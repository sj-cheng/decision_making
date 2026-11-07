"""Frozen 2x2 estimator mechanism experiment, gated new-history confirmation."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key]='1'

import argparse
from pathlib import Path
import json
import time
import hashlib
import numpy as np

import run_multimodal_estimator_validation as previous
from audit_l45_rollout_reliability import save,sha,observation
from solvers.corner_estimator import Estimator,VERSION,IMPORTANT_MASS
from solvers.estimator_particle_reference import ParticleEstimator
from estimator_audit_geometry import validate_paths

ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'current/l45_multimodal_estimator_20261025'
DEFAULT=ROOT/'current/l45_corner_turn_estimator_20261025'
METHODS=('particle','A','B','C','D','U0','U1')
FIELDS=('sample_coverage','mass_1m','nll','sample_nll','area90','coverage90','mean_error','spread','handoff_coverage')


def sources():
    names=['run_corner_estimator_validation.py','solvers/corner_estimator.py','solvers/multimodal_estimator.py',
        'solvers/local_pursuit.py','solvers/history_recovery.py','solvers/estimator_particle_reference.py',
        'run_multimodal_estimator_validation.py','estimator_audit_geometry.py','corner_estimator_scenarios.py',
        'prepare_corner_estimator_bank.py','tests/test_corner_estimator.py','L45_CORNER_TURN_ESTIMATION.md']
    return {str(ROOT/'code'/n):sha(ROOT/'code'/n) for n in names}


def freeze(output):
    frozen=json.loads((output/'bank_frozen.json').read_text())
    for path,h in frozen['artifacts'].items():
        if sha(Path(path))!=h:raise RuntimeError('Frozen new-history input changed')
    inputs={}
    for bank in (SOURCE/'bank',output/'bank'):
        for directory in ('controller_inputs','environment'):
            inputs.update({str(p):sha(p) for p in (bank/directory).glob('*.json')})
        inputs[str(bank/'manifest.json')]=sha(bank/'manifest.json')
    protocol=dict(version=VERSION,methods=list(METHODS),candidate='D',spacing=.5,sigma=.35,persistence=.9,
        particle_budget=64,important_region_mass=.01,statistic_seed=72120261025,bootstrap=10000,
        old_confirmation_is_development=True,controlled_cases=96,new_confirmation_cases=96,
        sources=sources(),inputs=inputs,no_mcts=True,no_training=True,no_deployment_change=True,
        confirmation_frozen_sha256=sha(output/'bank_frozen.json'))
    path=output/'experiment.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol:raise RuntimeError('Source/protocol changed; new output required')
    else:save(path,protocol)
    return protocol


def make_estimator(boxes,method,seed):
    return ParticleEstimator(boxes,seed) if method=='particle' else Estimator(boxes,method,seed)


def warmup():
    first=observation(dict(step=0,position=[-3.,6.],target=[2.,6.],remaining=80))
    hidden=observation(dict(step=1,position=[-3.,5.],target=None,remaining=79))
    boxes=np.array([[[-1.,1.],[-5.,5.]]])
    for m in METHODS:
        own=make_estimator(boxes,m,17);own.update(first);own.update(hidden)
        if m!='particle':validate_paths(own.witnesses(),np.array(own.observers),boxes)


def branch_diagnostics(own):
    if own.graph is None:return dict(important=0,missing=0,mass_error=0.)
    full=np.bincount(own.regions,weights=own.weights.sum(axis=1))
    empirical=np.bincount(own.regions[own.ids],weights=own.particle_weights,minlength=len(full))
    important=full>=IMPORTANT_MASS
    return dict(important=int(important.sum()),missing=int(np.sum(important&(empirical==0))),
        mass_error=float(np.max(np.abs(full[important]-empirical[important]))) if important.any() else 0.)


def evaluate_case(case):
    packet=case['packet'];boxes=np.array(packet['map'])
    order=list(METHODS);offset=int(hashlib.sha256(case['key'].encode()).hexdigest()[:8],16)%len(order)
    order=order[offset:]+order[:offset]
    models={m:make_estimator(boxes,m,case['seed']) for m in order}
    errors={};initialization={};frames=[];identity=True
    for m,e in models.items():
        started=time.perf_counter()
        try:
            for item in packet['history'][:-1]:e.update(observation(item))
        except (ValueError,RuntimeError) as exc:errors[m]=str(exc)
        initialization[m]=1000*(time.perf_counter()-started)
    for index,obs in enumerate(case['observations']):
        outputs={}
        for m,e in models.items():
            if m in errors:continue
            try:
                samples,weights=e.update(obs)
                points,marginal=e.distribution()
                paths=e.paths if m=='particle' else e.witnesses()
                if obs.target is None and paths is not None:
                    observers=e.history if m=='particle' else e.observers
                    violations,transitions=validate_paths(paths,np.array(observers),boxes)
                else:violations,transitions=0,0
                if violations:raise RuntimeError(f'Independent path certificate failed: {violations}')
                outputs[m]=(samples,weights,points,marginal,e.last_stats.copy(),transitions,
                            branch_diagnostics(e) if m!='particle' else dict(important=0,missing=0,mass_error=0.))
            except (ValueError,RuntimeError) as exc:errors[m]=str(exc)
        for a,b in [('A','B'),('C','D'),('U0','U1')]:
            if a in outputs and b in outputs and models[a].graph is not None:
                identity=identity and np.array_equal(models[a].weights,models[b].weights)
        if all(m in outputs for m in METHODS[1:]) and models['A'].graph is not None:
            identity=identity and all(np.array_equal(models['A'].reachable,models[m].reachable) for m in METHODS[2:])
        # No target truth is read until every arm's public-only update is done.
        truth=case['truths'][index]
        for m in METHODS:
            common=dict(method=m,frame=index,hidden=obs.target is None,handoff=index==0)
            if m not in outputs:
                frames.append(common|dict(failed=True,error=errors.get(m),sample_coverage=0.,mass_1m=0.,nll=-float(np.log(1e-12)),
                    sample_nll=-float(np.log(1e-12)),area90=400.,coverage90=0.,mean_error=30.,spread=30.,
                    update_ms=0.,build_ms=0.,transitions=0,important=0,missing=0,mass_error=0.))
                continue
            samples,weights,points,marginal,stats,transitions,branches=outputs[m]
            metric=previous.score(points,marginal,samples,truth)
            sample_nll=previous.score(samples,weights,samples,truth)['nll']
            frames.append(common|metric|dict(failed=False,sample_nll=sample_nll,update_ms=stats['update_ms'],
                build_ms=stats['build_ms'],transitions=transitions,unique_samples=len(np.unique(samples,axis=0)))|branches)
    summary={}
    for m in METHODS:
        fs=[f for f in frames if f['method']==m];hidden=[f for f in fs if f['hidden']]
        summary[m]={k:float(np.mean([f[k] for f in hidden])) if hidden else 0. for k in FIELDS if k!='handoff_coverage'}
        summary[m].update(handoff_coverage=fs[0]['sample_coverage'],failed=m in errors,failure=errors.get(m),initialization_ms=initialization[m],
            important=sum(f['important'] for f in hidden),missing=sum(f['missing'] for f in hidden),
            maximum_mass_error=max(f['mass_error'] for f in fs))
    return dict(key=case['key'],group=case['group'],density_and_range_identity=bool(identity),summary=summary,frames=frames)


def statistics(records):
    groups=sorted({r['group'] for r in records});rng=np.random.default_rng(72120261025)
    indices=rng.integers(0,len(groups),(10000,len(groups)))
    v={m:{k:np.array([np.mean([r['summary'][m][k] for r in records if r['group']==g]) for g in groups]) for k in FIELDS} for m in METHODS}
    summary={}
    for m in METHODS:
        fs=[f for r in records for f in r['frames'] if f['method']==m];times=[f['update_ms'] for f in fs if not f['failed']]
        init=[r['summary'][m]['initialization_ms'] for r in records]
        summary[m]={k:float(v[m][k].mean()) for k in FIELDS}
        summary[m].update(failed_cases=sum(r['summary'][m]['failed'] for r in records),
            important=sum(r['summary'][m]['important'] for r in records),missing=sum(r['summary'][m]['missing'] for r in records),
            maximum_mass_error=max(r['summary'][m]['maximum_mass_error'] for r in records),
            update_mean_ms=float(np.mean(times)) if times else None,update_p95_ms=float(np.percentile(times,95)) if times else None,
            initialization_mean_ms=float(np.mean(init)),initialization_p95_ms=float(np.percentile(init,95)),
            independently_verified_transitions=sum(f['transitions'] for f in fs))
    comparisons={};boots={}
    pairs=[('C','A'),('B','A'),('D','C'),('D','B'),('D','A'),('D','particle'),('D','U1'),('D','U0')]
    for a,b in pairs:
        label=a+'-'+b;comparisons[label]={}
        for field in FIELDS:
            delta=v[a][field]-v[b][field];boot=delta[indices].mean(axis=1)
            comparisons[label][field]=dict(mean=float(delta.mean()),ci95=np.percentile(boot,[2.5,97.5]).tolist())
            if field=='nll':boots[label]=boot
    comparisons['interaction']={}
    for field in FIELDS:
        delta=v['D'][field]-v['C'][field]-v['B'][field]+v['A'][field]
        comparisons['interaction'][field]=dict(mean=float(delta.mean()),ci95=np.percentile(delta[indices].mean(axis=1),[2.5,97.5]).tolist())
    main=['D-particle','D-U1'];means=np.array([comparisons[k]['nll']['mean'] for k in main])
    matrix=np.stack([boots[k] for k in main],axis=1)
    radius=float(np.quantile(np.max(np.abs(matrix-means),axis=1),.95))
    for k,mean in zip(main,means):comparisons[k]['nll']['simultaneous_ci95']=[float(mean-radius),float(mean+radius)]
    families={}
    for family in ('long_wall','wall_corner','multi_route'):
        selected=[r for r in records if r['group'].startswith(family)];ids=sorted({r['group'] for r in selected})
        families[family]={m:{k:float(np.mean([np.mean([r['summary'][m][k] for r in selected if r['group']==g]) for g in ids])) for k in FIELDS} for m in METHODS}
    checks=dict(no_failures=summary['D']['failed_cases']==0,density_identity=all(r['density_and_range_identity'] for r in records),
        handoff_coverage=summary['D']['handoff_coverage']>=.75,hidden_coverage=summary['D']['sample_coverage']>=.70,
        nll_vs_particle=comparisons['D-particle']['nll']['simultaneous_ci95'][1]<0,
        nll_vs_matched_feasible=comparisons['D-U1']['nll']['simultaneous_ci95'][1]<0,
        concentrated=summary['D']['area90']<=summary['U1']['area90'],wall_corner= families['wall_corner']['D']['handoff_coverage']>=.75,
        important_regions=summary['D']['missing']==0 and summary['D']['maximum_mass_error']<1e-10,
        runtime=summary['D']['update_p95_ms'] is not None and summary['D']['update_p95_ms']<=10.)
    pilot=dict(nll_gain=summary['D']['nll']<=summary['A']['nll']-.2,
        corner_gain=families['wall_corner']['D']['handoff_coverage']>=families['wall_corner']['A']['handoff_coverage']+4/32,
        no_failures=checks['no_failures'] and checks['density_identity'])
    return dict(groups=len(groups),cases=len(records),summary=summary,comparisons=comparisons,families=families,
        checks=checks,passed=all(checks.values()),pilot_checks=pilot,pilot_passed=all(pilot.values()))


def run(output,stage):
    protocol=freeze(output)
    if (output/(stage+'_results.json')).exists():raise RuntimeError('Completed stage cannot be overwritten')
    if stage=='confirmation':
        control=json.loads((output/'controlled_results.json').read_text())
        if not control['pilot_passed']:raise RuntimeError('Controlled gate failed; new histories remain unused')
    warmup()
    records=[];bank=SOURCE if stage=='controlled' else output
    for case in previous.confirmation_cases(bank):
        cached=output/stage/(case['key']+'.json')
        record=json.loads(cached.read_text()) if cached.exists() else evaluate_case(case)
        if not cached.exists():save(cached,record)
        records.append(record)
        save(output/'status.json',dict(stage=stage,completed=len(records),total=96,terminal=False,pid=os.getpid()))
        if len(records)%8==0:print(f'{stage}: {len(records)}/96',flush=True)
    result=statistics(records);result['sources_unchanged']=protocol['sources']==sources()
    if not result['sources_unchanged']:raise RuntimeError('Source changed during evaluation')
    save(output/(stage+'_results.json'),result)
    save(output/'status.json',dict(stage=stage+'_completed',terminal=True,completed=96,total=96,pid=None,
        estimator_accepted=False,search_comparison=False,neural_training=False,deployment_changed=False))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('controlled','confirmation'))
    parser.add_argument('--output',type=Path,default=DEFAULT);args=parser.parse_args();run(args.output.resolve(),args.stage)
