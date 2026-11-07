"""Single-mechanism closed-loop study: frozen-D rollout geometry replanning."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse,fcntl,json,time,hashlib,collections
from pathlib import Path
import numpy as np
import run_frozen_d_closed_loop as base
from solvers.frozen_d_search_interface import Controller as Original,loader,FROZEN
from solvers.rollout_replan import Controller as Replanned
from audit_frozen_d_search_interface import observation,independent_traces,quantiles,warmup
from corner_estimator_scenarios import construct
from verify_l45_occlusion_scenarios import verify_case

ROOT=base.ROOT
SOURCE=base.OUT
OUT=ROOT/'current/l45_rollout_replan_20261026'
METHODS=('G','T','T+Geo','T-R','T+Geo-R')
MAIN=(('T-R','T'),('T+Geo-R','T+Geo'),('T-R','G'),('T+Geo-R','G'))


def save(p,d):base.save(p,d)
def sha(p):return base.sha(p)


class BudgetedActor:
    """Common startup debit; all algorithmic behavior lives in frozen actors."""
    def __init__(self,actor,initialization_ms):
        self.actor=actor;self.initialization_ms=initialization_ms;self.first=True

    def __getattr__(self,name):return getattr(self.actor,name)

    def act(self,obs,fixed_simulations=None):
        charge=self.initialization_ms if self.first else 0.
        allowance=max(0.,200.-charge)
        self.actor.budget_ms=allowance
        try:action=self.actor.act(obs,fixed_simulations=fixed_simulations)
        finally:self.actor.budget_ms=200.;self.first=False
        self.actor.last_stats.update(charged_initialization_ms=charge,allowed_act_ms=allowance)
        return action


def initialize(packet,method,seed,cache_enabled=True):
    started=time.perf_counter()
    if method.endswith('-R'):
        actor=Replanned(packet['map'],packet['motion']['pursuer_axis_limit'],method,seed,cache_enabled=cache_enabled)
    else:actor=Original(packet['map'],packet['motion']['pursuer_axis_limit'],method,seed,budget_ms=200.)
    static=1000*(time.perf_counter()-started);stamp=time.perf_counter()
    for item in packet['history'][:-1]:actor.estimator.update(observation(item))
    history=1000*(time.perf_counter()-stamp)
    return BudgetedActor(actor,static+history),dict(static_setup_ms=static,history_initialization_ms=history,setup_breakdown=actor.setup)


def development_entries():return json.loads((SOURCE/'bank/manifest.json').read_text())['entries'][::2]


def inputs(task):
    bank=OUT/'bank' if task['stage']=='confirmation' else SOURCE/'bank'
    return tuple(json.loads((bank/task['entry'][k]).read_text()) for k in ('public','environment'))


def prepare():
    if (OUT/'bank_frozen.json').exists():return
    known=set()
    for p in (ROOT/'current').glob('**/controller_inputs/*.json'):
        d=json.loads(p.read_text())
        if 'map' in d and 'history' in d:
            known.add(hashlib.sha256(json.dumps(dict(map=d['map'],history=d['history']),sort_keys=True).encode()).hexdigest())
    entries=[];checks=[]
    for f,family in enumerate(('long_wall','wall_corner','multi_route')):
        for h,H in enumerate((4,6)):
            for v,speed in enumerate((1.1,1.3)):
                for variant in range(2):
                    seed=int(np.random.SeedSequence([83020261026,f,h,v,variant]).generate_state(1)[0])
                    rng=np.random.default_rng(seed);height=float(rng.uniform(6.2,6.9) if H==4 else rng.uniform(8.15,8.85))
                    env,pub=construct(family,H,speed,seed,height,variant,pause=variant)
                    check=verify_case(env,pub)
                    if not check['passed']:raise RuntimeError((env['id'],check['reasons']))
                    fingerprint=hashlib.sha256(json.dumps(dict(map=pub['map'],history=pub['history']),sort_keys=True).encode()).hexdigest()
                    if fingerprint in known:raise RuntimeError('Confirmation public history overlap')
                    known.add(fingerprint)
                    for kind,d in (('controller_inputs',pub),('environment',env)):save(OUT/'bank'/kind/(env['id']+'.json'),d)
                    entries.append(dict(id=env['id'],family=family,public=f'controller_inputs/{env["id"]}.json',environment=f'environment/{env["id"]}.json',geometry_history_sha256=fingerprint))
                    checks.append(check)
    save(OUT/'bank/manifest.json',dict(entries=entries,scope='new instances of three known topology families'))
    save(OUT/'bank_acceptance.json',dict(passed=True,groups=24,checks=checks))
    save(OUT/'bank_frozen.json',dict(seed=83020261026,outcomes_read=False,groups=24,files={str(p):sha(p) for p in (OUT/'bank').rglob('*.json')}))


def freeze():
    baseline=json.loads((SOURCE/'experiment.json').read_text())
    assert all(sha(p)==h for p,h in (baseline['sources_and_inputs']|baseline['protected_before']).items())
    names=('run_rollout_replan_experiment.py','solvers/rollout_replan.py','L45_ROLLOUT_REPLAN_20261026.md','tests/test_rollout_replan.py')
    config=dict(methods=list(METHODS),main_pairs=[list(x) for x in MAIN],budget_ms=200.,simulation_cap=1000,depth=8,
        single_change='recompute original G geometric action from simulated own position and fixed root weighted belief in rollout',
        source_hashes={str(ROOT/'code'/p):sha(ROOT/'code'/p) for p in names},
        original_sources_and_inputs=baseline['sources_and_inputs'],protected_sources=baseline['protected_before'],
        bank_frozen_sha256=sha(OUT/'bank_frozen.json'),baseline_experiment_sha256=sha(SOURCE/'experiment.json'),
        development_games=120,confirmation_games=240,bootstrap_draws=10000,bootstrap_seed=83120261026,
        startup_debited_from_first200ms_for_all_methods=True,root_cache='exact float position; decision-local',
        no_estimator_change=True,no_neural_training=True,no_deployment_change=True)
    path=OUT/'experiment.json'
    if path.exists():assert json.loads(path.read_text())==config,'Frozen source/config changed'
    else:save(path,config)
    frozen=json.loads((OUT/'bank_frozen.json').read_text())
    assert all(sha(p)==h for p,h in frozen['files'].items())
    loader.verify();return config


def tasks(stage):
    entries=development_entries() if stage=='development' else json.loads((OUT/'bank/manifest.json').read_text())['entries']
    requests=[]
    for i,entry in enumerate(entries):
        for wi,world in enumerate(('A','B')):
            seeds=np.random.SeedSequence([83220261026,int(entry['geometry_history_sha256'][:8],16),wi]).generate_state(3)
            offset=(i+wi)%5
            for method in METHODS[offset:]+METHODS[:offset]:
                requests.append(dict(key=entry['id']+'_'+world+'_'+method.replace('+','plus'),entry=entry,stage=stage,world=world,method=method,
                    seed=int(seeds[0]),opponent_seed=int(seeds[1]),environment_seed=int(seeds[2])))
    return requests


def signature(own):
    e=own.estimator
    return hashlib.sha256(e.particles.tobytes()+e.particle_weights.tobytes()+e.rng.tobytes()).hexdigest()


def fixed_signature(own,action):
    s=own.last_stats
    return dict(action=action.tolist(),rng=own.rng.tolist(),posterior=signature(own),
        visits=s['root_visits'],totals=s['root_totals'],root_counts=s['root_counts'],
        traces=hashlib.sha256(b''.join(x.tobytes() for x in own.traces)).hexdigest())


def preflight():
    path=OUT/'preflight.json'
    if path.exists():return json.loads(path.read_text())
    warmup();first=development_entries()[0];packet=json.loads((SOURCE/'bank'/first['public']).read_text())
    warm,_=initialize(packet,'T-R',17);warm.act(observation(packet['history'][-1]),fixed_simulations=2)
    rows=[];checks=[]
    for i,entry in enumerate(development_entries()):
        packet=json.loads((SOURCE/'bank'/entry['public']).read_text());seed=83320261026+i
        posterior=[]
        for method in METHODS[i%5:]+METHODS[:i%5]:
            for repeat in range(2):
                own,cost=initialize(packet,method,seed)
                started=time.perf_counter();action=own.act(observation(packet['history'][-1]));elapsed=1000*(time.perf_counter()-started)
                trace=np.concatenate(own.traces) if own.traces else np.zeros((0,10))
                errors=independent_traces(trace,own.public.boxes,own.public.speed)
                rows.append(dict(group=entry['id'],method=method,repeat=repeat,decision_ms=elapsed,
                    startup_plus_decision_ms=cost['static_setup_ms']+cost['history_initialization_ms']+elapsed,
                    statistics=own.last_stats,errors=errors.tolist(),transitions=len(trace),
                    output_legal=base.legal_action(np.array(packet['history'][-1]['position']),action,own.public.speed,own.public.boxes)))
                posterior.append(signature(own))
            if method.endswith('-R'):
                outputs=[]
                for _ in range(2):
                    actor,_=initialize(packet,method,seed)
                    action=actor.act(observation(packet['history'][-1]),fixed_simulations=32)
                    outputs.append(fixed_signature(actor,action))
                variants=[]
                for cache in (True,False):
                    actor,_=initialize(packet,method,seed,cache_enabled=cache)
                    action=actor.act(observation(packet['history'][-1]),fixed_simulations=8)
                    variants.append(fixed_signature(actor,action))
                checks.append(dict(group=entry['id'],method=method,fixed32_replay=outputs[0]==outputs[1],cache_direct_fixed8=variants[0]==variants[1]))
        checks.append(dict(group=entry['id'],common_posterior=len(set(posterior))==1))
        print(json.dumps(dict(stage='preflight',groups=i+1,total=12)),flush=True)
    timing={m:dict(decision=quantiles([r['decision_ms'] for r in rows if r['method']==m]),
        startup=quantiles([r['startup_plus_decision_ms'] for r in rows if r['method']==m]),
        min_simulations=min(r['statistics']['simulations'] for r in rows if r['method']==m)) for m in METHODS}
    passed=(all(all(v for k,v in c.items() if k not in ('group','method')) for c in checks)
        and all(r['output_legal'] and not any(r['errors']) for r in rows)
        and all(timing[m]['decision']['p95_ms']<=200 and timing[m]['startup']['p95_ms']<=200 and timing[m]['min_simulations']>0 for m in ('T-R','T+Geo-R')))
    result=dict(passed=passed,checks=checks,rows=rows,timing=timing,total_root_decisions=len(rows),
        independent_transitions=sum(r['transitions'] for r in rows),new_confirmation_evaluated=False)
    save(path,result);return result


def statistics(rows):
    groups=sorted({r['history_id'] for r in rows});index={(r['history_id'],r['world'],r['method']):r for r in rows}
    assert len(index)==len(groups)*10
    draws=np.random.default_rng(83120261026).integers(0,len(groups),(10000,len(groups)))
    pairs=MAIN+(('T+Geo-R','T-R'),)
    comparisons={};boots={}
    for a,b in pairs:
        label=a+'__'+b;comparisons[label]={}
        for metric in ('capture','rediscovered','completion_cost','rediscovery_cost'):
            def value(r):return r['first_rediscovery'] if metric=='rediscovery_cost' and r['first_rediscovery'] is not None else r['remaining_at_handoff'] if metric=='rediscovery_cost' else r[metric]
            delta=np.array([np.mean([value(index[(g,w,a)])-value(index[(g,w,b)]) for w in ('A','B')]) for g in groups])
            boot=delta[draws].mean(1)
            comparisons[label][metric]=dict(mean=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
            if metric=='capture':boots[label]=boot
    main=[a+'__'+b for a,b in MAIN];centers=np.array([comparisons[k]['capture']['mean'] for k in main])
    radius=float(np.quantile(np.max(np.abs(np.stack([boots[k] for k in main],axis=1)-centers),axis=1),.95))
    for label,center in zip(main,centers):
        c=comparisons[label];c['capture']['simultaneous_ci95']=[float(center-radius),float(center+radius)]
        method=label.split('__')[0]
        c['reliable_extra_capture_value']=bool(center-radius>0 and c['completion_cost']['ci95'][1]<=0 and all(r['failure'] is None for r in rows if r['method']==method))
    summary={m:base.aggregate([r for r in rows if r['method']==m]) for m in METHODS}
    for m,s in summary.items():
        selected=[r for r in rows if r['method']==m]
        s['rediscovery_cost']=float(np.mean([r['first_rediscovery'] if r['first_rediscovery'] is not None else r['remaining_at_handoff'] for r in selected]))
    return dict(groups=len(groups),games=len(rows),summary=summary,comparisons=comparisons,
        by_family={f:{m:base.aggregate([r for r in rows if r['family']==f and r['method']==m]) for m in METHODS} for f in ('long_wall','wall_corner','multi_route')},
        bootstrap_draws=10000,bootstrap_seed=83120261026,simultaneous_capture_radius=radius)


def run_stage(stage):
    rows=[];requests=tasks(stage);started=time.perf_counter()
    save(OUT/stage/'requests.json',requests)
    for i,task in enumerate(requests):
        row=base.cached(task)
        if row is None:row=base.commit(task,base.episode(task))
        rows.append(row)
        status=dict(stage=stage,completed=i+1,total=len(requests),elapsed_seconds=time.perf_counter()-started,pid=os.getpid(),training=False)
        save(OUT/'status.json',status);print(json.dumps(status),flush=True)
    save(OUT/stage/'results.json',rows)
    result=statistics(rows);save(OUT/stage/'statistics.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','preflight','pipeline'))
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'current/.l45_local_pursuit_serial.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='prepare':prepare();freeze();return
        freeze()
        base.OUT=OUT;base.inputs=inputs;base.initialize=initialize
        if not preflight()['passed']:
            save(OUT/'status.json',dict(stage='preflight_failed',pid=None,confirmation_evaluated=False));return
        if args.command=='preflight':return
        run_stage('development');result=run_stage('confirmation');freeze()
        c=result['comparisons']
        reliable_T=c['T-R__T']['reliable_extra_capture_value'] and c['T-R__G']['reliable_extra_capture_value']
        reliable_geo=c['T+Geo-R__T+Geo']['reliable_extra_capture_value'] and c['T+Geo-R__G']['reliable_extra_capture_value']
        save(OUT/'decision.json',dict(reliable_T_revision=reliable_T,reliable_TGeo_revision=reliable_geo,
            stop_modification_direction=not (reliable_T or reliable_geo),estimator_unchanged=bool(loader.verify()),
            neural_training=False,deployment_changed=False))
        save(OUT/'status.json',dict(stage='complete',completed=360,total=360,pid=None,training=False,deployment_changed=False))


if __name__=='__main__':main()
