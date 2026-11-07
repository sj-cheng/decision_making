"""Read-only engineering gate before any local-search efficacy experiment."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'

from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import fcntl
import numpy as np
from numba import njit
from solvers.frozen_d_search_interface import (
    Controller, Observation, METHODS, FROZEN, VERSION, loader, belief_view,
    sample_roots, sampled_one_step, propagate, COMPASS, sensor_visible)
from estimator_audit_geometry import point_legal, blocked

ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'current/l45_corner_turn_estimator_20261025'
OUTPUT=ROOT/'current/l45_weighted_search_interface_v2_20261026'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.tmp')
    pending.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    pending.replace(path)


def observation(item):
    return Observation(item['step'],np.asarray(item['position'],float),
        None if item['target'] is None else np.asarray(item['target'],float),item['remaining'])


@njit(cache=True)
def independent_traces(trace,boxes,speed):
    errors=np.zeros(4,np.int64)
    for row in trace:
        a,b,c,d=row[:2],row[2:4],row[4:6],row[6:8]
        errors[0]+=int(not (point_legal(a,boxes) and point_legal(b,boxes)
            and np.max(np.abs(b-a))<=speed+2e-7 and not blocked(a,b,boxes)))
        errors[1]+=int(not (point_legal(c,boxes) and point_legal(d,boxes)
            and np.max(np.abs(d-c))<=1.+2e-7 and not blocked(c,d,boxes)))
        seen=np.sum((b-d)**2)<=36.+1e-12 and not blocked(b,d,boxes)
        errors[2]+=int(seen!=bool(row[8]))
        errors[3]+=int((np.linalg.norm(b-d)<1.)!=bool(row[9]))
    return errors


@njit(cache=True)
def independent_edges(points,links,lengths,boxes):
    errors,total=0,0
    for i in range(len(points)):
        for k in range(lengths[i]):
            j=links[i,k]
            errors+=int(not (point_legal(points[i],boxes) and point_legal(points[j],boxes)
                and np.max(np.abs(points[j]-points[i]))<=1.+2e-7 and not blocked(points[i],points[j],boxes)))
            total+=1
    return errors,total


def predicted_joint(e, exported):
    points,links,lengths,probabilities,*_=e.graph
    if exported:
        ids,weights,modes=belief_view(e)
        joint=np.zeros_like(e.weights)
        for k in range(64):joint[ids[k]]+=weights[k]*modes[k]
    else:joint=e.weights
    result=np.zeros_like(joint)
    for i in np.flatnonzero(joint.sum(1)>0):
        prior=.9*joint[i]+.1*joint[i].sum()/9.
        np.add.at(result,links[i,:lengths[i]],probabilities[i,:lengths[i]]*prior)
    return joint,result


def probability_audit(e,obs,seed):
    ids,weights,modes=belief_view(e)
    draws=200000
    counts,joint_counts=sample_roots(ids,weights,modes,draws,seed)
    expected=weights[:,None]*modes
    root_error=np.abs(counts/draws-weights)
    joint_error=np.abs(joint_counts/draws-expected)
    root_tolerance=6*np.sqrt(weights*(1-weights)/draws)+2/draws
    joint_tolerance=6*np.sqrt(expected*(1-expected)/draws)+2/draws
    exported,predicted=predicted_joint(e,True)
    _,full_prediction=predicted_joint(e,False)
    points,links,lengths,probabilities,sensor,*_=e.graph
    # Exact negative-observation filter equivalence for BOTH full and 64-position priors.
    mismatches=[]
    for prior,prediction in ((e.weights,full_prediction),(exported,predicted)):
        allowed=[]
        from l45_corner_shared_20261025.local_pursuit import clear
        for q in points:
            allowed.append(np.sum((obs.position-q)**2)>(6.+1e-6)**2 or
                not clear(obs.position,q,sensor,np.zeros(2,np.int64)))
        manual=prediction*np.asarray(allowed)[:,None]
        manual/=manual.sum()
        actual,*_=propagate(prior,prior.sum(1)>0,obs.position,points,links,lengths,probabilities,sensor,1)
        mismatches.append(float(np.max(np.abs(manual-actual))))
    empirical=sampled_one_step(ids,weights,modes,links,lengths,probabilities,draws,seed+1)/draws
    error=np.abs(empirical-predicted)
    tolerance=6*np.sqrt(predicted*(1-predicted)/draws)+2/draws
    edge_errors,edge_count=independent_edges(points,links,lengths,e.boxes)
    # Independent exact-map sensor versus D's deliberately conservative guard.
    exact=np.array([sensor_visible(obs.position,q,e.boxes) for q in points])
    guarded=np.array([np.sum((obs.position-q)**2)<=(6.+1e-6)**2 and
        clear(obs.position,q,sensor,np.zeros(2,np.int64)) for q in points])
    sensor_disagreement_mass=float(predicted.sum(1)[exact!=guarded].sum())
    return dict(draws=draws,root_max_error=float(root_error.max()),
        root_sampling_passed=bool(np.all(root_error<=root_tolerance)),
        joint_sampling_passed=bool(np.all(joint_error<=joint_tolerance)),
        transition_sampling_passed=bool(np.all(error<=tolerance)),
        filter_max_error=max(mismatches),filter_equivalent=max(mismatches)<1e-12,
        frozen_probabilities_normalized=bool(np.allclose(
            probabilities.sum(1)[lengths>0],1.,atol=1e-12,rtol=0)),
        exported_full_prediction_tv=float(.5*np.abs(predicted-full_prediction).sum()),
        root_weights_uniform_tv=float(.5*np.abs(weights-1/64).sum()),
        graph_edges=int(edge_count),illegal_graph_edges=int(edge_errors),
        sensor_disagreement_mass=sensor_disagreement_mass,
        sensor_consistent=sensor_disagreement_mass<1e-12)


def warmup():
    started=time.perf_counter()
    first=Observation(0,np.array([-3.,6.]),np.array([2.,6.]),80)
    hidden=Observation(1,np.array([-3.,5.]),None,79)
    c=Controller([[[-1.,1.],[-5.,5.]]],1.1,'T+Geo',17)
    c.act(first)
    c.act(hidden,fixed_simulations=2)
    independent_traces(np.concatenate(c.traces),c.public.boxes,1.1)
    ids,weights,modes=belief_view(c.estimator)
    sample_roots(ids,weights,modes,2,17)
    _,links,lengths,probabilities,*_=c.estimator.graph
    sampled_one_step(ids,weights,modes,links,lengths,probabilities,2,17)
    independent_edges(c.estimator.graph[0],links,lengths,c.public.boxes)
    return dict(seconds=time.perf_counter()-started,excluded_from_warm_decisions=True,
        includes_first_load_or_jit=True)


def freeze(output):
    sources={str(ROOT/'code'/name):sha(ROOT/'code'/name) for name in (
        'L45_WEIGHTED_SEARCH_INTERFACE.md','solvers/frozen_d_search_interface.py',
        'audit_frozen_d_search_interface.py','estimator_audit_geometry.py','tests/test_frozen_d_search_interface.py')}
    manifest=SOURCE/'bank/manifest.json'
    bank=json.loads(manifest.read_text())['entries']
    inputs={str(manifest):sha(manifest)}
    for entry in bank:
        path=SOURCE/'bank'/entry['public']
        inputs[str(path)]=sha(path)
    protected=[ROOT/'code/param.py',ROOT/'code/problems/example8.py',ROOT/'code/solvers/c_puct.py',
        ROOT/'code/cpp/bindings.cpp',ROOT/'code/cpp/problems/example8.hpp']
    result=dict(version=VERSION,run_date='2026-10-26',methods=list(METHODS),sources=sources,inputs=inputs,
        frozen_manifest_sha256=sha(FROZEN/'manifest.json'),estimator_parameters_changed=False,
        deployment_before_sha256={str(p):sha(p) for p in protected},
        public_groups=24,paired_worlds_used=False,stage='engineering_development_only',
        no_efficacy_games=True,no_new_efficacy_confirmation_consumed=True,
        revision='own-motion boxes inflated1e-6; sensor slab parameter tolerance1e-12; frozen D unchanged',
        preceding_failure_results_sha256=sha(ROOT/'current/l45_weighted_search_interface_20261026/results.json'),
        budget_ms=50.,deadline_reserve_ms=2.,planning_depth=8,simulation_cap=1000,
        probability_draws=200000,probability_tolerance='6 binomial SD + 2/N per bin',
        methods_rotated=True,threads=1,training=False)
    path=output/'experiment.json'
    if path.exists():
        if json.loads(path.read_text())!=result:raise RuntimeError('Frozen audit configuration changed')
    else:save(path,result)
    return bank,result


def signature(c):
    e=c.estimator
    return hashlib.sha256(e.particles.tobytes()+e.particle_weights.tobytes()+e.rng.tobytes()).hexdigest()


def replay_audit(packet,method,seed,reference):
    # Fresh instances, public input only. No scorer/environment module imported.
    copies=[]
    for _world_label in ('external_world_A','external_world_B'):
        c=Controller(packet['map'],packet['motion']['pursuer_axis_limit'],method,seed)
        for item in packet['history'][:-1]:c.estimator.update(observation(item))
        action=c.act(observation(packet['history'][-1]),fixed_simulations=reference['simulations'])
        copies.append(dict(action=action.tolist(),root_counts=c.last_stats['root_counts'],
            root_visits=c.last_stats['root_visits'],root_totals=c.last_stats['root_totals'],
            estimator=signature(c),rng=c.rng.tolist(),
            traces_sha256=hashlib.sha256(b''.join(t.tobytes() for t in c.traces)).hexdigest()))
    return dict(replay_exact=copies[0]==copies[1],external_label_invariance=copies[0]==copies[1],
        # Compare to the timed original with its recorded number of simulations.
        matches_original=all(copies[0][k]==reference[k] for k in copies[0]),
        simulations=reference['simulations'],signature=copies[0])


def quantiles(values):
    return dict(n=len(values),mean_ms=float(np.mean(values)),p50_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values,95)),maximum_ms=float(np.max(values)),
        over_50ms=int(np.sum(np.array(values)>50.))) if values else dict(n=0)


def run(output):
    bank,protocol=freeze(output)
    if (output/'results.json').exists():raise RuntimeError('Completed audit cannot be overwritten')
    save(output/'warmup.json',warmup())
    records=[];probabilities=[];replays=[];init_rows=[];pair_identity=True
    for group,entry in enumerate(bank):
        packet=json.loads((SOURCE/'bank'/entry['public']).read_text())
        seed=73020261025+group
        expected={}
        order=list(METHODS[group%3:]+METHODS[:group%3])
        group_rows=[]
        for method in order:
            c=Controller(packet['map'],packet['motion']['pursuer_axis_limit'],method,seed)
            for item in packet['history']:
                obs=observation(item)
                action=c.act(obs)
                stats=c.last_stats.copy()
                state=signature(c)
                if obs.step not in expected:expected[obs.step]=state
                pair_identity=pair_identity and expected[obs.step]==state
                trace=np.concatenate(c.traces) if c.traces else np.zeros((0,10))
                errors=independent_traces(trace,c.public.boxes,c.public.speed)
                kind='visible' if obs.target is not None else 'occlusion_build' if stats['build_ms']>0 else 'ordinary_hidden'
                row=dict(group=entry['id'],family=entry['family'],method=method,step=obs.step,kind=kind,
                    action=action.tolist(),stats=stats,estimator_sha256=state,
                    simulated_transitions=len(trace),independent_errors=errors.tolist(),
                    trace_sha256=hashlib.sha256(trace.tobytes()).hexdigest())
                group_rows.append(row)
            # Reproduce real handoff initialization separately: static setup,
            # history[:-1], current observation exactly once, final validation.
            initialized=Controller(packet['map'],packet['motion']['pursuer_axis_limit'],method,seed)
            stamp=time.perf_counter()
            for item in packet['history'][:-1]:initialized.estimator.update(observation(item))
            history_ms=1000*(time.perf_counter()-stamp)
            initialized_action=initialized.act(observation(packet['history'][-1]))
            init_rows.append(dict(group=entry['id'],method=method,setup=initialized.setup,
                history_ms=history_ms,first_decision_ms=initialized.last_stats['total_ms'],
                history_plus_decision_ms=history_ms+initialized.last_stats['total_ms'],
                complete_startup_ms=sum(initialized.setup.values())+history_ms+initialized.last_stats['total_ms']))
            # Fixed-count replay of final recorded decision (new search RNG at
            # handoff rather than RNG advanced by earlier history decisions).
            if group%8==0:
                n=initialized.last_stats['simulations']
                ref=dict(action=initialized_action.tolist(),root_counts=initialized.last_stats['root_counts'],
                    root_visits=initialized.last_stats['root_visits'],root_totals=initialized.last_stats['root_totals'],
                    estimator=signature(initialized),rng=initialized.rng.tolist(),
                    traces_sha256=hashlib.sha256(b''.join(t.tobytes() for t in initialized.traces)).hexdigest(),simulations=n)
                replays.append(dict(group=entry['id'],method=method,**replay_audit(packet,method,seed,ref)))
            if method=='G':probabilities.append(dict(group=entry['id'],**probability_audit(c.estimator,obs,seed%4294967296)))
        records+=group_rows
        save(output/'groups'/(entry['id']+'.json'),dict(decisions=group_rows))
        save(output/'status.json',dict(stage='interface_audit',groups=group+1,total=24,pid=os.getpid(),training=False))
        print(json.dumps(dict(groups=group+1,total=24,decisions=len(records))),flush=True)
    rebuild=[]
    # A legal public-history round trip returns to visibility, then loses it
    # again; confirms graph construction recurs during normal operation.
    for group in (0,8,16):
        entry=bank[group];packet=json.loads((SOURCE/'bank'/entry['public']).read_text())
        c=Controller(packet['map'],packet['motion']['pursuer_axis_limit'],'G',73020261025+group)
        original=packet['history']
        stream=original+original[-2::-1]+original[1:]
        for t,item in enumerate(stream):
            obs=Observation(t,np.asarray(item['position'],float),None if item['target'] is None else np.asarray(item['target'],float),80-t)
            c.act(obs)
            if c.last_stats['build_ms']>0:
                rebuild.append(dict(group=entry['id'],step=t,stats=c.last_stats.copy()))
    summary={}
    for method in METHODS:
        selected=[r for r in records if r['method']==method]
        summary[method]={kind:quantiles([r['stats']['total_ms'] for r in selected if r['kind']==kind])
            for kind in ('visible','ordinary_hidden','occlusion_build')}
        summary[method]['all_decisions']=quantiles([r['stats']['total_ms'] for r in selected])
        summary[method]['history_initialization']=quantiles([r['history_ms'] for r in init_rows if r['method']==method])
        summary[method]['history_plus_first_decision']=quantiles([r['history_plus_decision_ms'] for r in init_rows if r['method']==method])
        summary[method]['complete_startup']=quantiles([r['complete_startup_ms'] for r in init_rows if r['method']==method])
        summary[method]['components']={kind:{field:quantiles([r['stats'][field] for r in selected if r['kind']==kind]) for field in
            ('update_ms','build_ms','candidate_ms','tree_prepare_ms','search_including_prepare_ms','final_check_ms')}
            for kind in ('visible','ordinary_hidden','occlusion_build')}
        summary[method]['simulations']=dict(total=sum(r['stats']['simulations'] for r in selected),
            zero_hidden=sum(r['kind']!='visible' and r['stats']['simulations']==0 for r in selected))
    checks=dict(paired_estimator_identity=pair_identity,
        root_weights=all(p['root_sampling_passed'] and p['joint_sampling_passed'] for p in probabilities),
        motion_kernel=all(p['filter_equivalent'] and p['transition_sampling_passed'] and p['frozen_probabilities_normalized'] for p in probabilities),
        sensor=all(p['sensor_consistent'] for p in probabilities),
        legal_graph=all(p['illegal_graph_edges']==0 for p in probabilities),
        legal_simulations=all(not any(r['independent_errors']) for r in records),
        replay=all(r['replay_exact'] and r['matches_original'] for r in replays),
        public_only_invariance=all(r['external_label_invariance'] for r in replays),
        frozen_estimator_verified=bool(loader.verify()),
        protected_sources_unchanged=all(sha(p)==h for p,h in protocol['deployment_before_sha256'].items()))
    runtime={method:all(summary[method][kind]['p95_ms']<=50. for kind in
        ('ordinary_hidden','occlusion_build','history_plus_first_decision')) for method in METHODS}
    accepted=all(checks.values()) and all(runtime.values())
    result=dict(interface_checks=checks,budget_checks=runtime,passed=accepted,summary=summary,
        groups=24,decision_rows=len(records),simulated_transitions=sum(r['simulated_transitions'] for r in records),
        probabilities=probabilities,replays=replays,initializations=init_rows,repeated_occlusions=rebuild,
        search_effectiveness_comparison_started=False,independent_effectiveness_confirmation_used=False,
        training=False,deployment_changed=False,host=dict(machine=platform.machine(),python=platform.python_version()))
    save(output/'results.json',result)
    save(output/'decision.json',dict(interface_passed=all(checks.values()),budget_passed=all(runtime.values()),
        efficacy_comparison_allowed=accepted,estimator_remains_frozen=True,training=False,deployment_changed=False,
        next_step='Freeze fresh controlled and independent efficacy protocol' if accepted else
            'Resolve demonstrated interface/runtime gate without changing frozen estimator probabilities; no efficacy claim'))
    save(output/'status.json',dict(stage='interface_passed' if accepted else 'stopped_at_preflight_gate',
        groups=24,total=24,passed=accepted,search_effectiveness_games=0,pid=None,training=False,deployment_changed=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (ROOT/'current/.l45_local_pursuit_serial.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.output)
