"""Bounded search-value study; immutable old algorithms and offline-only work."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from solvers.frozen_d_search_interface import Controller as Geometry, Observation, sensor_visible, loader, FROZEN
from solvers.search_value_belief import Controller as Search
from solvers.root_audit_puct import RootPUCT, cpp
from search_value_model import ModelMatchedTarget
from corner_estimator_scenarios import construct, original_model
from verify_l45_occlusion_scenarios import verify_case
from run_frozen_d_closed_loop import save, sha, legal_action
from audit_frozen_d_search_interface import observation, independent_traces
from analyze_search_value_validation import statistics, write_report

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'current/l45_search_value_20261026'
OLD=ROOT/'current/l45_closed_loop_D_20261026/bank'
METHODS=('G','OL200','FB200','OLH','FBH')
OPPONENTS=('matched','rootpuct')


def development_entries():
    entries=json.loads((OLD/'manifest.json').read_text())['entries']
    picked=[]
    for family in ('long_wall','wall_corner','multi_route'):
        for H,speed in ((4,1.1),(6,1.3)):
            picked.append(next(e for e in entries if e['family']==family and f'_H{H}_' in e['id'] and f'_v{speed:.1f}_' in e['id']))
    return picked


def prepare():
    if (OUT/'bank_frozen.json').exists():
        return
    known=set()
    for p in (ROOT/'current').glob('**/controller_inputs/*.json'):
        d=json.loads(p.read_text())
        if 'map' in d and 'history' in d:
            known.add(hashlib.sha256(json.dumps(dict(map=d['map'],history=d['history']),sort_keys=True).encode()).hexdigest())
    entries=[];checks=[]
    for f,family in enumerate(('long_wall','wall_corner','multi_route')):
        for h,H in enumerate((4,6)):
            for v,speed in enumerate((1.1,1.3)):
                seed=int(np.random.SeedSequence([84020261026,f,h,v]).generate_state(1)[0])
                rng=np.random.default_rng(seed)
                height=float(rng.uniform(6.2,6.9) if H==4 else rng.uniform(8.15,8.85))
                variant=(h+v)%2
                env,packet=construct(family,H,speed,seed,height,variant,pause=variant)
                check=verify_case(env,packet)
                if not check['passed']:
                    raise RuntimeError(('Independent geometric rejection',env['id'],check))
                digest=hashlib.sha256(json.dumps(dict(map=packet['map'],history=packet['history']),sort_keys=True).encode()).hexdigest()
                if digest in known:
                    raise RuntimeError('Confirmation overlaps historical public history')
                known.add(digest)
                for kind,data in (('controller_inputs',packet),('environment',env)):
                    save(OUT/'bank'/kind/(env['id']+'.json'),data)
                entries.append(dict(id=env['id'],family=family,public=f'controller_inputs/{env["id"]}.json',environment=f'environment/{env["id"]}.json',geometry_history_sha256=digest))
                checks.append(check)
    save(OUT/'bank/manifest.json',dict(entries=entries,groups=12,scope='new instances; three known topology families'))
    save(OUT/'bank_acceptance.json',dict(passed=True,checks=checks,no_algorithm_outcomes_used=True))
    save(OUT/'bank_frozen.json',dict(seed=84020261026,outcomes_read=False,groups=12,files={str(p):sha(p) for p in (OUT/'bank').rglob('*.json')}))


def freeze():
    names=('L45_SEARCH_VALUE_VALIDATION_20261026.md','run_search_value_validation.py','search_value_model.py',
        'solvers/search_value_belief.py','tests/test_search_value_model.py','tests/test_search_value_belief.py',
        'analyze_search_value_validation.py','audit_search_value_validation.py',
        'solvers/frozen_d_search_interface.py','solvers/root_audit_puct.py','corner_estimator_scenarios.py',
        'solvers/rollout_geometry_exact.py','solvers/rollout_replan.py',
        'verify_l45_occlusion_scenarios.py','estimator_audit_geometry.py','audit_frozen_d_search_interface.py',
        'run_frozen_d_closed_loop.py','problems/example8.py','solvers/c_puct.py')
    protected=('param.py','cpp/bindings.cpp','cpp/problems/example8.hpp','cpp/problems/problem.hpp')
    paths=[ROOT/'code'/p for p in names]+[Path(cpp.__file__),FROZEN/'manifest.json',OLD/'manifest.json',OUT/'bank_frozen.json']
    for e in development_entries():
        paths.extend(OLD/e[k] for k in ('public','environment'))
    prior=ROOT/'current/l45_rollout_replan_exact_20261026'
    old_names=('REPORT.md','decision.json','development/results.json','confirmation/results.json','DELIVERY_SHA256.json')
    config=dict(version='search_value_exact_belief_v1',methods=list(METHODS),opponents=list(OPPONENTS),
        budgets_ms=dict(G=200.,OL200=200.,FB200=200.,OLH=2000.,FBH=2000.),low_cap=1000,high_cap=4096,depth=8,
        development_groups=6,confirmation_groups=12,development_games=120,confirmation_games=240,
        bootstrap_draws=10000,bootstrap_seed=84120261026,development_minimum_useful_steps=1.,
        source_hashes={str(p):sha(p) for p in paths},protected_sources={str(ROOT/'code'/p):sha(ROOT/'code'/p) for p in protected},
        prior_results={str(prior/p):sha(prior/p) for p in old_names},frozen_D_changed=False,training=False,deployment_changed=False,
        matched_scope='fixed-root motion kernel only until first rediscovery; subsequent capture is a transfer metric',
        tree_belief='weighted64 induced root; exact fixed-graph prediction and exact simulated sensor conditioning')
    p=OUT/'experiment.json'
    if p.exists():
        assert json.loads(p.read_text())==config,'Frozen source or inputs changed'
    else:
        save(p,config)
        folder=OUT/'source_snapshot'
        for name in names:
            dst=folder/name;dst.parent.mkdir(parents=True,exist_ok=True)
            dst.write_bytes((ROOT/'code'/name).read_bytes())
    loader.verify()
    frozen=json.loads((OUT/'bank_frozen.json').read_text())
    assert all(sha(p)==h for p,h in frozen['files'].items())
    return config


def tasks(stage):
    bank=OUT/'bank' if stage=='confirmation' else OLD
    entries=json.loads((bank/'manifest.json').read_text())['entries'] if stage=='confirmation' else development_entries()
    result=[]
    for i,entry in enumerate(entries):
        digest=int(entry['geometry_history_sha256'][:8],16)
        for wi,world in enumerate(('A','B')):
            seeds=np.random.SeedSequence([84220261026,digest,wi]).generate_state(3)
            for oi,opponent in enumerate(OPPONENTS):
                order=list(METHODS[(i+wi+oi)%5:]+METHODS[:(i+wi+oi)%5])
                for method in order:
                    result.append(dict(key=f'{entry["id"]}_{world}_{opponent}_{method}',group=entry['id'],entry=entry,
                        family=entry['family'],world=world,method=method,opponent=opponent,stage=stage,
                        packet_path=str(bank/entry['public']),environment_path=str(bank/entry['environment']),
                        seed=int(seeds[0]),opponent_seed=int(seeds[1]),environment_seed=int(seeds[2]),
                        target_seed=int(seeds[0]),draw_index=wi))
    return result


def initialize(packet,method,seed):
    stamp=time.perf_counter()
    budget=2000. if method.endswith('H') else 200.
    if method=='G':
        own=Geometry(packet['map'],packet['motion']['pursuer_axis_limit'],'G',seed,budget_ms=budget)
    else:
        own=Search(packet['map'],packet['motion']['pursuer_axis_limit'],method[:2],seed,budget_ms=budget,
            simulations=4096 if method.endswith('H') else 1000)
    static=1000*(time.perf_counter()-stamp);stamp=time.perf_counter()
    for item in packet['history'][:-1]:
        own.estimator.update(observation(item))
    history=1000*(time.perf_counter()-stamp)
    return own,dict(static_setup_ms=static,history_initialization_ms=history,total_ms=static+history,budget_ms=budget)


def signature(e):
    return hashlib.sha256(e.particles.tobytes()+e.particle_weights.tobytes()+e.rng.tobytes()).hexdigest()


def episode(task,limit=None,fixed=None):
    packet=json.loads(Path(task['packet_path']).read_text())
    environment=json.loads(Path(task['environment_path']).read_text())
    boxes=np.asarray(packet['map']);speed=packet['motion']['pursuer_axis_limit']
    p=original_model(packet['map'],speed)
    own,initialization=initialize(packet,task['method'],task['seed'])
    if task['opponent']=='matched':
        other=ModelMatchedTarget(packet,task['target_seed'],task['draw_index'])
        state=np.zeros((p.state_dim,1))
        state[:2,0]=other.position
        state[4:6,0]=packet['history'][-1]['position']
        state[2:4,0]=[-9.8,-9.8];state[6:8,0]=[9.8,9.8]
        state[p.active_idxs,0]=[1,0,1,0]
        state[8,0]=packet['history'][-1]['step']
        matched_initial=dict(index=other.initial_index,mode=other.initial_mode,position=other.position.tolist(),
            seed=task['target_seed'],draw_index=task['draw_index'])
    else:
        world=next(w for w in environment['worlds'] if w['id']==task['world'])
        state=np.asarray(world['handoff_full_state'],float).reshape(-1,1)
        other=RootPUCT([None]*4,None,3000,0.,0.)
        other.seeded_state(p,task['opponent_seed'],task['environment_seed'])
        matched_initial=None
    if sensor_visible(state[4:6,0],state[:2,0],boxes):
        raise RuntimeError('Handoff must be hidden; do not resample for outcome')
    initial=state[:,0].tolist();remaining=80-int(state[8,0])
    trace=[];failure=None;first_seen=None
    for index in range(min(remaining,limit) if limit else remaining):
        start_cycle=time.perf_counter()
        seen=sensor_visible(state[4:6,0],state[:2,0],boxes)
        obs=Observation(int(round(state[8,0])),state[4:6,0].copy(),state[:2,0].copy() if seen else None,80-int(round(state[8,0])))
        charge=initialization['total_ms'] if index==0 else 0.
        own.budget_ms=max(0.,initialization['budget_ms']-charge)
        rng_before=own.rng.copy();stamp=time.perf_counter()
        try:
            action=own.act(obs,fixed_simulations=None if fixed is None else fixed[index])
        except Exception as exc:
            failure=dict(reason='controller_failure',error=repr(exc),step=obs.step)
            break
        own_ms=1000*(time.perf_counter()-stamp)
        stats=own.last_stats.copy()
        stats.update(charged_initialization_ms=charge,allowed_act_ms=own.budget_ms,registered_budget_ms=initialization['budget_ms'])
        own.budget_ms=initialization['budget_ms']
        # Opponent is evaluated from the common pre-action state. The matched
        # target ignores own action; RootPUCT never receives the chosen action.
        stamp=time.perf_counter()
        if task['opponent']=='matched':
            raw=other.step()-state[:2,0]
        else:
            raw=np.asarray(other.search(p,state.copy(),turn=0).best_action).reshape(-1)[:2].astype(float)
        shield=not legal_action(state[:2,0],raw,1.,boxes)
        other_action=np.zeros(2) if shield else raw.copy()
        if shield and task['opponent']=='matched':
            failure=dict(reason='matched_illegal_motion',step=obs.step)
        opponent_ms=1000*(time.perf_counter()-stamp)
        own_legal=legal_action(state[4:6,0],action,speed,boxes)
        other_legal=legal_action(state[:2,0],other_action,1.,boxes)
        if not own_legal or not other_legal:
            failure=dict(reason='own_illegal' if not own_legal else 'opponent_illegal',step=obs.step)
        joint=np.zeros((p.action_dim,1));joint[:2,0]=other_action;joint[4:6,0]=action
        stamp=time.perf_counter()
        nxt=state.copy() if failure else p.step(state,joint,p.dt)
        environment_ms=1000*(time.perf_counter()-stamp)
        if not failure and not (np.allclose(nxt[:2,0],state[:2,0]+other_action,rtol=0,atol=1e-10)
                and np.allclose(nxt[4:6,0],state[4:6,0]+action,rtol=0,atol=1e-10)):
            failure=dict(reason='joint_motion_mismatch',step=obs.step)
        cycle_ms=1000*(time.perf_counter()-start_cycle)
        simulated=np.concatenate(own.traces) if own.traces else np.zeros((0,10))
        errors=independent_traces(simulated,boxes,speed)
        if np.any(errors):
            failure=dict(reason='simulated_geometry_failure',step=obs.step,errors=errors.tolist())
        post_seen=bool(sensor_visible(nxt[4:6,0],nxt[:2,0],boxes))
        capture=bool(nxt[9,0]<.5 and nxt[11,0]<.5)
        if capture and not (np.linalg.norm(nxt[4:6,0]-nxt[:2,0])<1. and own_legal and other_legal):
            failure=dict(reason='invalid_capture',step=obs.step)
        if post_seen and first_seen is None:
            first_seen=index+1
        e=own.estimator
        trace.append(dict(step=obs.step,state=state[:,0].tolist(),next_state=nxt[:,0].tolist(),
            observation=dict(step=obs.step,position=obs.position.tolist(),target=None if obs.target is None else obs.target.tolist(),remaining=obs.remaining),
            own_action=action.tolist(),opponent_raw=raw.tolist(),opponent_action=other_action.tolist(),opponent_shield=shield,
            particles=e.particles.tolist(),weights=e.particle_weights.tolist(),estimator_sha256=signature(e),
            search_rng_before=rng_before.tolist(),search_rng_after=own.rng.tolist(),statistics=stats,
            own_ms=own_ms,opponent_ms=opponent_ms,environment_ms=environment_ms,cycle_ms=cycle_ms,
            post_seen=post_seen,capture=capture,own_legal=own_legal,opponent_legal=other_legal,
            simulated_transitions=len(simulated),simulation_errors=errors.tolist(),simulation_sha256=hashlib.sha256(simulated.tobytes()).hexdigest()))
        state=nxt
        if failure or capture or p.is_terminal(state):
            break
    capture=bool(trace and trace[-1]['capture'] and not failure)
    lat=[r['own_ms'] for r in trace]
    row=dict(key=task['key'],group=task['group'],history_id=task['group'],family=task['family'],world=task['world'],
        method=task['method'],opponent=task['opponent'],stage=task['stage'],capture=capture,first_seen=first_seen,
        original_remaining=remaining,cost=len(trace) if capture else remaining,
        rediscovery_cost=first_seen if first_seen is not None else remaining,failure=failure,
        executed_steps=len(trace),initialization=initialization,initial_total_ms=initialization['total_ms']+(lat[0] if lat else 0.),
        latencies_ms=lat,simulations=[r['statistics']['simulations'] for r in trace],
        hidden_simulations=[r['statistics']['simulations'] for r in trace if r['observation']['target'] is None],
        shield_count=sum(r['opponent_shield'] for r in trace),simulated_transitions=sum(r['simulated_transitions'] for r in trace),
        cycle_ms=[r['cycle_ms'] for r in trace],budget_ms=initialization['budget_ms'],
        registered_overruns=sum(r['own_ms']>initialization['budget_ms'] for r in trace),
        simulation_errors=np.sum([r['simulation_errors'] for r in trace],axis=0).tolist() if trace else [0]*4,
        reason='engineering_failure' if failure else 'capture' if capture else 'engineering_prefix' if limit else 'timeout')
    return dict(task=task,initial_state=initial,final_state=state[:,0].tolist(),matched_initial=matched_initial,summary=row,trace=trace)


def commit(task,payload):
    path=OUT/task['stage']/'episodes'/(task['key']+'.json');save(path,payload)
    row=payload['summary']|dict(episode=str(path),episode_sha256=sha(path))
    save(OUT/task['stage']/'cache'/(task['key']+'.json'),row)
    return row


def cached(task):
    p=OUT/task['stage']/'cache'/(task['key']+'.json')
    if not p.exists():return None
    r=json.loads(p.read_text());assert sha(r['episode'])==r['episode_sha256']
    return r


def preflight():
    path=OUT/'preflight.json'
    if path.exists():return json.loads(path.read_text())
    # JIT and imports are warmed before any timed decision, under one lock.
    stamp=time.perf_counter();warm=[]
    for task in tasks('development')[:5]:
        packet=json.loads(Path(task['packet_path']).read_text())
        own,_=initialize(packet,task['method'],task['seed'])
        own.act(observation(packet['history'][-1]),fixed_simulations=1)
    save(OUT/'warmup.json',dict(seconds=time.perf_counter()-stamp,excluded_from_decision_timing=True))
    for family in ('long_wall','wall_corner','multi_route'):
        sample=next(t for t in tasks('development') if t['family']==family and t['world']=='A' and t['opponent']=='matched')
        packet=json.loads(Path(sample['packet_path']).read_text())
        expected=None
        for method in METHODS:
            own,setup=initialize(packet,method,sample['seed']);own.budget_ms=max(0.,setup['budget_ms']-setup['total_ms'])
            started=time.perf_counter();action=own.act(observation(packet['history'][-1]));ms=1000*(time.perf_counter()-started)
            sim=np.concatenate(own.traces) if own.traces else np.zeros((0,10))
            errors=independent_traces(sim,np.asarray(packet['map']),packet['motion']['pursuer_axis_limit'])
            replay=[]
            for _ in range(2):
                clone,_=initialize(packet,method,sample['seed'])
                a=clone.act(observation(packet['history'][-1]),fixed_simulations=8 if method!='G' else 0)
                replay.append(dict(action=a.tolist(),rng=clone.rng.tolist(),stats={k:clone.last_stats[k] for k in ('root_visits','root_totals','root_counts')},
                    particles=clone.estimator.particles.tolist(),weights=clone.estimator.particle_weights.tolist(),
                    traces=hashlib.sha256(b''.join(t.tobytes() for t in clone.traces)).hexdigest()))
            state=signature(own.estimator)
            if expected is None:expected=state
            record=dict(family=family,method=method,actual_ms=ms,initial_total_ms=setup['total_ms']+ms,
                simulations=own.last_stats['simulations'],root_candidates=own.last_stats['root_candidates'],
                visited_roots=sum(v>0 for v in own.last_stats['root_visits']),
                replay_exact=replay[0]==replay[1],common_D=state==expected,simulation_errors=errors.tolist(),
                simulated_transitions=len(sim),action_legal=legal_action(packet['history'][-1]['position'],action,packet['motion']['pursuer_axis_limit'],np.asarray(packet['map'])))
            warm.append(record);print(json.dumps(dict(stage='preflight',**record)),flush=True)
    result=dict(passed=all(r['replay_exact'] and r['common_D'] and not any(r['simulation_errors']) and r['action_legal'] for r in warm),records=warm,
        engineering_only=True,low_budget_root_coverage_not_a_silent_failure=True,training=False)
    save(path,result);return result


def run_stage(stage):
    requests=tasks(stage);save(OUT/stage/'requests.json',requests)
    rows=[];started=time.perf_counter()
    for i,task in enumerate(requests):
        row=cached(task)
        if row is None:row=commit(task,episode(task))
        rows.append(row)
        status=dict(stage=stage,completed=i+1,total=len(requests),key=task['key'],pid=os.getpid(),
            elapsed_seconds=time.perf_counter()-started,failures=sum(r['failure'] is not None for r in rows),training=False)
        save(OUT/'status.json',status);print(json.dumps(status),flush=True)
        if row['failure']:
            save(OUT/stage/'partial_results.json',rows)
            save(OUT/'decision.json',dict(stop=True,reason='engineering_failure_during_closed_loop',key=row['key'],training=False))
            return None
    save(OUT/stage/'results.json',rows)
    stat=statistics(rows);save(OUT/stage/'statistics.json',stat)
    return stat


def pipeline():
    prepare();freeze()
    if not preflight()['passed']:
        save(OUT/'decision.json',dict(stop=True,reason='preflight_failed',training=False));return
    dev=run_stage('development')
    if dev is None:return
    from audit_search_value_validation import audit
    reviewed=audit(OUT,'development')
    if not reviewed.get('passed',not reviewed.get('errors')):
        save(OUT/'decision.json',dict(stop=True,reason='development_postrun_audit_failed',training=False));return
    proceed=bool(dev['decision']['development_continue'])
    save(OUT/'development_gate.json',dict(continue_to_confirmation=proceed,decision=dev['decision'],confirmation_outcomes_read=False))
    if proceed:
        confirmation=run_stage('confirmation')
        if confirmation is None:return
        checked=audit(OUT,'confirmation')
        if not checked.get('passed',not checked.get('errors')):
            save(OUT/'decision.json',dict(stop=True,reason='confirmation_postrun_audit_failed',training=False));return
        decision=confirmation['decision']
    else:
        decision=dev['decision']
    supported=bool(proceed and decision.get('rootpuct_followup',False))
    save(OUT/'decision.json',dict(stop=not supported,
        confirmation_run=proceed,development_gate_passed=proceed,statistical_decision=decision,
        reason='see_predeclared_grouped_endpoints',neural_training=False,deployment_changed=False))
    write_report(OUT)
    save(OUT/'status.json',dict(stage='complete',development_games=120,confirmation_games=240 if proceed else 0,
        stop=not supported,neural_training=False,deployment_changed=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('prepare','preflight','development','confirmation','pipeline'))
    args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'current/.l45_search_value_timing.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.command=='prepare':prepare()
        elif args.command=='pipeline':pipeline()
        else:
            prepare();freeze()
            if args.command=='preflight':preflight()
            else:
                assert preflight()['passed']
                if args.command=='confirmation':
                    assert json.loads((OUT/'development_gate.json').read_text())['continue_to_confirmation']
                run_stage(args.command)


if __name__=='__main__':main()
