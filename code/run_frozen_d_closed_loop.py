"""Frozen-D real environment capture study, public-history clustered inference.

Estimator/controller sources from the preceding audit are read-only. Each
method's adversary responds to its own state, with no future sharing.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse,collections,fcntl,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
from solvers.frozen_d_search_interface import Controller,METHODS,Observation,sensor_visible,loader,FROZEN
from solvers.root_audit_puct import RootPUCT,cpp
from corner_estimator_scenarios import construct,original_model
from verify_l45_occlusion_scenarios import verify_case
from estimator_audit_geometry import point_legal,blocked
from audit_frozen_d_search_interface import observation,independent_traces,warmup,quantiles

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'current/l45_closed_loop_D_20261026'
OLD=ROOT/'current/l45_corner_turn_estimator_20261025/bank'
BUDGET=200.


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def legal_action(at,action,speed,boxes):
    at=np.asarray(at);action=np.asarray(action)
    return bool(np.all(np.isfinite(action)) and np.max(np.abs(action))<=speed+2e-7
        and point_legal(at,boxes) and point_legal(at+action,boxes) and not blocked(at,at+action,boxes))


def prepare():
    if (OUT/'bank_frozen.json').exists():return
    known=set()
    for path in (ROOT/'current').glob('**/controller_inputs/*.json'):
        data=json.loads(path.read_text())
        if 'map' in data and 'history' in data:
            known.add(hashlib.sha256(json.dumps(dict(map=data['map'],history=data['history']),sort_keys=True).encode()).hexdigest())
    entries=[];checks=[]
    for f,family in enumerate(('long_wall','wall_corner','multi_route')):
        for h,H in enumerate((4,6)):
            for v,speed in enumerate((1.1,1.3)):
                for variant in range(2):
                    seed=int(np.random.SeedSequence([81020261026,f,h,v,variant]).generate_state(1)[0])
                    rng=np.random.default_rng(seed)
                    height=float(rng.uniform(6.2,6.9) if H==4 else rng.uniform(8.15,8.85))
                    env,packet=construct(family,H,speed,seed,height,variant,pause=variant)
                    check=verify_case(env,packet)
                    if not check['passed']:raise RuntimeError(('Independent geometry rejected',env['id'],check['reasons']))
                    fingerprint=hashlib.sha256(json.dumps(dict(map=packet['map'],history=packet['history']),sort_keys=True).encode()).hexdigest()
                    if fingerprint in known:raise RuntimeError('New public history overlaps historical inputs')
                    known.add(fingerprint)
                    for kind,data in (('controller_inputs',packet),('environment',env)):
                        save(OUT/'bank'/kind/(env['id']+'.json'),data)
                    entries.append(dict(id=env['id'],family=family,public=f'controller_inputs/{env["id"]}.json',
                        environment=f'environment/{env["id"]}.json',geometry_history_sha256=fingerprint))
                    checks.append(check)
    save(OUT/'bank/manifest.json',dict(entries=entries,scope='unseen instances of three registered topology families'))
    save(OUT/'bank_acceptance.json',dict(passed=True,checks=checks))
    paths=list((OUT/'bank').rglob('*.json'))+[OUT/'bank_acceptance.json']
    save(OUT/'bank_frozen.json',dict(seed=81020261026,groups=24,outcomes_read=False,files={str(p):sha(p) for p in paths}))


def development_entries():
    return json.loads((OLD/'manifest.json').read_text())['entries'][::2]


def freeze():
    names=('run_frozen_d_closed_loop.py','L45_CLOSED_LOOP_D_20261026.md','tests/test_frozen_d_closed_loop.py',
        'solvers/frozen_d_search_interface.py','solvers/root_audit_puct.py','solvers/c_puct.py','problems/example8.py',
        'corner_estimator_scenarios.py','estimator_audit_geometry.py','verify_l45_occlusion_scenarios.py','audit_frozen_d_search_interface.py')
    paths=[ROOT/'code'/name for name in names]+[Path(cpp.__file__),FROZEN/'manifest.json']
    for base,entries in ((OLD,development_entries()),(OUT/'bank',json.loads((OUT/'bank/manifest.json').read_text())['entries'])):
        paths.append(base/'manifest.json')
        for entry in entries:paths.extend(base/entry[k] for k in ('public','environment'))
    protected=[ROOT/'code/param.py',ROOT/'code/cpp/bindings.cpp',ROOT/'code/cpp/problems/example8.hpp']
    config=dict(protocol='frozen_D_closed_loop_capture_v1',budget_ms=BUDGET,high_level_dt_s=1.,
        methods=list(METHODS),simulations=1000,planning_depth=8,opponent='original RootPUCT / 3000 / depth100 / no network',
        development_groups=12,confirmation_groups=24,development_games=72,confirmation_games=144,
        sources_and_inputs={str(p):sha(p) for p in paths},protected_before={str(p):sha(p) for p in protected},
        bank_frozen_sha256=sha(OUT/'bank_frozen.json'),bootstrap_draws=10000,bootstrap_seed=81120261026,
        frozen_D_changed=False,deployment_changed=False,training=False,paired_unit='public history; retain A/B/all methods',
        old_50ms_gate_record_preserved=True,user_authorized_budget_revision=True)
    path=OUT/'experiment.json'
    if path.exists():
        if json.loads(path.read_text())!=config:raise RuntimeError('Frozen execution protocol/source changed')
    else:save(path,config)
    loader.verify()
    return config


def tasks(stage):
    entries=development_entries() if stage in ('engineering','development') else json.loads((OUT/'bank/manifest.json').read_text())['entries']
    if stage=='engineering':entries=entries[::4]
    result=[]
    for i,entry in enumerate(entries):
        for wi,world in enumerate(('A',) if stage=='engineering' else ('A','B')):
            seeds=np.random.SeedSequence([81220261026,0 if stage!='confirmation' else 1,
                int(entry['geometry_history_sha256'][:8],16),wi]).generate_state(3)
            offset=(i+wi)%3
            for method in METHODS[offset:]+METHODS[:offset]:
                result.append(dict(key=entry['id']+'_'+world+'_'+method.replace('+','plus'),entry=entry,
                    stage=stage,world=world,method=method,seed=int(seeds[0]),opponent_seed=int(seeds[1]),environment_seed=int(seeds[2])))
    return result


def inputs(task):
    base=OUT/'bank' if task['stage']=='confirmation' else OLD
    entry=task['entry']
    return json.loads((base/entry['public']).read_text()),json.loads((base/entry['environment']).read_text())


def initialize(packet,method,seed):
    started=time.perf_counter()
    own=Controller(packet['map'],packet['motion']['pursuer_axis_limit'],method,seed,budget_ms=BUDGET,simulations=1000)
    static_ms=1000*(time.perf_counter()-started)
    started=time.perf_counter()
    for item in packet['history'][:-1]:own.estimator.update(observation(item))
    return own,dict(static_setup_ms=static_ms,setup_breakdown=own.setup,
        history_initialization_ms=1000*(time.perf_counter()-started))


def actor_signature(own):
    e=own.estimator
    return hashlib.sha256(e.particles.tobytes()+e.particle_weights.tobytes()+e.rng.tobytes()).hexdigest()


def episode(task,limit=None,fixed=None):
    packet,environment=inputs(task)
    p=original_model(packet['map'],packet['motion']['pursuer_axis_limit'])
    world=next(w for w in environment['worlds'] if w['id']==task['world'])
    state=np.asarray(world['handoff_full_state'],float).reshape(-1,1)
    own,initialization=initialize(packet,task['method'],task['seed'])
    stamp=time.perf_counter()
    other=RootPUCT([None]*4,None,3000,0.,0.)
    other.seeded_state(p,task['opponent_seed'],task['environment_seed'])
    initialization['opponent_setup_ms']=1000*(time.perf_counter()-stamp)
    boxes=np.asarray(packet['map'],float);trace=[];failure=None;first_seen=None
    original_remaining=80-int(state[8,0]);initial_state=state[:,0].tolist()
    for index in range(min(original_remaining,limit) if limit else original_remaining):
        cycle_start=time.perf_counter()
        seen=sensor_visible(state[4:6,0],state[:2,0],boxes)
        obs=Observation(int(round(state[8,0])),state[4:6,0].copy(),state[:2,0].copy() if seen else None,80-int(round(state[8,0])))
        before_rng=own.rng.copy();started=time.perf_counter()
        try:
            action=own.act(obs,fixed_simulations=None if fixed is None else fixed[index])
        except Exception as exc:
            failed_ms=1000*(time.perf_counter()-started)
            failure=dict(reason='controller_failure',error=repr(exc),step=obs.step,observation=dict(step=obs.step,
                position=obs.position.tolist(),target=None if obs.target is None else obs.target.tolist(),remaining=obs.remaining),
                failed_call_ms=failed_ms,state=state[:,0].tolist())
            save(OUT/task['stage']/'failures'/(task['key']+'.json'),failure)
            break
        own_ms=1000*(time.perf_counter()-started);stats=own.last_stats.copy()
        # True state enters only the environment, actual adversary and scorer.
        # The adversary sees the same pre-action state, never own action.
        started=time.perf_counter();adversary=other.search(p,state.copy(),turn=0)
        raw=np.asarray(adversary.best_action).reshape(-1)[:2].astype(float)
        shield=not legal_action(state[:2,0],raw,1.,boxes)
        other_action=np.zeros(2) if shield else raw.copy()
        opponent_ms=1000*(time.perf_counter()-started)
        lawful=legal_action(state[4:6,0],action,own.public.speed,boxes)
        other_lawful=legal_action(state[:2,0],other_action,1.,boxes)
        if not lawful or not other_lawful:
            failure=dict(reason='own_illegal' if not lawful else 'opponent_illegal',step=obs.step)
        joint=np.zeros((p.action_dim,1));joint[4:6,0]=action;joint[:2,0]=other_action
        if failure is None:
            started=time.perf_counter();nxt=p.step(state,joint,p.dt)
            environment_ms=1000*(time.perf_counter()-started)
            # Original direct-motion environment must apply both actions once.
            if not (np.allclose(nxt[4:6,0],state[4:6,0]+action,rtol=0,atol=1e-10)
                    and np.allclose(nxt[:2,0],state[:2,0]+other_action,rtol=0,atol=1e-10)):
                failure=dict(reason='environment_motion_mismatch',step=obs.step)
        else:nxt=state.copy();environment_ms=0.
        post_seen=bool(sensor_visible(nxt[4:6,0],nxt[:2,0],boxes))
        cycle_ms=1000*(time.perf_counter()-cycle_start)
        # Independent audit and scoring start after actions and transition.
        simulated=np.concatenate(own.traces) if own.traces else np.zeros((0,10))
        errors=independent_traces(simulated,boxes,own.public.speed)
        if np.any(errors) and failure is None:failure=dict(reason='simulation_geometry_failure',step=obs.step,errors=errors.tolist())
        capture=bool(nxt[9,0]<.5 and nxt[11,0]<.5)
        if capture and not (np.linalg.norm(nxt[4:6,0]-nxt[:2,0])<1. and lawful and other_lawful):
            failure=dict(reason='invalid_capture',step=obs.step)
        if post_seen and first_seen is None:first_seen=index+1
        e=own.estimator;near=float(np.min(np.linalg.norm(e.particles-state[:2,0],axis=1)))
        record=dict(step=obs.step,state=state[:,0].tolist(),next_state=nxt[:,0].tolist(),
            observation=dict(step=obs.step,position=obs.position.tolist(),target=None if obs.target is None else obs.target.tolist(),remaining=obs.remaining),
            own_action=action.tolist(),opponent_raw=raw.tolist(),opponent_action=other_action.tolist(),opponent_shield=shield,
            particles=e.particles.tolist(),weights=e.particle_weights.tolist(),estimator_sha256=actor_signature(own),
            search_rng_before=before_rng.tolist(),search_rng_after=own.rng.tolist(),
            own_ms=own_ms,opponent_ms=opponent_ms,environment_ms=environment_ms,cycle_ms=cycle_ms,
            statistics=stats,post_seen=post_seen,capture=capture,own_legal=lawful,opponent_legal=other_lawful,
            simulated_transitions=len(simulated),simulation_errors=errors.tolist(),
            simulation_sha256=hashlib.sha256(simulated.tobytes()).hexdigest(),nearest_particle_error=near)
        trace.append(record);state=nxt
        if failure or capture or p.is_terminal(state):break
    capture=bool(trace and trace[-1]['capture'] and failure is None)
    reason='capture' if capture else failure['reason'] if failure else 'engineering_prefix' if limit and len(trace)==limit else 'timeout'
    hidden=[r for r in trace if r['observation']['target'] is None]
    lat=[r['own_ms'] for r in trace]
    positions=np.array([r['state'][4:6] for r in trace[-8:]])
    stationary=bool(len(positions)>=8 and np.max(np.ptp(positions,axis=0))<.05)
    lost_again=sum(bool(trace[i-1]['post_seen'] and not trace[i]['post_seen']) for i in range(1,len(trace)))
    row=dict(key=task['key'],history_id=task['entry']['id'],family=task['entry']['family'],world=task['world'],method=task['method'],
        stage=task['stage'],capture=int(capture),rediscovered=int(first_seen is not None),first_rediscovery=first_seen,
        executed_steps=len(trace),completion_cost=len(trace) if capture else original_remaining,remaining_at_handoff=original_remaining,
        reason=reason,failure=failure,initialization=initialization,
        cold_start_first_decision_ms=sum(initialization[k] for k in ('static_setup_ms','history_initialization_ms'))+(lat[0] if lat else 0.),
        latencies_ms=lat,cycle_ms=[r['cycle_ms'] for r in trace],opponent_ms=[r['opponent_ms'] for r in trace],
        over_50ms=sum(x>50 for x in lat),over_200ms=sum(x>200 for x in lat),over_high_level_period=sum(r['cycle_ms']>1000 for r in trace),
        shield_count=sum(r['opponent_shield'] for r in trace),hidden_steps=len(hidden),hidden_covered=sum(r['nearest_particle_error']<=1 for r in hidden),
        hidden_simulations=sum(r['statistics']['simulations'] for r in hidden),lost_again=lost_again,stationary_last8=stationary,
        simulated_transitions=sum(r['simulated_transitions'] for r in trace),simulation_errors=np.sum([r['simulation_errors'] for r in trace],axis=0).tolist() if trace else [0]*4,
        trajectory_sha256=hashlib.sha256(json.dumps(trace,sort_keys=True).encode()).hexdigest())
    return dict(task=task,initial_state=initial_state,final_state=state[:,0].tolist(),summary=row,trace=trace)


def commit(task,payload):
    folder=OUT/task['stage'];path=folder/'episodes'/(task['key']+'.json')
    save(path,payload)
    row=payload['summary']|dict(episode=str(path),episode_sha256=sha(path))
    save(folder/'cache'/(task['key']+'.json'),row)
    return row


def cached(task):
    path=OUT/task['stage']/'cache'/(task['key']+'.json')
    if not path.exists():return None
    row=json.loads(path.read_text())
    if sha(row['episode'])!=row['episode_sha256']:raise RuntimeError('Cached trajectory changed')
    return row


def preflight():
    if (OUT/'preflight.json').exists():return json.loads((OUT/'preflight.json').read_text())
    started=time.perf_counter();warmup()
    checks=[]
    for entry in development_entries()[::4]:
        packet=json.loads((OLD/entry['public']).read_text());states=[]
        for method in METHODS:
            own,_=initialize(packet,method,17)
            own.estimator.update(observation(packet['history'][-1]));states.append(actor_signature(own))
        checks.append(dict(history=entry['id'],common_estimator=len(set(states))==1))
    rows=[];replay=[]
    for task in tasks('engineering'):
        row=cached(task)
        payload=None
        if row is None:
            payload=episode(task,limit=6);row=commit(task,payload)
        rows.append(row)
        if payload is None:payload=json.loads(Path(row['episode']).read_text())
        second=episode(task,limit=len(payload['trace']),fixed=[r['statistics']['simulations'] for r in payload['trace']])
        fields=('state','next_state','observation','own_action','opponent_raw','opponent_action','particles','weights',
            'estimator_sha256','search_rng_before','search_rng_after','simulation_sha256')
        exact=len(second['trace'])==len(payload['trace']) and all(a[k]==b[k] for a,b in zip(payload['trace'],second['trace']) for k in fields)
        replay.append(dict(key=task['key'],exact=exact))
        print(json.dumps(dict(stage='engineering',completed=len(rows),total=9,replay=exact)),flush=True)
    interface=all(c['common_estimator'] for c in checks) and all(r['failure'] is None for r in rows) and all(r['exact'] for r in replay)
    lat=[v for r in rows for v in r['latencies_ms']];cycle=[v for r in rows for v in r['cycle_ms']]
    result=dict(passed=interface,common_history_checks=checks,replays=replay,rows=rows,
        budget_ms=BUDGET,latency=quantiles(lat),cycle=quantiles(cycle),elapsed_seconds=time.perf_counter()-started,
        informational_budget_gate='all methods same200ms; disclose overruns, do not silently exclude games',
        formal_development_started=False)
    save(OUT/'preflight.json',result)
    return result


def run_stage(stage):
    rows=[];started=time.perf_counter();requests=tasks(stage)
    save(OUT/stage/'requests.json',requests)
    for i,task in enumerate(requests):
        row=cached(task)
        if row is None:row=commit(task,episode(task))
        rows.append(row)
        status=dict(stage=stage,completed=i+1,total=len(requests),key=task['key'],
            captures=sum(r['capture'] for r in rows),controller_failures=sum(r['failure'] is not None for r in rows),
            elapsed_seconds=time.perf_counter()-started,pid=os.getpid(),training=False)
        save(OUT/'status.json',status);print(json.dumps(status),flush=True)
    save(OUT/stage/'results.json',rows)
    result=statistics(rows);save(OUT/stage/'statistics.json',result)
    return result


def aggregate(rows):
    latency=[v for r in rows for v in r['latencies_ms']]
    return dict(games=len(rows),capture=sum(r['capture'] for r in rows),capture_rate=float(np.mean([r['capture'] for r in rows])),
        rediscovered=sum(r['rediscovered'] for r in rows),rediscovery_rate=float(np.mean([r['rediscovered'] for r in rows])),
        completion_cost=float(np.mean([r['completion_cost'] for r in rows])),
        success_steps=float(np.mean([r['executed_steps'] for r in rows if r['capture']])) if any(r['capture'] for r in rows) else None,
        first_rediscovery_mean=float(np.mean([r['first_rediscovery'] for r in rows if r['rediscovered']])) if any(r['rediscovered'] for r in rows) else None,
        failure_reasons=dict(collections.Counter(r['reason'] for r in rows)),latency=quantiles(latency),
        cycle=quantiles([v for r in rows for v in r['cycle_ms']]),opponent_latency=quantiles([v for r in rows for v in r['opponent_ms']]),
        cold_start_first_decision=quantiles([r['cold_start_first_decision_ms'] for r in rows]),
        over_200ms=sum(r['over_200ms'] for r in rows),over_high_level_period=sum(r['over_high_level_period'] for r in rows),
        shield_count=sum(r['shield_count'] for r in rows),stationary_failures=sum(not r['capture'] and r['stationary_last8'] for r in rows),
        hidden_coverage=sum(r['hidden_covered'] for r in rows)/max(1,sum(r['hidden_steps'] for r in rows)),
        simulated_transitions=sum(r['simulated_transitions'] for r in rows))


def statistics(rows):
    histories=sorted({r['history_id'] for r in rows})
    index={(r['history_id'],r['world'],r['method']):r for r in rows}
    if len(index)!=len(histories)*6:raise RuntimeError('Missing paired groups')
    draws=np.random.default_rng(81120261026).integers(0,len(histories),(10000,len(histories)))
    comparisons={};bootstraps={}
    for a,b in (('T','G'),('T+Geo','T'),('T+Geo','G')):
        comparison={}
        for metric in ('capture','rediscovered','completion_cost'):
            delta=np.array([np.mean([index[(h,w,a)][metric]-index[(h,w,b)][metric] for w in ('A','B')]) for h in histories])
            boot=delta[draws].mean(1)
            comparison[metric]=dict(mean=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
                positive_groups=int((delta>0).sum()),negative_groups=int((delta<0).sum()))
            if metric=='capture':bootstraps[a+'-'+b]=boot
        comparisons[a+'-'+b]=comparison
    main=('T-G','T+Geo-T')
    centers=np.array([comparisons[k]['capture']['mean'] for k in main])
    radius=float(np.quantile(np.max(np.abs(np.stack([bootstraps[k] for k in main],1)-centers),axis=1),.95))
    for k,center in zip(main,centers):
        comparisons[k]['capture']['simultaneous_ci95']=[float(center-radius),float(center+radius)]
        method=k.split('-')[0]
        no_failure=all(r['failure'] is None for r in rows if r['method']==method)
        comparisons[k]['reliable_extra_capture_value']=bool(center-radius>0 and comparisons[k]['completion_cost']['ci95'][1]<=0 and no_failure)
    return dict(groups=len(histories),games=len(rows),summary={m:aggregate([r for r in rows if r['method']==m]) for m in METHODS},
        comparisons=comparisons,by_family={f:{m:aggregate([r for r in rows if r['method']==m and r['family']==f]) for m in METHODS}
            for f in ('long_wall','wall_corner','multi_route')},bootstrap_seed=81120261026,bootstrap_draws=10000,
        cluster_unit='public history; both hidden initial worlds retained',simultaneous_main_radius=radius)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','preflight','pipeline','statistics'))
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'current/.l45_local_pursuit_serial.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='prepare':prepare();freeze();return
        freeze()
        if args.command=='statistics':
            for stage in ('development','confirmation'):
                path=OUT/stage/'results.json'
                if path.exists():save(OUT/stage/'statistics.json',statistics(json.loads(path.read_text())))
            return
        if not preflight()['passed']:raise RuntimeError('Closed-loop interface preflight failed; effects not run')
        if args.command=='preflight':return
        development=run_stage('development')
        confirmation=run_stage('confirmation')
        config=freeze()
        decision=dict(complete=True,development=development['comparisons'],confirmation=confirmation['comparisons'],
            estimator_frozen=bool(loader.verify()),protected_files_unchanged=all(sha(p)==h for p,h in config['protected_before'].items()),
            reliable_T_over_G=confirmation['comparisons']['T-G']['reliable_extra_capture_value'],
            reliable_TGeo_over_T=confirmation['comparisons']['T+Geo-T']['reliable_extra_capture_value'],
            training_started=False,deployment_changed=False)
        save(OUT/'decision.json',decision)
        save(OUT/'status.json',dict(stage='complete',completed=216,total=216,pid=None,training=False,deployment_changed=False))


if __name__=='__main__':main()
