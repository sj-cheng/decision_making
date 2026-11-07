"""Frozen, serial local-observation mechanism experiment; no training or adoption."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import numpy as np
import torch
import numba
import run_l45_direction_validation as prior
from solvers.root_audit_puct import RootPUCT, cpp
from solvers.local_pursuit import (Controller,PublicMap,Observation,warmup,legal,clear,visible,safe,
                                  modal,predict,recover,guided,route,move)

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT/'code'
DEFAULT_OUTPUT = ROOT/'current/l45_local_pursuit_20261024'
SMOKE_OUTPUT = ROOT/'current/l45_local_pursuit_smoke_20261024'
METHODS = ('L','G','T','V','O')
MAPS = dict(open=[],corner=[[[-1.5,1.5],[-3.5,3.5]]],
            multipath=[[[-4.,-2.],[-4.5,4.5]],[[2.,4.],[-4.5,4.5]]])
PAIRS = (('G','L'),('T','G'),('V','T'),('V','G'))
o = prior.o
lock_json, locked_read = prior.lock_json, prior.locked_read


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return o.json_hash(value)


def config(smoke=False):
    return dict(schema=1,smoke=smoke,maps=MAPS,methods=list(METHODS),particles=64,
        scenarios_per_map=1 if smoke else 20,streams=1 if smoke else 2,
        namespace=6002026102401 if smoke else 6002026102402,bootstrap_seed=60020261024,
        bootstrap_draws=10000,range=6.,capture_radius=1.,horizon=80,depth=100,
        deadline_ms=0. if smoke else 50.,simulation_cap=3 if smoke else 1000,
        opponent_simulations=3 if smoke else 3000,opponent_depth=100,
        ratios=[1.1,1.3],noise_fraction=.1,root_mixture=[.5,.5],
        UCT=1.,progressive_widening=[2.,.5],gamma=1.,value_cutoff=False,neural=False,
        simultaneous=True,swept_collision=True,training_episodes=0,
        recovery='sequential_legal_paths_all_empty_observations;512_auxiliary;8_restarts;64_posterior',
        simulated_opponent='independent_evasive_geometric50_uniform50;full_position_sensor',
        exit_grouping='first_turn_of_pursuer_to_hypothesis_shortest_step_path',
        reobserve_nodes='obstacle_outside_corners_and_8_boundary_nodes;offset=.05',
        gate=dict(gain=.05,own_illegal_margin=.02,open_regression=.05),
        serial=True,threads=1,adoption=False,recommended='old0:novalue')


def fingerprints():
    files = [Path(__file__),CODE/'solvers/local_pursuit.py',CODE/'tests/test_l45_local_pursuit.py',
             CODE/'L45_LOCAL_PURSUIT.md',CODE/'local_observation.py',CODE/'problems/example8.py',
             CODE/'solvers/c_puct.py',CODE/'solvers/root_audit_puct.py',Path(cpp.__file__),
             CODE/'cpp/build_root_audit/build_manifest.json']
    return {str(p):sha(p) for p in files}


def scenario(seed, map_id, ratio):
    rng = np.random.default_rng(seed)
    public = PublicMap(MAPS[map_id],ratio)
    count = np.zeros(2,np.int64)
    while True:
        pursuer = rng.uniform(-8.8,8.8,2)
        angle, distance = rng.uniform(0,2*np.pi),rng.uniform(3.,6.)
        target = pursuer+distance*np.array([np.cos(angle),np.sin(angle)])
        if legal(pursuer,public.boxes,count) and legal(target,public.boxes,count) and visible(pursuer,target,public.boxes,count):
            break
    p = model(map_id,ratio)
    state = np.zeros((p.state_dim,1))
    state[p.state_idxs[0],0] = target
    state[p.state_idxs[1],0] = [-9.8,-9.8]
    state[p.state_idxs[2],0] = pursuer
    state[p.state_idxs[3],0] = [9.8,9.8]
    state[p.active_idxs,0] = [1.,0.,1.,0.]
    spec = dict(seed=int(seed),map_id=map_id,obstacles=MAPS[map_id],ratio=ratio,
        initial_state=state.tolist(),action_lims=p.action_lims.tolist(),control_lims=p.control_lims.tolist(),
        evader_speed=1.,pursuer_speed=ratio,
        streams=[dict(controller=int(rng.integers(1,2**32-1)),opponent=int(rng.integers(1,2**32-1)),
                      environment=int(rng.integers(1,2**31-1))) for _ in range(2)])
    spec['episode_hash'] = digest(spec)
    return spec


def model(map_id, ratio):
    from problems.problem import get_problem
    p = get_problem('example8')
    p.obstacles = [np.array(b,float) for b in MAPS[map_id]]
    p.use_minco_dynamics = False
    p.swept_collision = True
    p.current_evader_speed_lim,p.current_pursuer_speed_lim = 1.,ratio
    for r in range(4):
        lim = 1. if r < 2 else ratio
        p.action_lims[p.action_idxs[r]] = [[-lim,lim],[-lim,lim]]
        p.control_lims[p.action_idxs[r]] = p.action_lims[p.action_idxs[r]]/p.dt
    assert p.tf == 80 and p.dt == 1 and p.desired_distance == 1.
    assert p.pursuer_progress_reward_weight == .03 and p.evader_progress_reward_weight == .2
    return p


def prepare(args,smoke=False):
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    if (out/'experiment.lock.json').exists():
        verify(args,smoke)
        return locked_read(out/'scenarios.json')
    c = config(smoke)
    files,used,_,occurrences = prior.history(args)
    before = len(used)
    banks,engineering = {},{}
    for mi,map_id in enumerate(MAPS):
        banks[map_id] = []
        i = 0
        while len(banks[map_id]) < c['scenarios_per_map']:
            ratio = c['ratios'][len(banks[map_id])%2]
            seed = int(np.random.SeedSequence([c['namespace'],mi,i]).generate_state(1)[0]);i+=1
            spec = scenario(seed,map_id,ratio)
            physical = prior.physical_hash(spec)
            if physical not in used:
                used.add(physical);banks[map_id].append(spec)
        engineering[map_id] = scenario(int(np.random.SeedSequence([6002026102499,mi]).generate_state(1)[0]),map_id,1.1)
        if prior.physical_hash(engineering[map_id]) in used:
            raise ValueError('Engineering scene overlaps evaluation/history')
    lock_json(out/'scenarios.json',banks)
    lock_json(out/'engineering.json',engineering)
    lock_json(out/'scenario_independence.json',dict(previous_physical_states=before,previous_occurrences=occurrences,
        new_physical_states=sum(map(len,banks.values())),physical_overlap=0,
        generation='geometry_only; no_controller_outcome_screening',historical_files={str(p):sha(p) for p in files},
        scope='all recoverable prior scenario/case/bank/rollout JSON, including smoke; physical positions and action limits',
        initial_velocity_acceleration='all zero; inactive robots parked legally'))
    runtime = out/'runtime';runtime.mkdir(exist_ok=True)
    shutil.copyfile(Path(cpp.__file__),runtime/Path(cpp.__file__).name)
    shutil.copytree(CODE/'cpp/build_root_audit/source',runtime/'opponent_source',dirs_exist_ok=True)
    for path in (Path(__file__),CODE/'solvers/local_pursuit.py',CODE/'L45_LOCAL_PURSUIT.md'):
        shutil.copyfile(path,runtime/path.name)
    record = dict(config=c,implementation=fingerprints(),scenarios_sha256=sha(out/'scenarios.json'),
        engineering_sha256=sha(out/'engineering.json'),model_fingerprints={},
        model_note='No model is loaded by any method or Search Only opponent.',
        runtime=dict(python=sys.version,numpy=np.__version__,torch=str(torch.__version__),numba=numba.__version__,
                     platform=platform.platform(),executable=sys.executable,affinity=sorted(os.sched_getaffinity(0))))
    lock_json(out/'experiment.json',record)
    return banks


def verify(args,smoke=False):
    record = locked_read(args.output/'experiment.json')
    if record['config'] != config(smoke) or record['implementation'] != fingerprints():
        raise ValueError('Frozen protocol or implementation changed; refuse mixed results')
    if record['scenarios_sha256'] != sha(args.output/'scenarios.json') or record['engineering_sha256'] != sha(args.output/'engineering.json'):
        raise ValueError('Frozen scenarios changed')
    return record


def items(args,smoke=False):
    banks = locked_read(args.output/'scenarios.json')
    result = []
    for mi,bank in enumerate(banks.values()):
        for si,spec in enumerate(bank):
            rotation = (si+mi)%5
            for stream in range(config(smoke)['streams']):
                for method in METHODS[rotation:]+METHODS[:rotation]:
                    identity = dict(protocol=sha(args.output/'experiment.json'),scene=spec['episode_hash'],stream=stream,method=method)
                    result.append(dict(**identity,key=digest(identity),spec=spec))
    return result


def setup(item,smoke=False):
    spec,cfg = item['spec'],config(smoke)
    if digest({k:v for k,v in spec.items() if k!='episode_hash'}) != spec['episode_hash']:
        raise ValueError('Scenario identity changed')
    p = model(spec['map_id'],spec['ratio'])
    state = np.array(spec['initial_state'],float)
    stream = spec['streams'][item['stream']]
    own = Controller(PublicMap(spec['obstacles'],spec['ratio']),item['method'],stream['controller'],
                     cfg['simulation_cap'],cfg['deadline_ms'])
    other = RootPUCT([None]*4,None,cfg['opponent_simulations'],0.,0.)
    other.seeded_state(p,stream['opponent'],stream['environment'])
    return p,state,own,other


def snapshot(p,state,own,other,trace):
    return dict(state=state.copy(),controller=own.snapshot(),opponent=other.get_rng_state(p),trace=copy.deepcopy(trace))


def save_checkpoint(args,item,p,state,own,other,trace):
    path = args.output/'pending'/(item['key']+'.pt')
    o.save_torch(path,dict(key=item['key'],snapshot=snapshot(p,state,own,other,trace)))


def load_pending(args,item,p,own,other):
    path = args.output/'pending'/(item['key']+'.pt')
    if not path.exists(): return None
    record = o.load_torch(path)
    if record['key'] != item['key']: raise ValueError('Pending identity changed')
    snap = record['snapshot']
    own.restore(snap['controller']);other.set_rng_state(p,snap['opponent'])
    return snap['state'].copy(),copy.deepcopy(snap['trace'])


def observation(own,p,state):
    pr,en = state[p.state_idxs[2],0],state[p.state_idxs[0],0]
    obs = own.public.observe(int(round(state[p.time_idx,0])),pr,en)
    if own.method == 'O': obs.target = en.copy()
    return obs


def one_step(p,state,own,other,fixed=None):
    obs = observation(own,p,state)
    action = np.zeros((p.action_dim,1))
    selected = own.act(obs,fixed_simulations=fixed)
    action[p.action_idxs[2],0] = selected
    own_stats = copy.deepcopy(own.last_stats)
    # Same pre-decision state; no action selected above is given to the opponent.
    stamp = time.perf_counter()
    result = other.search(p,state.copy(),turn=0)
    action[p.action_idxs[0],0] = np.asarray(result.best_action).reshape(-1)[p.action_idxs[0]]
    opponent_ms = (time.perf_counter()-stamp)*1000
    native = other._problem_wrapper_for(p)
    nxt = np.asarray(native.step(state,action,p.dt),dtype=float).reshape(-1,1)
    reward = np.asarray(native.transition_reward(state,action,nxt)).reshape(-1).tolist()
    legal_own = safe(state[p.state_idxs[2],0],action[p.action_idxs[2],0],own.public.speed,own.public.boxes,np.zeros(2,np.int64))
    legal_other = safe(state[p.state_idxs[0],0],action[p.action_idxs[0],0],1.,own.public.boxes,np.zeros(2,np.int64))
    step = dict(state=state.copy(),action=action,next_state=nxt.copy(),reward=reward,observation=copy.deepcopy(obs),
        particles=own.particles.copy(),belief_rng=own.filter_rng.copy(),controller_rng=own.rng.copy(),
        own=own_stats,opponent_ms=opponent_ms,opponent_simulations=int(other.solver_settings.number_simulations),
        own_illegal=not bool(legal_own),opponent_illegal=not bool(legal_other),
        coverage_1m=bool(np.min(np.linalg.norm(own.particles-state[p.state_idxs[0],0],axis=1))<=1.),
        posterior_target_distance=float(np.min(np.linalg.norm(own.particles-state[p.state_idxs[0],0],axis=1))))
    return nxt,step


def episode(args,item,smoke=False,checkpoint=True,stop_after=None):
    p,state,own,other = setup(item,smoke)
    trace = []
    if checkpoint:
        pending = load_pending(args,item,p,own,other)
        if pending:state,trace = pending
    initial = snapshot(p,state,own,other,trace) if not trace else None
    started = time.perf_counter()
    while not p.is_terminal(state):
        state,step = one_step(p,state,own,other)
        trace.append(step)
        if checkpoint:save_checkpoint(args,item,p,state,own,other,trace)
        if stop_after is not None and len(trace)>=stop_after:
            return dict(interrupted=True,steps=len(trace))
    row = measure(item,p,state,trace)
    row['wall_seconds_this_run'] = time.perf_counter()-started
    detail = dict(key=item['key'],scenario=item['spec'],method=item['method'],stream=item['stream'],
        initial=initial,trajectory=trace,final=snapshot(p,state,own,other,[]))
    row['_record'] = detail
    return row


def measure(item,p,state,trace):
    own_illegal,opponent_illegal = any(t['own_illegal'] for t in trace),any(t['opponent_illegal'] for t in trace)
    capture = bool(not p.is_active(state,0) and not own_illegal and not opponent_illegal)
    losses,reacquired,lost,intervals = 0,0,0,[]
    duration = 0
    for t in trace:
        seen = t['observation'].target is not None
        if not seen:
            if duration == 0:losses+=1
            duration+=1;lost+=1
        elif duration:
            reacquired+=1;intervals.append(duration);duration=0
    if duration:intervals.append(duration)
    reason = 'capture' if capture else 'own_illegal' if own_illegal else 'opponent_illegal' if opponent_illegal else 'timeout'
    hidden = [t for t in trace if t['observation'].target is None]
    latencies = [t['own']['latency_ms'] for t in trace]
    sources = Counter(t['own']['source'] for t in trace)
    return dict(key=item['key'],scene_id=item['spec']['episode_hash'],map_id=item['spec']['map_id'],ratio=item['spec']['ratio'],
        stream=item['stream'],method=item['method'],legal_capture=int(capture),own_illegal=int(own_illegal),
        opponent_illegal=int(opponent_illegal),failure_reason=reason,steps=len(trace),
        completion_cost=len(trace) if capture else 80,lost_steps=lost,loss_events=losses,reacquisitions=reacquired,
        loss_durations=intervals,coverage_1m=sum(t['coverage_1m'] for t in hidden),hidden_observations=len(hidden),
        recovery_count=trace[-1]['own']['recoveries'] if trace else 0,latency_ms=latencies,
        opponent_ms=[t['opponent_ms'] for t in trace],overruns=sum(t['own']['overrun'] for t in trace),
        simulations=[t['own']['simulations'] for t in trace],transitions=sum(t['own']['transitions'] for t in trace),
        geometry_queries=sum(t['own']['geometry_queries'] for t in trace),selected_sources=dict(sources),
        trajectory_sha256=digest([dict(state=t['state'].tolist(),action=t['action'].tolist(),next_state=t['next_state'].tolist()) for t in trace]))


def commit(args,row):
    # Reuse the established artifact + metric + checksum commit order.
    prior.commit_row(args,row)


def cached(args,item):
    marker = args.output/'cache_checksums'/(item['key']+'.json')
    if not marker.with_suffix('.lock.json').exists():return None
    expected = locked_read(marker)['sha256']
    path = args.output/'cache'/(item['key']+'.json')
    if sha(path) != expected:raise ValueError('Committed row changed')
    row = o.read_json(path)
    if row['key'] != item['key']:raise ValueError('Cache identity changed')
    if sha(row['episode_record']['path']) != row['episode_record']['sha256']:raise ValueError('Committed episode changed')
    if digest({k:v for k,v in row.items() if k!='content_sha256'}) != row['content_sha256']:
        raise ValueError('Committed metric digest changed')
    return row


@contextmanager
def serial_lock(args):
    # Covers all this pipeline's processes/outputs, including smoke/profile.
    with (ROOT/'current/.l45_local_pursuit_serial.lock').open('a+') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another local-pursuit timing process is active')
        yield


def evaluate(args,smoke=False):
    verify(args,smoke)
    if not smoke:
        for name in ('checks.json','smoke_acceptance.json','profile.json'):
            if not (args.output/name).exists() or not o.read_json(args.output/name)['passed']:
                raise ValueError('Preflight required: '+name)
    tasks = items(args,smoke)
    assert len(tasks) == (15 if smoke else 600)
    lock_json(args.output/'requests.json',[{k:v for k,v in t.items() if k!='spec'} for t in tasks])
    start = time.perf_counter()
    rows,reused = [],0
    for i,item in enumerate(tasks):
        row = cached(args,item)
        if row is None:
            row = episode(args,item,smoke)
            commit(args,row)
        else:reused+=1
        rows.append(row)
        status = dict(stage='smoke' if smoke else 'evaluate',completed=i+1,total=len(tasks),reused=reused,
            elapsed_seconds=time.perf_counter()-start,current=dict(map=item['spec']['map_id'],method=item['method']),
            pid=os.getpid(),updated_unix=time.time())
        o.write_json(args.output/'status.json',status)
        print(json.dumps(status),flush=True)
    lock_json(args.output/'results.json',rows)
    return rows


def profile(args):
    verify(args)
    if (args.output/'profile.json').exists():return o.read_json(args.output/'profile.json')
    warmup()
    records=[]
    engineering = locked_read(args.output/'engineering.json')
    for spec in engineering.values():
        for method in METHODS:
            item=dict(key='profile_'+spec['map_id']+'_'+method,spec=spec,method=method,stream=0)
            profile_args=argparse.Namespace(output=args.output/"engineering_episodes")
            row=episode(profile_args,item,checkpoint=True)
            detail=row.pop("_record")
            o.save_torch(profile_args.output/(item["key"]+".pt"),detail)
            records.append(row)
            print(json.dumps(dict(stage='profile',map=spec['map_id'],method=method,seconds=row['wall_seconds_this_run'],steps=row['steps'])),flush=True)
    estimate = sum(r['wall_seconds_this_run'] for r in records)*40
    result=dict(passed=True,engineering_only=True,games=15,records=records,estimated_formal_seconds=estimate,
        estimated_formal_minutes=estimate/60,formula='3_maps * 5_methods full engineering games * 20 scenes/map * 2 streams',
        fixed_algorithms=True,warning='Throughput estimate only; initial engineering capture times vary.')
    lock_json(args.output/'profile.json',result)
    print(json.dumps(dict(stage='estimate',minutes=estimate/60)),flush=True)
    return result


def replay(args,key=None,all_episodes=False):
    cfg=locked_read(args.output/'experiment.json')['config']
    tasks=items(args,cfg['smoke'])
    tasks=[t for t in tasks if cached(args,t) is not None and (key is None or t['key']==key)]
    if not all_episodes and key is None:
        selected=[]
        for method in METHODS:
            t=next((t for t in tasks if t['method']==method),None)
            if t:selected.append(t)
        tasks=selected
    checks=[]
    for item in tasks:
        row=cached(args,item);record=o.load_torch(row['episode_record']['path'])
        p,state,own,other=setup(item,cfg['smoke'])
        for original in record['trajectory']:
            nxt,actual=one_step(p,state,own,other,fixed=original['own']['simulations'])
            for name in ('action','next_state','particles','belief_rng','controller_rng'):
                if not np.array_equal(actual[name],original[name]):raise AssertionError('Replay differs: '+item['key']+' '+name)
            state=nxt
        # Restore halfway including opponent RNG and observation history, then
        # rerun the suffix with the recorded counts as a second audit.
        checks.append(dict(key=item['key'],steps=len(record['trajectory']),passed=True))
    if not checks:raise ValueError('No committed episodes available to replay')
    result=dict(passed=True,fixed_recorded_simulations=True,episodes=checks)
    o.write_json(args.output/'replay.json',result)
    return result


def aggregate(rows):
    lat=[v for r in rows for v in r['latency_ms']]
    simulations=[v for r in rows for v in r['simulations']]
    hidden=sum(r['hidden_observations'] for r in rows)
    losses=sum(r['loss_events'] for r in rows)
    return dict(games=len(rows),capture_rate=float(np.mean([r['legal_capture'] for r in rows])),
        own_illegal_rate=float(np.mean([r['own_illegal'] for r in rows])),
        completion_cost=float(np.mean([r['completion_cost'] for r in rows])),
        mean_ms=float(np.mean(lat)),p95_ms=float(np.quantile(lat,.95)),overrun_rate=sum(r['overruns'] for r in rows)/max(1,len(lat)),
        mean_simulations=float(np.mean(simulations)),simulations=int(sum(simulations)),
        transitions=sum(r['transitions'] for r in rows),geometry_queries=sum(r['geometry_queries'] for r in rows),
        lost_steps=sum(r['lost_steps'] for r in rows),loss_events=losses,reacquisitions=sum(r['reacquisitions'] for r in rows),
        conditional_reacquisition_rate=sum(r['reacquisitions'] for r in rows)/losses if losses else None,
        hidden_coverage_1m=sum(r['coverage_1m'] for r in rows)/hidden if hidden else None,
        recoveries=sum(r['recovery_count'] for r in rows),failure_reasons=dict(Counter(r['failure_reason'] for r in rows)),
        selected_sources=dict(sum((Counter(r['selected_sources']) for r in rows),Counter())))


def statistics(rows,smoke=False):
    scenes=sorted({r['scene_id'] for r in rows})
    if len(rows)!=len(scenes)*5*(1 if smoke else 2):raise ValueError('Incomplete paired experiment')
    index={(r['method'],r['scene_id'],r['stream']):r for r in rows}
    if len(index)!=len(rows):raise ValueError('Duplicate result')
    rng=np.random.default_rng(60020261024)
    groups=[[i for i,s in enumerate(scenes) if next(r for r in rows if r['scene_id']==s)['map_id']==m] for m in MAPS]
    draws=np.concatenate([rng.choice(g,size=(10000,len(g)),replace=True) for g in groups],axis=1)
    comparisons={}
    for a,b in PAIRS:
        result={}
        for metric in ('legal_capture','own_illegal','completion_cost'):
            delta=np.array([np.mean([index[(a,s,j)][metric]-index[(b,s,j)][metric] for j in range(1 if smoke else 2)]) for s in scenes])
            result[metric]=dict(delta=float(delta.mean()),ci95=np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist(),
                gained=sum(index[(a,s,j)][metric]>index[(b,s,j)][metric] for s in scenes for j in range(1 if smoke else 2)),
                lost=sum(index[(a,s,j)][metric]<index[(b,s,j)][metric] for s in scenes for j in range(1 if smoke else 2)),
                scene_positive=int(np.sum(delta>0)),scene_negative=int(np.sum(delta<0)))
        comparisons[a+'-'+b]=result
    summary={m:aggregate([r for r in rows if r['method']==m]) for m in METHODS}
    by_map={map_id:{m:aggregate([r for r in rows if r['method']==m and r['map_id']==map_id]) for m in METHODS} for map_id in MAPS}
    by_ratio={str(ratio):{m:aggregate([r for r in rows if r['method']==m and r['ratio']==ratio]) for m in METHODS}
              for ratio in sorted({r['ratio'] for r in rows})}
    gates={}
    for ref in ('T','G'):
        d=comparisons['V-'+ref]
        gates[ref]=dict(capture_gain=d['legal_capture']['delta'],illegal_gain=d['own_illegal']['delta'],
            open_gain=by_map['open']['V']['capture_rate']-by_map['open'][ref]['capture_rate'])
        gates[ref]['passed']=bool(gates[ref]['capture_gain']>=.05-1e-12 and gates[ref]['illegal_gain']<=.02+1e-12 and gates[ref]['open_gain']>=-.05-1e-12)
    return dict(seed=60020261024,draws=10000,unit='physical_scene; average_streams_then_stratified_paired_bootstrap',
        summary=summary,by_map=by_map,by_ratio=by_ratio,comparisons=comparisons,gates=gates,
        passed=not smoke and all(v['passed'] for v in gates.values()),development_only=True,
        O='full_state_reference; not_an_upper_bound; excluded_from_gates')


def plots(args,rows,stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    dest=args.output/'figures';dest.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    for ax,key,title in zip(axes,('capture_rate','completion_cost','mean_ms'),('Legal capture rate','Completion cost (failure=80)','Mean decision latency (ms)')):
        ax.bar(METHODS,[stats['summary'][m][key] for m in METHODS]);ax.set_title(title)
    fig.tight_layout();fig.savefig(dest/'cost_capture.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for m in METHODS:
        latency=sorted(v for r in rows if r['method']==m for v in r['latency_ms'])
        axes[0].plot(latency,np.linspace(0,1,len(latency)),label=m)
        part=[r for r in rows if r['method']==m]
        axes[1].scatter([r['geometry_queries'] for r in part],[r['transitions'] for r in part],label=m,alpha=.5,s=10)
    axes[0].set(xlabel='Decision ms',ylabel='CDF');axes[0].legend()
    axes[1].set(xlabel='Geometry queries',ylabel='Simulated transitions');axes[1].legend()
    fig.tight_layout();fig.savefig(dest/'cost_distributions.png',dpi=160);plt.close(fig)
    examples=[]
    predicates=[('success',lambda r:r['legal_capture']),('failure',lambda r:not r['legal_capture']),('reacquisition',lambda r:r['reacquisitions']>0)]
    for label,test in predicates:
        candidates=[r for r in rows if r['method']=='V' and test(r)] or [r for r in rows if test(r)]
        for row in candidates[:1]:
            detail=o.load_torch(row['episode_record']['path']);trace=detail['trajectory']
            fig,ax=plt.subplots(figsize=(5,5))
            for box in MAPS[row['map_id']]:ax.add_patch(Rectangle((box[0][0],box[1][0]),box[0][1]-box[0][0],box[1][1]-box[1][0],color='gray'))
            for r,color,name in ((0,'tab:red','evader'),(2,'tab:blue','pursuer')):
                pts=np.array([t['state'][r*2:r*2+2,0] for t in trace]+[trace[-1]['next_state'][r*2:r*2+2,0]])
                ax.plot(*pts.T,color=color,label=name);ax.scatter(*pts[0],color=color,marker='x')
            hidden=[t for t in trace if t['observation'].target is None]
            if hidden:
                part=hidden[len(hidden)//2]['particles'];ax.scatter(*part.T,s=5,alpha=.4,c='green',label='hidden belief snapshot')
            ax.set(xlim=(-10,10),ylim=(-10,10),title=f"{label}: {row['method']} / {row['map_id']}",aspect='equal');ax.legend()
            fig.tight_layout();fig.savefig(dest/(label+'.png'),dpi=160);plt.close(fig)
            examples.append(dict(kind=label,key=row['key'],path=row['episode_record']['path']))
    o.write_json(args.output/'trajectory_examples.json',examples)


def report(args):
    cfg=locked_read(args.output/'experiment.json')['config']
    rows=locked_read(args.output/'results.json')
    stats=statistics(rows,cfg['smoke'])
    lock_json(args.output/'statistics.json',stats)
    plots(args,rows,stats)
    s=stats['summary']
    lines=['# 局部观测下一追一逃机制实验','',
        f"完成 {len(rows)} 局请求，{len({r['scene_id'] for r in rows})} 个物理场景；每场 {cfg['streams']} 个配对随机流。" ,
        '本轮不训练网络，不修改当前部署推荐。O 是全状态直接 G 参照，不是理论上界，不参与继续门槛。','',
        '|方法|合法捕获率|自身非法率|完成成本（失败=80）|均值/P95 ms|超时率|平均模拟数|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m in METHODS:
        v=s[m];lines.append(f"|{m}|{v['capture_rate']:.2%}|{v['own_illegal_rate']:.2%}|{v['completion_cost']:.2f}|{v['mean_ms']:.3f}/{v['p95_ms']:.3f}|{v['overrun_rate']:.2%}|{v['mean_simulations']:.1f}|")
    lines+=['','两流先在物理场景内平均，按三类地图分层做 10,000 次配对 bootstrap；固定统计种子 60020261024。',
            '区间描述这批开发场景的配对差异，不把两条随机流当成独立物理样本。','',
            '|比较|捕获率差（百分点）|95% 区间（百分点）|挽回/丢失（配对局）|','|---|---:|---|---:|']
    for name,v in stats['comparisons'].items():
        d=v['legal_capture'];lo,hi=d['ci95'];lines.append(f"|{name}|{100*d['delta']:+.2f}|[{100*lo:+.2f}, {100*hi:+.2f}]|{d['gained']}/{d['lost']}|")
    lines+=['','## 分地图结果','','|地图|L|G|T|V|O|','|---|---:|---:|---:|---:|---:|']
    for map_id,part in stats['by_map'].items():lines.append('|'+map_id+'|'+'|'.join(f"{part[m]['capture_rate']:.2%}" for m in METHODS)+'|')
    lines+=['','## 分速度比结果','','|速度比|L|G|T|V|O|','|---|---:|---:|---:|---:|---:|']
    for ratio,part in stats['by_ratio'].items():lines.append('|'+ratio+'|'+'|'.join(f"{part[m]['capture_rate']:.2%}" for m in METHODS)+'|')
    lines+=['','## 目标丢失与估计','','|方法|丢失步数|丢失事件|条件重发现率|隐藏时粒子 1m 覆盖率|恢复次数|',
            '|---|---:|---:|---:|---:|---:|']
    for m in METHODS:
        v=s[m];rate=v['conditional_reacquisition_rate'];cov=v['hidden_coverage_1m']
        lines.append(f"|{m}|{v['lost_steps']}|{v['loss_events']}|{rate:.2%}|{cov:.2%}|{v['recoveries']}|" if rate is not None and cov is not None else f"|{m}|{v['lost_steps']}|{v['loss_events']}|N/A|N/A|{v['recoveries']}|")
    lines+=['','覆盖率仅作评估：隐藏目标与最近粒子的距离不超过 1m。评估真值在动作确定后读取，不传回控制器。L 不维护多假设，其覆盖数不作滤波质量比较。',
            '条件重发现率 = 再次正观测事件 / 丢失事件；结束时仍丢失的事件保留在分母。','',
            '## 候选来源','','|来源|V 选用次数|','|---|---:|']
    lines += [f"|{k}|{v}|" for k,v in s['V']['selected_sources'].items()]
    lines+=['','## 开发继续判定','']
    for ref,v in stats['gates'].items():
        lines.append(f"V−{ref}：捕获率 {v['capture_gain']*100:+.2f} 点，自身非法率 {v['illegal_gain']*100:+.2f} 点，开阔捕获率 {v['open_gain']*100:+.2f} 点；{'通过' if v['passed'] else '未通过'}。")
    if stats['passed']:
        conclusion='V 同时通过两项开发门槛。冻结本方法；下一轮另行登记独立地图、不同对手和机器人动力学验证，本轮不自动扩展。'
    elif s['G']['hidden_coverage_1m'] is not None and s['G']['hidden_coverage_1m']<.5 and s['G']['capture_rate']<s['O']['capture_rate']-.05:
        conclusion='V 未通过门槛，隐藏时粒子覆盖不足且全状态参照更好，优先检查目标估计失真与遮挡后的支持集；不追加搜索预算。'
    elif stats['comparisons']['T-G']['legal_capture']['delta']>0:
        conclusion='V 未同时通过门槛。T 的点估计优于 G，规划值得继续研究；当前可见性/出口候选缺少达到门槛的额外收益证据。'
    else:
        conclusion='V 未同时通过门槛，搜索也未显示优于 G 的捕获率点估计；优先保留和检查简单粒子估计与几何控制。'
    if cfg['smoke']:conclusion='这是独立工程 smoke，不能用于研究收益判断。'
    lines += [conclusion,'','## 仍然失败的条件','']
    for map_id,part in stats['by_map'].items():lines.append(f"{map_id} / V：{part['V']['failure_reasons']}；隐藏时 1m 覆盖率 {part['V']['hidden_coverage_1m']}。")
    lines+=['','## 计算与实现边界','',
        'T/V 为根粒子采样 open-loop MCTS，节点按动作历史索引，没有按观测结果分支；不宣称完整信念空间规划。',
        '真实对手为原 Search Only：正式预算 3,000、深度 100、全状态、无策略网络和价值截断。模拟对手统一采用低成本全位置观测的逃离/均匀混合策略，不能等同于复刻 3,000 次搜索对手。',
        '50ms 包含估计、每决策一次的根候选生成、搜索和动作选择；仅在完整模拟边界停止，超时据实计入。预热和已知地图静态图构建另计，日志、评估真值和落盘在决策计时之外。',
        'V 出口定义冻结为追捕者到粒子最短绕障路径的首个转折节点；按合格粒子质量归并，并要求追捕者到达不晚于目标。',
        '几何查询数包含合法性/线段查询调用；环境转移数为追捕搜索中的模拟联合推进，另有逐步真实推进。完整分组、区间、非法率和成本见 statistics.json。',
        '计时运行单进程单线程，固定模拟数重放见 replay.json；逐步断点包含环境状态、观测历史、粒子及双方随机状态。','',
        '![捕获率与成本](figures/cost_capture.png)','![计算分布](figures/cost_distributions.png)']
    for file in ('success.png','failure.png','reacquisition.png'):
        if (args.output/'figures'/file).exists():lines+=['',f'![{file}](figures/{file})']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return stats


def check(args):
    import importlib.util
    spec=importlib.util.spec_from_file_location("local_pursuit_acceptance", CODE/"tests/test_l45_local_pursuit.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    run_checks=module.run_checks
    result=run_checks(args.output)
    o.write_json(args.output/'checks.json',result)
    return result


def smoke(args):
    target=argparse.Namespace(**vars(args));target.output=args.output/'smoke'
    prepare(target,True)
    evaluate(target,True)
    replay(target,all_episodes=True)
    report(target)
    result=dict(passed=True,requests=15,simulations=3,isolated_output=str(target.output),
                replay_sha256=sha(target.output/'replay.json'))
    o.write_json(args.output/'smoke_acceptance.json',result)
    return result


def watch(args):
    while True:
        path=args.output/'status.json'
        print(json.dumps(o.read_json(path) if path.exists() else dict(stage='not_started'),ensure_ascii=False),flush=True)
        if args.once or (path.exists() and o.read_json(path)['completed']==o.read_json(path)['total']):return
        time.sleep(5)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','check','smoke','profile','evaluate','report','replay','watch','pipeline'))
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--once',action='store_true',help='watch once; pipeline runs its finite frozen set once')
    parser.add_argument('--key')
    parser.add_argument('--all',action='store_true',help='replay all committed episodes')
    args=parser.parse_args(argv);args.output=args.output.resolve();return args


def main(argv=None):
    args=parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.command=='watch':return watch(args)
    with serial_lock(args):
        if args.command=='prepare':return prepare(args)
        if args.command=='check':warmup();return check(args)
        if args.command=='smoke':warmup();return smoke(args)
        if args.command=='profile':return profile(args)
        if args.command=='evaluate':warmup();return evaluate(args)
        if args.command=='report':return report(args)
        if args.command=='replay':warmup();return replay(args,args.key,args.all)
        if args.command=='pipeline':
            warmup();prepare(args);check(args);smoke(args);profile(args)
            evaluate(args);replay(args,all_episodes=True);report(args)
            o.write_json(args.output/'pipeline_complete.json',dict(passed=True,requests=600,training=False,adoption=False))


if __name__=='__main__':main()
