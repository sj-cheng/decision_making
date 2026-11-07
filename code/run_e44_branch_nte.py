"""E44: mechanism -> nine trained policies -> 1008 paired closed-loop runs.

Performance NEVER gates training or test. Correctness failures remain explicit.
Historical sources and accepted D are imported read-only. All progress commits
atomically and complete episodes/labels/checkpoints can be resumed.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import fcntl
import hashlib
import json
import math
import signal
import sys
import time
from pathlib import Path
import numpy as np
import torch
import run_search_value_validation as engine
from run_frozen_d_closed_loop import save, sha, legal_action
from corner_estimator_scenarios import construct
from verify_l45_occlusion_scenarios import verify_case
from audit_frozen_d_search_interface import observation, independent_traces
from solvers.frozen_d_search_interface import Controller as Geometry, loader, FROZEN
from solvers.e44_branch_nte import Controller
from solvers.nte_d_local import Node
from solvers.e44_routes import MAX_ROUTES, route_arrays
from learning.e44_policy import Network, Runtime, KINDS, weighted_nll, action_log_density

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'current/e44_branch_conditioned_nte_20261027'
BASE_SEED = 94420261027
SEEDS = (17,29,43)
MECHANISMS = ('uniform_wide','uniform_focused','route_wide','route_focused')
METHODS = ('G','HAND')+tuple(f'{k}{s}' for k in ('SINGLE','GMM','ROUTE','DIRECT') for s in SEEDS)
OPPONENTS = ('matched','rootpuct')
FEATURE_KEYS = ('maps','map_mask','particles','weights','history','history_mask','public','routes','route_mask','anchors')
MAX_LABELS = 128
WALL_SECONDS = 12*3600
SMOKE = False
ACTIVE = None
FORCED_TEACHER = None
SOURCE_NAMES = ('run_e44_branch_nte.py','solvers/e44_routes.py','solvers/e44_branch_nte.py',
    'learning/e44_policy.py','tests/test_e44_branch_nte.py','E44_PROTOCOL_20261027.md',
    'solvers/nte_d_local.py','learning/nte_d_local.py','solvers/search_value_belief.py',
    'solvers/frozen_d_search_interface.py','search_value_model.py','run_search_value_validation.py',
    'run_frozen_d_closed_loop.py','corner_estimator_scenarios.py','verify_l45_occlusion_scenarios.py',
    'audit_frozen_d_search_interface.py','estimator_audit_geometry.py','solvers/root_audit_puct.py',
    'solvers/rollout_geometry_exact.py','solvers/rollout_replan.py','solvers/c_puct.py','problems/example8.py',
    'prepare_l45_occlusion_scenarios.py','solvers/local_pursuit.py','analyze_search_value_validation.py')


def read(path):
    return json.loads(Path(path).read_text())


def emit(**values):
    record = dict(pid=os.getpid(),timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**values)
    save(OUT/'status.json',record)
    with (OUT/'events.jsonl').open('a') as f:
        f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)


def entries(split):
    return [e for e in read(OUT/'bank/manifest.json')['entries'] if e['split']==split]


def prepare():
    if (OUT/'experiment.json').exists():
        return verify()
    OUT.mkdir(parents=True,exist_ok=True)
    known_maps,known_histories = set(),set()
    for path in (ROOT/'current').glob('**/controller_inputs/*.json'):
        if path.is_relative_to(OUT):
            continue
        d = read(path)
        if 'map' in d and 'history' in d:
            known_maps.add(hashlib.sha256(json.dumps(d['map'],sort_keys=True).encode()).hexdigest())
            known_histories.add(hashlib.sha256(json.dumps(dict(map=d['map'],history=d['history']),sort_keys=True).encode()).hexdigest())
    rows,checks = [],[]
    counts = (('train',1 if SMOKE else 8),('validation',1 if SMOKE else 2),('test',1 if SMOKE else 6))
    for si,(split,repeats) in enumerate(counts):
        for fi,family in enumerate(('long_wall','wall_corner','multi_route')):
            for i in range(repeats):
                smoke_namespace = int(hashlib.sha256(OUT.name.encode()).hexdigest()[:8],16) if SMOKE else 0
                seed = int(np.random.SeedSequence([BASE_SEED,smoke_namespace,si,fi,i]).generate_state(1)[0])
                rng = np.random.default_rng(seed)
                H,speed = (4 if i%2==0 else 6),(1.1 if i%4<2 else 1.3)
                height = float(rng.uniform(6.2,6.9) if H==4 else rng.uniform(8.15,8.85))
                env,packet = construct(family,H,speed,seed,height,i%2,pause=i%2)
                check = verify_case(env,packet)
                assert check['passed'],check
                mh = hashlib.sha256(json.dumps(packet['map'],sort_keys=True).encode()).hexdigest()
                hh = hashlib.sha256(json.dumps(dict(map=packet['map'],history=packet['history']),sort_keys=True).encode()).hexdigest()
                assert mh not in known_maps and hh not in known_histories
                known_maps.add(mh);known_histories.add(hh)
                folder = OUT/'bank'/split
                for kind,data in (('controller_inputs',packet),('environment',env)):
                    save(folder/kind/(env['id']+'.json'),data)
                rows.append(dict(id=env['id'],family=family,split=split,
                    packet_path=str(folder/'controller_inputs'/(env['id']+'.json')),
                    environment_path=str(folder/'environment'/(env['id']+'.json')),
                    map_sha256=mh,geometry_history_sha256=hh))
                checks.append(check)
    save(OUT/'bank/manifest.json',dict(entries=rows,counts={s:sum(e['split']==s for e in rows) for s,_ in counts},
        split_unit='entire map and public history, retain all worlds and training seeds',test_outcomes_read=False))
    save(OUT/'bank_acceptance.json',dict(passed=True,geometry_only=True,records=checks))
    protected = {}
    for p in (FROZEN).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.nbc','.nbi'):
            protected[str(p)] = sha(p)
    for folder in ('l45_nte_d_local_baseline_20261027','l45_candidate_prior_30e_20261026','l45_candidate_prior_train_20261026'):
        for name in ('REPORT.md','experiment.json','training.json','models_frozen.json','decision.json','teacher_gate.json'):
            p = ROOT/'current'/folder/name
            if p.exists():
                protected[str(p)] = sha(p)
    for name in ('param.py','cpp/bindings.cpp','cpp/problems/example8.hpp'):
        protected[str(ROOT/'code'/name)] = sha(ROOT/'code'/name)
    protected[str(Path(engine.cpp.__file__).resolve())] = sha(engine.cpp.__file__)
    architecture = {k:dict(width=Network(k).width,parameters=sum(p.numel() for p in Network(k).parameters())) for k in KINDS}
    config = dict(experiment='E44',version='branch_conditioned_NTE_v1',smoke=SMOKE,
        registered_date='2026-10-27 Asia/Shanghai',wall_budget_seconds=WALL_SECONDS,
        train_maps=3 if SMOKE else 24,validation_maps=3 if SMOKE else 6,test_histories=3 if SMOKE else 18,
        train_states_per_map=2 if SMOKE else 24,validation_states_per_map=2 if SMOKE else 16,
        train_seeds=list(SEEDS),epochs=2 if SMOKE else 30,architecture=architecture,
        mechanisms=list(MECHANISMS),methods=list(METHODS),opponents=list(OPPONENTS),
        evaluation_games=84 if SMOKE else 1008,online_budget_ms=200.,teacher_budget_ms=2000.,
        labels_probe_budget_ms=16000.,maximum_paired_probe_draws=96,
        planning_physical_steps=8,discount=.95,reward='first rediscovery + capture',
        teacher_performance_gate=False,complete_all_networks_and_test_despite_negative_diagnostics=True,
        correctness_failure='explicit failure, preserve data; repair before continuing valid scientific stages',
        proposal_allocation=dict(wide='ceil(2 sqrt(N+1))',focused='min(12, ceil((N+1)^.25)); all tree nodes',
            route='region mass >= .01; nearest-centroid lawful sample; first waypoint, direct 8 sectors; stop; at most 12 modes',
            mixture='50% learned or handcrafted proposal and 50% uniform; shared legality rejection cap 64'),
        data='train/validation only G80% plus legal uniform20%; complete public histories; all visited teacher root actions retained',
        label_weights='visit-normalized empirical distribution; independent MC mean/SE saved; no low-visit deletion or performance threshold',
        NLL='common pre-legality tanh Gaussian mixture density in normalized Cartesian action; endpoints clip at common 1e-4 resolution',
        geometry_inputs='identical weighted sets, map, history and route descriptors for all networks',
        visible_stage='shared legal geometric pursuit; conclusions focus hidden control',
        new_value_network=False,new_opponent_model=False,D_changed=False,deployment_changed=False,
        matched_scope='fixed-root generated target kernel matches only before first rediscovery; not exact full posterior or post-reset match',
        primary_endpoint=dict(matched='rediscovery_cost',rootpuct='cost'),
        primary_comparisons=['ROUTE-SINGLE','ROUTE-GMM','ROUTE-HAND','ROUTE-DIRECT','ROUTE-G'],
        statistics='10000 public-history grouped paired bootstrap; 10 simultaneous primary intervals (5 contrasts x 2 target conditions); retain worlds/seeds',
        minimum_useful_cost_improvement=1.,reliable_gain='upper simultaneous CI <0; mean<=-1; all seeds improve; observed success does not decrease (not noninferiority proof); both compared arms have zero registered 200ms overruns',
        serial_timing=True,threads=1,cpu_affinity=sorted(os.sched_getaffinity(0)),
        source_hashes={str(ROOT/'code'/p):sha(ROOT/'code'/p) for p in SOURCE_NAMES},
        protected_hashes=protected,bank_hashes={str(p):sha(p) for p in (OUT/'bank').rglob('*.json')},
        exact_original_NTE_reproduction=False)
    save(OUT/'experiment.json',config)
    for name in SOURCE_NAMES:
        p = OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((ROOT/'code'/name).read_bytes())
    (OUT/'PROTOCOL.md').write_bytes((ROOT/'code/E44_PROTOCOL_20261027.md').read_bytes())
    write_live_report('协议已冻结；接口检查待执行。')
    emit(stage='registered',formal_training_started=False,evaluation_games=config['evaluation_games'])
    return config


def verify():
    loader.verify()
    config = read(OUT/'experiment.json')
    for category in ('source_hashes','protected_hashes','bank_hashes'):
        for p,h in config[category].items():
            if sha(p)!=h:
                raise RuntimeError(f'Frozen artifact changed: {p}')
    if (OUT/'models_frozen.json').exists():
        for p,h in read(OUT/'models_frozen.json')['files'].items():
            if sha(p)!=h:
                raise RuntimeError(f'Frozen model changed: {p}')
    return config


def teacher_choice():
    if FORCED_TEACHER is not None:
        return FORCED_TEACHER
    return read(OUT/'step1/selection.json')['selected']


def model_path(kind,seed,random=False):
    return OUT/('interface_models' if random else 'models')/f'{kind}_{seed}.npz'


def initialize(packet,method,seed):
    global ACTIVE
    started = time.perf_counter()
    boxes,speed = packet['map'],packet['motion']['pursuer_axis_limit']
    if method=='G':
        c = Geometry(boxes,speed,'G',seed,budget_ms=200.)
    elif method in MECHANISMS:
        proposal,allocation = method.split('_')
        c = Controller(boxes,speed,seed,proposal,allocation)
    elif method=='HAND':
        c = Controller(boxes,speed,seed,'route',teacher_choice().split('_')[1])
    else:
        prefix = next(k for k in ('SINGLE','GMM','ROUTE','DIRECT') if method.startswith(k))
        ts = int(method[len(prefix):]);kind = dict(SINGLE='single',GMM='gmm',ROUTE='route',DIRECT='route')[prefix]
        path = model_path(kind,ts,random=not (OUT/'models_frozen.json').exists())
        c = Controller(boxes,speed,seed,'neural',teacher_choice().split('_')[1],path,direct=prefix=='DIRECT')
    static = 1000*(time.perf_counter()-started);started = time.perf_counter()
    for item in packet['history'][:-1]:
        obs = observation(item);c.estimator.update(obs)
        if hasattr(c,'observe_history'):
            c.observe_history(obs)
    history = 1000*(time.perf_counter()-started)
    ACTIVE = c
    return c,dict(static_setup_ms=static,history_initialization_ms=history,total_ms=static+history,budget_ms=200.)


def request(entry,world,method,stage,opponent='rootpuct'):
    seeds = np.random.SeedSequence([BASE_SEED,int(SMOKE),int(entry['geometry_history_sha256'][:8],16),int(world=='B')]).generate_state(4)
    return dict(key=f'{entry["id"]}_{world}_{opponent}_{method}',group=entry['id'],family=entry['family'],
        stage=stage,world=world,method=method,opponent=opponent,packet_path=entry['packet_path'],
        environment_path=entry['environment_path'],seed=int(seeds[0]),opponent_seed=int(seeds[1]),
        environment_seed=int(seeds[2]),target_seed=int(seeds[3]),draw_index=int(world=='B'))


def run_episode(req,limit=None,fixed=None,initializer=None):
    old = engine.initialize;engine.initialize = initializer or initialize
    try:
        return engine.episode(req,limit=limit,fixed=fixed)
    finally:
        engine.initialize = old


def commit_episode(req,payload):
    path = OUT/req['stage']/'episodes'/(req['key']+'.json');save(path,payload)
    row = payload['summary']|dict(episode=str(path),episode_sha256=sha(path),
        rediscovered=payload['summary']['first_seen'] is not None)
    save(OUT/req['stage']/'cache'/(req['key']+'.json'),row)
    if row['failure'] is not None:
        raise RuntimeError(f'Engineering failure preserved at {path}: {row["failure"]}')
    return row


def cached_episode(req):
    path = OUT/req['stage']/'cache'/(req['key']+'.json')
    if path.exists():
        r = read(path)
        assert sha(r['episode'])==r['episode_sha256']
        if r['failure'] is not None:
            raise RuntimeError('Prior engineering failure must be investigated, not silently resampled')
        return r
    return None


def preflight():
    global FORCED_TEACHER
    if (OUT/'interface.json').exists():
        return read(OUT/'interface.json')
    FORCED_TEACHER = 'route_focused'
    torch.set_num_threads(1)
    folder = OUT/'interface_models';folder.mkdir(exist_ok=True)
    for seed in SEEDS:
        for kind in KINDS:
            torch.manual_seed(seed);Network(kind).export(model_path(kind,seed,True))
    stamp = time.perf_counter()
    first = read(entries('validation')[0]['packet_path'])
    for method in MECHANISMS+('SINGLE17','GMM17','ROUTE17','DIRECT17'):
        c,_ = initialize(first,method,317)
        c.act(observation(first['history'][-1]),fixed_simulations=2)
    save(OUT/'warmup.json',dict(seconds=time.perf_counter()-stamp,compile_excluded_from_decisions=True))
    records = []
    for family in ('long_wall','wall_corner','multi_route'):
        entry = next(e for e in entries('validation') if e['family']==family)
        packet = read(entry['packet_path']);obs = observation(packet['history'][-1])
        for method in MECHANISMS+('SINGLE17','GMM17','ROUTE17','DIRECT17'):
            copies = []
            for _ in range(2):
                c,_ = initialize(packet,method,317);a = c.act(obs,fixed_simulations=8)
                traces = np.concatenate(c.traces) if c.traces else np.zeros((0,10))
                assert not any(independent_traces(traces,c.public.boxes,c.public.speed))
                assert legal_action(obs.position,a,c.public.speed,c.public.boxes)
                copies.append((a.tolist(),c.public_signature(),c.last_stats['root_actions'],c.last_stats['root_visits']))
            assert copies[0]==copies[1]
            records.append(dict(family=family,method=method,replay_exact=True,public_only=True,legal=True))
        for opponent in OPPONENTS:
            for method in ('G','route_focused','SINGLE17','GMM17','ROUTE17','DIRECT17'):
                req = request(entry,'A',method,'engineering',opponent)
                payload = run_episode(req,limit=3,fixed=[4]*3)
                commit_episode(req,payload)
        emit(stage='interface',family=family,records=len(records))
    result = dict(passed=True,records=records,engineering_episodes=36,
        test_used=False,random_models_only=True,no_performance_gate=True)
    save(OUT/'interface.json',result);FORCED_TEACHER = None
    return result


def teacher_query(packet,method,seed):
    c,setup = initialize(packet,method,seed)
    c.budget_ms = max(0.,2000.-setup['total_ms'])
    start = time.perf_counter();a = c.act(observation(packet['history'][-1]),fixed_simulations=16 if SMOKE else None)
    decision_ms=1000*(time.perf_counter()-start)
    audit_started=time.perf_counter()
    traces = np.concatenate(c.traces) if c.traces else np.zeros((0,10))
    assert not any(independent_traces(traces,c.public.boxes,c.public.speed))
    return c,dict(action=a.tolist(),setup=setup,total_ms=setup['total_ms']+decision_ms,
        offline_audit_ms=1000*(time.perf_counter()-audit_started),
        statistics=c.last_stats,trace_sha256=hashlib.sha256(traces.tobytes()).hexdigest(),simulation_errors=[0]*4)


def independent_values(c,actions,seed,draws=96,budget_ms=16000.):
    # Probe uses forced first action then the SAME observation-conditioned public
    # geometric rollout used at search leaves; the full tree is evaluated apart.
    c.rng[:] = np.array([int(seed)%4294967296],np.int64)
    values,traces,ms = c.independent_rollouts(actions,draws=4 if SMOKE else draws,budget_ms=budget_ms)
    trace = np.concatenate(traces) if traces else np.zeros((0,10))
    errors = independent_traces(trace,c.public.boxes,c.public.speed)
    assert not any(errors)
    if not len(values):
        raise RuntimeError('No complete independent model draw: execution/data failure, not performance gate')
    return values,dict(draws=len(values),milliseconds=ms,simulation_errors=errors.tolist(),
        trace_sha256=hashlib.sha256(trace.tobytes()).hexdigest(),simulated_transitions=len(trace))


def step1():
    if (OUT/'step1/selection.json').exists():
        return read(OUT/'step1/selection.json')
    development = [e for family in ('long_wall','wall_corner','multi_route')
        for e in [x for x in entries('train') if x['family']==family][:1 if SMOKE else 4]]
    diagnostics = []
    for i,entry in enumerate(development):
        path = OUT/'step1/roots'/(entry['id']+'.json')
        if path.exists():
            diagnostics.append(read(path));continue
        packet = read(entry['packet_path']);queries={};controllers={};pool=[]
        def add(a):
            for j,b in enumerate(pool):
                if np.max(np.abs(np.asarray(a)-b))<1e-10:
                    return j
            pool.append(np.asarray(a).copy());return len(pool)-1
        indexes = {}
        for method in MECHANISMS:
            c,q = teacher_query(packet,method,BASE_SEED+i)
            assert c.root is not None and sum(c.root.visits)>0
            controllers[method],queries[method]=c,q
            indexes[method]=add(q['action'])
        g,_=initialize(packet,'G',BASE_SEED+i);ga=g.act(observation(packet['history'][-1]));indexes['G']=add(ga)
        # Fixed finite reference pool. First add each method's winner, then visit
        # order round-robin; selection never consults probe outcomes.
        for rank in range(64):
            for method in MECHANISMS:
                c = controllers[method];order=np.argsort(-np.asarray(c.root.visits),kind='stable')
                if rank<len(order) and len(pool)<24:
                    add(c.root.actions[int(order[rank])])
        c=controllers['uniform_focused']
        values,probe = independent_values(c,pool,BASE_SEED+i+411)
        split = max(1,len(values)//3)
        screen = values[:split];heldout=values[split:] if len(values)>split else values
        reference=int(np.argmax(screen.mean(0)))
        reference_means=screen.mean(0);reference_se=screen.std(0,ddof=1)/np.sqrt(len(screen)) if len(screen)>1 else np.zeros(len(pool))
        near = np.flatnonzero(reference_means >= reference_means.max()-.02-1.96*reference_se)
        result = dict(group=entry['id'],family=entry['family'],queries=queries,actions=[a.tolist() for a in pool],
            action_indexes=indexes,probe=probe,probe_returns=values.tolist(),reference_index=reference,
            screening_draws=split,heldout_draws=len(heldout),near_optimal_indexes=near.tolist(),
            comparisons={m:dict(empirical_regret=float((heldout[:,reference]-heldout[:,j]).mean()),
                paired_se=float((heldout[:,reference]-heldout[:,j]).std(ddof=1)/np.sqrt(len(heldout))) if len(heldout)>1 else 0.,
                near_optimal_hit=bool(j in near)) for m,j in indexes.items()},
            scope='finite-reference-pool, model diagnostic; not global optimal regret',performance_gate=False)
        save(path,result);diagnostics.append(result)
        emit(stage='step1_roots',completed=i+1,total=len(development))
    # Paired full closed-loop development in BOTH target conditions. No test map
    # enters teacher selection; failures of efficacy do not stop the pipeline.
    rows=[]
    for entry in development:
        for opponent in OPPONENTS:
            for world in ('A',) if SMOKE else ('A','B'):
                for method in ('G',)+MECHANISMS:
                    req=request(entry,world,method,'step1/closed_loop',opponent)
                    row=cached_episode(req)
                    if row is None:
                        payload=run_episode(req,limit=3 if SMOKE else None,fixed=[8]*3 if SMOKE else None)
                        row=commit_episode(req,payload)
                    rows.append(row)
                    emit(stage='step1_closed_loop',completed=len(rows),total=len(development)*len(OPPONENTS)*(1 if SMOKE else 2)*5)
    scores={m:float(np.mean([r['comparisons'][m]['empirical_regret'] for r in diagnostics])) for m in MECHANISMS}
    # Quantized 1e-6 tie, then deterministic method order. No significance gate.
    chosen=min(MECHANISMS,key=lambda m:(round(scores[m],6),MECHANISMS.index(m)))
    selected=dict(selected=chosen,criterion='minimum mean held-out finite-pool model regret; rounded 1e-6, fixed order tie',
        scores=scores,diagnostic_groups=len(diagnostics),closed_loop_games=len(rows),
        continue_to_training=True,performance_gate=False,test_outcomes_read=False)
    save(OUT/'step1/closed_loop/results.json',rows)
    save(OUT/'step1/selection.json',selected)
    emit(stage='teacher_selected',**selected)
    write_live_report('Step 1 已完成；无论性能正负，继续统一标签与三网络训练。')
    return selected


class Collector(Geometry):
    def __init__(self,boxes,speed,seed):
        super().__init__(boxes,speed,'G',seed,budget_ms=200.)
        self.records,self.packets=[],[]
        self.behavior_rng=np.random.default_rng(seed+812)

    def observe_history(self,obs):
        self.records.append(dict(step=obs.step,position=obs.position.tolist(),
            target=None if obs.target is None else obs.target.tolist(),remaining=obs.remaining))

    def act(self,obs,fixed_simulations=None):
        self.observe_history(obs)
        action=super().act(obs,fixed_simulations=0)
        self.packets.append(dict(map=self.public.boxes.tolist(),motion=dict(pursuer_axis_limit=self.public.speed),history=list(self.records)))
        if self.behavior_rng.random()<.2:
            for _ in range(64):
                a=self.behavior_rng.uniform(-self.public.speed,self.public.speed,2)
                if legal_action(obs.position,a,self.public.speed,self.motion_boxes):
                    return a
            return np.zeros(2)
        return action


def collect_packets(entry):
    path=OUT/'collection'/entry['split']/(entry['id']+'.json')
    if path.exists():
        return read(path)['packets']
    packet=read(entry['packet_path']);captured={}
    def init(packet,method,seed):
        start=time.perf_counter();c=Collector(packet['map'],packet['motion']['pursuer_axis_limit'],seed)
        for item in packet['history'][:-1]:
            obs=observation(item);c.estimator.update(obs);c.observe_history(obs)
        captured['own']=c;ms=1000*(time.perf_counter()-start)
        return c,dict(static_setup_ms=ms,history_initialization_ms=0.,total_ms=ms,budget_ms=200.)
    req=request(entry,'A','G','collection','rootpuct')
    payload=run_episode(req,limit=4 if SMOKE else None,initializer=init)
    assert payload['summary']['failure'] is None,payload['summary']['failure']
    packets=[dict(map=packet['map'],motion=packet['motion'],history=packet['history'][:j+1])
        for j in range(len(packet['history'])-1)]+captured['own'].packets
    save(path,dict(packets=packets,episode=payload,test_used=False))
    return packets


def label_data():
    if (OUT/'data/manifest.json').exists():
        manifest=read(OUT/'data/manifest.json')
        for p,h in manifest['files'].items():assert sha(p)==h
        return manifest
    config=read(OUT/'experiment.json');numbers={}
    for split in ('train','validation'):
        count=0
        for i,entry in enumerate(entries(split)):
            packets=collect_packets(entry)
            cap=config['train_states_per_map' if split=='train' else 'validation_states_per_map']
            # Even spacing on complete public trajectories; no teacher outcome
            # used to remove or reselect states. Visible states remain tagged.
            indexes=np.unique(np.linspace(0,len(packets)-1,min(cap,len(packets))).astype(int))
            for j in indexes:
                folder=OUT/'data'/split/entry['id'];folder.mkdir(parents=True,exist_ok=True)
                path=folder/f'{j:03d}.npz';meta=path.with_suffix('.json')
                if path.exists() and meta.exists():
                    assert sha(path)==read(meta)['npz_sha256'];count+=1;continue
                packet=packets[int(j)]
                c,query=teacher_query(packet,teacher_choice(),BASE_SEED+i*1009+int(j))
                obs=observation(packet['history'][-1]);c.deadline=float('inf')
                if c.root is None:
                    # Visible policy is a shared geometry action; no fabricated search.
                    node=Node(obs.position,np.ones((1,9))/9,0,0,list(c.public_records),True)
                    x=c.features(node);actions=np.asarray([query['action']])/c.public.speed
                    weights=np.ones(1);quality=dict(source='shared visible geometry',draws=0)
                    near_mask=np.ones(1)
                else:
                    x=c.features(c.root)
                    actions=np.asarray(c.root.actions)/c.public.speed
                    weights=np.asarray(c.root.visits,dtype=float)
                    keep=weights>0
                    actions,weights=actions[keep],weights[keep]
                    if not len(weights):raise RuntimeError('No completed teacher labels')
                    weights/=weights.sum()
                    values,probe=independent_values(c,actions*c.public.speed,BASE_SEED+i*1009+int(j)+199991)
                    quality=dict(source='public-model independent forced-action feedback rollout',**probe,
                        returns=values.tolist(),mean=values.mean(0).tolist(),
                        standard_error=(values.std(0,ddof=1)/np.sqrt(len(values))).tolist() if len(values)>1 else [0.]*len(actions),
                        all_low_signal_retained=True)
                    qmean=values.mean(0)
                    qse=values.std(0,ddof=1)/np.sqrt(len(values)) if len(values)>1 else np.zeros(len(actions))
                    near_mask=(qmean>=qmean.max()-.02-1.96*qse).astype(float)
                if len(actions)>MAX_LABELS:raise RuntimeError('Teacher label capacity exceeded; do not truncate silently')
                la,lw,ln=np.zeros((MAX_LABELS,2)),np.zeros(MAX_LABELS),np.zeros(MAX_LABELS)
                la[:len(actions)],lw[:len(actions)]=actions,weights
                ln[:len(actions)]=near_mask
                temp=path.with_suffix('.tmp.npz')
                np.savez_compressed(temp,**x,label_actions=la,label_weights=lw,label_near_mask=ln)
                temp.replace(path)
                save(meta,dict(group=entry['id'],family=entry['family'],public_packet=packet,teacher=query,
                    quality=quality,label_count=len(actions),npz_sha256=sha(path),performance_filter=False))
                count+=1
                emit(stage='labels_'+split,maps_completed=i,maps_total=len(entries(split)),states_completed=count,
                    current_group=entry['id'],visible=obs.target is not None)
        numbers[split]=count
    manifest=dict(counts=numbers,files={str(p):sha(p) for p in (OUT/'data').rglob('*') if p.is_file()},
        no_test_data=True,shared_labels_all_networks=True,no_cross_route_mean_targets=True)
    save(OUT/'data/manifest.json',manifest)
    return manifest


def dataset(split):
    files=sorted((OUT/'data'/split).rglob('*.npz'))
    items=[dict(np.load(p,allow_pickle=False)) for p in files]
    return {k:torch.from_numpy(np.stack([d[k] for d in items])).float() for k in items[0]}


def validation_hits(model,data):
    rng=np.random.default_rng(991)
    with torch.no_grad():
        logits,loc,ls=model(data)
        probs=torch.softmax(logits,-1).numpy();loc=loc.numpy();ls=ls.numpy()
    hits=[]
    for i in range(len(probs)):
        n=128;k=rng.choice(len(probs[i]),n,p=probs[i]);a=np.tanh(loc[i,k]+np.exp(ls[i,k])*rng.standard_normal((n,2)))
        valid=data['label_weights'][i].numpy()>0
        targets=data['label_actions'][i].numpy()[valid]
        weights=data['label_weights'][i].numpy()[valid]*data['label_near_mask'][i].numpy()[valid]
        weights/=weights.sum()
        within=np.any(np.max(np.abs(a[:,None,:]-targets[None,:,:]),axis=2)<=.15,axis=0)
        hits.append(float(weights@within))
    return float(np.mean(hits))


def train():
    if (OUT/'models_frozen.json').exists():
        verify();return read(OUT/'training.json')
    data,val=dataset('train'),dataset('validation')
    config=read(OUT/'experiment.json');epochs=config['epochs']
    folder=OUT/'models';folder.mkdir(exist_ok=True)
    records=[];torch.set_num_threads(1)
    save(OUT/'training.json',dict(started=True,completed=False,expected_models=9,records=[]))
    for seed in SEEDS:
        for kind in KINDS:
            target=OUT/'training'/f'{kind}_{seed}.json'
            if target.exists():
                record=read(target);assert sha(model_path(kind,seed))==record['model_sha256'];records.append(record);continue
            torch.manual_seed(seed);model=Network(kind)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
            rng=np.random.default_rng(seed);best=float('inf');selected=None;logs=[]
            state_path=OUT/'training'/f'{kind}_{seed}_resume.pt';state_path.parent.mkdir(exist_ok=True)
            begin=1
            if state_path.exists():
                s=torch.load(state_path,weights_only=False,map_location='cpu')
                model.load_state_dict(s['model']);optimizer.load_state_dict(s['optimizer']);rng.bit_generator.state=s['rng']
                best,selected,logs,begin=s['best'],s['selected'],s['logs'],s['epoch']+1
            for epoch in range(begin,epochs+1):
                model.train();total=0.;n=0
                for batch in np.array_split(rng.permutation(len(data['public'])),max(1,math.ceil(len(data['public'])/64))):
                    batch_data={k:v[batch] for k,v in data.items()}
                    optimizer.zero_grad();loss=weighted_nll(model,batch_data)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
                    total+=float(loss.detach())*len(batch);n+=len(batch)
                model.eval()
                with torch.no_grad():v=float(weighted_nll(model,val))
                if not np.isfinite(v):raise RuntimeError('Nonfinite validation loss')
                logs.append(dict(epoch=epoch,train_NLL=total/n,validation_NLL=v))
                if v<best:
                    best,selected=v,epoch;model.export(model_path(kind,seed))
                    torch.save(model.state_dict(),folder/f'{kind}_{seed}.pt')
                torch.save(dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    rng=rng.bit_generator.state,best=best,selected=selected,logs=logs),state_path.with_suffix('.tmp'))
                state_path.with_suffix('.tmp').replace(state_path)
                emit(stage='training',kind=kind,seed=seed,epoch=epoch,epochs=epochs,validation_NLL=v)
            model.load_state_dict(torch.load(folder/f'{kind}_{seed}.pt',weights_only=True))
            hit=validation_hits(model,val)
            record=dict(kind=kind,seed=seed,epochs=epochs,selected_epoch=selected,best_validation_NLL=best,
                parameters=sum(p.numel() for p in model.parameters()),
                validation_visit_weighted_near_optimal_hit_at_128=hit,
                hit_definition='proposal 128 samples within normalized L-infinity .15; common empirical near set mean>=max(mean)-.02-1.96SE; not global optimality',
                log=logs,model_sha256=sha(model_path(kind,seed)))
            save(target,record);records.append(record)
            save(OUT/'training.json',dict(started=True,completed=len(records)==9,records=records,
                train_states=len(data['public']),validation_states=len(val['public']),test_outcomes_read=False))
    save(OUT/'models_frozen.json',dict(files={str(p):sha(p) for p in folder.iterdir() if p.is_file()},
        selection='minimum validation Cartesian action NLL within fixed epochs; all nine models',test_outcomes_read=False))
    write_live_report('9 个策略网络训练完成并冻结；进入训练后验收及独立闭环。')
    return read(OUT/'training.json')


def trained_preflight():
    if (OUT/'trained_interface.json').exists():return read(OUT/'trained_interface.json')
    records=[]
    for family in ('long_wall','wall_corner','multi_route'):
        entry=next(e for e in entries('validation') if e['family']==family)
        packet=read(entry['packet_path']);obs=observation(packet['history'][-1])
        for method in METHODS:
            c,setup=initialize(packet,method,317)
            a=c.act(obs,fixed_simulations=8)
            assert legal_action(obs.position,a,c.public.speed,c.public.boxes)
            traces=np.concatenate(c.traces) if c.traces else np.zeros((0,10))
            assert not any(independent_traces(traces,c.public.boxes,c.public.speed))
            if getattr(c,'root',None) is not None:
                c.deadline=float('inf');x=c.features(c.root)
                if c.network is not None:
                    runtime=c.network;model=Network(runtime.kind,runtime.width).double()
                    model.load_state_dict({k:torch.from_numpy(v) for k,v in runtime.arrays.items()})
                    with torch.no_grad():lg,mu,ls=model({k:torch.from_numpy(x[k])[None] for k in FEATURE_KEYS})
                    expected=(torch.softmax(lg,-1).numpy()[0],mu.numpy()[0],ls.numpy()[0])
                    for aa,bb in zip(expected,runtime(x)):np.testing.assert_allclose(aa,bb,atol=1e-11,rtol=0)
            records.append(dict(family=family,method=method,legal=True,matched_numpy=True))
    result=dict(passed=True,records=records,test_outcomes_read=False);save(OUT/'trained_interface.json',result)
    return result


def audit_payload(payload):
    task=payload['task'];packet=read(task['packet_path']);e=loader.load_estimator(packet['map'],task['seed'])
    for item in packet['history'][:-1]:e.update(observation(item))
    for r in payload['trace']:
        obs=observation(r['observation']);e.update(obs)
        np.testing.assert_array_equal(e.particles,np.asarray(r['particles']))
        np.testing.assert_array_equal(e.particle_weights,np.asarray(r['weights']))
        assert engine.signature(e)==r['estimator_sha256']
        assert legal_action(obs.position,np.asarray(r['own_action']),packet['motion']['pursuer_axis_limit'],np.asarray(packet['map']))
        assert not any(r['simulation_errors'])
    return dict(frames=len(payload['trace']),D_reconstructed=True,simulation_online_audit=True)


def evaluate():
    rows=[];requests=[]
    for i,entry in enumerate(entries('test')):
        for opponent in OPPONENTS:
            for wi,world in enumerate(('A',) if SMOKE else ('A','B')):
                offset=(i+wi)%len(METHODS)
                for method in METHODS[offset:]+METHODS[:offset]:requests.append(request(entry,world,method,'evaluation',opponent))
    expected=read(OUT/'experiment.json')['evaluation_games']
    assert len(requests)==expected,(len(requests),expected)
    save(OUT/'evaluation/requests.json',requests)
    for i,req in enumerate(requests):
        row=cached_episode(req)
        if row is None:
            payload=run_episode(req,limit=3 if SMOKE else None,fixed=[8]*3 if SMOKE else None)
            audit=audit_payload(payload)
            row=commit_episode(req,payload)
            save(OUT/'evaluation/audits'/(req['key']+'.json'),audit)
        rows.append(row)
        if i%14==0:verify()
        emit(stage='evaluation',completed=i+1,total=expected,method=req['method'],opponent=req['opponent'],
            captures=sum(int(r['capture']) for r in rows),latest_cost=row['cost'])
    save(OUT/'evaluation/results.json',rows)
    return summarize(rows)


def summarize(rows):
    groups=sorted({r['group'] for r in rows});comparators=('SINGLE','GMM','HAND','DIRECT','G')
    def names(arm):return [arm] if arm in ('G','HAND') else [f'{arm}{s}' for s in SEEDS]
    def mean(group,opponent,arm,field):
        return float(np.mean([r[field] for r in rows if r['group']==group and r['opponent']==opponent and r['method'] in names(arm)]))
    comparisons=[(opp,b,'rediscovery_cost' if opp=='matched' else 'cost') for opp in OPPONENTS for b in comparators]
    diff=np.array([[mean(g,opp,'ROUTE',field)-mean(g,opp,b,field) for opp,b,field in comparisons] for g in groups])
    rng=np.random.default_rng(BASE_SEED+881)
    boot=diff[rng.integers(len(groups),size=(10000,len(groups)))].mean(1)
    center=diff.mean(0);radius=float(np.quantile(np.max(np.abs(boot-center),axis=1),.95))
    results=[]
    for j,(opp,b,field) in enumerate(comparisons):
        per_seed=[]
        for s in SEEDS:
            differences=[]
            for g in groups:
                a=float(np.mean([r[field] for r in rows if r['group']==g and r['opponent']==opp and r['method']==f'ROUTE{s}']))
                bn=b if b in ('G','HAND') else f'{b}{s}'
                bv=float(np.mean([r[field] for r in rows if r['group']==g and r['opponent']==opp and r['method']==bn]))
                differences.append(a-bv)
            per_seed.append(float(np.mean(differences)))
        success='capture' if opp=='rootpuct' else 'rediscovered'
        success_diff=float(np.mean([mean(g,opp,'ROUTE',success)-mean(g,opp,b,success) for g in groups]))
        interval=[float(center[j]-radius),float(center[j]+radius)]
        relevant=[r for r in rows if r['opponent']==opp and r['method'] in names('ROUTE')+names(b)]
        overruns=sum(sum(t>r['budget_ms'] for t in registered_latencies(r)) for r in relevant)
        task_signal=bool(interval[1]<0 and center[j]<=-1. and all(x<0 for x in per_seed) and success_diff>=0)
        reliable=task_signal and overruns==0
        results.append(dict(opponent=opp,comparison='ROUTE-'+b,endpoint=field,difference=float(center[j]),
            simultaneous95=interval,ordinary95=np.quantile(boot[:,j],[.025,.975]).tolist(),
            per_seed=per_seed,observed_success_difference=success_diff,reliable_gain=reliable,
            task_signal_before_timing=task_signal,registered_budget_overruns=overruns,
            conclusion='reliable_gain' if reliable else 'worse' if interval[0]>0 else 'insufficient_evidence'))
    methods={}
    for opp in OPPONENTS:
        methods[opp]={}
        for method in METHODS:
            rr=[r for r in rows if r['opponent']==opp and r['method']==method]
            lat=[v for r in rr for v in registered_latencies(r)];first=[r['initial_total_ms'] for r in rr]
            methods[opp][method]=dict(games=len(rr),captures=sum(int(r['capture']) for r in rr),
                rediscoveries=sum(int(r['first_seen'] is not None) for r in rr),
                cost=float(np.mean([r['cost'] for r in rr])),rediscovery_cost=float(np.mean([r['rediscovery_cost'] for r in rr])),
                latency={f'p{p}':float(np.percentile(lat,p)) for p in (50,95,99)},maximum_ms=max(lat),
                first_max_ms=max(first),over_200ms=sum(x>200 for x in lat),first_over_200ms=sum(x>200 for x in first))
            methods[opp][method]['outcome_counts']={reason:sum(r['reason']==reason for r in rr)
                for reason in sorted({r['reason'] for r in rr})}
    real={r['comparison']:r for r in results if r['opponent']=='rootpuct'}
    matched={r['comparison']:r for r in results if r['opponent']=='matched'}
    structural=('ROUTE-SINGLE','ROUTE-GMM','ROUTE-HAND','ROUTE-DIRECT')
    if all(real[k]['reliable_gain'] for k in structural) and real['ROUTE-G']['reliable_gain']:
        direction='expand_generalization_then_2v1'
    elif any(matched[k]['reliable_gain'] for k in structural) and not real['ROUTE-G']['reliable_gain']:
        direction='matched_mechanism_only_prioritize_opponent_model_diagnosis_not_unique_causal_attribution'
    else:
        direction='end_current_configuration_report_complete_evidence_no_automatic_module_or_retraining'
    stats=dict(groups=len(groups),games=len(rows),methods=methods,comparisons=results,group_differences=diff.tolist(),
        correction='10 simultaneous intervals, grouped paired bootstrap',direction=direction,
        matched_scope='generator match before first rediscovery only',smoke=SMOKE)
    save(OUT/'evaluation/statistics.json',stats)
    return stats


def registered_latencies(row):
    """Every returned decision, charging initialization exactly once."""
    return [row['initial_total_ms']]+row['latencies_ms'][1:] if row['latencies_ms'] else []


def write_live_report(state):
    lines=['# E44：冻结 D 下的分支条件连续动作扩展','',state,'',
        '本轮无训练前性能屏障。局部公开输入、冻结 D、三网络三种子、同条件标签；真实对手与首次重发现前生成核匹配目标分别报告。',
        '这是 NTE 式局部信念适配，不是原论文完整复现。旧实验和部署不变。','',
        '- [冻结协议](PROTOCOL.md)','- [机器配置与源码摘要](experiment.json)','- [进度](status.json)',
        '- [教师选择](step1/selection.json)','- [训练](training.json)','- [闭环统计](evaluation/statistics.json)','']
    if (OUT/'training.json').exists():
        t=read(OUT/'training.json');lines += [f'已完成训练模型：{len(t.get("records",[]))}/9。','']
    if (OUT/'evaluation/statistics.json').exists():
        s=read(OUT/'evaluation/statistics.json')
        lines += [f'闭环完成：{s["games"]} 局；独立公共历史：{s["groups"]} 组。', '',
            '|对手|方法|捕获/局数|完成成本|重发现成本|决策P95 ms|','|---|---|---:|---:|---:|---:|']
        for opp,methods in s['methods'].items():
            for m,r in methods.items():lines.append(f'|{opp}|{m}|{r["captures"]}/{r["games"]}|{r["cost"]:.2f}|{r["rediscovery_cost"]:.2f}|{r["latency"]["p95"]:.2f}|')
        lines += ['','|条件|比较|主要成本差|10项同时95%区间|判断|','|---|---|---:|---|---|']
        for r in s['comparisons']:lines.append(f'|{r["opponent"]}|{r["comparison"]}|{r["difference"]:+.3f}|{r["simultaneous95"]}|{r["conclusion"]}|')
        lines += ['',f'方向裁决：`{s["direction"]}`。不自动启动对手模型或多车扩展。',
            '点估计持平不等于等效或非劣；观察成功率护栏不是统计非劣证明。匹配目标重发现后的捕获不是全程模型匹配证据。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


class BudgetExpired(BaseException):
    pass


def pipeline():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare();verify()
    clock_path=OUT/'clock.json'
    if not clock_path.exists():save(clock_path,dict(started_unix=time.time(),wall_cap_seconds=WALL_SECONDS))
    start=read(clock_path)['started_unix'];remaining=max(0,int(WALL_SECONDS-(time.time()-start)))
    if remaining<=0:raise RuntimeError('Registered wall budget exhausted')
    def alarm(*_):raise BudgetExpired('Registered E44 total wall budget exhausted')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(remaining)
    try:
        preflight();step1();label_data();train();trained_preflight();stats=evaluate();verify()
        save(OUT/'decision.json',dict(status='complete',formal_training_models=9,
            evaluation_games=stats['games'],direction=stats['direction'],performance_gate=False,
            elapsed_seconds=time.time()-start,smoke=SMOKE))
        write_live_report('完整流水线已完成。' if not SMOKE else '隔离工程 smoke 完成，不作性能结论。')
        emit(stage='complete',models=9,evaluation_games=stats['games'],elapsed_seconds=time.time()-start)
    except BaseException as exc:
        state='budget_stop' if isinstance(exc,BudgetExpired) else 'execution_failure'
        save(OUT/'failure.json',dict(status=state,error=repr(exc),previous_status=read(OUT/'status.json'),elapsed_seconds=time.time()-start))
        write_live_report(f'流水线状态：{state}；现场已保留，性能结果不触发停止。')
        emit(stage=state,error=repr(exc));raise
    finally:
        signal.alarm(0)


def main():
    global OUT,SMOKE,FORCED_TEACHER
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','pipeline','smoke','status'))
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.output is not None:OUT=args.output.resolve()
    if args.stage=='smoke':
        SMOKE=True
        if args.output is None:OUT=ROOT/'current/e44_branch_conditioned_nte_smoke_20261027'
    torch.set_num_threads(1)
    if args.stage=='prepare':prepare()
    elif args.stage=='status':print(json.dumps(read(OUT/'status.json'),ensure_ascii=False,indent=2))
    else:pipeline()


if __name__=='__main__':main()
