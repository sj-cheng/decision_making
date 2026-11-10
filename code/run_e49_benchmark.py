"""Resumable, audited E49 implementation and benchmark coordinator.

Run with the project dm_env_py311 Python; all scientific artifacts are isolated.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ['TZ'] = 'Asia/Shanghai'

import argparse
import concurrent.futures as cf
import datetime
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'current/e49_persistent_pursuit_benchmark_20261108'
SEEDS = (17, 29, 43)
HORIZONS = (40, 60, 80, 100, 120, 160)


def stamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              allow_nan=False, default=serializable)+'\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_seed(*values):
    return int(hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:8], 16)


def identity(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()


def revision():
    names = ['run_e49_benchmark.py', 'e49_rules.py', 'e49_controllers.py',
             'e49_learning.py', 'e49_bank.py', 'e49_audit.py', 'e49_report.py',
             'e49_adaptation.py', 'e49_engineering.py']
    return {name: sha(ROOT/'code'/name) for name in names if (ROOT/'code'/name).exists()}


def event(out, stage, **values):
    row = dict(time=stamp(), stage=stage, **values)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'events.jsonl').open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=serializable)+'\n')
    previous = read(out/'status.json') if (out/'status.json').exists() else {}
    previous.update(row)
    write(out/'status.json', previous)
    print(json.dumps(row, ensure_ascii=False, default=serializable), flush=True)


def save_episode(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with gzip.open(tmp, 'wb') as handle:
        pickle.dump(payload, handle, protocol=5)
    tmp.replace(path)


def load_episode(path):
    with gzip.open(path, 'rb') as handle:
        return pickle.load(handle)


def observation(state, boxes, horizon):
    from solvers.e45_team_adapter import public_observation
    obs = public_observation(int(state[8, 0]), state[4:8, 0].reshape(2, 2),
                             state[:4, 0].reshape(2, 2), state[11:13, 0] > .5,
                             state[9:11, 0] > .5, boxes)
    obs['remaining'] = horizon-obs['step']
    return obs


def reset_history(case, horizon):
    # Geometry/public history is immutable; only the declared task clock changes.
    return [dict(frame, remaining=horizon-frame['step']) for frame in case['history']]


def run_episode(task, out):
    from e49_rules import create_environment, make_opponent
    from e49_controllers import install, make_controller
    from e49_audit import audit_frame, audit_planner, diagnostics
    from run_frozen_d_closed_loop import legal_action
    horizon = task['horizon']
    install(horizon, Path(out))
    case = read(task['case'])
    case['horizon'] = horizon
    boxes = np.asarray(case['map'], float).reshape(-1, 2, 2)
    env = create_environment(boxes, horizon=horizon, capture_dt=task.get('capture_dt', .01))
    begin = time.perf_counter()
    own = make_controller(case, task)
    construction_ms = 1000*(time.perf_counter()-begin)
    histories = []
    for frame in reset_history(case, horizon)[:-1]:
        begin = time.perf_counter()
        own.observe(frame)
        histories.append(1000*(time.perf_counter()-begin))
    state = np.asarray(next(w['initial'] for w in case['worlds'] if w['id'] == task['world']), float).reshape(-1, 1)
    if state[8, 0] != 0:
        raise RuntimeError('E49 handoff must start at t=0')
    opponent = make_opponent(env, task['seed'], simulations=task.get('opponent_simulations', 3000))
    trace = []
    try:
        for _ in range(min(horizon, task.get('limit') or horizon)):
            obs = observation(state, boxes, horizon)
            wall, cpu = time.perf_counter(), time.process_time()
            action = np.asarray(own.act(obs, fixed_simulations=task.get('fixed')), float)
            latency = 1000*(time.perf_counter()-wall)
            cpu_ms = 1000*(time.process_time()-cpu)
            if action.shape != (2, 2) or not np.isfinite(action).all():
                raise RuntimeError('Malformed pursuer action')
            for i in range(2):
                if not legal_action(state[4+2*i:6+2*i, 0], action[i], 1., boxes):
                    raise RuntimeError('Illegal actual pursuer command')
            rng_before = opponent.get_rng_state(env)
            raw = np.asarray(opponent.search(env, state.copy(), turn=0).best_action, float).reshape(-1)[:4].reshape(2, 2)
            enemy = raw.copy()
            shields = 0
            for j in range(2):
                if state[9+j, 0] <= .5:
                    enemy[j] = 0
                elif not legal_action(state[2*j:2*j+2, 0], enemy[j], 1.5, boxes):
                    enemy[j] = 0
                    shields += 1
            joint = np.r_[enemy.ravel(), action.ravel()][:, None]
            nxt = env.step(state, joint, 1.)
            if not env.is_valid(nxt) or not np.isfinite(nxt).all():
                raise RuntimeError('Invalid physical next state')
            stats = dict(own.last_stats)
            # Keep the controller's independent planner transition audit evidence.
            simulated = getattr(own, 'traces', [])
            audited_simulations = audit_planner(getattr(own,'e49_simulated_evidence',[]), boxes)
            frame = dict(observation=obs, state=state[:, 0].tolist(), next_state=nxt[:, 0].tolist(),
                         action=action.tolist(), opponent_action=enemy.tolist(), opponent_raw_action=raw.tolist(),
                         opponent_shields=shields, opponent_rng_before=rng_before,
                         stats=stats, latency_ms=latency, cpu_ms=cpu_ms,
                         independently_audited_simulated_transitions=audited_simulations,
                         simulated_transitions=sum(len(t) for t in simulated),
                         simulated_trace=np.concatenate(simulated) if simulated else np.zeros((0, 26)))
            frame['capture_events'] = audit_frame(frame, boxes, task.get('capture_dt', .01))
            trace.append(frame)
            state = nxt
            if not (state[9:11, 0] > .5).any():
                break
    except Exception:
        try:
            evidence = dict(task=task, state=state, trace=trace,
                            opponent_rng=opponent.get_rng_state(env),
                            traceback=traceback.format_exc())
            save_episode(Path(out)/'failures'/(identity(task)+f'.{time.time_ns()}.pkl.gz'), evidence)
        finally:
            raise
    extra = diagnostics(trace, horizon)
    decisions = len(trace)
    complete = extra['double_capture']
    latencies = [f['latency_ms'] for f in trace]
    summary = dict(**extra, method=task['arm'], train_seed=task.get('train_seed', 17),
                   decisions=decisions, completion_cost=decisions if complete else horizon,
                   completion_cost_normalized=(decisions if complete else horizon)/horizon,
                   utility=float(complete)-.0025*decisions,
                   latencies_ms=latencies, construction_ms=construction_ms, history_ms=histories,
                   cpu_ms=[f['cpu_ms'] for f in trace], opponent_shields=sum(f['opponent_shields'] for f in trace),
                   online_overruns=sum(t > task.get('budget_ms', 200.) for t in latencies),
                   zero_simulation_decisions=sum(f['stats'].get('simulations') == 0 for f in trace),
                   stop_decisions=sum(np.max(np.abs(f['action'])) < 1e-10 for f in trace),
                   simulated_transitions=sum(f['simulated_transitions'] for f in trace),
                   independent_actual_audit=True)
    return dict(task=task, summary=summary, trace=trace)


def alarm(*_):
    raise TimeoutError('E49 episode watchdog (900s)')


def execute(task, out):
    out = Path(out)
    key = identity(task)
    if task.get('case_sha256') and sha(task['case'])!=task['case_sha256']:
        raise RuntimeError('E49 case identity mismatch')
    if task.get('source_revision') and task['source_revision']!=revision():
        raise RuntimeError('E49 source changed during running task')
    if task.get('checkpoint_sha256') and sha(task['checkpoint'])!=task['checkpoint_sha256']:
        raise RuntimeError('E49 checkpoint identity mismatch')
    result_path = out/'results'/task['stage']/(key+'.json')
    if result_path.exists():
        row = read(result_path)
        if row['valid'] and sha(row['episode']) == row['episode_sha256']:
            return row
        raise RuntimeError('Prior E49 failure requires repair: '+str(result_path))
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(900)
    try:
        payload = run_episode(task, out)
        path = out/'episodes'/task['stage']/(key+'.pkl.gz')
        save_episode(path, payload)
        row = dict(valid=True, task=task, summary=payload['summary'], episode=str(path),
                   episode_sha256=sha(path), seconds=time.perf_counter()-started,
                   finished=stamp(), cpu=sorted(os.sched_getaffinity(0)))
    except Exception as exc:
        row = dict(valid=False, task=task, error=repr(exc), traceback=traceback.format_exc(),
                   seconds=time.perf_counter()-started, finished=stamp())
    finally:
        signal.alarm(0)
    write(result_path, row)
    return row


def cpu_layout(max_workers=8):
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        p = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = ((p/'physical_package_id').read_text().strip(), (p/'core_id').read_text().strip())
        cores.setdefault(key, []).append(cpu)
    physical = list(cores.values())
    reserved = [c for group in physical[:10] for c in group]
    if len(reserved) < 20 or len(physical) < 12:
        raise RuntimeError('Cannot reserve the registered 20 logical CPUs')
    selected = physical[-min(max_workers, len(physical)-10):]
    return dict(cpus=[v[0] for v in selected], idle_siblings=[c for v in selected for c in v[1:]],
                reserved=reserved,training=[c for group in physical[10:13] for c in group])


def worker_init(queue):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.sched_setaffinity(0, {queue.get()})
    from e49_rules import warmup_rules
    warmup_rules()


def timing_probe(argument):
    case_path,cpu = argument
    os.sched_setaffinity(0,{cpu})
    from e49_rules import warmup_rules
    from e49_controllers import install,make_controller
    warmup_rules();install(80)
    case=read(case_path);samples=[]
    for repeat in range(3):
        own=make_controller(case,dict(arm='A2',seed=14917,horizon=80,budget_ms=200.))
        for obs in reset_history(case,80)[:-1]:own.observe(obs)
        started=time.perf_counter()
        own.act(reset_history(case,80)[-1],fixed_simulations=16)
        elapsed=1000*(time.perf_counter()-started)
        if repeat:samples.append(elapsed)
    return samples


def calibrate_cpus(out,entry,requested):
    """Fixed-work concurrency gate with inactive sibling threads in our pools."""
    path=Path(out)/'timing_calibration.json'
    if path.exists():
        data=read(path)
        # Reuse a calibration only when it covers the requested worker count.
        # A later run may intentionally expand the pool after the user grants
        # more CPU capacity; that expansion must receive its own fixed-work
        # timing check instead of silently retaining the smaller pool.
        if (data['source_revision']==revision()
                and len(data['selected_cpus'])>=len(requested)
                and set(requested)<=set(data['selected_cpus'])):
            return requested
    context=mp.get_context('spawn')
    with cf.ProcessPoolExecutor(1,mp_context=context) as pool:
        reference=pool.submit(timing_probe,(entry['path'],requested[0])).result()
    cpus=list(requested);attempts=[]
    while True:
        with cf.ProcessPoolExecutor(len(cpus),mp_context=context) as pool:
            parallel=list(pool.map(timing_probe,[(entry['path'],cpu) for cpu in cpus]))
        pooled=[value for samples in parallel for value in samples]
        ratio=float(np.median(pooled)/np.median(reference))
        tail=float(np.percentile(pooled,95)/np.percentile(reference,95))
        passed=ratio<=1.35 and tail<=1.5
        attempts.append(dict(cpus=cpus,reference_ms=reference,parallel_ms=parallel,
                             median_ratio=ratio,p95_ratio=tail,passed=passed))
        write(path,dict(source_revision=revision(),attempts=attempts,
                         selected_cpus=cpus if passed else []))
        if passed:return cpus
        if len(cpus)==1:raise RuntimeError('E49 fixed-work timing gate failed even at1worker')
        cpus=cpus[:max(1,len(cpus)//2)]


def freeze_sources(out,specs):
    out=Path(out);sources=revision();models={}
    for spec in specs:
        if spec.get('checkpoint'):models[spec['checkpoint']]=sha(spec['checkpoint'])
    target=out/'source_snapshot'
    for name in sources:
        p=target/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((ROOT/'code'/name).read_bytes())
    # Include private native rules and binary identity in the frozen runtime.
    native={str(p):sha(p) for p in (out/'cpp/source').rglob('*') if p.is_file()}
    native.update({str(p):sha(p) for p in (out/'cpp/build').glob('bindings_e49*.so')})
    historical={}
    for base in (ROOT/'code/learning',ROOT/'code/solvers',ROOT/'code/problems',
                 ROOT/'third_party/mappo_on_policy/onpolicy',
                 ROOT/'current/l45_corner_turn_estimator_20261025/frozen_estimator'):
        for path in base.rglob('*.py'):
            if '__pycache__' in path.parts or '/cpp/build' in str(path):continue
            if path.name in sources and path.parent==ROOT/'code':continue
            historical[str(path)]=sha(path)
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.parent==ROOT/'code' and path.suffix=='.py' and path.name not in sources:
                historical[str(path)]=sha(path)
    data={str(out/name):sha(out/name) for name in ('bank_manifest.json','bank_allocation.json',
        'horizon_selection.json','adaptation/final_method_specs.json') if (out/name).exists()}
    record=dict(frozen_at=stamp(),source_revision=sources,model_hashes=models,
                native_hashes=native,historical_source_hashes=historical,data_hashes=data)
    write(out/'runtime_freeze.json',record)
    return record


def verify_freeze(out):
    freeze=read(Path(out)/'runtime_freeze.json')
    if freeze['source_revision']!=revision():raise RuntimeError('Frozen E49 source changed')
    for section in ('model_hashes','native_hashes','historical_source_hashes','data_hashes'):
        for path,digest in freeze[section].items():
            if sha(path)!=digest:raise RuntimeError('Frozen E49 artifact changed: '+path)
    return freeze


def batch(out, tasks, cpus):
    tasks = list(tasks)
    if not tasks:
        return []
    stage = tasks[0]['stage']
    event(out, stage, scheduled=len(tasks), completed=0)
    context = mp.get_context('spawn')
    queue = context.Queue()
    for cpu in cpus:
        queue.put(cpu)
    rows = []
    iterator = iter(tasks)
    with cf.ProcessPoolExecutor(len(cpus), mp_context=context,
                                initializer=worker_init, initargs=(queue,)) as pool:
        active = {}
        for _ in cpus:
            task = next(iterator, None)
            if task:
                active[pool.submit(execute, task, str(out))] = task
        while active:
            done, _ = cf.wait(active, return_when=cf.FIRST_COMPLETED, timeout=30)
            if not done:
                event(out, stage, scheduled=len(tasks), completed=len(rows), active=len(active))
            for future in done:
                task = active.pop(future)
                row = future.result()
                rows.append(row)
                event(out, stage, scheduled=len(tasks), completed=len(rows), last_method=task['arm'],
                      last_valid=row['valid'], last_seconds=row['seconds'])
                if not row['valid']:
                    for item in active:
                        item.cancel()
                    raise RuntimeError('E49 engineering failure: '+row['error'])
                task = next(iterator, None)
                if task:
                    active[pool.submit(execute, task, str(out))] = task
    return rows


def tasks_for(entries, specs, horizon, stage, streams=1, **kw):
    code = revision()
    for entry in entries:
        for world in ('AB', 'BA'):
            for stream in range(streams):
                seed = stable_seed('E49-20261108', entry['sha256'], world, stream)
                for spec in specs:
                    task = dict(spec, stage=stage, case=entry['path'], group=entry['id'],
                                case_sha256=entry['sha256'], family=entry['family'], suite=entry['suite'],
                                split=entry['split'], obstacle_tier=entry['obstacle_tier'],
                                world=world, stream=stream, seed=seed, horizon=horizon,
                                budget_ms=200., source_revision=code)
                    task.update(kw)
                    if task.get('checkpoint'):
                        task['checkpoint_sha256']=sha(task['checkpoint'])
                    yield task


def method_rates(rows):
    # Models/worlds/streams first averaged within each map, then equal map/tier.
    from collections import defaultdict
    cells = defaultdict(lambda: defaultdict(list))
    tiers = {}
    for row in rows:
        t = row['task']
        cells[t['arm']][t['group']].append(float(row['summary']['double_capture']))
        tiers[t['group']] = str(t['obstacle_tier'])
    result = {}
    for method, maps in cells.items():
        groups = defaultdict(list)
        for name, values in maps.items():
            groups[tiers[name]].append(float(np.mean(values)))
        result[method] = float(np.mean([np.mean(values) for values in groups.values()]))
    return result


def saturation(rates):
    if len(rates) != 12:
        raise RuntimeError('Horizon calibration requires all 12 methods')
    if sum(v >= .9 for v in rates.values()) >= 10:
        return 'easy'
    if sum(v <= .1 for v in rates.values()) >= 10:
        return 'hard'
    return 'usable'


def calibrate_horizon(out, entries, specs, cpus):
    target = Path(out)/'horizon_selection.json'
    if target.exists():
        record = read(target)
        if record['source_revision'] != revision():
            raise RuntimeError('Horizon calibration source revision changed; review required')
        return record['horizon']
    tested = []
    horizon, previous_kind, previous_horizon = 80, None, None
    route = []
    while True:
        rows = batch(out, tasks_for(entries, specs, horizon, f'calibration_H{horizon}'), cpus)
        rates = method_rates(rows)
        kind = saturation(rates)
        tested.append(dict(horizon=horizon, rates=rates, saturation=kind, episodes=len(rows)))
        write(Path(out)/'horizon_calibration_progress.json', tested)
        if kind == 'usable':
            reason = 'No universal floor or ceiling'
            break
        if previous_kind and kind != previous_kind:
            horizon = previous_horizon
            reason = 'Adjacent candidates crossed opposite saturation; keep previous candidate'
            break
        if not route:
            route = [60, 40] if kind == 'easy' else [100, 120, 160]
        if not route:
            reason = 'Candidate boundary remains saturated'
            break
        previous_kind, previous_horizon = kind, horizon
        horizon = route.pop(0)
        # Route exhaustion is handled after evaluating the boundary.
        if horizon in (40, 160):
            route = [None]
        if horizon is None:
            horizon = previous_horizon
            reason = 'Candidate boundary remains saturated'
            break
    write(target, dict(horizon=horizon, reason=reason, tested=tested,
                       source_revision=revision(), frozen_at=stamp(), formal_used=False))
    return horizon


def adaptation_decisions(rows):
    from collections import defaultdict
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row['task']['arm']][row['task']['group']].append(row['summary']['utility'])
    utilities = {arm: float(np.mean([np.mean(v) for v in maps.values()])) for arm, maps in groups.items()}
    rates = method_rates(rows)
    references = {'Single': [('SINGLE_PRIOR', 'GEOM_SEARCH')],
                  'L5': [('A3', 'A2'), ('A5', 'A4')],
                  'MAPPO_RouteAPF': [('MAPPO_RouteAPF', 'RouteAPF')], 'PPO_C': [('PPO_C', 'A4')]}
    decisions = {}
    for family, pairs in references.items():
        comparisons = [dict(method=a, reference=b, capture_deficit=rates[b]-rates[a],
                            utility_deficit=utilities[b]-utilities[a]) for a, b in pairs]
        decisions[family] = dict(triggered=any(c['capture_deficit'] >= .1-1e-10
                                               or c['utility_deficit'] >= .05-1e-10 for c in comparisons),
                                comparisons=comparisons)
    return decisions


def require_engineering(out):
    from e49_engineering import core_fingerprint
    path=Path(out)/'engineering_checks.json'
    if not path.exists() or not read(path).get('gate'):
        raise RuntimeError('Full E49 engineering gate has not passed')
    if read(path).get('core_end')!=core_fingerprint():
        raise RuntimeError('E49 engineering gate predates current physical/controller runtime')


def run_adaptation_subprocess(out,workers):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='3'
    dependencies=Path(out)/'python_deps'
    env['PYTHONPATH']=os.pathsep.join([str(dependencies),str(ROOT/'code'),env.get('PYTHONPATH','')])
    subprocess.run([sys.executable,str(Path(__file__)),'adapt','--out',str(out),
                    '--workers',str(workers)],env=env,check=True)


def formal(out,config,specs,horizon,cpus):
    require_engineering(out);verify_freeze(out)
    if not (Path(out)/'adaptation/final_method_specs.json').exists():
        raise RuntimeError('Adaptation decisions/selected models must be finalized')
    test=[entry for entry in config['entries'] if entry['split']=='test']
    jobs=list(tasks_for(test,specs,horizon,'formal',streams=2))
    if len(jobs)!=7392:raise RuntimeError('Wrong formal task matrix')
    write(Path(out)/'formal_task_manifest.json',dict(horizon=horizon,tasks=jobs,episodes=len(jobs)))
    event(out,'formal',formal_started=True,horizon=horizon,scheduled=len(jobs))
    rows=batch(out,jobs,cpus)
    verify_freeze(out)
    from e49_report import report
    stats=report(out,rows)
    event(out,'complete',status='complete',formal_started=True,formal_completed=len(rows),
          horizon=horizon,report=str(Path(out)/'final/REPORT.md'),
          capture_rates={k:v['double_capture'] for k,v in stats['methods'].items()})
    return rows


def run_pipeline(out,config,specs,workers):
    from e49_engineering import smoke,run_engineering,core_fingerprint
    gate=Path(out)/'engineering_checks.json'
    # Engineering never reads the formal test maps or method outcomes.
    if (not gate.exists() or not read(gate).get('gate')
            or read(gate).get('core_end')!=core_fingerprint()):
        event(out,'engineering',status='running')
        if not smoke(out,workers)['passed']:raise RuntimeError('E49 prefix/replay gate failed')
        if not run_engineering(out,workers)['gate']:raise RuntimeError('E49 full integration/sampling gate failed')
    require_engineering(out)
    dev=[e for e in config['entries'] if e['split']=='development']
    entry=next(e for e in dev if str(e['obstacle_tier'])=='24')
    event(out,'timing_calibration',status='running')
    cpus=calibrate_cpus(out,entry,cpu_layout(workers)['cpus'])
    freeze_sources(out,specs)
    event(out,'horizon_calibration',status='running',cpus=cpus)
    horizon=calibrate_horizon(out,dev,specs,cpus)
    verify_freeze(out)
    rows=batch(out,tasks_for(dev,specs,horizon,f'calibration_H{horizon}'),cpus)
    decisions=adaptation_decisions(rows)
    write(Path(out)/'adaptation_decisions.json',dict(horizon=horizon,families=decisions,created=stamp()))
    event(out,'adaptation',horizon=horizon,decisions=decisions)
    run_adaptation_subprocess(out,len(cpus))
    specs=read(Path(out)/'adaptation/final_method_specs.json')
    freeze_sources(out,specs)
    write(Path(out)/'frozen_methods.json',dict(horizon=horizon,methods=specs,
          source_revision=revision(),frozen_at=stamp()))
    formal(out,config,specs,horizon,cpus)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'engineering', 'calibrate', 'adapt', 'formal', 'pipeline', 'status'])
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    out = args.out
    if args.command == 'status':
        print(json.dumps(read(out/'status.json'), ensure_ascii=False, indent=2))
        return
    from e49_bank import prepare
    from e49_controllers import method_specs
    config = prepare(out, horizon=80)
    specs = method_specs()
    entries = config['entries']
    cpus = cpu_layout(args.workers)['cpus']
    write(out/'resource_layout.json', cpu_layout(args.workers))
    if args.command == 'prepare':
        event(out, 'prepared', maps=len(entries), methods=len(specs))
    elif args.command == 'engineering':
        from e49_engineering import smoke,run_engineering
        if not smoke(out,args.workers)['passed']:raise RuntimeError('E49 prefix/replay gate failed')
        result=run_engineering(out,args.workers)
        if not result['gate']:raise RuntimeError('E49 full integration/sampling gate failed')
        event(out,'engineering_passed',full_episodes=result['passed_full_episodes'])
    elif args.command == 'calibrate':
        require_engineering(out)
        freeze_sources(out,specs)
        dev = [e for e in entries if e['split'] == 'development']
        cpus=calibrate_cpus(out,next(e for e in dev if str(e['obstacle_tier'])=='24'),cpus)
        horizon = calibrate_horizon(out, dev, specs, cpus)
        rows = batch(out, tasks_for(dev, specs, horizon, f'calibration_H{horizon}'), cpus)
        decisions = adaptation_decisions(rows)
        write(out/'adaptation_decisions.json', dict(horizon=horizon, families=decisions, created=stamp()))
        event(out, 'calibrated', horizon=horizon, adaptation=decisions)
    elif args.command == 'formal':
        frozen=read(out/'frozen_methods.json');horizon=frozen['horizon']
        timing=read(out/'timing_calibration.json')
        formal(out,config,frozen['methods'],horizon,timing['selected_cpus'])
    elif args.command == 'adapt':
        require_engineering(out);verify_freeze(out)
        from e49_adaptation import adapt
        record=read(out/'adaptation_decisions.json')
        timing=read(out/'timing_calibration.json')
        os.sched_setaffinity(0,set(cpu_layout(args.workers)['training']))
        specs=adapt(out,config,specs,record['horizon'],record['families'],timing['selected_cpus'])
        write(out/'adaptation/final_method_specs.json',specs)
    elif args.command == 'pipeline':
        import fcntl
        lock=(out/'pipeline.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(out/'coordinator.json',dict(pid=os.getpid(),started=stamp(),command=sys.argv))
        try:
            run_pipeline(out,config,specs,args.workers)
        except BaseException as exc:
            event(out,'failed',status='failed',error=repr(exc),traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
