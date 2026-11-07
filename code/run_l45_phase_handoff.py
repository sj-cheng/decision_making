"""Frozen O/S phase handoff. No training, teacher calls or reward changes."""
import os
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_var] = '1'
import argparse
import contextlib
import copy
import fcntl
import io
import multiprocessing as mp
from pathlib import Path
import platform
import shutil
import threading
import time

import numpy as np
import torch
import run_l45_guarded_supervision as s
from benchmark_baselines import get_oracles, oracle_paths
from solvers.phase_puct import PrivatePUCT, PhaseSolver

o, v, f = s.o, s.v, s.f
ROOT = o.ROOT
SOURCE = s.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT/'current/l45_phase_handoff_20261023'
SMOKE_OUTPUT = ROOT/'current/l45_phase_handoff_smoke_20261023'
lock_json, locked_read = v.lock_json, v.locked_read
SPLITS = dict(validation_four=('four', 100), validation_none=('none', 50),
    confirm_four=('four', 400), confirm_none=('none', 100), guard_four=('four', 100),
    guard_none=('none', 100), timing_four=('four', 30), warmup=('four', 1),
    audit_four=('four', 1), audit_none=('none', 1))
SCIENCE = ('capture_count', 'double_capture', 'first_capture_time', 'capture_time', 'survival_time',
    'pursuer_return', 'evader_return', 'illegal_terminal', 'steps', 'legal_double_capture',
    'legal_survival', 'one_capture_timeout', 'pursuer_illegal', 'evader_illegal', 'completion_cost')


def config(args):
    return dict(schema=1, smoke=args.smoke, training_episodes=0, teacher_queries=0, optimizer_updates=0,
        budget=3 if args.smoke else 1000, search_only_budget=3 if args.smoke else 3000, depth=100,
        pursuer_betas=[.5, 0.], evader_betas=[.5, .1], completion_bonus=0.,
        candidate='H', candidate_phases=['O', 'S'], diagnostic_combinations=['OO', 'OS', 'SO', 'SS'],
        switch='next_real_decision_after_first_legal_capture; fixed_weights_within_each_search',
        evader_deployment='complete_O_configuration_including_internal_prediction',
        diagnostic_n=2 if args.smoke else 100, diagnostic_opponent='Search Only',
        diagnostic_minimum_net_successes=1, scenario_namespace=4802026102301 if args.smoke else 4802026102302,
        splits={name: [layout, 1 if args.smoke else n] for name,(layout,n) in SPLITS.items()},
        bootstrap_seed=48020261023, bootstrap_draws=20000,
        comparisons=[dict(id='H-O', candidate='H', reference='O'), dict(id='H-L', candidate='H', reference='L')],
        minimum_gain=.03, illegal_margin=.02, regression_margin=.03, no_automatic_expansion=True)


def settings(args, role=1):
    c = config(args)
    return dict(search_budget=c['budget'], search_only_budget=c['search_only_budget'],
        search_depth=100, beta_policy=.5, beta_value=0. if role == 1 else .1)


def fingerprints():
    paths = [Path(__file__), o.CODE/'solvers/phase_puct.py', o.CODE/'build_phase_search.py',
        o.CODE/'cpp/build_phase/build_manifest.json', *list((o.CODE/'cpp/build_phase').glob('*.so'))]
    if len(paths) != 5:
        raise ValueError('Build exactly one private phase extension first')
    return dict(private={str(p): o.sha(p) for p in paths}, historical=s.implementation_fingerprints(),
        platform=dict(machine=platform.machine(), system=platform.platform(), python=platform.python_version(),
            compiler=platform.python_compiler(), libc=list(platform.libc_ver()), numpy=np.__version__, torch=str(torch.__version__)))


def registry(args):
    return {name: dict(models=str(args.output/'models'/name), model_hash=o.model_fingerprint(args.output/'models'/name))
            for name in ('O', 'S', 'L')}


def prepare(args):
    out = args.output
    if out == ROOT or out == SOURCE or SOURCE in out.parents:
        raise ValueError('Output must be isolated')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'experiment.lock.json').exists():
        verify(args)
        return locked_read(out/'scenarios.json')
    s.verify(s.parse_args([]))
    prior = locked_read(SOURCE/'results.json')
    if prior['recommended'] != s.CONTROL:
        raise ValueError('Expected frozen old0 recommendation')
    selection = o.read_json(SOURCE/'training/S/seed0/selection.json')
    selected = next(r['model'] for r in selection['records'] if r['episode'] == 512)
    if o.sha(selected['source_checkpoint']) != selected['checkpoint_sha256']:
        raise ValueError('Full S512 checkpoint changed')
    origins = dict(O=SOURCE/'models/current', S=Path(selected['models']), L=SOURCE/'models/baseline')
    source = {str(SOURCE/name): o.sha(SOURCE/name) for name in
        ('results.json', 'recommended.json', 'experiment.json', 'scenarios.json', 'training/S/seed0/selection.json')}
    source[selected['source_checkpoint']] = selected['checkpoint_sha256']
    for origin in origins.values():
        source.update({str(origin/n): o.sha(origin/n) for n in o.MODEL_NAMES})
    for name, origin in origins.items():
        dest = out/'models'/name
        dest.mkdir(parents=True, exist_ok=True)
        for filename in o.MODEL_NAMES:
            shutil.copyfile(origin/filename, dest/filename)
    if registry(args)['S']['model_hash'] != selected['model_hash']:
        raise ValueError('S512 export differs from locked checkpoint')
    for robot in (0, 1):
        if len({registry(args)[name]['model_hash'][o.MODEL_NAMES[robot]] for name in ('O','S','L')}) != 1:
            raise ValueError('Evader actor must remain original')
    if len({registry(args)[name]['model_hash'][o.MODEL_NAMES[-1]] for name in ('O','S','L')}) != 1:
        raise ValueError('Original value network changed')
    history = s.history_files(args)
    used = s.history_scenes(history)
    old_n = len(used)
    banks = {}
    for phase, (name, (layout, n)) in enumerate(config(args)['splits'].items(), 1310):
        bank, index = [], 0
        while len(bank) < n:
            spec = o.make_scenario(o.scenario_seed(phase, config(args)['scenario_namespace'], index), layout)
            index += 1
            identity = s.previous.physical_hash(spec)
            if identity not in used:
                used.add(identity)
                bank.append(spec)
        banks[name] = bank
    oldbanks = locked_read(SOURCE/'scenarios.json')
    # Smoke uses a separate diagnostic bank; formal diagnostic intentionally
    # reuses every four-obstacle validation scene, regardless of outcome.
    diagnostic = banks['audit_four']+banks['audit_none'] if args.smoke else oldbanks['validation_four']
    if len(diagnostic) != config(args)['diagnostic_n']:
        raise ValueError('Incomplete diagnostic bank')
    historical_tasks = []
    if not args.smoke:
        srcargs = s.parse_args([])
        for item in s.requests(srcargs, [s.controls(srcargs)[0], selected], diagnostic,
                               opponents=('Search Only',)):
            row = s.read_cached(srcargs, item)
            name = 'O' if row['uid'] == s.CONTROL else 'S'
            historical_tasks.append(dict(prefix=name, spec=item['task']['spec'], row={k: row[k] for k in SCIENCE}))
            cache = SOURCE/'cache'/(item['key']+'.json')
            source[str(cache)] = o.sha(cache)
    record = dict(config=config(args), implementation=fingerprints(), source=source,
        history={str(p): o.sha(p) for p in history}, models=registry(args))
    lock_json(out/'diagnostic_bank.json', dict(scenes=diagnostic, expected=historical_tasks,
        intentional_history_reuse=not args.smoke))
    lock_json(out/'scenarios.json', banks)
    lock_json(out/'scenario_independence.json', dict(previous_physical_scenes=old_n,
        new_scenes=len(used)-old_n, overlap=0, diagnostic_history_exception=not args.smoke,
        scope='All recoverable l45 banks and rollout manifests including archives, smoke and unused scenes; not all NTE pretraining states.'))
    lock_json(out/'experiment.json', record)
    lock_json(out/'signature.json', dict(sha256=o.json_hash(dict(record=record,
        scenarios=o.sha(out/'scenarios.json'), diagnostic=o.sha(out/'diagnostic_bank.json')))))
    verify(args)
    return banks


def verify(args):
    record = locked_read(args.output/'experiment.json')
    if record['config'] != config(args) or record['implementation'] != fingerprints():
        raise ValueError('Protocol, implementation, build or platform changed')
    for path, sha in (record['source'] | record['history']).items():
        if o.sha(path) != sha:
            raise ValueError(f'Source or historical bank changed: {path}')
    if registry(args) != record['models']:
        raise ValueError('Frozen model changed')
    banks = locked_read(args.output/'scenarios.json')
    identities = [s.previous.physical_hash(x) for bank in banks.values() for x in bank]
    if len(identities) != len(set(identities)) or set(banks) != set(config(args)['splits']):
        raise ValueError('Duplicate or missing scenario splits')
    if set(identities) & s.history_scenes([Path(p) for p in record['history']]):
        raise ValueError('Scene leakage')
    for name,(layout,n) in config(args)['splits'].items():
        if len(banks[name]) != n or any(x['layout'] != layout for x in banks[name]):
            raise ValueError('Scene bank shape changed')
    diagnostic = locked_read(args.output/'diagnostic_bank.json')
    if len(diagnostic['scenes']) != config(args)['diagnostic_n']:
        raise ValueError('Missing diagnostic scenes')
    expected = o.json_hash(dict(record=record, scenarios=o.sha(args.output/'scenarios.json'),
        diagnostic=o.sha(args.output/'diagnostic_bank.json')))
    if locked_read(args.output/'signature.json')['sha256'] != expected:
        raise ValueError('Signature changed')
    return record


def private_solver(problem, model, method, role, requested, seed):
    if method == 'Policy Only':
        return f.make_solver(problem, model['models'], method, role, requested, seed)
    if o.model_fingerprint(model['models']) != model['model_hash']:
        raise ValueError('Worker model changed')
    paths, value_path = oracle_paths(str(ROOT), iteration=45, model_dir=model['models'])
    with contextlib.redirect_stdout(io.StringIO()):
        policies, value = get_oracles(problem, value_oracle_name='deterministic', value_oracle_path=value_path,
            policy_oracle_name='gaussian', policy_oracle_paths=paths)
        if method == 'Search Only':
            policies, value = [None]*problem.num_robots, None
        beta = (0.,0.) if method == 'Search Only' else (requested['beta_policy'], requested['beta_value'])
        solver = PrivatePUCT(policies, value,
            requested['search_only_budget'] if method == 'Search Only' else requested['search_budget'], *beta)
    solver.solver_wrapper.set_seed(int(seed))
    masked = o.RoleMaskedSolver(solver, o.TEAMS[role])
    masked.actual_settings = dict(method=method, budget=int(solver.solver_settings.number_simulations),
        depth=100, beta_policy=float(beta[0]), beta_value=float(beta[1]))
    return masked


def phase_pair(uid, role):
    if role == 0:
        return ('L','L') if uid == 'L' else ('O','O')
    return dict(H=('O','S'), O=('O','O'), L=('L','L'), S=('S','S'), SO=('S','O'))[uid]


def make_own(problem, reg, uid, role, requested, seed):
    before, after = phase_pair(uid, role)
    solvers = {name: private_solver(problem, reg[name], 'Search + Learning', role, requested, seed).solver
               for name in set((before, after))}
    return PhaseSolver(solvers, before, after, o.TEAMS[role])


def other_rng(solver, problem):
    return None if solver.actual_settings['method'] == 'Policy Only' else solver.solver.get_rng_state(problem)


def initial_stats():
    return dict(steps=0, reward=[0.,0.,0.,0.], first=None, last=None, capture_count=0,
        pursuer_illegal=0, evader_illegal=0)


def advance(problem, state, action, stats):
    reward = np.asarray(problem.reward(state, action), float).reshape(-1)
    nxt = problem.step(state, action, problem.dt)
    stats['reward'] = (np.asarray(stats['reward'])+reward).tolist()
    stats['steps'] += 1
    count = 2-problem.active_evader_count(nxt)
    if count > stats['capture_count']:
        if stats['first'] is None:
            stats['first'] = float(nxt[problem.time_idx, 0])
        stats['last'] = float(nxt[problem.time_idx, 0])
    stats['capture_count'] = int(count)
    violations = o.violations(problem, nxt)
    stats['evader_illegal'] |= int(violations[0])
    stats['pursuer_illegal'] |= int(violations[1])
    return nxt


def metrics(problem, state, stats):
    count, illegal = stats['capture_count'], bool(stats['pursuer_illegal'] or stats['evader_illegal'])
    t = float(state[problem.time_idx,0])
    return dict(capture_count=count, double_capture=int(count == 2),
        first_capture_time=stats['first'] if count else float('nan'),
        capture_time=stats['last'] if count == 2 else float('nan'), survival_time=t,
        pursuer_return=float(np.sum(np.asarray(stats['reward'])[list(o.TEAMS[1])])),
        evader_return=float(np.sum(np.asarray(stats['reward'])[list(o.TEAMS[0])])),
        illegal_terminal=int(illegal), steps=stats['steps'], legal_double_capture=int(count == 2 and not illegal),
        legal_survival=int(count < 2 and t >= problem.tf and not illegal),
        one_capture_timeout=int(count == 1 and t >= problem.tf and not illegal),
        pursuer_illegal=stats['pursuer_illegal'], evader_illegal=stats['evader_illegal'],
        completion_cost=stats['last'] if count == 2 and not illegal else float(problem.tf))


def science(row):
    return o.json_hash({k: row[k] for k in SCIENCE})


def simulate(problem, state, own, other, role, stats=None, stop_first=False, trace=None):
    stats = initial_stats() if stats is None else copy.deepcopy(stats)
    team = o.TeamSolver(other if role == 1 else own, own if role == 1 else other)
    decisions, switching = [], []
    while not problem.is_terminal(state):
        action = team.policy(problem, state)
        decisions.append([team.last_pursuer_latency_ms, team.last_evader_latency_ms, team.last_joint_latency_ms])
        if getattr(own, 'last_switched', False):
            switching.append(team.last_pursuer_latency_ms if role == 1 else team.last_evader_latency_ms)
        nxt = advance(problem, state, action, stats)
        if trace is not None:
            trace.append(dict(state=state.tolist(), action=action.tolist(), next_state=nxt.tolist()))
        state = nxt
        if stop_first and stats['capture_count'] > 0:
            break
        if stats['steps'] > int(problem.tf/problem.dt):
            raise RuntimeError('Exceeded original horizon')
    return state, stats, decisions, switching


def requests(args, uids, scenes, role=1, opponents=o.OPPONENTS, timing='parallel'):
    reg = registry(args)
    items = []
    for uid in uids:
        effective = dict(phases=phase_pair(uid, role), weights={n: reg[n]['model_hash'] for n in set(phase_pair(uid, role))})
        for spec in scenes:
            for opponent in opponents:
                identity = dict(protocol=o.sha(args.output/'experiment.json'), policy_id=o.json_hash(effective),
                    episode_hash=spec['episode_hash'], role=role, opponent=opponent, timing_context=timing,
                    own_settings=settings(args, role), opponent_settings=settings(args, 0), mode='search')
                key = o.json_hash(identity)
                task = dict(**identity, key=key, uid=uid, spec=spec, registry=reg)
                items.append(dict(key=key, uid=uid, task=task))
    return items


def evaluate_one(task, legacy=False):
    torch.set_num_threads(1)
    problem, state = o.restore_scenario(task['spec'])
    problem.clear_closed_loop_cache()
    role, uid, reg = task['role'], task['uid'], task['registry']
    for model in reg.values():
        if o.model_fingerprint(model['models']) != model['model_hash']:
            raise ValueError('Worker weights changed')
    before, after = phase_pair(uid, role)
    if legacy:
        if before != after:
            raise ValueError('Legacy cannot run hybrid')
        own = f.make_solver(problem, reg[before]['models'], 'Search + Learning', role,
            task['own_settings'], task['spec']['search_seeds'][role])
        other = f.make_solver(problem, reg['L']['models'], task['opponent'], 1-role,
            task['opponent_settings'], task['spec']['search_seeds'][1-role])
    else:
        own = make_own(problem, reg, uid, role, task['own_settings'], task['spec']['search_seeds'][role])
        other = private_solver(problem, reg['L'], task['opponent'], 1-role,
            task['opponent_settings'], task['spec']['search_seeds'][1-role])
    machine = v.machine_snapshot() if task['timing_context'] == 'single_process' else None
    started = time.perf_counter()
    if legacy:
        team = o.TeamSolver(other if role == 1 else own, own if role == 1 else other)
        original = o.run_episode(problem, team, state)
        illegal = original['illegal_terminal']
        original.update(legal_double_capture=int(original['double_capture'] and not illegal),
            legal_survival=int(original['capture_count'] < 2 and original['survival_time'] >= problem.tf and not illegal),
            one_capture_timeout=int(original['capture_count'] == 1 and original['survival_time'] >= problem.tf and not illegal),
            pursuer_illegal=int(original['pursuer_obstacle_collision'] or original['pursuer_out_of_bounds']),
            evader_illegal=int(original['evader_obstacle_collision'] or original['evader_out_of_bounds']))
        original['completion_cost'] = original['capture_time'] if original['legal_double_capture'] else float(problem.tf)
        row = {k: original[k] for k in SCIENCE}
        decisions, switching = [], []
    else:
        state, stats, decisions, switching = simulate(problem, state, own, other, role)
        row = metrics(problem, state, stats)
    row.update({k: task[k] for k in ('key','policy_id','role','opponent','timing_context','mode')})
    row.update(uid=uid, episode_hash=task['spec']['episode_hash'], layout=task['spec']['layout'],
        search_seeds=task['spec']['search_seeds'], episode_wall_ms=(time.perf_counter()-started)*1000.,
        phase_pair=[before, after], switches=getattr(own, 'switches', 0),
        actual_own=dict(task['own_settings'], phases=[before,after]), actual_opponent=other.actual_settings)
    if task['timing_context'] in ('warmup','single_process'):
        row.update(decision_latency_ms=decisions, switch_decision_ms=switching)
    if machine:
        row['timing_machine'] = dict(before=machine, after=v.machine_snapshot())
    return row


def read_cached(args, item):
    path = args.output/'cache'/(item['key']+'.json')
    if locked_read(args.output/'cache_checksums'/(item['key']+'.json'))['sha256'] != o.sha(path):
        raise ValueError('Committed cache file changed')
    row = o.read_json(path)
    digest = row.pop('content_sha256', None)
    task = item['task']
    if digest != o.json_hash(row) or any(row[k] != task[k] for k in
            ('key','policy_id','role','opponent','timing_context','mode')):
        raise ValueError('Cache digest or identity mismatch')
    if row['episode_hash'] != task['spec']['episode_hash'] or row['search_seeds'] != task['spec']['search_seeds']:
        raise ValueError('Unpaired scenes or seeds')
    if row['actual_own'] != dict(task['own_settings'], phases=list(phase_pair(item['uid'], task['role']))):
        raise ValueError('Cached role phase configuration changed')
    other = row['actual_opponent']
    requested = task['opponent_settings']
    if other['method'] != task['opponent']:
        raise ValueError('Cached opponent changed')
    if task['opponent'] != 'Policy Only':
        budget = requested['search_only_budget' if task['opponent'] == 'Search Only' else 'search_budget']
        beta = [0.,0.] if task['opponent'] == 'Search Only' else [requested['beta_policy'],requested['beta_value']]
        if other['budget'] != budget or other['depth'] != 100 or not np.allclose([other['beta_policy'],other['beta_value']], beta):
            raise ValueError('Cached opponent settings changed')
    row['uid'] = item['uid']  # H and O share the complete evader behavior.
    return row


def run_evaluation(args, name, items, pool, status):
    lock_json(args.output/name/'manifest.json', [{k: val for k,val in item.items() if k != 'task'} for item in items])
    unique = {item['key']: item for item in items}
    missing = []
    switches = 0
    for key,item in unique.items():
        if (args.output/'cache_checksums'/(key+'.lock.json')).exists():
            switches += read_cached(args, item)['switches']
        else:
            missing.append(item['task'])
    done = len(unique)-len(missing)
    status(name, completed=done, total=len(unique), label_requests=len(items), committed_switches=switches)
    iterator = pool.imap_unordered(evaluate_one, missing, chunksize=1) if pool else map(evaluate_one, missing)
    for row in iterator:
        row['content_sha256'] = o.json_hash(row)
        path = args.output/'cache'/(row['key']+'.json')
        o.write_json(path, row)
        lock_json(args.output/'cache_checksums'/(row['key']+'.json'),dict(sha256=o.sha(path)))
        switches += row['switches']
        done += 1
        if done % 10 == 0 or done == len(unique):
            status(name, completed=done, total=len(unique), label_requests=len(items), committed_switches=switches)
    rows = [read_cached(args, item) for item in items]
    summary = [dict(uid=uid, layout=layout, **v.aggregate(rows,uid,layout,items[0]['task']['role']))
        for uid in sorted({r['uid'] for r in rows}) for layout in sorted({r['layout'] for r in rows})]
    o.write_json(args.output/name/'summary.json', summary)
    return rows


def diagnostic_tasks(args):
    bank = locked_read(args.output/'diagnostic_bank.json')
    tasks = []
    for spec in bank['scenes']:
        for prefix in ('O','S'):
            identity = dict(protocol=o.sha(args.output/'experiment.json'), spec=spec, prefix=prefix,
                settings=settings(args), opponent_settings=settings(args,0), registry=registry(args), kind='prefix')
            key = o.json_hash(identity)
            tasks.append(dict(**identity, key=key, result=str(args.output/'prefix_cache'/(key+'.pt'))))
    return tasks


def read_segment(task):
    path = Path(task['result'])
    if locked_read(path.with_suffix('.json'))['sha256'] != o.sha(path):
        raise ValueError('Segment cache tampered')
    row = o.load_torch(path)
    if row['key'] != task['key'] or row['task'] != {k:v for k,v in task.items() if k != 'result'}:
        raise ValueError('Segment cache identity mismatch')
    return row


def prefix_one(task):
    torch.set_num_threads(1)
    p, state = o.restore_scenario(task['spec'])
    p.clear_closed_loop_cache()
    reg = task['registry']
    own = make_own(p, reg, task['prefix'], 1, task['settings'], task['spec']['search_seeds'][1])
    other = private_solver(p, reg['L'], 'Search Only', 0, task['opponent_settings'], task['spec']['search_seeds'][0])
    trace = []
    state, stats, _, _ = simulate(p, state, own, other, 1, stop_first=True, trace=trace)
    branch = not p.is_terminal(state) and stats['capture_count'] == 1 and not any(o.violations(p,state))
    geometry = None
    if branch:
        e = next(r for r in o.TEAMS[0] if p.is_active(state,r))
        pursuer = next(r for r in o.TEAMS[1] if p.is_active(state,r))
        geometry = dict(evader=e, pursuer=pursuer, first_capture_time=stats['first'],
            remaining_time=p.tf-float(state[p.time_idx,0]),
            distance=float(np.linalg.norm(state[p.state_idxs[e],0]-state[p.state_idxs[pursuer],0])),
            speed_ratio=task['spec']['pursuer_speed']/task['spec']['evader_speed'])
    return dict(key=task['key'], task={k:v for k,v in task.items() if k != 'result'},
        state=state, stats=stats, branch=branch, geometry=geometry, trajectory=trace,
        own_rng=own.get_rng_state(p), opponent_rng=other_rng(other,p), python_rng=o.rng_state())


def suffix_tasks(args, prefixes):
    tasks = []
    for prefix_task in prefixes:
        for after in ('O','S'):
            identity = dict(protocol=o.sha(args.output/'experiment.json'), prefix_task=prefix_task,
                prefix_sha256=o.sha(prefix_task['result']), after=after, kind='suffix')
            key = o.json_hash(identity)
            tasks.append(dict(**identity, key=key, result=str(args.output/'suffix_cache'/(key+'.pt'))))
    return tasks


def suffix_one(task):
    torch.set_num_threads(1)
    pt = task['prefix_task']
    if o.sha(pt['result']) != task['prefix_sha256']:
        raise ValueError('Branch prefix changed')
    prefix = read_segment(pt)
    p, _ = o.restore_scenario(pt['spec'])
    p.clear_closed_loop_cache()
    uid = dict(OO='O', OS='H', SO='SO', SS='S')[pt['prefix']+task['after']]
    own = make_own(p, pt['registry'], uid, 1, pt['settings'], pt['spec']['search_seeds'][1])
    other = private_solver(p, pt['registry']['L'], 'Search Only', 0,
        pt['opponent_settings'], pt['spec']['search_seeds'][0])
    # O/O and S/S need only one solver; restore the prefix's exact stream.
    own.set_rng_state(p, prefix['own_rng'])
    other.solver.set_rng_state(p, prefix['opponent_rng'])
    o.restore_rng(prefix['python_rng'])
    state, stats = prefix['state'].copy(), copy.deepcopy(prefix['stats'])
    trace = []
    if prefix['branch']:
        state, stats, _, _ = simulate(p,state,own,other,1,stats=stats,trace=trace)
    row = metrics(p,state,stats)
    row.update(combination=pt['prefix']+task['after'], episode_hash=pt['spec']['episode_hash'],
        branched=prefix['branch'], geometry=prefix['geometry'])
    return dict(key=task['key'], task={k:v for k,v in task.items() if k != 'result'},
        metrics=row, trajectory=trace, state=state, own_rng=own.get_rng_state(p), opponent_rng=other_rng(other,p))


def run_segments(args, name, tasks, worker, pool, status):
    lock_json(args.output/name/'manifest.json', tasks)
    missing = []
    switches = 0
    for task in tasks:
        if Path(task['result']).with_suffix('.lock.json').exists():
            cached = read_segment(task)
            switches += cached['own_rng']['switches']
        else:
            missing.append(task)
    done = len(tasks)-len(missing)
    status(name, completed=done,total=len(tasks),committed_switches=switches)
    iterator = pool.imap_unordered(worker,missing,chunksize=1) if pool else map(worker,missing)
    by_key = {t['key']: t for t in tasks}
    for row in iterator:
        path = Path(by_key[row['key']]['result'])
        o.save_torch(path,row)
        lock_json(path.with_suffix('.json'),dict(sha256=o.sha(path)))
        switches += row['own_rng']['switches']
        done += 1
        if done % 10 == 0 or done == len(tasks):
            status(name,completed=done,total=len(tasks),committed_switches=switches)
    return [read_segment(t) for t in tasks]


def diagnose(args, banks, pool, status):
    prefixes = diagnostic_tasks(args)
    run_segments(args,'diagnostic_prefixes',prefixes,prefix_one,pool,status)
    suffixes = suffix_tasks(args,prefixes)
    results = run_segments(args,'diagnostic_suffixes',suffixes,suffix_one,pool,status)
    rows = [r['metrics'] for r in results]
    indexed = {c:{r['episode_hash']:r for r in rows if r['combination'] == c} for c in ('OO','OS','SO','SS')}
    n = config(args)['diagnostic_n']
    if any(len(part) != n for part in indexed.values()):
        raise ValueError('Missing cross-handoff pairs')
    for record in locked_read(args.output/'diagnostic_bank.json')['expected']:
        actual = indexed[record['prefix']*2][record['spec']['episode_hash']]
        if science(actual) != science(record['row']):
            raise ValueError('OO/SS does not reproduce historical scientific result')
    pairs = {}
    for candidate,reference in (('OS','OO'),('SO','SS')):
        a,b = indexed[candidate],indexed[reference]
        gained = sum(a[k]['legal_double_capture'] > b[k]['legal_double_capture'] for k in a)
        lost = sum(a[k]['legal_double_capture'] < b[k]['legal_double_capture'] for k in a)
        branched = [k for k in a if a[k]['branched']]
        pairs[candidate+'-'+reference] = dict(gained=gained,lost=lost,net=gained-lost,
            delta=(gained-lost)/n, games=n, branched_games=len(branched),
            conditional_delta=(gained-lost)/len(branched) if branched else None,
            own_illegal_delta=float(np.mean([a[k]['pursuer_illegal']-b[k]['pursuer_illegal'] for k in a])))
    summary = {c:dict(games=n, successes=sum(r['legal_double_capture'] for r in part.values()),
        success_rate=float(np.mean([r['legal_double_capture'] for r in part.values()])),
        one_capture_timeout=sum(r['one_capture_timeout'] for r in part.values()),
        zero_capture=sum(r['capture_count'] == 0 for r in part.values()),
        pursuer_illegal=sum(r['pursuer_illegal'] for r in part.values())) for c,part in indexed.items()}
    primary = pairs['OS-OO']
    observed = primary['net'] >= 1 and primary['own_illegal_delta'] <= .02+1e-12
    return lock_json(args.output/'diagnostic_result.json',dict(passed=observed and not args.smoke,
        observed_gate=observed, smoke=args.smoke, pairs=pairs, summary=summary, rows=rows,
        historical_parity_checked=not args.smoke, only_candidate='H', bootstrap_claim=False))


def comparison(rows, reference, confidence=None):
    four = s.point_comparison(rows,'H',reference,'four')
    none = s.point_comparison(rows,'H',reference,'none')
    checks = s.performance_checks(four,none,gain=True)
    checks['no_obstacle_own_illegal'] = none['own_illegal_delta'] <= .02+1e-12
    if confidence is not None:
        checks['simultaneous_lower_positive'] = confidence['simultaneous_lower95'] > 0.
    return dict(four=four,none=none,checks=checks,passed=all(checks.values()))


def statistics(rows, draws=20000):
    specs = config(parse_args([]))['comparisons']
    indexed = {uid:v.indexed_rows(rows,uid,'four',1) for uid in ('H','O','L')}
    keys = sorted(indexed['H'])
    if any(set(part) != set(keys) for part in indexed.values()):
        raise ValueError('Missing statistical pairs')
    scenes = sorted({key[0] for key in keys})
    differences, output = [], []
    for spec in specs:
        a,b = indexed['H'],indexed[spec['reference']]
        if any(a[k]['search_seeds'] != b[k]['search_seeds'] for k in keys):
            raise ValueError('Statistical seed mismatch')
        for scene in scenes:
            if {op for sc,op in keys if sc == scene} != set(o.OPPONENTS):
                raise ValueError('All three opponents must be present')
        delta = {k:a[k]['legal_double_capture']-b[k]['legal_double_capture'] for k in keys}
        differences.append([np.mean([delta[k] for k in keys if k[0] == scene]) for scene in scenes])
        output.append(dict(**spec,delta=float(np.mean(list(delta.values()))),n_scenes=len(scenes),n_games=len(keys),
            gained=sum(x>0 for x in delta.values()),lost=sum(x<0 for x in delta.values())))
    differences = np.asarray(differences)
    observed = differences.mean(1)
    rng = np.random.default_rng(48020261023)
    samples = np.empty((draws,2))
    for start in range(0,draws,256):
        stop = min(start+256,draws)
        chosen = rng.integers(len(scenes),size=(stop-start,len(scenes)))
        samples[start:stop] = differences[:,chosen].mean(-1).T
    correction = max(0.,float(np.quantile(np.max(samples-observed,axis=1),.95)))
    for i,result in enumerate(output):
        result.update(ci95=np.quantile(samples[:,i],[.025,.975]).tolist(),simultaneous_lower95=float(observed[i]-correction))
    return dict(comparisons=output,draws=draws,bootstrap_seed=48020261023,
        method='max_centered_joint_one_sided_95',correction=correction)


def validate(args,banks,pool,status):
    if not locked_read(args.output/'diagnostic_result.json')['passed']:
        raise ValueError('Diagnostic gate did not authorize validation')
    rows = run_evaluation(args,'validation',requests(args,['H','O','L'],banks['validation_four']+banks['validation_none']),pool,status)
    comparisons = {ref:comparison(rows,ref) for ref in ('O','L')}
    passed = all(r['passed'] for r in comparisons.values()) and not args.smoke
    result = lock_json(args.output/'validation_result.json',dict(passed=passed,comparisons=comparisons,
        aggregates={uid:v.aggregate(rows,uid) for uid in ('H','O','L')}))
    if passed:
        lock_json(args.output/'candidate_lock.json',dict(candidate='H',phases=['O','S'],models=registry(args),
            implementation=fingerprints(),comparisons=config(args)['comparisons'],validation_sha256=o.sha(args.output/'validation_result.json')))
    return result


def confirm(args,banks,pool,status):
    if not locked_read(args.output/'validation_result.json')['passed']:
        raise ValueError('Validation did not authorize confirmation')
    lock = locked_read(args.output/'candidate_lock.json')
    if lock['candidate'] != 'H' or lock['models'] != registry(args) or lock['implementation'] != fingerprints() or \
            lock['validation_sha256'] != o.sha(args.output/'validation_result.json'):
        raise ValueError('Candidate lock changed')
    rows = run_evaluation(args,'confirmation',requests(args,['H','O','L'],banks['confirm_four']+banks['confirm_none']),pool,status)
    stats = statistics(rows)
    comparisons = {r['reference']:comparison(rows,r['reference'],r) for r in stats['comparisons']}
    return lock_json(args.output/'confirmation_result.json',dict(passed=all(r['passed'] for r in comparisons.values()) and not args.smoke,
        statistics=stats,comparisons=comparisons))


def guard(args,banks,pool,status):
    if not locked_read(args.output/'confirmation_result.json')['passed']:
        raise ValueError('Confirmation did not authorize regression')
    rows = run_evaluation(args,'evader_regression',requests(args,['H','O','L'],banks['guard_four']+banks['guard_none'],role=0),pool,status)
    comparisons = {ref:{layout:s.point_comparison(rows,'H',ref,layout,role=0) for layout in ('four','none')} for ref in ('O','L')}
    passed = all(r['delta'] >= -.03-1e-12 and r['own_illegal_delta'] <= .02+1e-12
        for layouts in comparisons.values() for r in layouts.values())
    return lock_json(args.output/'guard_result.json',dict(passed=passed,comparisons=comparisons,
        evader_complete_O=True,logical_requests=len(rows),unique_requests=len({i['key'] for i in requests(args,['H','O','L'],banks['guard_four']+banks['guard_none'],role=0)})))


def timing(args,banks,status):
    if not locked_read(args.output/'guard_result.json')['passed']:
        raise ValueError('Regression did not authorize timing')
    if mp.active_children():
        raise RuntimeError('Exit parallel pool before serial timing')
    path = args.output/'timing_result.json'
    if path.with_suffix('.lock.json').exists():
        return locked_read(path)
    for item in requests(args,['H','O','L'],banks['warmup'],timing='warmup'):
        evaluate_one(item['task'])
    items = []
    for index,spec in enumerate(banks['timing_four']):
        order = ['H','O','L']
        order = order[index%3:]+order[:index%3]
        for op in o.OPPONENTS:
            items.extend(requests(args,order,[spec],opponents=(op,),timing='single_process'))
    rows = run_evaluation(args,'serial_timing',items,None,status)
    summaries = {}
    for uid in ('H','O','L'):
        part = [r for r in rows if r['uid'] == uid]
        decisions = np.concatenate([np.asarray(r['decision_latency_ms']) for r in part])[:,0]
        switched = [x for r in part for x in r['switch_decision_ms']]
        summaries[uid] = dict(pursuer_mean_ms=float(decisions.mean()),pursuer_p95_ms=float(np.percentile(decisions,95)),
            switch_mean_ms=float(np.mean(switched)) if switched else None,switch_decisions=len(switched),
            episode_mean_ms=float(np.mean([r['episode_wall_ms'] for r in part])),episodes=len(part),
            per_opponent={op:dict(episode_mean_ms=float(np.mean([r['episode_wall_ms'] for r in part if r['opponent'] == op]))) for op in o.OPPONENTS})
    for row in summaries.values():
        row['ratio_to_O'] = row['pursuer_mean_ms']/summaries['O']['pursuer_mean_ms']
    return lock_json(path,summaries)


def audit_episode(task):
    old = evaluate_one(task,legacy=True)
    new = evaluate_one(task)
    if science(old) != science(new):
        raise ValueError('Original full episode parity failed')
    return dict(layout=task['spec']['layout'],role=task['role'],opponent=task['opponent'],episode_exact=True)


def audit(args,banks,status=lambda *a,**k:None):
    path = args.output/'audit_result.json'
    if path.with_suffix('.lock.json').exists():
        return locked_read(path)
    probes = []
    reg = registry(args)
    for layout in ('four','none'):
        spec = banks['audit_'+layout][0]
        for role in (0,1):
            p,state = o.restore_scenario(spec)
            for phase in (0,1):
                current = state.copy()
                if phase:
                    current[[p.active_idxs[0],p.active_idxs[2]],0] = 0.
                old = f.make_solver(p,reg['O']['models'],'Search + Learning',role,settings(args,role),spec['search_seeds'][role])
                new = make_own(p,reg,'O',role,settings(args,role),spec['search_seeds'][role])
                before = new.get_rng_state(p)
                action = new.policy(p,current)
                new.set_rng_state(p,before)
                if not np.array_equal(action,new.policy(p,current)) or not np.array_equal(action,old.policy(p,current)):
                    raise ValueError('Original action or RNG roundtrip parity failed')
                probes.append(dict(layout=layout,role=role,phase=phase,action_rng_exact=True))
    tasks = [item['task'] for role in (0,1) for layout in ('four','none')
        for item in requests(args,['O'],banks['audit_'+layout],role=role)]
    with mp.get_context('spawn').Pool(min(args.workers,len(tasks))) as pool:
        for index,probe in enumerate(pool.imap_unordered(audit_episode,tasks,chunksize=1),1):
            probes.append(probe)
            status('audit',completed=index,total=len(tasks))
    # Segmentation is compared to uninterrupted calls, including both RNGs
    # and every executable action, at the same (possibly smoke) budget.
    segmented = []
    spec = banks['audit_none'][0]
    for uid,prefix_name,after in (('O','O','O'),('S','S','S'),('H','O','S')):
        task = dict(protocol=o.sha(args.output/'experiment.json'),spec=spec,prefix=prefix_name,
            settings=settings(args),opponent_settings=settings(args,0),registry=reg,kind='prefix')
        task.update(key=o.json_hash(task),result=str(args.output/'audit_segments'/f'{uid}.pt'))
        prefix = prefix_one(task)
        o.save_torch(task['result'],prefix)
        lock_json(Path(task['result']).with_suffix('.json'),dict(sha256=o.sha(task['result'])))
        st = dict(protocol=o.sha(args.output/'experiment.json'),prefix_task=task,prefix_sha256=o.sha(task['result']),after=after,kind='suffix')
        st.update(key=o.json_hash(st),result=str(args.output/'audit_segments'/f'{uid}_suffix.pt'))
        suffix = suffix_one(st)
        p,state = o.restore_scenario(spec)
        own = make_own(p,reg,uid,1,settings(args),spec['search_seeds'][1])
        other = private_solver(p,reg['L'],'Search Only',0,settings(args,0),spec['search_seeds'][0])
        trace = []
        state,stats,_,_ = simulate(p,state,own,other,1,trace=trace)
        if science(suffix['metrics']) != science(metrics(p,state,stats)) or \
                o.json_hash(trace) != o.json_hash(prefix['trajectory']+suffix['trajectory']) or \
                suffix['own_rng'] != own.get_rng_state(p) or suffix['opponent_rng'] != other_rng(other,p):
            raise ValueError('Segmented trajectory or RNG failed parity')
        segmented.append(dict(uid=uid,exact=True,branched=prefix['branch']))
    return lock_json(path,dict(passed=True,original_parity=probes,segmented_parity=segmented,
        private_rng_only=True,unchanged_historical_sources=True))


def read_recommended(args):
    rec = locked_read(args.output/'recommended.json')
    if rec['experiment_sha256'] != o.sha(args.output/'experiment.json'):
        raise ValueError('Export protocol changed')
    for model in rec['registry'].values():
        if o.model_fingerprint(model['models']) != model['model_hash']:
            raise ValueError('Exported weights changed')
    if rec['implementation'] != fingerprints():
        raise ValueError('Exported implementation changed')
    for path,digest in rec['runtime_fingerprints'].items():
        if o.sha(path) != digest:
            raise ValueError('Exported adapter or private runtime changed')
    return rec


def load_recommended_solver(args,problem,role,seed):
    rec = read_recommended(args)
    return make_own(problem,rec['registry'],rec['uid'],role,settings(args,role),seed)


def replay(args,banks):
    rec = read_recommended(args)
    probes = []
    for role in (0,1):
        spec = banks['warmup'][0]
        p,state = o.restore_scenario(spec)
        for after in (False,True):
            current = state.copy()
            if after:
                current[[p.active_idxs[0],p.active_idxs[2]],0] = 0.
            expected = make_own(p,registry(args),rec['uid'],role,settings(args,role),spec['search_seeds'][role])
            actual = load_recommended_solver(args,p,role,spec['search_seeds'][role])
            if not np.array_equal(expected.policy(p,current),actual.policy(p,current)):
                raise ValueError('Export action replay mismatch')
            probes.append(dict(role=role,after_first=after,exact=True))
    # Complete trajectories exercise a real handoff and its continuous stream.
    item = requests(args,[rec['uid']],banks['warmup'],opponents=('Search Only',))[0]
    task = copy.deepcopy(item['task'])
    task['registry'] = rec['registry']
    if science(evaluate_one(task)) != science(evaluate_one(item['task'])):
        raise ValueError('Export full episode mismatch')
    return lock_json(args.output/'replay.json',dict(passed=True,uid=rec['uid'],probes=probes,full_episode_exact=True))


def optional(args,name):
    path = args.output/(name+'.json')
    return locked_read(path) if path.with_suffix('.lock.json').exists() else None


def finish(args,banks):
    verify(args)
    diagnostic = locked_read(args.output/'diagnostic_result.json')
    validation,confirmation,regression,measurements = [optional(args,n) for n in
        ('validation_result','confirmation_result','guard_result','timing_result')]
    for prerequisite,downstream,name in ((diagnostic,validation,'validation'),(validation,confirmation,'confirmation'),
                                        (confirmation,regression,'regression'),(regression,measurements,'timing')):
        if prerequisite and prerequisite['passed'] and downstream is None:
            raise ValueError(name+' remains incomplete')
        if downstream is not None and (not prerequisite or not prerequisite['passed']):
            raise ValueError(name+' exists without passing prerequisite')
    accepted = bool(regression and regression['passed'] and measurements and not args.smoke)
    uid = 'H' if accepted else 'O'
    exported = {}
    for name in ('O','S','L'):
        dest = args.output/'recommended/models'/name
        dest.mkdir(parents=True,exist_ok=True)
        origin = registry(args)[name]
        for filename in o.MODEL_NAMES:
            if not (dest/filename).exists():
                shutil.copyfile(Path(origin['models'])/filename,dest/filename)
        if o.model_fingerprint(dest) != origin['model_hash']:
            raise ValueError('Export copy changed')
        exported[name] = dict(models=str(dest),model_hash=origin['model_hash'])
    runtime_fingerprints = {}
    runtime_sources = [Path(__file__),o.CODE/'solvers/phase_puct.py',o.CODE/'build_phase_search.py',
        o.CODE/'cpp/build_phase/build_manifest.json',*list((o.CODE/'cpp/build_phase').glob('*.so'))]
    for origin in runtime_sources:
        dest = args.output/'recommended/runtime'/origin.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            shutil.copyfile(origin,dest)
        if o.sha(dest) != o.sha(origin):
            raise ValueError('Exported runtime copy changed')
        runtime_fingerprints[str(dest)] = o.sha(dest)
    rec = dict(schema='l45_phase_handoff_recommendation_v1',uid=uid,historical_uid='old0:novalue' if uid == 'O' else None,
        registry=exported,pursuer_phases=list(phase_pair(uid,1)),evader_phases=['O','O'],
        switch=config(args)['switch'],pursuer_settings=settings(args),evader_settings=settings(args,0),
        loader='run_l45_phase_handoff.load_recommended_solver',adapter='solvers.phase_puct.PhaseSolver',
        implementation=fingerprints(),runtime_fingerprints=runtime_fingerprints,experiment_sha256=o.sha(args.output/'experiment.json'),
        quality_gate_passed=accepted,previous_confirmed_recommendation_preserved=not accepted,smoke=args.smoke)
    lock_json(args.output/'recommended.json',rec)
    reason = 'adopted' if accepted else 'diagnostic_no_gain' if not diagnostic['passed'] else \
        'validation_no_gain' if not validation['passed'] else 'confirmation_failed' if not confirmation['passed'] else 'guard_failed'
    result = dict(reason=reason,recommended=uid,quality_gate_passed=accepted,smoke=args.smoke,
        training_episodes=0,teacher_queries=0,optimizer_updates=0,diagnostic=diagnostic,validation=validation,
        confirmation=confirmation,guard=regression,timing=measurements,replay=replay(args,banks),
        actual_stages=['prepare','audit','diagnose']+(['validate'] if validation else [])+
        (['confirm'] if confirmation else [])+(['guard'] if regression else [])+(['timing'] if measurements else [])+['report','replay'],
        continue_formal=False)
    lock_json(args.output/'results.json',result)
    report(args,banks,result)
    return result


def report(args,banks,result):
    d = result['diagnostic']
    lines = ['# 首捕前后交叉接管实验','',f"结束原因：{result['reason']}；推荐：{result['recommended']}；smoke：{args.smoke}。",
        '新增训练、教师查询和优化器更新均为0。O=推荐PPO第2048局；S=继承PPO的硬KL监督新增512局；L=原l45。',
        '追捕1000次/深度100/βπ=.5/βv=0；原对手、环境、奖励和观测保持。H仅在真实首捕后下一次搜索切换O→S。','',
        '|组合|合法双捕|零捕|首捕后超时|自身非法|','|---|---:|---:|---:|---:|']
    for combo,r in d['summary'].items():
        lines.append(f"|{combo}|{r['successes']}/{r['games']}={r['success_rate']:.2%}|{r['zero_capture']}|{r['one_capture_timeout']}|{r['pursuer_illegal']}|")
    for name,r in d['pairs'].items():
        lines.append(f"\n{name}：挽回{r['gained']}、丢失{r['lost']}、净增{r['net']}；全场景增益{r['delta']:+.2%}；实际可分支{r['branched_games']}局，条件增益{r['conditional_delta']}。")
    lines += ['', '历史诊断复用全部指定场景，不能用于独立效果确认。跨前缀结果不归因于单一几何因素；两后缀失败不证明局面无解。']
    val = result['validation']
    if val:
        lines += ['', '|模型|全新验证四障碍合法双捕|','|---|---:|']
        for uid,r in val['aggregates'].items():
            lines.append(f"|{uid}|{r['success_rate']:.2%}|")
        for ref,r in val['comparisons'].items():
            lines.append(f"H对{ref}：{r['four']['delta']:+.2%}；通过全部验证限制：{r['passed']}。")
    else:
        lines += ['', '诊断门槛未通过，未执行新场景验证；H相对两个参考的新场景增益未知。']
    lines += ['', '独立确认通过：'+str(bool(result['confirmation'] and result['confirmation']['passed']))+
        '；逃逸回归通过：'+str(bool(result['guard'] and result['guard']['passed']))+'。',
        '耗时变化见timing_result.json。' if result['timing'] else '未执行单进程计时，耗时变化未知。',
        '所有阶段、两项配对bootstrap、逐场景分支、候选锁定和推荐来源见results.json及各阶段记录。',
        '仅说明固定检查点组合，不宣称跨种子稳定或PPO的独立贡献。本轮结束，不自动延长或改变方法。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['pipeline','prepare','audit','diagnose','validate','confirm','guard','timing','report','watch','replay'],default='pipeline')
    parser.add_argument('--workers',type=int,default=80)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--once',action='store_true')
    args = parser.parse_args(argv)
    args.output = (args.output or (SMOKE_OUTPUT if args.smoke else DEFAULT_OUTPUT)).resolve()
    if args.workers < 1:
        parser.error('--workers must be positive')
    return args


def watch(args):
    while True:
        row = o.read_json(args.output/'status.json')
        parent = Path(f"/proc/{row['pid']}")
        workers = []
        if parent.exists():
            try:
                workers = [pid for pid in (parent/f"task/{row['pid']}/children").read_text().split()
                    if b'spawn_main' in Path(f'/proc/{pid}/cmdline').read_bytes()]
            except FileNotFoundError:
                pass
        print(dict(row,parent_alive=parent.exists(),live_workers=len(workers),
            committed_prefixes=len(list((args.output/'prefix_cache').glob('*.lock.json'))),
            committed_suffixes=len(list((args.output/'suffix_cache').glob('*.lock.json'))),
            committed_episodes=len(list((args.output/'cache_checksums').glob('*.lock.json')))),flush=True)
        if args.once:
            return
        time.sleep(5)


def main(argv=None):
    args = parse_args(argv)
    if args.stage == 'watch':
        return watch(args)
    args.output.mkdir(parents=True,exist_ok=True)
    handle = (args.output/'run.lock').open('a')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1)
    started,phase_started,last_phase,initial = time.time(),time.time(),None,0
    current,mutex = {},threading.Lock()
    def status(phase,**fields):
        nonlocal phase_started,last_phase,initial
        with mutex:
            if phase != last_phase:
                phase_started,last_phase,initial = time.time(),phase,fields.get('completed',0)
            row = dict(phase=phase,pid=os.getpid(),workers=args.workers,updated_unix=time.time(),**fields)
            if 'completed' in fields:
                rate = (fields['completed']-initial)/max(time.time()-phase_started,1e-9)
                row.update(rate=rate,eta_seconds=(fields['total']-fields['completed'])/rate if rate else None)
            current.clear();current.update(row)
            o.write_json(args.output/'status.json',row)
        print(row,flush=True)
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(5):
            with mutex:
                if current:
                    current['updated_unix'] = time.time()
                    o.write_json(args.output/'status.json',current)
    thread = threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        status('prepare')
        banks = prepare(args)
        if args.stage == 'prepare':
            status('prepared');return
        if args.stage in ('report','replay','timing'):
            value = finish(args,banks) if args.stage == 'report' else replay(args,banks) if args.stage == 'replay' else timing(args,banks,status)
            status('complete' if args.stage == 'report' else 'stage_complete',stage=args.stage)
            return value
        status('audit')
        audit(args,banks,status)
        if args.stage == 'audit':
            status('audited');return
        with mp.get_context('spawn').Pool(args.workers) as pool:
            if args.stage in ('pipeline','diagnose'):
                diagnostic = diagnose(args,banks,pool,status)
            else:
                diagnostic = locked_read(args.output/'diagnostic_result.json')
            validation = optional(args,'validation_result')
            confirmation = optional(args,'confirmation_result')
            regression = optional(args,'guard_result')
            if args.stage in ('pipeline','validate') and diagnostic['passed']:
                validation = validate(args,banks,pool,status)
            elif args.stage == 'validate':
                raise ValueError('Diagnostic failed')
            if args.stage in ('pipeline','confirm') and validation and validation['passed']:
                confirmation = confirm(args,banks,pool,status)
            elif args.stage == 'confirm':
                raise ValueError('Validation failed or incomplete')
            if args.stage in ('pipeline','guard') and confirmation and confirmation['passed']:
                regression = guard(args,banks,pool,status)
            elif args.stage == 'guard':
                raise ValueError('Confirmation failed or incomplete')
        if args.stage == 'pipeline':
            if regression and regression['passed']:
                timing(args,banks,status)
            value = finish(args,banks)
            status('complete',recommended=value['recommended'],quality_gate_passed=value['quality_gate_passed'],
                reason=value['reason'],elapsed_seconds=time.time()-started)
            return value
        status('stage_complete',stage=args.stage)
    except BaseException as error:
        status('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error))
        raise
    finally:
        stop.set();thread.join();handle.close()


if __name__ == '__main__':
    main()
