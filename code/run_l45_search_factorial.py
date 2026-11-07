"""Frozen 2x2 pursuit weights/search experiment; D is the sole candidate."""
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
import random
import shutil
import threading
import time
import types
import numpy as np
import torch
import run_l45_synchronous_search as frozen

torch.set_num_threads(1)
base, research = frozen.base, frozen.research
o, s, v, f = frozen.o, frozen.s, frozen.v, frozen.f
ROOT, CODE, SOURCE = o.ROOT, o.CODE, frozen.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT/'current/l45_search_factorial_20261024'
SMOKE_OUTPUT = ROOT/'current/l45_search_factorial_smoke_20261024'
PROTOCOL = CODE/'L45_SEARCH_FACTORIAL.md'
UIDS, REFS = ('A', 'B', 'C', 'D'), ('A', 'B', 'C')
SPLITS, SCIENCE = frozen.SPLITS, frozen.SCIENCE
lock_json, locked_read = frozen.lock_json, frozen.locked_read
DeploymentPUCT = frozen.DeploymentPUCT


def config(args):
    return dict(schema=1, smoke=args.smoke, training_episodes=0, teacher_queries=0,
        optimizer_updates=0, source_episode=2048, candidate='D',
        factors=dict(A=['L','legacy'], B=['O','legacy'], C=['L','sync'], D=['O','sync']),
        rule='whole_episode; unchanged frozen simultaneous solver; complete_O_evader_for_BCD',
        budget=3 if args.smoke else 1000, search_only_budget=3 if args.smoke else 3000,
        depth=100, pursuer_betas=[.5,0.], evader_betas=[.5,.1], completion_bonus=0.,
        scenario_namespace=5202026102401 if args.smoke else 5202026102402,
        splits={name:[layout,1 if args.smoke else n] for name,(layout,n) in SPLITS.items()},
        bootstrap_seed=52020261024, bootstrap_draws=20000,
        comparisons=[dict(id='D-'+ref, candidate='D', reference=ref) for ref in REFS],
        request_caps=dict(validation=24 if args.smoke else 1800, confirmation=6000,
            evader_regression=2400, serial_timing=360),
        minimum_gain=.03, illegal_margin=.02, regression_margin=.03,
        no_automatic_expansion=True, records='full_states_actions_roots_first_capture_rng')


def fingerprints():
    paths = [Path(__file__), PROTOCOL, CODE/'tests/test_l45_search_factorial.py']
    return dict(files={str(p):o.sha(p) for p in paths}, inherited=frozen.fingerprints())


# Bind generic functions without changing any historical module's globals.
for _name in ('settings','registry','verify','initial_stats','advance','metrics','science','other_rng','optional','watch'):
    _method = getattr(base, _name)
    globals()[_name] = types.FunctionType(_method.__code__, globals(), _name,
                                        _method.__defaults__, _method.__closure__)


def prepare(args):
    out = args.output
    if out == ROOT or out == SOURCE or SOURCE in out.parents:
        raise ValueError('Output must be isolated')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'experiment.lock.json').exists():
        verify(args)
        return locked_read(out/'scenarios.json')
    frozen.verify(frozen.parse_args([]))
    prior = locked_read(SOURCE/'results.json')
    if prior['recommended'] != 'O' or prior['reason'] != 'validation_no_gain':
        raise ValueError('Expected completed frozen synchronous validation')
    source = {str(SOURCE/name):o.sha(SOURCE/name) for name in
              ('experiment.json','results.json','replay.json','validation_result.json')}
    for name in ('O','L'):
        dest = out/'models'/name
        dest.mkdir(parents=True, exist_ok=True)
        for filename in o.MODEL_NAMES:
            origin = SOURCE/'models'/name/filename
            source[str(origin)] = o.sha(origin)
            shutil.copyfile(origin, dest/filename)
    reg = registry(args)
    for filename in (o.MODEL_NAMES[0], o.MODEL_NAMES[1], o.MODEL_NAMES[-1]):
        if reg['O']['model_hash'][filename] != reg['L']['model_hash'][filename]:
            raise ValueError('Frozen evader actors or original value differ')
    history = s.history_files(args)
    used = s.history_scenes(history)
    old_count, banks = len(used), {}
    for phase,(name,(layout,count)) in enumerate(config(args)['splits'].items(), 2000):
        cases, index = [], 0
        while len(cases) < count:
            spec = o.make_scenario(o.scenario_seed(phase, config(args)['scenario_namespace'], index), layout)
            index += 1
            identity = s.previous.physical_hash(spec)
            if identity not in used:
                used.add(identity)
                cases.append(spec)
        banks[name] = cases
    record = dict(config=config(args), implementation=fingerprints(), source=source,
        history={str(p):o.sha(p) for p in history}, models=reg)
    lock_json(out/'scenarios.json', banks)
    lock_json(out/'scenario_independence.json', dict(previous_physical_scenes=old_count,
        new_scenes=len(used)-old_count, overlap=0, history_reuse=False, all_phases_frozen=True,
        scope='all recoverable l45 scenes, archives, smoke, unused banks and actual rollout specs'))
    lock_json(out/'experiment.json', record)
    lock_json(out/'signature.json', dict(sha256=o.json_hash(dict(record=record,
        scenarios=o.sha(out/'scenarios.json')))))
    shutil.copyfile(PROTOCOL, out/'PLAN.md')
    verify(args)
    return banks


def phase_pair(uid, role):
    if uid not in UIDS or role not in (0,1):
        raise ValueError('Unknown configuration or role')
    name = 'L' if uid == 'A' or (uid == 'C' and role == 1) else 'O'
    return name, name


def effective(uid, role, reg):
    name = phase_pair(uid, role)[0]
    return dict(model=name, weights=reg[name]['model_hash'],
        algorithm='synchronous_DUCT' if role == 1 and uid in ('C','D') else 'legacy_PUCT')


def make_own(problem, reg, uid, role, requested, seed, solver_type=DeploymentPUCT):
    model = reg[phase_pair(uid, role)[0]]
    if o.model_fingerprint(model['models']) != model['model_hash']:
        raise ValueError('Worker weights changed')
    paths, value_path = research.h.oracle_paths(str(ROOT), iteration=45, model_dir=model['models'])
    with contextlib.redirect_stdout(io.StringIO()):
        policies, value = research.h.get_oracles(problem, value_oracle_name='deterministic',
            value_oracle_path=value_path, policy_oracle_name='gaussian', policy_oracle_paths=paths)
    raw = solver_type(policies, value, requested['search_budget'], requested['beta_policy'],
        requested['beta_value'], role == 1 and uid in ('C','D'), requested['search_depth'])
    raw.solver_wrapper.set_seed(int(seed))
    return o.RoleMaskedSolver(raw, o.TEAMS[role])


def requests(args, uids, scenes, role=1, opponents=o.OPPONENTS, timing='parallel'):
    reg, items = registry(args), []
    for uid in uids:
        behavior = effective(uid, role, reg)
        for spec in scenes:
            for opponent in opponents:
                identity = dict(protocol=o.sha(args.output/'experiment.json'), policy_id=o.json_hash(behavior),
                    episode_hash=spec['episode_hash'], role=role, opponent=opponent, timing_context=timing,
                    own_settings=settings(args,role), opponent_settings=settings(args,0), mode='search')
                key = o.json_hash(identity)
                task = dict(**identity, key=key, uid=uid, spec=spec, registry=reg, effective=behavior)
                items.append(dict(key=key, uid=uid, task=task))
    return items


def seed_task(task):
    seed = int(task['spec']['episode_hash'][:8],16)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_pair(task):
    p, state = o.restore_scenario(task['spec'])
    p.clear_closed_loop_cache()
    role = task['role']
    own = make_own(p, task['registry'], task['uid'], role, task['own_settings'],
                   task['spec']['search_seeds'][role])
    other = base.private_solver(p, task['registry']['L'], task['opponent'], 1-role,
                               task['opponent_settings'], task['spec']['search_seeds'][1-role])
    return p, state, own, other


def snapshot(p, state, stats, own, other):
    return dict(state=state.copy(), stats=copy.deepcopy(stats), time=float(state[p.time_idx,0]),
        remaining_time=float(p.tf-state[p.time_idx,0]), own_rng=own.solver.get_rng_state(p),
        opponent_rng=other_rng(other,p), python_rng=o.rng_state())


def restore_snapshot(task, snap):
    p, _, own, other = make_pair(task)
    own.solver.set_rng_state(p, snap['own_rng'])
    if snap['opponent_rng'] is not None:
        other.solver.set_rng_state(p, snap['opponent_rng'])
    o.restore_rng(snap['python_rng'])
    state = snap['state'].copy()
    if state.shape != (53,1) or float(state[p.time_idx,0]) != snap['time'] or \
            float(p.tf-state[p.time_idx,0]) != snap['remaining_time']:
        raise ValueError('Snapshot state or horizon changed')
    p.clear_closed_loop_cache()
    return p, state, own, other, copy.deepcopy(snap['stats'])


def simulate_record(p, state, own, other, role, stats=None, record=True):
    stats = initial_stats() if stats is None else copy.deepcopy(stats)
    team = o.TeamSolver(other if role == 1 else own, own if role == 1 else other)
    trace, roots, decisions = [], [], []
    first = None
    start_snapshot = snapshot(p,state,stats,own,other) if record else None
    deployment_ms = 0.
    while not p.is_terminal(state):
        phase = int(stats['capture_count'] > 0)
        before_cost = dict(own.solver.cost)
        action = team.policy(p,state)
        decision = [team.last_pursuer_latency_ms, team.last_evader_latency_ms, team.last_joint_latency_ms]
        decisions.append(decision)
        advance_clock = time.perf_counter()
        nxt = advance(p,state,action,stats)
        deployment_ms += decision[2] + (time.perf_counter()-advance_clock)*1000.
        if record:
            table = np.asarray(own.solver.last_result.root_audit).copy()
            if not np.isfinite(action).all() or not np.isfinite(table).all():
                raise ValueError('Nonfinite action or root statistics')
            trace.append(dict(state=state.copy(), action=action.copy(), next_state=nxt.copy()))
            roots.append(dict(phase=phase, table=table,
                cost={k:own.solver.cost[k]-before_cost[k] for k in before_cost}))
        state = nxt
        if record and first is None and phase == 0 and stats['capture_count'] > 0:
            first = snapshot(p,state,stats,own,other)
            first.update(next_decision=len(trace), branch_available=bool(stats['capture_count']==1 and
                not p.is_terminal(state) and not(stats['pursuer_illegal'] or stats['evader_illegal'])))
        if stats['steps'] > int(p.tf/p.dt):
            raise RuntimeError('Exceeded remaining physical horizon')
    detail = dict(initial=start_snapshot, first_capture=first, trajectory=trace, roots=roots,
        final=snapshot(p,state,stats,own,other) if record else None)
    return metrics(p,state,stats), detail, decisions, deployment_ms


def trajectory_digest(trace):
    import hashlib
    digest = hashlib.sha256()
    for row in trace:
        for key in ('state','action','next_state'):
            array = np.ascontiguousarray(row[key],dtype=np.float64)
            digest.update(array.tobytes())
    return digest.hexdigest()


def evaluate_one(task):
    seed_task(task)
    p,state,own,other = make_pair(task)
    started = time.perf_counter()
    row,detail,decisions,deployment_ms = simulate_record(p,state,own,other,task['role'])
    row.update({k:task[k] for k in ('key','policy_id','role','opponent','timing_context','mode')})
    row.update(uid=task['uid'], episode_hash=task['spec']['episode_hash'], layout=task['spec']['layout'],
        search_seeds=task['spec']['search_seeds'], actual_own=dict(task['own_settings'],effective=task['effective']),
        actual_opponent=other.actual_settings, own_cost=dict(own.solver.cost), own_mcts_calls=own.solver.calls,
        switches=0, intervention_events=[], episode_wall_ms=(time.perf_counter()-started)*1000.,
        deployment_episode_ms=deployment_ms, trajectory_sha256=trajectory_digest(detail['trajectory']))
    if task['timing_context'] in ('warmup','single_process'):
        row['decision_latency_ms'] = decisions
    row['_record'] = dict(key=task['key'], scenario=task['spec'], effective=task['effective'],
                          role=task['role'], opponent=task['opponent'], **detail)
    return row


def commit_row(args, row):
    detail = row.pop('_record')
    artifact = args.output/'episodes'/(row['key']+'.pt')
    # A row checksum is the commit point. Uncommitted artifacts can be replaced
    # on replay, including after a crash between artifact and metric writes.
    o.save_torch(artifact,detail)
    row['episode_record'] = dict(path=str(artifact),sha256=o.sha(artifact))
    row['content_sha256'] = o.json_hash(row)
    path = args.output/'cache'/(row['key']+'.json')
    o.write_json(path,row)
    lock_json(args.output/'cache_checksums'/(row['key']+'.json'),dict(sha256=o.sha(path)))


def read_cached(args,item):
    # Reuse the frozen identity, seed, actual opponent and role checks.
    row = frozen.read_cached(args,item)
    expected = args.output/'episodes'/(item['key']+'.pt')
    artifact = row['episode_record']
    if artifact['path'] != str(expected) or o.sha(expected) != artifact['sha256']:
        raise ValueError('Committed trajectory or snapshot changed')
    return row


def read_episode(args,item):
    row = read_cached(args,item)
    detail = o.load_torch(row['episode_record']['path'])
    if detail['key'] != item['key'] or detail['scenario'] != item['task']['spec'] or \
            detail['effective'] != item['task']['effective'] or detail['role'] != item['task']['role'] or \
            detail['opponent'] != item['task']['opponent'] or \
            trajectory_digest(detail['trajectory']) != row['trajectory_sha256'] or \
            len(detail['trajectory']) != row['own_mcts_calls']:
        raise ValueError('Episode record identity or trajectory changed')
    return row,detail


def run_evaluation(args,name,items,pool,status):
    if len(items) > config(args)['request_caps'][name]:
        raise ValueError('Stage request budget exceeded')
    manifest = [{k:val for k,val in item.items() if k != 'task'} for item in items]
    lock_json(args.output/name/'manifest.json',manifest)
    unique = {item['key']:item for item in items}
    missing = [item['task'] for key,item in unique.items()
               if not (args.output/'cache_checksums'/(key+'.lock.json')).exists()]
    for key,item in unique.items():
        if (args.output/'cache_checksums'/(key+'.lock.json')).exists():
            read_cached(args,item)
    done = len(unique)-len(missing)
    status(name,completed=done,total=len(unique),label_requests=len(items),reused=done,
           actual_execution_requests=len(missing))
    if pool and missing:
        pids = sorted({p.pid for p in pool._pool if p.is_alive()})
        evidence = dict(stage=name, pids=pids, workers=len(pids),
            threads={str(pid):len(list(Path('/proc',str(pid),'task').glob('*'))) for pid in pids})
        o.write_json(args.output/'worker_snapshot.json',evidence)
        lock_json(args.output/name/'worker_snapshots'/(str(os.getpid())+'.json'),evidence)
    iterator = pool.imap_unordered(evaluate_one,missing,chunksize=1) if pool else map(evaluate_one,missing)
    started = time.perf_counter()
    for row in iterator:
        commit_row(args,row)
        done += 1
        if done % 10 == 0 or done == len(unique):
            status(name,completed=done,total=len(unique),label_requests=len(items),
                   throughput_per_second=(done-(len(unique)-len(missing)))/max(.001,time.perf_counter()-started))
    rows = [read_cached(args,item) for item in items]
    lock_json(args.output/name/'summary.json', [dict(uid=uid,layout=layout,
        **v.aggregate(rows,uid,layout,items[0]['task']['role']))
        for uid in UIDS for layout in sorted({r['layout'] for r in rows})])
    return rows


def paired_index(rows,layout='four'):
    parts = {uid:v.indexed_rows(rows,uid,layout,1) for uid in UIDS}
    keys = sorted(parts['A'])
    if not keys or any(set(part) != set(keys) for part in parts.values()):
        raise ValueError('Missing factorial pairs')
    scenes = sorted({key[0] for key in keys})
    for scene in scenes:
        if {op for sc,op in keys if sc == scene} != set(o.OPPONENTS):
            raise ValueError('Missing opponent stratum')
    if any(parts[u][k]['search_seeds'] != parts['A'][k]['search_seeds'] for u in UIDS for k in keys):
        raise ValueError('Statistical seed mismatch')
    return parts,keys,scenes


def bootstrap_contrasts(rows,contrasts,draws=20000):
    parts,keys,scenes = paired_index(rows)
    differences, output = [], []
    for name,coefficients in contrasts.items():
        delta = {k:sum(coefficient*parts[uid][k]['legal_double_capture']
                       for uid,coefficient in coefficients.items()) for k in keys}
        differences.append([np.mean([delta[(scene,op)] for op in o.OPPONENTS]) for scene in scenes])
        output.append(dict(id=name,delta=float(np.mean(list(delta.values()))),n_scenes=len(scenes),
            n_games=len(keys),gained=sum(x>0 for x in delta.values()),lost=sum(x<0 for x in delta.values())))
    differences = np.asarray(differences)
    observed = differences.mean(1)
    rng = np.random.default_rng(52020261024)
    samples = np.empty((draws,len(contrasts)))
    for start in range(0,draws,256):
        stop = min(start+256,draws)
        indices = rng.integers(len(scenes),size=(stop-start,len(scenes)))
        samples[start:stop] = differences[:,indices].mean(-1).T
    correction = max(0.,float(np.quantile(np.max(samples-observed,axis=1),.95)))
    for i,row in enumerate(output):
        row.update(ci95=np.quantile(samples[:,i],[.025,.975]).tolist(),
                   simultaneous_lower95=float(observed[i]-correction))
    return dict(comparisons=output,draws=draws,bootstrap_seed=52020261024,correction=correction,
        method='physical_scene_paired_three_opponents; max_centered_joint_one_sided_95')


def statistics(rows,draws=20000):
    result = bootstrap_contrasts(rows,{'D-'+ref:{'D':1,ref:-1} for ref in REFS},draws)
    for row,ref in zip(result['comparisons'],REFS):
        row.update(candidate='D',reference=ref)
    return result


def factorial_statistics(rows,draws=20000):
    result = bootstrap_contrasts(rows,{'weights_legacy':{'B':1,'A':-1},
        'weights_sync':{'D':1,'C':-1},'search_L':{'C':1,'A':-1},
        'search_O':{'D':1,'B':-1},'interaction':{'D':1,'B':-1,'C':-1,'A':1}},draws)
    # These ordinary intervals are exploratory and never affect adoption.
    for row in result['comparisons']:
        row.pop('simultaneous_lower95')
    result.update(exploratory_only=True,interaction='(D-B)-(C-A)')
    result.pop('correction')
    return result


def comparison(rows,reference,confidence=None):
    paired_index(rows,'four')
    paired_index(rows,'none')
    four = s.point_comparison(rows,'D',reference,'four')
    none = s.point_comparison(rows,'D',reference,'none')
    checks = s.performance_checks(four,none,gain=True)
    checks['no_obstacle_own_illegal'] = none['own_illegal_delta'] <= .02+1e-12
    if confidence is not None:
        checks['simultaneous_lower_positive'] = confidence['simultaneous_lower95'] > 0
    return dict(four=four,none=none,checks=checks,passed=all(checks.values()))


def candidate_record(args):
    return dict(candidate='D',config=config(args),models=registry(args),implementation=fingerprints(),
                validation_sha256=o.sha(args.output/'validation_result.json'))


def validate(args,banks,pool,status):
    rows = run_evaluation(args,'validation',requests(args,UIDS,banks['validation_four']+banks['validation_none']),pool,status)
    comparisons = {ref:comparison(rows,ref) for ref in REFS}
    result = lock_json(args.output/'validation_result.json',dict(
        passed=all(x['passed'] for x in comparisons.values()) and not args.smoke,
        comparisons=comparisons,aggregates={uid:v.aggregate(rows,uid) for uid in UIDS},smoke=args.smoke))
    if result['passed']:
        lock_json(args.output/'candidate_lock.json',candidate_record(args))
    return result


def check_candidate(args):
    if args.smoke or not locked_read(args.output/'validation_result.json')['passed']:
        raise ValueError('Validation failed; confirmation forbidden')
    if locked_read(args.output/'candidate_lock.json') != candidate_record(args):
        raise ValueError('Candidate lock changed')


def confirm(args,banks,pool,status):
    check_candidate(args)
    rows = run_evaluation(args,'confirmation',requests(args,UIDS,banks['confirm_four']+banks['confirm_none']),pool,status)
    stats = statistics(rows)
    comparisons = {x['reference']:comparison(rows,x['reference'],x) for x in stats['comparisons']}
    return lock_json(args.output/'confirmation_result.json',dict(passed=all(x['passed'] for x in comparisons.values()),
        statistics=stats,comparisons=comparisons))


def guard(args,banks,pool,status):
    check_candidate(args)
    if not locked_read(args.output/'confirmation_result.json')['passed']:
        raise ValueError('Confirmation failed; regression forbidden')
    req = requests(args,UIDS,banks['guard_four']+banks['guard_none'],role=0)
    rows = run_evaluation(args,'evader_regression',req,pool,status)
    comparisons = {ref:{layout:s.point_comparison(rows,'D',ref,layout,role=0)
        for layout in ('four','none')} for ref in REFS}
    passed = all(x['delta'] >= -.03-1e-12 and x['own_illegal_delta'] <= .02+1e-12
                 for part in comparisons.values() for x in part.values())
    return lock_json(args.output/'guard_result.json',dict(passed=passed,comparisons=comparisons,
        evader_complete_O=True,logical_requests=len(req),unique_requests=len({i['key'] for i in req})))


def timing(args,banks,status):
    if not locked_read(args.output/'guard_result.json')['passed']:
        raise ValueError('Regression failed; timing forbidden')
    if mp.active_children():
        raise ValueError('Exit parallel pool before timing')
    if (args.output/'timing_result.lock.json').exists():
        return locked_read(args.output/'timing_result.json')
    for item in requests(args,UIDS,banks['warmup'],timing='warmup'):
        evaluate_one(item['task'])
    req = []
    for i,spec in enumerate(banks['timing_four']):
        order = UIDS[i%4:]+UIDS[:i%4]
        for op in o.OPPONENTS:
            req += requests(args,order,[spec],opponents=(op,),timing='single_process')
    rows = run_evaluation(args,'serial_timing',req,None,status)
    result = {}
    for uid in UIDS:
        part = [r for r in rows if r['uid']==uid]
        decisions = np.concatenate([np.asarray(r['decision_latency_ms'])[:,0] for r in part])
        episodes = [r['deployment_episode_ms'] for r in part]
        result[uid] = dict(pursuer_mean_ms=float(decisions.mean()),
            pursuer_p95_ms=float(np.percentile(decisions,95)),episode_mean_ms=float(np.mean(episodes)),
            episode_p95_ms=float(np.percentile(episodes,95)),episodes=len(part),
            own_cost={k:sum(r['own_cost'][k] for r in part) for k in part[0]['own_cost']})
    for row in result.values():
        row['ratio_to_B'] = row['pursuer_mean_ms']/result['B']['pursuer_mean_ms']
    return lock_json(args.output/'timing_result.json',result)


def audit_one(task):
    row = evaluate_one(task)
    detail = row.pop('_record')
    seed_task(task)
    p,state = o.restore_scenario(task['spec'])
    role,reg,uid = task['role'],task['registry'],task['uid']
    if uid in ('A','B'):
        own = f.make_solver(p,reg[phase_pair(uid,role)[0]]['models'],'Search + Learning',role,
                           task['own_settings'],task['spec']['search_seeds'][role])
        other = f.make_solver(p,reg['L']['models'],task['opponent'],1-role,
                             task['opponent_settings'],task['spec']['search_seeds'][1-role])
    else:
        if uid == 'C':
            own = frozen.make_own(p,reg,'C',role,task['own_settings'],task['spec']['search_seeds'][role])
        else:
            cfg = dict(budget=task['own_settings']['search_budget'],
                opponent_budget=task['own_settings']['search_only_budget'],depth=100)
            raw = research.make_solver(p,reg['O'],cfg,role==1,role,task['spec']['search_seeds'][role])
            own = o.RoleMaskedSolver(raw,o.TEAMS[role])
        other = base.private_solver(p,reg['L'],task['opponent'],1-role,
                                   task['opponent_settings'],task['spec']['search_seeds'][1-role])
    trace = []
    end,stats,_,_ = frozen.simulate(p,state,own,other,role,trace=trace)
    if science(metrics(p,end,stats)) != science(row) or trajectory_digest(trace) != row['trajectory_sha256']:
        raise ValueError('Frozen implementation full-episode parity failed')
    # Restoring does not reset the physical clock, horizon or either RNG stream.
    probes = [('initial',detail['initial'])]
    if detail['first_capture'] and detail['first_capture']['branch_available']:
        probes.append(('first_capture',detail['first_capture']))
    for name,snap in probes:
        p,state,own,other,stats = restore_snapshot(task,snap)
        rng_before = snapshot(p,state,stats,own,other)
        own.solver.set_rng_state(p,own.solver.get_rng_state(p))
        if not s.equal_tree(rng_before,snapshot(p,state,stats,own,other)):
            raise ValueError('RNG roundtrip consumed or changed state')
        resumed,resumed_detail,_,_ = simulate_record(p,state,own,other,role,stats)
        offset = 0 if name=='initial' else snap['next_decision']
        if science(resumed) != science(row) or \
           trajectory_digest(resumed_detail['trajectory']) != trajectory_digest(detail['trajectory'][offset:]) or \
           not s.equal_tree(resumed_detail['final']['own_rng'],detail['final']['own_rng']) or \
           not s.equal_tree(resumed_detail['final']['opponent_rng'],detail['final']['opponent_rng']) or \
           not s.equal_tree(resumed_detail['final']['python_rng'],detail['final']['python_rng']):
            raise ValueError('Snapshot continuation or both ending RNG streams changed')
    return dict(uid=uid,layout=task['spec']['layout'],role=role,opponent=task['opponent'],
                full_episode_exact=True,snapshot_probes=[name for name,_ in probes])


def audit(args,banks,status):
    if (args.output/'audit_result.lock.json').exists():
        return locked_read(args.output/'audit_result.json')
    tasks = [item['task'] for role in (0,1) for layout in ('four','none')
             for item in requests(args,UIDS,banks['audit_'+layout],role)]
    probes = []
    with mp.get_context('spawn').Pool(min(args.workers,len(tasks))) as pool:
        for row in pool.imap_unordered(audit_one,tasks,chunksize=1):
            probes.append(row)
            status('audit',completed=len(probes),total=len(tasks))
    return lock_json(args.output/'audit_result.json',dict(passed=True,probes=sorted(probes,
        key=lambda x:(x['uid'],x['layout'],x['role'],x['opponent'])),no_actor_updates=True,
        prior_engineering=str(SOURCE/'audit_result.json')))


_EXPORTED_SOLVERS = {}


def exported_solver_type(rec):
    """Load the copied extension and adapter, rather than the working .so."""
    import importlib.util
    identity = o.json_hash(rec['runtime_fingerprints'])
    if identity not in _EXPORTED_SOLVERS:
        runtime = Path(rec['runtime_adapter']).parent
        binary = next(runtime.glob('bindings_deployment*.so'))
        spec = importlib.util.spec_from_file_location('factorial_'+identity+'.bindings_deployment',binary)
        cpp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cpp)
        spec = importlib.util.spec_from_file_location('factorial_adapter_'+identity,runtime/'deployment_puct.py')
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        adapter.cpp = cpp
        for name in ('_problem_wrapper_for','create_cpp_policy_oracle','create_cpp_value_oracle'):
            setattr(adapter.DeploymentPUCT,name,adapter.private_method(getattr(adapter.C_PUCT,name)))
        _EXPORTED_SOLVERS[identity] = adapter.DeploymentPUCT
    return _EXPORTED_SOLVERS[identity]


def load_recommended_solver(args,problem,role,seed):
    rec = locked_read(args.output/'recommended.json')
    if rec['implementation'] != fingerprints():
        raise ValueError('Recommendation implementation changed')
    for path,digest in rec['runtime_fingerprints'].items():
        if o.sha(path) != digest:
            raise ValueError('Exported runtime changed')
    requested = rec['pursuer_settings'] if role else rec['evader_settings']
    return make_own(problem,rec['registry'],rec['uid'],role,requested,seed,
                    solver_type=exported_solver_type(rec))


def replay(args,banks):
    rec = locked_read(args.output/'recommended.json')
    probes = []
    for role in (0,1):
        spec = banks['warmup'][0]
        p,state = o.restore_scenario(spec)
        for phase in (0,1):
            current = state.copy()
            if phase:
                current[[p.active_idxs[0],p.active_idxs[2]],0] = 0.
            a = make_own(p,registry(args),rec['uid'],role,rec['pursuer_settings'] if role else rec['evader_settings'],
                         spec['search_seeds'][role])
            b = load_recommended_solver(args,p,role,spec['search_seeds'][role])
            if not np.array_equal(a.policy(p,current),b.policy(p,current)):
                raise ValueError('Exported role/phase action differs')
            probes.append(dict(role=role,phase=phase,exact=True))
    task = requests(args,[rec['uid']],banks['warmup'],opponents=('Search Only',))[0]['task']
    seed_task(task)
    p,state,own,other = make_pair(task)
    original,detail,_,_ = simulate_record(p,state,own,other,1)
    seed_task(task)
    p,state = o.restore_scenario(task['spec'])
    own = load_recommended_solver(args,p,1,task['spec']['search_seeds'][1])
    other = base.private_solver(p,rec['registry']['L'],task['opponent'],0,
                               task['opponent_settings'],task['spec']['search_seeds'][0])
    exported,export_detail,_,_ = simulate_record(p,state,own,other,1)
    if science(original) != science(exported) or \
       trajectory_digest(detail['trajectory']) != trajectory_digest(export_detail['trajectory']):
        raise ValueError('Exported runtime full episode differs')
    return lock_json(args.output/'replay.json',dict(passed=True,probes=probes,
        full_episode_exact=True,exported_binary_loaded=True,uid=rec['uid']))


def paired_analysis(rows):
    result = {}
    for layout in ('four','none'):
        parts,keys,scenes = paired_index(rows,layout)
        result[layout] = {}
        for ref in REFS:
            result[layout][ref] = {}
            for op in o.OPPONENTS:
                chosen = [key for key in keys if key[1]==op]
                deltas = {k:parts['D'][k]['legal_double_capture']-parts[ref][k]['legal_double_capture'] for k in chosen}
                result[layout][ref][op] = dict(gained=sum(x>0 for x in deltas.values()),
                    lost=sum(x<0 for x in deltas.values()),net=sum(deltas.values()),
                    gained_scenes=[k[0] for k,x in deltas.items() if x>0],
                    lost_scenes=[k[0] for k,x in deltas.items() if x<0])
    return result


def diagnostic_summary(args,items):
    output = {}
    for uid in UIDS:
        for layout in ('four','none'):
            for op in o.OPPONENTS:
                part = [item for item in items if item['uid']==uid and item['task']['spec']['layout']==layout
                        and item['task']['opponent']==op]
                rr,phase_cost = [], {str(phase):dict(calls=0,step_calls=0,policy_evaluations=0,value_evaluations=0)
                                    for phase in (0,1)}
                for item in part:
                    row,record = read_episode(args,item)
                    rr.append(row)
                    for root in record['roots']:
                        target = phase_cost[str(root['phase'])]
                        target['calls'] += 1
                        for key,val in root['cost'].items():
                            target[key] += val
                first = [r['first_capture_time'] for r in rr if np.isfinite(r['first_capture_time'])]
                second = [r['capture_time']-r['first_capture_time'] for r in rr if r['legal_double_capture']]
                timeouts = [r for r in rr if r['one_capture_timeout']]
                output[f'{uid}/{layout}/{op}'] = dict(games=len(rr),
                    successes=sum(r['legal_double_capture'] for r in rr),
                    zero_capture=sum(r['capture_count']==0 for r in rr),
                    one_capture_timeout=len(timeouts),own_illegal=sum(r['pursuer_illegal'] for r in rr),
                    mean_completion_cost=float(np.mean([r['completion_cost'] for r in rr])),
                    first_capture_conditional=dict(n=len(first),mean=float(np.mean(first)) if first else None),
                    remaining_time_conditional=dict(n=len(first),mean=float(np.mean([80-t for t in first])) if first else None),
                    second_capture_conditional=dict(n=len(second),mean=float(np.mean(second)) if second else None),
                    timeout_with_first_by_40=sum(r['first_capture_time']<=40 for r in timeouts),phase_cost=phase_cost)
    return output


def report(args,banks,result):
    items = requests(args,UIDS,banks['validation_four']+banks['validation_none'])
    rows = [read_cached(args,item) for item in items]
    if args.smoke:
        for item in items:
            read_episode(args,item)
        message = ('隔离 smoke 已完成24个低预算对局的提交与轨迹校验。\n'
            '无效果结论，不执行因素检验、独立确认、回归或正式采用。推荐保持 old0:novalue。\n')
        (args.output/'REPORT.md').write_text(message)
        (args.output/'CONCLUSION.md').write_text(message)
        return
    paired = lock_json(args.output/'paired_analysis.json',paired_analysis(rows))
    effects = lock_json(args.output/'factorial_validation.json',factorial_statistics(rows))
    lock_json(args.output/'diagnostics.json',diagnostic_summary(args,items))
    parts,keys,_ = paired_index(rows)
    lock_json(args.output/'paired_results.json',[dict(scene=k[0],opponent=k[1],
        models={uid:{name:parts[uid][k][name] for name in ('legal_double_capture','capture_count',
            'first_capture_time','capture_time','one_capture_timeout','pursuer_illegal','completion_cost')}
                for uid in UIDS}) for k in keys])
    if result['confirmation']:
        confirm_rows = [read_cached(args,item) for item in requests(args,UIDS,banks['confirm_four']+banks['confirm_none'])]
        lock_json(args.output/'factorial_confirmation.json',factorial_statistics(confirm_rows))
    lines = ['追捕权重 × 搜索方式：冻结 2×2 实验','',
        f"结束原因：{result['reason']}；推荐：{result['recommended']}。",
        'A=L旧搜索，B=O旧搜索，C=L同步搜索，D=O同步搜索。唯一候选D；无训练、教师标签或优化器更新。','',
        '|配置|Policy Only|Search+Learning|Search Only|四障碍平均|无障碍平均|',
        '|---|---:|---:|---:|---:|---:|']
    for uid in UIDS:
        rates = [float(np.mean([r['legal_double_capture'] for r in rows if r['uid']==uid and
                  r['layout']=='four' and r['opponent']==op])) for op in o.OPPONENTS]
        rates += [v.aggregate(rows,uid,layout)['success_rate'] for layout in ('four','none')]
        lines.append('|'+uid+'|'+'|'.join(f'{x:.2%}' for x in rates)+'|')
    lines += ['','验证门槛：']
    for ref,comp in result['validation']['comparisons'].items():
        changes = paired['four'][ref]
        gained,lost = sum(x['gained'] for x in changes.values()),sum(x['lost'] for x in changes.values())
        failed = [k for k,val in comp['checks'].items() if not val]
        lines.append(f"D−{ref}：{comp['four']['delta']*100:+.2f}点，挽回{gained}、丢失{lost}；失败项：{failed}。")
    lines += ['','因素差异与交互作用（验证集探索性普通95%区间，不用于选模）：']
    for row in effects['comparisons']:
        lo,hi = row['ci95']
        lines.append(f"{row['id']}：{row['delta']*100:+.2f}点，区间[{lo*100:+.2f},{hi*100:+.2f}]点。")
    for name in ('confirmation','guard'):
        lines += ['',name+'：'+('未执行' if result[name] is None else '通过' if result[name]['passed'] else '失败')]
    if result['confirmation']:
        lines += ['',str(result['confirmation']['statistics'])]
    lines += ['', '单进程计时：'+str(result['timing']) if result['timing'] else '单进程计时：未执行，部署耗时未知。',
        '首捕时间、剩余时间、第二捕耗时、零捕、超时及阶段成本见diagnostics.json；逐场景挽回与丢失见paired_analysis.json。',
        '完整状态/动作轨迹、首捕后双方与Python/NumPy/Torch随机状态、根动作访问和均值统计见episodes/*.pt。',
        '同步因子包含动作统计、联合推进及rollout的整体变化；不能单独归因于联合推进。',
        '不同模型形成的首捕局面不同，条件耗时和几何差异不作单因素因果解释。',
        '相同模拟预算不代表等算力；不由并行吞吐推断部署提速。固定检查点对照不证明跨种子稳定或PPO的普遍贡献。',
        '所有门槛通过才采用D；失败保留old0:novalue，不换候选、不追加样本、不自动进入其他训练。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (args.output/'CONCLUSION.md').write_text('\n'.join(lines[:16])+
        '\n\n确认：'+('未执行' if not result['confirmation'] else '通过' if result['confirmation']['passed'] else '失败')+
        '；逃逸回归：'+('未执行' if not result['guard'] else '通过' if result['guard']['passed'] else '失败')+
        '；单进程计时：'+('已完成' if result['timing'] else '未执行')+
        '\n最终推荐：'+('D（O权重+同步追捕搜索）' if result['quality_gate_passed'] else 'old0:novalue（B）')+
        '。\n本轮无训练。全部因素差异、交互区间与失败诊断见 REPORT.md、factorial_validation.json 和 diagnostics.json。\n')


def finish(args,banks):
    verify(args)
    validation,confirmation,regression,measurement = [optional(args,x) for x in
        ('validation_result','confirmation_result','guard_result','timing_result')]
    if validation is None:
        raise ValueError('Validation incomplete')
    for before,after in ((validation,confirmation),(confirmation,regression),(regression,measurement)):
        if before and before['passed'] and after is None:
            raise ValueError('Required downstream stage incomplete')
        if after is not None and (not before or not before['passed']):
            raise ValueError('Downstream without prerequisite')
    accepted = bool(regression and regression['passed'] and measurement and not args.smoke)
    uid = 'D' if accepted else 'B'
    exported = {}
    for name,origin in registry(args).items():
        dest = args.output/'recommended/models'/name
        dest.mkdir(parents=True,exist_ok=True)
        for filename in o.MODEL_NAMES:
            if not (dest/filename).exists():
                shutil.copyfile(Path(origin['models'])/filename,dest/filename)
        if o.model_fingerprint(dest) != origin['model_hash']:
            raise ValueError('Exported weights changed')
        exported[name] = dict(models=str(dest),model_hash=origin['model_hash'])
    runtime = {}
    for origin in [Path(__file__),PROTOCOL,CODE/'solvers/deployment_puct.py',
                   CODE/'cpp/solvers/simultaneous_puct.hpp',CODE/'cpp/build_deployment/build_manifest.json',
                   *list((CODE/'cpp/build_deployment').glob('*.so'))]:
        dest = args.output/'recommended/runtime'/origin.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            shutil.copyfile(origin,dest)
        if o.sha(dest) != o.sha(origin):
            raise ValueError('Exported runtime changed')
        runtime[str(dest)] = o.sha(dest)
    lock_json(args.output/'recommended.json',dict(uid=uid,registry=exported,
        historical_uid=None if accepted else 'old0:novalue',pursuer_model='O',evader_model='O',
        pursuer_search='synchronous_DUCT' if accepted else 'legacy_PUCT',evader_search='legacy_PUCT',
        pursuer_settings=settings(args),evader_settings=settings(args,0),quality_gate_passed=accepted,
        runtime_fingerprints=runtime,runtime_adapter=str(args.output/'recommended/runtime/deployment_puct.py'),
        implementation=fingerprints(),loader='run_l45_search_factorial.load_recommended_solver'))
    reason = 'smoke_only' if args.smoke else 'adopted' if accepted else \
        'validation_no_gain' if not validation['passed'] else 'confirmation_failed' if not confirmation['passed'] else 'guard_failed'
    result = dict(reason=reason,recommended=uid,historical_recommendation=None if accepted else 'old0:novalue',
        quality_gate_passed=accepted,validation=validation,confirmation=confirmation,guard=regression,timing=measurement,
        training_episodes=0,teacher_queries=0,optimizer_updates=0,smoke=args.smoke,
        replay=replay(args,banks),continue_formal=False)
    lock_json(args.output/'results.json',result)
    report(args,banks,result)
    return result


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='pipeline',choices=['pipeline','prepare','audit','validate','confirm','guard','timing','report','replay','watch'])
    p.add_argument('--workers',type=int,default=80);p.add_argument('--output',type=Path)
    p.add_argument('--smoke',action='store_true');p.add_argument('--once',action='store_true')
    args=p.parse_args(argv);args.output=(args.output or (SMOKE_OUTPUT if args.smoke else DEFAULT_OUTPUT)).resolve()
    if args.workers<1:p.error('workers must be positive')
    return args



def main(argv=None):
    args=parse_args(argv)
    if args.stage=='watch':return watch(args)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(pid=os.getpid(),phase='prepare',workers=args.workers,started=time.time())
        mutex=threading.Lock();stop=threading.Event()
        def status(phase,**fields):
            with mutex:
                if phase!=state['phase']:
                    state.pop('completed',None);state.pop('total',None);state['stage_started']=time.time()
                state.update(phase=phase,updated_unix=time.time(),**fields)
                children=[p.pid for p in mp.active_children() if p.is_alive()]
                threads={str(pid):len(list(Path('/proc',str(pid),'task').glob('*'))) for pid in children}
                state.update(actual_workers=len(children),worker_pids=children,worker_threads=threads,
                    committed_cache=len(list((args.output/'cache_checksums').glob('*.lock.json'))))
                o.write_json(args.output/'status.json',state)
        def heartbeat():
            while not stop.wait(5):
                with mutex:phase=state['phase']
                status(phase)
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:
            banks=prepare(args);status('prepared')
            if args.stage=='prepare':return
            if args.stage in ('pipeline','audit'):audit(args,banks,status)
            if args.stage=='audit':status('complete',reason='engineering_passed');return
            if not locked_read(args.output/'audit_result.json')['passed']:raise ValueError('Audit required')
            if args.stage=='confirm':check_candidate(args)
            if args.stage=='guard' and not locked_read(args.output/'confirmation_result.json')['passed']:
                raise ValueError('Confirmation failed; regression forbidden')
            if args.stage=='timing' and not locked_read(args.output/'guard_result.json')['passed']:
                raise ValueError('Regression failed; timing forbidden')
            if args.stage in ('pipeline','validate','confirm','guard'):
                with mp.get_context('spawn').Pool(args.workers) as pool:
                    if args.stage in ('pipeline','validate'):validation=validate(args,banks,pool,status)
                    else:validation=locked_read(args.output/'validation_result.json')
                    if args.stage in ('pipeline','confirm') and validation['passed']:
                        confirmation=confirm(args,banks,pool,status)
                    else:confirmation=optional(args,'confirmation_result')
                    if args.stage in ('pipeline','guard') and confirmation and confirmation['passed']:guard(args,banks,pool,status)
                if args.stage!='pipeline':status('complete',reason=args.stage+'_complete');return
            if args.stage in ('pipeline','timing') and optional(args,'guard_result') and optional(args,'guard_result')['passed']:timing(args,banks,status)
            if args.stage=='replay':replay(args,banks);status('complete',reason='replay_passed');return
            result=finish(args,banks)
            status('complete',reason=result['reason'],recommended=result['recommended'],elapsed_seconds=time.time()-state['started'])
        except BaseException as error:status('failed',reason=type(error).__name__+': '+str(error));raise
        finally:stop.set();thread.join(timeout=1)


if __name__=='__main__':main()
