"""Frozen AQ once after first capture; fresh paired deployment evaluation."""
import os
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_var] = '1'
import argparse
import contextlib
import copy
import fcntl
import io
import multiprocessing as mp
import random
from pathlib import Path
import shutil
import threading
import time
import numpy as np
import torch
import run_l45_root_action_audit as a
from benchmark_baselines import get_oracles, oracle_paths
from solvers.root_audit_puct import RootPUCT
from solvers.aq_once import AQOnceSolver

o, s, v, f = a.o, a.s, a.h.v, a.f
ROOT = o.ROOT
SOURCE = a.DEFAULT
DEFAULT_OUTPUT = ROOT/'current/l45_aq_once_20261024'
SMOKE_OUTPUT = ROOT/'current/l45_aq_once_smoke_20261024'
lock_json, locked_read = v.lock_json, v.locked_read
SPLITS = dict(validation_four=('four',100), validation_none=('none',50),
    confirm_four=('four',400), confirm_none=('none',100), guard_four=('four',100),
    guard_none=('none',100), timing_four=('four',30), warmup=('four',1),
    audit_four=('four',1), audit_none=('none',1))
SCIENCE = a.h.SCIENCE


def config(args):
    return dict(schema=1, smoke=args.smoke, training_episodes=0, teacher_queries=0, optimizer_updates=0,
        budget=3 if args.smoke else 1000, search_only_budget=3 if args.smoke else 3000, depth=100,
        pursuer_betas=[.5,0.], evader_betas=[.5,.1], completion_bonus=0., candidate='AQ',
        rule='first_real_decision_after_first_legal_capture: highest_other_distinct_root_Q_once_then_O',
        ties='native_root_order', dedup='masked_float32_exact; A0_then_first_native_representative',
        no_distinct_action='A0_and_consume_opportunity', evader_deployment='complete_O_including_prediction',
        scenario_namespace=5002026102401 if args.smoke else 5002026102402,
        splits={name:[layout,1 if args.smoke else n] for name,(layout,n) in SPLITS.items()},
        bootstrap_seed=50020261024, bootstrap_draws=20000,
        comparisons=[dict(id='AQ-O',candidate='AQ',reference='O'),dict(id='AQ-L',candidate='AQ',reference='L')],
        request_caps=dict(validation=1350,confirmation=4500,evader_regression=1800,serial_timing=270),
        minimum_gain=.03,illegal_margin=.02,regression_margin=.03,no_automatic_expansion=True)



def settings(args, role=1):
    c = config(args)
    return dict(search_budget=c['budget'], search_only_budget=c['search_only_budget'],
        search_depth=100, beta_policy=.5, beta_value=0. if role == 1 else .1)


def fingerprints():
    paths = [Path(__file__),o.CODE/'solvers/aq_once.py',o.CODE/'L45_AQ_ONCE.md']
    return dict(files={str(p):o.sha(p) for p in paths}, inherited=a.fingerprints())



def registry(args):
    return {name: dict(models=str(args.output/'models'/name), model_hash=o.model_fingerprint(args.output/'models'/name))
            for name in ('O', 'L')}


def prepare(args):
    out = args.output
    if out == ROOT or out == SOURCE or SOURCE in out.parents:
        raise ValueError('Output must be isolated')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'experiment.lock.json').exists():
        verify(args)
        return locked_read(out/'scenarios.json')
    prior = locked_read(SOURCE/'results.json')
    if prior['recommended'] != 'old0:novalue':
        raise ValueError('Expected original recommendation')
    source_record = locked_read(SOURCE/'experiment.json')
    if source_record['implementation'] != a.fingerprints() or source_record['models'] != a.registry(a.parse_args([])):
        raise ValueError('Historical implementation or model changed')
    origins = {name:SOURCE/'models'/name for name in ('O','L')}
    source = {str(SOURCE/name):o.sha(SOURCE/name) for name in
        ('results.json','experiment.json','posthoc_simple_baselines.json','NEXT_DECISION_REVIEW.md')}
    for name,origin in origins.items():
        dest=out/'models'/name
        dest.mkdir(parents=True,exist_ok=True)
        for filename in o.MODEL_NAMES:
            source[str(origin/filename)]=o.sha(origin/filename)
            shutil.copyfile(origin/filename,dest/filename)
    reg=registry(args)
    for filename in (o.MODEL_NAMES[0],o.MODEL_NAMES[1],o.MODEL_NAMES[-1]):
        if reg['O']['model_hash'][filename] != reg['L']['model_hash'][filename]:
            raise ValueError('Frozen evader or original value changed')
    history=s.history_files(args)
    used=s.history_scenes(history)
    old_n=len(used)
    banks={}
    for phase,(name,(layout,n)) in enumerate(config(args)['splits'].items(),1510):
        bank,index=[],0
        while len(bank)<n:
            spec=o.make_scenario(o.scenario_seed(phase,config(args)['scenario_namespace'],index),layout)
            index+=1
            identity=s.previous.physical_hash(spec)
            if identity not in used:
                used.add(identity);bank.append(spec)
        banks[name]=bank
    record=dict(config=config(args),implementation=fingerprints(),source=source,
        history={str(p):o.sha(p) for p in history},models=reg)
    lock_json(out/'scenarios.json',banks)
    lock_json(out/'scenario_independence.json',dict(previous_physical_scenes=old_n,
        new_scenes=len(used)-old_n,overlap=0,history_reuse=False,
        scope='All recoverable l45 banks and rollout manifests, archives, smoke and unused scenes; not all NTE pretraining states.'))
    lock_json(out/'experiment.json',record)
    lock_json(out/'signature.json',dict(sha256=o.json_hash(dict(record=record,scenarios=o.sha(out/'scenarios.json')))))
    shutil.copyfile(o.CODE/'L45_AQ_ONCE.md',out/'PLAN.md')
    verify(args)
    return banks



def verify(args):
    record=locked_read(args.output/'experiment.json')
    if record['config'] != config(args) or record['implementation'] != fingerprints():
        raise ValueError('Protocol, implementation, build or platform changed')
    for path,sha in (record['source']|record['history']).items():
        if o.sha(path)!=sha:
            raise ValueError('Source or historical scene changed: '+path)
    if registry(args)!=record['models']:
        raise ValueError('Frozen model changed')
    banks=locked_read(args.output/'scenarios.json')
    identities=[s.previous.physical_hash(x) for bank in banks.values() for x in bank]
    if len(identities)!=len(set(identities)) or set(banks)!=set(config(args)['splits']):
        raise ValueError('Duplicate or missing scenario splits')
    if set(identities)&s.history_scenes([Path(p) for p in record['history']]):
        raise ValueError('Scene leakage')
    for name,(layout,n) in config(args)['splits'].items():
        if len(banks[name])!=n or any(x['layout']!=layout for x in banks[name]):
            raise ValueError('Scene shape changed')
    expected=o.json_hash(dict(record=record,scenarios=o.sha(args.output/'scenarios.json')))
    if locked_read(args.output/'signature.json')['sha256']!=expected:
        raise ValueError('Signature changed')
    return record



def private_solver(problem, model, method, role, requested, seed, export=False):
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
        solver = RootPUCT(policies, value,
            requested['search_only_budget'] if method == 'Search Only' else requested['search_budget'], *beta, export_root=export)
    solver.solver_wrapper.set_seed(int(seed))
    masked = o.RoleMaskedSolver(solver, o.TEAMS[role])
    masked.actual_settings = dict(method=method, budget=int(solver.solver_settings.number_simulations),
        depth=100, beta_policy=float(beta[0]), beta_value=float(beta[1]))
    return masked


def phase_pair(uid, role):
    if uid not in ('AQ','O','L'):
        raise ValueError('Unknown candidate')
    return ('L','L') if uid=='L' else ('O','O')



def make_own(problem,reg,uid,role,requested,seed):
    name=phase_pair(uid,role)[0]
    enabled=uid=='AQ' and role==1
    raw=private_solver(problem,reg[name],'Search + Learning',role,requested,seed,export=enabled).solver
    return AQOnceSolver(raw,o.TEAMS[role],enabled)



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
        effective = dict(aq_once=uid=='AQ' and role==1, phases=phase_pair(uid, role), weights={n: reg[n]['model_hash'] for n in set(phase_pair(uid, role))})
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
    seed = int(task['spec']['episode_hash'][:8], 16)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    problem, state = o.restore_scenario(task['spec'])
    problem.clear_closed_loop_cache()
    role, uid, reg = task['role'], task['uid'], task['registry']
    for model in reg.values():
        if o.model_fingerprint(model['models']) != model['model_hash']:
            raise ValueError('Worker weights changed')
    before, after = phase_pair(uid, role)
    if legacy:
        if uid=='AQ' and role==1:
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
        aq_once=uid=='AQ' and role==1, intervention_events=getattr(own,'events',[]),
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
    if row['aq_once'] != (item['uid']=='AQ' and task['role']==1):
        raise ValueError('Cached AQ rule changed')
    if row['switches'] not in (0,1) or len(row['intervention_events']) != row['switches']:
        raise ValueError('Invalid single intervention count')
    row['uid'] = item['uid']  # AQ and O share the complete evader behavior.
    return row


def run_evaluation(args, name, items, pool, status):
    if name in config(args)['request_caps'] and len(items)>config(args)['request_caps'][name]:
        raise ValueError('Stage request budget exceeded')
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


def comparison(rows, reference, confidence=None):
    four = s.point_comparison(rows,'AQ',reference,'four')
    none = s.point_comparison(rows,'AQ',reference,'none')
    checks = s.performance_checks(four,none,gain=True)
    checks['no_obstacle_own_illegal'] = none['own_illegal_delta'] <= .02+1e-12
    if confidence is not None:
        checks['simultaneous_lower_positive'] = confidence['simultaneous_lower95'] > 0.
    return dict(four=four,none=none,checks=checks,passed=all(checks.values()))


def statistics(rows, draws=20000):
    specs = config(parse_args([]))['comparisons']
    indexed = {uid:v.indexed_rows(rows,uid,'four',1) for uid in ('AQ','O','L')}
    keys = sorted(indexed['AQ'])
    if any(set(part) != set(keys) for part in indexed.values()):
        raise ValueError('Missing statistical pairs')
    scenes = sorted({key[0] for key in keys})
    differences, output = [], []
    for spec in specs:
        a,b = indexed['AQ'],indexed[spec['reference']]
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
    rng = np.random.default_rng(50020261024)
    samples = np.empty((draws,2))
    for start in range(0,draws,256):
        stop = min(start+256,draws)
        chosen = rng.integers(len(scenes),size=(stop-start,len(scenes)))
        samples[start:stop] = differences[:,chosen].mean(-1).T
    correction = max(0.,float(np.quantile(np.max(samples-observed,axis=1),.95)))
    for i,result in enumerate(output):
        result.update(ci95=np.quantile(samples[:,i],[.025,.975]).tolist(),simultaneous_lower95=float(observed[i]-correction))
    return dict(comparisons=output,draws=draws,bootstrap_seed=50020261024,
        method='max_centered_joint_one_sided_95',correction=correction)


def validate(args,banks,pool,status):
    rows = run_evaluation(args,'validation',requests(args,['AQ','O','L'],banks['validation_four']+banks['validation_none']),pool,status)
    comparisons = {ref:comparison(rows,ref) for ref in ('O','L')}
    passed = all(r['passed'] for r in comparisons.values()) and not args.smoke
    result = lock_json(args.output/'validation_result.json',dict(passed=passed,comparisons=comparisons,
        aggregates={uid:v.aggregate(rows,uid) for uid in ('AQ','O','L')}))
    if passed:
        lock_json(args.output/'candidate_lock.json',dict(candidate='AQ',rule=config(args)['rule'],models=registry(args),
            implementation=fingerprints(),comparisons=config(args)['comparisons'],validation_sha256=o.sha(args.output/'validation_result.json')))
    return result


def confirm(args,banks,pool,status):
    if not locked_read(args.output/'validation_result.json')['passed']:
        raise ValueError('Validation did not authorize confirmation')
    lock = locked_read(args.output/'candidate_lock.json')
    if lock['candidate'] != 'AQ' or lock['models'] != registry(args) or lock['implementation'] != fingerprints() or \
            lock['validation_sha256'] != o.sha(args.output/'validation_result.json') or \
            lock['rule'] != config(args)['rule'] or lock['comparisons'] != config(args)['comparisons']:
        raise ValueError('Candidate lock changed')
    rows = run_evaluation(args,'confirmation',requests(args,['AQ','O','L'],banks['confirm_four']+banks['confirm_none']),pool,status)
    stats = statistics(rows)
    comparisons = {r['reference']:comparison(rows,r['reference'],r) for r in stats['comparisons']}
    return lock_json(args.output/'confirmation_result.json',dict(passed=all(r['passed'] for r in comparisons.values()) and not args.smoke,
        statistics=stats,comparisons=comparisons))


def guard(args,banks,pool,status):
    if not locked_read(args.output/'confirmation_result.json')['passed']:
        raise ValueError('Confirmation did not authorize regression')
    rows = run_evaluation(args,'evader_regression',requests(args,['AQ','O','L'],banks['guard_four']+banks['guard_none'],role=0),pool,status)
    comparisons = {ref:{layout:s.point_comparison(rows,'AQ',ref,layout,role=0) for layout in ('four','none')} for ref in ('O','L')}
    passed = all(r['delta'] >= -.03-1e-12 and r['own_illegal_delta'] <= .02+1e-12
        for layouts in comparisons.values() for r in layouts.values())
    return lock_json(args.output/'guard_result.json',dict(passed=passed,comparisons=comparisons,
        evader_complete_O=True,logical_requests=len(rows),unique_requests=len({i['key'] for i in requests(args,['AQ','O','L'],banks['guard_four']+banks['guard_none'],role=0)})))


def timing(args,banks,status):
    if not locked_read(args.output/'guard_result.json')['passed']:
        raise ValueError('Regression did not authorize timing')
    if mp.active_children():
        raise RuntimeError('Exit parallel pool before serial timing')
    path = args.output/'timing_result.json'
    if path.with_suffix('.lock.json').exists():
        return locked_read(path)
    for item in requests(args,['AQ','O','L'],banks['warmup'],timing='warmup'):
        evaluate_one(item['task'])
    items = []
    for index,spec in enumerate(banks['timing_four']):
        order = ['AQ','O','L']
        order = order[index%3:]+order[:index%3]
        for op in o.OPPONENTS:
            items.extend(requests(args,order,[spec],opponents=(op,),timing='single_process'))
    rows = run_evaluation(args,'serial_timing',items,None,status)
    summaries = {}
    for uid in ('AQ','O','L'):
        part = [r for r in rows if r['uid'] == uid]
        decisions = np.concatenate([np.asarray(r['decision_latency_ms']) for r in part])[:,0]
        switched = [x for r in part for x in r['switch_decision_ms']]
        summaries[uid] = dict(pursuer_mean_ms=float(decisions.mean()),pursuer_p95_ms=float(np.percentile(decisions,95)),
            switch_mean_ms=float(np.mean(switched)) if switched else None,switch_p95_ms=float(np.percentile(switched,95)) if switched else None,switch_decisions=len(switched),
            episode_mean_ms=float(np.mean([r['episode_wall_ms'] for r in part])),episode_p95_ms=float(np.percentile([r['episode_wall_ms'] for r in part],95)),episodes=len(part),
            per_opponent={op:dict(episode_mean_ms=float(np.mean([r['episode_wall_ms'] for r in part if r['opponent'] == op]))) for op in o.OPPONENTS})
    for row in summaries.values():
        row['ratio_to_O'] = row['pursuer_mean_ms']/summaries['O']['pursuer_mean_ms']
    return lock_json(path,summaries)


def audit_episode(task):
    old = evaluate_one(task,legacy=True)
    new = evaluate_one(task)
    if science(old)!=science(new):
        raise ValueError('Original full episode scientific parity failed')
    trajectories=[]
    for legacy in (True,False):
        p,state=o.restore_scenario(task['spec'])
        p.clear_closed_loop_cache()
        reg,role=task['registry'],task['role']
        if legacy:
            own=f.make_solver(p,reg['O']['models'],'Search + Learning',role,task['own_settings'],task['spec']['search_seeds'][role])
            other=f.make_solver(p,reg['L']['models'],task['opponent'],1-role,task['opponent_settings'],task['spec']['search_seeds'][1-role])
        else:
            own=make_own(p,reg,'O',role,task['own_settings'],task['spec']['search_seeds'][role])
            other=private_solver(p,reg['L'],task['opponent'],1-role,task['opponent_settings'],task['spec']['search_seeds'][1-role])
        trace=[]
        simulate(p,state,own,other,role,trace=trace)
        trajectories.append(o.json_hash(trace))
    if trajectories[0]!=trajectories[1]:
        raise ValueError('Original complete action and state parity failed')
    return dict(layout=task['spec']['layout'],role=task['role'],opponent=task['opponent'],episode_exact=True,trajectory_exact=True)



def audit(args,banks,status=lambda *a,**k:None):
    path=args.output/'audit_result.json'
    if path.with_suffix('.lock.json').exists():
        return locked_read(path)
    tasks=[item['task'] for role in (0,1) for layout in ('four','none')
        for item in requests(args,['O'],banks['audit_'+layout],role=role)]
    probes=[]
    with mp.get_context('spawn').Pool(min(args.workers,len(tasks))) as pool:
        for index,probe in enumerate(pool.imap_unordered(audit_episode,tasks,chunksize=1),1):
            probes.append(probe);status('audit',completed=index,total=len(tasks))
    segmented=[]
    reg=registry(args)
    for layout in ('four','none'):
        spec=banks['audit_'+layout][0]
        traces={}
        for uid in ('O','AQ'):
            seed = int(spec['episode_hash'][:8], 16)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            p,state=o.restore_scenario(spec);p.clear_closed_loop_cache()
            own=make_own(p,reg,uid,1,settings(args),spec['search_seeds'][1])
            other=private_solver(p,reg['L'],'Search Only',0,settings(args,0),spec['search_seeds'][0])
            prefix=[]
            state,stats,_,_=simulate(p,state,own,other,1,stop_first=True,trace=prefix)
            record=dict(state=state.copy(),stats=copy.deepcopy(stats),own_rng=own.get_rng_state(p),
                opponent_rng=other_rng(other,p),python_rng=o.rng_state(),spec=spec,trajectory=prefix)
            a.commit_pt(args.output/'audit_snapshots'/f'{layout}_{uid}.pt',record)
            suffix=[]
            final,total,_,_=simulate(p,state.copy(),own,other,1,stats=stats,trace=suffix)
            expected=(science(metrics(p,final,total)),o.json_hash(suffix),own.get_rng_state(p),other_rng(other,p))
            restored=a.read_pt(args.output/'audit_snapshots'/f'{layout}_{uid}.pt')
            p,_=o.restore_scenario(restored['spec']);p.clear_closed_loop_cache()
            own=make_own(p,reg,uid,1,settings(args),spec['search_seeds'][1])
            other=private_solver(p,reg['L'],'Search Only',0,settings(args,0),spec['search_seeds'][0])
            own.set_rng_state(p,restored['own_rng']);other.solver.set_rng_state(p,restored['opponent_rng'])
            o.restore_rng(restored['python_rng'])
            resumed=[]
            final,total,_,_=simulate(p,restored['state'],own,other,1,stats=restored['stats'],trace=resumed)
            actual=(science(metrics(p,final,total)),o.json_hash(resumed),own.get_rng_state(p),other_rng(other,p))
            if actual!=expected:
                raise ValueError('AQ prefix/suffix snapshot mismatch')
            traces[uid]=o.json_hash(prefix)
            segmented.append(dict(uid=uid,layout=layout,exact=True,interventions=own.switches))
        if traces['AQ']!=traces['O']:
            raise ValueError('AQ changed the first-capture prefix')
    return lock_json(path,dict(passed=True,original_parity=probes,segmented_parity=segmented,
        unchanged_historical_sources=True,no_extra_search=True))



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
    validation,confirmation,regression,measurements=[optional(args,n) for n in
        ('validation_result','confirmation_result','guard_result','timing_result')]
    if validation is None:
        raise ValueError('Validation remains incomplete')
    for prerequisite,downstream,name in ((validation,confirmation,'confirmation'),
            (confirmation,regression,'regression'),(regression,measurements,'timing')):
        if prerequisite and prerequisite['passed'] and downstream is None:
            raise ValueError(name+' remains incomplete')
        if downstream is not None and (not prerequisite or not prerequisite['passed']):
            raise ValueError(name+' exists without passing prerequisite')
    accepted=bool(regression and regression['passed'] and measurements and not args.smoke)
    uid='AQ' if accepted else 'O'
    exported={}
    for name,origin in registry(args).items():
        dest=args.output/'recommended/models'/name
        dest.mkdir(parents=True,exist_ok=True)
        for filename in o.MODEL_NAMES:
            if not (dest/filename).exists():
                shutil.copyfile(Path(origin['models'])/filename,dest/filename)
        if o.model_fingerprint(dest)!=origin['model_hash']:
            raise ValueError('Export copy changed')
        exported[name]=dict(models=str(dest),model_hash=origin['model_hash'])
    runtime_sources=[Path(__file__),o.CODE/'solvers/aq_once.py',o.CODE/'solvers/root_audit_puct.py',
        o.CODE/'cpp/build_root_audit/build_manifest.json',*list((o.CODE/'cpp/build_root_audit').glob('*.so'))]
    runtime_fingerprints={}
    for origin in runtime_sources:
        dest=args.output/'recommended/runtime'/origin.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            shutil.copyfile(origin,dest)
        if o.sha(dest)!=o.sha(origin):
            raise ValueError('Exported runtime copy changed')
        runtime_fingerprints[str(dest)]=o.sha(dest)
    rec=dict(schema='l45_aq_once_recommendation_v1',uid=uid,historical_uid='old0:novalue' if uid=='O' else None,
        registry=exported,pursuer_model='O',aq_once=accepted,evader_model='O',rule=config(args)['rule'] if accepted else 'original_A0',
        pursuer_settings=settings(args),evader_settings=settings(args,0),
        loader='run_l45_aq_once.load_recommended_solver',adapter='solvers.aq_once.AQOnceSolver',
        implementation=fingerprints(),runtime_fingerprints=runtime_fingerprints,experiment_sha256=o.sha(args.output/'experiment.json'),
        quality_gate_passed=accepted,previous_confirmed_recommendation_preserved=not accepted,smoke=args.smoke)
    lock_json(args.output/'recommended.json',rec)
    reason='smoke_only' if args.smoke else 'adopted' if accepted else 'validation_no_gain' if not validation['passed'] else \
        'confirmation_failed' if not confirmation['passed'] else 'guard_failed'
    result=dict(reason=reason,recommended=uid,quality_gate_passed=accepted,smoke=args.smoke,
        training_episodes=0,teacher_queries=0,optimizer_updates=0,validation=validation,confirmation=confirmation,
        guard=regression,timing=measurements,replay=replay(args,banks),
        actual_stages=['prepare','audit','validate']+(['confirm'] if confirmation else [])+
        (['guard'] if regression else [])+(['timing'] if measurements else [])+['report','replay'],continue_formal=False)
    lock_json(args.output/'results.json',result)
    report(args,banks,result)
    return result



def report(args,banks,result):
    lines=['# 首捕后一次 AQ 改选实验','',f"结束原因：{result['reason']}；推荐：{result['recommended']}；smoke：{args.smoke}。",
        '新增训练、教师标签和优化器更新均为 0。O=推荐 PPO 第2048局；L=原 l45 追捕关闭旧价值。',
        'AQ仅在首捕后下一次真实决策，排除原联合动作后选最高Q不同执行动作；一次后恢复O。逃逸完整沿用O。',
        '正式追捕1000次/深度100/βπ=.5/βv=0；Search Only对手3000次。历史事后结果仅用于提出假设，未合并统计。','',
        '|模型|四障碍合法双捕|无障碍合法双捕|','|---|---:|---:|']
    rows=[read_cached(args,i) for i in requests(args,['AQ','O','L'],banks['validation_four']+banks['validation_none'])]
    for uid in ('AQ','O','L'):
        rates=[v.aggregate(rows,uid,layout)['success_rate'] for layout in ('four','none')]
        lines.append(f'|{uid}|{rates[0]:.2%}|{rates[1]:.2%}|')
    for ref,r in result['validation']['comparisons'].items():
        lines.append(f"\nAQ对{ref}：四障碍{r['four']['delta']:+.2%}，无障碍{r['none']['delta']:+.2%}；全部验证限制通过：{r['passed']}。失败项目：{[k for k,vv in r['checks'].items() if not vv]}。")
        lines.append('逐对手配对结果：'+str(r['four']['per_opponent_delta']))
    own=[r for r in rows if r['uid']=='AQ']
    lines += ['',f"AQ验证实际改选机会 {sum(r['switches'] for r in own)}；无不同动作回退 {sum(e['fallback'] for r in own for e in r['intervention_events'])}。",
        '独立确认：'+('通过' if result['confirmation'] and result['confirmation']['passed'] else '失败' if result['confirmation'] else '未执行'),
        '逃逸回归：'+('通过' if result['guard'] and result['guard']['passed'] else '失败' if result['guard'] else '未执行'),
        '耗时变化：'+(str(result['timing']) if result['timing'] else '未执行单进程计时，未知。')]
    if result['confirmation']:
        for comparison in result['confirmation']['statistics']['comparisons']:
            lines.append(str(comparison))
    lines += ['', '详细门槛、锁定统计、每局指标与改选状态/动作见 results.json、validation/summary.json 和 cache。',
        '未通过门槛即停止，不追加样本、不改变排序、不自动训练评分器。仅支持固定检查点与规则的结论，不宣称跨种子稳定或PPO独立贡献。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['pipeline','prepare','audit','validate','confirm','guard','timing','report','watch','replay'],default='pipeline')
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
            validation=optional(args,'validation_result')
            confirmation=optional(args,'confirmation_result')
            regression=optional(args,'guard_result')
            if args.stage in ('pipeline','validate'):
                validation=validate(args,banks,pool,status)
            if args.stage in ('pipeline','confirm') and validation and validation['passed']:
                verify(args)
                confirmation = confirm(args,banks,pool,status)
            elif args.stage == 'confirm':
                raise ValueError('Validation failed or incomplete')
            if args.stage in ('pipeline','guard') and confirmation and confirmation['passed']:
                verify(args)
                regression = guard(args,banks,pool,status)
            elif args.stage == 'guard':
                raise ValueError('Confirmation failed or incomplete')
        if args.stage == 'pipeline':
            if regression and regression['passed']:
                verify(args)
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
