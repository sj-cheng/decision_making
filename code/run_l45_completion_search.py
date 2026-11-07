"""Frozen-model, solver-local double-capture utility experiment.

No training path. Screen once; confirm only locked eligible configurations.
"""
import os
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_var] = '1'

import argparse
import contextlib
import fcntl
import io
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import run_l45_search_feedback as s
from solvers.completion_puct import CompletionPUCT
from benchmark_baselines import get_oracles, oracle_paths

o, v, f = s.o, s.v, s.f
ROOT = o.ROOT
DEFAULT_OUTPUT = ROOT/'current/l45_completion_search_20261023'
SMOKE_OUTPUT = ROOT/'current/l45_completion_search_smoke_20261023'
SPLITS = dict(screen_four=('four', 60), screen_none=('none', 30),
              confirm_four=('four', 400), confirm_none=('none', 100),
              guard_four=('four', 100), guard_none=('none', 100),
              timing_four=('four', 30), warmup=('four', 1),
              audit_four=('four', 1), audit_none=('none', 1))
REFERENCES, CANDIDATES = ('R0', 'L0'), ('R1', 'L1')
lock_json, locked_read = v.lock_json, v.locked_read


def comparisons():
    return [dict(id=f'{a}-{b}', candidate=a, reference=b)
            for a, b in [('R1', 'R0'), ('R1', 'L0'), ('L1', 'R0'), ('L1', 'L0'), ('R1', 'L1')]]


def config(args):
    return dict(schema=1, smoke=args.smoke, no_training=True, completion_bonus=1.,
                scenario_namespace=4602026102301 if args.smoke else 4602026102302,
                budget=3 if args.smoke else 1000, search_only_budget=3 if args.smoke else 3000,
                depth=100, pursuer_betas=[.5, 0.], evader_betas=[.5, .1],
                splits={k: [layout, 1 if args.smoke else count] for k, (layout, count) in SPLITS.items()},
                bootstrap_seed=46020261023, bootstrap_draws=20000, comparison_specs=comparisons(),
                minimum_gain=.03, illegal_margin=.02, regression_margin=.03,
                full_correction='max_centered_cluster_bootstrap_one_sided_95',
                partial_correction='fixed_five_hypotheses_bonferroni_one_sided_99',
                no_automatic_training_or_sweep=True)


def settings(args, role=1, bonus=0.):
    cfg = config(args)
    return dict(search_budget=cfg['budget'], search_only_budget=cfg['search_only_budget'],
                search_depth=100, beta_policy=.5, beta_value=0. if role == 1 else .1,
                completion_bonus=float(bonus) if role == 1 else 0.)


def fingerprints():
    paths = [Path(__file__), o.CODE/'solvers/completion_puct.py', o.CODE/'build_completion_search.py',
             o.CODE/'cpp/build_completion/build_manifest.json']
    paths += list((o.CODE/'cpp/build_completion').glob('*.so'))
    if len(paths) != 5:
        raise ValueError('Build exactly one private completion extension before running')
    return dict(private={str(p): o.sha(p) for p in paths}, historical=s.implementation_fingerprints())


def models(args):
    result = []
    for uid in ('R0', 'L0', 'R1', 'L1'):
        directory = args.output/('models/current' if uid.startswith('R') else 'models/baseline')
        item = f.model(uid, directory, cfg='novalue')
        item.update(uid=uid, family='completion' if uid.endswith('1') else 'reference',
                    bonus=float(uid.endswith('1')))
        result.append(item)
    return result


def prepare(args):
    out = args.output
    if out == ROOT or out in (s.DEFAULT_OUTPUT, s.SMOKE_OUTPUT) or s.DEFAULT_OUTPUT in out.parents:
        raise ValueError('Output must be isolated from historical experiments')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'experiment.lock.json').exists():
        verify(args)
        return locked_read(out/'scenarios.json')
    s.verify(s.parse_args([]))
    history = s.history_files(args)
    used = s.history_scenes(history)
    old_count = len(used)
    source = {}
    for name in ('current', 'baseline'):
        origin, dest = s.DEFAULT_OUTPUT/'models'/name, out/'models'/name
        dest.mkdir(parents=True, exist_ok=True)
        for filename in o.MODEL_NAMES:
            shutil.copyfile(origin/filename, dest/filename)
            source[str(origin/filename)] = o.sha(origin/filename)
    for name in ('recommended.json', 'results.json', 'experiment.json'):
        path = s.DEFAULT_OUTPUT/name
        source[str(path)] = o.sha(path)
    banks = {}
    for phase, (name, (layout, count)) in enumerate(config(args)['splits'].items(), 1100):
        bank, index = [], 0
        while len(bank) < count:
            spec = o.make_scenario(o.scenario_seed(phase, config(args)['scenario_namespace'], index), layout)
            index += 1
            identity = s.previous.physical_hash(spec)
            if identity not in used:
                used.add(identity)
                bank.append(spec)
        banks[name] = bank
    record = dict(config=config(args), implementation=fingerprints(), source=source,
                  history={str(p): o.sha(p) for p in history},
                  models={name: o.model_fingerprint(out/'models'/name) for name in ('current', 'baseline')})
    lock_json(out/'scenarios.json', banks)
    lock_json(out/'scenario_independence.json', dict(previous_physical_scenes=old_count,
        new_scenes=len(used)-old_count, overlap=0, history_files=len(history),
        scope='All recoverable l45/PPO banks including unused reservations, actual rollouts, smoke and archives; not all NTE pretraining states.'))
    lock_json(out/'experiment.json', record)
    lock_json(out/'signature.json', dict(sha256=o.json_hash(dict(record=record, scenarios=o.sha(out/'scenarios.json')))))
    (out/'PLAN.md').write_text((s.DEFAULT_OUTPUT/'NEXT_PLAN.md').read_text())
    verify(args)
    return banks


def verify(args):
    out = args.output
    record = locked_read(out/'experiment.json')
    if record['config'] != config(args) or record['implementation'] != fingerprints():
        raise ValueError('Protocol or implementation changed; refuse mixed resume')
    for path, expected in (record['source'] | record['history']).items():
        if o.sha(path) != expected:
            raise ValueError(f'Historical source changed: {path}')
    for name, expected in record['models'].items():
        if o.model_fingerprint(out/'models'/name) != expected:
            raise ValueError('Frozen model changed')
    banks = locked_read(out/'scenarios.json')
    identities = [s.previous.physical_hash(spec) for bank in banks.values() for spec in bank]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate physical scenarios between stages')
    if set(banks) != set(config(args)['splits']):
        raise ValueError('Scenario splits changed')
    for name, (layout, count) in config(args)['splits'].items():
        if len(banks[name]) != count or any(spec['layout'] != layout for spec in banks[name]):
            raise ValueError('Scenario count or layout changed')
        for spec in banks[name]:
            o.restore_scenario(spec)
    expected = o.json_hash(dict(record=record, scenarios=o.sha(out/'scenarios.json')))
    if locked_read(out/'signature.json')['sha256'] != expected:
        raise ValueError('Scenario signature changed')
    return record


def make_own(problem, directory, role, requested, seed):
    if role == 0 and requested['completion_bonus'] != 0.:
        raise ValueError('Evader utility must remain original')
    paths, value_path = oracle_paths(str(ROOT), iteration=45, model_dir=str(directory))
    with contextlib.redirect_stdout(io.StringIO()):
        policies, value = get_oracles(problem, value_oracle_name='deterministic', value_oracle_path=value_path,
                                     policy_oracle_name='gaussian', policy_oracle_paths=paths)
        solver = CompletionPUCT(policies, value, requested['search_budget'],
                                requested['beta_value'], requested['completion_bonus'])
    solver.solver_wrapper.set_seed(int(seed))
    ss = solver.solver_settings
    actual = dict(method='Search + Learning', budget=int(ss.number_simulations), depth=int(ss.search_depth),
                  beta_policy=float(ss.beta_policy), beta_value=float(ss.beta_value),
                  completion_bonus=float(solver.solver_wrapper.completion_bonus), adapter='private_completion')
    if actual['completion_bonus'] != requested['completion_bonus']:
        raise ValueError('C++ completion setting mismatch')
    masked = o.RoleMaskedSolver(solver, o.TEAMS[role])
    masked.actual_settings = actual
    return masked


def requests(args, selected, scenes, role=1, opponents=o.OPPONENTS, timing='parallel'):
    protocol = o.sha(args.output/'experiment.json')
    items = []
    for item in selected:
        for spec in scenes:
            for opponent in opponents:
                identity = dict(protocol=protocol, policy_id=item['policy_id'], episode_hash=spec['episode_hash'],
                    own_settings=settings(args, role, item['bonus']), opponent_settings=settings(args, 0),
                    mode='search', role=role, opponent=opponent, timing_context=timing)
                key = o.json_hash(identity)
                task = dict(**identity, key=key, models=item['models'], model_hash=item['model_hash'],
                            baseline=str(args.output/'models/baseline'), spec=spec)
                items.append(dict(key=key, uid=item['uid'], label=item['uid'], family=item['family'],
                                  config='completion' if item['bonus'] else 'original', training_seed=None,
                                  model_hash=item['model_hash'], task=task))
    return items


def evaluate_one(task, legacy=False):
    torch.set_num_threads(1)
    if o.model_fingerprint(task['models']) != task['model_hash']:
        raise ValueError('Evaluation weights changed')
    problem, state = o.restore_scenario(task['spec'])
    role = task['role']
    factory = f.make_solver if legacy else None
    own = (factory(problem, task['models'], 'Search + Learning', role, task['own_settings'], task['spec']['search_seeds'][role])
           if legacy else make_own(problem, task['models'], role, task['own_settings'], task['spec']['search_seeds'][role]))
    other = f.make_solver(problem, task['baseline'], task['opponent'], 1-role,
                          task['opponent_settings'], task['spec']['search_seeds'][1-role])
    team = o.TeamSolver(own if role == 0 else other, own if role == 1 else other)
    decisions = []

    class TimedTeam:
        def policy(self, p, current):
            action = team.policy(p, current)
            if task['timing_context'] in ('single_process', 'warmup'):
                decisions.append([team.last_pursuer_latency_ms, team.last_evader_latency_ms, team.last_joint_latency_ms])
            return action

        def __getattr__(self, name):
            return getattr(team, name)

    machine = v.machine_snapshot() if task['timing_context'] == 'single_process' else None
    started = time.perf_counter()
    row = o.run_episode(problem, TimedTeam(), state)
    row['episode_wall_ms'] = (time.perf_counter()-started)*1000.
    illegal = row['illegal_terminal']
    row.update(legal_double_capture=int(row['double_capture'] and not illegal),
        legal_survival=int(row['capture_count'] < 2 and row['survival_time'] >= problem.tf and not illegal),
        one_capture_timeout=int(row['capture_count'] == 1 and row['survival_time'] >= problem.tf and not illegal),
        pursuer_illegal=int(row['pursuer_obstacle_collision'] or row['pursuer_out_of_bounds']),
        evader_illegal=int(row['evader_obstacle_collision'] or row['evader_out_of_bounds']))
    row['completion_cost'] = row['capture_time'] if row['legal_double_capture'] else float(problem.tf)
    row.update({k: task[k] for k in ('key', 'policy_id', 'mode', 'role', 'opponent', 'timing_context')})
    row.update(episode_hash=task['spec']['episode_hash'], layout=task['spec']['layout'],
        search_seeds=task['spec']['search_seeds'], actual_own=own.actual_settings,
        actual_opponent=dict(other.actual_settings, completion_bonus=0., adapter='historical'))
    diagnostics = [] if legacy else own.solver.diagnostics
    row['search_completion_signal'] = {phase: dict(
        root_queries=sum(d['before_first'] == before for d in diagnostics),
        tree_events=sum(d['tree_events'] for d in diagnostics if d['before_first'] == before),
        rollout_events=sum(d['rollout_events'] for d in diagnostics if d['before_first'] == before),
        queries_with_signal=sum(d['before_first'] == before and (d['tree_events']+d['rollout_events'] > 0)
                                for d in diagnostics)) for phase, before in [('before_first', True), ('after_first', False)]}
    if decisions:
        row['decision_latency_ms'] = decisions
    if machine:
        row['timing_machine'] = dict(before=machine, after=v.machine_snapshot())
    return row


def read_cached(args, item):
    row = v.read_cached(args, item)
    if row['actual_own']['completion_bonus'] != item['task']['own_settings']['completion_bonus'] or \
            row['actual_opponent']['completion_bonus'] != 0.:
        raise ValueError('Cached utility isolation mismatch')
    return row


def run_evaluation(args, name, items, pool, status):
    manifest = [{k: val for k, val in item.items() if k != 'task'} for item in items]
    if len({(item['uid'], item['key']) for item in items}) != len(items):
        raise ValueError('Duplicate evaluation request')
    lock_json(args.output/name/'manifest.json', manifest)
    cache = args.output/'cache'
    cache.mkdir(exist_ok=True)
    unique = {item['key']: item for item in items}
    for key, item in unique.items():
        if (cache/(key+'.json')).exists():
            read_cached(args, item)
    missing = [item['task'] for key, item in unique.items() if not (cache/(key+'.json')).exists()]
    done = len(unique)-len(missing)
    status(name, completed=done, total=len(unique), label_requests=len(items))
    iterator = pool.imap_unordered(evaluate_one, missing, chunksize=1) if pool else map(evaluate_one, missing)
    for row in iterator:
        row['content_sha256'] = o.json_hash(row)
        o.write_json(cache/(row['key']+'.json'), row)
        done += 1
        if done % 10 == 0 or done == len(unique):
            status(name, completed=done, total=len(unique), label_requests=len(items))
    rows = [read_cached(args, item) for item in items]
    summary = [dict(uid=uid, layout=layout, opponent=opponent, **o.summarize(part))
               for uid in sorted({r['uid'] for r in rows}) for layout in ('four', 'none')
               for opponent in o.OPPONENTS
               if (part := [r for r in rows if (r['uid'], r['layout'], r['opponent']) == (uid, layout, opponent)])]
    o.write_json(args.output/name/'summary.json', summary)
    return rows


def checks(rows, candidate, reference, confidence=None):
    four = s.point_comparison(rows, candidate, reference, 'four')
    none = s.point_comparison(rows, candidate, reference, 'none')
    result = s.performance_checks(four, none, gain=True)
    if confidence is not None:
        result['formal_lower_positive'] = confidence['formal_lower95'] > 0.
    return dict(four=four, none=none, checks=result, passed=all(result.values()))


def screen(args, banks, pool, status):
    rows = run_evaluation(args, 'screen', requests(args, models(args), banks['screen_four']+banks['screen_none']), pool, status)
    records = [dict(uid=uid, references={ref: checks(rows, uid, ref) for ref in REFERENCES}) for uid in CANDIDATES]
    eligible = [r['uid'] for r in records if all(c['passed'] for c in r['references'].values())]
    return lock_json(args.output/'screen_result.json', dict(candidates=records, eligible=eligible,
        aggregates=[dict(uid=uid, layout=layout, **v.aggregate(rows, uid, layout))
                    for uid in (*REFERENCES, *CANDIDATES) for layout in ('four', 'none')],
        selected_models=[m for m in models(args) if m['uid'] in eligible],
        completion_signal={uid: {phase: {field: sum(r['search_completion_signal'][phase][field]
            for r in rows if r['uid'] == uid and r['layout'] == 'four')
            for field in ('root_queries', 'tree_events', 'rollout_events', 'queries_with_signal')}
            for phase in ('before_first', 'after_first')} for uid in (*REFERENCES, *CANDIDATES)}
            if rows and 'search_completion_signal' in rows[0] else None,
        rule='Both references +3pp, strong opponent positive, prespecified regressions'))


def statistics(rows, eligible, draws=20000):
    specs = [spec for spec in comparisons() if spec['candidate'] in eligible and
             (spec['reference'] in REFERENCES or spec['reference'] in eligible)]
    indexed = {uid: v.indexed_rows(rows, uid, 'four') for spec in specs
               for uid in (spec['candidate'], spec['reference'])}
    if not specs:
        raise ValueError('No locked confirmation comparisons')
    keys = sorted(next(iter(indexed.values())))
    if any(set(part) != set(keys) for part in indexed.values()):
        raise ValueError('Missing physical pairs')
    scenes = sorted({scene for scene, _ in keys})
    positions = [[i for i, (scene, _) in enumerate(keys) if scene == physical] for physical in scenes]
    differences, outputs = [], []
    for spec in specs:
        a, b = indexed[spec['candidate']], indexed[spec['reference']]
        if any(a[key]['search_seeds'] != b[key]['search_seeds'] for key in keys):
            raise ValueError('Search seeds do not match')
        delta = np.array([a[key]['legal_double_capture']-b[key]['legal_double_capture'] for key in keys], float)
        differences.append([float(delta[idx].mean()) for idx in positions])
        outputs.append(dict(**spec, **s.point_comparison(rows, spec['candidate'], spec['reference']),
                            n_scenes=len(scenes), n_games=len(keys)))
    differences = np.asarray(differences)
    observed = differences.mean(1)
    rng = np.random.default_rng(46020261023)
    samples = np.empty((draws, len(specs)))
    for start in range(0, draws, 256):
        chosen = rng.integers(len(scenes), size=(min(256, draws-start), len(scenes)))
        samples[start:start+len(chosen)] = differences[:, chosen].mean(-1).T
    full = set(eligible) == set(CANDIDATES)
    correction = max(0., float(np.quantile(np.max(samples-observed, axis=1), .95))) if full else None
    for i, result in enumerate(outputs):
        result['ci95'] = np.quantile(samples[:, i], [.025, .975]).tolist()
        result['formal_lower95'] = float(observed[i]-correction if full else
            observed[i]-max(0., float(np.quantile(samples[:, i]-observed[i], .99))))
    return dict(comparisons=outputs, draws=draws, bootstrap_seed=46020261023,
                preregistered_comparison_count=5, observed_comparison_count=len(specs),
                correction='max_centered_joint_95' if full else 'fixed_five_bonferroni_99',
                note='Physical scenes jointly resampled; all opponents retained; frozen checkpoint inference only.')


def confirm(args, banks, pool, status):
    gate = locked_read(args.output/'screen_result.json')
    if not gate['eligible']:
        raise ValueError('Locked screen did not authorize confirmation')
    selected = [m for m in models(args) if m['uid'] in (*REFERENCES, *gate['eligible'])]
    if [m for m in selected if m['uid'] in gate['eligible']] != gate['selected_models']:
        raise ValueError('Locked candidates changed')
    rows = run_evaluation(args, 'confirmation', requests(args, selected, banks['confirm_four']+banks['confirm_none']), pool, status)
    stats = statistics(rows, gate['eligible'])
    indexed = {r['id']: r for r in stats['comparisons']}
    records = [dict(uid=uid, references={ref: checks(rows, uid, ref, indexed[f'{uid}-{ref}']) for ref in REFERENCES})
               for uid in gate['eligible']]
    return lock_json(args.output/'confirmation_result.json', dict(statistics=stats, candidates=records,
        eligible=[r['uid'] for r in records if all(c['passed'] for c in r['references'].values())],
        aggregates=[dict(uid=m['uid'], layout=layout, **v.aggregate(rows, m['uid'], layout))
                    for m in selected for layout in ('four', 'none')]))


def guard(args, banks, pool, status):
    eligible = locked_read(args.output/'confirmation_result.json')['eligible']
    if not eligible:
        raise ValueError('No independently confirmed candidate for evader regression')
    selected = [m for m in models(args) if m['uid'] in (*REFERENCES, *eligible)]
    rows = run_evaluation(args, 'evader_guard', requests(args, selected, banks['guard_four']+banks['guard_none'], role=0), pool, status)
    records = []
    for uid in eligible:
        parts = {f'{ref}/{layout}': s.point_comparison(rows, uid, ref, layout, role=0)
                 for ref in REFERENCES for layout in ('four', 'none')}
        records.append(dict(uid=uid, comparisons=parts,
                            passed=all(p['delta'] >= -.03-1e-12 and p['own_illegal_delta'] <= .02+1e-12 for p in parts.values())))
    return lock_json(args.output/'guard_result.json', dict(candidates=records,
        accepted=[r['uid'] for r in records if r['passed']], evader_utility='original'))


def timing(args, banks, status):
    accepted = locked_read(args.output/'guard_result.json')['accepted']
    if not accepted:
        raise ValueError('No regression-passing candidate')
    children = Path(f'/proc/{os.getpid()}/task/{os.getpid()}/children').read_text().split()
    if any(b'spawn_main' in Path(f'/proc/{pid}/cmdline').read_bytes() for pid in children):
        raise ValueError('Parallel workers must exit before timing')
    selected = [m for m in models(args) if m['uid'] in (*REFERENCES, *accepted)]
    for item in requests(args, selected, banks['warmup'], opponents=('Policy Only',), timing='warmup'):
        evaluate_one(item['task'])
    items = []
    for index, spec in enumerate(banks['timing_four']):
        offset = index % len(selected)
        items += requests(args, selected[offset:]+selected[:offset], [spec], opponents=('Policy Only',), timing='single_process')
    rows = run_evaluation(args, 'serial_timing', items, None, status)
    result = {}
    for model in selected:
        part = [r for r in rows if r['uid'] == model['uid']]
        decisions = np.concatenate([r['decision_latency_ms'] for r in part])[:, 0]
        result[model['uid']] = dict(pursuer_mean_ms=float(decisions.mean()), pursuer_p50_ms=float(np.median(decisions)),
            pursuer_p95_ms=float(np.percentile(decisions, 95)), episode_mean_ms=float(np.mean([r['episode_wall_ms'] for r in part])),
            decision_counts=[len(r['decision_latency_ms']) for r in part],
            machines=[r['timing_machine'] for r in part], episodes=len(part), decisions=len(decisions))
    for value in result.values():
        value['ratio_to_R0'] = value['pursuer_mean_ms']/result['R0']['pursuer_mean_ms']
    return lock_json(args.output/'timing_result.json', result)


def audit(args, banks):
    path = args.output/'audit_result.json'
    if path.with_suffix('.lock.json').exists():
        return locked_read(path)
    parity = []
    reference = models(args)[0]
    for layout in ('four', 'none'):
        spec = banks['audit_'+layout][0]
        problem, state = o.restore_scenario(spec)
        for phase in ('before_first', 'after_first'):
            current = state.copy()
            if phase == 'after_first':
                current[problem.active_idxs[0], 0] = 0.
            for role in (0, 1):
                requested = settings(args, role)
                old = f.make_solver(problem, reference['models'], 'Search + Learning', role, requested, spec['search_seeds'][role])
                new = make_own(problem, reference['models'], role, requested, spec['search_seeds'][role])
                if not np.array_equal(old.policy(problem, current), new.policy(problem, current)):
                    raise ValueError('Zero bonus does not exactly reproduce historical action')
                parity.append(dict(layout=layout, phase=phase, role=role, exact_action=True))
        for item in requests(args, [reference], [spec]):
            old, new = evaluate_one(item['task'], legacy=True), evaluate_one(item['task'])
            excluded = {'actual_own', 'actual_opponent', 'search_completion_signal', 'episode_wall_ms'}
            scientific = lambda row: {k: val for k, val in row.items() if k not in excluded and 'latency' not in k}
            if o.json_hash(scientific(old)) != o.json_hash(scientific(new)):
                raise ValueError('Zero bonus changed scientific episode results')
            parity.append(dict(layout=layout, opponent=item['task']['opponent'], exact_episode=True))
    return lock_json(path, dict(passed=True, zero_bonus_parity=parity,
        historical_sources_unchanged=True, isolated_private_build=True,
        scope='Actions before/after capture, both roles/maps; full episodes versus three original opponents'))


def read_recommended(args):
    rec = locked_read(args.output/'recommended.json')
    if rec['experiment_sha256'] != o.sha(args.output/'experiment.json') or o.model_fingerprint(rec['models']) != rec['model_hash']:
        raise ValueError('Recommended artifact changed')
    return rec


def load_recommended_solver(args, problem, role, seed):
    rec = read_recommended(args)
    return make_own(problem, rec['models'], role, rec['pursuer_settings' if role == 1 else 'evader_settings'], seed)


def replay(args, banks):
    rec = read_recommended(args)
    source = next(m for m in models(args) if m['uid'] == rec['uid'])
    problem, state = o.restore_scenario(banks['warmup'][0])
    for role in (0, 1):
        seed = banks['warmup'][0]['search_seeds'][role]
        expected = make_own(problem, source['models'], role, settings(args, role, source['bonus']), seed)
        actual = load_recommended_solver(args, problem, role, seed)
        if not np.array_equal(expected.policy(problem, state), actual.policy(problem, state)):
            raise ValueError('Recommended export replay changed')
    return lock_json(args.output/'replay.json', dict(uid=rec['uid'], exact_roles=[0, 1], passed=True))


def finish(args, banks):
    screen_result = locked_read(args.output/'screen_result.json')
    confirmation = locked_read(args.output/'confirmation_result.json') if (args.output/'confirmation_result.lock.json').exists() else None
    regression = locked_read(args.output/'guard_result.json') if (args.output/'guard_result.lock.json').exists() else None
    measurements = locked_read(args.output/'timing_result.json') if (args.output/'timing_result.lock.json').exists() else None
    if screen_result['eligible'] and confirmation is None:
        raise ValueError('Confirmation remains incomplete')
    if confirmation and confirmation['eligible'] and regression is None:
        raise ValueError('Evader regression remains incomplete')
    if regression and regression['accepted'] and measurements is None:
        raise ValueError('Serial timing remains incomplete')
    accepted = regression['accepted'] if regression else []
    uid = 'R0'
    if accepted:
        stats = {r['uid']: r for r in confirmation['aggregates'] if r['layout'] == 'four'}
        uid = min(accepted, key=lambda name: (-stats[name]['success_rate'], stats[name]['own_illegal_rate'],
                  stats[name]['completion_cost'], measurements[name]['pursuer_mean_ms'], name))
    chosen = next(m for m in models(args) if m['uid'] == uid)
    dest = args.output/'recommended/models'
    dest.mkdir(parents=True, exist_ok=True)
    for name in o.MODEL_NAMES:
        if not (dest/name).exists():
            shutil.copyfile(Path(chosen['models'])/name, dest/name)
    if o.model_fingerprint(dest) != chosen['model_hash']:
        raise ValueError('Exported recommendation differs from locked model')
    rec = dict(uid=uid, historical_uid='old0:novalue' if uid == 'R0' else None, models=str(dest),
        model_hash=chosen['model_hash'], pursuer_settings=settings(args, 1, chosen['bonus']),
        evader_settings=settings(args, 0), adapter='solvers.completion_puct.CompletionPUCT',
        loader='run_l45_completion_search.load_recommended_solver',
        quality_gate_passed=bool(accepted) and not args.smoke, smoke=args.smoke,
        experiment_sha256=o.sha(args.output/'experiment.json'),
        contribution='search utility; no network training' if accepted else 'no confirmed additional gain',
        timing=measurements, original_environment_reward_unchanged=True)
    lock_json(args.output/'recommended.json', rec)
    reason = 'adopted' if accepted else 'screen_no_gain' if not screen_result['eligible'] else 'no_candidate_confirmed'
    result = dict(reason=reason, smoke=args.smoke, recommended=uid, quality_gate_passed=rec['quality_gate_passed'],
                  training_episodes=0, screen=screen_result, confirmation=confirmation, guard=regression,
                  timing=measurements, replay=replay(args, banks),
                  actual_stages=['prepare', 'audit', 'screen']+(['confirm'] if confirmation else [])+
                    (['guard'] if regression else [])+(['timing'] if measurements else [])+['report', 'replay'])
    lock_json(args.output/'results.json', result)
    lines = ['# 固定模型：合法双捕完成目标进入 MCTS', '',
        f'结束原因：{reason}；smoke：{args.smoke}；推荐：{uid}。',
        '本轮没有训练网络；本方搜索完成奖励固定为1.0，真实环境与三个对手使用原奖励。', '',
        '|配置|筛选四障碍合法双捕|Search Only|无障碍|', '|---|---:|---:|---:|']
    screen_rows = [read_cached(args, item) for item in requests(args, models(args), banks['screen_four']+banks['screen_none'])]
    for name in ('R0', 'L0', 'R1', 'L1'):
        four, none = v.aggregate(screen_rows, name), v.aggregate(screen_rows, name, 'none')
        strong = [r for r in screen_rows if (r['uid'], r['layout'], r['opponent']) == (name, 'four', 'Search Only')]
        lines.append(f"|{name}|{four['success_rate']:.2%}|{np.mean([r['legal_double_capture'] for r in strong]):.2%}|{none['success_rate']:.2%}|")
    lines += ['', 'R=旧PPO第2048局；L=原l45；0=原搜索效用；1=本方合法完成双捕额外1.0。',
              '筛选只用于固定候选，不构成独立确认。没有通过门槛则保留old0:novalue。',
              f"实际阶段：{', '.join(result['actual_stages'])}。",
              'PPO额外贡献只有独立确认R1−L1校正下界大于零时才成立。',
              '耗时变化：见timing_result.json。' if measurements else '耗时变化未测量：没有完成全部采用门槛的候选。',
              '本轮结束，不自动追加训练、奖励系数扫描、样本或搜索预算。',
              '详细比较、统计和推荐配置见results.json与recommended.json。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if not args.smoke:
        for parent in (s.DEFAULT_OUTPUT, s.SOURCE):
            path = parent/'REPORT.md'
            marker = str(args.output/'REPORT.md')
            if marker not in path.read_text():
                with path.open('a') as stream:
                    stream.write(f'\n\n固定模型搜索完成目标实验已结束：{reason}，推荐{uid}。[报告]({marker})。\n')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['pipeline', 'prepare', 'audit', 'screen', 'confirm', 'guard', 'timing', 'report', 'watch', 'replay'], default='pipeline')
    parser.add_argument('--workers', type=int, default=80)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    args.output = (args.output or (SMOKE_OUTPUT if args.smoke else DEFAULT_OUTPUT)).resolve()
    if args.workers < 1:
        parser.error('--workers must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.stage == 'watch':
        while True:
            record = o.read_json(args.output/'status.json')
            pid = record['pid']
            parent = Path(f'/proc/{pid}')
            alive = parent.exists()
            workers = []
            if alive:
                try:
                    for child in (parent/f'task/{pid}/children').read_text().split():
                        if b'spawn_main' in Path(f'/proc/{child}/cmdline').read_bytes():
                            workers.append(int(child))
                except FileNotFoundError:
                    pass
            print(dict(record, parent_alive=alive, live_workers=len(workers),
                       committed_cache_files=len(list((args.output/'cache').glob('*.json')))), flush=True)
            if args.once:
                return
            time.sleep(5)
    args.output.mkdir(parents=True, exist_ok=True)
    handle = (args.output/'run.lock').open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(1)
    started, phase_started, current_phase = time.time(), time.time(), None
    initial_completed = 0

    def status(phase, **fields):
        nonlocal phase_started, current_phase, initial_completed
        if phase != current_phase:
            phase_started, current_phase = time.time(), phase
            initial_completed = fields.get('completed', 0)
        record = dict(phase=phase, pid=os.getpid(), workers=args.workers, updated_unix=time.time(), **fields)
        if 'completed' in fields:
            rate = (fields['completed']-initial_completed)/max(time.time()-phase_started, 1e-9)
            record.update(rate=rate, eta_seconds=(fields['total']-fields['completed'])/rate if rate else None)
        o.write_json(args.output/'status.json', record)
        print(record, flush=True)

    try:
        status('prepare')
        banks = prepare(args)
        if args.stage == 'prepare':
            status('prepared')
            return
        if args.stage in ('replay', 'report', 'timing'):
            value = replay(args, banks) if args.stage == 'replay' else finish(args, banks) if args.stage == 'report' else timing(args, banks, status)
            status('complete' if args.stage == 'report' else 'stage_complete', stage=args.stage)
            return value
        status('audit')
        audit(args, banks)
        if args.stage == 'audit':
            status('audited')
            return
        confirmation = regression = None
        with mp.get_context('spawn').Pool(args.workers) as pool:
            gate = screen(args, banks, pool, status) if args.stage in ('pipeline', 'screen') else locked_read(args.output/'screen_result.json')
            if args.stage in ('pipeline', 'confirm') and gate['eligible']:
                confirmation = confirm(args, banks, pool, status)
            elif args.stage in ('confirm', 'guard') and not gate['eligible']:
                raise ValueError('Screen did not authorize downstream stage')
            if args.stage == 'guard':
                confirmation = locked_read(args.output/'confirmation_result.json')
            if (args.stage == 'pipeline' and confirmation and confirmation['eligible']) or args.stage == 'guard':
                regression = guard(args, banks, pool, status)
        if args.stage == 'pipeline':
            if regression and regression['accepted']:
                timing(args, banks, status)
            value = finish(args, banks)
            status('complete', recommended=value['recommended'], quality_gate_passed=value['quality_gate_passed'],
                   reason=value['reason'], elapsed_seconds=time.time()-started)
            return value
        status('stage_complete', stage=args.stage)
    except BaseException as error:
        status('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
        raise
    finally:
        handle.close()


if __name__ == '__main__':
    main()
