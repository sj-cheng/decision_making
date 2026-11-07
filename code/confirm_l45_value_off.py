"""Frozen, performance-first confirmation of l45/PPO with and without old value.

This entry point has no training path. Historical runners and C++ stay unchanged.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import copy
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import threading
import time

import numpy as np
import torch

import run_l45_ppo_followup as followup

o = followup.o
previous = followup.previous
ROOT = o.ROOT
SOURCE = followup.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT / 'current/l45_value_off_confirm_20261023'
SMOKE_OUTPUT = ROOT / 'current/l45_value_off_smoke_20261023'
STARTUP_ARCHIVE = ROOT / 'current/l45_value_off_startup_archive_20261023'
CONFIGS = ('base', 'novalue')
SEEDS = (0, 1, 2)
BOOTSTRAP_SEED = 45020261023
SPLITS = dict(confirm_four=('four', 200), confirm_none=('none', 100),
              guard_four=('four', 100), guard_none=('none', 100),
              timing_four=('four', 30), warmup=('four', 1))


def locked_read(path):
    """One atomic commit contains the value; the readable JSON is a projection."""
    committed = o.read_json(path.with_suffix('.lock.json'))
    value = committed['value']
    if committed['sha256'] != o.json_hash(value):
        raise ValueError('Locked artifact digest changed')
    if not path.exists():
        o.write_json(path, value)  # Recover interruption between commit and projection.
    elif o.json_hash(o.read_json(path)) != committed['sha256']:
        raise ValueError('Locked artifact projection changed')
    return value


def lock_json(path, value):
    canonical = json.loads(json.dumps(value))
    if path.with_suffix('.lock.json').exists():
        if locked_read(path) != canonical:
            raise ValueError('Locked artifact changed')
    else:
        if path.exists() and o.json_hash(o.read_json(path)) != o.json_hash(canonical):
            raise ValueError('Uncommitted artifact differs')
        o.write_json(path.with_suffix('.lock.json'), dict(sha256=o.json_hash(canonical), value=canonical))
        o.write_json(path, canonical)
    return canonical


def config(args):
    return dict(smoke=args.smoke, search_budget=3 if args.smoke else 1000,
                scenario_namespace=2026102301 if args.smoke else 2026102302,
                search_only_budget=3 if args.smoke else 3000, search_depth=100,
                beta_policy=.5, beta_values=dict(base=.1, novalue=0.),
                splits={k: [layout, 1 if args.smoke else n]
                        for k, (layout, n) in SPLITS.items()},
                seeds=list(SEEDS), opponents=list(o.OPPONENTS),
                bootstrap_draws=20000, bootstrap_seed=BOOTSTRAP_SEED,
                minimum_gain=.05, illegal_margin=.02, regression_margin=.03,
                simultaneous_method='max_centered_cluster_bootstrap_one_sided_95',
                latency_hard_limit=None, single_checkpoint_adoption=True,
                no_training=True, continue_formal=False,
                evader_deployment_config='base')


def settings(args, cfg='base'):
    c = config(args)
    if cfg not in CONFIGS:
        raise ValueError('Unknown frozen search configuration')
    return dict(search_budget=c['search_budget'], search_only_budget=c['search_only_budget'],
                search_depth=100, beta_policy=.5, beta_value=c['beta_values'][cfg])


def source_registry():
    """Verify committed checkpoints, not stale progress/dashboard files."""
    args = followup.parse_args([])
    followup.verify(args)
    if o.read_json(SOURCE / 'status.json')['phase'] != 'complete':
        raise ValueError('Followup experiment must be complete')
    parent = previous.source_contract(followup.SOURCE)
    registry = []
    for item in parent['registry']:
        if item['label'] != 'baseline' and not item['label'].endswith('_final'):
            continue
        seed = item['seed']
        label = 'baseline' if seed is None else f'old{seed}'
        registry.append(dict(label=label, family='l45' if seed is None else 'old',
                             seed=seed, episode=item['episode'], source_models=item['models'],
                             checkpoint=item['checkpoint'],
                             checkpoint_sha256=item.get('checkpoint_sha256'),
                             model_hash=item['model_hash']))
    locked = followup.locked_read(SOURCE / 'reward_selection.json')
    for seed in SEEDS:
        directory = SOURCE / f'reward/A/seed{seed}'
        selected = o.read_json(directory / 'selection.json')
        committed = followup.load_checkpoint(args, directory)
        if selected['best_episode'] != 512 or selected['current_episode'] != 512 or \
                committed['episode'] != 512 or committed['selection']['best_episode'] != 512:
            raise ValueError('A checkpoint is not the previously locked episode 512')
        if any(selected[k] != v for k, v in committed['selection'].items()):
            raise ValueError('A selection differs from committed checkpoint')
        models = Path(selected['models'])
        if models.resolve() != (directory / 'models_ep512').resolve():
            raise ValueError('Unexpected A export path')
        digest = o.model_fingerprint(models)
        frozen = next(m for m in locked['models'] if m['label'] == f'A{seed}')
        if digest != frozen['model_hash'] or str(models) != frozen['models']:
            raise ValueError('A model differs from locked reward selection')
        for robot, name in enumerate(o.MODEL_NAMES[:4]):
            actual, expected = o.load_torch(models / name), committed['actors'][robot]
            if actual.keys() != expected.keys() or any(
                    not torch.equal(value, expected[key]) for key, value in actual.items()):
                raise ValueError('A model is not the committed actor')
        pointer = o.read_json(directory / 'latest.json')
        registry.append(dict(label=f'A{seed}', family='A', seed=seed, episode=512,
                             source_models=str(models), checkpoint=pointer['path'],
                             checkpoint_sha256=pointer['sha256'], model_hash=digest))
    if [m['label'] for m in registry] != ['baseline', 'old0', 'old1', 'old2', 'A0', 'A1', 'A2']:
        raise ValueError('Unexpected seven-model registry')
    baseline = registry[0]['model_hash']
    for item in registry:
        if any(item['model_hash'][name] != baseline[name]
               for name in (*o.MODEL_NAMES[:2], o.MODEL_NAMES[-1])):
            raise ValueError('Frozen evader actors or old search value changed')
    return registry


def additional_history_files(include_smoke=False):
    paths = [STARTUP_ARCHIVE / name / 'scenarios.json' for name in ('formal', 'smoke')]
    if include_smoke:
        paths.append(SMOKE_OUTPUT / 'scenarios.json')
    return [path for path in paths if path.exists()]


def history_scenes(include_smoke=False):
    used = followup.history_scenes()
    used.update(previous.physical_hash(s)
                for cases in o.read_json(SOURCE / 'scenarios.json').values() for s in cases)
    # Include recoverable actual rollout specs as well as all reserved banks.
    for path in sorted((SOURCE / 'reward').glob('*/seed*/rollouts/*.json')):
        used.update(previous.physical_hash(row['spec']) for row in o.read_json(path))
    for path in additional_history_files(include_smoke):
        used.update(previous.physical_hash(s) for cases in o.read_json(path).values() for s in cases)
    return used


def record(args):
    registry = source_registry()
    return dict(schema=1, config=config(args), registry=registry,
                source={str(p): o.sha(p) for p in (
                    SOURCE / 'experiment.json', SOURCE / 'signature.json',
                    SOURCE / 'results.json', SOURCE / 'scenarios.json',
                    SOURCE / 'reward_selection.json', SOURCE / 'reward_selection.lock.json')},
                additional_history={str(p): o.sha(p) for p in additional_history_files(not args.smoke)},
                implementation=dict(runner=o.sha(__file__), followup=o.sha(followup.__file__),
                                    previous=o.sha(previous.__file__), original=o.implementation_files()))


def verify(args):
    out = args.output
    current = record(args)
    if o.read_json(out / 'experiment.json') != current:
        raise ValueError('Source/configuration/implementation changed; refuse mixed resume')
    signature = o.json_hash(dict(record=current, scenarios=o.sha(out / 'scenarios.json')))
    if o.read_json(out / 'signature.json')['sha256'] != signature:
        raise ValueError('Frozen scenario fingerprint changed')
    for item in current['registry']:
        if o.model_fingerprint(out / 'models' / item['label']) != item['model_hash']:
            raise ValueError('Frozen model snapshot changed')
    return current


def prepare(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    current = record(args)
    if (out / 'experiment.json').exists():
        if o.read_json(out / 'experiment.json') != current:
            raise ValueError('Experiment fingerprint changed')
    else:
        for item in current['registry']:
            target = out / 'models' / item['label']
            target.mkdir(parents=True, exist_ok=True)
            for name in o.MODEL_NAMES:
                shutil.copyfile(Path(item['source_models']) / name, target / name)
            if o.model_fingerprint(target) != item['model_hash']:
                raise ValueError('Copied checkpoint fingerprint mismatch')
        o.write_json(out / 'experiment.json', current)
    used = history_scenes(not args.smoke)
    if not (out / 'scenarios.json').exists():
        initial_count = len(used)
        banks = {}
        for phase, (name, (layout, count)) in enumerate(config(args)['splits'].items(), 750):
            cases, index = [], 0
            while len(cases) < count:
                spec = o.make_scenario(o.scenario_seed(phase, config(args)['scenario_namespace'], index), layout)
                index += 1
                identity = previous.physical_hash(spec)
                if identity in used:
                    continue
                used.add(identity)
                cases.append(spec)
            banks[name] = cases
        o.write_json(out / 'scenarios.json', banks)
        o.write_json(out / 'scenario_independence.json', dict(
            previous_physical_scenes=initial_count, new_scenes=len(used)-initial_count,
            overlap=0, includes_followup_banks_and_actual_rollouts=True,
            limitation='Deduplicates recorded l45/PPO episode initial states; not all NTE pretraining states.'))
    banks = o.read_json(out / 'scenarios.json')
    if set(banks) != set(SPLITS):
        raise ValueError('Frozen scenario splits changed')
    identities = [previous.physical_hash(s) for cases in banks.values() for s in cases]
    if len(set(identities)) != len(identities) or history_scenes(not args.smoke).intersection(identities):
        raise ValueError('Physical scenario leakage')
    for name, (layout, count) in config(args)['splits'].items():
        if len(banks[name]) != count or any(s['layout'] != layout for s in banks[name]):
            raise ValueError('Scenario count/layout changed')
        for spec in banks[name]:
            o.restore_scenario(spec)  # Validate the complete spec, including search seeds.
    signature = dict(sha256=o.json_hash(dict(record=current, scenarios=o.sha(out / 'scenarios.json'))))
    if (out / 'signature.json').exists() and o.read_json(out / 'signature.json') != signature:
        raise ValueError('Frozen scenarios changed')
    o.write_json(out / 'signature.json', signature)
    verify(args)
    return banks


def models(args):
    result = []
    for source in o.read_json(args.output / 'experiment.json')['registry']:
        for cfg in CONFIGS:
            item = followup.model(source['label'], args.output / 'models' / source['label'],
                                  source['seed'], cfg)
            item.update(uid=f"{source['label']}:{cfg}", family=source['family'],
                        source_checkpoint=source['checkpoint'], episode=source['episode'])
            result.append(item)
    return result


def requests(args, selected, scenes, role=1, opponents=o.OPPONENTS, timing='parallel'):
    result = []
    protocol = o.sha(args.output / 'experiment.json')
    for item in selected:
        for spec in scenes:
            for opponent in opponents:
                own = settings(args, item['config'] if role == 1 else 'base')
                identity = dict(protocol=protocol, policy_id=item['policy_id'],
                                episode_hash=spec['episode_hash'], own_settings=own,
                                opponent_settings=settings(args), mode='search', role=role,
                                opponent=opponent, timing_context=timing)
                key = o.json_hash(identity)
                task = dict(**identity, key=key, models=item['models'], model_hash=item['model_hash'],
                            baseline=str(args.output / 'models/baseline'), spec=spec)
                result.append(dict(key=key, uid=item['uid'], label=item['label'],
                                   config=item['config'], training_seed=item['seed'],
                                   family=item['family'], model_hash=item['model_hash'], task=task))
    return result


def evaluate_one(task):
    torch.set_num_threads(1)
    if o.model_fingerprint(task['models']) != task['model_hash']:
        raise ValueError('Evaluation model fingerprint changed')
    p, state = o.restore_scenario(task['spec'])
    role = task['role']
    own = followup.make_solver(p, task['models'], 'Search + Learning', role,
                               task['own_settings'], task['spec']['search_seeds'][role])
    other = followup.make_solver(p, task['baseline'], task['opponent'], 1-role,
                                 task['opponent_settings'], task['spec']['search_seeds'][1-role])
    team = o.TeamSolver(own if role == 0 else other, own if role == 1 else other)
    decisions = []

    class TimedTeam:
        def policy(self, problem, current):
            action = team.policy(problem, current)
            if task['timing_context'] in ('single_process', 'warmup'):
                decisions.append([team.last_pursuer_latency_ms, team.last_evader_latency_ms,
                                  team.last_joint_latency_ms])
            return action

        def __getattr__(self, name):
            return getattr(team, name)

    timing_machine = machine_snapshot() if task['timing_context'] == 'single_process' else None
    started = time.perf_counter()
    metric = o.run_episode(p, TimedTeam(), state)
    metric['episode_wall_ms'] = (time.perf_counter() - started) * 1000.
    illegal = metric['illegal_terminal']
    metric.update(legal_double_capture=int(metric['double_capture'] and not illegal),
                  legal_survival=int(metric['capture_count'] < 2 and metric['survival_time'] >= p.tf and not illegal),
                  one_capture_timeout=int(metric['capture_count'] == 1 and metric['survival_time'] >= p.tf and not illegal),
                  pursuer_illegal=int(metric['pursuer_obstacle_collision'] or metric['pursuer_out_of_bounds']),
                  evader_illegal=int(metric['evader_obstacle_collision'] or metric['evader_out_of_bounds']))
    metric['completion_cost'] = metric['capture_time'] if metric['legal_double_capture'] else float(p.tf)
    metric.update({k: task[k] for k in ('key', 'policy_id', 'mode', 'role', 'opponent', 'timing_context')})
    metric.update(episode_hash=task['spec']['episode_hash'], layout=task['spec']['layout'],
                  search_seeds=task['spec']['search_seeds'], actual_own=own.actual_settings,
                  actual_opponent=other.actual_settings)
    if decisions:
        metric['decision_latency_ms'] = decisions
    if timing_machine:
        metric['timing_machine'] = dict(before=timing_machine, after=machine_snapshot())
    return metric


def read_cached(args, item):
    stored = o.read_json(args.output / 'cache' / (item['key'] + '.json'))
    digest = stored.pop('content_sha256', None)
    if digest != o.json_hash(stored):
        raise ValueError('Evaluation cache content changed')
    task = item['task']
    if any(stored[k] != task[k] for k in ('key', 'policy_id', 'mode', 'role', 'opponent', 'timing_context')) or \
            stored['episode_hash'] != task['spec']['episode_hash'] or \
            stored['search_seeds'] != task['spec']['search_seeds']:
        raise ValueError('Cached evaluation identity mismatch')
    # Verify effective parameters as well as requested task identity.
    for field, method in (('actual_own', 'Search + Learning'), ('actual_opponent', task['opponent'])):
        actual = stored[field]
        requested = task['own_settings' if field == 'actual_own' else 'opponent_settings']
        if actual['method'] != method:
            raise ValueError('Cached opponent method mismatch')
        if method != 'Policy Only':
            expected_budget = requested['search_only_budget' if method == 'Search Only' else 'search_budget']
            expected_betas = [0., 0.] if method == 'Search Only' else [requested['beta_policy'], requested['beta_value']]
            if actual['budget'] != expected_budget or actual['depth'] != 100 or not np.allclose(
                    [actual['beta_policy'], actual['beta_value']], expected_betas, atol=1e-7, rtol=0):
                raise ValueError('Cached actual solver parameters mismatch')
    stored.update({k: v for k, v in item.items() if k not in ('key', 'task')})
    return stored


def run_evaluation(args, name, items, pool, status):
    folder = args.output / name
    manifest = [{k: v for k, v in item.items() if k != 'task'} for item in items]
    if len({(m['uid'], m['key']) for m in manifest}) != len(manifest):
        raise ValueError('Duplicate evaluation requests')
    if (folder / 'manifest.json').exists() and o.read_json(folder / 'manifest.json') != manifest:
        raise ValueError('Frozen evaluation manifest changed')
    o.write_json(folder / 'manifest.json', manifest)
    cache = args.output / 'cache'
    cache.mkdir(exist_ok=True)
    unique = {item['key']: item['task'] for item in items}
    missing = [task for key, task in unique.items() if not (cache / (key + '.json')).exists()]
    completed = len(unique) - len(missing)
    status(name, completed=completed, total=len(unique), label_requests=len(items))
    iterator = pool.imap_unordered(evaluate_one, missing, chunksize=1) if pool else map(evaluate_one, missing)
    for row in iterator:
        row['content_sha256'] = o.json_hash(row)
        o.write_json(cache / (row['key'] + '.json'), row)
        completed += 1
        status(name, completed=completed, total=len(unique), label_requests=len(items))
    rows = [read_cached(args, item) for item in items]
    summary = []
    for uid in sorted({r['uid'] for r in rows}):
        for layout in sorted({r['layout'] for r in rows}):
            for opponent in o.OPPONENTS:
                part = [r for r in rows if (r['uid'], r['layout'], r['opponent']) == (uid, layout, opponent)]
                if part:
                    summary.append(dict(uid=uid, layout=layout, opponent=opponent, **o.summarize(part)))
    o.write_json(folder / 'summary.json', summary)
    return rows


def comparison_specs():
    specs = []
    for label in ('baseline', 'old0', 'old1', 'old2', 'A0', 'A1', 'A2'):
        for cfg in CONFIGS:
            uid = f'{label}:{cfg}'
            if uid != 'baseline:base':
                specs.append(dict(id=f'{uid}-baseline:base', candidate=uid, reference='baseline:base'))
    for label in ('old0', 'old1', 'old2', 'A0', 'A1', 'A2'):
        specs.append(dict(id=f'{label}:novalue-baseline:novalue',
                          candidate=f'{label}:novalue', reference='baseline:novalue'))
    if len(specs) != 19:
        raise AssertionError('Exactly 19 locked comparisons are required')
    return specs


def indexed_rows(rows, uid, layout, role=1):
    result = {}
    for row in rows:
        if (row['uid'], row['layout'], row['role']) != (uid, layout, role):
            continue
        key = (row['episode_hash'], row['opponent'])
        if key in result:
            raise ValueError('Duplicate paired row')
        result[key] = row
    if not result:
        raise ValueError(f'Missing paired rows: {uid}/{layout}/{role}')
    scenes = {k[0] for k in result}
    if any({op for scene, op in result if scene == s} != set(o.OPPONENTS) for s in scenes):
        raise ValueError('Missing opponent within a physical scene')
    return result


def paired_statistics(rows, specs, layout='four', role=1, draws=20000, simultaneous=True):
    """Fixed-checkpoint inference: cluster physical scenes, never training seeds."""
    metric = 'legal_double_capture' if role == 1 else 'legal_survival'
    side = 'pursuer_illegal' if role == 1 else 'evader_illegal'
    indexed = {uid: indexed_rows(rows, uid, layout, role)
               for s in specs for uid in (s['candidate'], s['reference'])}
    keys = sorted(next(iter(indexed.values())))
    if any(set(values) != set(keys) for values in indexed.values()):
        raise ValueError('Missing/unpaired evaluation scenes')
    scenes = sorted({scene for scene, _ in keys})
    positions = [[i for i, (scene, _) in enumerate(keys) if scene == s] for s in scenes]
    differences, outputs = [], []
    for spec in specs:
        candidate, reference = indexed[spec['candidate']], indexed[spec['reference']]
        if any(candidate[k]['search_seeds'] != reference[k]['search_seeds'] for k in keys):
            raise ValueError('Unpaired search seeds')
        delta = np.array([candidate[k][metric] - reference[k][metric] for k in keys], float)
        differences.append([float(delta[idx].mean()) for idx in positions])
        outputs.append(dict(**spec, delta=float(delta.mean()), n_scenes=len(scenes),
                            n_games=len(keys), own_illegal_delta=float(np.mean([
                                candidate[k][side] - reference[k][side] for k in keys])),
                            gained=sum(candidate[k][metric] > reference[k][metric] for k in keys),
                            lost=sum(candidate[k][metric] < reference[k][metric] for k in keys),
                            per_opponent_delta={op: float(np.mean([
                                candidate[k][metric] - reference[k][metric] for k in keys if k[1] == op]))
                                for op in o.OPPONENTS}))
    differences = np.asarray(differences)
    observed = differences.mean(1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty((draws, len(specs)), dtype=float)
    for start in range(0, draws, 256):
        stop = min(start+256, draws)
        chosen = rng.integers(len(scenes), size=(stop-start, len(scenes)))
        samples[start:stop] = differences[:, chosen].mean(-1).T
    correction = max(0., float(np.quantile(np.max(samples-observed, axis=1), .95)))
    for i, result in enumerate(outputs):
        result['ci95'] = np.quantile(samples[:, i], [.025, .975]).tolist()
        if simultaneous:
            result['simultaneous_lower95'] = float(observed[i] - correction)
    return dict(comparisons=outputs, draws=draws, n_scenes=len(scenes),
                simultaneous_comparisons=len(specs) if simultaneous else 0,
                max_centered_correction=correction if simultaneous else None,
                note='Physical scenes jointly resampled across opponents and fixed checkpoints; conditional on original l45.')


def aggregate(rows, uid, layout='four', role=1):
    part = list(indexed_rows(rows, uid, layout, role).values())
    metric = 'legal_double_capture' if role == 1 else 'legal_survival'
    side = 'pursuer_illegal' if role == 1 else 'evader_illegal'
    return dict(games=len(part), success_rate=float(np.mean([r[metric] for r in part])),
                own_illegal_rate=float(np.mean([r[side] for r in part])),
                opponent_illegal_rate=float(np.mean([r['evader_illegal' if role == 1 else 'pursuer_illegal'] for r in part])),
                one_capture_timeout_rate=float(np.mean([r['one_capture_timeout'] for r in part])),
                completion_cost=float(np.mean([r['completion_cost'] for r in part])))


def family_statistics(rows):
    result = []
    for family in ('old', 'A'):
        for cfg in CONFIGS:
            for layout in ('four', 'none'):
                specs = [dict(id=str(seed), candidate=f'{family}{seed}:{cfg}', reference=f'baseline:{cfg}')
                         for seed in SEEDS]
                individual = paired_statistics(rows, specs, layout, simultaneous=False)['comparisons']
                result.append(dict(family=family, config=cfg, layout=layout,
                                   mean_success_rate=float(np.mean([aggregate(rows, s['candidate'], layout)['success_rate'] for s in specs])),
                                   mean_delta=float(np.mean([r['delta'] for r in individual])),
                                   per_seed_delta=[r['delta'] for r in individual],
                                   all_seeds_nonnegative=all(r['delta'] >= -1e-12 for r in individual),
                                   fixed_checkpoint_comparisons=individual,
                                   note='All three seeds reported; a confirmed individual checkpoint does not establish seed stability.'))
    return result


def confirmation_result(args, rows):
    specs = comparison_specs()
    four = paired_statistics(rows, specs)
    none = paired_statistics(rows, specs, 'none', simultaneous=False)
    by_four = {s['id']: s for s in four['comparisons']}
    by_none = {s['id']: s for s in none['comparisons']}
    candidates = []
    for item in models(args):
        if item['uid'] == 'baseline:base':
            continue
        key = f"{item['uid']}-baseline:base"
        f, n = by_four[key], by_none[key]
        checks = dict(minimum_gain=f['delta'] >= .05-1e-12,
                      simultaneous_lower_positive=f['simultaneous_lower95'] > 0.,
                      own_illegal=f['own_illegal_delta'] <= .02+1e-12,
                      no_obstacle_regression=n['delta'] >= -.03-1e-12,
                      no_obstacle_illegal=n['own_illegal_delta'] <= .02+1e-12)
        same_id = f"{item['uid']}-baseline:{item['config']}"
        same = by_four.get(same_id)
        candidates.append(dict(uid=item['uid'], label=item['label'], config=item['config'],
                               checks=checks, pursuit_pass=all(checks.values()),
                               four=aggregate(rows, item['uid']), none=aggregate(rows, item['uid'], 'none'),
                               baseline_comparison=f, no_obstacle_comparison=n,
                               ppo_increment=same if item['label'] != 'baseline' else None,
                               ppo_increment_confirmed=bool(item['label'] != 'baseline' and same and
                                                            same['simultaneous_lower95'] > 0.)))
    return dict(smoke=args.smoke, four=four, none=none,
                baseline=dict(four=aggregate(rows, 'baseline:base'), none=aggregate(rows, 'baseline:base', 'none')),
                candidates=candidates, eligible=[c['uid'] for c in candidates if c['pursuit_pass']],
                families=family_statistics(rows), selection='locked_candidates_simultaneous_inference')


def confirm(args, banks, pool, status):
    items = requests(args, models(args), banks['confirm_four'] + banks['confirm_none'])
    expected = 14 * 3 * (len(banks['confirm_four']) + len(banks['confirm_none']))
    if len(items) != expected:
        raise AssertionError('Incorrect 14-configuration confirmation matrix')
    rows = run_evaluation(args, 'confirmation', items, pool, status)
    result = confirmation_result(args, rows)
    lock_json(args.output / 'confirmation_result.json', result)
    stage_report(args, 'confirmation', f"追捕确认完成，{len(result['eligible'])} 个配置进入回归。")
    return result


def guard_models(args, pursuit):
    labels = {uid.split(':')[0] for uid in pursuit['eligible']} | {'baseline'}
    return [m for m in models(args) if m['config'] == 'base' and m['label'] in labels]


def guard(args, banks, pool, status):
    pursuit = locked_read(args.output / 'confirmation_result.json')
    if not pursuit['eligible']:
        raise ValueError('No eligible pursuit candidate for regression')
    selected = guard_models(args, pursuit)
    rows = run_evaluation(args, 'evader_regression', requests(
        args, selected, banks['guard_four'] + banks['guard_none'], role=0), pool, status)
    result = {}
    for item in selected:
        comparisons = {}
        for layout in ('four', 'none'):
            spec = [dict(id=item['label'], candidate=item['uid'], reference='baseline:base')]
            value = paired_statistics(rows, spec, layout, role=0, simultaneous=False)['comparisons'][0]
            value['passed'] = value['delta'] >= -.03-1e-12 and value['own_illegal_delta'] <= .02+1e-12
            comparisons[layout] = value
        result[item['label']] = dict(passed=all(v['passed'] for v in comparisons.values()), comparisons=comparisons)
    accepted = [uid for uid in pursuit['eligible'] if result[uid.split(':')[0]]['passed']]
    output = dict(models=result, accepted=accepted, own_evader_config='base')
    lock_json(args.output / 'guard_result.json', output)
    stage_report(args, 'guard', f"逃逸回归完成，{len(accepted)} 个配置通过；逃逸使用原配置。")
    return output


def machine_snapshot():
    return dict(unix=time.time(), loadavg=list(os.getloadavg()), cpu_count=os.cpu_count(),
                affinity=sorted(os.sched_getaffinity(0)), pid=os.getpid(),
                threads={name: os.environ.get(name) for name in
                         ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})


def timing(args, banks, status):
    guard_result = locked_read(args.output / 'guard_result.json')
    accepted = set(guard_result['accepted'])
    if not accepted:
        raise ValueError('No regression-passing candidate for serial timing')
    # A live spawn worker would contaminate this experiment's serial timing.
    children = Path(f'/proc/{os.getpid()}/task/{os.getpid()}/children').read_text().split()
    if any(b'spawn_main' in Path(f'/proc/{pid}/cmdline').read_bytes() for pid in children):
        raise RuntimeError('Parallel evaluation workers must exit before timing')
    selected = [m for m in models(args) if m['uid'] in accepted | {'baseline:base'}]
    before = machine_snapshot()
    warmups = []
    for item in requests(args, selected, banks['warmup'], opponents=('Policy Only',), timing='warmup'):
        status('warmup', uid=item['uid'])
        row = evaluate_one(item['task'])  # Warm again after interruption; never count as measured trials.
        warmups.append(dict(uid=item['uid'], episode_wall_ms=row['episode_wall_ms']))
    o.write_json(args.output / 'serial_timing/warmup.json', dict(machine=before, trials=warmups))
    # Interleave configurations within scenes, rotating the first model to limit order bias.
    items = []
    for index, spec in enumerate(banks['timing_four']):
        offset = index % len(selected)
        order = selected[offset:] + selected[:offset]
        items.extend(requests(args, order, [spec], opponents=('Policy Only',), timing='single_process'))
    rows = run_evaluation(args, 'serial_timing', items, None, status)
    result = {}
    for model in selected:
        part = [r for r in rows if r['uid'] == model['uid']]
        decisions = np.concatenate([np.asarray(r['decision_latency_ms']) for r in part], axis=0)
        if any(r['timing_context'] != 'single_process' for r in part):
            raise ValueError('Parallel latency included in serial timing')
        result[model['uid']] = dict(episodes=len(part), decisions=len(decisions),
            pursuer_mean_ms=float(decisions[:, 0].mean()), pursuer_p50_ms=float(np.median(decisions[:, 0])),
            pursuer_p95_ms=float(np.percentile(decisions[:, 0], 95)),
            joint_mean_ms=float(decisions[:, 2].mean()),
            episode_mean_ms=float(np.mean([r['episode_wall_ms'] for r in part])),
            episode_p50_ms=float(np.median([r['episode_wall_ms'] for r in part])),
            timing_session_pids=sorted({r['timing_machine']['before']['pid'] for r in part}),
            machine_before=before, machine_after=machine_snapshot(), timing_context='single_process')
    base = result['baseline:base']['pursuer_mean_ms']
    for value in result.values():
        value['pursuer_time_ratio_to_baseline'] = value['pursuer_mean_ms'] / base
    # Resuming reuses recorded timings, rather than replacing them with fresh measurements.
    path = args.output / 'timing_result.json'
    if path.exists():
        existing = locked_read(path)
        if {k: v['pursuer_mean_ms'] for k, v in existing.items()} != {k: v['pursuer_mean_ms'] for k, v in result.items()}:
            raise ValueError('Locked timing measurements changed')
        result = existing
    else:
        lock_json(path, result)
    stage_report(args, 'timing', '单进程计时完成；耗时不作为成功率提升的否决门槛。')
    return result


def selection_key(candidate, timings):
    label = candidate['label']
    family_order = 0 if label == 'baseline' else (1 if label.startswith('A') else 2)
    seed = -1 if label == 'baseline' else int(label[-1])
    return (-candidate['four']['success_rate'], candidate['four']['own_illegal_rate'],
            candidate['four']['completion_cost'], timings[candidate['uid']]['pursuer_mean_ms'],
            family_order, seed, CONFIGS.index(candidate['config']))


def scientific_digest(row):
    excluded = ('key', 'uid', 'label', 'config', 'training_seed', 'family', 'model_hash',
                'timing_context', 'episode_wall_ms', 'content_sha256', 'timing_machine')
    return o.json_hash({k: v for k, v in row.items() if k not in excluded and 'latency' not in k})


def read_recommended(args):
    value = locked_read(args.output / 'recommended.json')
    directory = Path(value['models'])
    if directory.resolve() != (args.output / 'recommended/models').resolve() or \
            o.model_fingerprint(directory) != value['model_hash']:
        raise ValueError('Recommended model fingerprint changed')
    if value['pursuer_settings'] != settings(args, value['config']) or value['evader_settings'] != settings(args):
        raise ValueError('Recommended role configuration changed')
    if value['experiment_sha256'] != o.sha(args.output / 'experiment.json'):
        raise ValueError('Recommendation belongs to another experiment')
    source = next((m for m in models(args) if m['uid'] == value['uid']), None)
    if source is None or source['model_hash'] != value['model_hash']:
        raise ValueError('Recommendation is not a frozen candidate')
    return value


def load_recommended_solver(args, problem, role, search_seed):
    value = read_recommended(args)
    requested = value['pursuer_settings' if role == 1 else 'evader_settings']
    return followup.make_solver(problem, value['models'], 'Search + Learning', role, requested, search_seed)


def replay(args, banks):
    value = read_recommended(args)
    source = next(m for m in models(args) if m['uid'] == value['uid'])
    exported = dict(source, models=value['models'])
    spec = banks['confirm_four'][0]
    expected_item = requests(args, [source], [spec], opponents=('Policy Only',))[0]
    expected = read_cached(args, expected_item)
    task = requests(args, [exported], [spec], opponents=('Policy Only',), timing='replay')[0]['task']
    actual = evaluate_one(task)
    if scientific_digest(actual) != scientific_digest(expected):
        raise ValueError('Exported recommendation did not reproduce original search episode')
    result = dict(uid=value['uid'], episode_hash=spec['episode_hash'], opponent='Policy Only',
                  scientific_result_exact_match=True, replay_not_added_to_confirmation=True,
                  metrics={key: actual[key] for key in ('legal_double_capture', 'one_capture_timeout', 'capture_count')})
    o.write_json(args.output / 'replay.json', result)
    return result


def finish(args, banks):
    verify(args)
    pursuit = locked_read(args.output / 'confirmation_result.json')
    accepted, timings = [], {}
    if pursuit['eligible']:
        regression = locked_read(args.output / 'guard_result.json')
        accepted = regression['accepted']
        if accepted:
            timings = locked_read(args.output / 'timing_result.json')
            if set(timings) != set(accepted) | {'baseline:base'}:
                raise ValueError('Missing serial timings for accepted candidates')
    candidates = [c for c in pursuit['candidates'] if c['uid'] in accepted]
    selected = min(candidates, key=lambda c: selection_key(c, timings)) if candidates else None
    uid = selected['uid'] if selected else 'baseline:base'
    item = next(m for m in models(args) if m['uid'] == uid)
    target = args.output / 'recommended/models'
    if target.exists():
        if o.model_fingerprint(target) != item['model_hash']:
            raise ValueError('Existing recommended export differs; refuse overwrite')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.parent / '.models.pending'
        if pending.exists():
            shutil.rmtree(pending)
        pending.mkdir()
        for name in o.MODEL_NAMES:
            shutil.copyfile(Path(item['models']) / name, pending / name)
        if o.model_fingerprint(pending) != item['model_hash']:
            raise ValueError('Recommended export fingerprint mismatch')
        pending.replace(target)
    value = dict(schema='l45_search_recommendation_v1', uid=uid, label=item['label'],
                 config=item['config'], models=str(target), model_hash=item['model_hash'],
                 source_models=item['models'], source_checkpoint=item['source_checkpoint'], episode=item['episode'],
                 pursuer_settings=settings(args, item['config']), evader_settings=settings(args),
                 confirmed_improvement=bool(selected and not args.smoke), smoke=args.smoke,
                 selected_metrics=selected, timing=timings.get(uid),
                 baseline_timing=timings.get('baseline:base'),
                 experiment_sha256=o.sha(args.output / 'experiment.json'),
                 no_training=True, continue_formal=False)
    lock_json(args.output / 'recommended.json', value)
    reproduced = replay(args, banks)
    reason = 'quality_confirmed' if selected else ('evader_regression_failed' if pursuit['eligible'] else 'no_candidate_confirmed')
    result = dict(reason=reason, smoke=args.smoke, quality_gate_passed=value['confirmed_improvement'],
                  recommended=uid, baseline=pursuit['baseline'], selected=selected,
                  eligible=pursuit['eligible'], regression_accepted=accepted,
                  timing=timings, replay=reproduced, all_seed_results=pursuit['families'],
                  no_training=True, continue_formal=False, original_l45_overwritten=False)
    lock_json(args.output / 'results.json', result)
    report(args, result)
    stage_report(args, 'complete', f"本轮结束：{reason}；推荐 {uid}，正式提升确认：{value['confirmed_improvement']}。")
    return result


def report(args, result):
    pursuit = locked_read(args.output / 'confirmation_result.json')
    base = pursuit['baseline']['four']['success_rate']
    lines = ['# 现成模型与价值配置独立确认', '',
             f"功能 smoke：{args.smoke}；结束原因：{result['reason']}；推荐：{result['recommended']}。",
             f"正式提升确认：{result['quality_gate_passed']}。原 l45 未覆盖，无新增训练。", '',
             f'原 l45 四障碍合法双捕率：{base:.2%}。下面均为同场景配对比较。', '',
             '| 配置 | 四障碍双捕 | 相对原 baseline | 同时95%下界 | 无障碍双捕 | 追捕门槛 |',
             '|---|---:|---:|---:|---:|---|']
    for c in pursuit['candidates']:
        comparison = c['baseline_comparison']
        lines.append(f"| {c['uid']} | {c['four']['success_rate']:.2%} | {comparison['delta']*100:+.2f} pp | "
                     f"{comparison['simultaneous_lower95']*100:+.2f} pp | {c['none']['success_rate']:.2%} | {c['pursuit_pass']} |")
    lines += ['', '采用规则允许一个独立确认有效的检查点；所有种子均完整报告，不能据此声称 PPO 跨种子稳定。',
              '19 项比较共同按物理场景 bootstrap；三个对手与全部固定模型共同抽样。',
              '关闭旧价值的配置收益与 PPO 相对同配置 l45 的增益分开报告。', '']
    selected = result['selected']
    if selected:
        lines += [f"最终采用 {selected['uid']}，相对主 baseline 提高 {selected['baseline_comparison']['delta']*100:.2f} 个百分点。"]
        if selected['ppo_increment']:
            increment = selected['ppo_increment']
            lines += [f"PPO 相对同配置 l45 的增益 {increment['delta']*100:+.2f} 个百分点；"
                      f"额外收益确认：{selected['ppo_increment_confirmed']}。"]
        else:
            lines += ['收益来自搜索价值配置变化，不属于 PPO 收益。']
        t = result['timing'][selected['uid']]
        lines += [f"单进程追捕平均 {t['pursuer_mean_ms']:.2f} ms/步，P50 {t['pursuer_p50_ms']:.2f}，"
                  f"P95 {t['pursuer_p95_ms']:.2f}；相对原配置耗时 {t['pursuer_time_ratio_to_baseline']:.3f} 倍。"]
    else:
        lines += ['没有候选完成全部验收，继续使用原 l45＋原配置；未自动追加测试或训练。']
    lines += ['', '首次捕获后超时、双方非法、对手分组及三种子结果见 confirmation/summary.json、confirmation_result.json。',
              '逃逸仍使用原配置；回归只检查候选完整模型的内部预测影响。',
              '计时只来自 serial_timing，机器负载随结果保存；不根据并行吞吐宣称部署加速。',
              '模型及角色配置见 recommended.json；导出原搜索加载重放检查通过。',
              '本轮结束，不自动进入第二捕训练。', '']
    (args.output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def stage_report(args, name, text):
    target = args.output / 'phase_reports' / (name + '.md')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f'# 价值配置确认：{name}\n\n{text}\n', encoding='utf-8')
    if args.output == DEFAULT_OUTPUT.resolve() and not args.smoke:
        main = ROOT / 'current/search_rl_diagnosis_20261022/REPORT.md'
        marker = f'<!-- l45-value-confirm-{name} -->'
        if marker not in main.read_text():
            with main.open('a') as stream:
                stream.write(f'\n{marker}\n\n### 现成模型确认：{name}\n\n{text}\n\n'
                             f'[阶段报告]({os.path.relpath(target, main.parent)})\n')


def audit(args, banks):
    verify(args)
    spec = banks['confirm_four'][0]
    p, state = o.restore_scenario(spec)
    selected = {m['uid']: m for m in models(args)}
    parameters = {}
    for cfg in CONFIGS:
        solver = followup.make_solver(p, selected[f'baseline:{cfg}']['models'], 'Search + Learning', 1,
                                      settings(args, cfg), spec['search_seeds'][1])
        parameters[cfg] = solver.actual_settings
    own = followup.make_solver(p, selected['baseline:base']['models'], 'Search + Learning', 1,
                               settings(args), spec['search_seeds'][1])
    historical = o.make_solver(p, selected['baseline:base']['models'], 'Search + Learning', 1,
                               settings(args), spec['search_seeds'][1])
    if not np.array_equal(own.policy(p, state), historical.policy(p, state)):
        raise ValueError('Original search configuration failed reproduction')
    expected = comparison_specs()
    lock_json(args.output / 'comparisons.json', expected)
    result = dict(original_configuration_exact_match=True, effective_parameters=parameters,
                  comparisons=len(expected), verified_models=7, no_training=True)
    lock_json(args.output / 'audit.json', result)
    stage_report(args, 'audit', '七套模型、C++ 参数、原配置复现及 19 项预设比较核验通过。')
    return result


def live_pids(output):
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            if (entry / 'stat').read_text().rsplit(')', 1)[1].split()[0] in ('Z', 'X'):
                continue
            argv = (entry / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if not argv or not Path(argv[0]).name.startswith('python'):
                continue
            cwd = (entry / 'cwd').resolve()
            indices = [i for i, arg in enumerate(argv) if arg.endswith(Path(__file__).name) and
                       (Path(arg) if Path(arg).is_absolute() else cwd / arg).resolve() == Path(__file__).resolve()]
            if not indices or '-c' in argv[:indices[0]] or '-m' in argv[:indices[0]]:
                continue
            stage = next((arg.split('=', 1)[1] for arg in argv if arg.startswith('--stage=')), None)
            if '--stage' in argv:
                stage = argv[argv.index('--stage')+1]
            if stage == 'watch':
                continue
            target = DEFAULT_OUTPUT
            if '--output' in argv:
                target = Path(argv[argv.index('--output')+1])
            for arg in argv:
                if arg.startswith('--output='):
                    target = Path(arg.split('=', 1)[1])
            if (target if target.is_absolute() else cwd / target).resolve() == output.resolve():
                found.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            pass
    return sorted(found)


def dashboard(args):
    out = args.output
    state = o.read_json(out / 'status.json') if (out / 'status.json').exists() else {}
    pids = live_pids(out)
    complete = state.get('phase') == 'complete' and (out / 'results.json').exists() and (out / 'recommended.json').exists()
    lines = ['l45 现成模型独立确认（只评测，不训练）',
             f"状态：{'已完成' if complete else ('运行中 '+str(pids) if pids else '未运行／进度已保存')}",
             f"阶段：{state.get('phase', '未启动')}；worker 配置：{state.get('workers', '--')}",
             f'输出：{out}']
    if pids:
        try:
            children = Path(f'/proc/{pids[0]}/task/{pids[0]}/children').read_text().split()
            count = sum(b'spawn_main' in Path(f'/proc/{pid}/cmdline').read_bytes() for pid in children)
            lines.append(f'实际并行 worker：{count}')
        except OSError:
            lines.append('实际 worker：进程切换中')
    for name in ('confirmation', 'evader_regression', 'serial_timing'):
        manifest = out / name / 'manifest.json'
        if manifest.exists():
            keys = {item['key'] for item in o.read_json(manifest)}
            completed = sum((out / 'cache' / (key+'.json')).exists() for key in keys)
            lines.append(f'{name}：{completed}/{len(keys)}')
    if pids and state.get('rate', 0) > 0 and state.get('total', 0) > state.get('completed', 0):
        eta = (state['total']-state['completed']) / state['rate'] / 60
        lines.append(f"当前阶段 {state['rate']:.3f} 局/秒，剩余约 {eta:.1f} 分钟（不含后续阶段）")
    if complete:
        result = o.read_json(out / 'results.json')
        lines.append(f"推荐：{result['recommended']}；正式提升确认：{result['quality_gate_passed']}")
    lines.append('并行吞吐只用于 ETA；部署耗时单进程测量。关闭看板不会停止任务。')
    return '\n'.join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=80)
    parser.add_argument('--stage', choices=('pipeline', 'prepare', 'audit', 'confirm', 'guard',
                                          'timing', 'report', 'watch', 'replay'), default='pipeline')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.workers < 1:
        parser.error('workers must be positive')
    if args.smoke and args.output == DEFAULT_OUTPUT.resolve():
        parser.error('Smoke needs an isolated output')
    for source in (SOURCE, followup.SOURCE, followup.FINAL_SOURCE, o.SOURCE_MODELS):
        source = source.resolve()
        if args.output == source or args.output.is_relative_to(source) or source.is_relative_to(args.output):
            parser.error('Output overlaps a protected historical source')
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(1)
    if args.stage == 'watch':
        try:
            while True:
                print(('' if args.once else '\033[2J\033[H') + dashboard(args), flush=True)
                if args.once:
                    return
                time.sleep(3)
        except KeyboardInterrupt:
            return
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'run.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError('Another runner owns this output') from error
    latest, clocks = {}, {}
    state_lock = threading.RLock()
    stopped = threading.Event()

    def status(phase, **fields):
        if 'completed' in fields:
            now = time.monotonic()
            begin, initial = clocks.setdefault(phase, (now, fields['completed']))
            fields['rate'] = (fields['completed']-initial) / max(now-begin, 1e-9)
        value = dict(phase=phase, pid=os.getpid(), workers=args.workers, updated_unix=time.time(), **fields)
        with state_lock:
            latest.clear()
            latest.update(value)
            o.write_json(args.output / 'status.json', value)
        print(o.json_hash(value)[:8] + ' ' + str(value), flush=True)

    def heartbeat():
        while not stopped.wait(5):
            with state_lock:
                if latest:
                    latest['heartbeat_unix'] = time.time()
                    o.write_json(args.output / 'status.json', latest)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        status('checking')
        banks = prepare(args)
        if args.stage == 'prepare':
            status('prepared')
            return
        if args.stage == 'replay':
            status('replay')
            result = replay(args, banks)
            status('complete', **result)
            return
        if args.stage == 'report':
            result = finish(args, banks)
            status('complete', reason=result['reason'], quality_gate_passed=result['quality_gate_passed'])
            return
        status('audit')
        audit(args, banks)
        if args.stage == 'audit':
            status('audited')
            return
        if args.stage in ('pipeline', 'confirm', 'guard'):
            with mp.get_context('spawn').Pool(args.workers) as pool:
                if args.stage in ('pipeline', 'confirm'):
                    pursuit = confirm(args, banks, pool, status)
                else:
                    pursuit = locked_read(args.output / 'confirmation_result.json')
                if (args.stage == 'pipeline' and pursuit['eligible']) or args.stage == 'guard':
                    guard(args, banks, pool, status)
            # Pool is fully gone before any deployment timing.
        if args.stage in ('pipeline', 'timing'):
            pursuit = locked_read(args.output / 'confirmation_result.json')
            if pursuit['eligible']:
                regression = locked_read(args.output / 'guard_result.json')
                if regression['accepted']:
                    timing(args, banks, status)
        if args.stage == 'pipeline':
            status('summarizing')
            result = finish(args, banks)
            status('complete', reason=result['reason'], quality_gate_passed=result['quality_gate_passed'])
        else:
            status('stage_complete', stage=args.stage)
    except KeyboardInterrupt:
        status('interrupted')
        raise
    except Exception as error:
        status('failed', error=repr(error))
        raise
    finally:
        stopped.set()
        thread.join(timeout=1)
        lock.close()


if __name__ == '__main__':
    main()
