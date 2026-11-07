"""Paired continuation from old0:novalue; only the frozen opponent mix changes.

Historical runners are read-only libraries. This module never calls their main
or pipeline functions. Every scientific choice is frozen before sampling.
"""
import os
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_var] = '1'

import argparse
import copy
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import threading
import time

import numpy as np
import torch

import confirm_l45_value_off as v

o, f, previous = v.o, v.followup, v.previous
ROOT = o.ROOT
SOURCE = v.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT / 'current/l45_strong_opponent_20261023'
SMOKE_OUTPUT = ROOT / 'current/l45_strong_opponent_smoke_20261023'
ARMS, SEEDS = ('A', 'B'), (0, 1, 2)
CONTROL, BASELINE = 'old0:novalue', 'baseline:novalue'
SPLITS = dict(validation_four=('four', 60), validation_none=('none', 30),
              confirm_four=('four', 200), confirm_none=('none', 100),
              guard_four=('four', 100), guard_none=('none', 100),
              timing_four=('four', 30), warmup=('four', 1))
lock_json, locked_read = v.lock_json, v.locked_read
paired_statistics, aggregate = v.paired_statistics, v.aggregate
evaluate_one, read_cached = v.evaluate_one, v.read_cached
scientific_digest = v.scientific_digest


def config(args):
    return dict(schema=1, smoke=args.smoke, episodes=64 if args.smoke else 512,
                batch=64, eval_every=64 if args.smoke else 256, seeds=list(SEEDS),
                source_episode=2048, max_training_episodes=384 if args.smoke else 3072,
                scenario_namespace=4502026102304 if args.smoke else 4502026102305,
                search_budget=3 if args.smoke else 1000,
                search_only_budget=3 if args.smoke else 3000, search_depth=100,
                pursuer_betas=[.5, 0.], evader_betas=[.5, .1],
                mixes=dict(A=[32, 32, 0], B=[16, 16, 32]),
                lr=3e-5, critic_lr=3e-4, ppo_epochs=4, minibatch=256,
                clip=.1, gamma=1., gae_lambda=.95, entropy=.001,
                target_kl=.02, gradient_clip=.5,
                splits={k: [layout, 1 if args.smoke else n] for k, (layout, n) in SPLITS.items()},
                minimum_gain=.03, illegal_margin=.02, regression_margin=.03,
                bootstrap_draws=20000, bootstrap_seed=45020261023,
                simultaneous_method='max_centered_cluster_bootstrap_one_sided_95',
                comparison_specs=comparison_specs(), automatic_extra_training=False)


def settings(args, role=1):
    c = config(args)
    return dict(search_budget=c['search_budget'], search_only_budget=c['search_only_budget'],
                search_depth=100, beta_policy=.5, beta_value=0. if role == 1 else .1)


def comparison_specs():
    specs = [dict(id=f'{arm}{seed}-{ref}', candidate=f'{arm}{seed}', reference=ref)
             for ref in (CONTROL, BASELINE) for arm in ARMS for seed in SEEDS]
    specs += [dict(id=f'B{seed}-A{seed}', candidate=f'B{seed}', reference=f'A{seed}') for seed in SEEDS]
    assert len(specs) == 15
    return specs


def history_files(args):
    """All recoverable l45/PPO banks and actual rollout specs, including archives."""
    paths = set()
    for root in (ROOT / 'current').glob('l45*'):
        if not root.is_dir() or root.resolve() == args.output.resolve():
            continue
        paths.update(root.rglob('scenarios.json'))
        paths.update(root.glob('**/rollouts/*.json'))
    return sorted(p for p in paths if p.is_file() and not p.is_relative_to(args.output))


def history_scenes(paths):
    used = set()
    for path in paths:
        data = o.read_json(path)
        specs = ([s for group in data.values() for s in group] if isinstance(data, dict)
                 else [row['spec'] for row in data])
        for spec in specs:
            o.restore_scenario(spec)
            used.add(previous.physical_hash(spec))
    return used


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_tree(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    return left == right


def source_record():
    args = v.parse_args([])
    v.verify(args)
    rec = v.read_recommended(args)
    if rec['uid'] != CONTROL or rec['episode'] != 2048 or rec['smoke']:
        raise ValueError('Unexpected current recommendation')
    initial = o.load_torch(rec['source_checkpoint'])
    if initial['episode'] != 2048 or any(k not in initial for k in
            ('critic', 'actor_optimizer', 'critic_optimizer', 'rng')):
        raise ValueError('Full episode-2048 continuation state is required')
    for robot, name in enumerate(o.MODEL_NAMES[:4]):
        if not equal_tree(initial['actors'][robot], o.load_torch(Path(rec['models']) / name)):
            raise ValueError('Recommended actors differ from full source checkpoint')
    for name, rate in (('actor_optimizer', 3e-5), ('critic_optimizer', 3e-4)):
        if initial[name]['param_groups'][0]['lr'] != rate or not initial[name]['state']:
            raise ValueError('Expected populated original Adam state')
    paths = [SOURCE / n for n in ('recommended.json', 'recommended.lock.json', 'experiment.json',
                                   'signature.json', 'results.json')]
    paths += [Path(rec['source_checkpoint'])]
    paths += [Path(rec['models']) / n for n in o.MODEL_NAMES]
    paths += [SOURCE / 'models/baseline' / n for n in o.MODEL_NAMES]
    return dict(recommendation=rec, fingerprints={str(p): o.sha(p) for p in paths})


def prepare(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'experiment.json'
    if path.with_suffix('.lock.json').exists():
        record = locked_read(path)
        verify(args)
        return o.read_json(out / 'scenarios.json')
    source = source_record()
    files = history_files(args)
    record = dict(config=config(args), source=source,
                  history={str(p): o.sha(p) for p in files},
                  implementation=dict(runner=o.sha(__file__), value_off=o.sha(v.__file__),
                                      followup=o.sha(f.__file__), previous=o.sha(previous.__file__),
                                      original=o.implementation_files()))
    rec = source['recommendation']
    for label, origin in (('current', Path(rec['models'])), ('baseline', SOURCE / 'models/baseline')):
        dest = out / 'models' / label
        dest.mkdir(parents=True, exist_ok=True)
        for name in o.MODEL_NAMES:
            shutil.copyfile(origin / name, dest / name)
    initial = out / 'initial/source.pt'
    initial.parent.mkdir(exist_ok=True)
    shutil.copyfile(rec['source_checkpoint'], initial)
    record['initial_sha256'] = o.sha(initial)
    record['controls'] = {label: o.model_fingerprint(out / 'models' / label) for label in ('current', 'baseline')}
    used = history_scenes(files)
    historical_count = len(used)
    banks = {}
    splits = {f'train{seed}': ['four', config(args)['episodes']] for seed in SEEDS}
    splits.update(config(args)['splits'])
    for phase, (name, (layout, count)) in enumerate(splits.items(), 910):
        cases, index = [], 0
        while len(cases) < count:
            spec = o.make_scenario(o.scenario_seed(phase, config(args)['scenario_namespace'], index), layout)
            index += 1
            identity = previous.physical_hash(spec)
            if identity not in used:
                used.add(identity)
                cases.append(spec)
        banks[name] = cases
    lock_json(out / 'scenarios.json', banks)
    lock_json(path, record)
    lock_json(out / 'signature.json', dict(sha256=o.json_hash(dict(record=record, scenarios=o.sha(out / 'scenarios.json')))))
    lock_json(out / 'scenario_independence.json', dict(
        previous_physical_scenes=historical_count, new_scenes=len(used)-historical_count,
        overlap=0, history_files=len(files), paired_A_B_training_is_only_shared_exception=True,
        scope='All recoverable current/l45*/**/scenarios.json and **/rollouts/*.json, including smoke, startup archives and actual rollouts; not all NTE pretraining states.'))
    verify(args)
    return banks


def verify(args):
    out = args.output
    record = locked_read(out / 'experiment.json')
    if record['config'] != config(args):
        raise ValueError('Protocol changed; refuse mixed resume')
    impl = dict(runner=o.sha(__file__), value_off=o.sha(v.__file__), followup=o.sha(f.__file__),
                previous=o.sha(previous.__file__), original=o.implementation_files())
    if record['implementation'] != impl:
        raise ValueError('Implementation changed; refuse mixed resume')
    for path, expected in (record['source']['fingerprints'] | record['history']).items():
        if o.sha(path) != expected:
            raise ValueError(f'Historical source changed: {path}')
    if o.sha(out / 'initial/source.pt') != record['initial_sha256']:
        raise ValueError('Full initial checkpoint changed')
    for label, digest in record['controls'].items():
        if o.model_fingerprint(out / 'models' / label) != digest:
            raise ValueError('Frozen control changed')
    locked_read(out / 'scenarios.json')
    expected = o.json_hash(dict(record=record, scenarios=o.sha(out / 'scenarios.json')))
    if locked_read(out / 'signature.json')['sha256'] != expected:
        raise ValueError('Scenario/protocol signature changed')
    return record


def model(uid, directory, seed=None, episode=0, arm=None, checkpoint=None):
    item = f.model(uid, directory, seed, cfg='novalue')
    item.update(uid=uid, family=arm or 'control', arm=arm, episode=episode,
                source_checkpoint=str(checkpoint) if checkpoint else None)
    if checkpoint:
        item['checkpoint_sha256'] = o.sha(checkpoint)
    return item


def controls(args):
    return [model(CONTROL, args.output / 'models/current'), model(BASELINE, args.output / 'models/baseline')]


def requests(args, selected, scenes, role=1, opponents=o.OPPONENTS, timing='parallel'):
    result = []
    protocol = o.sha(args.output / 'experiment.json')
    for item in selected:
        for spec in scenes:
            for opponent in opponents:
                identity = dict(protocol=protocol, policy_id=item['policy_id'], episode_hash=spec['episode_hash'],
                                own_settings=settings(args, role), opponent_settings=settings(args, 0),
                                mode='search', role=role, opponent=opponent, timing_context=timing)
                key = o.json_hash(identity)
                task = dict(**identity, key=key, models=item['models'], model_hash=item['model_hash'],
                            baseline=str(args.output / 'models/baseline'), spec=spec)
                result.append(dict(key=key, uid=item['uid'], label=item['label'], config='novalue',
                                   training_seed=item['seed'], family=item['family'],
                                   model_hash=item['model_hash'], task=task))
    return result


def run_evaluation(args, name, items, pool, status):
    folder = args.output / name
    manifest = [{k: value for k, value in item.items() if k != 'task'} for item in items]
    if len({(m['uid'], m['key']) for m in manifest}) != len(manifest):
        raise ValueError('Duplicate evaluation request')
    lock_json(folder / 'manifest.json', manifest)
    cache = args.output / 'cache'
    cache.mkdir(exist_ok=True)
    unique = {item['key']: item for item in items}
    for item in unique.values():
        if (cache / (item['key'] + '.json')).exists():
            read_cached(args, item)  # Fail before spending compute on corrupted caches.
    missing = [item['task'] for item in unique.values() if not (cache / (item['key'] + '.json')).exists()]
    completed = len(unique)-len(missing)
    status(name, completed=completed, total=len(unique), label_requests=len(items))
    iterator = pool.imap_unordered(evaluate_one, missing, chunksize=1) if pool else map(evaluate_one, missing)
    for row in iterator:
        row['content_sha256'] = o.json_hash(row)
        o.write_json(cache / (row['key'] + '.json'), row)
        completed += 1
        status(name, completed=completed, total=len(unique), label_requests=len(items))
    rows = [read_cached(args, item) for item in items]
    summaries = []
    for uid in sorted({r['uid'] for r in rows}):
        for layout in sorted({r['layout'] for r in rows}):
            for opponent in o.OPPONENTS:
                part = [r for r in rows if (r['uid'], r['layout'], r['opponent']) == (uid, layout, opponent)]
                if part:
                    summaries.append(dict(uid=uid, layout=layout, opponent=opponent, **o.summarize(part)))
    o.write_json(folder / 'summary.json', summaries)
    return rows


def point_comparison(rows, candidate, reference=CONTROL, layout='four', role=1):
    left = v.indexed_rows(rows, candidate, layout, role)
    right = v.indexed_rows(rows, reference, layout, role)
    if set(left) != set(right) or any(left[k]['search_seeds'] != right[k]['search_seeds'] for k in left):
        raise ValueError('Missing pairs or mismatched search seeds')
    metric = 'legal_double_capture' if role == 1 else 'legal_survival'
    side = 'pursuer_illegal' if role == 1 else 'evader_illegal'
    return dict(delta=float(np.mean([left[k][metric]-right[k][metric] for k in left])),
                own_illegal_delta=float(np.mean([left[k][side]-right[k][side] for k in left])),
                per_opponent_delta={op: float(np.mean([left[k][metric]-right[k][metric]
                                                       for k in left if k[1] == op])) for op in o.OPPONENTS})


def performance_checks(four, none, gain=False, confidence=False):
    checks = dict(own_illegal=four['own_illegal_delta'] <= .02+1e-12,
                  other_opponents=all(four['per_opponent_delta'][op] >= -.03-1e-12 for op in o.OPPONENTS[:2]),
                  no_obstacle_regression=none['delta'] >= -.03-1e-12)
    if gain:
        checks.update(minimum_gain=four['delta'] >= .03-1e-12,
                      search_only_improves=four['per_opponent_delta']['Search Only'] > 0.)
    if confidence:
        checks['simultaneous_lower_positive'] = four['simultaneous_lower95'] > 0.
    return checks


def opponent_schedule(args, seed, completed):
    """Common 16+16 positions, then replace the remaining 32 in B."""
    pa = ['Policy Only']*16 + ['Search + Learning']*16 + ['Policy Only']*16 + ['Search + Learning']*16
    pb = ['Policy Only']*16 + ['Search + Learning']*16 + ['Search Only']*32
    rng = np.random.default_rng(o.scenario_seed(930, config(args)['scenario_namespace'] % 2**32 + seed, completed))
    order = rng.permutation(64).tolist()
    return dict(A=[pa[i] for i in order], B=[pb[i] for i in order], common_positions=[i for i, j in enumerate(order) if j < 32])


def directory(args, arm, seed):
    return args.output / f'training/{arm}/seed{seed}'


def assert_frozen(args, states):
    for robot in o.TEAMS[0]:
        expected = o.load_torch(args.output / 'models/baseline' / o.MODEL_NAMES[robot])
        if not equal_tree(states[robot], expected):
            raise ValueError('Frozen evader actor changed')


def load_runtime(args, arm, seed):
    dest = directory(args, arm, seed)
    dest.mkdir(parents=True, exist_ok=True)
    initial = o.load_torch(args.output / 'initial/source.pt')
    if (dest / 'latest.json').exists():
        saved = f.load_checkpoint(args, dest)
        pointer = o.read_json(dest / 'latest.json')
        if pointer['pending_validation'] != saved['pending_validation'] or pointer['commit'] != saved['commit']:
            raise ValueError('Committed pending phase/counter changed')
        if saved['arm'] != arm or saved['seed'] != seed or saved['initial_sha'] != o.sha(args.output / 'initial/source.pt'):
            raise ValueError('Shared initialization or arm/seed changed')
    else:
        random.seed(o.scenario_seed(931, config(args)['scenario_namespace'] % 2**32, seed))
        np.random.seed(o.scenario_seed(932, config(args)['scenario_namespace'] % 2**32, seed))
        torch.manual_seed(o.scenario_seed(933, config(args)['scenario_namespace'] % 2**32, seed))
        saved = copy.deepcopy(initial)
        saved.update(episode=0, source_episode=2048, pending_validation=True, best_episode=0,
                     best_score=None, arm=arm, seed=seed, rng=o.rng_state(), commit=0,
                     initial_sha=o.sha(args.output / 'initial/source.pt'),
                     signature=o.json_hash(o.read_json(args.output / 'signature.json')),
                     experiment_sha=o.sha(args.output / 'experiment.json'))
        saved = f.commit_checkpoint(dest, saved, dict(records=[], best=None))
    assert_frozen(args, saved['actors'])
    actors = o.load_actors(o.original_problem(), states=saved['actors'])
    for robot in o.TEAMS[0]:
        for parameter in actors[robot].parameters():
            parameter.requires_grad_(False)
    critic = o.TrainingCritic()
    critic.load_state_dict(saved['critic'], strict=True)
    actor_opt = torch.optim.Adam([p for r in o.TEAMS[1] for p in actors[r].parameters()], lr=3e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    actor_opt.load_state_dict(saved['actor_optimizer'])
    critic_opt.load_state_dict(saved['critic_optimizer'])
    return dict(arm=arm, seed=seed, directory=dest, saved=saved, actors=actors, critic=critic,
                actor_opt=actor_opt, critic_opt=critic_opt)


def validate_runtime(args, banks, runtime, baseline_rows, pool, status):
    saved, dest = runtime['saved'], runtime['directory']
    if not saved['pending_validation']:
        return
    episode = saved['episode']
    exported = f.export_checkpoint(args, dest, episode, saved['actors'])
    pointer = o.read_json(dest / 'latest.json')
    descriptor = model(f"{runtime['arm']}{runtime['seed']}", exported, runtime['seed'], episode,
                       runtime['arm'], pointer['path'])
    rows = run_evaluation(args, f'training/{runtime["arm"]}/seed{runtime["seed"]}/validation_ep{episode}',
                          requests(args, [descriptor], banks['validation_four']+banks['validation_none']), pool, status)
    combined = rows + [r for r in baseline_rows if r['uid'] == CONTROL]
    four = point_comparison(combined, descriptor['uid'])
    none = point_comparison(combined, descriptor['uid'], layout='none')
    checks = performance_checks(four, none)
    entry = dict(model=descriptor, episode=episode, eligible=all(checks.values()), checks=checks,
                 four=aggregate(rows, descriptor['uid']), none=aggregate(rows, descriptor['uid'], 'none'),
                 comparison=four, none_comparison=none)
    selection = copy.deepcopy(saved['selection'])
    selection['records'].append(entry)
    eligible = [r for r in selection['records'] if r['eligible']]
    selection['best'] = min(eligible, key=lambda r: (-r['four']['success_rate'], r['four']['own_illegal_rate'],
                                                   r['four']['completion_cost'], r['episode']))
    saved.update(pending_validation=False, best_episode=selection['best']['episode'])
    runtime['saved'] = f.commit_checkpoint(dest, saved, selection)
    o.write_json(dest / 'selection.json', selection)


def sampling_tasks(args, banks, runtime):
    saved, dest = runtime['saved'], runtime['directory']
    completed, arm, seed = saved['episode'], runtime['arm'], runtime['seed']
    snapshot = dest / f'sampling/ep{completed:06d}.pt'
    payload = dict(actors=saved['actors'], episode=completed, arm=arm, seed=seed,
                   signature=saved['signature'])
    meta = snapshot.with_suffix('.json')
    if meta.with_suffix('.lock.json').exists():
        if locked_read(meta)['sha256'] != o.sha(snapshot) or not equal_tree(payload, o.load_torch(snapshot)):
            raise ValueError('Immutable sampling snapshot changed')
    else:
        o.save_torch(snapshot, payload)
        lock_json(meta, dict(sha256=o.sha(snapshot)))
    schedule = opponent_schedule(args, seed, completed)
    tasks = []
    for offset in range(64):
        index = completed+offset
        identity = dict(arm=arm, seed=seed, index=index, snapshot_episode=completed,
                        snapshot_sha256=o.sha(snapshot), episode_hash=banks[f'train{seed}'][index]['episode_hash'],
                        sampling_seed=o.scenario_seed(934, config(args)['scenario_namespace'] % 2**32 + seed, index),
                        opponent=schedule[arm][offset], settings=settings(args, 0),
                        baseline_hash=o.model_fingerprint(args.output / 'models/baseline'), signature=saved['signature'])
        key = o.json_hash(identity)
        tasks.append(dict(**identity, key=key, snapshot=str(snapshot),
                          baseline=str(args.output / 'models/baseline'), spec=banks[f'train{seed}'][index],
                          result=str(dest / f'rollout_cache/{key}.pt')))
    lock_json(dest / f'batches/{completed:06d}.json', dict(schedule=schedule, tasks=tasks))
    return tasks


def rollout(task):
    """Original reward and environment; paired streams, independent trajectories."""
    torch.set_num_threads(1)
    if o.sha(task['snapshot']) != task['snapshot_sha256'] or o.model_fingerprint(task['baseline']) != task['baseline_hash']:
        raise ValueError('Sampling source changed')
    checkpoint = o.cached_weights(task['snapshot'])
    if any(checkpoint[k] != task[k] for k in ('arm', 'seed', 'signature')) or checkpoint['episode'] != task['snapshot_episode']:
        raise ValueError('Cross-arm or stale sampling snapshot')
    p, state = o.restore_scenario(task['spec'])
    actors = o.load_actors(p, states=checkpoint['actors'])
    other = f.make_solver(p, task['baseline'], task['opponent'], 0, task['settings'], task['spec']['search_seeds'][0])
    torch.manual_seed(task['sampling_seed'])
    np.random.seed(task['sampling_seed'])
    random.seed(task['sampling_seed'])
    rows, rewards, capture_events = [], [], []
    captures, first, second, first_state = 0, None, None, None
    while not p.is_terminal(state) and len(rows) < int(p.tf / p.dt):
        obs = np.array([p.policy_encoding(state, r).reshape(-1) for r in o.TEAMS[1]], np.float32)
        limits = np.array([p.action_lims[p.action_idxs[r]] for r in o.TEAMS[1]], np.float32)
        active = np.array([p.is_active(state, r) for r in o.TEAMS[1]], np.float32)
        action = other.policy(p, state)
        raw = np.zeros((2, 2), np.float32)
        means, stds, logp = raw.copy(), np.ones_like(raw), 0.
        for local, robot in enumerate(o.TEAMS[1]):
            if active[local]:
                with torch.no_grad():
                    dist = o.distribution(actors[robot], torch.from_numpy(obs[local]), torch.from_numpy(limits[local]))
                    sampled = dist.sample()
                    logp += float(dist.log_prob(sampled).sum())
                raw[local], means[local], stds[local] = sampled.numpy(), dist.mean.numpy(), dist.stddev.numpy()
                action[p.action_idxs[robot], 0] = np.clip(raw[local], limits[local, :, 0], limits[local, :, 1])
        reward = p.reward(state, action)
        nxt = p.step(state, action, p.dt)
        team_reward = float(np.mean(reward[list(o.TEAMS[1]), 0]))
        rows.append(dict(obs=obs, limits=limits, raw=raw, active=active, old_mean=means, old_std=stds,
                         old_logp=logp, critic=o.critic_features(p, state), reward=team_reward,
                         done=bool(p.is_terminal(nxt))))
        rewards.append(team_reward)
        count = 2-p.active_evader_count(nxt)
        if count > captures:
            capture_events.append(dict(time=float(nxt[p.time_idx, 0]), state=nxt.tolist(),
                                       active=[bool(p.is_active(nxt, r)) for r in range(4)]))
            if first is None:
                first, first_state = float(nxt[p.time_idx, 0]), nxt.tolist()
            if count == 2:
                second = float(nxt[p.time_idx, 0])
        captures, state = count, nxt
    if not rows or not rows[-1]['done']:
        raise RuntimeError('Unexpected empty or truncated on-policy episode')
    illegal = o.violations(p, state)
    timed_out = float(state[p.time_idx, 0]) >= p.tf and captures < 2 and not any(illegal)
    return dict(key=task['key'], arm=task['arm'], seed=task['seed'], index=task['index'],
                snapshot_episode=checkpoint['episode'], snapshot_sha256=task['snapshot_sha256'],
                spec=task['spec'], opponent=task['opponent'], rows=rows, actual_opponent=other.actual_settings,
                stats=dict(steps=len(rows), environment_return=sum(rewards), legal_double_capture=int(captures == 2 and not any(illegal)),
                           pursuer_illegal=int(illegal[1]), evader_illegal=int(illegal[0]), first_capture_time=first,
                           capture_time=second, after_first_capture_state=first_state, capture_events=capture_events,
                           evader_speed=task['spec']['evader_speed'], pursuer_speed=task['spec']['pursuer_speed'],
                           timeout_type=('one_capture' if captures == 1 else 'zero_capture') if timed_out else None))


def validate_rollout(task, episode):
    if any(episode[k] != task[k] for k in ('key', 'arm', 'seed', 'index', 'snapshot_episode', 'snapshot_sha256', 'opponent')):
        raise ValueError('Cross-group or cross-update sample')
    if episode['spec'] != task['spec']:
        raise ValueError('Sample scenario changed')
    actual, requested = episode['actual_opponent'], task['settings']
    if actual['method'] != task['opponent']:
        raise ValueError('Sampling opponent changed')
    if task['opponent'] != 'Policy Only':
        budget = requested['search_only_budget' if task['opponent'] == 'Search Only' else 'search_budget']
        betas = [0., 0.] if task['opponent'] == 'Search Only' else [.5, .1]
        if actual['budget'] != budget or actual['depth'] != 100 or not np.allclose(
                [actual['beta_policy'], actual['beta_value']], betas, atol=1e-7, rtol=0):
            raise ValueError('Effective training opponent parameters changed')


def read_rollout(task):
    path = Path(task['result'])
    if locked_read(path.with_suffix('.json'))['sha256'] != o.sha(path):
        raise ValueError('Sampling cache content changed')
    episode = o.load_torch(path)
    validate_rollout(task, episode)
    return episode


def sample_batch(args, tasks, pool, status, phase):
    missing = []
    for task in tasks:
        if Path(task['result']).with_suffix('.lock.json').exists():
            read_rollout(task)
        else:
            missing.append(task)
    completed = len(tasks)-len(missing)
    status(phase, completed=completed, total=len(tasks), batch_episode=tasks[0]['snapshot_episode'])
    iterator = pool.imap_unordered(rollout, missing, chunksize=1) if pool else map(rollout, missing)
    by_key = {task['key']: task for task in tasks}
    for episode in iterator:
        task = by_key[episode['key']]
        validate_rollout(task, episode)
        path = Path(task['result'])
        o.save_torch(path, episode)
        lock_json(path.with_suffix('.json'), dict(sha256=o.sha(path)))
        completed += 1
        status(phase, completed=completed, total=len(tasks), batch_episode=task['snapshot_episode'])


def update_runtime(args, runtime, tasks, status):
    saved, dest = runtime['saved'], runtime['directory']
    episodes = [read_rollout(task) for task in tasks]
    if len(episodes) != 64 or [e['index'] for e in episodes] != list(range(saved['episode'], saved['episode']+64)):
        raise ValueError('Incomplete or reordered on-policy batch')
    o.restore_rng(saved['rng'])
    status('ppo_update', arm=runtime['arm'], seed=runtime['seed'], episode=saved['episode'])
    metric = o.update_ppo(runtime['actors'], runtime['critic'], runtime['actor_opt'], runtime['critic_opt'], episodes, config(args))
    actors = o.actor_states(runtime['actors'])
    assert_frozen(args, actors)
    completed = saved['episode']+64
    metric.update(arm=runtime['arm'], seed=runtime['seed'], episode=completed,
                  double_capture_episodes=sum(e['stats']['legal_double_capture'] for e in episodes),
                  opponents={op: sum(e['opponent'] == op for e in episodes) for op in o.OPPONENTS})
    payload = copy.deepcopy(saved)
    payload.update(actors=actors, critic=runtime['critic'].state_dict(),
                   actor_optimizer=runtime['actor_opt'].state_dict(), critic_optimizer=runtime['critic_opt'].state_dict(),
                   episode=completed, rng=o.rng_state(), last_metrics=metric,
                   pending_validation=completed % config(args)['eval_every'] == 0,
                   last_batch=dict(manifest=str(dest / f'batches/{saved["episode"]:06d}.json'),
                                   keys=[task['key'] for task in tasks]))
    # Commit full state before any validation or progress projection. An interrupted
    # update replays from the last state and the same immutable batch/RNG.
    runtime['saved'] = f.commit_checkpoint(dest, payload, saved['selection'])
    project_runtime(runtime, episodes)


def project_runtime(runtime, episodes=None):
    saved, dest = runtime['saved'], runtime['directory']
    if saved['episode']:
        batch = locked_read(Path(saved['last_batch']['manifest']))
        episodes = episodes or [read_rollout(task) for task in batch['tasks']]
        o.write_json(dest / f'updates/{saved["episode"]:06d}.json', saved['last_metrics'])
        o.write_json(dest / f'rollouts/{saved["episode"]:06d}.json',
                     [dict(spec=e['spec'], opponent=e['opponent'], stats=e['stats']) for e in episodes])
    o.write_json(dest / 'progress.json', dict(arm=runtime['arm'], seed=runtime['seed'], episode=saved['episode'],
                                             pending_validation=saved['pending_validation'], best_episode=saved['best_episode']))
    o.write_json(dest / 'selection.json', saved['selection'])


def train_pair(args, banks, seed, baseline_rows, pool, status):
    runtimes = [load_runtime(args, arm, seed) for arm in ARMS]
    for runtime in runtimes:
        project_runtime(runtime)
    while True:
        verify(args)
        for runtime in runtimes:
            validate_runtime(args, banks, runtime, baseline_rows, pool, status)
        completed = min(r['saved']['episode'] for r in runtimes)
        if completed == config(args)['episodes']:
            break
        active = [r for r in runtimes if r['saved']['episode'] == completed]
        tasks = {r['arm']: sampling_tasks(args, banks, r) for r in active}
        sample_batch(args, [t for r in active for t in tasks[r['arm']]], pool, status,
                     f'{"pilot" if seed == 0 else "expand"}_sampling_seed{seed}_ep{completed}')
        for runtime in active:
            update_runtime(args, runtime, tasks[runtime['arm']], status)
    for runtime in runtimes:
        project_runtime(runtime)
    return [r['saved']['selection']['best'] for r in runtimes]


def baseline_validation(args, banks, pool, status):
    return run_evaluation(args, 'validation_controls',
                          requests(args, controls(args), banks['validation_four']+banks['validation_none']), pool, status)


def verify_models(items):
    for item in items:
        if o.model_fingerprint(item['models']) != item['model_hash'] or f.policy_id(item['models']) != item['policy_id']:
            raise ValueError('Locked selected model changed')
        if item.get('source_checkpoint') and o.sha(item['source_checkpoint']) != item['checkpoint_sha256']:
            raise ValueError('Locked selected full checkpoint changed')


def pilot(args, banks, pool, status):
    path = args.output / 'pilot_result.json'
    if path.with_suffix('.lock.json').exists():
        result = locked_read(path)
        verify_models([r['model'] for r in result['selected']])
        return result
    baseline_rows = baseline_validation(args, banks, pool, status)
    selected = train_pair(args, banks, 0, baseline_rows, pool, status)
    gates = []
    for entry in selected:
        checks = performance_checks(entry['comparison'], entry['none_comparison'], gain=True)
        gates.append(dict(uid=entry['model']['uid'], checks=checks, passed=all(checks.values())))
    observed = any(g['passed'] for g in gates)
    result = dict(selected=selected, gates=gates, passed=observed and not args.smoke,
                  expansion_authorized=observed and not args.smoke, observed_validation_gate=observed, smoke=args.smoke,
                  note='Validation screening only; this is not an independently confirmed gain.')
    return lock_json(path, result)


def expand(args, banks, pool, status):
    gate = locked_read(args.output / 'pilot_result.json')
    if not gate['passed'] or not gate['expansion_authorized']:
        raise ValueError('Locked pilot did not authorize expansion')
    verify_models([e['model'] for e in gate['selected']])
    path = args.output / 'checkpoint_selection.json'
    if path.with_suffix('.lock.json').exists():
        result = locked_read(path)
        verify_models(result['models'])
        return result
    baseline_rows = baseline_validation(args, banks, pool, status)
    selected = list(gate['selected'])
    for seed in (1, 2):
        selected.extend(train_pair(args, banks, seed, baseline_rows, pool, status))
    selected.sort(key=lambda e: (e['model']['arm'], e['model']['seed']))
    if [e['model']['uid'] for e in selected] != ['A0', 'A1', 'A2', 'B0', 'B1', 'B2']:
        raise ValueError('All six checkpoint selections are required')
    return lock_json(path, dict(selected=selected, models=[e['model'] for e in selected],
                               controls=controls(args), selection_split='validation_only',
                               experiment_sha256=o.sha(args.output / 'experiment.json')))


def selected_models(args):
    selected = locked_read(args.output / 'checkpoint_selection.json')
    if selected['experiment_sha256'] != o.sha(args.output / 'experiment.json'):
        raise ValueError('Selections belong to another experiment')
    items = selected['controls']+selected['models']
    if sorted(m['uid'] for m in selected['models']) != ['A0', 'A1', 'A2', 'B0', 'B1', 'B2'] or \
            sorted(m['uid'] for m in selected['controls']) != sorted([CONTROL, BASELINE]):
        raise ValueError('Incomplete locked checkpoint registry')
    verify_models(items)
    return items


def confirmation_result(args, rows):
    specs = comparison_specs()
    four = paired_statistics(rows, specs)
    none = paired_statistics(rows, specs, 'none', simultaneous=False)
    by_four = {s['id']: s for s in four['comparisons']}
    by_none = {s['id']: s for s in none['comparisons']}
    candidates = []
    for arm in ARMS:
        for seed in SEEDS:
            uid = f'{arm}{seed}'
            fc, nc = by_four[f'{uid}-{CONTROL}'], by_none[f'{uid}-{CONTROL}']
            checks = performance_checks(fc, nc, gain=True, confidence=True)
            candidates.append(dict(uid=uid, arm=arm, seed=seed, checks=checks, pursuit_pass=all(checks.values()),
                                   four=aggregate(rows, uid), none=aggregate(rows, uid, 'none'),
                                   control_comparison=fc, baseline_comparison=by_four[f'{uid}-{BASELINE}'],
                                   no_obstacle_comparison=nc))
    return dict(smoke=args.smoke, four=four, none=none, candidates=candidates,
                eligible=[c['uid'] for c in candidates if c['pursuit_pass']],
                control=dict(four=aggregate(rows, CONTROL), none=aggregate(rows, CONTROL, 'none')),
                strong_opponent_contribution=[by_four[f'B{seed}-A{seed}'] for seed in SEEDS],
                note='All fixed checkpoints reported. No resampling of continuation seeds; all share one pretrained and one PPO starting point.')


def confirm(args, banks, pool, status):
    models = selected_models(args)  # Locked before any confirmation cache is read.
    items = requests(args, models, banks['confirm_four']+banks['confirm_none'])
    expected = 8*3*(len(banks['confirm_four'])+len(banks['confirm_none']))
    if len(items) != expected:
        raise ValueError('Confirmation matrix must contain six candidates and two controls')
    rows = run_evaluation(args, 'confirmation', items, pool, status)
    return lock_json(args.output / 'confirmation_result.json', confirmation_result(args, rows))


def guard(args, banks, pool, status):
    pursuit = locked_read(args.output / 'confirmation_result.json')
    if not pursuit['eligible']:
        raise ValueError('No pursuit-passing candidate')
    models = [m for m in selected_models(args) if m['uid'] in set(pursuit['eligible']) | {CONTROL}]
    rows = run_evaluation(args, 'evader_regression',
                          requests(args, models, banks['guard_four']+banks['guard_none'], role=0), pool, status)
    candidates = []
    for uid in pursuit['eligible']:
        comparisons = {layout: point_comparison(rows, uid, layout=layout, role=0) for layout in ('four', 'none')}
        checks = {layout: c['delta'] >= -.03-1e-12 and c['own_illegal_delta'] <= .02+1e-12
                  for layout, c in comparisons.items()}
        candidates.append(dict(uid=uid, comparisons=comparisons, checks=checks, passed=all(checks.values())))
    return lock_json(args.output / 'guard_result.json', dict(candidates=candidates,
                     accepted=[c['uid'] for c in candidates if c['passed']], evader_settings=settings(args, 0)))


def timing(args, banks, status):
    accepted = set(locked_read(args.output / 'guard_result.json')['accepted'])
    if not accepted:
        raise ValueError('No regression-passing candidate for timing')
    children = Path(f'/proc/{os.getpid()}/task/{os.getpid()}/children').read_text().split()
    if any(b'spawn_main' in Path(f'/proc/{pid}/cmdline').read_bytes() for pid in children):
        raise RuntimeError('Parallel workers must exit before single-process timing')
    models = [m for m in selected_models(args) if m['uid'] in accepted | {CONTROL}]
    before = v.machine_snapshot()
    for item in requests(args, models, banks['warmup'], opponents=('Policy Only',), timing='warmup'):
        status('warmup', uid=item['uid'])
        evaluate_one(item['task'])
    items = []
    for index, spec in enumerate(banks['timing_four']):
        offset = index % len(models)
        items.extend(requests(args, models[offset:]+models[:offset], [spec],
                              opponents=('Policy Only',), timing='single_process'))
    rows = run_evaluation(args, 'serial_timing', items, None, status)
    result = {}
    for item in models:
        part = [r for r in rows if r['uid'] == item['uid']]
        decisions = np.concatenate([np.asarray(r['decision_latency_ms']) for r in part], axis=0)
        result[item['uid']] = dict(episodes=len(part), decisions=len(decisions),
            pursuer_mean_ms=float(decisions[:, 0].mean()), pursuer_p50_ms=float(np.median(decisions[:, 0])),
            pursuer_p95_ms=float(np.percentile(decisions[:, 0], 95)),
            episode_mean_ms=float(np.mean([r['episode_wall_ms'] for r in part])),
            episode_p50_ms=float(np.median([r['episode_wall_ms'] for r in part])),
            episode_wall_ms=[r['episode_wall_ms'] for r in part],
            decision_counts=[len(r['decision_latency_ms']) for r in part],
            timing_session_pids=sorted({r['timing_machine']['before']['pid'] for r in part}),
            machine_before=before, machine_after=v.machine_snapshot(), timing_context='single_process')
    base = result[CONTROL]['pursuer_mean_ms']
    for value in result.values():
        value['pursuer_time_ratio_to_current'] = value['pursuer_mean_ms']/base
    path = args.output / 'timing_result.json'
    if path.with_suffix('.lock.json').exists():
        old = locked_read(path)
        if {k: r['pursuer_mean_ms'] for k, r in old.items()} != {k: r['pursuer_mean_ms'] for k, r in result.items()}:
            raise ValueError('Timing measurements changed')
        return old
    return lock_json(path, result)


def selection_key(candidate, timings, descriptors):
    item = next(m for m in descriptors if m['uid'] == candidate['uid'])
    return (-candidate['four']['success_rate'], candidate['four']['own_illegal_rate'],
            candidate['four']['completion_cost'], timings[candidate['uid']]['pursuer_mean_ms'],
            ARMS.index(item['arm']), item['seed'], item['episode'])


def read_recommended(args):
    rec = locked_read(args.output / 'recommended.json')
    if rec['experiment_sha256'] != o.sha(args.output / 'experiment.json') or \
            Path(rec['models']).resolve() != (args.output / 'recommended/models').resolve() or \
            o.model_fingerprint(rec['models']) != rec['model_hash']:
        raise ValueError('Recommended model fingerprint changed')
    if rec['pursuer_settings'] != settings(args) or rec['evader_settings'] != settings(args, 0):
        raise ValueError('Recommended role configuration changed')
    expected = controls(args) if rec['uid'] == CONTROL else selected_models(args)
    source = next(m for m in expected if m['uid'] == rec['uid'])
    if source['model_hash'] != rec['model_hash']:
        raise ValueError('Recommendation is not a frozen candidate')
    return rec


def load_recommended_solver(args, problem, role, search_seed):
    rec = read_recommended(args)
    return f.make_solver(problem, rec['models'], 'Search + Learning', role,
                         rec['pursuer_settings' if role == 1 else 'evader_settings'], search_seed)


def replay(args, banks):
    rec = read_recommended(args)
    descriptors = controls(args) if rec['uid'] == CONTROL else selected_models(args)
    item = next(m for m in descriptors if m['uid'] == rec['uid'])
    split = 'confirm_four' if (args.output / 'confirmation_result.lock.json').exists() else 'validation_four'
    spec = banks[split][0]
    expected = read_cached(args, requests(args, [item], [spec], opponents=('Policy Only',))[0])
    exported = dict(item, models=rec['models'])
    actual = evaluate_one(requests(args, [exported], [spec], opponents=('Policy Only',), timing='replay')[0]['task'])
    if scientific_digest(actual) != scientific_digest(expected):
        raise ValueError('Recommended export failed original-search replay')
    result = dict(uid=rec['uid'], episode_hash=spec['episode_hash'], scientific_result_exact_match=True,
                  replay_not_added_to_confirmation=True)
    o.write_json(args.output / 'replay.json', result)
    return result


def finish(args, banks):
    verify(args)
    gate = locked_read(args.output / 'pilot_result.json')
    pursuit, regression, timings, selected = None, None, {}, None
    descriptors = controls(args)
    if gate['passed']:
        descriptors = selected_models(args)
        pursuit = locked_read(args.output / 'confirmation_result.json')
        if pursuit['eligible']:
            regression = locked_read(args.output / 'guard_result.json')
            if regression['accepted']:
                timings = locked_read(args.output / 'timing_result.json')
                if set(timings) != set(regression['accepted']) | {CONTROL}:
                    raise ValueError('Incomplete candidate timing matrix')
                candidates = [c for c in pursuit['candidates'] if c['uid'] in regression['accepted']]
                selected = min(candidates, key=lambda c: selection_key(c, timings, descriptors))
    # Smoke can exercise selection/export but never advertises a formal gain.
    uid = selected['uid'] if selected else CONTROL
    item = next(m for m in descriptors if m['uid'] == uid)
    target = args.output / 'recommended/models'
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.parent / '.models.pending'
        if pending.exists():
            shutil.rmtree(pending)
        pending.mkdir()
        for name in o.MODEL_NAMES:
            shutil.copyfile(Path(item['models']) / name, pending / name)
        if o.model_fingerprint(pending) != item['model_hash']:
            raise ValueError('Recommendation copy differs')
        pending.replace(target)
    if o.model_fingerprint(target) != item['model_hash']:
        raise ValueError('Existing recommended export differs')
    confirmed = selected is not None and not args.smoke
    reason = 'quality_confirmed' if confirmed else ('smoke_only' if args.smoke else
             ('pilot_no_gain' if not gate['passed'] else 'no_candidate_confirmed'))
    source = o.read_json(args.output / 'experiment.json')['source']['recommendation']
    value = dict(schema='l45_strong_opponent_recommendation_v1', uid=uid, models=str(target),
                 model_hash=item['model_hash'], source_models=item['models'], source_checkpoint=item['source_checkpoint'] or source['source_checkpoint'],
                 source_episode=2048, added_episodes=item['episode'], arm=item['arm'], seed=item['seed'],
                 pursuer_settings=settings(args), evader_settings=settings(args, 0), smoke=args.smoke,
                 confirmed_improvement=confirmed, previous_confirmed_improvement_preserved=True,
                 selected_metrics=selected, timing=timings.get(uid), control_timing=timings.get(CONTROL),
                 experiment_sha256=o.sha(args.output / 'experiment.json'))
    lock_json(args.output / 'recommended.json', value)
    replay_result = replay(args, banks)
    selection_records = []
    for arm in ARMS:
        for seed in SEEDS:
            path = directory(args, arm, seed) / 'selection.json'
            selection_records.append(dict(arm=arm, seed=seed, executed=path.exists(),
                                          selection=o.read_json(path) if path.exists() else None))
    stages = ['prepare', 'audit', 'pilot']
    if gate['passed']:
        stages += ['expand', 'confirm']
        if pursuit['eligible']:
            stages += ['guard']
            if regression['accepted']:
                stages += ['timing']
    result = dict(smoke=args.smoke, reason=reason, quality_gate_passed=confirmed, recommended=uid,
                  pilot=gate, pursuit=pursuit, regression=regression, timings=timings,
                  selections=selection_records, actual_stages=stages+['report', 'replay'],
                  replay=replay_result, continue_formal=False,
                  training_episodes=sum(o.read_json(directory(args, a, s) / 'progress.json')['episode']
                                        for a in ARMS for s in SEEDS if (directory(args, a, s) / 'progress.json').exists()))
    lock_json(args.output / 'results.json', result)
    report(args, result)
    return result


def report(args, result):
    text = ['# l45 强对手续训结果', '', f"结束原因：{result['reason']}；功能 smoke：{args.smoke}。",
            f"新增训练：{result['training_episodes']} 局；实际阶段：{', '.join(result['actual_stages'])}。", '',
            f"最终采用：`{result['recommended']}`；独立确认新增收益：{result['quality_gate_passed']}。"]
    pursuit = result['pursuit']
    if pursuit:
        text += ['', f"新确认集当前推荐起点：{pursuit['control']['four']['success_rate']:.2%}。", '',
                 '| 检查点 | 四障碍双捕率 | 相对当前推荐 | 同时95%下界 | 无障碍 | 追捕门槛 |',
                 '|---|---:|---:|---:|---:|---|']
        for c in pursuit['candidates']:
            e = next(s['selection']['best']['episode'] for s in result['selections'] if f"{s['arm']}{s['seed']}" == c['uid'])
            delta = c['control_comparison']
            text.append(f"| {c['uid']} +{e} | {c['four']['success_rate']:.2%} | {100*delta['delta']:+.2f} pp | {100*delta['simultaneous_lower95']:+.2f} pp | {c['none']['success_rate']:.2%} | {c['pursuit_pass']} |")
        text += ['', '强对手配比 B 对同种子 A：']
        for c in pursuit['strong_opponent_contribution']:
            text.append(f"- {c['id']}：{100*c['delta']:+.2f} pp，普通95%区间 {[round(100*x, 2) for x in c['ci95']]} pp，同时下界 {100*c['simultaneous_lower95']:+.2f} pp。")
        text += ['', '仅同时下界大于零时确认该固定检查点的额外配比收益；完整报告所有续训种子，不宣称跨预训练起点稳定。']
    else:
        text += ['', '试点未提供扩大依据，停止于种子0；没有读取独立确认结果。', '',
                 '| 组别 | 选优新增局数 | 验证增益 | Search Only增益 | 试点通过 |',
                 '|---|---:|---:|---:|---|']
        for e, g in zip(result['pilot']['selected'], result['pilot']['gates']):
            text.append(f"| {g['uid']} | {e['episode']} | {100*e['comparison']['delta']:+.2f} pp | {100*e['comparison']['per_opponent_delta']['Search Only']:+.2f} pp | {g['passed']} |")
    rec = locked_read(args.output / 'recommended.json')
    if result['quality_gate_passed']:
        c = rec['selected_metrics']
        text += ['', f"最好检查点为 {rec['uid']} 新增第 {rec['added_episodes']} 局，相对当前推荐提高 {100*c['control_comparison']['delta']:.2f} 个百分点。"]
    else:
        text += ['', '本轮未确认新增收益，保留 old0:novalue 及此前已独立确认的历史提升。不能把验证波动作为最终提升结论。']
    if rec['timing']:
        t = rec['timing']
        text += ['', f"单进程追捕平均 {t['pursuer_mean_ms']:.2f} ms/步，P50 {t['pursuer_p50_ms']:.2f}，P95 {t['pursuer_p95_ms']:.2f}；相对当前推荐 {t['pursuer_time_ratio_to_current']:.3f} 倍。", '完整墙钟、步数和机器负载见 timing_result.json。']
    else:
        text += ['', '耗时变化：未测量；没有通过全部成功率及回归门槛的候选，未启动单进程计时。']
    text += ['', '选模、15项配对统计、回归及实际执行情况见 results.json；场景排除范围见 scenario_independence.json。',
             '512局是探索预算；试点失败不证明已收敛，也不否定所有PPO方案。',
             '本实验结束，不自动追加训练、改奖励、增大搜索或启动其他路线。', '']
    (args.output / 'REPORT.md').write_text('\n'.join(text))
    if not args.smoke:
        for path in (SOURCE / 'REPORT.md', ROOT / 'current/search_rl_diagnosis_20261022/REPORT.md'):
            link = f"\n\n本轮强对手续训结果：[报告]({args.output / 'REPORT.md'})；[机器可读结果]({args.output / 'results.json'})。\n"
            if str(args.output / 'REPORT.md') not in path.read_text():
                # Prior REPORT.md is editorial only and excluded from source fingerprints.
                with path.open('a') as stream:
                    stream.write(link)


def audit(args, banks):
    verify(args)
    records = o.read_json(args.output / 'experiment.json')
    used = history_scenes([Path(p) for p in records['history']])
    identities = [previous.physical_hash(s) for group in banks.values() for s in group]
    if len(identities) != len(set(identities)) or used.intersection(identities):
        raise ValueError('Physical scene leakage')
    splits = {f'train{s}': ('four', config(args)['episodes']) for s in SEEDS} | config(args)['splits']
    if set(banks) != set(splits):
        raise ValueError('Scenario split set changed')
    for name, (layout, count) in splits.items():
        if len(banks[name]) != count or any(s['layout'] != layout for s in banks[name]):
            raise ValueError('Scenario count/layout changed')
    initial = o.load_torch(args.output / 'initial/source.pt')
    for arm in ARMS:
        for seed in SEEDS:
            schedule = opponent_schedule(args, seed, 0)
            if [schedule[arm].count(op) for op in o.OPPONENTS] != config(args)['mixes'][arm]:
                raise ValueError('Opponent count mismatch')
            if any(schedule['A'][i] != schedule['B'][i] for i in schedule['common_positions']):
                raise ValueError('Common paired opponents moved')
    exported = f.export_checkpoint(args, args.output / 'audit_export', 0, initial['actors'])
    if f.policy_id(exported) != f.policy_id(args.output / 'models/current'):
        raise ValueError('Zero-update export changed actor/value tensors')
    first = controls(args)[0]
    spec = banks['warmup'][0]
    probes = requests(args, [first, dict(first, uid='export', label='export', models=str(exported))],
                      [spec], opponents=('Policy Only',))
    left, right = [evaluate_one(t['task']) for t in probes]
    if scientific_digest(left) != scientific_digest(right):
        raise ValueError('Zero-update original-search replay differs')
    p, _ = o.restore_scenario(spec)
    actual = []
    for role in (0, 1):
        for method in o.OPPONENTS:
            solver = f.make_solver(p, args.output / 'models/baseline', method, role,
                                   settings(args, role), spec['search_seeds'][role])
            actual.append(dict(role=role, **solver.actual_settings))
    result = dict(passed=True, initial_source_episode=initial['episode'],
                  full_critic_and_adam_preserved=True, zero_update_export_and_replay_exact_match=True,
                  physical_overlap=0, actual_cpp_settings=actual,
                  smoke=args.smoke, audit_does_not_read_confirmation=True)
    return lock_json(args.output / 'audit_result.json', result)


def live_pids(output):
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid() or (entry / 'stat').read_text().rsplit(')', 1)[1].split()[0] in ('Z', 'X'):
                continue
            argv = (entry / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if not argv or not Path(argv[0]).name.startswith('python') or not any(Path(a).name == Path(__file__).name for a in argv):
                continue
            stage = next((a.split('=', 1)[1] for a in argv if a.startswith('--stage=')), None)
            if '--stage' in argv:
                stage = argv[argv.index('--stage')+1]
            if stage == 'watch':
                continue
            target = DEFAULT_OUTPUT
            if '--output' in argv:
                target = Path(argv[argv.index('--output')+1])
            for a in argv:
                if a.startswith('--output='):
                    target = Path(a.split('=', 1)[1])
            if not target.is_absolute():
                target = (entry / 'cwd').resolve() / target
            if target.resolve() == output.resolve():
                found.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return sorted(found)


def dashboard(args):
    out = args.output
    state = o.read_json(out / 'status.json') if (out / 'status.json').exists() else {}
    pids = live_pids(out)
    done = state.get('phase') == 'complete' and all((out / n).exists() for n in ('results.json', 'REPORT.md', 'recommended.json'))
    lines = ['l45 强对手续训：核验 → 试点 → 条件扩展 → 独立确认 → 回归 → 单进程计时 → 汇总',
             f"状态：{'已完成' if done else ('运行中 '+str(pids) if pids else '未运行／进度已保存')}；当前阶段：{state.get('phase', '未启动')}",
             f"配置 workers：{state.get('workers', '--')}；输出：{out}"]
    if pids:
        try:
            children = Path(f'/proc/{pids[0]}/task/{pids[0]}/children').read_text().split()
            actual = sum(b'spawn_main' in Path(f'/proc/{p}/cmdline').read_bytes() for p in children)
            lines.append(f'实际并行 worker：{actual}')
        except OSError:
            lines.append('实际 worker：进程切换中')
    for arm in ARMS:
        for seed in SEEDS:
            path = directory(args, arm, seed) / 'latest.json'
            if path.exists():
                value = o.read_json(path)
                lines.append(f"{arm} seed{seed}：新增 {value['episode']}/{config(args)['episodes']} 局；待验证 {value['pending_validation']}")
    for name in ('validation_controls', 'confirmation', 'evader_regression', 'serial_timing'):
        path = out / name / 'manifest.json'
        if path.exists():
            keys = {r['key'] for r in o.read_json(path)}
            count = sum((out / 'cache' / (key+'.json')).exists() for key in keys)
            lines.append(f'{name} 实际缓存：{count}/{len(keys)}')
    if pids and state.get('rate', 0) > 0 and state.get('total', 0) > state.get('completed', 0):
        eta = (state['total']-state['completed'])/state['rate']/60
        lines.append(f"当前阶段 {state['rate']:.3f} 局/秒，剩余约 {eta:.1f} 分钟；不包含后续阶段。")
    if done:
        result = o.read_json(out / 'results.json')
        lines.append(f"最终采用 {result['recommended']}；确认新增收益 {result['quality_gate_passed']}")
    lines.append('并行吞吐仅用于ETA；部署耗时独立单进程测量。关闭看板不会停止实验。')
    return '\n'.join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('pipeline', 'prepare', 'audit', 'pilot', 'expand', 'confirm',
                                          'guard', 'timing', 'report', 'watch', 'replay'), default='pipeline')
    parser.add_argument('--workers', type=int, default=80)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.workers < 1:
        parser.error('workers must be positive')
    if args.smoke and args.output == DEFAULT_OUTPUT.resolve():
        parser.error('Smoke requires an isolated output')
    for source in (SOURCE, v.SOURCE, f.SOURCE, f.FINAL_SOURCE, o.SOURCE_MODELS):
        source = source.resolve()
        if args.output == source or args.output.is_relative_to(source) or source.is_relative_to(args.output):
            parser.error('Output overlaps protected historical data')
    if args.output == ROOT or not args.output.name:
        parser.error('Use a dedicated experiment directory')
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(1)
    if args.stage == 'watch':
        try:
            while True:
                print(('' if args.once else '\033[2J\033[H')+dashboard(args), flush=True)
                if args.once:
                    return
                time.sleep(3)
        except KeyboardInterrupt:
            return
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'run.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    latest, clocks = {}, {}
    state_lock, stopped = threading.RLock(), threading.Event()

    def status(phase, **fields):
        if 'completed' in fields:
            now = time.monotonic()
            begin, initial = clocks.setdefault(phase, (now, fields['completed']))
            fields['rate'] = (fields['completed']-initial)/max(now-begin, 1e-9)
        value = dict(phase=phase, pid=os.getpid(), workers=args.workers, updated_unix=time.time(), **fields)
        with state_lock:
            latest.clear()
            latest.update(value)
            o.write_json(args.output / 'status.json', value)
        if fields.get('completed', 0) % 10 == 0 or fields.get('completed') == fields.get('total'):
            print(value, flush=True)

    def heartbeat():
        while not stopped.wait(5):
            with state_lock:
                if latest:
                    latest['heartbeat_unix'] = time.time()
                    o.write_json(args.output / 'status.json', latest)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        status('prepare')
        banks = prepare(args)
        if args.stage == 'prepare':
            status('prepared')
            return
        if args.stage in ('replay', 'report', 'timing'):
            result = (replay(args, banks) if args.stage == 'replay' else
                      finish(args, banks) if args.stage == 'report' else timing(args, banks, status))
            status('complete' if args.stage == 'report' else 'stage_complete', stage=args.stage)
            return result
        status('audit')
        audit(args, banks)
        if args.stage == 'audit':
            status('audited')
            return
        with mp.get_context('spawn').Pool(args.workers) as pool:
            if args.stage in ('pipeline', 'pilot'):
                gate = pilot(args, banks, pool, status)
            else:
                gate = locked_read(args.output / 'pilot_result.json')
            if args.stage in ('expand', 'confirm', 'guard') and not gate['passed']:
                raise ValueError('Locked pilot did not authorize downstream stages')
            if args.stage in ('pipeline', 'expand') and gate['passed']:
                expand(args, banks, pool, status)
            elif args.stage == 'expand':
                raise ValueError('Locked pilot did not authorize expansion')
            if args.stage in ('pipeline', 'confirm') and gate['passed']:
                pursuit = confirm(args, banks, pool, status)
            elif args.stage == 'guard':
                pursuit = locked_read(args.output / 'confirmation_result.json')
            else:
                pursuit = None
            if (args.stage == 'pipeline' and pursuit and pursuit['eligible']) or args.stage == 'guard':
                guard(args, banks, pool, status)
        if args.stage == 'pipeline':
            if pursuit and pursuit['eligible'] and locked_read(args.output / 'guard_result.json')['accepted']:
                timing(args, banks, status)
            status('report')
            result = finish(args, banks)
            status('complete', recommended=result['recommended'], quality_gate_passed=result['quality_gate_passed'], reason=result['reason'])
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
