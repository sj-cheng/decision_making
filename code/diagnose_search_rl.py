"""Read-only paired checkpoint diagnostics for the fixed PPO validation suite.

Writes new artifacts only. This script does not modify training/evaluation code.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
import benchmark_local_obstacles as benchmark
from learning.local_rl import PolicyBundle
from local_observation import TEAMS, make_episode
from train_search_rl import episode_seed

_models = {}
_output = None
_execution = 'mean'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def init_worker(source, output, execution='mean'):
    global _models, _output, _execution
    torch.set_num_threads(1)
    source = Path(source)
    _models = {name: PolicyBundle.load(source / filename) for name, filename in
               [('distilled', 'distilled.pt'), ('best_role0', 'best_role0.pt'),
                ('best_role1', 'best_role1.pt'), ('final', 'ppo_final.pt')]}
    _output = Path(output)
    _execution = execution


def evaluate_task(task):
    variant, role, layout, index, training_seed = task
    seed = episode_seed(training_seed, 80 + ('two', 'four', 'random').index(layout), role, index)
    spec = make_episode(seed, layout)
    candidate = _models[f'best_role{role}' if variant == 'best' else variant]
    captured = []
    original_build = benchmark.build_controller

    def capture_build(*args, **kwargs):
        controller = original_build(*args, **kwargs)
        if args[2] == role and (variant == 'distilled' or _execution == 'stochastic'):
            original_act = controller.act

            def act(obs, *positional, **named):
                if _execution == 'stochastic':
                    named['stochastic'] = True
                result = original_act(obs, *positional, **named)
                if variant == 'distilled':
                    captured.append((result[1]['features'].copy(), obs.limits.copy(), obs.active.copy()))
                return result

            controller.act = act
        return controller

    benchmark.build_controller = capture_build
    try:
        row = benchmark.evaluate_episode(spec, candidate, _models['distilled'], role, 'ppo', 'distilled')
    finally:
        benchmark.build_controller = original_build
    row.update(variant=variant, validation_index=index, execution=_execution)
    write_json(_output / 'episodes' / f'{variant}_role{role}_{layout}_{index:02}.json', row)
    if captured:
        features, limits, active = (np.asarray(values) for values in zip(*captured))
        np.savez_compressed(_output / 'common_observations' / f'role{role}_{layout}_{index:02}.npz',
                            features=features, limits=limits, active=active,
                            seed=seed, role=role, episode_hash=spec.fingerprint)
    return row


def paired_comparison(rows, baseline, candidate, role):
    metric = 'legal_survival' if role == 0 else 'double_capture'
    old = {r['episode_hash']: r for r in rows if r['role'] == role and r['variant'] == baseline}
    pairs = [(old[r['episode_hash']], r) for r in rows if r['role'] == role and r['variant'] == candidate]
    delta = np.array([new[metric] - previous[metric] for previous, new in pairs])
    rng = np.random.default_rng(20261022)
    boot = rng.choice(delta, (20000, len(delta)), replace=True).mean(1)
    return dict(baseline=baseline, candidate=candidate, role=role, metric=metric, n=len(pairs),
                success_delta=float(delta.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
                improved=int(np.sum(delta > 0)), regressed=int(np.sum(delta < 0)),
                unchanged=int(np.sum(delta == 0)),
                changed_episodes=[dict(seed=new['seed'], layout=new['layout'],
                                       episode_hash=new['episode_hash'], before=previous[metric], after=new[metric],
                                       before_own_collision=previous['evader_obstacle_collision' if role == 0 else 'pursuer_obstacle_collision'],
                                       after_own_collision=new['evader_obstacle_collision' if role == 0 else 'pursuer_obstacle_collision'])
                                  for previous, new in pairs if previous[metric] != new[metric]],
                caveat='Paired validation episodes, one training seed. CI excludes training-seed uncertainty; suite used for checkpoint selection.')


def action_diagnostics(output, source, execution='mean'):
    models = {name: PolicyBundle.load(source / filename) for name, filename in
              [('distilled', 'distilled.pt'), ('best_role0', 'best_role0.pt'),
               ('best_role1', 'best_role1.pt'), ('final', 'ppo_final.pt')]}
    report = {}
    with torch.no_grad():
        for role in range(2):
            observations = [np.load(path) for path in sorted((output / 'common_observations').glob(f'role{role}_*.npz'))]
            features = np.concatenate([f['features'] for f in observations])
            limits = np.concatenate([f['limits'] for f in observations])
            active = np.concatenate([f['active'] for f in observations]).astype(bool)
            distributions = {}
            role_report = dict(active_actor_decisions=int(active.sum()), models={}, comparisons={})
            for variant in ('distilled', 'best', 'final'):
                bundle = models[f'best_role{role}' if variant == 'best' else variant]
                means, stds, half_ranges, saturation_probs = [], [], [], []
                for local, robot in enumerate(TEAMS[role]):
                    x = torch.tensor(features[:, local][active[:, local]], dtype=torch.float32)
                    lims = torch.tensor(limits[:, local][active[:, local]], dtype=torch.float32)
                    dist = bundle.actors[robot].distribution(x, lims)
                    half = (lims[..., 1] - lims[..., 0]) * .5
                    means.append(dist.mean.numpy())
                    stds.append(dist.scale.numpy())
                    half_ranges.append(half.numpy())
                    saturation_probs.append((dist.cdf(lims[..., 0]) + 1. - dist.cdf(lims[..., 1])).numpy())
                mean, std, half = np.concatenate(means), np.concatenate(stds), np.concatenate(half_ranges)
                distributions[variant] = (mean, std, half)
                normalized_mean = mean / half
                role_report['models'][variant] = dict(mean_abs_normalized_action=float(np.abs(normalized_mean).mean()),
                    mean_coordinate_near_boundary_95=float((np.abs(normalized_mean) >= .95).mean()),
                    std_physical_mean=float(std.mean()), std_normalized_mean=float((std / half).mean()),
                    gaussian_sample_coordinate_clipping_probability=float(np.concatenate(saturation_probs).mean()))
            for a, b in [('distilled', 'best'), ('distilled', 'final'), ('best', 'final')]:
                mean_a, std_a, half = distributions[a]
                mean_b, std_b, _ = distributions[b]
                delta = (mean_b - mean_a) / half
                kl = np.log(std_b / std_a) + (std_a ** 2 + (mean_a - mean_b) ** 2) / (2 * std_b ** 2) - .5
                role_report['comparisons'][f'{a}_to_{b}'] = dict(
                    normalized_mean_rmse=float(np.sqrt(np.mean(delta ** 2))),
                    normalized_mean_abs_p95=float(np.percentile(np.abs(delta), 95)),
                    gaussian_kl_per_actor_mean=float(kl.sum(-1).mean()),
                    gaussian_kl_per_actor_p95=float(np.percentile(kl.sum(-1), 95)))
            report[f'role{role}'] = role_report
    report['method'] = (f'Identical active actor inputs from 120 distilled-vs-distilled validation trajectories; '
                        f'tested distilled execution={execution}, fixed opponent execution=mean. '
                        'KL is pre-clipping physical Normal KL, not clipped action distribution KL. Limits are symmetric.')
    write_json(output / 'action_drift.json', report)
    return report


def history_diagnostics(source, output):
    roots = [source.parents[1] / name / source.name for name in ('search_rl_v1', 'search_rl_v2_parallel', 'search_rl_v3_overlap')]
    metrics, episodes, validations = [], [], []
    for segment in roots:
        for filename, destination in [('train_metrics.jsonl', metrics), ('ppo_episodes.jsonl', episodes)]:
            path = segment / filename
            if path.exists():
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    row['segment'] = segment.parent.name
                    destination.append(row)
        for path in sorted(segment.glob('validation_role*_ep*.json')):
            row = json.loads(path.read_text())
            role = int(path.stem.split('_')[1][-1])
            epoch = int(path.stem.rsplit('ep', 1)[1])
            own = 'evader' if role == 0 else 'pursuer'
            validations.append(dict(segment=segment.parent.name, role=role, episode=epoch, score=row['score'],
                successes=sum(r['legal_survival' if role == 0 else 'double_capture'] for r in row['episodes']),
                own_collisions=sum(r[f'{own}_obstacle_collision'] for r in row['episodes']),
                opponent_collisions=sum(r[f'{"pursuer" if role == 0 else "evader"}_obstacle_collision'] for r in row['episodes']),
                own_illegal=sum(r[f'{own}_illegal'] for r in row['episodes']),
                n=len(row['episodes']), path=str(path)))
    write_json(output / 'validation_history.json', validations)
    with (output / 'training_metrics.csv').open('w') as handle:
        keys = sorted({k for r in metrics for k in r})
        writer = csv.DictWriter(handle, keys)
        writer.writeheader(); writer.writerows(metrics)
    bins = [(0, 512), (512, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000)]
    training_summary = []
    for role in range(2):
        role_episodes = [r for r in episodes if r['stats']['role'] == role]
        for lo, hi in bins:
            chosen = role_episodes[lo:hi]
            updates = [r for r in metrics if r['role'] == role and lo < r['episodes'][role] <= hi]
            if not chosen:
                continue
            stats = [r['stats'] for r in chosen]
            training_summary.append(dict(role=role, start_episode=lo, end_episode=hi, n=len(chosen),
                task_success_rate=float(np.mean([r['survival' if role == 0 else 'double'] for r in stats])),
                own_illegal_rate=float(np.mean([r['illegal'][role] for r in stats])),
                opponent_illegal_rate=float(np.mean([r['illegal'][1-role] for r in stats])),
                mean_episode_steps=float(np.mean([r['steps'] for r in stats])),
                reward_per_episode=float(np.mean([r['returns'][role] for r in stats])),
                mean_update_kl=float(np.mean([r['kl'] for r in updates])) if updates else None,
                updates_mean_kl_over_03=int(sum(r['kl'] > .03 for r in updates)),
                update_count=len(updates), mean_anchor=float(np.mean([r['anchor'] for r in updates])) if updates else None,
                mean_value_loss=float(np.mean([r['value_loss'] for r in updates])) if updates else None))
    report = dict(training_bins=training_summary, validation_history=validations,
                  episode_count=len(episodes), update_count=len(metrics),
                  caveat='Training bins differ in curriculum and opponent sampling, so their success rates are not controlled learning-curve estimates. Validation is fixed.')
    write_json(output / 'training_history.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--training-seed', type=int, default=0)
    parser.add_argument('--execution', choices=('mean', 'stochastic'), default='mean')
    args = parser.parse_args()
    args.source = args.source.resolve(); args.output = args.output.resolve()
    for name in ('episodes', 'common_observations'):
        (args.output / name).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    provenance = dict(source=str(args.source), training_seed=args.training_seed, workers=args.workers, execution=args.execution,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoints={name: hashlib.sha256((args.source / name).read_bytes()).hexdigest()
                     for name in ('distilled.pt', 'best_role0.pt', 'best_role1.pt', 'ppo_final.pt')},
        protocol='Reuse validation seed namespaces 80, 81, 82; 20 episodes/layout/role; fixed deterministic distilled opponent; tested execution recorded separately; no MCTS.')
    write_json(args.output / 'config.json', provenance)
    tasks = [(variant, role, layout, index, args.training_seed) for variant in ('distilled', 'best', 'final')
             for role in range(2) for layout in ('two', 'four', 'random') for index in range(20)]
    started = time.perf_counter()
    rows = []
    with mp.get_context('spawn').Pool(args.workers, initializer=init_worker,
                                      initargs=(str(args.source), str(args.output), args.execution)) as pool:
        for index, row in enumerate(pool.imap_unordered(evaluate_task, tasks, chunksize=1), 1):
            rows.append(row)
            if index % 30 == 0:
                print(f'validation {index}/{len(tasks)} elapsed={time.perf_counter()-started:.1f}s', flush=True)
    rows.sort(key=lambda r: (r['variant'], r['role'], r['layout'], r['validation_index']))
    write_json(args.output / 'episodes.json', rows)
    with (args.output / 'episodes.csv').open('w') as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for variant in ('distilled', 'best', 'final'):
        for role in range(2):
            chosen = [r for r in rows if r['variant'] == variant and r['role'] == role]
            metric = 'legal_survival' if role == 0 else 'double_capture'
            own = 'evader' if role == 0 else 'pursuer'
            opponent = 'pursuer' if role == 0 else 'evader'
            summary[f'{variant}/role{role}'] = dict(n=len(chosen), successes=sum(r[metric] for r in chosen),
                own_collisions=sum(r[f'{own}_obstacle_collision'] for r in chosen),
                opponent_collisions=sum(r[f'{opponent}_obstacle_collision'] for r in chosen),
                own_out_of_bounds=sum(r[f'{own}_out_of_bounds'] for r in chosen),
                mean_steps=float(np.mean([r['steps'] for r in chosen])),
                mean_return=float(np.mean([r[f'{own}_return'] for r in chosen])),
                failures=[dict(seed=r['seed'], layout=r['layout'], own_collision=r[f'{own}_obstacle_collision'],
                               opponent_collision=r[f'{opponent}_obstacle_collision'], steps=r['steps'])
                          for r in chosen if not r[metric]],
                by_layout={layout: dict(successes=sum(r[metric] for r in chosen if r['layout'] == layout),
                             own_collisions=sum(r[f'{own}_obstacle_collision'] for r in chosen if r['layout'] == layout))
                           for layout in ('two', 'four', 'random')})
    write_json(args.output / 'validation_summary.json', summary)
    paired = [paired_comparison(rows, a, b, role) for a, b in [('distilled', 'best'), ('best', 'final'), ('distilled', 'final')]
              for role in range(2)]
    write_json(args.output / 'paired_comparison.json', paired)
    action_diagnostics(args.output, args.source, args.execution)
    history_diagnostics(args.source, args.output)
    print(json.dumps({k: {f: v[f] for f in ('n', 'successes', 'own_collisions', 'opponent_collisions', 'own_out_of_bounds')}
                      for k, v in summary.items()}, indent=2), flush=True)
    print(f'Complete. elapsed={time.perf_counter()-started:.1f}s output={args.output}', flush=True)


if __name__ == '__main__':
    main()
