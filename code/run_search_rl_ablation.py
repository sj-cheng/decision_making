"""Controlled short PPO diagnostic; never a continuation of the pilot run.

Compose each team's episode-512 actor/critic/normalizer, reset Adam equally,
then compare a 2 x 2 LR/anchor grid on a fixed late curriculum window. The
original total training horizon remains 5000. Frozen opponents, episode seeds,
and update order are paired, while each arm samples its own on-policy data
after the common initial batch. No production checkpoints are modified.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.optim import Adam

import train_search_rl as trainer
from benchmark_local_obstacles import evaluation_task
from learning.local_rl import PolicyBundle, atomic_torch_save, joint_log_prob
from local_observation import TEAMS, make_episode

ARMS = [('lr3e5_anchor0', 3e-5, 'original_linear'),
        ('lr1e5_anchor0', 1e-5, 'original_linear'),
        ('lr3e5_anchor01', 3e-5, 'constant'),
        ('lr1e5_anchor01', 1e-5, 'constant')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(bundle, role=None):
    state = bundle.state_dict()
    digest = hashlib.sha256()
    for group, selected in [('actors', range(4) if role is None else TEAMS[role]),
                            ('critics', range(2) if role is None else [role])]:
        for i in selected:
            for name, value in sorted(state[group][i].items()):
                digest.update(f'{group}/{i}/{name}'.encode())
                digest.update(value.detach().cpu().numpy().tobytes())
    digest.update(json.dumps(state['norms'] if role is None else state['norms'][role], sort_keys=True).encode())
    return digest.hexdigest()


def actual_anchor(schedule, index, horizon, maximum=.1):
    return maximum if schedule == 'constant' else maximum * max(0., 1.-index/(.2*horizon))


def compose_initial(source):
    bundle = PolicyBundle.load(source/'best_role0.pt')
    selected = []
    for role in range(2):
        path = source/f'best_role{role}.pt'
        team = PolicyBundle.load(path)
        for robot in TEAMS[role]:
            bundle.actors[robot].load_state_dict(team.actors[robot].state_dict())
        bundle.critics[role].load_state_dict(team.critics[role].state_dict())
        bundle.norms[role].__dict__.update(copy.deepcopy(team.norms[role].__dict__))
        selected.append(dict(role=role, path=str(path.resolve()), sha256=sha(path),
                             source_episodes=team.metadata.get('episodes_per_role')))
    bundle.refresh()
    bundle.metadata.update(stage='controlled_late_window_diagnostic', selected_roles=selected,
                           diagnostic_note='Team-specific early best, fresh equal Adam; not exact continuation.')
    return bundle


def probe(bundle, anchor, role, results):
    rows = [row for e in results for row in e['rows']]
    tensor = lambda key: torch.tensor(np.asarray([row[key] for row in rows]), dtype=torch.float32)
    obs, raw, limits, active = [tensor(key) for key in ('obs', 'raw', 'limits', 'active')]
    with torch.no_grad():
        logp = joint_log_prob(bundle.actors, obs, raw, limits, active, role)
        means = []; anchor_dist = []; distributions = []
        for local, robot in enumerate(TEAMS[role]):
            scale = ((limits[:, local, :, 1]-limits[:, local, :, 0])*.5).clamp_min(1e-6)
            dist = bundle.actors[robot].distribution(obs[:, local], limits[:, local])
            mean = dist.mean / scale
            ref = anchor.actors[robot].distribution(obs[:, local], limits[:, local]).mean / scale
            means.append(mean)
            distributions.append((dist.mean, dist.stddev, active[:, local]))
            anchor_dist.append(((mean-ref).square().sum(-1)*active[:, local]).sum()/active[:, local].sum().clamp_min(1))
    return dict(logp=logp, old_logp=tensor('old_logp'), means=torch.cat(means),
                distributions=distributions, anchor_mse=float(torch.stack(anchor_dist).mean()))


def gaussian_kl(before, after):
    total = torch.zeros_like(before['logp'])
    for (mean0,std0,active),(mean1,std1,_) in zip(before['distributions'],after['distributions']):
        total += ((std1/std0).log()+(std0.square()+(mean0-mean1).square())/(2*std1.square())-.5).sum(-1)*active
    return float(total.mean())


def actor_vector(bundle, role):
    return torch.cat([p.detach().flatten().clone() for robot in TEAMS[role] for p in bundle.actors[robot].parameters()])


def task_plan(args, seed, role, offset, initial_wire, anchor_wire):
    tasks = []
    for position in range(args.batch):
        index = args.start_index + offset + position
        rng = np.random.default_rng(trainer.episode_seed(seed, 303, role, index))
        draw = rng.random()
        kind = 'search_distilled' if draw >= .9 else 'early_history' if draw >= .45 else 'distilled'
        tasks.append(dict(seed=trainer.episode_seed(seed, 2, role, index), role=role,
                          progress=index/args.horizon, kind='ppo', bundle=initial_wire,
                          opponent=initial_wire if kind == 'early_history' else anchor_wire,
                          search_opponent=kind == 'search_distilled', opponent_budget=args.opponent_budget,
                          budget=1000, depth=args.depth, training_index=index, opponent_kind=kind))
    return tasks


def train(args, out, initial, anchor, pool):
    initial_wire = trainer.wire_weights(initial.state_dict())
    anchor_wire = trainer.wire_weights(anchor.state_dict())
    plan_path = out/'task_plan.jsonl'
    if not plan_path.exists():
        with plan_path.open('w') as handle:
            for seed in args.seeds:
                for offset in range(0, args.episodes, args.batch):
                    for role in range(2):
                        for task in task_plan(args, seed, role, offset, initial_wire, anchor_wire):
                            row = {k:v for k,v in task.items() if k not in ('bundle','opponent')}
                            row['training_seed'] = seed
                            handle.write(json.dumps(row)+'\n')
    for seed in args.seeds:
        contexts = []
        for name, lr, schedule in ARMS:
            folder = out/f'seed{seed}'/name
            folder.mkdir(parents=True, exist_ok=True)
            bundle = initial.copy()
            bundle.metadata.update(training_seed=seed, ablation_arm=name)
            actors = [Adam([p for robot in TEAMS[t] for p in bundle.actors[robot].parameters()], lr=lr) for t in range(2)]
            critics = [Adam(bundle.critics[t].parameters(), lr=3e-4) for t in range(2)]
            counters = [0, 0]
            if (folder/'training_state.pt').exists():
                saved = torch.load(folder/'training_state.pt', map_location='cpu', weights_only=False)
                bundle.load_state_dict(saved['bundle']); counters = saved['counters']
                for optimizer, state in zip(actors+critics, saved['optimizers']):
                    optimizer.load_state_dict(state)
            contexts.append(dict(name=name, lr=lr, schedule=schedule, folder=folder, bundle=bundle,
                                 actors=actors, critics=critics, counters=counters))
        for offset in range(0, args.episodes, args.batch):
            for role in range(2):
                pending = [ctx for ctx in contexts if ctx['counters'][role] == offset]
                if not pending:
                    continue
                before_hashes = [state_hash(ctx['bundle'], role) for ctx in pending]
                # Safe sharing requires unchanged own actor and critic/norm too.
                shared = len(set(before_hashes)) == 1
                rollout_groups = pending[:1] if shared else pending
                all_tasks = []
                for ctx in rollout_groups:
                    tasks = task_plan(args, seed, role, offset, initial_wire, anchor_wire)
                    weights = trainer.wire_weights(ctx['bundle'].state_dict())
                    for task in tasks:
                        task['bundle'] = weights
                    all_tasks.extend(tasks)
                tick = time.perf_counter()
                results = trainer.run_tasks(pool, all_tasks)
                rollout_seconds = time.perf_counter()-tick
                for i, ctx in enumerate(pending):
                    chunk = results[:args.batch] if shared else results[i*args.batch:(i+1)*args.batch]
                    group_tasks = all_tasks[:args.batch] if shared else all_tasks[i*args.batch:(i+1)*args.batch]
                    bundle = ctx['bundle']
                    before = probe(bundle, anchor, role, chunk)
                    parameters_before = actor_vector(bundle, role)
                    coefficient = actual_anchor(ctx['schedule'], args.start_index+offset, args.horizon)
                    options = argparse.Namespace(anchor=coefficient, ppo_epochs=4, minibatch=256,
                                                 clip=.1, entropy=.001)
                    # Identical RNG namespace for minibatch shuffling; trajectories
                    # diverge by arm after their own first actor update.
                    update_seed = trainer.episode_seed(seed, 304, role, offset)
                    random.seed(update_seed); np.random.seed(update_seed); torch.manual_seed(update_seed)
                    update_started = time.perf_counter()
                    metrics = trainer.ppo_update(bundle, anchor, role, chunk, ctx['actors'][role],
                                                 ctx['critics'][role], options, progress=0.)
                    after = probe(bundle, anchor, role, chunk)
                    delta = after['logp']-before['logp']
                    logp_error = (before['logp']-before['old_logp']).abs()
                    if float(logp_error.max()) > 2e-3:
                        raise RuntimeError(f'On-policy old_logp mismatch: {float(logp_error.max())}')
                    difference = actor_vector(bundle, role)-parameters_before
                    ctx['counters'][role] += args.batch
                    record = dict(seed=seed, arm=ctx['name'], role=role, actor_lr=ctx['lr'],
                                  anchor_schedule=ctx['schedule'], actual_anchor=coefficient,
                                  start_index=args.start_index+offset, end_index=args.start_index+offset+args.batch,
                                  original_horizon=args.horizon, diagnostic_episodes=ctx['counters'].copy(),
                                  opponent_search_episodes=sum(t['search_opponent'] for t in group_tasks),
                                  shared_initial_trajectory_batch=shared, physical_rollout_groups=len(rollout_groups),
                                  rollout_seconds=rollout_seconds, update_seconds=time.perf_counter()-update_started,
                                  initial_state_sha256=before_hashes[i], final_state_sha256=state_hash(bundle, role),
                                  post_update_empirical_kl=float((delta.exp()-1-delta).mean()),
                                  post_update_exact_gaussian_kl=gaussian_kl(before, after),
                                  post_update_clip_fraction=float(((delta.exp()-1).abs()>.1).float().mean()),
                                  rollout_old_logp_max_abs_error=float(logp_error.max()),
                                  rollout_old_logp_mean_abs_error=float(logp_error.mean()),
                                  normalized_action_mean_change_rms=float((after['means']-before['means']).square().mean().sqrt()),
                                  anchor_mse_before=before['anchor_mse'], anchor_mse_after=after['anchor_mse'],
                                  parameter_delta_l2=float(difference.norm()),
                                  parameter_delta_relative=float(difference.norm()/parameters_before.norm()), **metrics)
                    with (ctx['folder']/'metrics.jsonl').open('a') as handle:
                        handle.write(json.dumps(record)+'\n')
                    with (ctx['folder']/'episodes.jsonl').open('a') as handle:
                        for task, result in zip(group_tasks, chunk):
                            handle.write(json.dumps(dict(seed=seed, role=role, training_index=task['training_index'],
                                      curriculum_progress=task['progress'], opponent_kind=task['opponent_kind'],
                                      actor_state_sha256=before_hashes[i], shared_initial_batch=shared,
                                      spec=result['spec'], stats=result['stats']))+'\n')
                    bundle.metadata.update(diagnostic_episodes=ctx['counters'].copy(),
                                           late_window_index=[args.start_index+c for c in ctx['counters']])
                    atomic_torch_save(dict(bundle=bundle.state_dict(), counters=ctx['counters'],
                                            optimizers=[o.state_dict() for o in ctx['actors']+ctx['critics']]),
                                      ctx['folder']/'training_state.pt')
                    bundle.save(ctx['folder']/'policy.pt')
                    print('ABLATION '+json.dumps(record), flush=True)


def evaluate(args, out, pool, split, episodes, phase):
    folder = out/split
    folder.mkdir(exist_ok=True)
    checkpoints = [('initial', None, out/'initial.pt')]
    checkpoints += [(name, seed, out/f'seed{seed}'/name/'policy.pt') for seed in args.seeds for name,_,_ in ARMS]
    tasks = []; manifest = []
    for arm, training_seed, checkpoint in checkpoints:
        for role in range(2):
            for li, layout in enumerate(('two','four','random')):
                for index in range(episodes):
                    spec = make_episode(trainer.episode_seed(20261022, phase+li, role, index), layout)
                    key = f'{arm}_seed{training_seed}_role{role}_{layout}_{index}'
                    task = dict(spec=asdict(spec), checkpoint=str(checkpoint), reference=str(out/'distilled.pt'),
                                role=role, method='ppo', opponent='distilled', budget=1000, depth=args.depth,
                                arm=arm, training_seed=training_seed, key=key)
                    manifest.append(dict(key=key, episode_fingerprint=spec.fingerprint, checkpoint_sha256=sha(checkpoint),
                                         spec=asdict(spec)))
                    if not (folder/(key+'.json')).exists():
                        tasks.append(task)
    trainer.write_json(folder/'manifest.json', manifest)
    iterator = pool.imap(evaluation_task, tasks, chunksize=1) if pool else map(evaluation_task, tasks)
    for done, (task, row) in enumerate(zip(tasks, iterator), 1):
        row.update(arm=task['arm'], diagnostic_training_seed=task['training_seed'], key=task['key'])
        trainer.write_json(folder/(task['key']+'.json'), row)
        if done % 100 == 0 or done == len(tasks):
            print(f'{split} {done}/{len(tasks)}', flush=True)
    return [json.loads((folder/(entry['key']+'.json')).read_text()) for entry in manifest]


def summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row['arm'], row['diagnostic_training_seed'], row['role']), []).append(row)
    result = []
    for (arm, seed, role), records in grouped.items():
        success = 'legal_survival' if role == 0 else 'double_capture'
        own = 'evader' if role == 0 else 'pursuer'
        result.append(dict(arm=arm, seed=seed, role=role, n=len(records), successes=int(sum(r[success] for r in records)),
                           success_rate=float(np.mean([r[success] for r in records])),
                           own_illegal=float(np.mean([r[own+'_illegal'] for r in records])),
                           any_illegal=float(np.mean([bool(r['pursuer_illegal'] or r['evader_illegal']) for r in records])),
                           own_collision=float(np.mean([r[own+'_obstacle_collision'] for r in records])),
                           mean_return=float(np.mean([r[own+'_return'] for r in records]))))
    return result


def paired_comparisons(rows, seeds):
    lookup = {(r['arm'], r['diagnostic_training_seed'], r['role'], r['episode_hash']):r for r in rows}
    reports = []
    rng = np.random.default_rng(79480)
    for baseline in ('initial','lr3e5_anchor0'):
        for name, _, _ in ARMS:
            if name == baseline:
                continue
            for role in range(2):
                metric = 'legal_survival' if role == 0 else 'double_capture'
                changes = []
                for seed in seeds:
                    records = [r for r in rows if r['arm'] == name and r['diagnostic_training_seed'] == seed and r['role'] == role]
                    changes.append([r[metric]-lookup[(baseline, None if baseline == 'initial' else seed, role, r['episode_hash'])][metric] for r in records])
                changes = np.asarray(changes)
                sampled = []
                for _ in range(2000):
                    selected = rng.integers(len(seeds), size=len(seeds))
                    # A confirmation episode is shared across training seeds.
                    # Resample the same episode indices for all selected seeds.
                    episode_indices = rng.integers(changes.shape[1], size=changes.shape[1])
                    sampled.append(changes[selected][:,episode_indices].mean())
                reports.append(dict(baseline=baseline, candidate=name, role=role, success_delta=float(changes.mean()),
                                    per_seed_delta=changes.mean(1).tolist(), paired_bootstrap_ci95=np.percentile(sampled,[2.5,97.5]).tolist(),
                                    wins=int((changes>0).sum()), losses=int((changes<0).sum()), training_seeds=len(seeds),
                                    caveat='Two fresh diagnostic seeds from one pretrained seed; short window, not formal evidence.'))
    return reports


def write_report(args, out, validation, confirmation):
    summaries = {split:summary(rows) for split,rows in [('validation',validation),('confirmation',confirmation)]}
    rank = validation_ranking(summaries['validation'])
    comparisons = paired_comparisons(confirmation, args.seeds)
    selected = rank[0][2]
    selected_confirm = [r for r in comparisons if r['candidate']==selected and r['baseline']=='initial']
    clear = bool(selected_confirm and any(r['paired_bootstrap_ci95'][0]>0 for r in selected_confirm)
                 and all(min(r['per_seed_delta'])>=0 for r in selected_confirm))
    report = dict(config=vars(args), selected_by_validation=selected, validation_ranking=rank,
                  summaries=summaries, confirmation_comparisons=comparisons,
                  short_window_signal=clear, continue_formal=False,
                  formal_reason='Require longer-window replicated gains and full budget-1000 search confirmation before 50k.',
                  notes=['Controlled late-window diagnostic, not exact pilot continuation.',
                         'Both roles start from their own episode-512 best actor/critic/value normalizer; all Adam states reset.',
                         'Original horizon stays 5000; curriculum uses actual indices, original anchor is zero in this late window.',
                         'Frozen early opponent history replaces the changing original pool to isolate LR/anchor.',
                         'Only the initial identical-policy batch is shared; later trajectories are arm-specific on-policy.',
                         'Validation phases 180-182 select; disjoint confirmation phases 190-192 are not used to tune.',
                         'Confirmation evaluates deterministic pure policies against distilled opponents; search remains unverified.'])
    trainer.write_json(out/'report.json', report)
    lines = ['# PPO late-window controlled diagnostic', '',
             f'{args.episodes} episodes per role; late index {args.start_index} to {args.start_index+args.episodes-1}; original horizon {args.horizon}.',
             'Each arm starts from the same team-specific episode-512 actor, critic and value normalizer, with fresh equal Adam.',
             'This is a short controlled diagnostic, not an exact continuation. Frozen opponents are paired across arms.', '',
             '| Arm | Seed | Role | Validation | Confirmation | Own collisions (confirmation) |',
             '|---|---:|---:|---:|---:|---:|']
    for row in summaries['confirmation']:
        val = next(s for s in summaries['validation'] if (s['arm'],s['seed'],s['role'])==(row['arm'],row['seed'],row['role']))
        lines.append(f"| {row['arm']} | {row['seed']} | {row['role']} | {val['successes']}/{val['n']} | {row['successes']}/{row['n']} | {row['own_collision']:.1%} |")
    lines += ['',f'Validation-selected arm: `{selected}`. Clear short-window signal: `{clear}`.',
              'Formal 50k training is not launched. Full search confirmation is pending.', '',
              'Role 0 = evaders; role 1 = pursuers. Success requires legal endings for both teams.',
              'Only two new diagnostic training seeds share a single pretrained seed; uncertainty is preliminary.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,4),sharey=True)
    names = ['initial']+[arm[0] for arm in ARMS]
    for role, ax in enumerate(axes):
        for j,seed in enumerate(args.seeds):
            values=[]
            for name in names:
                values.append(next(s['success_rate'] for s in summaries['confirmation'] if s['arm']==name and s['seed']==(None if name=='initial' else seed) and s['role']==role))
            ax.plot(range(len(names)),values,'o-',label=f'seed {seed}')
        ax.set_xticks(range(len(names)),names,rotation=25,ha='right')
        ax.set_title('Evader legal survival' if role==0 else 'Pursuer double capture')
        ax.set_ylim(0,1.05); ax.grid(alpha=.2); ax.legend()
    fig.suptitle(f'Disjoint confirmation: {3*args.confirmation_per_layout} episodes / role / diagnostic seed')
    fig.tight_layout(); fig.savefig(out/'confirmation.png',dpi=150); plt.close(fig)
    print(json.dumps(dict(selected=selected, short_window_signal=clear, continue_formal=False)),flush=True)


def validation_ranking(records):
    rank = []
    for name, _, _ in ARMS:
        items = [s for s in records if s['arm'] == name]
        rank.append((float(np.mean([s['success_rate'] for s in items])), -float(np.mean([s['any_illegal'] for s in items])), name))
    return sorted(rank, reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=trainer.ROOT/'current/search_rl_v3_overlap/pilot_seed0')
    parser.add_argument('--output', type=Path, default=trainer.ROOT/'current/search_rl_diagnosis_20261022/ablation')
    parser.add_argument('--seeds', nargs='+', type=int, default=[17,29])
    parser.add_argument('--episodes', type=int, default=128)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--start-index', type=int, default=1024)
    parser.add_argument('--horizon', type=int, default=5000)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--opponent-budget', type=int, default=200)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--validation-per-layout', type=int, default=10)
    parser.add_argument('--confirmation-per-layout', type=int, default=20)
    args = parser.parse_args()
    if args.episodes % args.batch or min(args.episodes,args.batch,args.workers) <= 0:
        parser.error('positive episodes must be a multiple of batch')
    if args.start_index+args.episodes > args.horizon:
        parser.error('diagnostic window exceeds the original horizon')
    args.source = str(args.source.resolve()); args.output = str(args.output.resolve())
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    signature = dict(config=vars(args), source_best=[sha(Path(args.source)/f'best_role{t}.pt') for t in range(2)],
                     distilled=sha(Path(args.source)/'distilled.pt'), trainer_hash=trainer.implementation_hash(),
                     runner_sha256=sha(__file__))
    if (out/'experiment.json').exists() and json.loads((out/'experiment.json').read_text()) != signature:
        raise ValueError('Output contains a different experiment; use a fresh directory')
    trainer.write_json(out/'experiment.json', signature)
    torch.set_num_threads(1); random.seed(20261022); np.random.seed(20261022); torch.manual_seed(20261022)
    initial = compose_initial(Path(args.source)); anchor = PolicyBundle.load(Path(args.source)/'distilled.pt')
    if not (out/'initial.pt').exists(): initial.save(out/'initial.pt')
    if not (out/'distilled.pt').exists(): anchor.save(out/'distilled.pt')
    trainer.write_json(out/'initial_manifest.json', dict(state_sha256=state_hash(initial), metadata=initial.metadata))
    with mp.get_context('spawn').Pool(args.workers,initializer=trainer.worker_init) if args.workers>1 else trainer._NoPool() as pool:
        train(args,out,initial,anchor,pool)
        validation = evaluate(args,out,pool,'validation',args.validation_per_layout,180)
        # Validation-only selection is frozen before the confirmation jobs run.
        val_summary = summary(validation)
        ranking = validation_ranking(val_summary)
        trainer.write_json(out/'selection_before_confirmation.json', dict(selected=ranking[0][2],ranking=ranking))
        confirmation = evaluate(args,out,pool,'confirmation',args.confirmation_per_layout,190)
    write_report(args,out,validation,confirmation)


if __name__ == '__main__':
    main()
