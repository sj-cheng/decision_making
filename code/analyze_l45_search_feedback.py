#!/usr/bin/env python3
"""Read-only analysis of saved S/J batches; no search, environment rollout or update."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

import run_l45_search_feedback as m


def distributions(actors, rows):
    obs = torch.tensor(np.asarray([r['obs'] for r in rows]), dtype=torch.float32)
    limits = torch.tensor(np.asarray([r['limits'] for r in rows]), dtype=torch.float32)
    active = torch.tensor(np.asarray([r['active'] for r in rows]), dtype=torch.float32)
    with torch.no_grad():
        dists = [m.o.distribution(actors[robot], obs[:, local], limits[:, local])
                 for local, robot in enumerate(m.o.TEAMS[1])]
    return (torch.stack([d.mean for d in dists], 1),
            torch.stack([d.stddev for d in dists], 1), limits, active)


def fit_error(means, targets, limits, active):
    half = .5 * (limits[..., 1] - limits[..., 0])
    per_robot = ((means - targets) / half).square().mean(-1)
    return (per_robot * active).sum(-1) / active.sum(-1)


def gradient_check(actors, critic, episodes, labels):
    rows = [r for ep in episodes for r in ep['rows']]
    positions = {(ep['index'], step): offset + step
                 for offset, ep in m._episode_offsets(episodes)
                 for step in range(len(ep['rows']))}
    tensor = lambda key: torch.tensor(np.asarray([r[key] for r in rows]), dtype=torch.float32)
    obs, limits, active, raw = (tensor(k) for k in ('obs', 'limits', 'active', 'raw'))
    labelled = torch.zeros(len(rows), dtype=torch.bool)
    targets = torch.zeros_like(raw)
    for label in labels:
        idx = positions[(label['index'], label['step'])]
        labelled[idx] = True
        targets[idx] = torch.tensor(label['target'])
    with torch.no_grad():
        values = critic(tensor('critic')).numpy()
    advantages = []
    for offset, ep in m._episode_offsets(episodes):
        part = ep['rows']
        adv, _ = m.o.advantages([r['reward'] for r in part],
                              values[offset:offset+len(part)], [r['done'] for r in part])
        advantages.extend(adv)
    adv = torch.tensor(np.asarray(advantages), dtype=torch.float32)
    adv = (adv-adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
    ratio = (m.o.joint_log_prob(actors, obs, raw, limits, active)-tensor('old_logp')).exp()
    entropy = sum((m.o.distribution(actors[robot], obs[:, local], limits[:, local]).entropy().sum(-1)
                   * active[:, local]).sum() / active[:, local].sum().clamp_min(1.)
                  for local, robot in enumerate(m.o.TEAMS[1]))*.5
    ppo = -torch.minimum(ratio*adv, ratio.clamp(.9, 1.1)*adv).mean() - .001*entropy
    search = m.search_loss(actors, obs, limits, active, targets, labelled)
    params = [p for robot in m.o.TEAMS[1] for p in actors[robot].parameters()]
    def vector(loss):
        gradients = torch.autograd.grad(loss, params, allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                          for p, g in zip(params, gradients)])
    gs, gp = vector(search), vector(ppo)
    return dict(scope='initial full batch only; gradients, no optimizer step',
                search_gradient_norm=float(gs.norm()), ppo_gradient_norm=float(gp.norm()),
                weighted_ppo_to_search_norm=float(.1*gp.norm()/gs.norm().clamp_min(1e-12)),
                cosine=float(torch.dot(gs, gp)/(gs.norm()*gp.norm()).clamp_min(1e-12)))


def evaluation_check(args):
    banks = m.o.read_json(args.output/'scenarios.json')
    scenes = banks['validation_four'] + banks['validation_none']
    stages = [('references', 'validation_controls', m.controls(args))]
    for arm in m.ARMS:
        selection = m.o.read_json(args.output/f'training/{arm}/seed0/selection.json')
        for entry in selection['records']:
            if entry['episode']:
                label = f"{arm}{entry['episode']}"
                stages.append((label, f"training/{arm}/seed0/validation_ep{entry['episode']}", [entry['model']]))
    allrows = {}
    for label, folder, models in stages:
        tasks = m.requests(args, models, scenes)
        manifest = m.locked_read(args.output/folder/'manifest.json')
        if {(r['key'], r['uid']) for r in tasks} != {(r['key'], r['uid']) for r in manifest}:
            raise ValueError('Validation manifest changed')
        for task in tasks:
            row = m.read_cached(args, task)
            uid = row['uid'] if label == 'references' else label
            allrows.setdefault(uid, {})[(row['layout'], row['episode_hash'], row['opponent'])] = row
    summary, inversions, different = {}, [], []
    for uid, indexed in allrows.items():
        part = [r for k, r in indexed.items() if k[0] == 'four']
        failed = [r for r in part if not r['legal_double_capture']]
        summary[uid] = dict(games=len(part), success=sum(r['legal_double_capture'] for r in part),
                            failed=len(failed), one_capture_timeout=sum(r['one_capture_timeout'] for r in failed),
                            strong_opponent_failures=sum(r['opponent'] == 'Search Only' for r in failed),
                            own_illegal=sum(r['pursuer_illegal'] for r in part))
    for left, right in itertools.combinations(allrows, 2):
        for key in sorted(set(allrows[left]) & set(allrows[right])):
            a, b = allrows[left][key], allrows[right][key]
            if key[0] != 'four' or any(r['pursuer_illegal'] or r['evader_illegal'] for r in [a, b]) or a['legal_double_capture'] == b['legal_double_capture']:
                continue
            winner, loser = (a, b) if a['legal_double_capture'] else (b, a)
            wn, ln = (left, right) if a['legal_double_capture'] else (right, left)
            row = dict(episode_hash=key[1], opponent=key[2], successful_model=wn, failed_model=ln,
                       successful_return=winner['pursuer_return'], failed_return=loser['pursuer_return'],
                       successful_first_capture_time=winner['first_capture_time'],
                       successful_finish_time=winner['capture_time'], failed_first_capture_time=loser['first_capture_time'])
            different.append(row)
            if loser['pursuer_return'] > winner['pursuer_return']:
                inversions.append(row)
    return dict(scope='existing validation only; correlated model-pair observations, not independent confirmation',
                summary=summary, discordant_model_pair_observations=len(different),
                failed_higher_return_pair_observations=len(inversions),
                distinct_physical_cases_with_inversion=len({r['episode_hash'] for r in inversions}),
                inversions=inversions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=m.DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output = args.output.resolve()
    torch.set_num_threads(1)
    protocol_args = m.parse_args(['--output', str(args.output)])
    m.verify(protocol_args)
    records, gradient = [], None
    for arm in m.ARMS:
        directory = args.output / f'training/{arm}/seed0'
        for manifest_path in sorted((directory/'batches').glob('*.json')):
            if manifest_path.name.endswith('.lock.json'):
                continue
            manifest = m.locked_read(manifest_path)
            episodes = [m.read_rollout(t) for t in manifest['tasks']]
            before_ep = episodes[0]['snapshot_episode']
            source = m.o.load_torch(manifest['tasks'][0]['snapshot'])
            checkpoint = sorted((directory/'checkpoints').glob(f'*_ep{before_ep+64:06d}.pt'))[0]
            updated = m.o.load_torch(checkpoint)
            before = m.o.load_actors(m.o.original_problem(), states=source['actors'])
            after = m.o.load_actors(m.o.original_problem(), states=updated['actors'])
            rows = [r for ep in episodes for r in ep['rows']]
            old_mu, old_sd, limits, active = distributions(before, rows)
            new_mu, new_sd, _, _ = distributions(after, rows)
            mean_kl = .5*((old_mu-new_mu)/new_sd).square()
            variance_kl = (new_sd/old_sd).log()+.5*((old_sd/new_sd).square()-1.)
            reduce = lambda x: float((x.sum(-1)*active).sum(-1).mean())
            half = .5*(limits[..., 1]-limits[..., 0])
            mask = active.bool().unsqueeze(-1).expand_as(old_sd)
            teaching = m.locked_read(directory/f'teachers/batch{before_ep:06d}.json')
            labels = [label for task in teaching['tasks'] for label in m.read_teacher(task)['labels']]
            positions = {(ep['index'], step): offset+step for offset, ep in m._episode_offsets(episodes)
                         for step in range(len(ep['rows']))}
            indices = torch.tensor([positions[(r['index'], r['step'])] for r in labels])
            targets = torch.tensor([r['target'] for r in labels])
            old_error = fit_error(old_mu[indices], targets, limits[indices], active[indices])
            new_error = fit_error(new_mu[indices], targets, limits[indices], active[indices])
            phase_mask = active[indices].sum(-1) == 2
            normalized_shift = fit_error(new_mu, old_mu, limits, active).mean().sqrt()
            metric = json.loads((directory/f'updates/{before_ep+64:06d}.json').read_text())
            record = dict(arm=arm, episode=before_ep+64,
                          source_sha256=m.o.sha(manifest['tasks'][0]['snapshot']),
                          checkpoint_sha256=m.o.sha(checkpoint),
                          mse_before=float(old_error.mean()), mse_after=float(new_error.mean()),
                          mse_relative_change=float(new_error.mean()/old_error.mean()-1.),
                          phases={name: dict(n=int(mask_.sum()), before=float(old_error[mask_].mean()),
                                             after=float(new_error[mask_].mean()))
                                  for name, mask_ in [('before_first', phase_mask), ('after_first', ~phase_mask)]},
                          mean_kl_component=reduce(mean_kl), variance_kl_component=reduce(variance_kl),
                          exact_kl=reduce(mean_kl+variance_kl),
                          old_std_median_relative_to_half_range=float(torch.median((old_sd/half)[mask])),
                          new_std_median_relative_to_half_range=float(torch.median((new_sd/half)[mask])),
                          normalized_mean_rmse=float(normalized_shift),
                          recorded_kl=metric['exact_kl_by_epoch'][-1], actor_epochs=metric['actor_epochs'])
            if abs(record['exact_kl']-record['recorded_kl']) > max(1e-5, abs(record['exact_kl'])*1e-4):
                raise ValueError('Recomputed KL does not reproduce the saved update')
            records.append(record)
            print(json.dumps(record), flush=True)
            if arm == 'S' and before_ep == 0:
                critic = m.o.TrainingCritic()
                critic.load_state_dict(m.o.load_torch(args.output/'initial/source.pt')['critic'])
                gradient = gradient_check(before, critic, episodes, labels)
    result = dict(source='saved pilot data only', no_environment_episodes=True,
                  no_search_queries=True, no_optimizer_updates=True,
                  records=records, initial_gradient_check=gradient)
    target = args.output/'review'
    target.mkdir(exist_ok=True)
    m.o.write_json(target/'update_diagnostics.json', result)
    evaluation = evaluation_check(protocol_args)
    m.o.write_json(target/'evaluation_diagnostics.json', evaluation)
    print('GRADIENT', gradient)
    print('EVALUATION', {k: v for k, v in evaluation.items() if k != 'inversions'})


if __name__ == '__main__':
    main()
