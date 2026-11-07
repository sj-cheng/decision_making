"""Read existing guarded-S artifacts; never run search, rollouts or updates."""
import json
from pathlib import Path

import numpy as np
import torch

import run_l45_guarded_supervision as m
from analyze_l45_search_feedback import distributions, fit_error


def main():
    torch.set_num_threads(1)
    args = m.parse_args([])
    m.verify(args)
    directory = args.output / 'training/S/seed0'
    records, training = [], []
    for manifest_path in sorted((directory / 'batches').glob('*.json')):
        if manifest_path.name.endswith('.lock.json'):
            continue
        manifest = m.locked_read(manifest_path)
        episodes = [m.read_rollout(task) for task in manifest['tasks']]
        start = episodes[0]['snapshot_episode']
        source = m.o.load_torch(manifest['tasks'][0]['snapshot'])
        before_checkpoint = sorted((directory / 'checkpoints').glob(f'*_ep{start:06d}.pt'))[-1]
        checkpoint = sorted((directory / 'checkpoints').glob(f'*_ep{start+64:06d}.pt'))[-1]
        before_saved, after_saved = map(m.o.load_torch, [before_checkpoint, checkpoint])
        assert m.equal_tree(source['actors'], before_saved['actors'])
        before = m.o.load_actors(m.o.original_problem(), states=source['actors'])
        after = m.o.load_actors(m.o.original_problem(), states=after_saved['actors'])
        rows = [row for ep in episodes for row in ep['rows']]
        mu0, sd0, limits, active = distributions(before, rows)
        mu1, sd1, _, _ = distributions(after, rows)
        kl = (((sd1/sd0).log() + .5*((sd0/sd1).square()-1.)
               + .5*((mu0-mu1)/sd1).square()).sum(-1)*active).sum(-1)
        teaching = m.locked_read(directory / f'teachers/batch{start:06d}.json')
        labels = [label for task in teaching['tasks'] for label in m.read_teacher(task)['labels']]
        positions = {(ep['index'], step): offset+step
                     for offset, ep in m._episode_offsets(episodes) for step in range(len(ep['rows']))}
        indices = torch.tensor([positions[(label['index'], label['step'])] for label in labels])
        targets = torch.tensor([label['target'] for label in labels])
        e0 = fit_error(mu0[indices], targets, limits[indices], active[indices])
        e1 = fit_error(mu1[indices], targets, limits[indices], active[indices])
        metric = m.o.read_json(directory / f'updates/{start+64:06d}.json')
        assert abs(float(kl.mean())-metric['final_exact_kl']) < 1e-5
        assert abs(float(e1.mean())-metric['search_loss']) < 1e-6
        # Recover the exact minibatch permutations from the committed RNG.
        # Forward/backward/Adam in this runner use no random draws.
        generator = torch.Generator()
        generator.set_state(before_saved['rng']['torch'])
        orders = [torch.randperm(len(rows), generator=generator) for _ in range(4)]
        assert torch.equal(generator.get_state(), after_saved['rng']['torch'])
        labelled = set(indices.tolist())
        used = set()
        for step in metric['accepted_step_trace']:
            if step['accepted']:
                offset = step['minibatch']*256
                used.update(labelled.intersection(orders[step['epoch']][offset:offset+256].tolist()))
        phases = {}
        for phase, count in [('before_first', 2), ('after_first', 1)]:
            mask = active.sum(-1) == count
            label_mask = mask[indices]
            phases[phase] = dict(rows=int(mask.sum()), labels=int(label_mask.sum()),
                mse_before=float(e0[label_mask].mean()), mse_after=float(e1[label_mask].mean()),
                mean_kl=float(kl[mask].mean()), max_state_kl=float(kl[mask].max()))
        records.append(dict(episode=start+64, samples=len(rows), labels=len(labels),
            labels_in_accepted_gradients=len(used), mse_before=float(e0.mean()), mse_after=float(e1.mean()),
            relative_fit_improvement=float(1.-e1.mean()/e0.mean()), phases=phases,
            actor_steps=metric['actor_steps'], actor_epochs=metric['actor_epochs'],
            checkpoint_sha256=m.o.sha(checkpoint)))
        training.extend(dict(opponent=ep['opponent'], **ep['stats']) for ep in episodes)

    banks = m.locked_read(args.output / 'scenarios.json')
    selection = m.o.read_json(directory / 'selection.json')
    model = next(r['model'] for r in selection['records'] if r['episode'] == 512)
    tasks = m.requests(args, m.controls(args)+[model], banks['validation_four'])
    indexed = {}
    for task in tasks:
        row = m.read_cached(args, task)
        indexed.setdefault(row['uid'], {})[(row['episode_hash'], row['opponent'])] = row
    assert all(len(rows) == 300 for rows in indexed.values())
    old, new = indexed[m.CONTROL], indexed['S0']
    assert old.keys() == new.keys()
    paired = {}
    for group in ['all', *m.o.OPPONENTS]:
        keys = [key for key in old if group == 'all' or key[1] == group]
        gained = [key for key in keys if not old[key]['legal_double_capture'] and new[key]['legal_double_capture']]
        lost = [key for key in keys if old[key]['legal_double_capture'] and not new[key]['legal_double_capture']]
        paired[group] = dict(n=len(keys), gained=len(gained), lost=len(lost),
            losses_with_zero_capture=sum(new[key]['capture_count'] == 0 for key in lost),
            losses_with_one_capture_timeout=sum(new[key]['one_capture_timeout'] for key in lost))
    assert paired['all']['gained'] == 28 and paired['all']['lost'] == 23
    result = dict(scope='Retrospective diagnostics only; no new environment, search or optimizer calls.',
        source_results_sha256=m.o.sha(args.output / 'results.json'), batches=records,
        queried_labels=sum(r['labels'] for r in records),
        labels_in_accepted_gradients=sum(r['labels_in_accepted_gradients'] for r in records),
        mean_within_batch_relative_fit_improvement=float(np.mean([r['relative_fit_improvement'] for r in records])),
        training_after_first_steps=sum(r['phases']['after_first']['rows'] for r in records),
        training={op: dict(n=sum(r['opponent'] == op for r in training),
            successes=sum(r['legal_double_capture'] for r in training if r['opponent'] == op))
            for op in m.o.OPPONENTS}, paired_validation=paired,
        limitations=['Used-label counts measure gradient exposure, not an independent effective sample size.',
            'Fit errors use training labels, not held-out labels or deployment-state labels.',
            'Average batch KL is not a per-state or initial-to-final deployment guarantee.',
            'Observed phase failure counts do not identify which phase caused the outcome.'])
    target = m.ROOT / 'current/l45_experiment_retrospective_20261023'
    target.mkdir(exist_ok=True)
    m.o.write_json(target / 'guarded_diagnostics.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
