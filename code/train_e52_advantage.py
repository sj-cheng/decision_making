"""Fit candidate advantages on train maps, select on validation, score test once."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from solvers.e52_advantage import AdvantageMLP, features
from solvers.e52_hmcts import HierarchicalMCTS


def group_arrays(group):
    records = [r for r in group.get('records', ()) if r['first_action_feasible']]
    if len(records) < 2:
        return None
    state = deepcopy(group['public'])
    # Earlier traces stored the last measurement and its cycle rather than
    # the predicted/decayed belief seen by the online feature encoder.
    for track in state['tracks']:
        if 'timestamp' in track:
            age = state['cycle'] - track['timestamp']
            track['position'] = (np.asarray(track['position']) + age*np.asarray(track['velocity'])).tolist()
            track['confidence'] *= np.exp(-.3*age)
    x = features(state, [r['endpoint'] for r in records],
                 [HierarchicalMCTS.OPTIONS.index(r['option']) for r in records],
                 [r['duration'] for r in records])
    y = np.asarray([r['advantage'] for r in records], np.float32)
    return x, y


def prepare(groups):
    batches = {split: [] for split in ('train', 'validation', 'test')}
    maps = {split: set() for split in batches}
    for group in groups:
        arrays = group_arrays(group)
        if arrays is None:
            continue
        split = group['split']
        x, y = arrays
        batches[split].append((x, y))
        maps[split].add(group['map_hash'])
    if not all(batches.values()) or any(
        maps[a]&maps[b] for a,b in (('train','validation'),('train','test'),('validation','test'))
    ):
        raise ValueError('need nonempty groups with disjoint map splits')
    return batches


def metrics(model, groups):
    torch = model.torch
    regrets, baseline, wins, total, mse = [], [], 0, 0, []
    with torch.inference_mode():
        for x, y in groups:
            prediction = model.forward(torch.from_numpy(x)).numpy()
            regrets.append(float(max(y) - y[prediction.argmax()]))
            baseline.append(float(max(y)))  # geometric reference has zero advantage
            mse.extend((y-prediction)**2)
            target_delta = y[:,None]-y[None,:]
            predicted_delta = prediction[:,None]-prediction[None,:]
            mask = target_delta > .05
            wins += int((predicted_delta[mask]>0).sum())
            total += int(mask.sum())
    return dict(groups=len(groups), mean_regret=float(np.mean(regrets)),
                geometry_mean_regret=float(np.mean(baseline)), mse=float(np.mean(mse)),
                ranked_pairs=total, pair_accuracy=wins/total if total else None)


def test_breakdown(model, groups):
    by_condition = {}
    for group in groups:
        if group['split'] != 'test':
            continue
        arrays = group_arrays(group)
        if arrays is not None:
            by_condition.setdefault(group['condition'], []).append(arrays)
    unknown = by_condition.get('unknown', [])
    stationary_selected = stationary_suboptimal = 0
    with model.torch.inference_mode():
        for x, y in unknown:
            best = int(model.forward(model.torch.from_numpy(x)).argmax())
            if x[best,2] < .05:
                stationary_selected += 1
                stationary_suboptimal += bool(max(y)-y[best] > .05)
    return dict(by_condition={k: metrics(model,v) for k,v in by_condition.items()},
                no_belief_predictions=len(unknown),
                no_belief_stationary_selected=stationary_selected,
                no_belief_stationary_suboptimal=stationary_suboptimal)


def train(batches, seed, epochs, architecture='mlp'):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = AdvantageMLP(batches['train'][0][0].shape[1], architecture)
    rows = torch.from_numpy(np.concatenate([x for x,_ in batches['train']]))
    model.center = rows.mean(dim=0)
    model.scale = rows.std(dim=0).clamp(min=.1)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=.001, weight_decay=.01)
    rng = np.random.default_rng(seed)
    best = None
    for epoch in range(epochs):
        model.model.train()
        order = rng.permutation(len(batches['train']))
        for offset in range(0, len(order), 16):
            losses = []
            for idx in order[offset:offset+16]:
                x, y = batches['train'][idx]
                t = torch.from_numpy(y)
                p = model.forward(torch.from_numpy(x))
                delta = t[:,None]-t[None,:]
                mask = delta>.05
                loss = torch.nn.functional.smooth_l1_loss(p, t)
                if mask.any():
                    difference = p[:,None]-p[None,:]
                    loss += .25*torch.nn.functional.softplus(-difference[mask]).mean()
                losses.append(loss)
            optimizer.zero_grad()
            torch.stack(losses).mean().backward()
            optimizer.step()
        model.model.eval()
        result = metrics(model, batches['validation'])
        if best is None or (result['mean_regret'], result['mse']) < best[:2]:
            best = result['mean_regret'], result['mse'], epoch+1, deepcopy(model.model.state_dict())
    model.model.load_state_dict(best[3])
    return model, best[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--architecture', choices=('mlp', 'transformer'), default='mlp')
    a = p.parse_args()
    data = json.loads(a.dataset.read_text())
    if data['state'] != 'complete':
        raise SystemExit('finish candidate collection before training')
    batches = prepare(data['groups'])
    model, epoch = train(batches, 52, a.epochs, a.architecture)
    report = dict(selected_epoch=epoch, dataset=str(a.dataset), seed=52, architecture=model.architecture,
                  test_breakdown=test_breakdown(model, data['groups']),
                  **{s: metrics(model, b) for s,b in batches.items()})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    model.torch.save(dict(schema=model.schema, width=model.width, architecture=model.architecture,
                         model=model.model.state_dict(),
                         center=model.center, scale=model.scale), a.output)
    a.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
