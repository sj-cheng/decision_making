"""Train and audit the public-information E50 leaf-value checkpoint.

The teacher is a transparent local pursuit score over belief support.  The
checkpoint is trained offline; closed-loop code only loads the resulting
NumPy weights and never receives simulator truth.
"""
import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
import repo_paths
repo_paths.install()
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from solvers.e50_leaf_value import LeafValueNet


def make_rows(seed=20261117, count=1600):
    rng = np.random.default_rng(seed)
    observations, endpoints, labels = [], [], []
    for _ in range(int(count)):
        pose = rng.uniform(-8., 8., 2)
        velocity = rng.uniform(-1., 1., 2)
        heading = float(rng.uniform(-np.pi, np.pi))
        has_belief = bool(rng.random() >= .35)
        targets = (pose + rng.uniform(-8., 8., (rng.integers(1, 3), 2))
                   if has_belief else np.zeros((0, 2)))
        targets = np.clip(targets, -9., 9.)
        belief = SimpleNamespace(particles=targets, weights=np.ones(len(targets)))
        obs = SimpleNamespace(pose=pose, velocity=velocity, heading=heading,
                              peer_messages=(), cycle=0, belief=belief)
        for _ in range(5):
            endpoint = np.clip(pose + rng.uniform(-2., 2., 2), -9., 9.)
            if len(targets):
                nearest = float(np.min(np.linalg.norm(targets - endpoint, axis=1)))
                label = -nearest - .02 * float(np.linalg.norm(endpoint - pose))
            else:
                label = .10 * float(np.linalg.norm(endpoint - pose))
            observations.append(obs); endpoints.append(endpoint); labels.append(label)
    return observations, np.asarray(endpoints), np.asarray(labels)


def run(output, seed=20261117, count=1600):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    obs, endpoints, labels = make_rows(seed, count)
    split = int(.8 * len(labels))
    order = np.arange(len(labels)); np.random.default_rng(seed + 1).shuffle(order)
    train_idx, test_idx = order[:split], order[split:]
    model = LeafValueNet.fit([obs[i] for i in train_idx], endpoints[train_idx], labels[train_idx])
    pred = np.asarray([model(obs[i], endpoints[i]) for i in test_idx])
    truth = labels[test_idx]
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    corr = float(np.corrcoef(pred, truth)[0, 1])
    # Ranking is the property consumed by MCTS candidate selection.
    rank = float(np.mean(np.argsort(np.argsort(pred)) == np.argsort(np.argsort(truth))))
    rank_corr = float(np.corrcoef(np.argsort(np.argsort(pred)),
                                  np.argsort(np.argsort(truth)))[0, 1])
    checkpoint = output / 'leaf_value_v1.npz'; model.save(checkpoint)
    result = dict(schema=LeafValueNet.SCHEMA, seed=int(seed), samples=int(len(labels)),
                  train_samples=int(len(train_idx)), test_samples=int(len(test_idx)),
                  feature_dim=int(len(model.weights)), test_rmse=rmse,
                  test_correlation=corr, exact_rank_fraction=rank,
                  rank_correlation=rank_corr,
                  checkpoint=str(checkpoint), teacher='nearest belief support plus motion regularizer',
                  simulator_truth_used=False)
    (output / 'REPORT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=20261117); p.add_argument('--count', type=int, default=1600)
    print(json.dumps(run(p.parse_args().output, p.parse_args().seed, p.parse_args().count), ensure_ascii=False))
