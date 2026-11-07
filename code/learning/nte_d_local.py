"""Small public-belief policy/value networks, research-only NTE adaptation.

All features are supplied as public sets. Hidden states and teacher scores are
absent from this module's inference interface. NumPy inference avoids PyTorch
thread/device setup inside timed decisions.
"""
import numpy as np
import torch
from torch import nn


def public_features(boxes, positions, weights, at, records, speed, remaining,
                    planning_remaining, seen_once):
    at = np.asarray(at, dtype=np.float64)
    positions, weights = np.asarray(positions), np.asarray(weights)
    if positions.shape != (64, 2) or weights.shape != (64,):
        raise ValueError('Expected 64 weighted public positions')
    if (not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError('Invalid public probability weights')
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 2, 2)
    maps = np.zeros((8, 4), np.float64)
    map_mask = np.zeros(8, np.float64)
    if len(b) > 8:
        raise ValueError('Map exceeds preregistered capacity')
    maps[:len(b), :2] = (b.mean(axis=2)-at)/20.
    maps[:len(b), 2:] = (b[:, :, 1]-b[:, :, 0])/20.
    map_mask[:len(b)] = 1.
    particles = np.column_stack(((positions-at)/20., weights))
    history, history_mask = np.zeros((17, 7)), np.zeros(17)
    selected = list(records[-16:])
    old = [r for r in records[:-16] if r['target'] is not None]
    if old:
        selected = [old[-1]]+selected
    step = 80-int(remaining)
    for i, r in enumerate(selected):
        history[i, 0] = (step-r['step'])/80.
        history[i, 1:3] = (np.asarray(r['position'])-at)/20.
        history[i, 3] = float(r['target'] is not None)
        if r['target'] is not None:
            history[i, 4:6] = (np.asarray(r['target'])-at)/20.
        history[i, 6], history_mask[i] = 1., 1.
    target = records[-1]['target'] if records else None
    public = np.array([at[0]/10., at[1]/10., speed/2., remaining/80.,
                       float(target is not None), 0., 0., float(seen_once),
                       planning_remaining/8.])
    if target is not None:
        public[5:7] = (np.asarray(target)-at)/20.
    return dict(maps=maps, map_mask=map_mask, particles=particles,
                weights=weights.copy(), history=history, history_mask=history_mask,
                public=public)


class Network(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind not in ('policy', 'value'):
            raise ValueError(kind)
        self.kind = kind
        self.map_encoder = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.particle_encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.history_encoder = nn.Sequential(nn.Linear(7, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(105, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                  nn.Linear(64, 4 if kind == 'policy' else 1))

    def forward(self, x):
        m = (self.map_encoder(x['maps'])*x['map_mask'].unsqueeze(-1)).sum(1)
        p = (self.particle_encoder(x['particles'])*x['weights'].unsqueeze(-1)).sum(1)
        h = (self.history_encoder(x['history'])*x['history_mask'].unsqueeze(-1)).sum(1)
        y = self.head(torch.cat((m, p, h, x['public']), dim=-1))
        return (y[:, :2].tanh(), y[:, 2:].clamp(-6., 1.)) if self.kind == 'policy' else 2.*y[:, 0].sigmoid()

    def export(self, path):
        values = {'kind': np.array(self.kind)}
        for name, value in self.state_dict().items():
            values[name] = value.detach().cpu().numpy().astype(np.float64)
        np.savez_compressed(path, **values)


class Runtime:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.kind = str(data['kind'])
            self.arrays = {k: np.ascontiguousarray(data[k]) for k in data.files if k != 'kind'}

    def _block(self, x, name, layers):
        for i in range(layers):
            index = 2*i
            x = x@self.arrays[f'{name}.{index}.weight'].T+self.arrays[f'{name}.{index}.bias']
            if name != 'head' or i < layers-1:
                x = np.maximum(0., x)
        return x

    def __call__(self, x):
        a = (self._block(x['maps'], 'map_encoder', 2)*x['map_mask'][:, None]).sum(0)
        b = (self._block(x['particles'], 'particle_encoder', 2)*x['weights'][:, None]).sum(0)
        c = (self._block(x['history'], 'history_encoder', 2)*x['history_mask'][:, None]).sum(0)
        y = self._block(np.concatenate((a, b, c, x['public'])), 'head', 3)
        if self.kind == 'policy':
            return np.tanh(y[:2]), np.clip(y[2:], -6., 1.)
        return float(2./(1.+np.exp(-np.clip(y[0], -700., 700.))))
