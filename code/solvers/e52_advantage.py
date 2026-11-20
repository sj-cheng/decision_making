"""Public-input features and a small model for matched branch advantages."""
from pathlib import Path

import numpy as np


def public_state(obs):
    belief = obs.belief
    tracks = list(getattr(belief, 'tracks', {}).values())
    particles = np.asarray(belief.particles).reshape(-1, 2)
    velocities = np.asarray(getattr(belief, 'velocities', [t.velocity for t in tracks])).reshape(-1, 2)
    if len(velocities) != len(particles):
        velocities = np.zeros_like(particles)
    return dict(
        pose=np.asarray(obs.pose).tolist(), velocity=np.asarray(obs.velocity).tolist(),
        acceleration=np.zeros(2).tolist() if obs.acceleration is None else np.asarray(obs.acceleration).tolist(),
        heading=float(obs.heading), cycle=int(obs.cycle),
        boxes=np.asarray(obs.local_map.obstacle_boxes()).tolist(),
        tracks=[dict(position=p.tolist(), velocity=v.tolist(), confidence=float(w))
                for p,v,w in zip(particles, velocities, belief.weights)],
        peers=[dict(pose=m.pose.tolist(), velocity=m.velocity.tolist(), age=obs.cycle-m.cycle)
               for m in obs.peer_messages],
    )


def features(state, endpoints, options, durations):
    """Return one row per action. Use the same transformation in training and search."""
    pose = np.asarray(state['pose'])
    delta = np.asarray(endpoints) - pose
    boxes = np.asarray(state['boxes']).reshape(-1, 2, 2) - pose[None, :, None]
    tracks = np.asarray([t['position'] for t in state['tracks']]).reshape(-1, 2) - pose
    n = len(delta)
    reach = np.linalg.norm(delta, axis=1)
    heading = float(state['heading'])
    common = [*state['velocity'], *state['acceleration'], np.sin(heading), np.cos(heading),
              min(len(tracks), 2)]
    if state['peers']:
        peer = state['peers'][0]
        common += [*(np.asarray(peer['pose'])-pose), *peer['velocity'], min(peer['age'], 10)/10., 1.]
    else:
        common += [0.] * 6
    if len(tracks):
        distances = np.linalg.norm(tracks[None] - delta[:, None], axis=-1)
        nearest = distances.argmin(axis=1)
        velocities = np.asarray([t['velocity'] for t in state['tracks']])
        weights = np.asarray([t['confidence'] for t in state['tracks']])
        target_features = np.column_stack((tracks[nearest], velocities[nearest],
                                           distances[np.arange(n), nearest], weights[nearest]))
    else:
        target_features = np.zeros((n, 6))
    if len(boxes):
        # Point-to-box clearance at several fractions of the candidate chord.
        # This is context for the critic; MINCO still checks the actual curve.
        points = delta[:, None, None, :] * np.array([0., .25, .5, .75, 1.])[None, :, None, None]
        gap = np.maximum(np.maximum(boxes[None, None, :, :, 0]-points,
                                   points-boxes[None, None, :, :, 1]), 0.)
        clearance = np.linalg.norm(gap, axis=-1).min(axis=-1).clip(max=5.)
    else:
        clearance = np.full((n, 5), 5.)
    return np.column_stack((delta, reach, durations, np.eye(4)[options],
                            np.tile(common, (n, 1)), target_features, clearance)).astype(np.float32)


class AdvantageMLP:
    schema = 'e52_branch_advantage_v1'

    def __init__(self, width=32, architecture='mlp'):
        import torch
        self.torch = torch
        self.width = width
        self.architecture = architecture
        if architecture == 'mlp':
            self.model = torch.nn.Sequential(
                torch.nn.Linear(width, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1),
            )
        elif architecture == 'transformer' and width == 32:
            self.model = _token_model(torch.nn)
        else:
            raise ValueError(f'unsupported advantage architecture: {architecture}, width {width}')
        self.center = torch.zeros(width)
        self.scale = torch.ones(width)

    def forward(self, rows):
        return self.model((rows-self.center)/self.scale).squeeze(-1)

    def rank_actions(self, observation, endpoints, options, durations):
        matrix = features(public_state(observation), endpoints, options, durations)
        with self.torch.inference_mode():
            return self.forward(self.torch.from_numpy(matrix)).numpy()

    @classmethod
    def load(cls, path):
        import torch
        saved = torch.load(Path(path), map_location='cpu', weights_only=True)
        if saved['schema'] != cls.schema:
            raise ValueError('unsupported E52 branch advantage checkpoint')
        model = cls(saved['width'], saved.get('architecture', 'mlp'))
        model.model.load_state_dict(saved['model'])
        model.center, model.scale = saved['center'], saved['scale']
        model.model.eval()
        return model


def _token_model(nn):
    """Use attention over existing feature groups without adding information."""
    import torch

    class TokenModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Action, ego, teammate, tracked target, and local clearance.
            self.sizes = (8, 7, 6, 6, 5)
            self.projections = nn.ModuleList(nn.Linear(size, 32) for size in self.sizes)
            self.encoder = nn.TransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=64, dropout=0.,
                batch_first=True,
            )
            self.output = nn.Linear(32, 1)

        def forward(self, x):
            tokens = torch.stack([project(block) for project,block in
                                  zip(self.projections, x.split(self.sizes, dim=-1))], dim=1)
            return self.output(self.encoder(tokens)[:, 0])

    return TokenModel()
