"""Warm-start compatible actors and CTDE critics for the local protocol."""
import copy
import hashlib
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from local_observation import BASE_DIM, OBS_DIM, SCHEMA, PROTOCOL
from learning.feedforward import FeedForward

CRITIC_DIM = 29 + 4 * 5 + 2 * OBS_DIM


class LocalActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.psi = FeedForward([
            ['Linear', OBS_DIM, 64], ['Linear', 64, 64],
            ['Linear', 64, 64], ['Linear', 64, 4]], 'relu', 'cpu')
        self._numpy = None

    def warm_start(self, path):
        old = torch.load(path, map_location='cpu', weights_only=False)
        new = self.state_dict()
        for key in new:
            if key == 'psi.layers.0.weight':
                if old[key].shape[1] != BASE_DIM:
                    raise ValueError('Warm start requires the V7 270-input checkpoint')
                new[key].zero_(); new[key][:, :BASE_DIM] = old[key]
            else:
                new[key] = old[key]
        self.load_state_dict(new)
        self.refresh()
        return self

    def distribution(self, obs, limits):
        raw = self.psi(obs)
        center = (limits[..., 0] + limits[..., 1]) * .5
        half = (limits[..., 1] - limits[..., 0]) * .5
        mean = center + half * torch.tanh(raw[..., :2])
        return Normal(mean, torch.exp(.5 * raw[..., 2:].clamp(-8., 2.)))

    def refresh(self):
        self._numpy = [(layer.weight.detach().cpu().numpy().copy(), layer.bias.detach().cpu().numpy().copy()) for layer in self.psi.layers]

    def numpy_parameters(self, obs, limits):
        if self._numpy is None:
            self.refresh()
        x = np.asarray(obs, np.float32)
        for i, (weight, bias) in enumerate(self._numpy):
            x = x @ weight.T + bias
            if i + 1 < len(self._numpy):
                x = np.maximum(x, 0.)
        limits = np.asarray(limits)
        mean = .5 * (limits[..., 0] + limits[..., 1]) + .5 * (limits[..., 1] - limits[..., 0]) * np.tanh(x[..., :2])
        std = np.exp(.5 * np.clip(x[..., 2:], -8., 2.))
        return mean, std

    def action(self, obs, limits, rng=None):
        mean, std = self.numpy_parameters(obs, limits)
        raw = mean if rng is None else mean + std * rng.normal(size=mean.shape)
        logp = np.sum(-.5 * ((raw - mean) / std) ** 2 - np.log(std) - .5 * np.log(2 * np.pi), axis=-1)
        return np.clip(raw, np.asarray(limits)[..., 0], np.asarray(limits)[..., 1]), raw, logp


class LocalCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(CRITIC_DIM, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def critic_features(problem, state, actor_features):
    physical = np.asarray(state[:29, 0], np.float32).copy()
    physical[:8] /= 10.; physical[8] /= 80.; physical[13:] /= 2.4
    geometry = np.zeros((4, 5), np.float32)
    if len(problem.obstacles) > 4:
        raise ValueError('Critic map encoding supports up to four obstacles')
    for i, box in enumerate(problem.obstacles):
        geometry[i] = np.r_[np.asarray(box).reshape(-1) / 10., 1.]
    return np.r_[physical, geometry.reshape(-1), np.asarray(actor_features).reshape(-1)].astype(np.float32)


class RunningValueNorm:
    def __init__(self):
        self.count = 0; self.mean = 0.; self.m2 = 0.

    @property
    def std(self):
        return max((self.m2 / max(self.count, 1)) ** .5, .1) if self.count else 1.

    def update(self, values):
        values = np.asarray(values, np.float64).reshape(-1)
        if not len(values):
            return
        n, mean, m2 = len(values), float(values.mean()), float(((values - values.mean()) ** 2).sum())
        total = self.count + n; delta = mean - self.mean
        self.m2 += m2 + delta * delta * self.count * n / total
        self.mean += delta * n / total; self.count = total

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


def advantages(rewards, values, dones, gamma=1., lam=.95, bootstrap=0.):
    rewards = np.asarray(rewards, np.float32); values = np.asarray(values, np.float32)
    adv = np.zeros_like(rewards); carry = 0.
    for i in reversed(range(len(rewards))):
        nxt = bootstrap if i + 1 == len(rewards) else values[i + 1]
        live = 1. - float(dones[i])
        delta = rewards[i] + gamma * nxt * live - values[i]
        carry = delta + gamma * lam * live * carry
        adv[i] = carry
    return adv, adv + values


def joint_log_prob(actors, obs, raw, limits, active, team):
    return sum(actors[robot].distribution(obs[:, local], limits[:, local]).log_prob(raw[:, local]).sum(-1) * active[:, local]
               for local, robot in enumerate((0, 1) if team == 0 else (2, 3)))


def atomic_torch_save(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp); os.replace(tmp, path)


class PolicyBundle:
    def __init__(self, model_dir=None, iteration=45, scratch=False):
        self.actors = [LocalActor() for _ in range(4)]
        self.critics = [LocalCritic() for _ in range(2)]
        self.norms = [RunningValueNorm(), RunningValueNorm()]
        self.metadata = dict(protocol=PROTOCOL, observation_schema=SCHEMA, obs_dim=OBS_DIM,
                             action_distribution='physical_normal_then_clip_v1',
                             reward='fixed_team_mean_environment_v1', gamma=1., value_validated=False)
        if not scratch:
            if model_dir is None:
                model_dir = Path(__file__).resolve().parents[2] / 'current/memory_observation_v3/models'
            for robot, actor in enumerate(self.actors):
                path=Path(model_dir) / f'model_policy_l{iteration}_i{robot}.pt'
                actor.warm_start(path)
                self.metadata.setdefault('warm_start',[]).append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        self.refresh()

    def refresh(self):
        for actor in self.actors:
            actor.refresh()

    def state_dict(self):
        return dict(actors=[a.state_dict() for a in self.actors], critics=[c.state_dict() for c in self.critics],
                    norms=[copy.deepcopy(n.__dict__) for n in self.norms], metadata=copy.deepcopy(self.metadata))

    def load_state_dict(self, state):
        if state['metadata'].get('observation_schema') != SCHEMA or state['metadata'].get('protocol') != PROTOCOL:
            raise ValueError('Incompatible local policy checkpoint')
        for a, s in zip(self.actors, state['actors']):
            a.load_state_dict(s)
        for c, s in zip(self.critics, state['critics']):
            c.load_state_dict(s)
        for n, s in zip(self.norms, state['norms']):
            n.__dict__.update(s)
        self.metadata = copy.deepcopy(state['metadata']); self.refresh()

    def save(self, path):
        atomic_torch_save(self.state_dict(), path)

    @classmethod
    def load(cls, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        result = cls(scratch=True); result.load_state_dict(state.get('bundle', state)); return result

    def copy(self):
        result = PolicyBundle(scratch=True); result.load_state_dict(copy.deepcopy(self.state_dict())); return result


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
