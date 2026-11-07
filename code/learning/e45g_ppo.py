"""Public-belief shared actor with MAPPO or IPPO critic, no expert actions.

Both actors see exactly E45's public packet. MAPPO's training-only critic
additionally sees normalized true target coordinates; IPPO's critic does not.
PPO scores the original latent Gaussian sample. Deterministic map-only safety
shortening is part of the environment, not a rejection-conditioned density.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from learning.e45b_joint_network import KEYS, public_features
from solvers.frozen_d_search_interface import PublicMap
from solvers.e45f_support_refinement import RefinedTeamBelief, refined_posterior
from solvers.e45_team_adapter import team_hidden_mask
from run_frozen_d_closed_loop import legal_action
from l45_corner_shared_20261025.corner_estimator import geometry_labels, export_regions


class PublicEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        for name, n in [('maps', 4), ('agents', 6), ('particles', 3), ('history', 14)]:
            setattr(self, name, nn.Sequential(nn.Linear(n, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh()))
        self.trunk = nn.Sequential(nn.Linear(200, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())

    def forward(self, x):
        maps = (self.maps(x['maps']) * x['map_mask'][..., None]).sum(1)
        agents = self.agents(x['agents']).flatten(1)
        particles = (self.particles(x['particles']) * x['weights'][..., None]).sum(2).flatten(1)
        history = (self.history(x['history']) * x['history_mask'][..., None]).sum(1)
        history /= x['history_mask'].sum(1, keepdim=True).clamp_min(1)
        return self.trunk(torch.cat([maps, agents, particles, history, x['public']], -1))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PublicEncoder()
        self.head = nn.Sequential(nn.Linear(136, 64), nn.Tanh(), nn.Linear(64, 2))
        self.log_std = nn.Parameter(torch.full((2,), -.3))
        nn.init.orthogonal_(self.head[-1].weight, .01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        z = self.encoder(x)[:, None, :].expand(-1, 2, -1)
        identity = torch.eye(2, device=z.device)[None].expand(len(z), -1, -1)
        mean = self.head(torch.cat([z, x['agents'], identity], -1))
        return mean, self.log_std.clamp(-2.5, .5).exp().expand_as(mean)


class Critic(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.encoder = PublicEncoder()
        self.head = nn.Sequential(nn.Linear(136 + (4 if kind == 'MAPPO' else 0), 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('value_mean', torch.tensor(0.))
        self.register_buffer('value_scale', torch.tensor(1.))
        self.register_buffer('value_count', torch.tensor(1e-4))
        self.register_buffer('value_var', torch.tensor(1.))

    def normalized(self, x, privileged):
        z = self.encoder(x)[:, None, :].expand(-1, 2, -1)
        identity = torch.eye(2, device=z.device)[None].expand(len(z), -1, -1)
        parts = [z, x['agents'], identity]
        if self.kind == 'MAPPO':
            if privileged is None:
                raise ValueError('MAPPO critic requires training-only privileged state')
            parts.append(privileged[:, None, :].expand(-1, 2, -1))
        return self.head(torch.cat(parts, -1)).squeeze(-1)

    def forward(self, x, privileged=None):
        return self.normalized(x, privileged) * self.value_scale + self.value_mean

    @torch.no_grad()
    def update_normalization(self, returns):
        old_mean, old_scale = self.value_mean.clone(), self.value_scale.clone()
        n = returns.numel(); mean = returns.mean(); var = returns.var(unbiased=False)
        total = self.value_count + n; delta = mean - old_mean
        new_mean = old_mean + delta * n / total
        new_var = (self.value_var * self.value_count + var * n + delta.square() * self.value_count * n / total) / total
        new_scale = new_var.clamp_min(1e-4).sqrt()
        self.head[-1].weight.mul_(old_scale / new_scale)
        self.head[-1].bias.copy_((old_scale * self.head[-1].bias + old_mean - new_mean) / new_scale)
        self.value_mean.copy_(new_mean); self.value_scale.copy_(new_scale)
        self.value_var.copy_(new_var); self.value_count.copy_(total)


def latent_log_prob(mean, std, raw):
    return Normal(mean, std).log_prob(raw).sum(-1)


def gae(rewards, values, gamma=1., lam=.95):
    """Complete team episodes: retirements do not truncate team return credit."""
    result = np.zeros_like(values); carry = np.zeros(2)
    for t in range(len(rewards) - 1, -1, -1):
        nxt = values[t + 1] if t + 1 < len(values) else np.zeros(2)
        carry = rewards[t] + gamma * nxt - values[t] + gamma * lam * carry
        result[t] = carry
    return result, result + values


def execution_legal_action(at, action, speed, boxes):
    """Require the same endpoint to remain legal in Python and C++ float32."""
    endpoint = np.asarray(at) + np.asarray(action)
    if not np.all(endpoint >= -10.) or not np.all(endpoint <= 10.):
        return False
    if not legal_action(at, action, speed, boxes):
        return False
    point = endpoint.astype(np.float32)
    obstacles = np.asarray(boxes, dtype=np.float32).reshape(-1, 2, 2)
    return not bool(np.any(np.all((point[None, :] >= obstacles[:, :, 0])
                                  & (point[None, :] <= obstacles[:, :, 1]), axis=1)))


def safe_action(positions, normalized, active, speed, boxes):
    """Bounded, deterministic, target-free backtracking; every arm shares it."""
    result = np.zeros((2, 2)); changed = 0
    for i in range(2):
        if not active[i]:
            continue
        raw = np.clip(normalized[i], -1., 1.) * speed
        for k in range(14):
            candidate = raw * 2. ** (-k)
            if execution_legal_action(positions[i], candidate, speed, boxes):
                result[i] = candidate; changed += int(k > 0); break
        else:
            changed += 1
    return result, changed


class TeamBelief(RefinedTeamBelief):
    """Also recover failures raised inside frozen single-observer propagation.

    The F wrapper only repaired zero mass after the second observer. This
    process-local G wrapper reconstructs the same public negative history on
    the declared 0.25 lattice if the earlier propagation itself raises.
    """
    def update(self, obs):
        rngs = [e.rng.copy() for e in self.estimators]
        try:
            return super().update(obs)
        except RuntimeError as exc:
            if not str(exc).startswith(('Empty weighted public-history support', 'Empty team-conditioned finite-grid support')):
                raise
            recovered = []
            for j, e in enumerate(self.estimators):
                if not obs['active_e'][j]:
                    self.known[j] = None; continue
                positive = [q for i, q in enumerate(obs['reports'][j]) if obs['active_p'][i] and q is not None]
                if positive:
                    # A failed target cannot precede a new positive reset.
                    self.known[j] = np.asarray(positive[0], float)
                    from solvers.frozen_d_search_interface import Observation
                    at = next(i for i, q in enumerate(obs['reports'][j]) if obs['active_p'][i] and q is not None)
                    e.step = obs['step'] - 1
                    e.update(Observation(obs['step'], np.asarray(obs['positions'][at]), self.known[j], obs['remaining']))
                    continue
                result = refined_posterior(self.boxes, self.public_history, j)
                if result is None:
                    raise RuntimeError('G refinement also exhausted; original=' + str(exc)) from exc
                graph, weights, parents, observers = result
                e.graph, e.weights, e.parents = graph, weights, parents
                e.reachable = weights.sum(1) > 0; e.regions = geometry_labels(graph[0], self.boxes)
                e.observers = [o[0] for o in observers]; e.geometry_means = None
                positives = [q for item in self.public_history for i, q in enumerate(item['reports'][j])
                             if item['active_p'][i] and q is not None]
                e.last = np.asarray(positives[-1]); e.rng[:] = rngs[j]; e.step = obs['step']
                e.ids, e.particle_weights, e.labels = export_regions(weights, e.regions, e.rng, 64)
                e.particles = graph[0][e.ids].copy(); self.known[j] = None
                e.last_stats.update(support_points=int(e.reachable.sum()), mode_mass=weights.sum(0).tolist())
                recovered.append(dict(step=obs['step'], target=j, spacing=.25, support_points=int(e.reachable.sum()), entry='inner_propagation'))
            self.refinements.extend(recovered); self.step = obs['step']
            self.last_stats = dict(adaptive_grid_refinement=recovered, min_conditioning_mass=0.)


def install_belief(evidence):
    from solvers.e45f_support_refinement import install
    install(evidence)
    import solvers.e45_team_adapter as adapter
    import solvers.e45b_joint_nte as joint
    import run_e45b_role_free as base
    adapter.TeamBelief = joint.TeamBelief = base.TeamBelief = TeamBelief
    from solvers.e45f_geometry_capacity import geometric_candidates_capacity
    adapter.geometric_candidates = geometric_candidates_capacity


class Controller:
    def __init__(self, boxes, speed, seed, actor):
        self.public = PublicMap(boxes, speed); self.belief = TeamBelief(boxes, seed)
        self.actor = actor; self.actor.eval(); self.records = []; self.last_stats = {}
        self.rng = np.random.default_rng(seed + 94572030)

    def observe(self, obs):
        self.belief.update(obs); self.records.append(obs)

    def features(self, obs):
        samples = np.zeros((2, 64, 2)); weights = np.zeros((2, 64))
        for j, e in enumerate(self.belief.estimators):
            if obs['active_e'][j]:
                samples[j] = e.particles; weights[j] = e.particle_weights
        return public_features(self.public.boxes, np.asarray(obs['positions']), obs['active_p'], obs['active_e'],
                               samples, weights, self.belief.known, self.records, self.public.speed, obs['remaining'])

    @torch.no_grad()
    def sample(self, obs, critic=None, privileged=None, deterministic=False):
        self.observe(obs); features = self.features(obs)
        batch = {k: torch.as_tensor(v[None], dtype=torch.float32) for k, v in features.items()}
        mean, std = self.actor(batch)
        raw = mean.numpy()[0] if deterministic else mean.numpy()[0] + std.numpy()[0] * self.rng.standard_normal((2, 2))
        action, changed = safe_action(np.asarray(obs['positions']), np.tanh(raw), obs['active_p'], self.public.speed, self.public.boxes)
        logp = latent_log_prob(mean, std, torch.as_tensor(raw[None], dtype=torch.float32)).numpy()[0]
        values = critic(batch, torch.as_tensor(privileged[None], dtype=torch.float32) if privileged is not None else None).numpy()[0] if critic else np.zeros(2)
        self.last_stats = dict(simulations=0, joint_action_search=False, assignment_used=False, roles_used=False,
                               safety_shortened=changed, team_belief=self.belief.last_stats)
        return action, features, raw.astype(np.float32), logp, values

    def act(self, obs, fixed_simulations=None):
        return self.sample(obs, deterministic=True)[0]
