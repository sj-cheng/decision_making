"""E45-I public-observation/APF adapters for the unmodified author RMAPPO.

Upstream owns actor, critic, distributions, optimizers and PPO updates. This
module owns the task adapter and complete-episode recurrent minibatches.
"""
from pathlib import Path
import sys
import numpy as np
import torch

UPSTREAM = Path(__file__).resolve().parents[2] / 'third_party/mappo_on_policy'
sys.path.insert(0, str(UPSTREAM))
from onpolicy.config import get_config
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from learning.e45b_joint_network import KEYS, public_features
from learning.e45g_ppo import TeamBelief, safe_action, gae
from solvers.frozen_d_search_interface import PublicMap
from solvers.e45_team_adapter import optimal_assignment

OBS_DIM = 814
CENT_DIM = 816
HIDDEN = 64
DEFAULT_APF = np.array([1., .8, .2, .6], dtype=np.float64)


def settings():
    args = get_config().parse_args([])
    for key, value in dict(algorithm_name='rmappo', hidden_size=HIDDEN,
        use_recurrent_policy=True, use_naive_recurrent_policy=False,
        recurrent_N=1, data_chunk_length=10, ppo_epoch=5, num_mini_batch=2,
        use_ReLU=False, use_popart=False, use_valuenorm=False,
        use_policy_active_masks=True, use_value_active_masks=True,
        use_feature_normalization=True, lr=3e-4, critic_lr=3e-4,
        entropy_coef=.01, clip_param=.2, max_grad_norm=.5,
        use_wandb=False, gamma=1., gae_lambda=.95).items():
        setattr(args, key, value)
    return args


def policy(kind, device='cpu'):
    # The author code dispatches by the Box class name, not by gym version.
    from gym.spaces import Box
    dimensions = 4 if kind == 'MAPPO_APF' else 2
    args = settings()
    model = R_MAPPOPolicy(args, Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32),
        Box(-np.inf, np.inf, (CENT_DIM,), dtype=np.float32),
        Box(-np.inf, np.inf, (dimensions,), dtype=np.float32), torch.device(device))
    return model, R_MAPPO(args, model, torch.device(device))


def actor_inputs(features):
    public = np.concatenate([np.asarray(features[k], np.float32).ravel() for k in KEYS])
    result = np.c_[np.repeat(public[None], 2, axis=0), np.eye(2, dtype=np.float32)]
    assert result.shape == (2, OBS_DIM)
    return result.astype(np.float32)


def critic_inputs(observations, privileged):
    # Both agents estimate the SAME team value, including after one retires.
    # No identity or hidden target coordinates enter the deployed actor.
    common = np.r_[observations[0, :-2], np.asarray(privileged, np.float32)]
    return np.repeat(common[None], 2, axis=0).astype(np.float32)


class PublicState:
    def __init__(self, boxes, speed, seed):
        self.public = PublicMap(boxes, speed)
        self.belief = TeamBelief(boxes, seed)
        self.records = []
        self.last_stats = {}

    def observe(self, obs):
        self.belief.update(obs)
        self.records.append(obs)

    def features(self, obs):
        samples = np.zeros((2, 64, 2)); weights = np.zeros((2, 64))
        for j, estimator in enumerate(self.belief.estimators):
            if obs['active_e'][j]:
                samples[j] = estimator.particles; weights[j] = estimator.particle_weights
        return public_features(self.public.boxes, np.asarray(obs['positions']),
            obs['active_p'], obs['active_e'], samples, weights, self.belief.known,
            self.records, self.public.speed, obs['remaining'])


def repulsion(at, boxes, goal):
    """Bounded AABB/boundary repulsion and goal-oriented tangential field."""
    away = np.zeros(2); tangent = np.zeros(2)
    pairs = []
    for box in boxes:
        closest = np.clip(at, box[:, 0], box[:, 1])
        delta = at - closest
        pairs.append((delta, float(np.linalg.norm(delta))))
    for axis in range(2):
        for sign in (-1., 1.):
            delta = np.zeros(2); delta[axis] = -sign
            pairs.append((delta, 10. - sign * at[axis]))
    for delta, distance in pairs:
        if distance >= 2.5:
            continue
        distance = max(distance, .08)
        normal = delta / max(np.linalg.norm(delta), 1e-12)
        strength = min(8., max(0., 1. / distance - 1. / 2.5) / distance)
        away += normal * strength
        turn = np.array([-normal[1], normal[0]])
        if np.dot(turn, goal - at) < 0:
            turn = -turn
        tangent += turn * strength
    return away, tangent


def apf_action(own, obs, raw=None, coefficients=None):
    """Public-belief assignment + attraction/repulsion/tangent APF adaptation.

    This is not AG fallback or a claimed full DACOOP/OPEN reproduction.
    Physical actions use exactly the common G/H deterministic legal mapping.
    """
    positions = np.asarray(obs['positions'], float)
    goals = np.zeros((2, 2)); cost = np.full((2, 2), 1e9)
    for j in np.flatnonzero(obs['active_e']):
        estimator = own.belief.estimators[j]
        known = own.belief.known[j]
        goals[j] = known if known is not None else np.average(
            estimator.particles, axis=0, weights=estimator.particle_weights)
        for i in np.flatnonzero(obs['active_p']):
            cost[i, j] = np.average(np.linalg.norm(estimator.particles - positions[i], axis=1),
                                    weights=estimator.particle_weights)
    assignment = optimal_assignment(cost, obs['active_p'], obs['active_e'])
    coefficients = DEFAULT_APF if coefficients is None else np.asarray(coefficients)
    gains = np.tile(coefficients, (2, 1))
    if raw is not None:
        gains *= np.exp(np.log(4.) * np.tanh(raw))
    normalized = np.zeros((2, 2))
    for i, j in assignment.items():
        pull = goals[j] - positions[i]
        pull /= max(np.linalg.norm(pull), 1e-12)
        away, tangent = repulsion(positions[i], own.public.boxes, goals[j])
        separation = np.zeros(2)
        for other in np.flatnonzero(obs['active_p']):
            if other != i:
                delta = positions[i] - positions[other]
                distance = max(np.linalg.norm(delta), .1)
                if distance < 2.:
                    separation += delta / distance * min(4., 1. / distance - .5)
        force = gains[i, 0] * pull + gains[i, 1] * away + gains[i, 2] * separation + gains[i, 3] * tangent
        # The task bounds EACH displacement coordinate (L-infinity speed).
        # Euclidean-unit attraction would needlessly slow diagonal pursuit.
        amplitude = min(1., np.max(np.abs(goals[j]-positions[i])) / own.public.speed)
        normalized[i] = amplitude * force / max(1e-12, np.max(np.abs(force)))
    action, shortened = safe_action(positions, normalized, obs['active_p'], own.public.speed, own.public.boxes)
    own.last_stats = dict(simulations=0, joint_action_search=False, assignment_used=True,
        assignment={str(i):int(j) for i, j in assignment.items()}, roles_used=False,
        safety_shortened=shortened, apf_coefficients=gains.tolist(),
        team_belief=own.belief.last_stats)
    return action


class APFController(PublicState):
    def __init__(self, boxes, speed, seed, coefficients=None):
        super().__init__(boxes, speed, seed)
        self.coefficients = coefficients

    def act(self, obs, fixed_simulations=None):
        self.observe(obs)
        return apf_action(self, obs, coefficients=self.coefficients)


class RecurrentController(PublicState):
    def __init__(self, boxes, speed, seed, actor, kind, deterministic=True, coefficients=None):
        super().__init__(boxes, speed, seed)
        self.actor = actor; self.actor.eval(); self.kind = kind
        self.deterministic = deterministic; self.coefficients = coefficients
        self.hidden = np.zeros((2, 1, HIDDEN), np.float32)
        self.samples = []

    @torch.no_grad()
    def observe(self, obs):
        # Causal prefix observations warm the actor memory; no actions execute.
        super().observe(obs)
        _, _, hidden = self.actor(actor_inputs(self.features(obs)), self.hidden,
                                  np.ones((2, 1), np.float32), deterministic=True)
        self.hidden = hidden.numpy()

    @torch.no_grad()
    def act(self, obs, fixed_simulations=None):
        PublicState.observe(self, obs)
        features = self.features(obs); inputs = actor_inputs(features)
        before = self.hidden.copy(); masks = np.ones((2, 1), np.float32)
        raw, logp, hidden = self.actor(inputs, before, masks, deterministic=self.deterministic)
        self.hidden = hidden.numpy(); raw = raw.numpy(); logp = logp.numpy()
        if self.kind == 'MAPPO_APF':
            action = apf_action(self, obs, raw, self.coefficients)
        else:
            action, changed = safe_action(np.asarray(obs['positions']), np.tanh(raw),
                obs['active_p'], self.public.speed, self.public.boxes)
            self.last_stats = dict(simulations=0, joint_action_search=False, assignment_used=False,
                roles_used=False, safety_shortened=changed, team_belief=self.belief.last_stats)
        self.samples.append(dict(obs=inputs, actor_hidden=before, raw=raw, logp=logp,
            active=np.asarray(obs['active_p'], np.float32)[:, None]))
        if not self.deterministic:
            from solvers.e45b_joint_nte import coverage_potential
            self.samples[-1]['phi'] = float(coverage_potential(
                np.asarray(obs['positions']),np.asarray(obs['active_p']),np.asarray(obs['active_e']),
                features['particles'][:,:,:2]*20+features['public'][:2]*10,
                features['weights'],self.public.speed,self.public.vertices,self.public.dist,self.public.boxes))
        return action


class EpisodeBuffer:
    """Exact complete-episode returns; padded chunks never cross episodes.

    The upstream trainer consumes this through its public buffer interface.
    Padding is masked in BOTH losses. Fully retired actor chunks are excluded.
    Every active real sample is visited exactly once in each PPO epoch.
    """
    def __init__(self, episodes):
        self.episodes = episodes
        self.lengths = [len(e['obs']) for e in episodes]
        for name in ('returns', 'value_preds', 'active_masks'):
            key = {'active_masks':'active', 'value_preds':'values'}.get(name, name)
            value = np.concatenate([e[key] for e in episodes])
            setattr(self, name, np.concatenate([value, np.zeros_like(value[:1])]))

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        chunks = []; offset = 0
        keys = ('central', 'obs', 'raw', 'values', 'returns', 'active', 'logp')
        for episode, length in zip(self.episodes, self.lengths):
            for agent in range(2):
                for start in range(0, length, data_chunk_length):
                    stop = min(start + data_chunk_length, length); size = stop - start
                    if not episode['active'][start:stop, agent].any():
                        continue
                    item = {}
                    for key in keys:
                        data = episode[key][start:stop, agent]
                        item[key] = np.pad(data, [(0, data_chunk_length-size)] + [(0,0)]*(data.ndim-1))
                    item['adv'] = np.pad(advantages[offset+start:offset+stop, agent], ((0,data_chunk_length-size),(0,0)))
                    item['masks'] = np.r_[np.ones(size), np.zeros(data_chunk_length-size)].astype(np.float32)[:,None]
                    item['actor_hidden'] = episode['actor_hidden'][start, agent]
                    item['critic_hidden'] = episode['critic_hidden'][start, agent]
                    chunks.append(item)
            offset += length
        if len(chunks) < num_mini_batch:
            raise RuntimeError('Not enough active recurrent chunks for registered minibatches')
        for ids in np.array_split(torch.randperm(len(chunks)).numpy(), num_mini_batch):
            selected = [chunks[i] for i in ids]
            def flat(key):
                value = np.stack([item[key] for item in selected], axis=1)
                return value.reshape(-1, *value.shape[2:]).astype(np.float32)
            yield (flat('central'), flat('obs'),
                np.stack([item['actor_hidden'] for item in selected]),
                np.stack([item['critic_hidden'] for item in selected]),
                flat('raw'), flat('values'), flat('returns'), flat('masks'),
                flat('active'), flat('logp'), flat('adv'), None)


@torch.no_grad()
def finish_episodes(episodes, model):
    """Replay centralized critic in parallel; privileged state is training-only."""
    device = model.device; count = len(episodes)
    hidden = torch.zeros(count*2, 1, HIDDEN, device=device)
    for episode in episodes:
        n = len(episode['obs'])
        episode['central'] = np.stack([critic_inputs(x,p) for x,p in zip(episode['obs'],episode['privileged'])])
        episode['values'] = np.zeros((n,2,1), np.float32)
        episode['critic_hidden'] = np.zeros((n,2,1,HIDDEN), np.float32)
    for step in range(max(len(e['obs']) for e in episodes)):
        central = np.zeros((count,2,CENT_DIM), np.float32); masks = np.zeros((count,2,1),np.float32)
        before = hidden.cpu().numpy().reshape(count,2,1,HIDDEN)
        for i,e in enumerate(episodes):
            if step < len(e['obs']):
                central[i] = e['central'][step]; masks[i] = 1
                e['critic_hidden'][step] = before[i]
        values,hidden = model.critic(central.reshape(-1,CENT_DIM),hidden,masks.reshape(-1,1))
        values = values.cpu().numpy().reshape(count,2,1)
        for i,e in enumerate(episodes):
            if step < len(e['obs']):
                e['values'][step] = values[i]
    for e in episodes:
        _, returns = gae(e['reward'][:,None], e['values'][...,0], gamma=1., lam=.95)
        e['returns'] = returns[...,None]
    return EpisodeBuffer(episodes)
