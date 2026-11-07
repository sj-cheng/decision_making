"""E46 public-information proposals, paired references and supervised students.

New experiment adapters only. Historical E45 controllers/data are not edited.
"""
from __future__ import annotations

import collections
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from learning.e45b_joint_network import Network, Runtime, public_features
from solvers.e45b_joint_nte import Controller, simultaneous_capture, ETA, TIME_COST
from solvers.e45f_baselines import GeometryController
from solvers.frozen_d_search_interface import move, safe, target_step

ROOT = Path(__file__).resolve().parents[2]
OLD_MODELS = ROOT / 'current/e45c_budget_scaling_20261028/learning/x20/round_12/models'


def geometric_options(own, node):
    """Same retained positions/routes as original GEOM; no assignment."""
    if hasattr(node, 'geometric_options'):
        return node.geometric_options
    own.features(node)
    p = own.public
    options = []
    for i in range(2):
        choices = [np.zeros(2)]
        if node.ap[i]:
            for j in range(2):
                if not node.ae[j]:
                    continue
                mass = np.asarray(node.weights[j])
                ids = np.unique(np.minimum(63, np.searchsorted(np.cumsum(mass), (np.arange(8)+.5)/8)))
                for k in ids:
                    own.allowed()
                    if mass[k] > 0:
                        choices.append(move(node.positions[i], node.samples[j, k], p.speed,
                                            p.vertices, p.dist, p.boxes, np.zeros(2, np.int64)))
            for vertex in p.vertices:
                own.allowed()
                choices.append(move(node.positions[i], vertex, p.speed, p.vertices,
                                    p.dist, p.boxes, np.zeros(2, np.int64)))
        unique = []
        for action in choices:
            if not any(np.max(np.abs(action-other)) < 1e-10 for other in unique):
                unique.append(action)
        options.append(np.asarray(unique))
    node.geometric_options = options
    return options


def draw_random(own, node):
    """Original non-neural global branch, including its amplitude mixture."""
    for _ in range(64):
        own.allowed()
        a = own.proposal_rng.uniform(-1, 1, (2, 2))
        if own.proposal_rng.random() < .5:
            a /= np.maximum(np.max(np.abs(a), axis=1, keepdims=True), 1e-12)
        if own.proposal_rng.random() < .02:
            a[:] = 0
        a *= own.public.speed
        a[~node.ap] = 0
        if legal(own, node, a):
            return a
        own.rejections += 1
    own.fallbacks += 1
    return np.zeros((2, 2))


def legal(own, node, a):
    return all(safe(node.positions[i], a[i], own.public.speed, own.motion_boxes,
                    np.zeros(2, np.int64)) for i in range(2))


def draw_geometry(own, node):
    choices = geometric_options(own, node)
    a = np.array([c[own.proposal_rng.integers(len(c))] for c in choices])
    a[~node.ap] = 0
    own.allowed()
    if legal(own, node, a):
        return a
    own.rejections += 1
    return draw_random(own, node)


class MixedController(Controller):
    """Explicit .25 random + .75 * (alpha network + (1-alpha) geometry)."""
    def __init__(self, *args, alpha=.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.source_counts = collections.Counter()

    def propose(self, node, direct=False):
        u = self.proposal_rng.random()
        if u >= .75:
            self.source_counts['random'] += 1
            return draw_random(self, node)
        if u >= .75*self.alpha:
            self.source_counts['geometry'] += 1
            return draw_geometry(self, node)
        self.source_counts['network'] += 1
        params = self.parameters(node)
        for _ in range(64):
            self.allowed()
            a = self.runtime.sample(params, self.proposal_rng)*self.public.speed
            a[~node.ap] = 0
            if legal(self, node, a):
                return a
            self.rejections += 1
        self.fallbacks += 1
        return np.zeros((2, 2))


def candidate_pool(own, node, size=16):
    """Bounded 16 complete joint actions, .75 geometry / .25 global draws."""
    actions = []
    for _ in range(size*64):
        own.allowed()
        a = draw_geometry(own, node) if own.proposal_rng.random() < .75 else draw_random(own, node)
        if not any(np.max(np.abs(a-b)) < 1e-10 for b in actions):
            actions.append(a)
        if len(actions) >= size:
            break
    if not actions:
        actions.append(np.zeros((2, 2)))
    return np.asarray(actions)


class RankNetwork(nn.Module):
    """Public state encoder + score of a complete 4-D joint candidate."""
    def __init__(self):
        super().__init__()
        self.encoder = Network('single')
        self.score = nn.Sequential(nn.Linear(self.encoder.width+4, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, actions):
        e = self.encoder
        m = (e.map_encoder(x['maps'])*x['map_mask'][..., None]).sum(1)
        a = e.agent_encoder(x['agents']).flatten(1)
        p = (e.particle_encoder(x['particles'])*x['weights'][..., None]).sum(2).flatten(1)
        h = (e.history_encoder(x['history'])*x['history_mask'][..., None]).sum(1)
        h /= x['history_mask'].sum(1, keepdim=True).clamp_min(1)
        z = e.trunk(torch.cat((m, a, p, h, x['public']), -1))
        return self.score(torch.cat((z[:, None].expand(-1, actions.shape[1], -1), actions), -1)).squeeze(-1)

    def export(self, path):
        np.savez_compressed(path, **{k:v.detach().cpu().numpy().astype(np.float64)
                                    for k,v in self.state_dict().items()})


class RankRuntime:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.w = {k:data[k] for k in data.files}

    def block(self, x, prefix):
        for i in (0, 2):
            x = np.maximum(0, x @ self.w[f'{prefix}.{i}.weight'].T + self.w[f'{prefix}.{i}.bias'])
        return x

    def __call__(self, x, actions):
        m = (self.block(x['maps'], 'encoder.map_encoder')*x['map_mask'][:,None]).sum(0)
        a = self.block(x['agents'], 'encoder.agent_encoder').reshape(-1)
        p = (self.block(x['particles'], 'encoder.particle_encoder')*x['weights'][...,None]).sum(1).reshape(-1)
        h = (self.block(x['history'], 'encoder.history_encoder')*x['history_mask'][:,None]).sum(0)/max(1,x['history_mask'].sum())
        z = self.block(np.r_[m,a,p,h,x['public']], 'encoder.trunk')
        v = np.concatenate((np.broadcast_to(z,(len(actions),len(z))),actions),axis=1)
        v = np.maximum(0,v@self.w['score.0.weight'].T+self.w['score.0.bias'])
        return (v@self.w['score.2.weight'].T+self.w['score.2.bias'])[:,0]


class PoolController(Controller):
    """Same bounded pool for uniform, geometric and learned priorities."""
    def __init__(self, *args, ordering='uniform', rank_path=None, force_root=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ordering = ordering
        self.ranker = RankRuntime(rank_path) if rank_path else None
        self.force_root = force_root

    def pool(self, node):
        if not hasattr(node, 'pool_actions'):
            node.pool_actions = candidate_pool(self, node)
            if self.ordering == 'learned':
                scores = self.ranker(self.features(node),node.pool_actions.reshape(-1,4)/self.public.speed)
                order = np.argsort(-scores,kind='stable')
            elif self.ordering == 'geometric':
                from solvers.e45b_joint_nte import coverage_potential
                self.features(node)
                p = self.public
                scores = []
                for action in node.pool_actions:
                    self.allowed()
                    scores.append(coverage_potential(node.positions+action,node.ap,node.ae,
                        node.samples,node.weights,p.speed,p.vertices,p.dist,p.boxes))
                order = np.argsort(-np.asarray(scores),kind='stable')
            else:
                order = self.proposal_rng.permutation(len(node.pool_actions))
            node.pool_order = order
            node.pool_next = 0
        return node.pool_actions

    def propose(self, node, direct=False):
        self.pool(node)
        k = int(node.pool_order[node.pool_next % len(node.pool_order)])
        node.pool_next += 1
        return node.pool_actions[k].copy()

    def select(self, node):
        if not (self.force_root and node is self.root):
            return super().select(node)
        if not node.actions:
            node.actions = list(self.pool(node).copy())
            node.visits = [0]*len(node.actions)
            node.totals = [0.]*len(node.actions)
        for i,n in enumerate(node.visits):
            if not n:
                return i
        n = np.asarray(node.visits)
        return int(np.argmax(np.asarray(node.totals)/n+.1*np.sqrt(np.log(node.n+1)/n)))


def make_controller(case, spec):
    kind = spec['arm']
    seed = int(spec['seed'])
    model_seed = int(spec.get('train_seed',17))
    args = (case['map'],case['speed'],seed)
    kwargs = dict(budget_ms=float(spec.get('budget_ms',200)),use_value=False)
    if kind == 'GEOM':
        own = GeometryController(*args,use_policy=False,**kwargs)
    elif kind == 'OLD_JOINT':
        own = Controller(*args,OLD_MODELS/f'joint_{model_seed}.npz',use_policy=True,
                         use_value=True,budget_ms=kwargs['budget_ms'])
    elif kind in ('PRIOR','STUDENT'):
        own = Controller(*args,spec.get('checkpoint') or OLD_MODELS/f'single_{model_seed}.npz',
                         use_policy=True,**kwargs)
    elif kind == 'MIX':
        own = MixedController(*args,spec.get('checkpoint') or OLD_MODELS/f'single_{model_seed}.npz',
                              use_policy=True,alpha=spec['alpha'],**kwargs)
    elif kind.startswith('POOL_'):
        own = PoolController(*args,use_policy=False,ordering=kind[5:].lower(),
                             rank_path=spec.get('checkpoint'),force_root=spec.get('force_root',False),**kwargs)
    else:
        raise ValueError(kind)
    if 'search_seed' in spec:
        s=int(spec['search_seed'])
        own.rng[:]=(s+1)%4294967296
        own.proposal_rng=np.random.default_rng(s+2)
        own.world_rng=np.random.default_rng(s+3)
    return own


def prepare_public(packet, spec, simulations=0):
    """Only map, speed, public records and explicit RNG seed are accepted."""
    case = dict(map=packet['map'],speed=packet['speed'])
    started = time.perf_counter()
    own = make_controller(case,spec)
    construction = 1000*(time.perf_counter()-started)
    started = time.perf_counter()
    for obs in packet['records'][:-1]:
        own.observe(obs)
    history = 1000*(time.perf_counter()-started)
    action = own.act(packet['records'][-1],fixed_simulations=simulations)
    return own,action,dict(construction_ms=construction,history_ms=history)


def root_labels(own):
    root = own.root
    actions = np.zeros((16,4));weights = np.zeros(16)
    visits = np.asarray(root.visits)
    keep = np.flatnonzero(visits>0)
    if not len(keep):
        raise RuntimeError('No completed teacher root simulations; no fabricated zero label')
    actions[:len(keep)] = np.asarray(root.actions)[keep].reshape(-1,4)/own.public.speed
    weights[:len(keep)] = visits[keep]/visits[keep].sum()
    return actions,weights


def reference_values(packet, actions, worlds, seed, horizon=8):
    """Paired posterior-world 8-step reference, public Single17 continuation.

    Hidden indices only generate target transitions, captures and observations.
    No model/real hidden state is passed into the continuation policy.
    Selection worlds and reporting worlds use independent seed namespaces.
    """
    spec = dict(arm='PRIOR',seed=packet['seed'],train_seed=17)
    own,_,_ = prepare_public(packet,spec,0)
    own.deadline = float('inf')
    root = own.root
    own.features(root)
    initial_phi = own.potential(root)
    result = np.zeros((len(actions),worlds))
    for ai,first in enumerate(actions):
        for wi in range(worlds):
            rng_seed = int(np.random.SeedSequence([seed,wi]).generate_state(1)[0])
            own.world_rng = np.random.default_rng(rng_seed)
            own.proposal_rng = np.random.default_rng(rng_seed+1)
            own.export_rng = np.array([(rng_seed+4)%4294967296],np.int64)
            transition_rng = np.array([(rng_seed+2)%4294967296],np.int64)
            ids,modes = own.sample_world()
            root.children.clear()
            node = root
            elapsed = 0
            complete = False
            for depth in range(min(horizon,own.remaining)):
                a = np.asarray(first) if depth == 0 else own.propose(node,direct=True)
                if not legal(own,node,a):
                    raise RuntimeError('Illegal candidate in public reference')
                targets = np.zeros((2,2))
                for j,g in enumerate(own.graphs):
                    if node.ae[j]:
                        ids[j],modes[j] = target_step(ids[j],modes[j],g[1],g[2],g[3],transition_rng)
                        targets[j] = g[0][ids[j]]
                positions = node.positions+a
                ap,ae,visible = simultaneous_capture(positions,targets,node.ap,node.ae,own.public.boxes)
                elapsed += 1
                complete = not ae.any()
                if complete or depth+1 >= own.remaining:
                    result[ai,wi] = float(complete)-TIME_COST*elapsed-ETA*initial_phi
                    break
                node.actions = [a]
                observed = np.array([ids[j] if visible[j]>=0 else visible[j] for j in range(2)])
                node,_ = own.child(node,0,positions,ap,ae,observed)
            else:
                result[ai,wi] = -TIME_COST*elapsed+ETA*(own.potential(node)-initial_phi)
    root.children.clear()
    return result
