"""Frozen E46 methods with public-node keyed candidate pools and direct controls.

This adapter does not modify any E46 source or checkpoint. Pool sampling, ordering
and rollout random streams are separate; matching public nodes share a menu.
"""
from __future__ import annotations

import hashlib
import json
import time
import collections

import numpy as np

from learning.e46_geometry_teacher import (
    PoolController, candidate_pool, make_controller as old_controller,
)
from solvers.e45b_joint_nte import coverage_potential


def stable_seed(*values):
    raw = json.dumps(values, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:4], 'little')


def pool_hash(actions):
    a = np.asarray(actions, dtype='<f8')
    return hashlib.sha256(a.tobytes()).hexdigest()


def public_node_key(node):
    if not hasattr(node, 'followup_public_key'):
        # Records and the team's simulated positions/masks are public at a node;
        # no sampled hidden indices, controller names, or network seeds occur.
        node.followup_public_key = stable_seed(
            node.records, node.positions.tolist(), node.ap.tolist(), node.ae.tolist(),
        )
    return node.followup_public_key


class SharedPoolController(PoolController):
    def __init__(self, *args, injected_root=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.injected_root = None if injected_root is None else np.asarray(injected_root, float)
        self.expansion_order = []

    def features(self, node):
        if node.features is None and node.samples is None:
            old = self.export_rng
            self.export_rng = np.array([stable_seed(public_node_key(node), 'export') or 1], np.int64)
            try:
                return super().features(node)
            finally:
                self.export_rng = old
        return super().features(node)

    def pool(self, node):
        if hasattr(node, 'pool_actions'):
            return node.pool_actions
        key = public_node_key(node)
        old = self.proposal_rng
        self.proposal_rng = np.random.default_rng(stable_seed(key, 'candidate-pool'))
        try:
            actions = (self.injected_root.copy()
                       if node is self.root and self.injected_root is not None
                       else candidate_pool(self, node))
        finally:
            self.proposal_rng = old
        if self.ordering == 'learned':
            scores = self.ranker(self.features(node), actions.reshape(-1, 4) / self.public.speed)
            order = np.argsort(-scores, kind='stable')
        elif self.ordering == 'geometric':
            self.features(node)
            p = self.public
            scores = []
            for action in actions:
                self.allowed()
                scores.append(coverage_potential(
                    node.positions + action, node.ap, node.ae, node.samples,
                    node.weights, p.speed, p.vertices, p.dist, p.boxes))
            scores = np.asarray(scores)
            order = np.argsort(-scores, kind='stable')
        else:
            scores = np.zeros(len(actions))
            order = np.random.default_rng(stable_seed(key, 'uniform-order')).permutation(len(actions))
        node.pool_actions, node.pool_order, node.pool_next = actions, order, 0
        node.pool_scores = np.asarray(scores)
        return actions

    def propose(self, node, direct=False):
        action = super().propose(node, direct)
        if node is self.root:
            self.expansion_order.append(int(node.pool_order[(node.pool_next-1) % len(node.pool_order)]))
        return action

    def act(self, obs, fixed_simulations=None):
        self.expansion_order = []
        action = super().act(obs, fixed_simulations)
        root = self.root
        self.last_stats['root_expansion_order'] = list(self.expansion_order)
        if hasattr(root, 'pool_actions'):
            self.last_stats.update(
                root_pool_sha256=pool_hash(root.pool_actions),
                root_pool_order=root.pool_order.tolist(),
                root_pool_scores=root.pool_scores.tolist(),
                root_pool_actions=root.pool_actions.reshape(-1, 4).tolist(),
            )
        self.last_stats['actual_max_depth'] = max(
            (len(t) for t in self.traces), default=0)
        self.last_stats['shared_pool_version'] = 'public-node-keyed-v1'
        return action


def instrument(own):
    totals = collections.defaultdict(float)
    counts = collections.Counter()
    for name in ('observe', 'features', 'potential', 'prepare_models', 'pool',
                 'parameters', 'propose', 'trajectory'):
        if not hasattr(own, name):
            continue
        fn = getattr(own, name)
        def wrapped(*args, _fn=fn, _name=name, **kwargs):
            start = time.perf_counter()
            try:
                return _fn(*args, **kwargs)
            finally:
                totals[_name] += 1000*(time.perf_counter()-start)
                counts[_name] += 1
        setattr(own, name, wrapped)
    act = own.act
    def timed(*args, **kwargs):
        totals.clear(); counts.clear()
        start = time.perf_counter()
        result = act(*args, **kwargs)
        own.last_stats.update(
            component_totals_ms=dict(totals), component_calls=dict(counts),
            wrapper_ms=1000*(time.perf_counter()-start),
            component_scope='nested inclusive timings; do not add',
        )
        return result
    own.act = timed
    return own


def make_controller(case, spec):
    arm = spec['arm']
    if arm in ('A1', 'A2', 'A3', 'A4', 'A5'):
        ordering = dict(A1='uniform', A2='geometric', A3='learned',
                        A4='geometric', A5='learned')[arm]
        own = SharedPoolController(
            case['map'], case['speed'], int(spec['seed']),
            ordering=ordering, rank_path=spec.get('checkpoint'),
            use_policy=False, use_value=False, direct=arm in ('A4', 'A5'),
            budget_ms=float(spec.get('budget_ms', 200)),
            injected_root=spec.get('injected_root'),
        )
    else:
        converted = dict(spec, arm={'R1': 'GEOM', 'R2': 'PRIOR'}.get(arm, arm))
        own = old_controller(case, converted)
    if spec.get('search_seed') is not None:
        seed = int(spec['search_seed'])
        own.rng[:] = (seed+1) % 4294967296
        own.world_rng = np.random.default_rng(seed+3)
        own.proposal_rng = np.random.default_rng(seed+2)
    if spec.get('profile'):
        own = instrument(own)
    return own


def prepare_public(packet, spec, simulations=0):
    start = time.perf_counter()
    own = make_controller(dict(map=packet['map'], speed=packet['speed']), spec)
    construction = 1000*(time.perf_counter()-start)
    start = time.perf_counter()
    for obs in packet['records'][:-1]:
        own.observe(obs)
    history = 1000*(time.perf_counter()-start)
    action = own.act(packet['records'][-1], fixed_simulations=simulations)
    return own, action, dict(construction_ms=construction, history_ms=history)
