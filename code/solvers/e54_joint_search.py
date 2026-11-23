"""Two-pursuer joint-action PUCT with one synchronized target belief per node."""
from dataclasses import dataclass, field, replace
from itertools import product
import math
import time
from types import SimpleNamespace

import numpy as np

from solvers.e53_capture import capture_samples
from solvers.e52_hmcts import LocalSnapshot, Node
from solvers.e50_distributed_runtime import LocalPlan


@dataclass
class JointNode:
    states: tuple
    depth: int = 0
    edges: list = field(default_factory=list)
    expanded: bool = False
    visits: int = 0
    total: float = 0.
    reward: float = 0.
    bonus: float = 0.
    prior: float = 1.
    moves: tuple = ()


def root_states(observations, world):
    maps = [LocalSnapshot(observations[i].local_map) for i in range(2)]
    if world is None:
        tracks = {}
        for obs in observations.values():
            for tid, track in obs.belief.tracks.items():
                if tid not in tracks or track.timestamp > tracks[tid].timestamp:
                    tracks[tid] = track
        now = observations[0].timestamp
        points = [t.position + (now-t.timestamp)*t.velocity for t in tracks.values()]
        velocities = [t.velocity for t in tracks.values()]
        boxes = np.unique(np.concatenate([m.boxes for m in maps]), axis=0)
    else:
        boxes, points, velocities = world
    belief = SimpleNamespace(particles=np.asarray(points).reshape(-1, 2),
                             velocities=np.asarray(velocities).reshape(-1, 2),
                             weights=np.ones(len(points)), terminal=False)
    for snapshot in maps:
        snapshot.boxes = boxes
    return tuple(replace(observations[i], belief=belief, local_map=maps[i]) for i in range(2))


def expand(node, controllers, deadline):
    node.expanded = True
    roots = []
    for i, obs in enumerate(node.states):
        local = Node(obs, depth=node.depth)
        controllers[i]._expand(local, deadline, keep_prefix=True)
        roots.append(local)
    # Keep candidate pairs lazy: create a successor only on the first visit.
    node.edges = sorted(product(roots[0].children, roots[1].children),
                        key=lambda pair: -pair[0].prior*pair[1].prior)


def successor(node, moves, controllers):
    before = node.states[0].belief
    future = moves[0].obs.belief
    masks = [m.obs.belief.captured_mask for m in moves]
    captured = masks[0] | masks[1]
    if len(captured) > 1 and captured.any():
        paths = np.stack([m.prefix for m in moves], axis=1)
        times = np.linspace(0., 1., len(paths))[:, None, None]
        targets = np.clip(before.particles + times*before.velocities, -9., 9.)
        hits, _ = capture_samples(paths, targets, np.ones(2, bool),
                                  np.ones(len(captured), bool), node.states[0].local_map.boxes)
        captured = hits >= 0
    alive = ~captured
    belief = SimpleNamespace(particles=future.predictions[alive], velocities=before.velocities[alive],
                             weights=before.weights[alive], terminal=len(alive)>0 and not alive.any())
    states = tuple(replace(m.obs, belief=belief) for m in moves)
    reward = sum(m.reward for m in moves) + 4.*(captured.sum() - sum(m.sum() for m in masks))
    if len(alive):
        # Reward progress towards every tracked target using its nearest pursuer.
        def distances(states, targets):
            poses = np.stack([s.pose for s in states])
            return np.linalg.norm(poses[:, None] - targets, axis=2).min(axis=0)
        reward += (distances(node.states, before.particles)-distances(states, future.predictions)).mean()
    bonus = sum(controllers[i]._leaf_value(m)[0] for i,m in enumerate(moves)) if not belief.terminal else 0.
    return JointNode(states, depth=node.depth+1, moves=moves, reward=reward, bonus=bonus,
                     prior=moves[0].prior*moves[1].prior)


def joint_plans(observations, controllers, world=None):
    config = controllers[0]
    started = time.perf_counter()
    deadline = None if config.time_budget_ms is None else started + 2*config.time_budget_ms/1000
    root = JointNode(root_states(observations, world))
    for ctrl in controllers.values():
        ctrl.midpoint, ctrl.evaluations, ctrl.nae_queries = 50, 0, 0
    children = {}
    levels = [0]*(config.depth+1)
    for _ in range(config.high_simulations):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        path = []
        node = root
        while node.depth < config.depth and not node.states[0].belief.terminal:
            if not node.expanded:
                expand(node, controllers, deadline)
            if not node.edges:
                break
            visited = children.setdefault(id(node), [])
            if len(visited) < min(len(node.edges), math.ceil(math.sqrt(node.visits+1))):
                child = successor(node, node.edges[len(visited)], controllers)
                visited.append(child)
                levels[child.depth] += 1
            else:
                child = max(visited, key=lambda c: c.total/max(1,c.visits)
                            + 1.4*c.prior*math.sqrt(node.visits+1)/(1+c.visits))
            path.append(child)
            node = child
        if not path:
            break
        value = node.bonus
        for child in reversed(path):
            value = child.reward + config.gamma*value
            child.visits += 1
            child.total += value
        root.visits += 1
    best = max(children.get(id(root), []), key=lambda c:(c.visits,c.total), default=None)
    stats = dict(mode='joint_global' if world is not None else 'joint_shared',
                 simulations=root.visits, nodes_per_depth=levels, elapsed_ms=(time.perf_counter()-started)*1000)
    plans = {}
    for i,obs in observations.items():
        ctrl = controllers[i]
        ctrl.last_search = dict(evaluations=ctrl.evaluations, nae_queries=ctrl.nae_queries)
        if best is None:
            plans[i] = LocalPlan(i, obs.pose, np.zeros(2), 0., 0, -1, False)
            continue
        move = best.moves[i]
        ctrl.last_search['predicted_targets'] = move.obs.belief.predictions.tolist()
        plans[i] = LocalPlan(i, move.endpoint, move.endpoint-obs.pose, best.total/best.visits,
                             best.visits, move.option, duration=move.duration, pass_through=move.pass_through)
    return plans, stats
