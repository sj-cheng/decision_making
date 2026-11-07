"""Research-only weighted frozen-D adapter and finite-depth open-loop UCT probe.

No environment object or hidden state is accepted. The immutable estimator is
loaded in its private namespace. This module does not claim search efficacy.
"""
from pathlib import Path
import importlib.util
import time
import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT/'current/l45_corner_turn_estimator_20261025/frozen_estimator'
spec = importlib.util.spec_from_file_location('accepted_d_loader', FROZEN/'loader.py')
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
loader.verify()
loader.load_estimator([], 0)  # Activate the checked private import namespace.
from l45_corner_shared_20261025.local_pursuit import (  # noqa: E402
    PublicMap, Observation, safe, visible, move, route, uniform)
from l45_corner_shared_20261025.multimodal_estimator import propagate  # noqa: E402

VERSION = 'frozen_D_weighted_open_loop_interface_v2'
METHODS = ('G', 'T', 'T+Geo')
COMPASS = np.array([[0., 0.], [-1., -1.], [-1., 0.], [-1., 1.],
                    [0., -1.], [0., 1.], [1., -1.], [1., 0.], [1., 1.]])


@njit(cache=True)
def sensor_visible(at, q, boxes):
    """Closed-box sensor with a rounding guard in segment parameter space.

    Separate implementation from the orientation/edge audit. The old slab
    comparison can call an exact corner tangent clear after a 1-ULP error.
    No box expansion or change to D's observation conditioning is made here.
    """
    if np.sum((at-q)**2) > 36.+1e-12:
        return False
    for box in boxes:
        low, high = 0., 1.
        possible = True
        for axis in range(2):
            delta = q[axis]-at[axis]
            if abs(delta) < 1e-12:
                if at[axis] < box[axis,0] or at[axis] > box[axis,1]:
                    possible = False
            else:
                u = (box[axis,0]-at[axis])/delta
                v = (box[axis,1]-at[axis])/delta
                low, high = max(low,min(u,v)), min(high,max(u,v))
        if possible and low <= high+1e-12:
            return False
    return True


@njit(cache=True)
def categorical(probabilities, rng):
    value = uniform(rng)*probabilities.sum()
    total = 0.
    for i in range(len(probabilities)):
        total += probabilities[i]
        if value < total:
            return i
    return len(probabilities)-1


@njit(cache=True)
def root_sample(ids, weights, modes, rng):
    sample = categorical(weights, rng)
    return sample, ids[sample], categorical(modes[sample], rng)


@njit(cache=True)
def target_step(index, mode, links, lengths, probabilities, rng):
    # This is precisely propagate(method=1)'s latent Markov factorization.
    if uniform(rng) >= .9:
        mode = min(8, int(9*uniform(rng)))
    if lengths[index] <= 0:
        raise RuntimeError('Simulation has no lawful motion edge')
    k = categorical(probabilities[index, :lengths[index], mode], rng)
    return links[index, k], mode


@njit(cache=True)
def sample_roots(ids, weights, modes, draws, seed):
    rng = np.array([seed], np.int64)
    counts = np.zeros(64, np.int64)
    joint = np.zeros((64, 9), np.int64)
    for _ in range(draws):
        i, _, m = root_sample(ids, weights, modes, rng)
        counts[i] += 1
        joint[i, m] += 1
    return counts, joint


@njit(cache=True)
def sampled_one_step(ids, weights, modes, links, lengths, probabilities, draws, seed):
    rng = np.array([seed], np.int64)
    result = np.zeros((len(links), 9), np.int64)
    for _ in range(draws):
        _, i, m = root_sample(ids, weights, modes, rng)
        j, n = target_step(i, m, links, lengths, probabilities, rng)
        result[j, n] += 1
    return result


def belief_view(estimator):
    """64 weighted positions plus conditional latent intent, never target truth."""
    if estimator.graph is None:
        raise ValueError('Hidden search requires a public-history motion graph')
    weights = estimator.particle_weights.copy()
    if (weights.shape != (64,) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or abs(weights.sum()-1.) > 1e-12):
        raise ValueError('Invalid 64-sample importance weights')
    modes = estimator.weights[estimator.ids].copy()
    normalizers = modes.sum(axis=1)
    if np.any(normalizers <= 0):
        raise ValueError('Exported sample has no conditional motion mode')
    modes /= normalizers[:, None]
    return estimator.ids.copy(), weights, modes


@njit(cache=True)
def geometric_candidates(at, samples, weights, labels, speed, vertices, dist, boxes, motion_boxes):
    """Strong weighted geometry: inspect every retained region and map vertex."""
    count = np.zeros(2, np.int64)
    actions = np.zeros((64, 2))
    actions[:9] = COMPASS*speed
    n = 9
    for r in range(int(labels.max())+1):
        mass = 0.
        center = np.zeros(2)
        for i in range(64):
            if labels[i] == r:
                center += weights[i]*samples[i]
                mass += weights[i]
        if mass <= 0:
            continue
        center /= mass
        winner, distance = -1, 1e100
        for i in range(64):
            if labels[i] == r and np.sum((samples[i]-center)**2) < distance:
                winner, distance = i, np.sum((samples[i]-center)**2)
        action = move(at, samples[winner], speed, vertices, dist, boxes, count)
        if n >= 64:
            raise RuntimeError('Candidate capacity exceeded')
        actions[n] = action
        n += 1
    for v in vertices:
        if n >= 64:
            raise RuntimeError('Candidate capacity exceeded')
        actions[n] = move(at, v, speed, vertices, dist, boxes, count)
        n += 1
    # Remove duplicates and invalid proposals before evaluating their score.
    unique = np.zeros_like(actions)
    used = 0
    for j in range(n):
        if not safe(at, actions[j], speed, motion_boxes, count):
            continue
        duplicate = False
        for k in range(used):
            if np.max(np.abs(actions[j]-unique[k])) < 1e-10:
                duplicate = True
        if not duplicate:
            unique[used] = actions[j]
            used += 1
    best, best_score = 0, -1e100
    for j in range(used):
        nxt = at+unique[j]
        score = 0.
        for i in range(64):
            steps, _ = route(nxt, samples[i], speed, vertices, dist, boxes, count)
            score += weights[i]*(-steps+int(sensor_visible(nxt, samples[i], boxes)))
        if score > best_score:
            best, best_score = j, score
    return unique[:used].copy(), unique[best].copy()


@njit(cache=True)
def one_simulation(at0, speed, horizon, points, links, lengths, probabilities,
                   ids, weights, modes, roots, base, boxes, motion_boxes, rng,
                   children, tree_actions, visits, totals, size):
    sample, target, mode = root_sample(ids, weights, modes, rng)
    at = at0.copy()
    node = 0
    path = np.zeros(horizon+1, np.int64)
    pathn = 1
    expanded, seen_once = False, False
    rewards = np.zeros(horizon)
    trace = np.zeros((horizon, 10))
    count = np.zeros(2, np.int64)
    steps, reason = 0, 1  # reason=1: registered planning leaf; 0: capture
    for depth in range(horizon):
        if not expanded:
            options = roots if node == 0 else COMPASS*speed
            child, chosen, best = -1, -1, -1e100
            for k in range(len(options)):
                if not safe(at, options[k], speed, motion_boxes, count):
                    continue
                c = children[node, k]
                if c < 0:
                    chosen, child = k, size
                    children[node, k] = child
                    tree_actions[child] = options[k]
                    size += 1
                    expanded = True
                    break
                value = totals[c]/max(1, visits[c])+np.sqrt(2*np.log(visits[node]+1)/max(1, visits[c]))
                if value > best:
                    best, chosen, child = value, k, c
            if child < 0:
                raise RuntimeError('No lawful tree action, including stop')
            action = options[chosen].copy()
            node = child
            path[pathn] = node
            pathn += 1
        else:
            action = base.copy()
            if not safe(at, action, speed, motion_boxes, count):
                action[:] = 0.
        if not safe(at, action, speed, motion_boxes, count):
            raise RuntimeError('Illegal simulated own action')
        new_target, mode = target_step(target, mode, links, lengths, probabilities, rng)
        if not safe(points[target], points[new_target]-points[target], 1., boxes, count):
            raise RuntimeError('Illegal simulated motion edge')
        nxt = at+action
        seen = sensor_visible(nxt, points[new_target], boxes)
        capture = np.linalg.norm(nxt-points[new_target]) < 1.
        rewards[depth] = int(seen and not seen_once)+int(capture)
        trace[depth, :2], trace[depth, 2:4] = at, nxt
        trace[depth, 4:6], trace[depth, 6:8] = points[target], points[new_target]
        trace[depth, 8], trace[depth, 9] = int(seen), int(capture)
        seen_once = seen_once or seen
        at, target = nxt, new_target
        steps += 1
        if capture:
            reason = 0
            break
    for j in range(pathn):
        value = 0.
        for t in range(steps-1, max(0, j-1)-1, -1):
            value = rewards[t]+.95*value
        visits[path[j]] += 1
        totals[path[j]] += value
    return size, trace[:steps].copy(), sample, reason


class Controller:
    def __init__(self, boxes, speed, method, seed, budget_ms=50., simulations=1000):
        if method not in METHODS:
            raise ValueError(method)
        started = time.perf_counter()
        self.estimator = loader.load_estimator(boxes, seed)
        stamp = time.perf_counter()
        self.public = PublicMap(boxes, speed)
        self.motion_boxes = self.public.boxes.copy()
        self.motion_boxes[:,:,0] -= 1e-6
        self.motion_boxes[:,:,1] += 1e-6
        self.method, self.seed = method, seed
        self.budget_ms, self.simulations = budget_ms, simulations
        self.rng = np.array([(seed+73120261025) % 4294967296], np.int64)
        self.setup = dict(loader_ms=1000*(stamp-started), static_map_ms=1000*(time.perf_counter()-stamp))
        self.last_stats, self.traces = {}, []

    def act(self, obs, fixed_simulations=None):
        start = time.perf_counter()
        self.estimator.update(obs)
        update_ms = 1000*(time.perf_counter()-start)
        e, p = self.estimator, self.public
        stamp = time.perf_counter()
        if obs.target is not None:
            count = np.zeros(2, np.int64)
            base = move(obs.position, obs.target, p.speed, p.vertices, p.dist, p.boxes, count)
            roots, view = base.reshape(1, 2), None
        else:
            view = belief_view(e)
            roots, base = geometric_candidates(obs.position, e.particles, e.particle_weights,
                e.regions[e.ids], p.speed, p.vertices, p.dist, p.boxes, self.motion_boxes)
            if self.method == 'T':
                roots = COMPASS*p.speed
        candidate_ms = 1000*(time.perf_counter()-stamp)
        stamp = time.perf_counter()
        completed, size = 0, 1
        cap = self.simulations if fixed_simulations is None else fixed_simulations
        self.traces = []
        root_counts = np.zeros(64, np.int64)
        terminations = np.zeros(2, np.int64)
        action = base.copy()
        visits, totals = None, None
        # If public update/candidates have exhausted the budget, return G's
        # checked fallback and record an overrun, never hide estimation cost.
        eligible = (view is not None and self.method != 'G' and
                    (fixed_simulations is not None or 1000*(time.perf_counter()-start) < self.budget_ms-2.))
        prepare_ms = 0.
        if eligible:
            children = np.full((cap+1, 64), -1, np.int32)
            tree_actions = np.zeros((cap+1, 2))
            visits, totals = np.zeros(cap+1, np.int64), np.zeros(cap+1)
            prepare_ms = 1000*(time.perf_counter()-stamp)
            points, links, lengths, probabilities, *_ = e.graph
            for _ in range(cap):
                if fixed_simulations is None and 1000*(time.perf_counter()-start) >= self.budget_ms-2.:
                    break
                size, trace, sample, reason = one_simulation(obs.position, p.speed, min(8, obs.remaining),
                    points, links, lengths, probabilities, *view, roots, base, p.boxes, self.motion_boxes, self.rng,
                    children, tree_actions, visits, totals, size)
                self.traces.append(trace)
                root_counts[sample] += 1
                terminations[reason] += 1
                completed += 1
            if completed:
                candidates = children[0][children[0] >= 0]
                winner = candidates[np.argmax(visits[candidates])]
                action = tree_actions[winner].copy()
        search_ms = 1000*(time.perf_counter()-stamp)
        stamp = time.perf_counter()
        if not safe(obs.position, action, p.speed, p.boxes, np.zeros(2, np.int64)):
            raise RuntimeError('Illegal final action')
        final_check_ms = 1000*(time.perf_counter()-stamp)
        total_ms = 1000*(time.perf_counter()-start)
        self.last_stats = dict(update_ms=update_ms, build_ms=e.last_stats['build_ms'],
            candidate_ms=candidate_ms, tree_prepare_ms=prepare_ms, search_including_prepare_ms=search_ms,
            final_check_ms=final_check_ms, total_ms=total_ms, over_50ms=total_ms > 50.,
            simulations=completed, root_counts=root_counts.tolist(), terminations=terminations.tolist(),
            root_visits=[] if visits is None else visits[children[0][children[0] >= 0]].tolist(),
            root_totals=[] if totals is None else totals[children[0][children[0] >= 0]].tolist(),
            root_candidates=len(roots), source='visible_geometry' if obs.target is not None else
                'weighted_uct_probe' if completed else 'weighted_geometry', planning_depth=min(8, obs.remaining))
        return action
