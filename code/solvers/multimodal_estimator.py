"""Public-observation weighted multimodal grid filter, estimator-only version.

No simulator state, robot identity of an unseen target, or route label is an
input. Exact lawful edges and every negative observation constrain all arms.
The persistent arm retains a weighted distribution over position and motion
mode. Exported samples keep mode mass with importance weights; weak modes are
not promoted to uniform probability. This is a discretized model, not a claim
of calibrated uncertainty or complete continuous reachable support.
"""
import copy
import time

import numpy as np
from numba import njit

from solvers.local_pursuit import clear, legal, uniform, visible
from solvers.history_recovery import GEOMETRY_MARGIN

VERSION = "public_multimodal_grid_v1"
SPACING = .5
SIGMA = .35
PERSISTENCE = .9
MODES = np.array([[0., 0.], [-1., -1.], [-1., 0.], [-1., 1.],
                  [0., -1.], [0., 1.], [1., -1.], [1., 0.], [1., 1.]])


@njit(cache=True)
def make_graph(last, boxes):
    count = np.zeros(2, np.int64)
    axes = np.empty((2, 83))
    sizes = np.zeros(2, np.int64)
    for axis in range(2):
        values = np.empty(83)
        values[:2] = np.array([-10., 10.])
        n = 2
        for k in range(-40, 41):
            q = last[axis]+k*.5
            if -10. <= q <= 10.:
                values[n] = q
                n += 1
        values = np.unique(values[:n])
        axes[axis, :len(values)] = values
        sizes[axis] = len(values)
    nx, ny = sizes
    points = np.empty((nx*ny, 2))
    motion, sensor = boxes.copy(), boxes.copy()
    for b in range(len(boxes)):
        for axis in range(2):
            motion[b, axis, 0] -= GEOMETRY_MARGIN
            motion[b, axis, 1] += GEOMETRY_MARGIN
            sensor[b, axis, 0] += GEOMETRY_MARGIN
            sensor[b, axis, 1] -= GEOMETRY_MARGIN
    free = np.zeros(nx*ny, np.bool_)
    for i in range(nx*ny):
        points[i] = np.array([axes[0, i//ny], axes[1, i%ny]])
        free[i] = legal(points[i], motion, count)
    links = np.full((len(points), 49), -1, np.int32)
    probabilities = np.zeros((len(points), 49, 9))
    lengths = np.zeros(len(points), np.int32)
    for i in range(len(points)):
        if not free[i]:
            continue
        x, y = i//ny, i%ny
        for xx in range(max(0, x-4), min(nx, x+5)):
            for yy in range(max(0, y-4), min(ny, y+5)):
                j = xx*ny+yy
                delta = points[j]-points[i]
                if (free[j] and np.max(np.abs(delta)) <= 1.+1e-10
                        and clear(points[i], points[j], motion, count)):
                    k = lengths[i]
                    if k >= 49:
                        raise ValueError("Unexpected lattice degree")
                    links[i, k] = j
                    lengths[i] += 1
                    for mode in range(9):
                        probabilities[i, k, mode] = np.exp(-.5*np.sum((delta-MODES[mode])**2)/(SIGMA**2))
        for mode in range(9):
            z = probabilities[i, :lengths[i], mode].sum()
            if z > 0:
                probabilities[i, :lengths[i], mode] /= z
    root = np.argmin(np.sum((points-last)**2, axis=1))
    if np.linalg.norm(points[root]-last) > 1e-8 or not free[root]:
        raise ValueError("Last measurement has no legal exact lattice anchor")
    return points, links, lengths, probabilities, sensor, int(root), int(count[0])


@njit(cache=True)
def propagate(weights, reachable, observer, points, links, lengths, probabilities,
              sensor_boxes, method):
    n = len(points)
    result = np.zeros_like(weights)
    mask = np.zeros(n, np.bool_)
    parents = np.full(n, -1, np.int32)
    count = np.zeros(2, np.int64)
    allowed = np.zeros(n, np.bool_)
    for j in range(n):
        # Definitely invisible, including the same numerical guard as recovery.
        allowed[j] = (np.sum((observer-points[j])**2) > (6.+GEOMETRY_MARGIN)**2
                      or not clear(observer, points[j], sensor_boxes, count))
    for i in range(n):
        if not reachable[i]:
            continue
        total = weights[i].sum()
        for k in range(lengths[i]):
            j = links[i, k]
            if not allowed[j]:
                continue
            mask[j] = True
            if parents[j] < 0:
                parents[j] = i
            if method == 0:  # deterministic iid/uniform-motion reference
                result[j, 0] += total/lengths[i]
            elif method == 1:  # persistent weighted motion mixture
                for mode in range(9):
                    prior = PERSISTENCE*weights[i, mode]+(1.-PERSISTENCE)*total/9.
                    result[j, mode] += prior*probabilities[i, k, mode]
            else:  # feasible support only, explicitly *not* probabilistic
                result[j, 0] = 1.
    z = result.sum()
    if not np.isfinite(z) or z <= 0:
        raise RuntimeError("Empty weighted public-history support")
    result /= z
    return result, mask, parents, int(count[0])


@njit(cache=True)
def export_modes(weights, rng, number):
    """Mode-stratified samples with preserved posterior mode probabilities."""
    mass = weights.sum(axis=0)
    counts = np.zeros(9, np.int64)
    for m in range(9):
        if mass[m] > 1e-12:
            counts[m] = 1
    for _ in range(number-counts.sum()):
        best = np.argmax(number*mass-counts)
        counts[best] += 1
    ids = np.empty(number, np.int64)
    output_weights = np.empty(number)
    labels = np.empty(number, np.int64)
    cursor = 0
    for m in range(9):
        if counts[m] == 0:
            continue
        cumulative = np.cumsum(weights[:, m])
        offset = uniform(rng)
        for k in range(counts[m]):
            value = (k+offset)*mass[m]/counts[m]
            i = np.searchsorted(cumulative, value, side="right")
            ids[cursor] = min(len(weights)-1, i)
            output_weights[cursor] = mass[m]/counts[m]
            labels[cursor] = m
            cursor += 1
    return ids, output_weights, labels


class Estimator:
    def __init__(self, boxes, method="persistent", seed=0, particles=64):
        if method not in ("diffuse", "persistent", "feasible"):
            raise ValueError(method)
        if particles != 64:
            raise ValueError("This registered version fixes the particle budget at 64")
        self.boxes = np.asarray(boxes, float).reshape(-1, 2, 2)
        self.method, self.number = method, particles
        self.rng = np.array([int(seed) % 4294967296], np.int64)
        self.step, self.last, self.graph = -1, None, None
        self.parents, self.observers = [], []
        self.weights, self.reachable = None, None
        self.particles, self.particle_weights = None, None
        self.last_stats = {}

    def update(self, obs):
        started = time.perf_counter()
        if self.step >= 0 and obs.step != self.step+1:
            raise ValueError("Nonconsecutive observation")
        if obs.remaining != 80-obs.step or not np.all(np.isfinite(obs.position)):
            raise ValueError("Bad public clock/position")
        if not legal(np.asarray(obs.position), self.boxes, np.zeros(2, np.int64)):
            raise ValueError("Illegal observer position")
        build_ms, geometry = 0., 0
        if obs.target is not None:
            count = np.zeros(2, np.int64)
            target = np.asarray(obs.target, float)
            if not legal(target, self.boxes, count) or not visible(obs.position, target, self.boxes, count):
                raise ValueError("Illegal positive measurement")
            self.last = target.copy()
            self.graph = None
            self.parents, self.observers = [], []
            self.particles = np.tile(target, (64, 1))
            self.particle_weights = np.ones(64)/64
            self.weights, self.reachable = None, None
        else:
            if self.last is None:
                raise ValueError("Positive initialization required")
            if self.graph is None:
                stamp = time.perf_counter()
                self.graph = make_graph(self.last, self.boxes)
                build_ms = 1000*(time.perf_counter()-stamp)
                points, links, lengths, probabilities, sensor, root, queries = self.graph
                geometry += queries
                self.weights = np.zeros((len(points), 9))
                self.weights[root] = np.ones(9)/9 if self.method == "persistent" else np.array([1.,0.,0.,0.,0.,0.,0.,0.,0.])
                self.reachable = np.zeros(len(points), bool)
                self.reachable[root] = True
            points, links, lengths, probabilities, sensor, root, queries = self.graph
            self.weights, self.reachable, parent, queries = propagate(
                self.weights, self.reachable, np.asarray(obs.position), points, links, lengths,
                probabilities, sensor, ("diffuse", "persistent", "feasible").index(self.method))
            geometry += queries
            self.parents.append(parent)
            self.observers.append(np.asarray(obs.position).copy())
            ids, self.particle_weights, self.labels = export_modes(self.weights, self.rng, 64)
            self.ids = ids
            self.particles = points[ids].copy()
        self.step = obs.step
        self.last_stats = dict(update_ms=1000*(time.perf_counter()-started), build_ms=build_ms,
            geometry_queries=geometry, support_points=int(self.reachable.sum()) if self.graph else 1,
            mode_mass=self.weights.sum(axis=0).tolist() if self.graph else [1.]+[0.]*8)
        return self.particles.copy(), self.particle_weights.copy()

    def distribution(self):
        if self.graph is None:
            return self.last.reshape(1, 2).copy(), np.ones(1)
        marginal = self.weights.sum(axis=1)
        ids = np.flatnonzero(marginal > 0)
        return self.graph[0][ids].copy(), marginal[ids].copy()

    def witnesses(self):
        if self.graph is None:
            return self.particles[:, None, :].copy()
        paths = np.empty((64, len(self.parents)+1, 2))
        paths[:, 0] = self.last
        for k, index in enumerate(self.ids):
            for t in range(len(self.parents)-1, -1, -1):
                paths[k, t+1] = self.graph[0][index]
                index = self.parents[t][index]
        return paths

    def snapshot(self):
        return copy.deepcopy(self.__dict__)

    def restore(self, snapshot):
        self.__dict__.update(copy.deepcopy(snapshot))
