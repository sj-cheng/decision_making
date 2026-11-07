"""Instrumented original motion/particle baseline; no hidden state inputs."""
import time
import numpy as np
from numba import njit
from solvers.local_pursuit import uniform, legal, clear, visible, recover


@njit(cache=True)
def step_particles(particles, ancestors, observer, boxes, rng, systematic):
    survivors = np.empty_like(particles)
    parents = np.empty(len(particles), np.int64)
    survivor_indices = np.empty(len(particles), np.int64)
    n = 0
    for i, q in enumerate(particles):
        candidate = q+np.array([2*uniform(rng)-1, 2*uniform(rng)-1])
        count = np.zeros(2, np.int64)
        if legal(candidate, boxes, count) and clear(q, candidate, boxes, count) and not visible(observer, candidate, boxes, count):
            survivors[n] = candidate
            parents[n] = ancestors[i]
            survivor_indices[n] = i
            n += 1
    out, lineage = particles.copy(), ancestors.copy()
    indices = np.arange(len(particles))
    if n:
        offset = uniform(rng) if systematic else 0.
        for i in range(len(particles)):
            j = min(n-1, int((i+offset)*n/len(particles))) if systematic else min(n-1, int(uniform(rng)*n))
            out[i], lineage[i] = survivors[j], parents[j]
            indices[i] = survivor_indices[j]
    return out, lineage, n, indices


class ParticleEstimator:
    def __init__(self, boxes, seed, systematic=False):
        self.boxes = np.asarray(boxes, float).reshape(-1, 2, 2)
        self.rng = np.array([(int(seed)+60020261024) % 4294967296], np.int64)
        self.systematic = systematic
        self.last = None
        self.history = []
        self.particles = np.zeros((64, 2))
        self.particle_weights = np.ones(64)/64
        self.ancestors = np.arange(64)
        self.paths = None
        self.last_stats = {}

    def update(self, obs):
        started = time.perf_counter()
        n, recovered = 64, False
        if obs.target is not None:
            self.last = obs.target.copy()
            self.history = []
            self.particles[:] = obs.target
            self.ancestors = np.arange(64)
            self.paths = self.particles[:,None,:].copy()
        else:
            if self.last is None:
                raise ValueError("Positive initialization required")
            self.history.append(obs.position.copy())
            self.particles, self.ancestors, n, indices = step_particles(self.particles, self.ancestors,
                obs.position, self.boxes, self.rng, self.systematic)
            if n and self.paths is not None:
                self.paths = np.concatenate((self.paths[indices],self.particles[:,None,:]),axis=1)
            if not n:
                self.particles, support = recover(self.last, np.asarray(self.history), self.boxes,
                                                  self.rng, np.zeros(2, np.int64))
                if not support:
                    raise RuntimeError("Original particle recovery failed")
                self.ancestors = np.arange(64)
                self.paths = None  # historical recovery did not save ancestors
                recovered = True
        self.last_stats = dict(update_ms=1000*(time.perf_counter()-started), build_ms=0.,
            survivors=int(n), unique_particles=len(np.unique(self.particles, axis=0)),
            unique_ancestors=len(np.unique(self.ancestors)), recovered=recovered)
        return self.particles.copy(), self.particle_weights.copy()

    def distribution(self):
        return self.particles.copy(), self.particle_weights.copy()
