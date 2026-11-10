"""Voronoi area-minimisation pursuit (Zhou et al. 2016) on an obstacle grid.

Both teams have a 1.0 per-axis limit, so travel time is Chebyshev distance and
the partition is a unit-cost 8-connected BFS. Every pursuer heads for the
centroid of the boundary it shares with the focused evader's cell; that shrinks
the evader cell fastest. A pursuer that shares no boundary, or is already close,
runs straight at the evader. The focused evader is the visible one with the
smallest cell. With no visible evader the controller falls back to the AG
belief-search actions, unchanged.
"""
import time
import numpy as np
from numba import njit
from solvers.frozen_d_search_interface import move, safe
from solvers.e45_team_adapter import TeamController

RESOLUTION = .25
CELLS = int(round(20./RESOLUTION))
CLOSE = 2.


def free_grid(boxes, margin=.05):
    centers = -10.+RESOLUTION*(np.arange(CELLS)+.5)
    x, y = np.meshgrid(centers, centers, indexing='ij')
    free = np.ones((CELLS, CELLS), np.bool_)
    for box in np.asarray(boxes, float).reshape(-1, 2, 2):
        free &= ~((x >= box[0, 0]-margin) & (x <= box[0, 1]+margin) & (y >= box[1, 0]-margin) & (y <= box[1, 1]+margin))
    return free, centers


@njit(cache=True)
def partition(free, sources):
    n = free.shape[0]
    label = np.full((n, n), -1, np.int64)
    queue = np.zeros((n*n, 2), np.int64)
    head, tail = 0, 0
    for k in range(len(sources)):
        i, j = sources[k]
        if label[i, j] < 0:
            label[i, j] = k; queue[tail] = (i, j); tail += 1
    while head < tail:
        i, j = queue[head]; head += 1
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0: continue
                a, b = i+di, j+dj
                if a < 0 or b < 0 or a >= n or b >= n or not free[a, b] or label[a, b] >= 0: continue
                if di != 0 and dj != 0 and not (free[i+di, j] and free[i, j+dj]): continue
                label[a, b] = label[i, j]; queue[tail] = (a, b); tail += 1
    return label


@njit(cache=True)
def shared_boundary(label, site, other):
    n = label.shape[0]
    count = 0; si = 0.; sj = 0.
    for i in range(n):
        for j in range(n):
            if label[i, j] != site: continue
            touch = False
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i+di, j+dj
                    if 0 <= a < n and 0 <= b < n and label[a, b] == other: touch = True
            if touch:
                count += 1; si += i; sj += j
    if count == 0: return -1., -1., 0
    return si/count, sj/count, count


@njit(cache=True)
def nearest_cell(label, site, ci, cj):
    n = label.shape[0]
    best = 1e18; bi = -1; bj = -1
    for i in range(n):
        for j in range(n):
            if label[i, j] != site: continue
            d = (i-ci)**2+(j-cj)**2
            if d < best: best = d; bi = i; bj = j
    return bi, bj


class VoronoiController(TeamController):
    def __init__(self, boxes, speed, seed, budget_ms=200.):
        super().__init__(boxes, speed, seed, 'AG', budget_ms=budget_ms)
        self.free, self.centers = free_grid(boxes)
        index = np.argwhere(self.free)
        self.free_index = index
        self.free_points = self.centers[index]

    def cell(self, point):
        k = int(np.argmin(np.sum((self.free_points-np.asarray(point))**2, axis=1)))
        return self.free_index[k]

    def act(self, obs, fixed_simulations=None):
        from e49_controllers import optimal_assignment
        started = time.perf_counter(); self.observe(obs)
        cost, options = self.geometry(obs)
        ap = np.asarray(obs['active_p'], bool); ae = np.asarray(obs['active_e'], bool)
        positions = np.asarray(obs['positions'], float)
        visible = [j for j in np.flatnonzero(ae) if self.belief.known[j] is not None]
        if not visible:
            assignment = optimal_assignment(cost, ap, ae, False); action = np.zeros((2, 2))
            for i, j in assignment.items(): action[i] = options[i, j]
            self.last_stats = dict(total_ms=1000*(time.perf_counter()-started), simulations=0, mode='search',
                                   assignment={str(i): int(j) for i, j in assignment.items()}, team_belief=self.belief.last_stats)
            return action
        pursuers = list(np.flatnonzero(ap))
        sites = [self.cell(positions[i]) for i in pursuers]+[self.cell(self.belief.known[j]) for j in visible]
        label = partition(self.free, np.asarray(sites, np.int64))
        areas = [int(np.sum(label == len(pursuers)+k)) for k in range(len(visible))]
        focus = int(np.argmin(areas)); target = np.asarray(self.belief.known[visible[focus]], float)
        evader_site = len(pursuers)+focus
        p = self.public; count = np.zeros(2, np.int64); action = np.zeros((2, 2)); modes = {}
        for k, i in enumerate(pursuers):
            ci, cj, size = shared_boundary(label, evader_site, k)
            goal = target; mode = 'chase'
            if size > 0 and np.max(np.abs(target-positions[i])) > CLOSE:
                bi, bj = nearest_cell(label, evader_site, ci, cj)
                goal = np.array([self.centers[bi], self.centers[bj]]); mode = 'boundary'
            step = move(positions[i], goal, p.speed, p.vertices, p.dist, p.boxes, count)
            if not safe(positions[i], step, p.speed, p.boxes, count):
                step = move(positions[i], target, p.speed, p.vertices, p.dist, p.boxes, count); mode = 'chase'
            action[i] = step; modes[str(i)] = mode
        self.last_stats = dict(total_ms=1000*(time.perf_counter()-started), simulations=0, mode='voronoi',
                               focus=int(visible[focus]), evader_cell_cells=areas, pursuer_modes=modes,
                               team_belief=self.belief.last_stats)
        return action
