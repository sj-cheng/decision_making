"""Geometry escape at the evader's own speed limit; diagnostic opponent only.

Same finite lookahead as e49_geometric_evader, but steps reach the 1.5 per-axis
limit instead of 1.0. Each pursuer is modelled as a map-aware direct chaser.
"""
from types import SimpleNamespace
import time
import numpy as np
from numba import njit
from solvers.frozen_d_search_interface import PublicMap, move, route, safe, sensor_visible


def actions(limit):
    steps = [[x, y] for scale in (limit, .5*limit) for x in (-scale, 0., scale) for y in (-scale, 0., scale)
             if x != 0 or y != 0]
    return np.array(steps+[[0., 0.]])


@njit(cache=True)
def escape_one(at, pursuers, active, boxes, vertices, pdist, edist, pursuer_speed, menu, limit, depth=4, width=8):
    states = np.zeros((width, 6)); states[0, :2] = at; states[0, 2:] = pursuers.reshape(4)
    first = np.zeros((width, 2)); scores = np.zeros(width); alive = np.ones(width, np.bool_); n = 1
    count = np.zeros(2, np.int64)
    for level in range(depth):
        ns = np.zeros((width*len(menu), 6)); nf = np.zeros((width*len(menu), 2))
        value = np.full(width*len(menu), -1e12); live = np.zeros(width*len(menu), np.bool_); size = 0
        for k in range(n):
            if not alive[k]:
                ns[size] = states[k]; nf[size] = first[k]; value[size] = scores[k]-100.; size += 1; continue
            q = states[k, :2]; pp = states[k, 2:].reshape(2, 2); nextp = pp.copy()
            for i in range(2):
                if active[i]: nextp[i] += move(pp[i], q, pursuer_speed, vertices, pdist, boxes, count)
            for action in menu:
                if not safe(q, action, limit, boxes, count): continue
                nq = q+action; minimum = 100.; geodesic = 100.; hidden = True
                for i in range(2):
                    if active[i]:
                        minimum = min(minimum, np.sqrt(np.sum((nq-nextp[i])**2)))
                        geodesic = min(geodesic, float(route(nextp[i], nq, 1., vertices, edist, boxes, count)[0]))
                        if sensor_visible(nextp[i], nq, boxes): hidden = False
                survives = minimum >= 1.
                margin = min(minimum, 10.)+.25*min(geodesic, 20.)+(.4 if hidden else 0.)
                boundary = max(0., min(10.-abs(nq[0]), 10.-abs(nq[1])))
                score = scores[k]+(.9**level)*(margin+.05*min(boundary, 2.))
                if not survives: score -= 1000.+100.*(depth-level)
                ns[size, :2] = nq; ns[size, 2:] = nextp.reshape(4)
                nf[size] = action if level == 0 else first[k]; value[size] = score; live[size] = survives; size += 1
        order = np.argsort(-value[:size]); n = min(width, size)
        for k in range(n):
            idx = order[k]; states[k] = ns[idx]; first[k] = nf[idx]; scores[k] = value[idx]; alive[k] = live[idx]
    return first[0], scores[0], alive[0]


class ScatterEscapeEvader:
    """Full-state opponent with the RootPUCT call surface used by run_episode."""
    def __init__(self, limit=1.5):
        self.limit = float(limit); self.menu = actions(self.limit); self.public = None; self.records = []

    def get_rng_state(self, problem):
        return dict(solver='deterministic', problem='deterministic')

    def search(self, problem, state, turn=0):
        if self.public is None: self.public = PublicMap(problem.obstacles, problem.current_pursuer_speed_lim)
        start = time.perf_counter(); p = self.public; out = np.zeros((2, 2)); ap = state[problem.active_idxs[2:], 0] > .5
        for j in range(2):
            if state[problem.active_idxs[j], 0] > .5:
                out[j], _, _ = escape_one(state[2*j:2*j+2, 0], state[4:8, 0].reshape(2, 2), ap,
                                          p.boxes, p.vertices, p.dist, p.edist, p.speed, self.menu, self.limit)
        self.records.append(1000*(time.perf_counter()-start))
        return SimpleNamespace(best_action=np.r_[out.ravel(), np.zeros(4)], success=True)
