"""Predicted captures on the simulator's 100 Hz grid, using planner-visible boxes."""
import numpy as np
from numba import njit
from solvers.local_pursuit import clear

@njit(cache=True)
def capture_samples(pursuers, evaders, active_p, active_e, boxes):
    """Interpolate aligned 50 Hz paths at half steps; match nearest clear pairs."""
    hits = np.full(len(active_e), -1)
    owners = np.full(len(active_e), -1)
    count = np.zeros(1, np.int64)
    for sample in range(2 * len(pursuers) - 1):
        i = sample // 2
        used = np.zeros(len(active_p), np.bool_)
        while True:
            distance = 1.
            best_p = best_e = -1
            for p in range(len(active_p)):
                if not active_p[p] or used[p]:
                    continue
                for e in range(len(active_e)):
                    if not active_e[e] or hits[e] >= 0:
                        continue
                    pp, ee = pursuers[i,p], evaders[i,e]
                    if sample % 2:
                        pp = (pp + pursuers[i+1,p]) * .5
                        ee = (ee + evaders[i+1,e]) * .5
                    dx, dy = pp[0]-ee[0], pp[1]-ee[1]
                    d = dx*dx + dy*dy
                    if d < distance and clear(pp, ee, boxes, count):
                        distance, best_p, best_e = d, p, e
            if best_e < 0:
                break
            used[best_p] = True
            hits[best_e], owners[best_e] = sample, best_p
    return hits, owners
