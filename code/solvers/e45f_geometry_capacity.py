"""E45-F geometry candidate buffer compatibility, enumeration unchanged.

64 samples contribute at most64 retained regions. Plus9 compass actions and
all public vertices gives an exact safe capacity bound. No pruning or scoring change.
"""
import numpy as np
from numba import njit
from solvers.frozen_d_search_interface import COMPASS,move,safe,route,sensor_visible


@njit(cache=True)
def geometric_candidates_capacity(at, samples, weights, labels, speed, vertices, dist, boxes, motion_boxes):
    """Strong weighted geometry: inspect every retained region and map vertex."""
    count = np.zeros(2, np.int64)
    capacity = 9 + len(samples) + len(vertices)
    actions = np.zeros((capacity, 2))
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
        if n >= capacity:
            raise RuntimeError('Candidate capacity exceeded')
        actions[n] = action
        n += 1
    for v in vertices:
        if n >= capacity:
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
