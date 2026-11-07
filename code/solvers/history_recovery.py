"""Bounded public-history recovery with explicit legal path witnesses.

Keep the historical stochastic filter/recovery as the first attempt. If its
finite samples die out, enumerate a finite point lattice through *all* empty
observations since the last measurement. Every returned endpoint has a legal
ancestral path. This is a discrete feasible-support fallback, not a calibrated
posterior, a continuous-support guarantee, or a hidden-world repair.
"""
import numpy as np
from numba import njit

from solvers.local_pursuit import legal, clear, visible, uniform, recover as stochastic_recover


RESOLUTIONS = (.5, .25)
GEOMETRY_MARGIN = 1e-6


@njit(cache=True)
def lattice_paths(last, observers, boxes, spacing, rng, count, number=64):
    """Return particles and witnesses, or explicit zero support.

    The lattice is anchored at the last measurement and includes exact arena
    boundaries. Parent pointers certify every empty-observation transition.
    No point/segment is accepted on an outer-cell approximation alone.
    """
    h = len(observers)
    result = np.zeros((number, 2))
    paths = np.zeros((number, h+1, 2))
    if (h > 80 or not np.all(np.isfinite(last)) or not np.all(np.isfinite(observers))
            or not legal(last, boxes, count)):
        return result, paths, 0
    if h == 0:
        result[:] = last
        paths[:, 0] = last
        return result, paths, number
    # Certify witnesses away from floating-point corner tangencies. Inflate
    # obstacles for motion and deflate them for negative sensor evidence, so
    # neither a grazing path nor an ambiguously occluded sample is certified.
    motion_boxes, sensor_boxes = boxes.copy(), boxes.copy()
    for b in range(len(boxes)):
        for axis in range(2):
            motion_boxes[b, axis, 0] -= GEOMETRY_MARGIN
            motion_boxes[b, axis, 1] += GEOMETRY_MARGIN
            sensor_boxes[b, axis, 0] += GEOMETRY_MARGIN
            sensor_boxes[b, axis, 1] -= GEOMETRY_MARGIN
    # A fixed world-center grid can miss a thin feasible sliver at a boundary.
    # Preserve axis-limit endpoints reachable from the actual measurement and
    # both exact arena edges. The construction still uses public data only.
    extent = int(np.ceil(20./spacing))
    xs = np.empty(2*extent+3)
    ys = np.empty_like(xs)
    xs[0], xs[1], ys[0], ys[1] = -10., 10., -10., 10.
    xn, yn = 2, 2
    for k in range(-extent, extent+1):
        x, y = last[0]+k*spacing, last[1]+k*spacing
        if -10. <= x <= 10.:
            xs[xn] = x
            xn += 1
        if -10. <= y <= 10.:
            ys[yn] = y
            yn += 1
    xs, ys = np.unique(xs[:xn]), np.unique(ys[:yn])
    nx, ny = len(xs), len(ys)
    cells = nx*ny
    radius = int(round(1./spacing))+2
    points = np.empty((cells, 2))
    free = np.zeros(cells, np.bool_)
    for i in range(cells):
        points[i] = np.array([xs[i//ny], ys[i%ny]])
        free[i] = legal(points[i], motion_boxes, count)
    parents = np.full((h, cells), -1, np.int32)
    prior = np.zeros(cells, np.bool_)
    for t in range(h):
        current = np.zeros(cells, np.bool_)
        for i in range(cells):
            # Reject samples that might be visible within numerical margin.
            might_be_visible = (np.sum((observers[t]-points[i])**2) <= (6.+GEOMETRY_MARGIN)**2
                                and clear(observers[t], points[i], sensor_boxes, count))
            if not free[i] or might_be_visible:
                continue
            if t == 0:
                if np.max(np.abs(points[i]-last)) <= 1.+1e-10 and clear(last, points[i], motion_boxes, count):
                    current[i] = True
                    parents[t, i] = -2
                continue
            x, y = i//ny, i%ny
            found = False
            for dx in range(-radius, radius+1):
                xx = x+dx
                if xx < 0 or xx >= nx:
                    continue
                for dy in range(-radius, radius+1):
                    yy = y+dy
                    if yy < 0 or yy >= ny:
                        continue
                    j = xx*ny+yy
                    if (prior[j] and np.max(np.abs(points[j]-points[i])) <= 1.+1e-10
                            and clear(points[j], points[i], motion_boxes, count)):
                        current[i] = True
                        parents[t, i] = j
                        found = True
                        break
                if found:
                    break
        prior = current
        if not prior.any():
            return result, paths, 0
    ids = np.flatnonzero(prior)
    # Stratify across endpoint order instead of duplicating one survivor. This
    # is uniform feasible-endpoint sampling, not probability/path weighting.
    offset = uniform(rng)
    for k in range(number):
        i = ids[min(len(ids)-1, int((k+offset)*len(ids)/number))]
        result[k] = points[i]
        paths[k, 0] = last
        for t in range(h-1, -1, -1):
            paths[k, t+1] = points[i]
            if t > 0:
                i = parents[t, i]
    return result, paths, number


@njit(cache=True)
def recover(last, observers, boxes, rng, count):
    # Extended counters are local to v2: stochastic calls, lattice attempts,
    # exhausted lattice recoveries, certified endpoint samples.
    count[2] += 1
    particles, support = stochastic_recover(last, observers, boxes, rng, count)
    if support:
        return particles, support
    for spacing in RESOLUTIONS:
        count[3] += 1
        particles, witnesses, support = lattice_paths(last, observers, boxes, spacing, rng, count)
        if support:
            count[5] += support
            return particles, support
    count[4] += 1
    return particles, 0
