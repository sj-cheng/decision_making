"""Process-local speed parameterization; never edits the frozen E46/E47 stack.

Known evader bounds enter the physical model, D grid/means/noise, public-history
recovery and simulated-motion checks. Spatial resolutions scale with the bound
so a full-speed displacement remains represented by two coarse grid intervals.
At bound 1 all original numerical kernels are retained.
"""
import importlib
import numpy as np
from numba import njit

from solvers.frozen_d_search_interface import PublicMap, safe
from solvers.local_pursuit import clear, legal, edge
from solvers.history_recovery import GEOMETRY_MARGIN


@njit(cache=True)
def speed_lattice(last, boxes, spacing, speed):
    axes = []
    for axis in range(2):
        values = [-10., 10.]
        extent = int(np.ceil(20 / spacing))
        for k in range(-extent, extent + 1):
            q = last[axis] + k * spacing
            if -10. <= q <= 10.:
                values.append(q)
        axes.append(np.unique(np.asarray(values)))
    nx, ny = len(axes[0]), len(axes[1])
    points = np.empty((nx * ny, 2))
    motion, sensor = boxes.copy(), boxes.copy()
    motion[:, :, 0] -= GEOMETRY_MARGIN
    motion[:, :, 1] += GEOMETRY_MARGIN
    sensor[:, :, 0] += GEOMETRY_MARGIN
    sensor[:, :, 1] -= GEOMETRY_MARGIN
    count = np.zeros(2, np.int64)
    free = np.zeros(len(points), np.bool_)
    for i in range(len(points)):
        points[i] = np.array([axes[0][i // ny], axes[1][i % ny]])
        free[i] = legal(points[i], motion, count)
    radius = int(np.ceil(speed / spacing)) + 2
    capacity = (2 * int(np.ceil(speed / spacing)) + 3) ** 2
    links = np.full((len(points), capacity), -1, np.int32)
    lengths = np.zeros(len(points), np.int32)
    for i in range(len(points)):
        if not free[i]:
            continue
        x, y = i // ny, i % ny
        for xx in range(max(0, x-radius), min(nx, x+radius+1)):
            for yy in range(max(0, y-radius), min(ny, y+radius+1)):
                j = xx * ny + yy
                if (free[j] and np.max(np.abs(points[j]-points[i])) <= speed+1e-10
                        and clear(points[i], points[j], motion, count)):
                    if lengths[i] >= capacity:
                        raise RuntimeError('Speed lattice degree bound exceeded')
                    links[i, lengths[i]] = j
                    lengths[i] += 1
    root = int(np.argmin(np.sum((points-last)**2, axis=1)))
    if not free[root] or np.linalg.norm(points[root]-last) > 1e-8:
        raise RuntimeError('No lawful public anchor')
    return points, links, lengths, sensor, root, int(count[0])


@njit(cache=True)
def speed_probabilities(points, links, lengths, means, speed):
    result = np.zeros((len(points), links.shape[1], 9))
    sigma = .35 * speed
    for i in range(len(points)):
        for k in range(lengths[i]):
            delta = points[links[i, k]]-points[i]
            for m in range(9):
                result[i, k, m] = np.exp(-.5*np.sum((delta-means[i, m])**2)/sigma**2)
        for m in range(9):
            z = result[i, :lengths[i], m].sum()
            if z > 0:
                result[i, :lengths[i], m] /= z
    return result


@njit(cache=True)
def speed_map_probabilities(points, links, lengths, last, boxes, vertices, dist, speed):
    modes = np.array([[0., 0.], [-1., -1.], [-1., 0.], [-1., 1.],
                      [0., -1.], [0., 1.], [1., -1.], [1., 0.], [1., 1.]])
    count = np.zeros(2, np.int64)
    goals = np.empty((9, 2)); goals[0] = last
    tails = np.full((9, len(vertices)), 1000000, np.int64)
    for m in range(1, 9):
        goals[m] = np.clip(last+20.*modes[m], -9.95, 9.95)
        if not legal(goals[m], boxes, count):
            raise ValueError('Public fixed-intent goal is illegal')
        for j in range(len(vertices)):
            if clear(vertices[j], goals[m], boxes, count):
                tail = edge(vertices[j], goals[m], speed)
                for k in range(len(vertices)):
                    tails[m, k] = min(tails[m, k], dist[k, j]+tail)
    means = np.zeros((len(points), 9, 2))
    for i in range(len(points)):
        if lengths[i] == 0:
            continue
        head = np.full(len(vertices), 1000000, np.int64)
        for j in range(len(vertices)):
            if clear(points[i], vertices[j], boxes, count):
                head[j] = edge(points[i], vertices[j], speed)
        for m in range(1, 9):
            best, first = 1000000, -1
            if clear(points[i], goals[m], boxes, count):
                best = edge(points[i], goals[m], speed)
            for j in range(len(vertices)):
                candidate = head[j]+tails[m, j]
                if candidate < best:
                    best, first = candidate, j
            if best >= 1000000:
                continue
            waypoint = vertices[first] if first >= 0 else goals[m]
            delta = waypoint-points[i]
            action = delta/max(1., np.max(np.abs(delta))/speed)
            if safe(points[i], action, speed, boxes, count):
                means[i, m] = action
    return speed_probabilities(points, links, lengths, means, speed), means, int(count[0])


_ORIGINAL = None


def install(speed, out):
    """Install/revert hooks only in this process; output and weights stay isolated."""
    global _ORIGINAL
    import run_l5_followup as follow
    import run_e45b_role_free as base
    import solvers.e45b_joint_nte as joint
    import solvers.e45f_support_refinement as refine
    from run_frozen_d_closed_loop import legal_action
    from corner_estimator_scenarios import original_model
    from e45c_capture_audit import capture_errors
    from e45b_sensor_audit import audit_visibility
    from estimator_audit_geometry import point_legal, blocked
    frozen = importlib.import_module('l45_corner_shared_20261025.corner_estimator')
    if _ORIGINAL is None:
        _ORIGINAL = [(frozen, n, getattr(frozen, n)) for n in ('make_graph', 'map_probabilities', 'PublicMap')]
        _ORIGINAL += [(joint, n, getattr(joint, n)) for n in ('make_graph', 'map_probabilities', 'PublicMap', 'safe')]
        _ORIGINAL += [(refine, n, getattr(refine, n)) for n in ('lattice', 'map_probabilities', 'motion_probabilities', 'PublicMap')]
    for module, name, value in _ORIGINAL:
        setattr(module, name, value)
    follow.configure(out)
    speed = float(speed)

    if speed != 1.:
        def lattice(last, boxes, spacing):
            return speed_lattice(last, boxes, spacing*speed, speed)[:5]

        def make_graph(last, boxes):
            points, links, lengths, sensor, root, queries = speed_lattice(last, boxes, .5*speed, speed)
            # D and tree callers replace this placeholder with map-guided probabilities.
            return points, links, lengths, np.zeros((len(points), links.shape[1], 9)), sensor, root, queries

        def probabilities(points, links, lengths, last, boxes, vertices, dist):
            return speed_map_probabilities(points, links, lengths, last, boxes, vertices, dist, speed)

        def public_map(boxes, limit):
            return PublicMap(boxes, speed if float(limit) == 1. else limit)

        def safe_target(at, action, limit, boxes, count):
            return safe(at, action, speed if float(limit) == 1. else limit, boxes, count)

        frozen.make_graph = joint.make_graph = make_graph
        frozen.map_probabilities = joint.map_probabilities = refine.map_probabilities = probabilities
        frozen.PublicMap = joint.PublicMap = refine.PublicMap = public_map
        joint.safe = safe_target
        refine.lattice = lattice
        refine.motion_probabilities = lambda p, l, n, m: speed_probabilities(p, l, n, m, speed)

    def environment(boxes, pursuer_speed):
        if float(pursuer_speed) == 1.:
            raise ValueError('This adapter reserves legacy limit=1 for evaders; pursuer limit must differ')
        p = original_model(boxes, pursuer_speed)
        p.current_evader_speed_lim = speed
        for j in range(2):
            p.action_lims[p.action_idxs[j]] = [[-speed, speed], [-speed, speed]]
            p.control_lims[p.action_idxs[j]] = p.action_lims[p.action_idxs[j]]/p.dt
        p.clear_closed_loop_cache()
        return p

    def audit(trace, boxes, pursuer_speed):
        errors = np.zeros(4, np.int64)
        for row in trace:
            for i in range(2):
                a, b = row[2*i:2*i+2], row[4+2*i:6+2*i]
                if (not point_legal(a, boxes) or not point_legal(b, boxes)
                        or np.max(np.abs(b-a)) > pursuer_speed+2e-7 or blocked(a, b, boxes)):
                    errors[0] += 1
                if not row[16+i] and np.max(np.abs(b-a)) > 1e-10:
                    errors[0] += 1
            for j in range(2):
                if row[18+j]:
                    a, b = row[8+2*j:10+2*j], row[12+2*j:14+2*j]
                    if (not point_legal(a, boxes) or not point_legal(b, boxes)
                            or np.max(np.abs(b-a)) > speed+2e-7 or blocked(a, b, boxes)):
                        errors[1] += 1
        errors[2] = capture_errors(trace)
        errors[3] = audit_visibility(trace, boxes)
        return errors

    base.original_model = environment
    base.legal_action = lambda at, a, v, boxes: legal_action(at, a, speed if float(v) == 1. else v, boxes)
    base.audit_simulated = audit
    old_factory = follow.follow.make_controller

    def factory(case, task):
        if task['arm'] == 'AG':
            from solvers.e45_team_adapter import TeamController
            return TeamController(case['map'], case['speed'], task['seed'], 'AG', budget_ms=task['budget_ms'])
        return old_factory(case, task)

    base.make_controller = factory
    return dict(revision='speed-bound-scaled-D-v1', evader_speed=speed,
                grid_spacing=.5*speed, sigma=.35*speed, pursuer_factory=factory)
