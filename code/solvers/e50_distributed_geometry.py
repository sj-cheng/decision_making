"""Single-agent role-free geometry and MINCO candidate evaluation."""
import numpy as np

from solvers.e50_interfaces import quintic_coefficients, sample_quintic


def _segment_safe(proxy, start, endpoint, samples=32):
    if proxy is None:
        return True
    if hasattr(proxy, 'segment_safe'):
        return bool(proxy.segment_safe(start, endpoint, samples=samples, interpolation=True))
    boxes = np.asarray(proxy.obstacle_boxes(), float).reshape(-1, 2, 2) if hasattr(proxy, 'obstacle_boxes') else np.zeros((0, 2, 2))
    points = np.linspace(np.asarray(start, float), np.asarray(endpoint, float), max(2, int(samples)))
    return not bool(np.any(((points[:, None, :] >= boxes[None, :, :, 0]) &
                            (points[:, None, :] <= boxes[None, :, :, 1])).all(axis=-1))) if len(boxes) else True


def _local_tangent_candidates(start, proxy, max_distance, margin=.25):
    """Generate wall-edge and corner bypass endpoints from local LiDAR boxes."""
    if proxy is None or not hasattr(proxy, 'obstacle_boxes'):
        return []
    boxes = np.asarray(proxy.obstacle_boxes(), float).reshape(-1, 2, 2)
    out = []
    for box in boxes:
        center = (box[:, 0] + box[:, 1]) * .5
        delta = center - start
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-9:
            continue
        tangent = np.array([-delta[1], delta[0]]) / norm
        for direction in (tangent, -tangent):
            out.append(start + direction * float(max_distance))
        low, high = box[:, 0] - margin, box[:, 1] + margin
        for corner in (np.array([low[0], low[1]]), np.array([low[0], high[1]]),
                       np.array([high[0], low[1]]), np.array([high[0], high[1]])):
            d = corner - start
            n = float(np.linalg.norm(d))
            if n > 1e-9:
                out.append(start + d * min(1., float(max_distance) / n))
    return out


def _local_free_ray_candidates(start, proxy, max_distance, agent_id=0, cycle=0):
    if proxy is None or not hasattr(proxy, 'free_space_endpoints'):
        return []
    try:
        points = [np.asarray(x, dtype=float) for x in
                  proxy.free_space_endpoints(max_distance=max_distance,
                                             min_clearance=.4, max_candidates=12)]
        return points
    except (TypeError, ValueError):
        return []


def single_agent_endpoints(start, particles, weights, proxy=None, *, duration=2.,
                           max_distance=2., max_candidates=16, samples=32,
                           heading=0., cycle=0, previous_action=None, agent_id=0):
    """Return local belief/intercept/tangent endpoints for one agent.

    The generator has no target assignment or teammate state.  It only uses
    local belief support and the local LiDAR proxy; unknown space is never
    converted into an obstacle-free claim.
    """
    start = np.asarray(start, dtype=float)
    particles = np.asarray(particles, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if start.shape != (2,) or particles.ndim != 2 or particles.shape[1] != 2:
        raise ValueError('invalid start/particles')
    if weights.shape != (len(particles),) or np.any(weights < 0):
        raise ValueError('invalid belief weights')
    candidates = [start.copy()]
    if previous_action is not None:
        previous_action = np.asarray(previous_action, float).reshape(-1)
        if previous_action.shape == (2,) and np.linalg.norm(previous_action) > 1e-6:
            direction = previous_action / np.linalg.norm(previous_action)
            candidates.append(start + direction * float(max_distance))
    ids = np.flatnonzero(weights > 0)
    if len(ids):
        ids = ids[np.argsort(weights[ids])[::-1]]
        ids = ids[:max(4, min(len(ids), 8))]
        for i in ids:
            delta = particles[i] - start
            norm = float(np.linalg.norm(delta))
            if norm > 1e-9:
                candidates.append(start + delta * min(1., max_distance / norm))
                unit = delta / norm
                for sign in (-1., 1.):
                    tangent = np.array([-unit[1], unit[0]]) * sign
                    candidates.append(start + tangent * min(max_distance, norm))
    free_candidates = _local_free_ray_candidates(start, proxy, max_distance,
                                                 agent_id=agent_id, cycle=cycle)
    tangent_candidates = _local_tangent_candidates(start, proxy, max_distance)
    # Prefer directly observed free rays before the larger corner/tangent set;
    # otherwise a visible target can fill the candidate budget with blocked
    # intercept variants and leave no frontier action at all.
    candidates.extend(free_candidates)
    candidates.extend(tangent_candidates)
    # A camera can legitimately have no target report at the beginning of an
    # episode.  Keep local exploration in the candidate set instead of
    # collapsing to a zero action; the directions are body-frame agnostic here
    # and the runtime can rotate them using its current heading if available.
    # LiDAR/occupancy filtering below removes blocked rays.
    if len(ids) == 0:
        radius = float(max_distance)
        # Rotate the exploration fan over replanning cycles so a forward-only
        # camera eventually searches behind the initial heading as the gimbal
        # sweeps its permitted sector.
        phase = float(heading) + (int(cycle) % 8) * (2. * np.pi / 8.)
        for angle in phase + np.linspace(0., 2. * np.pi, 8, endpoint=False):
            candidates.append(start + radius * np.array([np.cos(angle), np.sin(angle)]))
    unique = []
    for endpoint in candidates:
        endpoint = np.asarray(endpoint, dtype=float)
        if not _segment_safe(proxy, start, endpoint, samples=samples):
            continue
        if not any(np.linalg.norm(endpoint - other) < 1e-9 for other in unique):
            unique.append(endpoint)
        if len(unique) >= int(max_candidates):
            break
    return tuple(unique or [start.copy()])


def evaluate_minco_endpoint(start, endpoint, *, duration=2., sample_dt=.02,
                            start_velocity=None, start_acceleration=None,
                            max_speed=np.inf, max_acceleration=np.inf,
                            proxy=None, samples=None):
    """Compile one endpoint with the same quintic semantics used by E50.

    Returns a public score record suitable for a local MCTS value function.
    ``proxy`` is optional because an executor may supply a richer occupancy
    checker; when present it is queried over the full sampled track.
    """
    start = np.asarray(start, dtype=float); endpoint = np.asarray(endpoint, dtype=float)
    if start.shape != (2,) or endpoint.shape != (2,) or duration <= 0 or sample_dt <= 0:
        raise ValueError('invalid endpoint trajectory request')
    v0 = np.zeros(2) if start_velocity is None else np.asarray(start_velocity, dtype=float)
    a0 = np.zeros(2) if start_acceleration is None else np.asarray(start_acceleration, dtype=float)
    if v0.shape != (2,) or a0.shape != (2,):
        raise ValueError('invalid initial derivatives')
    coeff = quintic_coefficients(start[None], endpoint[None], float(duration),
                                 v0[None], np.zeros((1, 2)), a0[None], np.zeros((1, 2)))[0]
    positions, velocities, accelerations = sample_quintic(coeff, float(duration), float(sample_dt))
    speed = float(np.max(np.linalg.norm(velocities, axis=1)))
    accel = float(np.max(np.linalg.norm(accelerations, axis=1)))
    collision = False
    collision = not _segment_safe(proxy, start, endpoint, samples=max(2, len(positions)))
    feasible = bool(speed <= float(max_speed) + 1e-9 and accel <= float(max_acceleration) + 1e-9
                    and not collision)
    score = (0. if feasible else 1.) + .01 * speed + .001 * accel
    return dict(feasible=feasible, collision=bool(collision), score=float(score),
                max_speed=speed, max_acceleration=accel,
                positions=positions, velocities=velocities, accelerations=accelerations,
                coefficients=coeff)
