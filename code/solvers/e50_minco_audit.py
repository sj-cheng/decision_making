"""MINCO execution audit for existing E46-E49 controller actions.

The controller still returns the historical one-second relative displacement.
This adapter replays that command through Example8's existing closed-loop
MINCO implementation and returns sampled trajectories for fair comparison.
It does not change any historical controller or benchmark protocol.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MincoAudit:
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    times: np.ndarray
    collision: bool
    max_speed: float
    max_acceleration: float
    raw_targets: np.ndarray = None
    feasible_targets: np.ndarray = None


def _trajectory_hits_boxes(positions, boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 2, 2)
    if not len(boxes):
        return False
    inside = ((positions[:, :, None, :] >= boxes[None, None, :, :, 0]) &
              (positions[:, :, None, :] <= boxes[None, None, :, :, 1]))
    return bool(np.any(inside.all(axis=-1)))


def obstacle_aware_minco_action(problem, state, action, obstacles, *, dt=2.,
                                sample_dt=.02, robots=(2, 3)):
    """Choose a MINCO-feasible local detour before execution.

    This is part of the trajectory planner: candidates are converted to the
    same endpoint-constrained MINCO tracks and scored by displacement change.
    It does not inspect truth geometry or overwrite a command after planning.
    """
    state = np.asarray(state, float).reshape(-1, 1)
    from solvers.e50_compat_smoke import replay_endpoint_semantic
    base = np.asarray(action, float).reshape(problem.action_dim, 1).copy()
    boxes = np.asarray(obstacles, float).reshape(-1, 2, 2)
    if not len(boxes):
        return base, dict(applied=False, reason='no_local_obstacles')
    # A stationary MINCO track is a valid planner candidate when every
    # forward endpoint would enter a measured local patch.
    candidates = [base, np.zeros_like(base)]
    for robot in robots:
        sl = problem.action_idxs[robot]
        vec = base[sl, 0].copy()
        for angle in (np.deg2rad(30.), -np.deg2rad(30.)):
            rotated = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]]) @ vec
            trial = base.copy(); trial[sl, 0] = rotated; candidates.append(trial)
    best = None
    for candidate in candidates:
        audit, feasible = replay_endpoint_semantic(problem, state, candidate,
                                                    dt=dt, sample_dt=sample_dt,
                                                    robots=robots)
        hit = _trajectory_hits_boxes(audit.positions, boxes)
        score = (float(hit), 0. if feasible else 1.,
                 float(np.linalg.norm(candidate - base)),
                 float(audit.max_acceleration))
        if best is None or score < best[0]:
            best = (score, candidate, audit, feasible, hit)
    _, chosen, audit, feasible, hit = best
    return chosen, dict(applied=bool(np.linalg.norm(chosen-base) > 1e-9),
                        collision=bool(hit), feasible=bool(feasible),
                        max_acceleration=float(audit.max_acceleration))


def _sample_segments(problem, segments, sample_dt):
    positions, velocities, accelerations, times = [], [], [], []
    elapsed = 0.
    for segment in segments:
        duration = float(segment['duration'])
        planner = problem.get_closed_loop_planner(duration)
        coeff = segment['coeff_state']
        count = max(2, int(np.ceil(duration / sample_dt)) + 1)
        for t in np.linspace(0., duration, count, endpoint=False):
            positions.append(planner.get_pos(coeff, t).reshape(2))
            velocities.append(planner.get_vel(coeff, t).reshape(2))
            accelerations.append(planner.get_acc(coeff, t).reshape(2))
            times.append(elapsed + float(t))
        elapsed += duration
    if segments:
        segment = segments[-1]; duration = float(segment['duration'])
        planner = problem.get_closed_loop_planner(duration)
        coeff = segment['coeff_state']
        positions.append(planner.get_pos(coeff, duration).reshape(2))
        velocities.append(planner.get_vel(coeff, duration).reshape(2))
        accelerations.append(planner.get_acc(coeff, duration).reshape(2))
        times.append(elapsed)
    return (np.asarray(positions), np.asarray(velocities), np.asarray(accelerations),
            np.asarray(times))


def replay_minco(problem, state, action, *, dt=1., sample_dt=.02, robots=None):
    """Replay all selected robot commands through the existing MINCO backend."""
    state = np.asarray(state, dtype=float).reshape(-1, 1)
    action = np.asarray(action, dtype=float).reshape(problem.action_dim, 1)
    robots = tuple(range(problem.num_robots)) if robots is None else tuple(robots)
    tracks = []
    raw_targets, feasible_targets = [], []
    for robot in robots:
        if not problem.is_active(state, robot):
            p = state[problem.state_idxs[robot], 0].copy()
            tracks.append((p[None], np.zeros((1, 2)), np.zeros((1, 2)), np.zeros(1)))
            raw_targets.append(p.copy()); feasible_targets.append(p.copy())
            continue
        result = problem.rollout_robot_state(state, action, robot, float(dt))
        tracks.append(_sample_segments(problem, result[-1], sample_dt))
        raw_targets.append(np.asarray(result[0], dtype=float).reshape(2))
        feasible_targets.append(np.asarray(result[1], dtype=float).reshape(2))
    n = max(len(x[0]) for x in tracks)
    def pad(index):
        out = np.zeros((n, len(tracks), 2))
        for j, track in enumerate(tracks):
            values = track[index]
            out[:, j] = values[np.minimum(np.arange(n), len(values)-1)]
        return out
    times = np.linspace(0., float(dt), n)
    positions, velocities, accelerations = pad(0), pad(1), pad(2)
    boxes = np.asarray(getattr(problem, 'obstacles', []), dtype=float).reshape(-1, 2, 2)
    collision = False
    if len(boxes):
        inside = ((positions[:, :, None, :] >= boxes[None, None, :, :, 0]) &
                  (positions[:, :, None, :] <= boxes[None, None, :, :, 1]))
        collision = bool(np.any(inside.all(axis=-1)))
    return MincoAudit(positions, velocities, accelerations, times, collision,
                      float(np.max(np.linalg.norm(velocities, axis=-1))),
                      float(np.max(np.linalg.norm(accelerations, axis=-1))),
                      np.asarray(raw_targets), np.asarray(feasible_targets))


def replay_minco_with_planner(problem, state, action, planner_cls, *, dt=1.,
                              sample_dt=.02, robots=None):
    """Run the same replay with an isolated planner class, then restore state."""
    old_get = problem.get_closed_loop_planner
    old_planners = getattr(problem, '_closed_loop_planners', None)
    old_coeff = getattr(problem, '_coeff_state_cache', None)
    cache = {}

    def get_planner(piece_dt):
        key = round(float(piece_dt), 8)
        if key not in cache:
            cache[key] = planner_cls(piece_dt, getattr(problem, 'closed_loop_ratio', .1))
        return cache[key]

    problem.get_closed_loop_planner = get_planner
    problem._closed_loop_planners = cache
    problem._coeff_state_cache = {}
    try:
        return replay_minco(problem, state, action, dt=dt, sample_dt=sample_dt, robots=robots)
    finally:
        problem.get_closed_loop_planner = old_get
        if old_planners is not None:
            problem._closed_loop_planners = old_planners
        if old_coeff is not None:
            problem._coeff_state_cache = old_coeff
