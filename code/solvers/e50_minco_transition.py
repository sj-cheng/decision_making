"""Synchronous multi-robot MINCO transition for compatibility experiments.

This is an execution-layer adapter. It does not replace the E49 direct-step
physics and is only used when a complete MINCO trajectory is intentionally
being audited.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MincoTransition:
    state: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    times: np.ndarray
    capture_events: tuple
    obstacle_violation: bool


def _track(problem, state, action, robot, dt, sample_dt):
    if not problem.is_active(state, robot):
        point = state[problem.state_idxs[robot], 0].copy()
        n = max(2, int(np.ceil(dt / sample_dt)) + 1)
        return (np.repeat(point[None, :], n, axis=0), np.zeros((n, 2)),
                np.zeros((n, 2)), np.linspace(0., dt, n))
    result = problem.rollout_robot_state(state, action, robot, dt)
    from solvers.e50_minco_audit import _sample_segments
    return _sample_segments(problem, result[-1], sample_dt)


def minco_transition(problem, state, action, *, dt=1., capture_dt=.01,
                     sample_dt=None, capture_fn=None, retire_pursuer=False):
    """Advance all robots on synchronized MINCO tracks and apply captures."""
    state = np.asarray(state, dtype=float).reshape(-1, 1)
    action = np.asarray(action, dtype=float).reshape(problem.action_dim, 1)
    if dt <= 0 or capture_dt <= 0:
        raise ValueError('dt and capture_dt must be positive')
    sample_dt = float(capture_dt if sample_dt is None else sample_dt)
    tracks = [_track(problem, state, action, r, float(dt), sample_dt)
              for r in range(problem.num_robots)]
    n = max(len(t[0]) for t in tracks)
    times = np.linspace(0., float(dt), n)
    positions = np.zeros((n, problem.num_robots, 2))
    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)
    for r, track in enumerate(tracks):
        idx = np.minimum(np.arange(n), len(track[0])-1)
        positions[:, r] = track[0][idx]
        velocities[:, r] = track[1][idx]
        accelerations[:, r] = track[2][idx]
    boxes = np.asarray(getattr(problem, 'obstacles', []), dtype=float).reshape(-1, 2, 2)
    inside = ((positions[:, :, None, :] >= boxes[None, None, :, :, 0]) &
              (positions[:, :, None, :] <= boxes[None, None, :, :, 1])) if len(boxes) else np.zeros((n, problem.num_robots, 0, 2), bool)
    obstacle_violation = bool(np.any(inside.all(axis=-1)))
    active_p = np.array([problem.is_active(state, r) for r in problem.pursuers], dtype=bool)
    active_e = np.array([problem.is_active(state, r) for r in problem.evaders], dtype=bool)
    events = []
    captured_positions = {}
    if capture_fn is None:
        def capture_fn(p, e, ap, ae, b):
            return []
    for k, t in enumerate(times):
        pursuers = positions[k, list(problem.pursuers)]
        evaders = positions[k, list(problem.evaders)].copy()
        # Captured targets no longer move, while active pursuers keep their
        # precomputed MINCO trajectory, matching the persistent-pursuit rule.
        for j in range(2):
            if j in captured_positions:
                evaders[j] = captured_positions[j]
        pairs = capture_fn(pursuers, evaders, active_p, active_e, boxes)
        for pair in pairs:
            pi, ei = int(pair[0]), int(pair[1])
            if not active_p[pi] or not active_e[ei]:
                continue
            active_e[ei] = False
            if retire_pursuer:
                active_p[pi] = False
            captured_positions[ei] = evaders[ei].copy()
            events.append(dict(time=float(t), pursuer=pi, evader=ei,
                               position=evaders[ei].copy()))
    nxt = state.copy()
    for local, robot in enumerate(problem.evaders):
        nxt[problem.state_idxs[robot], 0] = positions[-1, robot]
        nxt[problem.vel_idxs[robot], 0] = velocities[-1, robot]
        nxt[problem.acc_idxs[robot], 0] = accelerations[-1, robot]
    for local, robot in enumerate(problem.pursuers):
        nxt[problem.state_idxs[robot], 0] = positions[-1, robot]
        nxt[problem.vel_idxs[robot], 0] = velocities[-1, robot]
        nxt[problem.acc_idxs[robot], 0] = accelerations[-1, robot]
    for local, robot in enumerate(problem.evaders):
        nxt[problem.active_idxs[robot], 0] = active_e[local]
    for local, robot in enumerate(problem.pursuers):
        nxt[problem.active_idxs[robot], 0] = active_p[local]
    nxt[problem.time_idx, 0] = state[problem.time_idx, 0] + dt
    return MincoTransition(nxt, positions, velocities, accelerations, times,
                           tuple(events), obstacle_violation)
