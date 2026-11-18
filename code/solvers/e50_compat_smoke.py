"""Small paired execution audit for existing E46--E49 actions.

This is intentionally a controller-agnostic smoke helper. It compares the
legacy direct propagation with the existing Example8 MINCO backend and records
sensor summaries; it does not change any historical controller or claim an
unknown-map benchmark result.
"""
from dataclasses import dataclass
import numpy as np

from solvers.e50_lidar_map import raycast_lidar
from solvers.e50_local_obstacle import LocalObstacleProxy
from solvers.e50_endpoint_trajectory import feasible_endpoint_trajectory
from solvers.e50_minco_audit import (MincoAudit, replay_minco,
                                     replay_minco_with_planner)


def _collision(problem, positions):
    boxes = np.asarray(getattr(problem, 'obstacles', []), dtype=float).reshape(-1, 2, 2)
    if not len(boxes):
        return False
    inside = (positions[:, :, None, :] >= boxes[None, None, :, :, 0]) & \
             (positions[:, :, None, :] <= boxes[None, None, :, :, 1])
    return bool(np.any(inside.all(axis=-1)))


def replay_direct(problem, state, action, *, dt=1., sample_dt=.02, robots=None):
    """Replay an action with the historical direct displacement semantics."""
    state = np.asarray(state, dtype=float).reshape(-1, 1)
    action = np.asarray(action, dtype=float).reshape(problem.action_dim, 1)
    robots = tuple(range(problem.num_robots)) if robots is None else tuple(robots)
    starts, ends, velocities = [], [], []
    for robot in robots:
        start = state[problem.state_idxs[robot], 0].copy()
        if problem.is_active(state, robot):
            velocity = np.asarray(problem.action_to_velocity(action, robot, dt), dtype=float).reshape(2)
            end = np.asarray(problem.propagate_robot_state(state, velocity.reshape(2, 1), dt, robot), dtype=float).reshape(2)
        else:
            velocity = np.zeros(2); end = start.copy()
        starts.append(start); ends.append(end); velocities.append(velocity)
    n = max(2, int(np.ceil(float(dt) / sample_dt)) + 1)
    alpha = np.linspace(0., 1., n)[:, None, None]
    positions = np.asarray(starts)[None] + alpha * (np.asarray(ends)-np.asarray(starts))[None]
    vel = np.broadcast_to(np.asarray(velocities)[None], positions.shape).copy()
    acc = np.zeros_like(positions)
    times = np.linspace(0., float(dt), n)
    return MincoAudit(positions, vel, acc, times, _collision(problem, positions),
                      float(np.max(np.linalg.norm(vel, axis=-1))), 0.)


def replay_endpoint_semantic(problem, state, action, *, dt=1., sample_dt=.02, robots=None):
    """Execute legacy displacement goals with endpoint-constrained polynomials."""
    state = np.asarray(state, dtype=float).reshape(-1, 1)
    action = np.asarray(action, dtype=float).reshape(problem.action_dim, 1)
    robots = tuple(range(problem.num_robots)) if robots is None else tuple(robots)
    tracks = []
    feasible = True
    for robot in robots:
        start = state[problem.state_idxs[robot], 0].copy()
        if not problem.is_active(state, robot):
            n = max(2, int(np.ceil(dt / sample_dt)) + 1)
            tracks.append((np.repeat(start[None], n, axis=0), np.zeros((n, 2)),
                           np.zeros((n, 2)), np.linspace(0., dt, n)))
            continue
        goal = start + action[problem.action_idxs[robot], 0]
        v0 = state[problem.vel_idxs[robot], 0].copy()
        a0 = state[problem.acc_idxs[robot], 0].copy()
        max_v, max_a = problem.get_robot_dynamic_limits(robot)
        tr = feasible_endpoint_trajectory(start[None], goal[None], dt, sample_dt,
                                          v0[None], a0[None], max_v, max_a)
        feasible &= tr.feasible
        tracks.append((tr.positions[:, 0], tr.velocities[:, 0],
                       tr.accelerations[:, 0], tr.times))
    n = max(len(x[0]) for x in tracks)
    positions = np.zeros((n, len(tracks), 2)); velocities = positions.copy(); accelerations = positions.copy()
    for j, track in enumerate(tracks):
        idx = np.minimum(np.arange(n), len(track[0])-1)
        positions[:, j] = track[0][idx]; velocities[:, j] = track[1][idx]; accelerations[:, j] = track[2][idx]
    audit = MincoAudit(positions, velocities, accelerations, np.linspace(0., dt, n),
                       _collision(problem, positions),
                       float(np.max(np.linalg.norm(velocities, axis=-1))),
                       float(np.max(np.linalg.norm(accelerations, axis=-1))))
    return audit, feasible


@dataclass(frozen=True)
class CompatibilityResult:
    direct: MincoAudit
    minco: MincoAudit
    endpoint_delta: np.ndarray
    lidar_ranges: np.ndarray
    direct_lidar_safe: np.ndarray
    minco_lidar_safe: np.ndarray
    minco_error_to_feasible: np.ndarray


def compare_execution(problem, state, action, *, dt=1., sample_dt=.02, robots=None,
                      lidar_rays=256, lidar_range=10., minco_planner_cls=None):
    """Return paired direct/MINCO audits and simulator-only LiDAR summaries."""
    robots = tuple(range(problem.num_robots)) if robots is None else tuple(robots)
    direct = replay_direct(problem, state, action, dt=dt, sample_dt=sample_dt, robots=robots)
    if minco_planner_cls is None:
        minco = replay_minco(problem, state, action, dt=dt, sample_dt=sample_dt, robots=robots)
    else:
        minco = replay_minco_with_planner(problem, state, action, minco_planner_cls,
                                          dt=dt, sample_dt=sample_dt, robots=robots)
    endpoint_delta = minco.positions[-1] - direct.positions[-1]
    start = np.asarray([state[problem.state_idxs[r], 0] for r in robots])
    lidar = np.asarray([raycast_lidar(p, getattr(problem, 'obstacles', []),
                                      num_rays=lidar_rays, max_range=lidar_range)[1]
                        for p in start])
    direct_safe, minco_safe = [], []
    for i, point in enumerate(start):
        angles, ranges = raycast_lidar(point, getattr(problem, 'obstacles', []),
                                       num_rays=lidar_rays, max_range=lidar_range)
        proxy = LocalObstacleProxy(point, angles, ranges, lidar_range)
        direct_safe.append(all(proxy.endpoint_safe(q) for q in direct.positions[:, i]))
        minco_safe.append(all(proxy.endpoint_safe(q) for q in minco.positions[:, i]))
    feasible_error = (minco.positions[-1] - minco.feasible_targets
                      if minco.feasible_targets is not None else endpoint_delta)
    return CompatibilityResult(direct, minco, endpoint_delta, lidar,
                               np.asarray(direct_safe, bool), np.asarray(minco_safe, bool),
                               feasible_error)
