"""Frozen camera/gimbal observation model for E50 simulation.

The real camera detector is treated as a frozen module.  This simulator only
models its public range/bearing output and the pan/tilt gimbal geometry.
"""
from dataclasses import dataclass
import math
import numpy as np


def wrap_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def segment_clear(start, end, boxes):
    """Return whether a line of sight misses every axis-aligned obstacle."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    delta = end - start
    for box in np.asarray(boxes, float).reshape(-1, 2, 2):
        t0, t1 = 0.0, 1.0
        for axis in range(2):
            if abs(delta[axis]) < 1e-12:
                if box[axis, 0] <= start[axis] <= box[axis, 1]:
                    return False
                continue
            a = (box[axis, 0] - start[axis]) / delta[axis]
            b = (box[axis, 1] - start[axis]) / delta[axis]
            t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
            if t0 > t1:
                break
        else:
            # Touching an obstacle is an occlusion in the sensor model.
            if t1 >= t0 and t1 >= 0.0 and t0 <= 1.0 and t0 < 1.0 - 1e-9:
                return False
    return True


@dataclass
class CameraFrame:
    reports: list
    visible: np.ndarray
    covered: np.ndarray
    gimbal_yaw: np.ndarray
    state: list
    lost_reason: list
    measurements: list
    fov_rad: float = 0.
    max_range: float = 20.


class CameraGimbalTracker:
    """Per-pursuer search/lock tracker with a 60 degree camera FOV."""
    def __init__(self, pursuers=2, *, fov_deg=60., gimbal_limit_deg=90.,
                 sweep_rate_deg_s=960., max_range=20., noise_std=0., seed=0):
        self.fov = math.radians(float(fov_deg))
        self.limit = math.radians(float(gimbal_limit_deg))
        self.rate = math.radians(float(sweep_rate_deg_s))
        self.max_range = float(max_range)
        self.noise_std = float(noise_std)
        if self.fov <= 0 or self.limit <= 0 or self.rate <= 0 or self.max_range <= 0:
            raise ValueError('invalid camera parameters')
        self.yaw = np.zeros(int(pursuers), dtype=float)
        self.direction = np.ones(int(pursuers), dtype=float)
        self.locked = [None] * int(pursuers)
        self.state = ['SEARCH'] * int(pursuers)
        self.rng = np.random.default_rng(seed)

    def _advance_search(self, i, dt):
        remaining = self.rate * float(dt)
        while remaining > 1e-12:
            boundary = self.limit - self.yaw[i] if self.direction[i] > 0 else -self.limit - self.yaw[i]
            move = min(abs(boundary), remaining) * self.direction[i]
            self.yaw[i] += move
            remaining -= abs(move)
            if abs(boundary) <= abs(move) + 1e-12:
                self.yaw[i] = np.clip(self.yaw[i], -self.limit, self.limit)
                self.direction[i] *= -1.

    def _aim(self, i, target_angle, dt):
        error = wrap_angle(target_angle - self.yaw[i])
        step = np.clip(error, -self.rate * float(dt), self.rate * float(dt))
        self.yaw[i] = float(np.clip(self.yaw[i] + step, -self.limit, self.limit))

    def _step_once(self, observer_positions, target_positions, active_p, active_e,
                   headings, obstacles, *, dt=1.):
        observers = np.asarray(observer_positions, float).reshape(-1, 2)
        targets = np.asarray(target_positions, float).reshape(-1, 2)
        headings = np.asarray(headings, float).reshape(-1)
        active_p, active_e = np.asarray(active_p, bool), np.asarray(active_e, bool)
        if len(observers) != len(self.yaw) or len(headings) != len(observers):
            raise ValueError('camera observer shape mismatch')
        reports = [[None for _ in range(len(observers))] for _ in range(len(targets))]
        visible = np.zeros((len(observers), len(targets)), bool)
        covered = np.zeros_like(visible)
        measurements = [[None for _ in range(len(targets))] for _ in range(len(observers))]
        lost_reason = [None] * len(observers)
        for i, observer in enumerate(observers):
            if not active_p[i]:
                self.locked[i], self.state[i] = None, 'INACTIVE'
                continue
            lock = self.locked[i]
            if lock is not None and (lock >= len(targets) or not active_e[lock]):
                lock = self.locked[i] = None
            if lock is None:
                self.state[i] = 'SEARCH'
                candidates = []
                for j, target in enumerate(targets):
                    if not active_e[j]:
                        continue
                    d = target - observer; distance = float(np.linalg.norm(d))
                    if distance > self.max_range or distance <= 1e-9:
                        continue
                    bearing = wrap_angle(math.atan2(d[1], d[0]) - headings[i])
                    rel = wrap_angle(bearing - self.yaw[i])
                    in_fov = abs(rel) <= self.fov / 2. + 1e-9
                    if in_fov:
                        covered[i, j] = True
                    if in_fov and segment_clear(observer, target, obstacles):
                        candidates.append((distance, j, bearing))
                if candidates:
                    _, lock, _ = min(candidates)
                    self.locked[i] = lock
                    self.state[i] = 'TRACK'
                else:
                    self._advance_search(i, dt)
            if self.locked[i] is not None:
                lock = self.locked[i]
                target = targets[lock]
                d = target - observer; distance = float(np.linalg.norm(d))
                bearing = wrap_angle(math.atan2(d[1], d[0]) - headings[i])
                if distance > self.max_range or abs(bearing) > self.limit + self.fov / 2. + 1e-9:
                    self.locked[i] = None; self.state[i] = 'SEARCH'; lost_reason[i] = 'out_of_fov'
                    self._advance_search(i, dt)
                    continue
                if not segment_clear(observer, target, obstacles):
                    self.locked[i] = None; self.state[i] = 'SEARCH'; lost_reason[i] = 'occlusion'
                    self._advance_search(i, dt)
                    continue
                self._aim(i, bearing, dt)
                rel = wrap_angle(bearing - self.yaw[i])
                if abs(rel) <= self.fov / 2. + 1e-9:
                    noisy_r = max(0., distance + self.rng.normal(0., self.noise_std))
                    noisy_b = wrap_angle(bearing + self.rng.normal(0., self.noise_std / max(distance, 1.)))
                    estimate = observer + noisy_r * np.array([math.cos(headings[i] + noisy_b),
                                                               math.sin(headings[i] + noisy_b)])
                    reports[lock][i] = estimate.tolist()
                    visible[i, lock] = True
                    covered[i, lock] = True
                    measurements[i][lock] = dict(range=float(noisy_r), bearing=float(noisy_b),
                                                 target_id=int(lock), gimbal_yaw=float(self.yaw[i]))
        return CameraFrame(reports, visible, covered, self.yaw.copy(), list(self.state),
                           lost_reason, measurements, self.fov, self.max_range)

    def step(self, observer_positions, target_positions, active_p, active_e,
             headings, obstacles, *, dt=1.):
        """Advance the gimbal in 50 ms sensor frames within one planner step."""
        count = max(1, int(math.ceil(float(dt) / .05)))
        frame = None
        reasons = [None] * len(self.yaw)
        for _ in range(count):
            frame = self._step_once(observer_positions, target_positions, active_p,
                                    active_e, headings, obstacles, dt=float(dt) / count)
            for i, reason in enumerate(frame.lost_reason):
                if reason is not None:
                    reasons[i] = reason
        frame.lost_reason = reasons
        return frame
