"""Interfaces for the E50 real-deployment-oriented simulation path.

The existing E49 controllers continue to use their legacy map/action objects.
These small, validated value objects are the boundary for the new path:
SLAM/LiDAR -> belief -> joint plan -> polynomial trajectory -> executor.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


def _array(value, shape=None, dtype=float):
    out = np.asarray(value, dtype=dtype).copy()
    if shape is not None and out.shape != shape:
        raise ValueError(f"expected shape {shape}, got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite value")
    return out


@dataclass(frozen=True)
class LidarScan:
    """Range-only LiDAR observation shared by simulation and ROS adapters."""
    frame_id: str
    stamp: float
    angles: np.ndarray
    ranges: np.ndarray
    max_range: float

    def __post_init__(self):
        angles = _array(self.angles)
        ranges = _array(self.ranges)
        if not self.frame_id or angles.ndim != 1 or ranges.shape != angles.shape:
            raise ValueError("invalid lidar scan shape")
        if self.max_range <= 0 or np.any(ranges < 0) or np.any(ranges > self.max_range):
            raise ValueError("invalid lidar ranges")
        object.__setattr__(self, "angles", angles)
        object.__setattr__(self, "ranges", ranges)

    def __iter__(self):
        # Keep compatibility with callers that previously unpacked angles,
        # ranges from a simulator helper.
        yield self.angles
        yield self.ranges


@dataclass(frozen=True)
class LocalMapSnapshot:
    """A timestamped local/shared map consumed by one planning cycle."""
    frame_id: str
    stamp: float
    resolution: float
    origin: np.ndarray
    occupancy: np.ndarray
    unknown: np.ndarray
    distance_field: Optional[np.ndarray] = None

    def __post_init__(self):
        if not self.frame_id or self.resolution <= 0 or not np.isfinite(self.stamp):
            raise ValueError("invalid map metadata")
        origin = _array(self.origin, (2,))
        occupancy = np.asarray(self.occupancy, dtype=np.float32).copy()
        unknown = np.asarray(self.unknown, dtype=bool).copy()
        if occupancy.ndim != 2 or unknown.shape != occupancy.shape:
            raise ValueError("occupancy and unknown must be matching 2-D arrays")
        if not np.all(np.isfinite(occupancy)) or np.any((occupancy < 0) | (occupancy > 1)):
            raise ValueError("occupancy must be in [0, 1]")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "unknown", unknown)
        if self.distance_field is not None:
            distance = _array(self.distance_field)
            if distance.shape != occupancy.shape or np.any(distance < 0):
                raise ValueError("invalid distance field")
            object.__setattr__(self, "distance_field", distance)


@dataclass(frozen=True)
class BeliefState:
    """Public target belief; hidden target truth is never stored here."""
    particles: np.ndarray
    weights: np.ndarray
    visible: np.ndarray
    confidence: np.ndarray
    stamp: float

    def __post_init__(self):
        particles = np.asarray(self.particles, dtype=float).copy()
        weights = np.asarray(self.weights, dtype=float).copy()
        visible = np.asarray(self.visible, dtype=bool).copy()
        confidence = _array(self.confidence)
        if particles.ndim != 3 or particles.shape[-1] != 2:
            raise ValueError("particles must have shape (targets, particles, 2)")
        if weights.shape != particles.shape[:2] or visible.shape != (particles.shape[0],):
            raise ValueError("belief shapes do not agree")
        if (confidence.shape != visible.shape or not np.all(np.isfinite(particles))
                or not np.all(np.isfinite(weights)) or np.any(weights < 0)
                or np.any((confidence < 0) | (confidence > 1))):
            raise ValueError("invalid belief values")
        sums = weights.sum(axis=1)
        if np.any((sums > 0) & ~np.isclose(sums, 1.0, atol=1e-6)):
            raise ValueError("each non-empty belief must be normalized")
        if not np.isfinite(self.stamp):
            raise ValueError("invalid belief timestamp")
        object.__setattr__(self, "particles", particles)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "visible", visible)
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class JointPlan:
    """A role-free two-pursuer high-level proposal for one replanning window."""
    waypoints: np.ndarray
    duration: float
    mode: Tuple[str, str] = ("pursue", "pursue")
    target_hint: Tuple[int, int] = (-1, -1)
    unknown_cost: float = 0.0
    geometry_score: float = 0.0

    def __post_init__(self):
        points = _array(self.waypoints)
        if points.shape != (2, 2, 2):
            raise ValueError("waypoints must have shape (2 pursuers, 2 points, 2D)")
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("duration must be positive")
        if len(self.mode) != 2 or len(self.target_hint) != 2:
            raise ValueError("joint plan must contain two pursuer entries")
        object.__setattr__(self, "waypoints", points)


@dataclass(frozen=True)
class TrajectoryCommand:
    """Sampled executable trajectories plus their source plan identity."""
    stamp: float
    dt: float
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    plan: JointPlan
    sequence: int = 0

    def __post_init__(self):
        positions = _array(self.positions)
        velocities = _array(self.velocities)
        accelerations = _array(self.accelerations)
        if positions.ndim != 3 or positions.shape[-1] != 2 or positions.shape[1] != 2:
            raise ValueError("trajectory positions must have shape (samples, 2, 2)")
        if velocities.shape != positions.shape or accelerations.shape != positions.shape:
            raise ValueError("trajectory arrays must have matching shapes")
        if self.dt <= 0 or not np.isfinite(self.stamp):
            raise ValueError("invalid trajectory timing")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "accelerations", accelerations)


def quintic_coefficients(start, goal, duration, start_velocity=None, goal_velocity=None,
                         start_acceleration=None, goal_acceleration=None):
    """Return minimum-jerk quintic coefficients for one or more 2-D points."""
    start, goal = _array(start), _array(goal)
    if start.shape != goal.shape or start.shape[-1] != 2 or duration <= 0:
        raise ValueError("start/goal must match and duration must be positive")
    v0 = np.zeros_like(start) if start_velocity is None else _array(start_velocity, start.shape)
    vf = np.zeros_like(start) if goal_velocity is None else _array(goal_velocity, start.shape)
    a0 = np.zeros_like(start) if start_acceleration is None else _array(start_acceleration, start.shape)
    af = np.zeros_like(start) if goal_acceleration is None else _array(goal_acceleration, start.shape)
    T = float(duration)
    c = np.zeros(start.shape + (6,), dtype=float)
    c[..., 0], c[..., 1], c[..., 2] = start, v0, a0 / 2.
    dp = goal - (start + v0*T + .5*a0*T*T)
    dv = vf - (v0 + a0*T)
    da = af - a0
    c[..., 3] = 10*dp/T**3 - 4*dv/T**2 + .5*da/T
    c[..., 4] = -15*dp/T**4 + 7*dv/T**3 - da/T**2
    c[..., 5] = 6*dp/T**5 - 3*dv/T**4 + .5*da/T**3
    return c


def sample_quintic(coefficients, duration, dt):
    """Sample coefficients into position, velocity and acceleration arrays."""
    c = _array(coefficients)
    if c.shape[-2:] != (2, 6) or duration <= 0 or dt <= 0:
        raise ValueError("coefficients must have shape (..., 2, 6)")
    n = max(2, int(np.ceil(duration / dt)) + 1)
    t = np.linspace(0., float(duration), n)
    p = np.zeros((n,) + c.shape[:-1]); v = p.copy(); a = p.copy()
    for k in range(6):
        p += t.reshape((-1,) + (1,)*(c.ndim-1))**k * c[..., k]
        if k >= 1: v += k*t.reshape((-1,) + (1,)*(c.ndim-1))**(k-1) * c[..., k]
        if k >= 2: a += k*(k-1)*t.reshape((-1,) + (1,)*(c.ndim-1))**(k-2) * c[..., k]
    return p, v, a


def plan_to_trajectory(plan, starts, dt=0.02):
    """Convert a JointPlan's first waypoint into a two-agent trajectory."""
    starts = _array(starts, (2, 2))
    coeff = quintic_coefficients(starts, plan.waypoints[:, 0], plan.duration)
    p, v, a = sample_quintic(coeff, plan.duration, dt)
    return TrajectoryCommand(0., dt, p, v, a, plan)
