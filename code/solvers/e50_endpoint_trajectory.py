"""Endpoint-semantic polynomial trajectories for the E50 compatibility audit.

Legacy E46-E49 actions are one-cycle displacement goals. This adapter reaches
that goal within the cycle while carrying measured initial velocity and
acceleration. Terminal derivatives are configurable; they are not implicitly
treated as hidden zero-state assumptions.
"""
from dataclasses import dataclass
import numpy as np

from solvers.e50_interfaces import quintic_coefficients, sample_quintic


@dataclass(frozen=True)
class EndpointTrajectory:
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    times: np.ndarray
    terminal_velocity: np.ndarray
    terminal_acceleration: np.ndarray
    feasible: bool
    max_speed: float
    max_acceleration: float


def endpoint_trajectory(start, goal, duration=1., dt=.02, initial_velocity=None,
                        initial_acceleration=None, terminal_velocity=None,
                        terminal_acceleration=None, max_speed=None,
                        max_acceleration=None):
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    if start.shape != goal.shape or start.ndim != 2 or start.shape[1] != 2:
        raise ValueError('start and goal must have shape (agents,2)')
    zeros = np.zeros_like(start)
    v0 = zeros if initial_velocity is None else np.asarray(initial_velocity, float)
    a0 = zeros if initial_acceleration is None else np.asarray(initial_acceleration, float)
    vf = v0.copy() if terminal_velocity is None else np.asarray(terminal_velocity, float)
    af = a0.copy() if terminal_acceleration is None else np.asarray(terminal_acceleration, float)
    for value in (v0, a0, vf, af):
        if value.shape != start.shape or not np.all(np.isfinite(value)):
            raise ValueError('invalid endpoint derivative')
    coeff = quintic_coefficients(start, goal, duration, v0, vf, a0, af)
    positions, velocities, accelerations = sample_quintic(coeff, duration, dt)
    # Example8 stores independent x/y limits. Use the largest component here
    # so feasibility matches the simulator's per-axis dynamic constraints;
    # callers can still compute vector norms from the returned samples.
    max_v = float(np.max(np.abs(velocities)))
    max_a = float(np.max(np.abs(accelerations)))
    feasible = ((max_speed is None or max_v <= float(max_speed) + 1e-9) and
                (max_acceleration is None or max_a <= float(max_acceleration) + 1e-9))
    return EndpointTrajectory(positions, velocities, accelerations,
                              np.linspace(0., float(duration), len(positions)),
                              vf, af, feasible, max_v, max_a)


def feasible_endpoint_trajectory(start, goal, duration=1., dt=.02,
                                 initial_velocity=None, initial_acceleration=None,
                                 max_speed=None, max_acceleration=None):
    """Choose a smooth terminal velocity that respects current limits.

    Receding-horizon execution may discard the planned tail at the next
    decision. We therefore try state-continuous velocity first, then braking
    and goal-directed alternatives when a target reversal makes it infeasible.
    """
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    v0 = np.zeros_like(start) if initial_velocity is None else np.asarray(initial_velocity, float)
    a0 = np.zeros_like(start) if initial_acceleration is None else np.asarray(initial_acceleration, float)
    candidates = [v0.copy(), np.zeros_like(v0)]
    candidates.append(np.clip((goal-start)/float(duration), -float(max_speed), float(max_speed))
                     if max_speed is not None else (goal-start)/float(duration))
    results = [endpoint_trajectory(start, goal, duration, dt, v0, a0, vf, a0,
                                   max_speed, max_acceleration) for vf in candidates]
    feasible = [r for r in results if r.feasible]
    if feasible:
        return min(feasible, key=lambda r: (r.max_acceleration, r.max_speed))
    return min(results, key=lambda r: (max(r.max_speed/(max_speed or np.inf),
                                           r.max_acceleration/(max_acceleration or np.inf)),
                                       r.max_acceleration, r.max_speed))
