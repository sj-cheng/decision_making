"""Simple, role-aware baselines used by the reproducible benchmark.

The benchmark keeps the solver interface used by :mod:`run.py`: every solver
returns one joint action with shape ``[problem.action_dim, 1]``.  A solver can
be restricted to a team so that the other team's action is supplied by a
second solver.
"""

import time

import numpy as np

from solvers.solver import Solver


def _active(problem, state, robot):
    return not hasattr(problem, "is_active") or problem.is_active(state, robot)


def _clip_robot_action(problem, action, robot):
    idx = problem.action_idxs[robot]
    lims = problem.action_lims[idx, :]
    action[idx, :] = np.clip(action[idx, :], lims[:, 0:1], lims[:, 1:2])


class RandomBaseline(Solver):
    """Uniform random actions, optionally restricted to one team."""

    solver_name = "Random"

    def __init__(self, team=None):
        self.team = None if team is None else tuple(int(x) for x in team)

    def policy(self, problem, state):
        action = np.zeros((problem.action_dim, 1), dtype=float)
        robots = range(problem.num_robots) if self.team is None else self.team
        for robot in robots:
            if _active(problem, state, robot):
                action[problem.action_idxs[robot], :] = problem.sample_action()[problem.action_idxs[robot], :]
        return action


class GreedyPursuitEvasion(Solver):
    """Nearest-target pursuit / nearest-pursuer fleeing baseline.

    The rule uses only the current state, role membership and the current
    action limits.  The small boundary term prevents the fleeing baseline from
    being an artificially weak out-of-bounds policy.
    """

    solver_name = "Greedy"

    def __init__(self, team=None, boundary_weight=0.20):
        self.team = None if team is None else tuple(int(x) for x in team)
        self.boundary_weight = float(boundary_weight)

    def _direction(self, problem, state, robot):
        pos = state[problem.state_idxs[robot], :]
        if robot in getattr(problem, "pursuers", []):
            targets = [e for e in getattr(problem, "evaders", []) if _active(problem, state, e)]
            sign = 1.0
        else:
            targets = [p for p in getattr(problem, "pursuers", []) if _active(problem, state, p)]
            sign = -1.0
        if not targets:
            return np.zeros((2, 1), dtype=float)

        target = min(
            targets,
            key=lambda other: float(np.linalg.norm(pos - state[problem.state_idxs[other], :])),
        )
        direction = sign * (state[problem.state_idxs[target], :] - pos)

        # Boundary repulsion is only relevant for evaders.  It keeps the
        # baseline meaningful instead of rewarding accidental early exits.
        if sign < 0:
            low = problem.state_lims[problem.state_idxs[robot], 0:1]
            high = problem.state_lims[problem.state_idxs[robot], 1:2]
            extent = np.maximum(high - low, 1e-8)
            normalized = (pos - low) / extent
            boundary = (0.5 - normalized) * 2.0
            direction = direction + self.boundary_weight * boundary
        norm = float(np.linalg.norm(direction))
        return direction / norm if norm > 1e-10 else np.zeros((2, 1), dtype=float)

    def policy(self, problem, state):
        action = np.zeros((problem.action_dim, 1), dtype=float)
        robots = range(problem.num_robots) if self.team is None else self.team
        for robot in robots:
            if not _active(problem, state, robot):
                continue
            idx = problem.action_idxs[robot]
            direction = self._direction(problem, state, robot)
            lims = problem.action_lims[idx, :]
            # Use the largest isotropic step that remains inside the per-axis
            # action bounds.
            magnitude = float(np.min(np.abs(lims[:, 1:2])))
            action[idx, :] = direction * magnitude
            _clip_robot_action(problem, action, robot)
        return action


class TeamSolver(Solver):
    """Compose independent solvers for the two teams."""

    solver_name = "TeamSolver"

    def __init__(self, evader_solver, pursuer_solver):
        self.evader_solver = evader_solver
        self.pursuer_solver = pursuer_solver
        # These fields are populated on every policy call.  They let the
        # benchmark report the cost of each team's solver separately instead
        # of attributing the combined TeamSolver cost to both methods.
        self.last_evader_latency_ms = np.nan
        self.last_pursuer_latency_ms = np.nan
        self.last_joint_latency_ms = np.nan

    def policy(self, problem, state):
        joint_started = time.perf_counter()
        if self.evader_solver is self.pursuer_solver:
            solver_started = time.perf_counter()
            joint = self.evader_solver.policy(problem, state)
            solver_elapsed_ms = (time.perf_counter() - solver_started) * 1000.0
            evader_action = joint
            pursuer_action = joint
            self.last_evader_latency_ms = solver_elapsed_ms
            self.last_pursuer_latency_ms = solver_elapsed_ms
        else:
            solver_started = time.perf_counter()
            evader_action = self.evader_solver.policy(problem, state)
            self.last_evader_latency_ms = (time.perf_counter() - solver_started) * 1000.0

            solver_started = time.perf_counter()
            pursuer_action = self.pursuer_solver.policy(problem, state)
            self.last_pursuer_latency_ms = (time.perf_counter() - solver_started) * 1000.0
        action = np.zeros((problem.action_dim, 1), dtype=float)
        for robot in getattr(problem, "evaders", []):
            action[problem.action_idxs[robot], :] = evader_action[problem.action_idxs[robot], :]
        for robot in getattr(problem, "pursuers", []):
            action[problem.action_idxs[robot], :] = pursuer_action[problem.action_idxs[robot], :]
        self.last_joint_latency_ms = (time.perf_counter() - joint_started) * 1000.0
        return action


class RoleMaskedSolver(Solver):
    """Restrict an existing full-joint solver to one team."""

    def __init__(self, solver, team):
        self.solver = solver
        self.team = tuple(int(x) for x in team)
        self.solver_name = getattr(solver, "solver_name", "Masked")

    def policy(self, problem, state):
        # Search/policy/MAPPO solvers can avoid computing the other team's
        # action entirely.  Older generic solvers keep the legacy fallback so
        # this wrapper remains usable outside the benchmark.
        if getattr(self.solver, "supports_team_policy", False):
            proposed = self.solver.policy(problem, state, team=self.team)
        else:
            proposed = self.solver.policy(problem, state)
        action = np.zeros((problem.action_dim, 1), dtype=float)
        for robot in self.team:
            action[problem.action_idxs[robot], :] = proposed[problem.action_idxs[robot], :]
        return action
