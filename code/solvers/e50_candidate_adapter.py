"""Compatibility bridge from the frozen E46 candidate pool to E50 plans."""
import numpy as np

from learning.e46_geometry_teacher import candidate_pool
from solvers.e50_geometry_plans import legacy_actions_to_plans


def candidate_plans(own, node, size=16, duration=1.0):
    """Reuse E46's public geometry/random pool without changing its semantics."""
    actions = candidate_pool(own, node, size=size)
    return legacy_actions_to_plans(actions, node.positions, duration=duration)


def plan_action(plan, starts):
    """Return the legacy endpoint displacement for a compatibility executor."""
    starts = np.asarray(starts, dtype=float)
    if starts.shape != (2, 2):
        raise ValueError('starts must have shape (2,2)')
    return np.asarray(plan.waypoints[:, 0] - starts, dtype=float)
