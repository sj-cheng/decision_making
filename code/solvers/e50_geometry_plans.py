"""Role-free high-level candidate generation for the E50 simulation path.

This module deliberately wraps legacy action proposals instead of changing the
frozen E49 controller. A later adapter can replace the endpoint generator with
LIDAR/map-aware geometry while keeping the JointPlan contract unchanged.
"""
from typing import Callable, Optional
import numpy as np

from solvers.e50_interfaces import JointPlan


def _goal_plan(starts, goals, duration, mode=("pursue", "pursue"), target_hint=(-1, -1)):
    starts = np.asarray(starts, dtype=float)
    goals = np.asarray(goals, dtype=float)
    # The two waypoint slots are intentionally both populated. The first is
    # the immediate endpoint for the legacy one-second action; the second is
    # reserved for the future short-horizon MINCO corridor.
    return JointPlan(np.stack((goals, goals), axis=1), duration, mode, target_hint)


def legacy_actions_to_plans(actions, starts, duration=1.0):
    """Wrap legacy (N,2,2) joint displacements as JointPlan candidates."""
    actions = np.asarray(actions, dtype=float)
    starts = np.asarray(starts, dtype=float)
    if actions.ndim != 3 or actions.shape[1:] != (2, 2) or starts.shape != (2, 2):
        raise ValueError("expected actions (N,2,2) and starts (2,2)")
    return tuple(_goal_plan(starts, starts + action, duration) for action in actions)


def belief_plan_candidates(starts, particles, weights, duration=1.0, max_candidates=16,
                           is_valid: Optional[Callable[[JointPlan], bool]] = None):
    """Generate role-free pursuit/cover proposals from public target beliefs.

    Each target contributes its highest-weight and weighted-quantile support
    points. Pairing those points across the two pursuers creates joint plans;
    no target assignment is forced and invalid plans can be filtered by the
    caller's local-map safety predicate.
    """
    starts = np.asarray(starts, dtype=float)
    particles = np.asarray(particles, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if starts.shape != (2, 2) or particles.ndim != 3 or particles.shape[-1] != 2:
        raise ValueError("invalid starts or particles shape")
    if weights.shape != particles.shape[:2]:
        raise ValueError("weights do not match particles")
    candidates = []
    for target in range(particles.shape[0]):
        active = weights[target] > 0
        if not np.any(active):
            continue
        ids = np.flatnonzero(active)
        order = ids[np.argsort(weights[target, ids])[::-1]]
        chosen = list(order[:3])
        for quantile in (.25, .5, .75):
            cumulative = np.cumsum(weights[target, ids])
            total = cumulative[-1]
            if total > 0:
                chosen.append(int(ids[min(len(ids)-1, np.searchsorted(cumulative, quantile*total))]))
        for idx in dict.fromkeys(chosen):
            goal = particles[target, idx]
            for swap in (False, True):
                goals = np.array([goal, goal], dtype=float)
                if swap:
                    goals = goals[::-1]
                mode = ("pursue", "pursue")
                hint = (target, target)
                plan = _goal_plan(starts, goals, duration, mode, hint)
                if is_valid is None or is_valid(plan):
                    candidates.append(plan)
                if len(candidates) >= max_candidates:
                    return tuple(candidates)
    if not candidates:
        candidates.append(_goal_plan(starts, starts, duration, ("wait", "wait")))
    return tuple(candidates)


def local_lidar_plan_candidates(starts, particles, weights, proxies, duration=2.0,
                                max_candidates=16, samples=32):
    """Generate belief plans while accepting only locally measured-safe paths.

    ``proxies`` contains one conservative range-only obstacle proxy per robot.
    Unknown space is never promoted to free space: a candidate is retained
    only when both straight local segments pass their measured-ray checks.
    If every proposal is rejected, the returned wait plan is the sole safe
    fallback.
    """
    starts = np.asarray(starts, dtype=float)
    if starts.shape != (2, 2) or len(proxies) != 2:
        raise ValueError("expected starts (2,2) and two local proxies")

    def is_valid(plan):
        goals = plan.waypoints[:, 0]
        return all(proxy.segment_safe(starts[i], goals[i], samples=samples)
                   for i, proxy in enumerate(proxies))

    return belief_plan_candidates(starts, particles, weights, duration=duration,
                                  max_candidates=max_candidates, is_valid=is_valid)
