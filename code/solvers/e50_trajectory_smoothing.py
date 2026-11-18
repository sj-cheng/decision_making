"""Cross-cycle action continuity for the E50 rolling executor."""
import numpy as np


def rate_limit_displacement(desired, previous, *, max_delta=.6):
    """Limit per-robot goal displacement changes between replans.

    The first cycle is handled by the caller.  This helper only changes a
    target when its Euclidean change from the previous target exceeds the
    configured envelope, preserving the method's direction as much as
    possible while preventing an instantaneous reversal at a replan boundary.
    """
    desired = np.asarray(desired, dtype=float)
    previous = np.asarray(previous, dtype=float)
    if desired.shape != (2, 2) or previous.shape != desired.shape:
        raise ValueError('desired and previous must have shape (2,2)')
    if not np.isfinite(desired).all() or not np.isfinite(previous).all():
        raise ValueError('actions must be finite')
    max_delta = float(max_delta)
    if max_delta <= 0 or not np.isfinite(max_delta):
        raise ValueError('max_delta must be positive and finite')
    result = desired.copy()
    for robot in range(2):
        delta = desired[robot] - previous[robot]
        norm = float(np.linalg.norm(delta))
        if norm > max_delta:
            result[robot] = previous[robot] + delta * (max_delta / norm)
    return result
