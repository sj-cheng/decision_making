"""E50-only corrected MINCO planner.

Example8's historical planner is left untouched. Its free-terminal-condition
row uses M^{-1}; the equivalent MincoTrajFactory derivation and the gradient
condition require M^{-T}. This isolated subclass permits paired old/corrected
audits before changing any shared benchmark.
"""
import numpy as np

from problems.example8 import (ClosedLoopMincoPlanner, POLY_DEGREE,
                                PolyTraj)


class CorrectedClosedLoopMincoPlanner(ClosedLoopMincoPlanner):
    @staticmethod
    def construct_minco_m2(pieceT):
        mat_m = np.zeros((POLY_DEGREE + 1, POLY_DEGREE + 1), dtype=float)
        for i in range(POLY_DEGREE - 1):
            mat_m[i, :] = PolyTraj.construct_beta(0.0, i)
        mat_m[-2, :] = PolyTraj.construct_beta(pieceT, 0)
        mat_m[-1, :] = PolyTraj.construct_beta(pieceT, 1)
        mat_m_inv = np.linalg.inv(mat_m)
        mat_supp = np.array([[0, 0, 0, 0, 0, 1]], dtype=float) @ mat_m_inv.T
        mat_supp = mat_supp @ ClosedLoopMincoPlanner.construct_bbint(pieceT, rank=3)
        mat_m[-1, :] = mat_supp[-1, :]
        return mat_m


def free_terminal_gradient(planner, pieceT):
    """Return the terminal free-condition residual under M^{-T} H."""
    mat = np.zeros((POLY_DEGREE + 1, POLY_DEGREE + 1), dtype=float)
    for i in range(POLY_DEGREE - 1):
        mat[i, :] = PolyTraj.construct_beta(0., i)
    mat[-2, :] = PolyTraj.construct_beta(pieceT, 0)
    mat[-1, :] = PolyTraj.construct_beta(pieceT, 1)
    inv = np.linalg.inv(mat)
    row = np.array([[0, 0, 0, 0, 0, 1.]], dtype=float) @ inv.T
    return row @ planner.construct_bbint(pieceT, rank=3)
