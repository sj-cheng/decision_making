"""Numerical audit helpers for Example8's existing MINCO implementation."""
import numpy as np

from problems.example8 import ClosedLoopMincoPlanner, MincoTrajFactory, PolyTraj


def audit_polytraj_boundary(duration=1.0):
    q0 = np.zeros((3, 2)); qT = np.array([[1., -2.], [0., 0.], [0., 0.]])
    poly = PolyTraj.init_by_qT(q0, qT, [0., duration])
    return dict(position_start=poly.get_pos(0.).tolist(), position_end=poly.get_pos(duration).tolist(),
                velocity_start=poly.get_vel(0.).tolist(), velocity_end=poly.get_vel(duration).tolist(),
                acceleration_start=poly.get_acc(0.).tolist(), acceleration_end=poly.get_acc(duration).tolist())


def audit_closed_loop(piece_dt=.1, ratio=.1, iterations=10):
    planner = ClosedLoopMincoPlanner(piece_dt, ratio)
    coeff = planner.coeff_state_from_kinematics([0., 0.], [0., 0.], [0., 0.])
    target = np.array([[1., 0.]])
    for _ in range(int(iterations)):
        coeff = planner.iterate(target, coeff)
    return dict(piece_dt=piece_dt, ratio=ratio, planning_t=planner.planning_t,
                spectral_radius=float(np.max(np.abs(np.linalg.eigvals(planner.mat_F_stab)))),
                endpoint=planner.get_pos(coeff, piece_dt).reshape(2).tolist(),
                matrix_condition=float(np.linalg.cond(planner.construct_minco_m2(planner.planning_t))))


def audit_parameter_grid(pieces=(.02, .05, .1, .2), ratios=(.1, .5, 1.)):
    rows = []
    for piece in pieces:
        for ratio in ratios:
            try:
                p = ClosedLoopMincoPlanner(piece, ratio)
                rows.append(dict(piece_dt=piece, ratio=ratio, planning_t=p.planning_t,
                                 spectral_radius=float(np.max(np.abs(np.linalg.eigvals(p.mat_F_stab)))),
                                 matrix_condition=float(np.linalg.cond(p.construct_minco_m2(p.planning_t)))))
            except Exception as exc:
                rows.append(dict(piece_dt=piece, ratio=ratio, error=repr(exc)))
    return rows


def audit_factory_without_cost():
    factory = MincoTrajFactory()
    try:
        factory.getMatrixMInv([0., 1.], 2)
    except Exception as exc:
        return dict(status='error', error=type(exc).__name__, message=str(exc))
    return dict(status='ok')
