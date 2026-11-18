import unittest
import numpy as np

from problems.example8 import ClosedLoopMincoPlanner, PolyTraj


class E50MincoMathTests(unittest.TestCase):
    def test_quintic_boundary_conditions_are_exact(self):
        q0 = np.array([[0.2, -0.3], [0.4, 0.1], [0.0, -0.2]])
        qT = np.array([[1.2, 0.7], [-0.1, 0.2], [0.3, 0.0]])
        poly = PolyTraj.init_by_qT(q0, qT, [0., 1.])
        np.testing.assert_allclose(poly.get_pos(0.), q0[0], atol=1e-10)
        np.testing.assert_allclose(poly.get_vel(0.), q0[1], atol=1e-10)
        np.testing.assert_allclose(poly.get_acc(0.), q0[2], atol=1e-10)
        np.testing.assert_allclose(poly.get_pos(1.), qT[0], atol=1e-10)
        np.testing.assert_allclose(poly.get_vel(1.), qT[1], atol=1e-10)
        np.testing.assert_allclose(poly.get_acc(1.), qT[2], atol=1e-10)

    def test_default_closed_loop_is_stable_but_not_one_second_endpoint_tracking(self):
        planner = ClosedLoopMincoPlanner(.1, .1)
        self.assertLess(max(abs(np.linalg.eigvals(planner.mat_F_stab))), 1.)
        coeff = planner.coeff_state_from_kinematics([0., 0.], [0., 0.], [0., 0.])
        target = np.array([[1., 1.]])
        for _ in range(10):
            coeff = planner.iterate(target, coeff)
        endpoint = planner.get_pos(coeff, .1).ravel()
        self.assertGreater(endpoint[0], 0.)
        self.assertLess(endpoint[0], .5)

    def test_default_dynamic_bounds_clip_one_meter_command(self):
        planner = ClosedLoopMincoPlanner(.1, .1)
        coeff = planner.coeff_state_from_kinematics([0., 0.], [0., 0.], [0., 0.])
        lower, upper = planner.bound_all(coeff, 20, 1., 1.)
        self.assertLess(float(upper[0, 0]), .5)
        self.assertGreater(float(lower[0, 0]), -.5)


if __name__ == '__main__': unittest.main()
