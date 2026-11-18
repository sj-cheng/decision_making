import unittest
import numpy as np

from solvers.e50_endpoint_trajectory import endpoint_trajectory, feasible_endpoint_trajectory


class E50EndpointTrajectoryTests(unittest.TestCase):
    def test_reaches_goal_and_carries_nonzero_state(self):
        start = np.array([[0., 0.]])
        goal = np.array([[.5, .2]])
        v0 = np.array([[.1, -.05]])
        a0 = np.array([[.02, .01]])
        vf = np.array([[.15, -.02]])
        af = np.array([[.01, .01]])
        tr = endpoint_trajectory(start, goal, initial_velocity=v0,
                                 initial_acceleration=a0, terminal_velocity=vf,
                                 terminal_acceleration=af)
        np.testing.assert_allclose(tr.positions[0], start, atol=1e-10)
        np.testing.assert_allclose(tr.positions[-1], goal, atol=1e-10)
        np.testing.assert_allclose(tr.velocities[0], v0, atol=1e-10)
        np.testing.assert_allclose(tr.velocities[-1], vf, atol=1e-10)
        self.assertTrue(tr.feasible)

    def test_reports_dynamic_infeasibility_without_silent_clipping(self):
        tr = endpoint_trajectory([[0., 0.]], [[2., 0.]], max_speed=.5,
                                 max_acceleration=.5)
        self.assertFalse(tr.feasible)
        np.testing.assert_allclose(tr.positions[-1], [[2., 0.]], atol=1e-10)

    def test_receding_horizon_brakes_before_target_reversal(self):
        tr = feasible_endpoint_trajectory([[-2.62, .22]], [[-3.62, -.72]], 2., .02,
                                           [[.23, .94]], [[0., 0.]], 1.5, 6.)
        self.assertTrue(tr.feasible)
        self.assertLessEqual(tr.max_speed, 1.5 + 1e-9)


if __name__ == '__main__': unittest.main()
