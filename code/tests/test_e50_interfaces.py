import unittest
import numpy as np

from solvers.e50_interfaces import (BeliefState, JointPlan, LocalMapSnapshot,
                                    plan_to_trajectory, quintic_coefficients,
                                    sample_quintic)


class E50InterfaceTests(unittest.TestCase):
    def test_map_and_belief_validate_public_shapes(self):
        m = LocalMapSnapshot('map', 1.0, .1, [0, 0], np.zeros((4, 5)), np.ones((4, 5), bool))
        self.assertEqual(m.occupancy.shape, (4, 5))
        b = BeliefState(np.zeros((2, 8, 2)), np.full((2, 8), 1/8), [True, False], [.9, .2], 1.)
        self.assertEqual(b.particles.shape, (2, 8, 2))

    def test_quintic_hits_boundary_conditions(self):
        c = quintic_coefficients([[0., 0.]], [[2., -1.]], 1.)
        p, v, a = sample_quintic(c, 1., .01)
        np.testing.assert_allclose(p[0, 0], [0., 0.])
        np.testing.assert_allclose(p[-1, 0], [2., -1.])
        np.testing.assert_allclose(v[[0, -1]], 0., atol=1e-10)
        np.testing.assert_allclose(a[[0, -1]], 0., atol=1e-9)

    def test_joint_plan_produces_two_agent_trajectory(self):
        plan = JointPlan(np.array([[[1., 0.], [0., 0.]], [[0., 1.], [0., 0.]]]), 1.)
        command = plan_to_trajectory(plan, [[0., 0.], [1., 1.]])
        self.assertEqual(command.positions.shape[1:], (2, 2))
        np.testing.assert_allclose(command.positions[-1], [[1., 0.], [0., 1.]])

    def test_unknown_map_and_invalid_belief_are_rejected(self):
        with self.assertRaises(ValueError):
            LocalMapSnapshot('map', 1., .1, [0, 0], np.zeros((2, 2)), np.zeros((3, 2), bool))
        with self.assertRaises(ValueError):
            BeliefState(np.zeros((1, 2, 2)), np.ones((1, 2)), [True], [1.2], 0.)


if __name__ == '__main__':
    unittest.main()
