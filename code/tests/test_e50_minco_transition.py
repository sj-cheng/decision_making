import unittest
import numpy as np

from problems.example8 import Example8
from solvers.e50_minco_transition import minco_transition


class E50MincoTransitionTests(unittest.TestCase):
    def test_synchronous_transition_preserves_robot_slots(self):
        p = Example8(); p.obstacles = []
        state = np.zeros((p.state_dim, 1))
        for robot, point in enumerate(([-2., 0.], [2., 0.], [-2., -2.], [2., -2.])):
            state[p.state_idxs[robot], 0] = point
        state[p.active_idxs, 0] = 1.
        action = np.zeros((p.action_dim, 1)); action[p.action_idxs[2], 0] = [1., 0.]
        result = minco_transition(p, state, action, capture_dt=.05)
        self.assertEqual(result.positions.shape[1:], (4, 2))
        self.assertAlmostEqual(result.state[p.time_idx, 0], 1.)
        self.assertFalse(result.obstacle_violation)
        self.assertEqual(result.capture_events, ())

    def test_obstacle_violation_is_reported_from_sampled_tracks(self):
        p = Example8(); p.obstacles = [np.array([[-1.95, -1.5], [-2.5, -1.5]])]
        state = np.zeros((p.state_dim, 1))
        for robot, point in enumerate(([-2., 0.], [2., 0.], [-2., -2.], [2., -2.])):
            state[p.state_idxs[robot], 0] = point
        state[p.active_idxs, 0] = 1.
        action = np.zeros((p.action_dim, 1)); action[p.action_idxs[2], 0] = [2., 0.]
        result = minco_transition(p, state, action, capture_dt=.05)
        self.assertTrue(result.obstacle_violation)


if __name__ == '__main__': unittest.main()
