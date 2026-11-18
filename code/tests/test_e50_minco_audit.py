import unittest
import numpy as np

from problems.example8 import Example8
from solvers.e50_minco_audit import replay_minco


class E50MincoAuditTests(unittest.TestCase):
    def test_existing_example8_minco_backend_is_replayable(self):
        problem = Example8()
        problem.obstacles = []
        state = np.zeros((problem.state_dim, 1))
        state[problem.state_idxs[0], 0] = [-2., 0.]
        state[problem.state_idxs[1], 0] = [2., 0.]
        state[problem.state_idxs[2], 0] = [-2., -2.]
        state[problem.state_idxs[3], 0] = [2., -2.]
        state[problem.active_idxs, 0] = 1.
        action = np.zeros((problem.action_dim, 1))
        action[problem.action_idxs[2], 0] = [1., 0.]
        audit = replay_minco(problem, state, action, robots=(2, 3), sample_dt=.05)
        self.assertGreater(len(audit.times), 2)
        self.assertEqual(audit.positions.shape[1:], (2, 2))
        self.assertFalse(audit.collision)
        self.assertGreater(audit.positions[-1, 0, 0], -2.)

    def test_collision_flag_requires_both_coordinates_inside_a_box(self):
        problem = Example8()
        problem.obstacles = [np.array([[0., 1.], [0., 1.]])]
        state = np.zeros((problem.state_dim, 1))
        for robot, point in enumerate(([-2., -2.], [2., 2.], [-2., -2.], [2., 2.])):
            state[problem.state_idxs[robot], 0] = point
        state[problem.active_idxs, 0] = 1.
        action = np.zeros((problem.action_dim, 1))
        audit = replay_minco(problem, state, action, robots=(2, 3), sample_dt=.05)
        self.assertFalse(audit.collision)


if __name__ == '__main__':
    unittest.main()
