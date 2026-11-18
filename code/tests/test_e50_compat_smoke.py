import unittest
import numpy as np

from problems.example8 import Example8
from solvers.e50_compat_smoke import compare_execution, replay_direct, replay_endpoint_semantic


class E50CompatibilitySmokeTests(unittest.TestCase):
    def make_state(self, problem):
        state = np.zeros((problem.state_dim, 1))
        for robot, point in enumerate(([-2., 0.], [2., 0.], [-2., -2.], [2., -2.])):
            state[problem.state_idxs[robot], 0] = point
        state[problem.active_idxs, 0] = 1.
        return state

    def test_direct_and_minco_are_both_audited(self):
        p = Example8(); p.obstacles = []
        s = self.make_state(p); action = np.zeros((p.action_dim, 1))
        action[p.action_idxs[2], 0] = [1., 0.]
        result = compare_execution(p, s, action, robots=(2, 3), sample_dt=.05, lidar_rays=8)
        self.assertEqual(result.endpoint_delta.shape, (2, 2))
        self.assertFalse(result.direct.collision)
        self.assertFalse(result.minco.collision)
        self.assertEqual(result.lidar_ranges.shape, (2, 8))
        self.assertTrue(result.direct_lidar_safe.all())
        self.assertTrue(result.minco_lidar_safe.all())
        self.assertEqual(result.minco_error_to_feasible.shape, (2, 2))

    def test_direct_collision_is_visible_to_audit(self):
        p = Example8(); p.obstacles = [np.array([[-1., 1.], [-2.5, -1.5]])]; p.swept_collision = False
        s = self.make_state(p); action = np.zeros((p.action_dim, 1))
        action[p.action_idxs[2], 0] = [2., 0.]
        result = replay_direct(p, s, action, robots=(2, 3), sample_dt=.05)
        self.assertTrue(result.collision)

    def test_endpoint_semantic_executor_reaches_legacy_goal(self):
        p = Example8(); p.obstacles = []
        s = self.make_state(p); action = np.zeros((p.action_dim, 1))
        action[p.action_idxs[2], 0] = [.02, .01]
        audit, feasible = replay_endpoint_semantic(p, s, action, robots=(2, 3), sample_dt=.05)
        self.assertTrue(feasible)
        np.testing.assert_allclose(audit.positions[-1, 0], s[p.state_idxs[2], 0] + action[p.action_idxs[2], 0], atol=1e-8)


if __name__ == '__main__': unittest.main()
