import unittest
import numpy as np

from solvers.e50_geometry_plans import (belief_plan_candidates,
                                        legacy_actions_to_plans,
                                        local_lidar_plan_candidates)
from solvers.e50_interfaces import LidarScan
from solvers.e50_local_obstacle import LocalObstacleProxy


class E50GeometryPlanTests(unittest.TestCase):
    def test_legacy_actions_are_wrapped_without_role_assignment(self):
        starts = np.zeros((2, 2))
        actions = np.array([[[1., 0.], [0., 1.]], [[0., 0.], [0., 0.]]])
        plans = legacy_actions_to_plans(actions, starts)
        self.assertEqual(len(plans), 2)
        np.testing.assert_allclose(plans[0].waypoints[:, 0], actions[0])
        self.assertEqual(plans[0].target_hint, (-1, -1))

    def test_belief_candidates_are_joint_and_bounded(self):
        starts = np.array([[0., 0.], [1., 1.]])
        particles = np.array([
            [[2., 0.], [3., 0.], [4., 0.]],
            [[-2., 0.], [-3., 0.], [-4., 0.]],
        ])
        weights = np.array([[.6, .3, .1], [.5, .3, .2]])
        plans = belief_plan_candidates(starts, particles, weights, max_candidates=8)
        self.assertGreater(len(plans), 0)
        self.assertLessEqual(len(plans), 8)
        self.assertTrue(all(p.waypoints.shape == (2, 2, 2) for p in plans))
        self.assertTrue(all(p.mode == ("pursue", "pursue") for p in plans))

    def test_invalid_candidates_can_be_filtered(self):
        starts = np.zeros((2, 2))
        particles = np.array([[[1., 0.]], [[2., 0.]]])
        weights = np.ones((2, 1))
        plans = belief_plan_candidates(starts, particles, weights, is_valid=lambda p: False)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].mode, ("wait", "wait"))

    def test_local_lidar_candidates_reject_unknown_or_blocked_paths(self):
        starts = np.zeros((2, 2))
        particles = np.array([[[1., 0.]], [[1., 0.]]])
        weights = np.ones((2, 1))
        angles = np.linspace(-np.pi, np.pi, 64, endpoint=False)
        ranges = np.full(64, 3.)
        ranges[np.argmin(np.abs(angles))] = .5
        scan = LidarScan('base', 0., angles, ranges, 3.)
        proxies = [LocalObstacleProxy.from_scan(starts[i], scan, inflation=.05)
                   for i in range(2)]
        plans = local_lidar_plan_candidates(starts, particles, weights, proxies)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].mode, ('wait', 'wait'))


if __name__ == '__main__':
    unittest.main()
