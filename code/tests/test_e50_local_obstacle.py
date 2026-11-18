import unittest
import numpy as np

from solvers.e50_lidar_map import raycast_scan
from solvers.e50_local_obstacle import LocalObstacleProxy, project_action_to_local_scan


class E50LocalObstacleTests(unittest.TestCase):
    def test_proxy_rejects_motion_past_a_measured_return(self):
        scan = raycast_scan([0., 0.], [[[2., 3.], [-1., 1.]]], num_rays=64, max_range=8.)
        proxy = LocalObstacleProxy.from_scan([0., 0.], scan, inflation=.1)
        self.assertTrue(proxy.endpoint_safe([1., 0.]))
        self.assertFalse(proxy.endpoint_safe([2.5, 0.]))
        self.assertFalse(proxy.segment_safe([0., 0.], [2.5, 0.]))

    def test_neighbor_beam_limit_is_conservative(self):
        scan = raycast_scan([0., 0.], [[[1., 1.2], [-.1, .1]]], num_rays=16, max_range=6.)
        proxy = LocalObstacleProxy.from_scan([0., 0.], scan, inflation=.05)
        # A point just off the center beam remains rejected if either adjacent
        # beam reports the nearer hit.
        self.assertFalse(proxy.endpoint_safe([1.05, .08]))

    def test_interpolated_limit_preserves_beam_locality(self):
        scan = raycast_scan([0., 0.], [[[1., 1.2], [-.1, .1]]], num_rays=16, max_range=6.)
        proxy = LocalObstacleProxy.from_scan([0., 0.], scan, inflation=.05)
        self.assertFalse(proxy.endpoint_safe([1.05, 0.], interpolation=True))
        self.assertTrue(proxy.endpoint_safe([1.05, .8], interpolation=True))

    def test_action_projection_returns_local_safe_fallback(self):
        scan = raycast_scan([0., 0.], [[[1., 1.2], [-.1, .1]]], num_rays=32, max_range=6.)
        proxies = [LocalObstacleProxy.from_scan([0., 0.], scan, inflation=.05)] * 2
        action = project_action_to_local_scan(np.zeros((2, 2)), np.array([[1., 0.], [1., 0.]]), proxies)
        self.assertTrue(all(proxies[i].segment_safe(np.zeros(2), action[i], interpolation=True)
                            for i in range(2)))
        self.assertLessEqual(action[0, 0], .75)

    def test_action_projection_can_bypass_a_nearby_wall(self):
        scan = raycast_scan([0., 0.], [[[.1, .3], [-1., 1.]]], num_rays=128, max_range=6.)
        proxies = [LocalObstacleProxy.from_scan([0., 0.], scan, inflation=.05)] * 2
        action = project_action_to_local_scan(np.zeros((2, 2)),
                                              np.array([[1., 0.], [1., 0.]]), proxies)
        self.assertGreater(abs(action[0, 1]), 0.)
        self.assertTrue(proxies[0].segment_safe(np.zeros(2), action[0], interpolation=True))


if __name__ == '__main__':
    unittest.main()
