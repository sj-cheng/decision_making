import unittest
import numpy as np

from solvers.e50_distributed_geometry import single_agent_endpoints, evaluate_minco_endpoint
from solvers.e50_interfaces import LidarScan
from solvers.e50_local_obstacle import LocalObstacleProxy


class DistributedGeometryTests(unittest.TestCase):
    def test_endpoints_are_single_agent_and_local_scan_checked(self):
        angles = np.linspace(-np.pi, np.pi, 128, endpoint=False)
        ranges = np.full(128, 5.)
        ranges[np.argmin(np.abs(angles))] = .6
        proxy = LocalObstacleProxy.from_scan([0., 0.], LidarScan('base', 0., angles, ranges, 5.), .05)
        points = np.array([[1., 0.], [0., 1.], [-1., 0.]])
        endpoints = single_agent_endpoints([0., 0.], points, [1., 0., 0.], proxy,
                                           max_distance=1., max_candidates=8)
        self.assertTrue(all(np.asarray(p).shape == (2,) for p in endpoints))
        self.assertTrue(all(proxy.segment_safe([0., 0.], p, interpolation=True) for p in endpoints))

    def test_minco_candidate_reports_dynamics_and_track(self):
        result = evaluate_minco_endpoint([0., 0.], [0.5, 0.], duration=2., sample_dt=.02,
                                         max_speed=2., max_acceleration=10.)
        self.assertTrue(result['feasible'])
        self.assertEqual(result['positions'].shape[1], 2)
        self.assertGreater(result['max_speed'], 0.)


if __name__ == '__main__':
    unittest.main()
