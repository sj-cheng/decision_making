import unittest
import numpy as np

from solvers.e50_interfaces import LidarScan
from solvers.e50_local_scan_memory import LocalScanMemory


class E50LocalScanMemoryTests(unittest.TestCase):
    def test_memory_merges_hits_and_expires_old_scans(self):
        scan = LidarScan('base', 0., np.array([0., np.pi/2]), np.array([2., 3.]), 5.)
        memory = LocalScanMemory(resolution=.5, max_age=1.)
        memory.update([0., 0.], scan)
        self.assertEqual(len(memory.points), 2)
        memory.update([0., 0.], LidarScan('base', 2., scan.angles, scan.ranges, 5.))
        self.assertEqual(len(memory.points), 2)
        self.assertTrue(np.all(memory.stamps == 2.))

    def test_recent_hits_guard_trajectory_points(self):
        scan = LidarScan('base', 0., np.array([0.]), np.array([2.]), 5.)
        memory = LocalScanMemory(max_age=2.)
        memory.update([0., 0.], scan)
        self.assertFalse(memory.endpoint_safe([2., 0.], inflation=.1))
        self.assertTrue(memory.endpoint_safe([0., 2.], inflation=.1))
        self.assertFalse(memory.trajectory_safe([[0., 0.], [2., 0.]], inflation=.1))

    def test_grid_keeps_ray_direction_and_occupied_hit(self):
        scan = LidarScan('base', 0., np.array([0., np.pi / 2]),
                         np.array([2., 3.]), 5.)
        memory = LocalScanMemory(resolution=.5)
        memory.update([0., 0.], scan)
        self.assertFalse(memory.grid_trajectory_safe([[1.9, 0.]], inflation=.1))
        self.assertTrue(memory.grid_trajectory_safe([[0., 2.]], inflation=.1))


if __name__ == '__main__': unittest.main()
