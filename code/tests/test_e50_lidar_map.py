import unittest
import numpy as np

from solvers.e50_lidar_map import (raycast_lidar, raycast_scan, scan_to_snapshot,
                                    unknown_fraction, scan_to_obstacle_boxes,
                                    points_to_obstacle_boxes)


class E50LidarMapTests(unittest.TestCase):
    def test_raycast_returns_nearest_obstacle_distance(self):
        angles, distances = raycast_lidar([0., 0.], [[[2., 3.], [-1., 1.]]], num_rays=4, max_range=10.)
        self.assertEqual(len(angles), 4)
        self.assertAlmostEqual(distances[0], 2.)
        self.assertAlmostEqual(distances[2], 10.)

    def test_typed_scan_is_unpackable_and_timestamped(self):
        scan = raycast_scan([0., 0.], [], num_rays=8, stamp=3.5)
        angles, ranges = scan
        self.assertEqual(scan.frame_id, 'base_link')
        self.assertEqual(scan.stamp, 3.5)
        self.assertEqual(angles.shape, ranges.shape)

    def test_scan_marks_free_ray_and_occupied_hit(self):
        m = scan_to_snapshot([[2., 0.]], [0., 0.], resolution=1., shape=(5, 5), origin=(-2., -2.), max_range=5.)
        self.assertFalse(m.unknown[2, 2])
        self.assertFalse(m.unknown[2, 3])
        self.assertEqual(m.occupancy[2, 4], 1.)
        self.assertEqual(m.occupancy[2, 2], 0.)
        self.assertGreater(unknown_fraction(m), 0.)

    def test_out_of_range_points_do_not_create_false_obstacles(self):
        m = scan_to_snapshot([[10., 0.], [np.nan, 1.]], [0., 0.], resolution=1., shape=(5, 5), origin=(-2., -2.), max_range=2.)
        self.assertTrue(np.all(m.unknown))
        self.assertTrue(np.all(m.occupancy == 0.))

    def test_scan_hits_become_local_planner_patches(self):
        scan = raycast_scan([0., 0.], [[[2., 3.], [-1., 1.]]],
                            num_rays=128, max_range=10.)
        boxes = scan_to_obstacle_boxes([0., 0.], scan, inflation=.1)
        self.assertGreater(len(boxes), 0)
        self.assertTrue(np.all(boxes[:, :, 0] <= boxes[:, :, 1]))
        # The front wall is represented near x=2, not as the full global map.
        self.assertTrue(np.any((boxes[:, 0, 0] < 2.1) & (boxes[:, 0, 1] > 1.8)))

    def test_max_range_scan_does_not_invent_obstacles(self):
        scan = raycast_scan([0., 0.], [], num_rays=32, max_range=10.)
        self.assertEqual(scan_to_obstacle_boxes([0., 0.], scan).shape, (0, 2, 2))

    def test_accumulated_hits_are_capped_for_e49_map_tokens(self):
        points = np.array([[float(i), 0.] for i in range(40)])
        boxes = points_to_obstacle_boxes(points, max_boxes=5, merge_distance=.01)
        self.assertLessEqual(len(boxes), 5)


if __name__ == '__main__':
    unittest.main()

# Kept separate from the class above so the rolling-map contract is explicit.
class E50LidarPlanningMapTests(unittest.TestCase):
    def test_rolling_map_expires_old_hits_and_exposes_patches(self):
        from solvers.e50_lidar_map import LidarPlanningMap
        m = LidarPlanningMap(max_age=1., max_points=32, inflation=.1)
        scan = raycast_scan([0., 0.], [[[2., 3.], [-1., 1.]]], num_rays=64,
                            max_range=10., stamp=0.)
        m.update([0., 0.], scan)
        self.assertGreater(len(m.points), 0)
        self.assertGreater(len(m.obstacle_boxes()), 0)
        empty = raycast_scan([0., 0.], [], num_rays=64, max_range=10., stamp=2.)
        m.update([0., 0.], empty)
        self.assertEqual(len(m.points), 0)

    def test_recent_free_rays_provide_frontier_endpoints(self):
        from solvers.e50_lidar_map import LidarPlanningMap
        m = LidarPlanningMap(max_age=1., max_points=32, inflation=.1)
        scan = raycast_scan([0., 0.], [[[4., 5.], [-1., 1.]]],
                            num_rays=64, max_range=10., stamp=0.)
        m.update([0., 0.], scan)
        endpoints = m.free_space_endpoints(max_distance=2., max_candidates=4)
        self.assertGreater(len(endpoints), 0)
        self.assertTrue(all(np.linalg.norm(np.asarray(x)) <= 2.01 for x in endpoints))
