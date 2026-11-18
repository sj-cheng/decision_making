import unittest
import numpy as np

from solvers.e50_lidar_map import raycast_scan, lidar_target_visibility
from solvers.e50_local_scan_memory import LocalScanMemory
from run_e50_rolling_smoke import _apply_evader_step, _mask_history_observation
from make_e50_legacy_gif import _path_at


class LidarPlanningBridgeTests(unittest.TestCase):
    def test_replayed_reports_follow_local_geometry(self):
        frame = dict(step=0, remaining=80, positions=[[0., 0.], [0., 1.]],
                     active_p=[True, True], active_e=[True, True],
                     reports=[[[2., 0.], None], [None, None]])
        boxes = np.array([[[1., 2.], [-.2, .2]]])
        masked = _mask_history_observation(frame, boxes)
        self.assertIsNone(masked['reports'][0][0])

    def test_hit_cells_export_as_legacy_geometry(self):
        memory = LocalScanMemory(resolution=.2, max_age=2.)
        scan = raycast_scan(np.array([0., 0.]), np.array([[[1., 2.], [1., 2.]]]),
                            num_rays=360, max_range=5.)
        memory.update(np.array([0., 0.]), scan)
        boxes = memory.obstacle_boxes(inflation=.05, max_boxes=24)
        self.assertGreater(len(boxes), 0)
        self.assertTrue(np.all(boxes[:, :, 0] <= boxes[:, :, 1]))
        # No synthetic obstacle is exported when every ray is a max-range miss.
        clear = LocalScanMemory(resolution=.2, max_age=2.)
        clear_scan = raycast_scan(np.array([0., 0.]), np.zeros((0, 2, 2)),
                                  num_rays=64, max_range=5.)
        clear.update(np.array([0., 0.]), clear_scan)
        self.assertEqual(clear.obstacle_boxes().shape, (0, 2, 2))

    def test_target_report_requires_a_clear_lidar_return(self):
        clear = lidar_target_visibility([[0., 0.]], [[3., 0.]],
                                        np.empty((0, 2, 2)), num_rays=256)
        self.assertTrue(bool(clear[0, 0]))
        wall = np.array([[[1., 2.], [-1., 1.]]])
        blocked = lidar_target_visibility([[0., 0.]], [[3., 0.]], wall,
                                          num_rays=256)
        self.assertFalse(bool(blocked[0, 0]))

    def test_target_report_uses_discrete_ray_width(self):
        # The vehicle footprint lets a 256-ray scan identify a target between
        # adjacent ray centrelines, while a static wall still occludes it.
        visible = lidar_target_visibility([[0., 0.]], [[3., .03]],
                                          np.empty((0, 2, 2)), num_rays=256)
        self.assertTrue(bool(visible[0, 0]))

    def test_gif_path_builder_keeps_evader_history(self):
        row = dict(cycles=[
            dict(executed_trajectory_positions=[[[0., 0.], [0., 1.]],
                                                  [[.2, 0.], [.2, 1.]]],
                 evader_position=[[1., 0.], [1., 1.]]),
            dict(executed_trajectory_positions=[[[.2, 0.], [.2, 1.]],
                                                  [[.4, 0.], [.4, 1.]]],
                 evader_position=[[2., 0.], [2., 1.]])])
        _, current, history = _path_at(row, np.array([[-1., 0.], [-1., 1.]]), 1, 1.)
        self.assertEqual(history[0].shape, (3, 2))
        np.testing.assert_allclose(current[0], [2., 0.])

    def test_evader_boundary_rejection_is_recorded_as_zero_applied_step(self):
        applied, legal = _apply_evader_step(np.array([9.9, 0.]), np.array([1., 0.]),
                                            np.empty((0, 2, 2)))
        self.assertFalse(legal)
        np.testing.assert_array_equal(applied, [0., 0.])


if __name__ == '__main__':
    unittest.main()
