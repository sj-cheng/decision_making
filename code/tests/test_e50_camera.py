import math
import unittest
import numpy as np

from solvers.e50_camera import CameraGimbalTracker, segment_clear, wrap_angle


class CameraModelTests(unittest.TestCase):
    def test_angle_wrap(self):
        self.assertAlmostEqual(wrap_angle(3*math.pi), -math.pi)

    def test_line_of_sight(self):
        self.assertTrue(segment_clear([0, 0], [3, 0], []))
        self.assertFalse(segment_clear([0, 0], [3, 0], [[[1, -1], [2, 1]]]))

    def test_detection_returns_range_bearing_estimate(self):
        tracker = CameraGimbalTracker(1, sweep_rate_deg_s=90.)
        frame = tracker.step([[0., 0.]], [[3., 0.]], [True], [True], [0.], [], dt=.1)
        self.assertTrue(frame.visible[0, 0])
        self.assertEqual(frame.state[0], 'TRACK')
        self.assertAlmostEqual(frame.measurements[0][0]['range'], 3., places=6)
        self.assertAlmostEqual(frame.measurements[0][0]['bearing'], 0., places=6)
        np.testing.assert_allclose(frame.reports[0][0], [3., 0.])

    def test_occlusion_loses_lock(self):
        tracker = CameraGimbalTracker(1, sweep_rate_deg_s=960.)
        first = tracker.step([[0., 0.]], [[3., 0.]], [True], [True], [0.], [], dt=.1)
        self.assertTrue(first.visible[0, 0])
        lost = tracker.step([[0., 0.]], [[3., 0.]], [True], [True], [0.],
                             [[[1., -1.], [2., 1.]]], dt=.1)
        self.assertFalse(lost.visible[0, 0])
        self.assertEqual(lost.lost_reason[0], 'occlusion')
        self.assertEqual(lost.state[0], 'SEARCH')

    def test_outside_mechanical_envelope_is_lost(self):
        tracker = CameraGimbalTracker(1, sweep_rate_deg_s=960.)
        tracker.locked[0] = 0
        tracker.state[0] = 'TRACK'
        frame = tracker.step([[0., 0.]], [[-3., 0.]], [True], [True], [0.], [], dt=.1)
        self.assertFalse(frame.visible[0, 0])
        self.assertEqual(frame.lost_reason[0], 'out_of_fov')

    def test_sweep_reaches_plus_120_envelope_but_not_plus_121(self):
        tracker = CameraGimbalTracker(1, sweep_rate_deg_s=90.)
        # First step sweeps the head to its positive mechanical limit.
        target = [math.cos(math.radians(119.)), math.sin(math.radians(119.))]
        tracker.step([[0., 0.]], [target], [True], [True], [0.], [], dt=1.)
        visible = tracker.step([[0., 0.]], [target], [True], [True], [0.], [], dt=.01)
        self.assertTrue(visible.visible[0, 0])
        self.assertAlmostEqual(abs(visible.gimbal_yaw[0]), math.radians(90.), places=3)


if __name__ == '__main__':
    unittest.main()
