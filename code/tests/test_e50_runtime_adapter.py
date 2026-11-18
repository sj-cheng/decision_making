import unittest
import numpy as np

from solvers.e50_interfaces import BeliefState, LidarScan
from solvers.e50_runtime_adapter import E50RuntimeAdapter


class _Time:
    def __init__(self, sec, nanosec=0): self.sec, self.nanosec = sec, nanosec


class _Header:
    def __init__(self, stamp, frame_id='base'): self.stamp, self.frame_id = stamp, frame_id


class _Point:
    def __init__(self, x, y): self.x, self.y = x, y


class _Pose:
    def __init__(self, x, y): self.position = _Point(x, y)


class _PoseArray:
    def __init__(self): self.header = _Header(_Time(1, 500000000)); self.poses = [_Pose(0, 0), _Pose(1, 1)]


class _Scan:
    header = _Header(_Time(1, 500000000), 'laser')
    angle_min = -1.; angle_increment = .5; ranges = [2., 3., 4.]; range_max = 10.


class _Belief:
    header = _Header(_Time(1, 500000000))
    particles = np.zeros((2, 2, 2)); weights = np.full((2, 2), .5)
    active = np.array([True, True]); confidence = np.array([.8, .7])


def belief(stamp):
    return BeliefState(np.zeros((2, 2, 2)), np.full((2, 2), .5),
                       np.array([False, False]), np.array([.2, .2]), stamp)


class E50RuntimeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = E50RuntimeAdapter()
        angles = np.linspace(-np.pi, np.pi, 8, endpoint=False)
        self.scan = LidarScan('base', 1., angles, np.full(8, 5.), 5.)
        self.adapter.update_poses(1., np.zeros((2, 2)))
        self.adapter.update_scan(0, self.scan); self.adapter.update_scan(1, self.scan)
        self.adapter.update_belief(belief(1.))

    def test_synchronized_snapshot_is_live(self):
        snapshot = self.adapter.snapshot(1.05)
        self.assertFalse(snapshot.degraded)
        np.testing.assert_allclose(snapshot.poses, 0.)

    def test_missing_or_stale_input_degrades(self):
        self.adapter.update_poses(0., np.zeros((2, 2)))
        snapshot = self.adapter.snapshot(1.)
        self.assertTrue(snapshot.degraded)
        self.assertEqual(snapshot.reason, 'stale_sensor')

    def test_timestamp_skew_degrades(self):
        self.adapter.update_belief(belief(1.3))
        snapshot = self.adapter.snapshot(1.)
        self.assertTrue(snapshot.degraded)
        self.assertEqual(snapshot.reason, 'timestamp_skew')

    def test_degraded_snapshot_gates_command_to_stop(self):
        snapshot = self.adapter.snapshot(1.5)
        command = np.ones((2, 2))
        np.testing.assert_allclose(self.adapter.gate_command(snapshot, command), 0.)
        live = self.adapter.snapshot(1.05)
        np.testing.assert_allclose(self.adapter.gate_command(live, command), command)

    def test_record_is_json_safe_and_marks_override(self):
        snapshot = self.adapter.snapshot(1.5); command = np.ones((2, 2))
        gated = self.adapter.gate_command(snapshot, command)
        record = self.adapter.record(snapshot, command, gated)
        self.assertTrue(record['command_overridden'])
        self.assertEqual(record['reason'], 'stale_sensor')

    def test_ros_message_adapters_feed_live_snapshot(self):
        adapter = E50RuntimeAdapter()
        adapter.update_pose_message(_PoseArray())
        adapter.update_scan_message(0, _Scan()); adapter.update_scan_message(1, _Scan())
        adapter.update_belief_message(_Belief())
        snapshot = adapter.snapshot(1.55)
        self.assertFalse(snapshot.degraded)
        np.testing.assert_allclose(snapshot.poses, [[0., 0.], [1., 1.]])
        self.assertEqual(snapshot.scans[0].frame_id, 'laser')


if __name__ == '__main__': unittest.main()
