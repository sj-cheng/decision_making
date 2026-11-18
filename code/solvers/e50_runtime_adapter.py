"""Runtime-neutral SLAM/LiDAR/belief boundary for E50 hardware integration."""
from dataclasses import dataclass
import numpy as np

from solvers.e50_interfaces import BeliefState, LidarScan


def _ros_stamp(value):
    """Convert ROS1/ROS2 time-like values or numeric stamps to seconds."""
    if hasattr(value, 'to_sec'):
        return float(value.to_sec())
    if hasattr(value, 'sec'):
        return float(value.sec) + float(getattr(value, 'nanosec', getattr(value, 'nsec', 0))) * 1e-9
    return float(value)


def ros_pose_array_to_poses(message):
    poses = getattr(message, 'poses', message)
    rows = []
    for item in poses:
        position = getattr(item, 'position', item)
        rows.append([float(position.x), float(position.y)])
    result = np.asarray(rows, dtype=float)
    if result.shape != (2, 2) or not np.isfinite(result).all():
        raise ValueError('ROS pose array must contain two finite 2-D poses')
    return _ros_stamp(getattr(getattr(message, 'header', None), 'stamp', 0.)), result


def ros_laserscan_to_lidar(message, frame_id='base_link'):
    header = getattr(message, 'header', None)
    stamp = _ros_stamp(getattr(header, 'stamp', getattr(message, 'stamp', 0.)))
    ranges = np.asarray(message.ranges, dtype=float)
    angles = float(message.angle_min) + np.arange(len(ranges)) * float(message.angle_increment)
    max_range = float(getattr(message, 'range_max', np.nanmax(ranges)))
    return LidarScan(str(getattr(header, 'frame_id', frame_id)), stamp,
                     angles, ranges, max_range)


def ros_belief_to_state(message):
    stamp = _ros_stamp(getattr(getattr(message, 'header', None), 'stamp',
                               getattr(message, 'stamp', 0.)))
    particles = np.asarray(getattr(message, 'particles'), dtype=float)
    weights = np.asarray(getattr(message, 'weights'), dtype=float)
    active = np.asarray(getattr(message, 'active'), dtype=bool)
    confidence = np.asarray(getattr(message, 'confidence'), dtype=float)
    return BeliefState(particles, weights, active, confidence, stamp)


@dataclass(frozen=True)
class AgentSnapshot:
    """One agent's deployment input; peer loss is not a team-wide failure."""
    agent_id: int
    stamp: float
    pose: np.ndarray
    scan: LidarScan
    belief: BeliefState
    degraded: bool = False
    reason: str = ''

    def __post_init__(self):
        if int(self.agent_id) < 0 or not np.isfinite(float(self.stamp)):
            raise ValueError('invalid agent snapshot identity')
        pose = np.asarray(self.pose, dtype=float)
        if pose.shape != (2,) or not np.isfinite(pose).all():
            raise ValueError('agent pose must have shape (2,)')
        object.__setattr__(self, 'pose', pose.copy())


@dataclass(frozen=True)
class RuntimeSnapshot:
    stamp: float
    poses: np.ndarray
    scans: tuple
    belief: BeliefState
    degraded: bool
    reason: str = ''


class E50RuntimeAdapter:
    def __init__(self, *, max_pose_age=.15, max_scan_age=.15, max_skew=.10):
        if min(max_pose_age, max_scan_age, max_skew) <= 0:
            raise ValueError('runtime ages must be positive')
        self.max_pose_age = float(max_pose_age); self.max_scan_age = float(max_scan_age)
        self.max_skew = float(max_skew)
        self._poses = None; self._pose_stamp = None
        self._scans = [None, None]; self._belief = None

    def update_poses(self, stamp, poses):
        poses = np.asarray(poses, dtype=float)
        if poses.shape != (2, 2) or not np.isfinite(poses).all():
            raise ValueError('poses must have shape (2,2) and be finite')
        self._poses = poses.copy(); self._pose_stamp = float(stamp)

    def update_pose_message(self, message):
        stamp, poses = ros_pose_array_to_poses(message)
        self.update_poses(stamp, poses)

    def update_scan(self, robot, scan):
        if int(robot) not in (0, 1) or not isinstance(scan, LidarScan):
            raise ValueError('robot must be 0/1 and scan must be LidarScan')
        self._scans[int(robot)] = scan

    def update_scan_message(self, robot, message):
        self.update_scan(robot, ros_laserscan_to_lidar(message))

    def update_belief(self, belief):
        if not isinstance(belief, BeliefState):
            raise TypeError('belief must be BeliefState')
        self._belief = belief

    def update_belief_message(self, message):
        self.update_belief(ros_belief_to_state(message))

    def snapshot(self, now):
        now = float(now)
        if self._poses is None or self._belief is None or any(s is None for s in self._scans):
            return RuntimeSnapshot(now, np.zeros((2, 2)), tuple(self._scans), self._belief,
                                   True, 'missing_input')
        stamps = [self._pose_stamp, *(s.stamp for s in self._scans), self._belief.stamp]
        if any(now - stamp > limit for stamp, limit in
               ((self._pose_stamp, self.max_pose_age),
                *[(s.stamp, self.max_scan_age) for s in self._scans])):
            return RuntimeSnapshot(now, self._poses.copy(), tuple(self._scans), self._belief,
                                   True, 'stale_sensor')
        if max(stamps) - min(stamps) > self.max_skew:
            return RuntimeSnapshot(now, self._poses.copy(), tuple(self._scans), self._belief,
                                   True, 'timestamp_skew')
        return RuntimeSnapshot(now, self._poses.copy(), tuple(self._scans), self._belief, False, '')

    @staticmethod
    def gate_command(snapshot, command):
        """Apply the hardware safety gate: degraded input means stop both cars."""
        command = np.asarray(command, dtype=float)
        if command.shape != (2, 2) or not np.isfinite(command).all():
            raise ValueError('command must have shape (2,2) and be finite')
        return np.zeros((2, 2), dtype=float) if snapshot.degraded else command.copy()

    @staticmethod
    def gate_agent_command(degraded, command):
        """Gate one command without stopping a healthy teammate."""
        command = np.asarray(command, dtype=float)
        if command.shape != (2,) or not np.isfinite(command).all():
            raise ValueError('agent command must have shape (2,) and be finite')
        return np.zeros(2, dtype=float) if bool(degraded) else command.copy()

    @staticmethod
    def record(snapshot, requested, gated):
        """Return a JSON-safe audit record for hardware replay logs."""
        requested = np.asarray(requested, dtype=float); gated = np.asarray(gated, dtype=float)
        if requested.shape != (2, 2) or gated.shape != (2, 2):
            raise ValueError('commands must have shape (2,2)')
        return dict(stamp=float(snapshot.stamp), degraded=bool(snapshot.degraded),
                    reason=str(snapshot.reason), requested_command=requested.tolist(),
                    gated_command=gated.tolist(),
                    command_overridden=bool(np.max(np.abs(requested - gated)) > 1e-12))
