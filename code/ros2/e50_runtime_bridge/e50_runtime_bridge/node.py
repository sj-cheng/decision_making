"""ROS2 bridge; the controller hook receives only gated E50 snapshots.

The repository test environment intentionally has no ROS dependency.  Import
ROS inside ``main`` so this module remains inspectable and unit-testable there.
Belief topic encoding is Float64MultiArray:
``[stamp, n, x0, y0, weight0, active0, confidence0, ...]``; each row is one
target with one support particle at the bridge boundary.
"""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / 'code') not in sys.path:
    sys.path.insert(0, str(ROOT / 'code'))
from solvers.e50_runtime_adapter import E50RuntimeAdapter
from solvers.e50_interfaces import BeliefState


def decode_belief(message):
    values = np.asarray(message.data, dtype=float)
    if len(values) < 2:
        raise ValueError('belief message is too short')
    stamp, count = float(values[0]), int(values[1])
    width = 5
    if count < 0 or len(values) != 2 + count * width:
        raise ValueError('belief message has invalid count/width')
    rows = values[2:].reshape(count, width)
    particles = rows[:, :2]
    weights = rows[:, 2]
    active = rows[:, 3] > .5
    confidence = rows[:, 4]
    # The transport uses one support particle per target to stay independent
    # of a project-specific belief message definition.
    return BeliefState(particles.reshape(count, 1, 2), np.ones((count, 1)),
                       active.reshape(count), confidence.reshape(count), stamp)


def decode_command(message):
    values = np.asarray(message.data, dtype=float).reshape(-1)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError('command message must contain two finite values')
    return values.copy()


def main():
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseArray, Twist
        from sensor_msgs.msg import LaserScan
        from std_msgs.msg import Float64MultiArray
    except ImportError as exc:
        raise RuntimeError('ROS2 dependencies are required on the robot image') from exc

    class Bridge(Node):
        def __init__(self):
            super().__init__('e50_runtime_bridge')
            self.adapter = E50RuntimeAdapter()
            self.latest_command = np.zeros((2, 2), dtype=float)
            self.cmd = [self.create_publisher(Twist, '/e50/cmd0', 10),
                        self.create_publisher(Twist, '/e50/cmd1', 10)]
            self.create_subscription(PoseArray, '/e50/poses', self.pose_cb, 10)
            self.create_subscription(LaserScan, '/e50/lidar0', lambda m: self.scan_cb(0, m), 10)
            self.create_subscription(LaserScan, '/e50/lidar1', lambda m: self.scan_cb(1, m), 10)
            self.create_subscription(Float64MultiArray, '/e50/belief', self.belief_cb, 10)
            self.create_subscription(Float64MultiArray, '/e50/requested_cmd0', lambda m: self.command_cb(0, m), 10)
            self.create_subscription(Float64MultiArray, '/e50/requested_cmd1', lambda m: self.command_cb(1, m), 10)
            self.create_timer(.02, self.tick)

        def pose_cb(self, message): self.adapter.update_pose_message(message)
        def scan_cb(self, robot, message): self.adapter.update_scan_message(robot, message)
        def belief_cb(self, message): self.adapter.update_belief(decode_belief(message))
        def command_cb(self, robot, message): self.latest_command[int(robot)] = decode_command(message)
        def tick(self):
            now = self.get_clock().now().nanoseconds * 1e-9
            snap = self.adapter.snapshot(now)
            gated = self.adapter.gate_command(snap, self.latest_command)
            for i, publisher in enumerate(self.cmd):
                msg = Twist(); msg.linear.x = float(gated[i, 0]); msg.linear.y = float(gated[i, 1]); publisher.publish(msg)

    rclpy.init(); rclpy.spin(Bridge()); rclpy.shutdown()
