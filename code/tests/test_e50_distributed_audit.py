import unittest
import numpy as np

from solvers.e50_distributed_audit import (DistributedAuditConfig,
                                           simulate_distributed_execution)


class E50DistributedAuditTests(unittest.TestCase):
    def test_delayed_commands_and_pose_are_explicit(self):
        commands = np.arange(12., dtype=float).reshape(3, 2, 2)
        poses = commands + 100.
        result = simulate_distributed_execution(
            commands, poses, DistributedAuditConfig(command_delay=1, pose_delay=1))
        np.testing.assert_allclose(result.commands[0], 0.)
        np.testing.assert_allclose(result.commands[1], commands[0])
        np.testing.assert_allclose(result.poses[1], poses[0])
        self.assertTrue(any(e['event'] == 'stop_degraded' for e in result.events))
        self.assertTrue(any(e['event'] == 'pose_delayed' for e in result.events))

    def test_packet_drop_stops_each_robot_without_reusing_stale_command(self):
        commands = np.ones((3, 2, 2))
        commands[1, 1] = 7.
        poses = np.zeros_like(commands)
        result = simulate_distributed_execution(
            commands, poses, DistributedAuditConfig(drop_cycles=(1,)))
        np.testing.assert_allclose(result.commands[0], commands[0])
        np.testing.assert_allclose(result.commands[1], 0.)
        np.testing.assert_allclose(result.commands[2], commands[2])
        self.assertEqual(sum(e['event'] == 'packet_dropped' for e in result.events), 2)


if __name__ == '__main__':
    unittest.main()
