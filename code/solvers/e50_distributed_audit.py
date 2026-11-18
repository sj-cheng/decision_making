"""Deterministic communication and sensing perturbation audit for E50.

The scheduler is deliberately independent of a controller.  It models the
execution boundary: each robot may receive a delayed or dropped command, and
the safe response to a missing command is a zero command for that robot.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DistributedAuditConfig:
    command_delay: int = 0
    pose_delay: int = 0
    drop_cycles: tuple = ()

    def __post_init__(self):
        if int(self.command_delay) < 0 or int(self.pose_delay) < 0:
            raise ValueError("delays must be non-negative")
        if any(int(c) < 0 for c in self.drop_cycles):
            raise ValueError("drop cycles must be non-negative")


@dataclass(frozen=True)
class DistributedAuditResult:
    commands: np.ndarray
    poses: np.ndarray
    events: tuple


def simulate_distributed_execution(commands, poses, config=DistributedAuditConfig()):
    """Replay joint commands with packet/pose delay and per-robot stop fallback.

    ``commands`` has shape ``(cycles, 2, 2)`` and represents the centrally
    selected command at each one-second decision. ``poses`` has shape
    ``(cycles, 2, 2)`` and is the pose available to the planner at each cycle.
    The returned command tensor is what each robot actually executes.
    """
    commands = np.asarray(commands, dtype=float)
    poses = np.asarray(poses, dtype=float)
    if commands.ndim != 3 or commands.shape[1:] != (2, 2):
        raise ValueError("commands must have shape (cycles,2,2)")
    if poses.shape != commands.shape:
        raise ValueError("poses must match commands shape")
    if not np.isfinite(commands).all() or not np.isfinite(poses).all():
        raise ValueError("commands and poses must be finite")
    delay = int(config.command_delay)
    pose_delay = int(config.pose_delay)
    drops = {int(c) for c in config.drop_cycles}
    actual = np.zeros_like(commands)
    events = []
    pending = [[] for _ in range(2)]
    for cycle in range(len(commands)):
        for robot in range(2):
            if cycle in drops:
                events.append(dict(cycle=cycle, robot=robot, event="packet_dropped"))
            else:
                pending[robot].append((cycle + delay, commands[cycle, robot].copy()))
            due = [item for item in pending[robot] if item[0] <= cycle]
            pending[robot] = [item for item in pending[robot] if item[0] > cycle]
            if due:
                actual[cycle, robot] = due[-1][1]
                events.append(dict(cycle=cycle, robot=robot, event="command_applied",
                                   source_cycle=int(due[-1][0] - delay)))
            else:
                actual[cycle, robot] = 0.
                events.append(dict(cycle=cycle, robot=robot, event="stop_degraded"))
        source = max(0, cycle - pose_delay)
        if source != cycle:
            events.append(dict(cycle=cycle, event="pose_delayed", source_cycle=source))
    delayed_poses = poses[[max(0, i - pose_delay) for i in range(len(poses))]]
    return DistributedAuditResult(actual, delayed_poses, tuple(events))
