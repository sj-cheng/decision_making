"""Privileged sparse-reward H-MCTS evader for E53.

The evader searches over high-level escape options and legal MINCO endpoint
actions.  Search state contains the exact two-team positions because this is a
strong-opponent baseline, while the executor still checks the full sampled
trajectory before it is used.
"""
from dataclasses import dataclass, field
from types import SimpleNamespace
from copy import deepcopy
from collections import Counter
import math
import time

import numpy as np

from solvers.e50_distributed_geometry import evaluate_minco_endpoint, reachable_waypoint
from solvers.local_pursuit import clear, move, safe
from solvers.e50_camera import CameraGimbalTracker
from solvers.e53_capture import capture_samples


OPTIONS = ("escape", "open_space", "cover", "split", "recover")


@dataclass
class _Node:
    evaders: np.ndarray
    pursuers: np.ndarray
    active_e: np.ndarray
    active_p: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    depth: int = 0
    parent: object = None
    action: np.ndarray = None
    options: tuple = ()
    reward: float = 0.0
    prior: float = 1.0
    visits: int = 0
    total: float = 0.0
    children: list = field(default_factory=list)
    pending: list | None = None
    cameras: object = None
    headings: object = None
    visibility: object = None


class EvaderHMCTS:
    """Full-state H-MCTS with sparse window and terminal rewards."""

    TIME_BUDGET_MS = None

    def __init__(self, *, evader_speed=1.5, max_acceleration=1.5,
                 depth=3, simulations=256, window=8, duration=2.0,
                 seed=17, cruise=False):
        self.evader_speed = float(evader_speed)
        self.max_acceleration = float(max_acceleration)
        self.depth = int(depth)
        self.simulations = int(simulations)
        self.window = int(window)
        if min(self.window, self.depth, self.simulations) < 1:
            raise ValueError("positive depth, window and simulation count required")
        self.camera_state = [CameraGimbalTracker(1) for _ in range(2)]
        self.headings = np.zeros(2)
        self.visibility = np.zeros((2,2), bool)
        self.duration = float(duration)
        self.cruise = cruise
        self.seed = int(seed)
        self.public = None
        self.records = []
        self.last_search = {}
        self.previous_action = np.zeros((2, 2), dtype=float)
        self.previous_velocity = np.zeros((2, 2), dtype=float)
        self.previous_acceleration = np.zeros((2, 2), dtype=float)
        self._rng = np.random.default_rng(self.seed)

    def set_execution_state(self, action=None, velocity=None, acceleration=None):
        if action is not None:
            self.previous_action = np.asarray(action, dtype=float).reshape(2, 2).copy()
        if velocity is not None:
            self.previous_velocity = np.asarray(velocity, dtype=float).reshape(2, 2).copy()
        if acceleration is not None:
            self.previous_acceleration = np.asarray(acceleration, dtype=float).reshape(2, 2).copy()

    def _init_map(self, problem):
        if self.public is not None:
            return
        self.public = SimpleNamespace(boxes=np.asarray(problem.obstacles, dtype=float).reshape(-1, 2, 2))

    def set_camera_state(self, trackers, headings, visibility):
        self.camera_state = trackers
        self.headings = np.asarray(headings, float).copy()
        self.visibility = np.asarray(visibility, bool).copy()

    def prepare(self, problem, position):
        """Load JIT kernels before the first timed decision."""
        self._init_map(problem)
        point = np.asarray(position, dtype=float).copy()
        count = np.zeros(1, np.int64)
        boxes = self.public.boxes
        safe(point, np.zeros(2), self.evader_speed, boxes, count)
        clear(point, point, boxes, count)
        move(point, point, 1.0, np.zeros((0, 2)), np.zeros((0, 0), np.int64), boxes, count)

    def cruise_track(self, position, velocity, acceleration, direction):
        duration = self.duration
        delta = self.evader_speed * direction - velocity
        change = np.linalg.norm(delta)
        target = velocity + delta * min(1., self.max_acceleration * duration / (1.6 * max(change, 1e-9)))
        endpoint = position + duration * (velocity + target) / 2.
        return evaluate_minco_endpoint(position, endpoint, duration=duration,
            start_velocity=velocity, start_acceleration=acceleration, terminal_velocity=target,
            max_speed=self.evader_speed, max_acceleration=self.max_acceleration,
            continuous_speed=True, continuous_acceleration=True)

    def _action_candidates(self, node):
        """Build option-labelled candidate actions and filter with MINCO."""
        boxes = self.public.boxes
        evaders, pursuers = node.evaders, node.pursuers
        active_e, active_p = node.active_e, node.active_p
        result = []
        for e in range(2):
            if not active_e[e]:
                result.append([(np.zeros(2), "recover", None)])
                continue
            q = evaders[e]
            nearest = int(np.argmin([np.linalg.norm(q - p) if active_p[i] else 1e6
                                      for i, p in enumerate(pursuers)]))
            away = q - pursuers[nearest] if active_p.any() else np.array([1., 0.])
            norm = max(1e-9, float(np.linalg.norm(away)))
            away /= norm
            other = evaders[1 - e] - q
            onorm = max(1e-9, float(np.linalg.norm(other)))
            split = -other / onorm
            speed = np.linalg.norm(node.velocity[e])
            forward = node.velocity[e] / speed if speed > .1 else away
            lateral = np.array([-forward[1], forward[0]])
            directions = {
                "escape": [away + forward, away, np.array([-away[1], away[0]]), np.array([away[1], -away[0]])],
                "open_space": [forward, forward + lateral, forward - lateral],
                "cover": [np.array([-away[1], away[0]]), np.array([away[1], -away[0]])],
                "split": [split, np.array([-split[1], split[0]]), np.array([split[1], -split[0]])],
                "recover": [np.zeros(2)],
            }
            entries = []
            for option in OPTIONS:
                for direction in directions[option]:
                    length = float(np.linalg.norm(direction))
                    displacement = np.zeros(2) if length < 1e-9 else direction / length * min(2., self.evader_speed * self.duration)
                    endpoint = reachable_waypoint(q, q + displacement, node.velocity[e],
                                                  self.duration, self.evader_speed, self.max_acceleration)
                    displacement = endpoint - q
                    if not safe(q, displacement, self.evader_speed * self.duration, boxes, np.zeros(1, np.int64)):
                        continue
                    if any(np.linalg.norm(endpoint - p) < 1.0 for i, p in enumerate(pursuers) if active_p[i]):
                        continue
                    track = evaluate_minco_endpoint(
                        q, endpoint, duration=self.duration, start_velocity=node.velocity[e],
                        start_acceleration=node.acceleration[e], max_speed=self.evader_speed,
                        max_acceleration=self.max_acceleration, proxy=None,
                        continuous_speed=True, continuous_acceleration=True, pass_through=True)
                    if not track["feasible"] or not self._path_safe(track["positions"]):
                        continue
                    entries.append((displacement, option, track))
            if self.cruise:
                for direction in (forward, away):
                    track = self.cruise_track(q, node.velocity[e], node.acceleration[e], direction)
                    endpoint = track['positions'][-1]
                    if (track['feasible'] and self._path_safe(track['positions']) and
                            all(np.linalg.norm(endpoint-p) >= 1. for i,p in enumerate(pursuers) if active_p[i])):
                        entries.append((endpoint-q, 'cruise', track))
            if not entries:
                entries = [(np.zeros(2), "recover", None)]
            result.append(entries[:16])
        return result

    def _path_safe(self, points):
        points = np.asarray(points, dtype=float)
        if np.any(np.abs(points) > 9.0):
            return False
        if len(self.public.boxes) == 0:
            return True
        return all(bool(clear(start, end, self.public.boxes, np.zeros(1, np.int64)))
                   for start, end in zip(points[:-1], points[1:]))

    def _transition(self, node, tracks):
        ev = node.evaders.copy()
        pu = node.pursuers.copy()
        velocity = np.zeros_like(ev)
        acceleration = np.zeros_like(ev)
        for e, track in enumerate(tracks):
            if track is not None:
                # Only the first second of the two-second polynomial is executed.
                mid = (len(track['positions']) - 1) // 2
                ev[e] = track['positions'][mid]
                velocity[e] = track['velocities'][mid]
                acceleration[e] = track['accelerations'][mid]
        for p in range(2):
            if node.active_p[p]:
                target = ev[int(np.argmin(np.where(node.active_e, np.linalg.norm(ev - pu[p], axis=1), np.inf)))]
                pu[p] += move(pu[p], target, 1.0, np.zeros((0, 2)),
                               np.zeros((0, 0), np.int64), self.public.boxes, np.zeros(1, np.int64))
        grid = np.linspace(0., 1., 51)[:, None, None]
        pp = node.pursuers + grid * (pu - node.pursuers)
        ee = np.repeat(node.evaders[None], 51, axis=0)
        for e, track in enumerate(tracks):
            if track is not None:
                ee[:, e] = track['positions'][:51]
        hits, owners = capture_samples(pp, ee, node.active_p, node.active_e, self.public.boxes)
        active_e = node.active_e & (hits < 0)
        events = [(owners[e], e) for e in np.flatnonzero(hits >= 0)]
        return ev, pu, active_e, events, velocity, acceleration

    def _advance_camera(self, parent, child):
        child.cameras = deepcopy(parent.cameras)
        child.headings = parent.headings.copy()
        child.visibility = np.zeros((2, 2), bool)
        for p, camera in enumerate(child.cameras):
            delta = child.pursuers[p] - parent.pursuers[p]
            if np.linalg.norm(delta) > 1e-6:
                child.headings[p] = math.atan2(delta[1], delta[0])
            frame = camera.step(child.pursuers[p:p+1], child.evaders, child.active_p[p:p+1],
                                child.active_e, child.headings[p:p+1], self.public.boxes)
            child.visibility[p] = frame.visible[0]

    def _reward(self, child, events):
        if events:
            self.events['capture'] += len(events)
            return -100. * len(events)
        if child.depth != self.depth and child.depth % self.window:
            return 0.
        start = child.parent
        while start.depth and start.depth % self.window:
            start = start.parent
        self.events['survival'] += 1
        reward = 2.
        if np.any(start.visibility & ~child.visibility & child.active_e):
            self.events['lost_visibility'] += 1
            reward += 8.
        if np.linalg.norm(start.pursuers[0]-start.pursuers[1]) <= 4. < np.linalg.norm(child.pursuers[0]-child.pursuers[1]):
            self.events['separation'] += 1
            reward += 6.
        return reward

    def _candidates(self, node):
        candidates = self._action_candidates(node)
        pairs = []
        for first in candidates[0]:
            for second in candidates[1]:
                action = np.vstack((first[0], second[0]))
                pairs.append((action, (first[1], second[1]), first[2], second[2]))
        # Sample distinct joint actions without privileging their enumeration order.
        seen = set()
        unique = []
        for row in pairs:
            key = tuple(np.round(row[0].ravel(), 5))
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
        self._rng.shuffle(unique)
        return unique[:64]

    def _child(self, node):
        action, options, first, second = node.pending.pop()
        ev, pu, active, events, vel, acc = self._transition(node, (first, second))
        child = _Node(ev, pu, active, node.active_p, vel, acc, node.depth+1,
                      node, action, options)
        if self.cruise and node.depth == 0:
            child.cruise_tracks = tuple(track if opt == "cruise" else None for track,opt in zip((first,second),options))
        self._advance_camera(node, child)
        child.reward = self._reward(child, events)
        node.children.append(child)
        return child

    @staticmethod
    def _backup(path):
        value = 0.
        for item in reversed(path[1:]):
            value = item.reward + .98 * value
            item.visits += 1
            item.total += value
        path[0].visits += 1
        path[0].total += value

    def _select(self, node):
        if not node.children:
            return None
        log_n = math.log(max(1, node.visits))
        return max(node.children, key=lambda child: child.total / max(1, child.visits) +
                   1.4 * child.prior * math.sqrt(log_n / max(1, child.visits)))

    def search(self, problem, state, turn=0):
        started = time.perf_counter()
        self.events = Counter()
        self._init_map(problem)
        state = np.asarray(state, dtype=float).reshape(-1, 1)
        evaders = state[:4, 0].reshape(2, 2).copy()
        pursuers = state[4:8, 0].reshape(2, 2).copy()
        active_e = np.array([problem.is_active(state, robot) for robot in problem.evaders], dtype=bool)
        active_p = np.array([problem.is_active(state, robot) for robot in problem.pursuers], dtype=bool)
        root = _Node(evaders, pursuers, active_e, active_p, self.previous_velocity.copy(), self.previous_acceleration.copy())
        root.cameras, root.headings, root.visibility = self.camera_state, self.headings, self.visibility
        deepest = 0
        for _ in range(self.simulations):
            node = root
            path = [node]
            while node.depth < self.depth and node.active_e.any():
                if node.pending is None:
                    node.pending = self._candidates(node)
                # Progressively admit shuffled pairs so a wide root still gets lookahead.
                if node.pending and len(node.children) < math.ceil(math.sqrt(node.visits+1)):
                    node = self._child(node)
                elif node.children:
                    node = self._select(node)
                else:
                    break
                path.append(node)
            deepest = max(deepest, node.depth)
            self._backup(path)
        simulations = root.visits
        best = max(root.children, key=lambda child: (child.visits, child.total), default=None)
        action = np.zeros((2, 2)) if best is None else best.action.copy()
        options = ("recover", "recover") if best is None else best.options
        elapsed = 1000.0 * (time.perf_counter() - started)
        window = dict(options=list(options), survived=bool(active_e.any()),
                      lost_visibility=bool(best is not None and
                          np.any(root.visibility & ~best.visibility & best.active_e)),
                      max_depth=deepest, reward_events=dict(self.events),
                      pursuer_separation=float(np.linalg.norm(pursuers[0] - pursuers[1])),
                      search_ms=elapsed, feasible=best is not None,
                      simulations=simulations, time_budget_ms=self.TIME_BUDGET_MS,
                      simulation_cap=self.simulations, depth=int(self.depth),
                      window_cycles=int(self.window), reward=float(best.reward if best else 0.0),
                      speed=float(np.max(np.linalg.norm(self.previous_velocity, axis=1))),
                      acceleration=float(np.max(np.linalg.norm(self.previous_acceleration, axis=1))),
                      # This is the one-step opponent prediction used by the
                      # selected branch.  The runner compares it with the
                      # next physical pursuer state after execution.
                      predicted_pursuers=(best.pursuers.tolist() if best is not None
                                           else pursuers.tolist()))
        self.selected_cruise = getattr(best, "cruise_tracks", (None, None))
        self.last_search = window
        self.records.append(window)
        self.previous_action = action.copy()
        return SimpleNamespace(best_action=np.r_[action.ravel(), np.zeros(4)], success=best is not None,
                               metadata=window)
