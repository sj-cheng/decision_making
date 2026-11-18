"""Central-free E50 runtime primitives.

This module is deliberately independent of the historical joint controller.
It provides the message boundary and a small factorised MCTS shell: each
agent searches its own endpoint candidates while predicting the teammate from
the last valid intent message.  Sensor truth and global maps never enter the
objects below.
"""
from dataclasses import dataclass
import math
import numpy as np


def _finite(x, shape=None):
    a = np.asarray(x, dtype=float)
    if shape is not None and a.shape != shape:
        raise ValueError(f"expected shape {shape}, got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("non-finite value")
    return a.copy()


@dataclass(frozen=True)
class PeerMessage:
    sender_id: int
    seq: int
    cycle: int
    timestamp: float
    ttl: int
    pose: np.ndarray
    velocity: np.ndarray
    heading: float
    active: bool
    beliefs: tuple = ()
    map_delta: tuple = ()
    intent: np.ndarray = None

    def __post_init__(self):
        if int(self.sender_id) < 0 or int(self.seq) < 0 or int(self.cycle) < 0:
            raise ValueError("invalid peer identity")
        if int(self.ttl) < 0 or not np.isfinite(float(self.timestamp)):
            raise ValueError("invalid peer timing")
        object.__setattr__(self, "pose", _finite(self.pose, (2,)))
        object.__setattr__(self, "velocity", _finite(self.velocity, (2,)))
        if not np.isfinite(float(self.heading)):
            raise ValueError("invalid heading")
        if self.intent is not None:
            object.__setattr__(self, "intent", _finite(self.intent, (2,)))

    def valid_at(self, cycle):
        return int(cycle) - int(self.cycle) <= int(self.ttl) and int(cycle) >= int(self.cycle)


@dataclass(frozen=True)
class AgentObservation:
    agent_id: int
    cycle: int
    timestamp: float
    pose: np.ndarray
    velocity: np.ndarray
    heading: float
    local_map: object
    belief: object
    peer_messages: tuple = ()
    previous_prefix: np.ndarray = None

    def __post_init__(self):
        if int(self.agent_id) < 0 or int(self.cycle) < 0:
            raise ValueError("invalid agent identity")
        if not np.isfinite(float(self.timestamp)):
            raise ValueError("invalid timestamp")
        object.__setattr__(self, "pose", _finite(self.pose, (2,)))
        object.__setattr__(self, "velocity", _finite(self.velocity, (2,)))
        if self.previous_prefix is not None:
            object.__setattr__(self, "previous_prefix", _finite(self.previous_prefix))


@dataclass(frozen=True)
class LocalPlan:
    agent_id: int
    endpoint: np.ndarray
    action: np.ndarray
    value: float
    visits: int
    candidate_id: int
    minco_feasible: bool = True

    def __post_init__(self):
        object.__setattr__(self, "endpoint", _finite(self.endpoint, (2,)))
        object.__setattr__(self, "action", _finite(self.action, (2,)))
        if not np.isfinite(float(self.value)) or int(self.visits) < 0:
            raise ValueError("invalid local plan")


class PeerMailbox:
    """Per-agent mailbox with sequence, TTL, duplicate and sender isolation."""
    def __init__(self, agent_id):
        self.agent_id = int(agent_id)
        self._latest = {}

    def ingest(self, message):
        if not isinstance(message, PeerMessage) or message.sender_id == self.agent_id:
            return False
        old = self._latest.get(message.sender_id)
        if old is not None and message.seq <= old.seq:
            return False
        self._latest[message.sender_id] = message
        return True

    def valid(self, cycle):
        return tuple(m for m in self._latest.values() if m.valid_at(cycle))

    def ages(self, cycle):
        return {m.sender_id: int(cycle) - m.cycle for m in self._latest.values()}


class PeerBus:
    """Deterministic peer-message transport used by simulation and replay."""
    def __init__(self, agents=(0, 1), *, delay=0, drop_cycles=()):
        self.agents = tuple(int(x) for x in agents)
        self.delay = int(delay)
        self.drop_cycles = {int(x) for x in drop_cycles}
        if self.delay < 0:
            raise ValueError('delay must be non-negative')
        self._pending = []

    def publish(self, message):
        if not isinstance(message, PeerMessage):
            raise TypeError('message must be PeerMessage')
        if message.cycle in self.drop_cycles:
            return False
        self._pending.append((message.cycle + self.delay, message))
        return True

    def deliver(self, cycle, mailboxes):
        cycle = int(cycle)
        due = [x for x in self._pending if x[0] <= cycle]
        self._pending = [x for x in self._pending if x[0] > cycle]
        delivered = 0
        for _, message in due:
            for agent_id, mailbox in mailboxes.items():
                if int(agent_id) != message.sender_id and mailbox.ingest(message):
                    delivered += 1
        return delivered


class FactorisedMCTS:
    """Small value-backed MCTS over one agent's candidate endpoints.

    ``candidate_fn`` and ``value_fn`` receive only the AgentObservation and
    candidate endpoint.  The class intentionally has no joint action or map
    access, making information-boundary tests straightforward.
    """
    def __init__(self, candidate_fn, value_fn, *, simulations=32, seed=0, gamma=.98):
        self.candidate_fn = candidate_fn
        self.value_fn = value_fn
        self.simulations = int(simulations)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)
        self.value_calls = 0
        self.rollout_calls = 0
        if self.simulations < 1 or not 0 < self.gamma <= 1:
            raise ValueError("invalid MCTS settings")

    def act(self, observation):
        candidates = [np.asarray(x, float) for x in self.candidate_fn(observation)]
        if not candidates:
            raise ValueError("candidate_fn returned no endpoints")
        if any(x.shape != (2,) or not np.isfinite(x).all() for x in candidates):
            raise ValueError("invalid endpoint candidate")
        n = np.zeros(len(candidates), dtype=int)
        q = np.zeros(len(candidates), dtype=float)
        for _ in range(self.simulations):
            unvisited = np.flatnonzero(n == 0)
            if len(unvisited):
                k = int(unvisited[0])
            else:
                bonus = .7 * np.sqrt(np.log(max(2, int(n.sum()))) / n)
                k = int(np.argmax(q / n + bonus))
            endpoint = candidates[k]
            value = float(self.value_fn(observation, endpoint))
            self.value_calls += 1
            if not np.isfinite(value):
                raise ValueError("value_fn returned non-finite value")
            n[k] += 1
            q[k] += value
        # Select only an evaluated child.  With a small budget (the runtime
        # smoke uses one simulation) an unvisited child has an artificial
        # zero score and could otherwise beat a visited child with a negative
        # value, producing a plan whose ``visits`` is zero.
        evaluated = np.flatnonzero(n > 0)
        if len(evaluated) == 0:  # defensive; simulations is validated >= 1
            raise RuntimeError("MCTS produced no evaluated candidate")
        k = int(evaluated[np.argmax(q[evaluated] / n[evaluated])])
        endpoint = candidates[k]
        return LocalPlan(observation.agent_id, endpoint,
                         endpoint - observation.pose, float(q[k] / max(1, n[k])),
                         int(n[k]), k)


class DistributedEpisodeRunner:
    """Run independent controllers with a peer-message barrier.

    ``observation_fn(agent_id, cycle, peer_messages, previous_plan)`` must
    construct an AgentObservation from that agent's own sensors.  The runner
    never combines sensor inputs or plans, so it is suitable for simulation
    and for replacing the same callback with a ROS adapter later.
    """
    def __init__(self, controllers, *, delay=0, drop_cycles=(), ttl=2):
        if not controllers:
            raise ValueError('at least one controller is required')
        self.controllers = {int(k): v for k, v in controllers.items()}
        self.mailboxes = {k: PeerMailbox(k) for k in self.controllers}
        self.bus = PeerBus(tuple(self.controllers), delay=delay,
                           drop_cycles=drop_cycles)
        self.ttl = int(ttl)
        if self.ttl < 0:
            raise ValueError('ttl must be non-negative')

    def run(self, cycles, observation_fn, *, timestamp_fn=None):
        logs = []
        previous = {k: None for k in self.controllers}
        seq = {k: 0 for k in self.controllers}
        for cycle in range(int(cycles)):
            delivered = self.bus.deliver(cycle, self.mailboxes)
            plans = {}
            observations = {}
            for agent_id, controller in self.controllers.items():
                peers = self.mailboxes[agent_id].valid(cycle)
                observation = observation_fn(agent_id, cycle, peers, previous[agent_id])
                if not isinstance(observation, AgentObservation):
                    raise TypeError('observation_fn must return AgentObservation')
                plan = controller.act(observation)
                if not isinstance(plan, LocalPlan) or plan.agent_id != agent_id:
                    raise TypeError('controller must return its own LocalPlan')
                plans[agent_id] = plan
                observations[agent_id] = observation
            for agent_id, plan in plans.items():
                seq[agent_id] += 1
                observation = observations[agent_id]
                self.bus.publish(make_peer_message(
                    agent_id, seq[agent_id], cycle,
                    float(cycle if timestamp_fn is None else timestamp_fn(cycle)),
                    observation.pose, observation.velocity, observation.heading,
                    intent=plan.endpoint, ttl=self.ttl,
                ))
                previous[agent_id] = plan
            logs.append(dict(cycle=cycle, delivered=int(delivered),
                             plans={k: dict(endpoint=v.endpoint.tolist(),
                                            action=v.action.tolist(),
                                            value=float(v.value), visits=int(v.visits))
                                    for k, v in plans.items()},
                             peer_ages={k: self.mailboxes[k].ages(cycle)
                                        for k in self.controllers}))
        return logs


def make_peer_message(agent_id, seq, cycle, timestamp, pose, velocity, heading,
                      *, beliefs=(), map_delta=(), intent=None, ttl=2, active=True):
    return PeerMessage(int(agent_id), int(seq), int(cycle), float(timestamp), int(ttl),
                       pose, velocity, float(heading), bool(active), tuple(beliefs),
                       tuple(map_delta), intent)


def make_local_geometry_mcts(*, value_net=None, simulations=32, seed=0,
                            duration=2., max_distance=2., max_speed=np.inf,
                            max_acceleration=np.inf, max_candidates=8,
                            candidate_samples=16):
    """Build a single-agent geometry + MINCO + leaf-value controller."""
    from solvers.e50_distributed_geometry import (
        single_agent_endpoints, evaluate_minco_endpoint)
    cache = {}

    class _CycleProxy:
        def __init__(self, boxes, free_endpoints=()):
            self.boxes = np.asarray(boxes, float).reshape(-1, 2, 2)
            self._free_endpoints = tuple(np.asarray(x, float).copy()
                                         for x in free_endpoints)
        def obstacle_boxes(self): return self.boxes
        def free_space_endpoints(self, max_distance=2., min_clearance=.4,
                                 max_candidates=12):
            return list(self._free_endpoints)[:int(max_candidates)]
        def segment_safe(self, start, endpoint, samples=32, interpolation=True):
            points = np.linspace(np.asarray(start, float), np.asarray(endpoint, float), max(2, int(samples)))
            # The simulator world boundary is part of the known vehicle
            # operating envelope even when no obstacle return is present.
            # Reject candidates that would run into that boundary so the
            # executor does not repeatedly truncate the same prefix.
            if np.any(points < -9.0) or np.any(points > 9.0):
                return False
            if not len(self.boxes): return True
            inside = ((points[:, None, :] >= self.boxes[None, :, :, 0]) &
                      (points[:, None, :] <= self.boxes[None, :, :, 1])).all(axis=-1)
            return not bool(np.any(inside))

    def proxy(observation):
        key = (int(observation.agent_id), int(observation.cycle))
        if key not in cache:
            source = observation.local_map
            boxes = source.obstacle_boxes() if hasattr(source, 'obstacle_boxes') else np.zeros((0, 2, 2))
            free = (source.free_space_endpoints(max_distance=2., min_clearance=.4,
                                                max_candidates=12)
                    if hasattr(source, 'free_space_endpoints') else ())
            cache[key] = _CycleProxy(boxes, free)
            if len(cache) > 8:
                cache.pop(next(iter(cache)))
        return cache[key]

    def candidates(observation):
        belief = getattr(observation, 'belief', None)
        particles = np.asarray(getattr(belief, 'particles', np.zeros((0, 2))), float).reshape(-1, 2)
        weights = np.asarray(getattr(belief, 'weights', np.zeros(0)), float).reshape(-1)
        if len(particles) != len(weights):
            particles, weights = np.zeros((0, 2)), np.zeros(0)
        endpoints = list(single_agent_endpoints(
            observation.pose, particles, weights, proxy(observation),
            duration=duration, max_distance=max_distance,
            max_candidates=max_candidates, samples=candidate_samples,
            heading=observation.heading, cycle=observation.cycle,
            previous_action=observation.previous_prefix,
            agent_id=observation.agent_id))
        return tuple(endpoints)

    def value(observation, endpoint):
        track = evaluate_minco_endpoint(
            observation.pose, endpoint, duration=duration,
            start_velocity=observation.velocity, max_speed=max_speed,
            max_acceleration=max_acceleration, proxy=proxy(observation))
        belief = getattr(observation, 'belief', None)
        particles = getattr(belief, 'particles', np.zeros((0, 2)))
        if len(np.asarray(particles).reshape(-1, 2)) == 0:
            return .10 * float(np.linalg.norm(np.asarray(endpoint) - observation.pose)) - float(track['score'])
        learned = 0. if value_net is None else float(value_net(observation, endpoint))
        # A trained value network adjusts the geometric/MINCO score; it does
        # not replace local feasibility and frontier reasoning.  The untrained
        # public prior keeps its historical scale for exploration smoke tests.
        if value_net is not None and getattr(value_net, 'weights', None) is not None:
            learned *= .1
        return learned - float(track['score'])

    return FactorisedMCTS(candidates, value, simulations=simulations, seed=seed)
