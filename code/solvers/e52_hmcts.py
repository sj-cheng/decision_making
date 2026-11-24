"""Independent option/endpoint PUCT trees over a public local predictive model.

Imagined states use the current LiDAR snapshot and constant-velocity beliefs;
no future sensor frame or simulator state is available to search.
"""
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
import time
import numpy as np
from solvers.e50_distributed_geometry import (DEFAULT_MAX_ACCELERATION, _local_tangent_candidates,
    reachable_waypoint, braking_waypoints, stop_duration, evaluate_minco_endpoint, single_agent_endpoints, validate_max_acceleration)
from solvers.e50_distributed_runtime import LocalPlan
from solvers.e52_nar_nae import observation_features
from solvers.e53_capture import capture_samples


class LocalSnapshot:
    def __init__(self, source):
        self.boxes = np.asarray(source.obstacle_boxes(), float).reshape(-1, 2, 2).copy()
        self.free = tuple(np.asarray(x, float).copy() for x in source.free_space_endpoints(
            max_distance=2., min_clearance=.4, max_candidates=12)) if hasattr(source, 'free_space_endpoints') else ()

    def obstacle_boxes(self): return self.boxes
    def free_space_endpoints(self, **kwargs): return self.free
    def path_safe(self, points):
        points = np.asarray(points)
        if np.any(np.abs(points) > 9.): return False
        return not bool(np.any(((points[:, None] >= self.boxes[None, :, :, 0]) &
                                 (points[:, None] <= self.boxes[None, :, :, 1])).all(axis=-1)))
    def segment_safe(self, start, endpoint, samples=32, **kwargs):
        return self.path_safe(np.linspace(start, endpoint, max(2, samples)))


@dataclass
class Node:
    obs: object
    depth: int = 0
    parent: object = None
    endpoint: object = None
    option: int = 0
    prior: float = 1.
    reward: float = 0.
    visits: int = 0
    total: float = 0.
    children: list = field(default_factory=list)
    budget_blocked: bool = field(default=False, init=False)
    expanded: bool = False
    duration: float = 2.
    value_bonus: float | None = None
    pass_through: bool = False
    prefix: object = None


class HierarchicalMCTS:
    OPTIONS = ('chase', 'frontier', 'bypass', 'recover')
    NAE_STRATEGIES = ('legacy', 'tracked', 'leaf_residual', 'budget_leaf_residual')
    MAX_LEAF_BONUS = .15

    def __init__(self, *, nar=None, nae=None, high_simulations=256, max_candidates=8,
                 max_distance=2., duration=2., depth=3, time_budget_ms=None, gamma=.98,
                 nae_strategy='legacy', max_acceleration=None, max_speed=None):
        self.nar, self.nae = nar, nae
        if nae_strategy not in self.NAE_STRATEGIES:
            raise ValueError(f'unknown NAE strategy: {nae_strategy}')
        self.nae_strategy = nae_strategy
        self.high_simulations, self.max_candidates = int(high_simulations), int(max_candidates)
        self.max_distance, self.duration = float(max_distance), float(duration)
        self.shared_speed = max_speed is not None
        self.max_speed = 2. if max_speed is None else float(max_speed)
        if not np.isfinite(self.max_speed) or self.max_speed <= 0:
            raise ValueError('max speed must be positive and finite')
        self.continuous_acceleration = max_acceleration is not None
        self.max_acceleration = (DEFAULT_MAX_ACCELERATION if max_acceleration is None
                                 else validate_max_acceleration(max_acceleration))
        self.depth, self.time_budget_ms, self.gamma = int(depth), time_budget_ms, float(gamma)
        if min(self.high_simulations, self.max_candidates, self.depth) < 1 or (time_budget_ms is not None and time_budget_ms <= 0):
            raise ValueError('invalid E52 search budget')
        self.last_search = {}

    def _imagined(self, obs, pose, track):
        particles = np.asarray(obs.belief.particles, float).reshape(-1, 2)
        velocities = getattr(obs.belief, 'velocities', None)
        if velocities is None:
            tracks = getattr(obs.belief, 'tracks', {})
            velocities = np.asarray([v.velocity for v in tracks.values()], float).reshape(-1, 2)
        if len(velocities) != len(particles): velocities = np.zeros_like(particles)
        predictions = np.clip(particles + velocities, -9., 9.)
        prefix = track.get('positions')
        positions = (prefix[:self.midpoint+1] if prefix is not None else
                     np.linspace(obs.pose, pose, 51))
        targets = np.clip(particles + np.linspace(0., 1., len(positions))[:, None, None]
                          * velocities, -9., 9.)
        captured = np.zeros(len(particles), dtype=bool)
        if len(particles) and np.any(np.linalg.norm(positions[:, None] - targets, axis=2) < 1.05):
            hits, _ = capture_samples(positions[:, None], targets, np.ones(1, bool),
                                      np.ones(len(particles), bool), obs.local_map.boxes)
            captured = hits >= 0
        weights = np.asarray(obs.belief.weights, float)
        if len(weights) != len(particles): weights = np.ones(len(particles), float)
        # Each support point represents a tracked target, not a fresh target at
        # every depth. Remove predicted captures before the next transition.
        # This terminates an exhausted local tracking branch; an initially
        # empty belief still explores normally and no global target truth is used.
        terminal = bool(getattr(obs.belief, 'terminal', False) or
                        (len(particles) > 0 and np.all(captured)))
        belief = SimpleNamespace(particles=predictions[~captured], weights=weights[~captured],
                                 velocities=velocities[~captured], terminal=terminal,
                                 predictions=predictions, captured_mask=captured, captured_count=int(np.sum(captured)))
        delta = pose-obs.pose
        return replace(obs, pose=pose.copy(), velocity=track['velocities'][self.midpoint].copy(),
                       heading=float(np.arctan2(delta[1], delta[0])), acceleration=track["accelerations"][self.midpoint].copy(), cycle=obs.cycle+1,
                       timestamp=obs.timestamp+1., belief=belief, previous_prefix=delta)

    def _reward(self, obs, future, option, endpoint):
        if getattr(obs.belief, 'terminal', False): return 0.
        particles = np.asarray(obs.belief.particles, float).reshape(-1, 2)
        move = float(np.linalg.norm(future.pose-obs.pose))
        value = .08*move - .05
        if len(particles):
            old = np.linalg.norm(particles-obs.pose, axis=1)
            predicted = future.belief.predictions
            new = np.linalg.norm(predicted-future.pose, axis=1)
            value += float(np.min(old)-np.min(new)) + 4.*future.belief.captured_count
            before = any(obs.local_map.segment_safe(obs.pose, p) for p in particles)
            after = any(obs.local_map.segment_safe(future.pose, p) for p in future.belief.particles)
            value += .5*float(after and not before)
        else:
            value += .2*move
        if option == 2 and len(particles):
            value += .1*move  # retain progress while bypassing locally mapped walls
        for peer in obs.peer_messages:
            if peer.intent is not None:
                value -= .2*np.exp(-float(np.linalg.norm(endpoint-peer.intent)))
        return value

    def _stop_duration(self, velocity, acceleration):
        return stop_duration(velocity, acceleration, self.max_acceleration)

    def _stops(self, pose, velocity, acceleration):
        return braking_waypoints(pose, velocity, acceleration, self.max_acceleration, self.duration)

    def _braking_actions(self, observation):
        acceleration = observation.acceleration
        return self._stops(observation.pose, observation.velocity,
                           np.zeros(2) if acceleration is None else acceleration)

    def _can_brake(self, local_map, track):
        """Check that the first executed second leaves a feasible short stop."""
        if len(track['positions']) <= self.midpoint:
            return True
        pose = track['positions'][self.midpoint]
        velocity = track['velocities'][self.midpoint]
        acceleration = track['accelerations'][self.midpoint]
        if self.continuous_acceleration:
            if max(np.linalg.norm(velocity), np.linalg.norm(acceleration)) < 1e-6:
                return True
            for endpoint, duration in self._stops(pose, velocity, acceleration):
                stop = evaluate_minco_endpoint(pose, endpoint, duration=duration,
                    start_velocity=velocity, start_acceleration=acceleration,
                    max_speed=self.max_speed, continuous_speed=self.shared_speed, max_acceleration=self.max_acceleration, continuous_acceleration=True)
                if stop['feasible'] and local_map.path_safe(stop['positions']):
                    return True
            return False
        if np.linalg.norm(velocity) < 1e-6:
            return True
        duration = self._stop_duration(velocity, acceleration)
        endpoint = pose + .5*duration*velocity + duration**2*acceleration/12.
        # Most paths are far from obstacles. The stopping polynomial with
        # known initial derivatives lies within this conservative local bound.
        radius = duration*np.linalg.norm(velocity) + duration**2*np.linalg.norm(acceleration)
        boxes = local_map.boxes
        near = np.any(((pose+radius >= boxes[:, :, 0]) & (pose-radius <= boxes[:, :, 1])).all(axis=1))
        if not near and np.max(np.abs(pose)) + radius < 9.:
            return True
        stop = evaluate_minco_endpoint(pose, endpoint, duration=duration,
            start_velocity=velocity, start_acceleration=acceleration,
            max_speed=self.max_speed, continuous_speed=self.shared_speed, max_acceleration=self.max_acceleration, proxy=None)
        return stop['feasible'] and local_map.path_safe(stop['positions'])

    def _expand(self, node, deadline, *, keep_prefix=False):
        node.expanded = True
        obs = node.obs
        if getattr(obs.belief, 'terminal', False): return
        particles = np.asarray(obs.belief.particles, float).reshape(-1, 2)
        # Each high-level option has its own geometry, then a lower endpoint choice.
        pools = []
        braking = self._braking_actions(obs)
        for option in range(4):
            if option == 0 and not len(particles): continue
            if option == 3:
                # Recovery remains one existing option with at most eight lower
                # candidates, also available while the camera has no target.
                points = [endpoint for endpoint, _ in braking]
                if len(particles):
                    target = particles[np.argmin(np.linalg.norm(particles-obs.pose,axis=1))]
                    delta = target-obs.pose; norm = max(1e-9,float(np.linalg.norm(delta)))
                    side = np.array([-delta[1],delta[0]])/norm*self.max_distance
                    visibility = [obs.pose+side,obs.pose-side] + list(obs.local_map.free_space_endpoints())
                    visibility.sort(key=lambda p: (not obs.local_map.segment_safe(p,target),float(np.linalg.norm(p-target))))
                    points.extend(visibility)
                if not points: continue
            elif option == 2:
                points = _local_tangent_candidates(obs.pose, obs.local_map, self.max_distance)
            else:
                points = single_agent_endpoints(obs.pose, particles if option in (0,3) else np.zeros((0,2)),
                    np.asarray(obs.belief.weights) if option in (0,3) else np.zeros(0), obs.local_map,
                    max_distance=self.max_distance, max_candidates=self.max_candidates,
                    heading=obs.heading + (np.pi/2 if option==3 else 0.), cycle=obs.cycle,
                    previous_action=obs.previous_prefix, agent_id=obs.agent_id)
            if self.shared_speed and option != 3:
                points = [reachable_waypoint(obs.pose, point, obs.velocity, self.duration,
                                             self.max_speed, self.max_acceleration) for point in points]
            generator = getattr(self, 'waypoint_generator', None)
            if generator is not None and node.depth == 0 and option == int(not len(particles)):
                # Reserve two slots in a non-braking option. Learned endpoints
                # undergo the same polynomial, path and braking checks below.
                points = list(points[:self.max_candidates-2]) + list(generator.propose(obs))
            pools.append((option, list(points)[:self.max_candidates]))
        # Round-robin options prevents the time budget from starving exploration.
        seen = set()
        for k in range(self.max_candidates):
            for option, points in pools:
                if k >= len(points): continue
                if self.evaluations >= self.high_simulations or (deadline is not None and time.perf_counter() >= deadline):
                    node.budget_blocked = self.evaluations >= self.high_simulations
                    self._normalize(node); return
                endpoint = np.asarray(points[k], float)
                duration = self.duration
                if option == 3 and k < len(braking):
                    duration = braking[k][1]
                key = (*np.round(endpoint, 6), duration)
                if key in seen: continue
                seen.add(key)
                self.evaluations += 1
                moving_waypoint = self.shared_speed and option != 3 and np.linalg.norm(endpoint-obs.pose) > 1e-6
                # Try passing through the same decision point, retaining a
                # stopped endpoint when passing through is infeasible.
                for pass_through in ((True, False) if moving_waypoint else (False,)):
                    track = evaluate_minco_endpoint(obs.pose, endpoint, duration=duration,
                        start_velocity=obs.velocity, start_acceleration=obs.acceleration,
                        max_speed=self.max_speed, continuous_speed=self.shared_speed,
                        max_acceleration=self.max_acceleration, proxy=obs.local_map,
                        continuous_acceleration=self.continuous_acceleration, pass_through=pass_through)
                    if not track['feasible'] or not obs.local_map.path_safe(track['positions']):
                        continue
                    if node.depth == 0 and not self._can_brake(obs.local_map, track):
                        continue
                    break
                else:
                    continue
                if len(track['positions']) <= self.midpoint:
                    # A braking polynomial shorter than one cycle stays at its
                    # zero-velocity/acceleration endpoint for the remaining time.
                    for name in ('positions', 'velocities', 'accelerations'):
                        samples = track[name]
                        track[name] = np.concatenate((samples, np.repeat(samples[-1:], self.midpoint+1-len(samples), axis=0)))
                future = self._imagined(obs, track['positions'][self.midpoint], track)
                logit = self.nar.score(obs, endpoint, option)[0] if self.nar is not None and not hasattr(self.nar, 'rank_actions') else 0.
                node.children.append(Node(future, node.depth+1, node, endpoint.copy(), option,
                    float(np.clip(logit, -20., 20.)), self._reward(obs, future, option, endpoint), duration=duration, pass_through=pass_through,
                    prefix=track["positions"][:self.midpoint+1] if keep_prefix else None))
        self._normalize(node)

    def _normalize(self, node):
        if node.children:
            logits = np.array([x.prior for x in node.children])
            if hasattr(self.nar, 'rank_actions'):
                logits = np.clip(self.nar.rank_actions(node.obs, [c.endpoint for c in node.children],
                                 [c.option for c in node.children], [c.duration for c in node.children]), -20, 20)
            probs = np.exp(logits-logits.max()); probs /= probs.sum()
            for child, prior in zip(node.children, probs): child.prior = float(prior)

    def _leaf_value(self, node):
        """Return an optional edge-Q replacement or a bounded ranking bonus."""
        parent = node.parent
        if self.nae is None or parent is None or getattr(node.obs.belief, 'terminal', False):
            return 0., False
        if self.nae_strategy != 'legacy' and not len(parent.obs.belief.particles):
            return 0., False
        if self.nae_strategy not in ('leaf_residual', 'budget_leaf_residual'):
            self.nae_queries += 1
            return self.nae.score(parent.obs, node.endpoint, node.option)[1], True
        budget_leaf = (self.nae_strategy == 'budget_leaf_residual' and
                       node.budget_blocked and not node.children)
        if (node.depth != self.depth and not budget_leaf) or len(parent.children) < 2:
            return 0., False
        if node.value_bonus is None:
            peers = parent.children
            endpoints = [c.endpoint for c in peers]
            options = [c.option for c in peers]
            if hasattr(self.nae, 'rank_actions'):
                scores = self.nae.rank_actions(parent.obs, endpoints, options, [c.duration for c in peers])
            else:
                scores = self.nae.value_scores(parent.obs, endpoints, options)
            self.nae_queries += len(peers)
            # Calibrate against the feasible sibling actions at this node.
            # Keep the geometric edge reward and use bounded preferences only.
            centered = scores - np.median(scores)
            bonuses = self.MAX_LEAF_BONUS * np.tanh(centered)
            if getattr(self, "audit_ranking", False):
                raw = np.array([child.reward for child in peers])
                base_order = np.argsort(-raw, kind="stable")
                adjusted_order = np.argsort(-(raw+self.gamma*bonuses), kind="stable")
                self.nae_peer_groups += 1
                self.nae_peer_reorders += int(not np.array_equal(base_order, adjusted_order))
                self.nae_leaf_winner_changes += int(base_order[0] != adjusted_order[0])
            for child, bonus in zip(peers, bonuses):
                child.value_bonus = float(bonus)
        return node.value_bonus, False

    def act(self, observation):
        started = time.perf_counter()
        deadline = None if self.time_budget_ms is None else started + max(.001, (self.time_budget_ms-10.)/1000.)
        snapshot = LocalSnapshot(observation.local_map)
        # Support public protocol-shaped observations as well as AgentObservation.
        from solvers.e50_distributed_runtime import AgentObservation
        obs = AgentObservation(observation.agent_id, observation.cycle, getattr(observation,'timestamp',float(observation.cycle)),
            observation.pose.copy(), observation.velocity.copy(), observation.heading, snapshot,
            observation.belief, observation.peer_messages, observation.previous_prefix, getattr(observation,"acceleration",None))
        self.midpoint = int(round(1./.02)); self.evaluations = 0
        self.nae_queries = 0
        self.nae_peer_groups = self.nae_peer_reorders = self.nae_leaf_winner_changes = 0
        root = Node(obs); deepest = 0; simulations = 0
        self._expand(root, deadline)
        while root.children and simulations < self.high_simulations and (deadline is None or time.perf_counter() < deadline):
            node = root; path = [root]
            while node.children and node.depth < self.depth:
                # An option is selected using aggregated child statistics, then an endpoint.
                options = sorted(set(c.option for c in node.children))
                def option_score(o):
                    children = [c for c in node.children if c.option==o]
                    n = sum(c.visits for c in children)
                    return sum(c.total for c in children)/max(1,n) + 1.4*sum(c.prior for c in children)*np.sqrt(node.visits+1)/(1+n)
                option = max(options, key=option_score)
                node = max((c for c in node.children if c.option==option),
                    key=lambda c: c.total/max(1,c.visits)+1.4*c.prior*np.sqrt(path[-1].visits+1)/(1+c.visits))
                path.append(node)
                if node.visits == 0: break
            if node.depth < self.depth and not node.expanded and node.visits:
                self._expand(node, deadline)
            deepest = max(deepest, node.depth)
            value, replaces_edge = self._leaf_value(node)
            # NAE is trained on an action return including its immediate reward.
            # Keep that leaf Q unchanged, then add/discount preceding edges once.
            # Terminal local branches have no learned continuation; no-NAE
            # ablations accumulate the original finite-depth transition rewards.
            for child in reversed(path[1:]):
                if not (replaces_edge and child is node):
                    value = child.reward + self.gamma*value
                child.visits += 1; child.total += value
            root.visits += 1; simulations += 1
        visited = [c for c in root.children if c.visits]
        selected = max(visited, key=lambda c:(c.visits,c.total/c.visits)) if visited else max(root.children, key=lambda c:c.reward, default=None)
        elapsed = (time.perf_counter()-started)*1000.
        self.last_search = dict(elapsed_ms=elapsed, evaluations=self.evaluations, simulations=simulations,
            max_depth=deepest, nar_enabled=self.nar is not None, nae_enabled=self.nae is not None,
            nae_strategy=self.nae_strategy, nae_queries=self.nae_queries,
            root=[dict(endpoint=c.endpoint.tolist(), option=self.OPTIONS[c.option], duration=c.duration, visits=c.visits,
                       pass_through=bool(c.pass_through), value=c.total/max(1,c.visits), features=observation_features(obs,c.endpoint,level=c.option).tolist()) for c in root.children])
        if getattr(self, "audit_ranking", False):
            self.last_search['selected_root_action'] = (None if selected is None else
                (selected.endpoint.tolist(),float(selected.duration),bool(selected.pass_through)))
            self.last_search.update(peer_groups=self.nae_peer_groups, peer_reorders=self.nae_peer_reorders,
                                    leaf_winner_changes=self.nae_leaf_winner_changes)
        if selected is not None:
            predicted = np.asarray(getattr(selected.obs.belief, 'predictions',
                                           selected.obs.belief.particles), float).reshape(-1, 2)
            self.last_search['predicted_targets'] = predicted.tolist()
        if selected is None:
            return LocalPlan(obs.agent_id, obs.pose, np.zeros(2), 0., 0, -1, False)
        return LocalPlan(obs.agent_id, selected.endpoint, selected.endpoint-obs.pose,
                         selected.total/selected.visits if selected.visits else selected.reward, selected.visits, next(i for i,c in enumerate(root.children) if c is selected), duration=selected.duration, pass_through=selected.pass_through)
