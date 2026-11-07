"""Public-belief continuous neural expansion on the frozen D planning model.

This is an NTE-style belief adaptation, not an exact paper reproduction. The
generative simulator owns a sampled target index; action proposal, UCT,
rollout and value functions accept public nodes only.
"""
import hashlib
import time
import numpy as np
from learning.nte_d_local import public_features, Runtime
from solvers.search_value_belief import Controller as BeliefController, _BudgetAbort, DISCOUNT, HORIZON
from solvers.frozen_d_search_interface import (
    belief_view, root_sample, target_step, sensor_visible, safe, move)
from search_value_model import root_joint, predict_joint, condition_joint
from l45_corner_shared_20261025.corner_estimator import export_regions

VERSION = 'local_belief_continuous_nte_D_v1'


def record(obs):
    return dict(step=int(obs.step), position=np.asarray(obs.position).tolist(),
                target=None if obs.target is None else np.asarray(obs.target).tolist(),
                remaining=int(obs.remaining))


class Node:
    def __init__(self, at, joint, observed, depth, records, seen_once):
        self.at, self.joint = at.copy(), joint
        self.observation, self.depth = int(observed), int(depth)
        self.records, self.seen_once = records, bool(seen_once)
        self.actions, self.visits, self.totals, self.squares = [], [], [], []
        self.children, self.n, self.prediction = {}, 0, None
        self.features = self.parameters = self.geometry = None


class Controller(BeliefController):
    def __init__(self, boxes, speed, seed, mode='M0', policy=None, value=None,
                 budget_ms=200., simulations=100000):
        super().__init__(boxes, speed, 'FB', seed, budget_ms, simulations)
        if mode not in ('M0', 'NPI', 'NPIV', 'DIRECT'):
            raise ValueError(mode)
        self.mode = mode
        self.policy = None if policy is None else Runtime(policy)
        self.value = None if value is None else Runtime(value)
        if mode != 'M0' and self.policy is None:
            raise ValueError('Neural arm needs a frozen policy')
        if mode == 'NPIV' and self.value is None:
            raise ValueError('NPIV needs a frozen value model')
        self.public_records = []
        self.proposal_rng = np.random.default_rng((int(seed)+72720261027) % (2**64))
        self.value_rng = np.random.default_rng((int(seed)+72820261027) % (2**64))
        self.horizon, self.root_remaining = 8, 80
        self.feature_ms = self.inference_ms = self.proposal_ms = 0.
        self.proposals = self.rejections = self.stop_fallbacks = self.value_calls = 0
        self.completed_returns = []

    def observe_history(self, obs):
        r = record(obs)
        if self.public_records and self.public_records[-1]['step'] == r['step']:
            if self.public_records[-1] != r:
                raise ValueError('Conflicting public observation at same step')
        else:
            self.public_records.append(r)

    def features(self, node):
        if node.features is None:
            started = time.perf_counter()
            if node.depth == 0:
                positions = self.estimator.particles.copy()
                weights = self.estimator.particle_weights.copy()
            else:
                ids, weights, _ = self._operation(export_regions, node.joint,
                    self.estimator.regions, self.belief_rng, 64)
                positions = self.estimator.graph[0][ids]
                self.exports += 1
            node.features = public_features(self.public.boxes, positions, weights,
                node.at, node.records, self.public.speed,
                self.root_remaining-node.depth, self.horizon-node.depth, node.seen_once)
            self.feature_ms += 1000*(time.perf_counter()-started)
        return node.features

    def parameters(self, node):
        if node.parameters is None:
            x = self.features(node)
            started = time.perf_counter()
            node.parameters = self.policy(x)
            self.inference_ms += 1000*(time.perf_counter()-started)
        return node.parameters

    def propose(self, node, direct=False):
        """No sampled hidden target, target mode, or realized return accepted."""
        started = time.perf_counter()
        neural = self.mode != 'M0' and (direct or self.proposal_rng.random() < .5)
        parameters = self.parameters(node) if neural else None
        for _ in range(64):
            self._allowed()
            if neural:
                mu, logvar = parameters
                a = np.clip(mu+np.exp(.5*logvar)*self.proposal_rng.standard_normal(2), -1., 1.)*self.public.speed
            else:
                a = self.proposal_rng.uniform(-self.public.speed, self.public.speed, 2)
            self.proposals += 1
            if safe(node.at, a, self.public.speed, self.motion_boxes, np.zeros(2, np.int64)):
                self.proposal_ms += 1000*(time.perf_counter()-started)
                return a
            self.rejections += 1
        self.stop_fallbacks += 1
        self.proposal_ms += 1000*(time.perf_counter()-started)
        return np.zeros(2)

    def select(self, node):
        width = max(1, int(np.ceil(2.*np.sqrt(node.n+1.))))
        if len(node.actions) < width:
            action = self.propose(node)
            # Duplicate clipping endpoints / stop do not create false branches.
            match = next((i for i, a in enumerate(node.actions) if np.max(np.abs(a-action)) < 1e-10), None)
            if match is None:
                node.actions.append(action)
                node.visits.append(0); node.totals.append(0.); node.squares.append(0.)
                return len(node.actions)-1
        unvisited = [i for i, n in enumerate(node.visits) if n == 0]
        if unvisited:
            return unvisited[0]
        v = np.asarray(node.visits)
        return int(np.argmax(np.asarray(node.totals)/v+np.sqrt(2.*np.log(node.n+1.)/v)))

    def rollout_action(self, node):
        if node.geometry is None:
            node.geometry = self._geometry(node.at, node.joint, node.observation)
        return node.geometry.copy()

    def next_public_node(self, node, action, prediction, observed):
        at = node.at+action
        posterior = self._posterior(prediction, at, observed)
        target = None if observed < 0 else self.estimator.graph[0][observed].tolist()
        records = node.records+[dict(step=80-self.root_remaining+node.depth+1,
            position=at.tolist(), target=target, remaining=self.root_remaining-node.depth-1)]
        return Node(at, posterior, observed, node.depth+1, records,
                    node.seen_once or observed >= 0)

    def trajectory(self, root_view, forced_action=None, use_value=False):
        e, p = self.estimator, self.public
        points, links, lengths, probabilities, *_ = e.graph
        sample, target, mode = root_sample(*root_view, self.rng)
        node, rewards, path, trace = self.root, [], [], []
        tree = forced_action is None
        reason, tail = 1, 0.
        try:
            while node.depth < self.horizon:
                self._allowed()
                if tree:
                    chosen = self.select(node)
                    action = node.actions[chosen].copy()
                    path.append((node, chosen, node.depth))
                elif node.depth == 0 and forced_action is not None:
                    action = forced_action.copy()
                else:
                    action = self.rollout_action(node)
                prediction = self._prediction(node)
                if not safe(node.at, action, p.speed, self.motion_boxes, np.zeros(2, np.int64)):
                    raise RuntimeError('Illegal simulated pursuit action')
                nxt_target, mode = target_step(target, mode, links, lengths, probabilities, self.rng)
                if not safe(points[target], points[nxt_target]-points[target], 1., p.boxes, np.zeros(2, np.int64)):
                    raise RuntimeError('Illegal simulated target action')
                nxt = node.at+action
                seen = bool(sensor_visible(nxt, points[nxt_target], p.boxes))
                capture = bool(np.linalg.norm(nxt-points[nxt_target]) < 1.)
                rewards.append(float(seen and not node.seen_once)+float(capture))
                trace.append(np.r_[node.at, nxt, points[target], points[nxt_target], float(seen), float(capture)])
                if capture:
                    reason = 0
                    break
                if node.depth+1 >= self.horizon:
                    break
                observed = int(nxt_target) if seen else -1
                key = (chosen, observed) if tree else None
                child = node.children.get(key) if tree else None
                if child is None:
                    child = self.next_public_node(node, action, prediction, observed)
                    if tree:
                        node.children[key] = child
                        self.nodes.append(child)
                        self.positive_branches += int(observed >= 0)
                        self.negative_branches += int(observed < 0)
                    if tree and use_value and self.value_rng.random() < .5:
                        x = self.features(child)
                        started = time.perf_counter()
                        tail = self.value(x)
                        self.inference_ms += 1000*(time.perf_counter()-started)
                        self.value_calls += 1
                        break
                    tree = False
                node, target = child, nxt_target
            self._allowed()
        except _BudgetAbort:
            return np.asarray(trace).reshape(-1, 10), int(sample), 2, None
        value = tail
        returns = np.empty(len(rewards))
        for depth in range(len(rewards)-1, -1, -1):
            value = rewards[depth]+DISCOUNT*value
            returns[depth] = value
        if forced_action is None:
            for visited, chosen, depth in path:
                visited.n += 1; visited.visits[chosen] += 1
                visited.totals[chosen] += float(returns[depth])
                visited.squares[chosen] += float(returns[depth]**2)
        return np.asarray(trace).reshape(-1, 10), int(sample), reason, float(returns[0])

    def act(self, obs, fixed_simulations=None):
        started = time.perf_counter()
        self.deadline = float('inf') if fixed_simulations is not None else started+self.budget_ms/1000.
        self.operation_ms = 1.
        self.queries = self.exports = self.conditionings = 0
        self.positive_branches = self.negative_branches = 0
        self.minimum_ess, self.filter_mass_minimum = 64., 1.
        self.geometry_preparation_ms = 0.
        self.feature_ms = self.inference_ms = self.proposal_ms = 0.
        self.proposals = self.rejections = self.stop_fallbacks = self.value_calls = 0
        self.observe_history(obs)
        self.estimator.update(obs)
        update_ms = 1000*(time.perf_counter()-started)
        self.horizon, self.root_remaining = min(HORIZON, obs.remaining), int(obs.remaining)
        self.nodes, self.traces, self.completed_returns = [], [], []
        counts, terminations = np.zeros(64, np.int64), np.zeros(3, np.int64)
        action = np.zeros(2)
        self.root = None
        stamp = time.perf_counter()
        if obs.target is not None:
            action = move(obs.position, obs.target, self.public.speed, self.public.vertices,
                          self.public.dist, self.public.boxes, np.zeros(2, np.int64))
            if not safe(obs.position, action, self.public.speed, self.motion_boxes, np.zeros(2, np.int64)):
                action = np.zeros(2); self.stop_fallbacks += 1
        else:
            self.root = Node(obs.position, root_joint(self.estimator), -1, 0,
                             list(self.public_records), False)
            self.nodes.append(self.root)
        prepare_ms = 1000*(time.perf_counter()-stamp)
        stamp = time.perf_counter()
        completed = attempted = 0
        cap = self.simulations if fixed_simulations is None else int(fixed_simulations)
        if self.root is not None:
            try:
                if self.mode == 'DIRECT':
                    action = self.propose(self.root, direct=True)
                else:
                    view = belief_view(self.estimator)
                    for _ in range(cap):
                        self._allowed()
                        trace, sample, reason, value = self.trajectory(view, use_value=self.mode == 'NPIV')
                        self.traces.append(trace)
                        attempted += 1; counts[sample] += 1; terminations[reason] += 1
                        if reason == 2:
                            break
                        completed += 1; self.completed_returns.append(value)
                    if completed:
                        action = self.root.actions[int(np.argmax(self.root.visits))].copy()
            except _BudgetAbort:
                if completed:
                    action = self.root.actions[int(np.argmax(self.root.visits))].copy()
                else:
                    self.stop_fallbacks += 1
        search_ms = 1000*(time.perf_counter()-stamp)
        stamp = time.perf_counter()
        if not safe(obs.position, action, self.public.speed, self.public.boxes, np.zeros(2, np.int64)):
            raise RuntimeError('Illegal final action')
        final_ms = 1000*(time.perf_counter()-stamp)
        self.last_stats = dict(version=VERSION, mode=self.mode, update_ms=update_ms,
            build_ms=self.estimator.last_stats['build_ms'], candidate_ms=prepare_ms,
            tree_prepare_ms=prepare_ms, search_including_prepare_ms=search_ms,
            feature_ms=self.feature_ms, inference_ms=self.inference_ms,
            proposal_including_features_ms=self.proposal_ms,
            final_check_ms=final_ms, total_ms=1000*(time.perf_counter()-started),
            simulations=completed, attempted_simulations=attempted,
            root_counts=counts.tolist(), terminations=terminations.tolist(),
            root_actions=[] if self.root is None else [a.tolist() for a in self.root.actions],
            root_visits=[] if self.root is None else self.root.visits.copy(),
            root_totals=[] if self.root is None else self.root.totals.copy(),
            root_squares=[] if self.root is None else self.root.squares.copy(),
            root_candidates=0 if self.root is None else len(self.root.actions),
            planning_depth=self.horizon, nodes=len(self.nodes), value_calls=self.value_calls,
            positive_observation_nodes=self.positive_branches,
            negative_observation_nodes=self.negative_branches,
            exact_conditionings=self.conditionings, geometry_queries=self.queries,
            planning_exports=self.exports, proposals=self.proposals,
            rejected_proposals=self.rejections, stop_fallbacks=self.stop_fallbacks,
            source='visible_geometry' if obs.target is not None else 'continuous_'+self.mode,
            minimum_observation_probability=self.filter_mass_minimum)
        return action

    def public_signature(self):
        e = self.estimator
        arrays = [e.particles, e.particle_weights, e.rng, self.rng]
        arrays += [a for t in self.traces for a in [t]]
        return hashlib.sha256(b''.join(np.asarray(a).tobytes() for a in arrays)).hexdigest()

    def independent_rollouts(self, actions, draws=96, budget_ms=2000., paired=True):
        """Held-out fixed-action MC. No search allocation enters its estimates."""
        started = time.perf_counter()
        self.deadline = started+budget_ms/1000.
        if self.root.depth == 0:
            view = belief_view(self.estimator)
        else:
            ids, weights, _ = export_regions(self.root.joint, self.estimator.regions, self.belief_rng, 64)
            modes = self.root.joint[ids].copy()
            modes /= modes.sum(1, keepdims=True)
            view = (ids, weights, modes)
        values, traces = [], []
        for _ in range(draws):
            before = self.rng.copy()
            row, row_traces = [], []
            for action in actions:
                if paired:
                    self.rng[:] = before
                trace, _, reason, value = self.trajectory(view, np.asarray(action), use_value=False)
                if reason == 2:
                    return np.asarray(values).reshape(-1, len(actions)), traces, 1000*(time.perf_counter()-started)
                row.append(value); row_traces.append(trace)
            # A common stream defines paired counterfactual worlds. Each next
            # draw's fresh seed is independent of action/realized return.
            self.rng[:] = np.array([int(np.random.SeedSequence([int(before[0]), 72920261027]).generate_state(1)[0])], np.int64)
            values.append(row); traces.extend(row_traces)
        return np.asarray(values).reshape(-1, len(actions)), traces, 1000*(time.perf_counter()-started)
