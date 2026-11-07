"""Research-only exact finite-grid belief search: open-loop / observation feedback.

The online estimator is the immutable accepted D.  Inside each decision the
planning model is D's *fixed root graph*, initialized from its weighted 64-sample
export.  Planning posteriors are Bayes posteriors for that model, not a claim to
reproduce D's positive-measurement graph reset.  No environment is accepted.

OL and FB execute the same code.  The sole mechanism switch is whether a
simulated sensor observation conditions the planning belief and tree history.
Both use the same legal root candidates, deeper compass actions, UCT, rollout
geometry, target kernel, reward, discount and physical planning horizon.
"""
import time
import numpy as np
from numba import njit

from solvers.frozen_d_search_interface import (
    Controller as OriginalController, COMPASS, belief_view, geometric_candidates,
    root_sample, target_step, sensor_visible, safe, move,
)
from l45_corner_shared_20261025.corner_estimator import export_regions
from l45_corner_shared_20261025.local_pursuit import clear, edge
from solvers.rollout_geometry_exact import geometric_candidates_exact
from search_value_model import root_joint, predict_joint, condition_joint

VERSION = 'fixed_root_exact_belief_feedback_v1'
DISCOUNT = .95
HORIZON = 8


@njit(cache=True)
def prepare_all_tails(points, vertices, dist, boxes, speed):
    """Exact public map cost tails, independent of posterior or hidden samples."""
    targets = np.empty((len(points)+len(vertices), 2))
    targets[:len(points)] = points
    targets[len(points):] = vertices
    tails = np.full((len(targets), len(vertices)), 1000000, np.int64)
    count = np.zeros(2, np.int64)
    for t in range(len(targets)):
        for j in range(len(vertices)):
            if clear(vertices[j], targets[t], boxes, count):
                cost = edge(vertices[j], targets[t], speed)
                for i in range(len(vertices)):
                    tails[t, i] = min(tails[t, i], dist[i, j]+cost)
    return tails


@njit(cache=True)
def export_geometry_inputs(ids, weights, regions, points, vertices, all_tails):
    """Preserve original region center/order/ties and look up exact map tails."""
    samples = points[ids]
    labels = regions[ids]
    targets = np.empty((64+len(vertices), 2))
    targets[:64] = samples
    targets[64:] = vertices
    tails = np.empty((len(targets), len(vertices)), np.int64)
    for i in range(64):
        tails[i] = all_tails[ids[i]]
    tails[64:] = all_tails[len(points):]
    goals = np.empty(64, np.int64)
    number = 0
    for r in range(int(labels.max())+1):
        mass = 0.
        center = np.zeros(2)
        for i in range(64):
            if labels[i] == r:
                center += weights[i]*samples[i]
                mass += weights[i]
        if mass <= 0:
            continue
        center /= mass
        winner, distance = -1, 1e100
        for i in range(64):
            if labels[i] == r and np.sum((samples[i]-center)**2) < distance:
                winner, distance = i, np.sum((samples[i]-center)**2)
        if number >= 64:
            raise RuntimeError('Candidate capacity exceeded')
        goals[number] = winner
        number += 1
    for j in range(len(vertices)):
        if number >= 64:
            raise RuntimeError('Candidate capacity exceeded')
        goals[number] = 64+j
        number += 1
    return targets, goals[:number].copy(), tails


class _BudgetAbort(Exception):
    pass


class _Node:
    """Public simulated history and its immutable conditional distribution."""
    __slots__ = ('at', 'joint', 'observation', 'options', 'prediction', 'children',
                 'visits', 'totals', 'n', 'geometry', 'depth')

    def __init__(self, at, joint, observation, options, depth):
        self.at = at.copy()
        self.joint = joint
        self.observation = observation
        self.options = options
        self.prediction = None
        self.children = {}
        self.visits = np.zeros(len(options), np.int64)
        self.totals = np.zeros(len(options))
        self.n = 0
        self.geometry = None
        self.depth = depth


class Controller(OriginalController):
    def __init__(self, boxes, speed, method, seed, budget_ms=200., simulations=1000):
        if method not in ('OL', 'FB'):
            raise ValueError(method)
        super().__init__(boxes, speed, 'T+Geo', seed, budget_ms, simulations)
        self.research_method = method
        self.feedback = method == 'FB'
        # Particle export for policy computation never consumes the simulator
        # target RNG.  Conditional distributions themselves are exact arrays.
        self.belief_rng = np.array([(seed+86120261026) % 4294967296], np.int64)
        self.root = None
        self.nodes = []
        self.deadline = float('inf')
        self.operation_ms = 2.
        self.queries = 0
        self.exports = 0
        self.conditionings = 0
        self.positive_branches = 0
        self.negative_branches = 0
        self.minimum_ess = 64.
        self.filter_mass_minimum = 1.
        self.exact_geometry_reuse = True
        self._geometry_graph = None
        self._geometry_tails = None
        self.geometry_preparation_ms = 0.

    def _allowed(self):
        if time.perf_counter() >= self.deadline-.002:
            raise _BudgetAbort()

    def _operation(self, callback, *args):
        # A guarded admission followed by a return-time check prevents a partial
        # simulation from reaching backpropagation when a query overruns.
        if time.perf_counter()+(2.*self.operation_ms+2.)/1000. >= self.deadline:
            raise _BudgetAbort()
        started = time.perf_counter()
        value = callback(*args)
        self.operation_ms = max(self.operation_ms, 1000*(time.perf_counter()-started))
        self._allowed()
        return value

    def _prediction(self, node):
        if node.prediction is None:
            e = self.estimator
            node.prediction = self._operation(predict_joint, node.joint, e.graph[1], e.graph[2], e.graph[3])
        return node.prediction

    def _posterior(self, prediction, at, observed):
        if not self.feedback:
            return prediction
        e = self.estimator
        posterior = self._operation(condition_joint, prediction, at, e.graph[0], self.public.boxes, observed)
        self.conditionings += 1
        # The exact event probability is scored for numerical diagnostics only.
        if observed >= 0:
            mass = float(prediction[observed].sum())
        else:
            support = posterior.sum(axis=1) > 0
            mass = float(prediction[support].sum())
        self.filter_mass_minimum = min(self.filter_mass_minimum, mass)
        return posterior

    def _options(self, at):
        proposals = COMPASS*self.public.speed
        count = np.zeros(2, np.int64)
        answer = np.asarray([a for a in proposals
                             if safe(at, a, self.public.speed, self.motion_boxes, count)])
        if not len(answer):
            raise RuntimeError('No legal compass action, including stop')
        return answer

    def _geometry(self, at, joint, observed=-1):
        """Policy receives a distribution and a public sensor measurement only."""
        p, e = self.public, self.estimator
        if self.feedback and observed >= 0:
            action = self._operation(move, at, e.graph[0][observed], p.speed,
                                     p.vertices, p.dist, p.boxes, np.zeros(2, np.int64))
            if not safe(at, action, p.speed, self.motion_boxes, np.zeros(2, np.int64)):
                # The same checked stop is used for any numerical tangent.
                action = np.zeros(2)
        else:
            ids, weights, _labels = self._operation(export_regions, joint, e.regions, self.belief_rng, 64)
            self.exports += 1
            self.minimum_ess = min(self.minimum_ess, float(1./np.sum(weights**2)))
            if self.exact_geometry_reuse:
                if self._geometry_graph is not e.graph:
                    stamp = time.perf_counter()
                    self._geometry_tails = self._operation(prepare_all_tails, e.graph[0],
                        p.vertices, p.dist, p.boxes, p.speed)
                    self._geometry_graph = e.graph
                    self.geometry_preparation_ms += 1000*(time.perf_counter()-stamp)
                prepared = self._operation(export_geometry_inputs, ids, weights, e.regions,
                    e.graph[0], p.vertices, self._geometry_tails)
                _, action = self._operation(geometric_candidates_exact, at, e.graph[0][ids], weights,
                                           e.regions[ids], p.speed, p.vertices, p.dist,
                                           p.boxes, self.motion_boxes, *prepared)
            else:
                _, action = self._operation(geometric_candidates, at, e.graph[0][ids], weights,
                                           e.regions[ids], p.speed, p.vertices, p.dist,
                                           p.boxes, self.motion_boxes)
        self.queries += 1
        return action

    def _select(self, node):
        # No current simulated target position/mode is in this interface.
        unvisited = np.flatnonzero(node.visits == 0)
        if len(unvisited):
            return int(unvisited[0])
        values = node.totals/node.visits + np.sqrt(2.*np.log(node.n+1)/node.visits)
        return int(np.argmax(values))

    def _simulation(self, root_view, horizon):
        e, p = self.estimator, self.public
        points, links, lengths, probabilities, *_ = e.graph
        sample, target, mode = root_sample(*root_view, self.rng)
        node = self.root
        at = node.at.copy()
        joint = node.joint
        observed = -1
        rewards = []
        path = []
        trace = []
        in_tree = True
        seen_once = False
        reason = 1
        try:
            for depth in range(horizon):
                self._allowed()
                if in_tree:
                    chosen = self._select(node)
                    action = node.options[chosen].copy()
                    path.append((node, chosen, depth))
                    prediction = self._prediction(node)
                else:
                    action = self._geometry(at, joint, observed)
                    prediction = self._operation(predict_joint, joint, links, lengths, probabilities)
                if not safe(at, action, p.speed, self.motion_boxes, np.zeros(2, np.int64)):
                    raise RuntimeError('Illegal simulated pursuit action')
                nxt_target, mode = target_step(target, mode, links, lengths, probabilities, self.rng)
                if not safe(points[target], points[nxt_target]-points[target], 1., p.boxes, np.zeros(2, np.int64)):
                    raise RuntimeError('Illegal simulated target action')
                nxt = at+action
                seen = bool(sensor_visible(nxt, points[nxt_target], p.boxes))
                capture = bool(np.linalg.norm(nxt-points[nxt_target]) < 1.)
                rewards.append(float(seen and not seen_once)+float(capture))
                trace.append(np.r_[at, nxt, points[target], points[nxt_target], float(seen), float(capture)])
                seen_once = seen_once or seen
                sensor_observation = int(nxt_target) if seen else -1
                # The simulator's hidden sample is allowed to generate this
                # observation; only the observation crosses into the policy.
                observed = sensor_observation if self.feedback else -1
                if capture:
                    reason = 0
                    break
                if depth+1 < horizon:
                    if in_tree:
                        key = (chosen, observed) if self.feedback else (chosen, -2)
                        child = node.children.get(key)
                        if child is None:
                            posterior = self._posterior(prediction, nxt, observed)
                            child = _Node(nxt, posterior, observed, self._options(nxt), depth+1)
                            node.children[key] = child
                            self.nodes.append(child)
                            if self.feedback:
                                self.positive_branches += int(observed >= 0)
                                self.negative_branches += int(observed < 0)
                            in_tree = False
                        node = child
                        joint = child.joint
                    else:
                        joint = self._posterior(prediction, nxt, observed)
                at, target = nxt, nxt_target
            self._allowed()
        except _BudgetAbort:
            return np.asarray(trace, dtype=float).reshape(-1, 10), int(sample), 2
        value = 0.
        returns = np.empty(len(rewards))
        for depth in range(len(rewards)-1, -1, -1):
            value = rewards[depth]+DISCOUNT*value
            returns[depth] = value
        for visited, chosen, depth in path:
            visited.n += 1
            visited.visits[chosen] += 1
            visited.totals[chosen] += returns[depth]
        return np.asarray(trace, dtype=float).reshape(-1, 10), int(sample), reason

    def act(self, obs, fixed_simulations=None):
        started = time.perf_counter()
        self.deadline = float('inf') if fixed_simulations is not None else started+self.budget_ms/1000.
        self.operation_ms = 2.
        self.queries = self.exports = self.conditionings = 0
        self.positive_branches = self.negative_branches = 0
        self.minimum_ess, self.filter_mass_minimum = 64., 1.
        self.geometry_preparation_ms = 0.
        self.estimator.update(obs)
        update_ms = 1000*(time.perf_counter()-started)
        e, p = self.estimator, self.public
        stamp = time.perf_counter()
        if obs.target is not None:
            action = move(obs.position, obs.target, p.speed, p.vertices, p.dist, p.boxes, np.zeros(2, np.int64))
            roots = action.reshape(1, 2)
            view = None
        else:
            roots, action = geometric_candidates(obs.position, e.particles, e.particle_weights,
                                                 e.regions[e.ids], p.speed, p.vertices, p.dist,
                                                 p.boxes, self.motion_boxes)
            view = belief_view(e)
        candidate_ms = 1000*(time.perf_counter()-stamp)
        self.root, self.nodes, self.traces = None, [], []
        completed = attempted = 0
        counts = np.zeros(64, np.int64)
        terminations = np.zeros(3, np.int64)
        preparation_ms = 0.
        stamp = time.perf_counter()
        cap = self.simulations if fixed_simulations is None else int(fixed_simulations)
        if view is not None:
            prepare_start = time.perf_counter()
            self.root = _Node(obs.position, root_joint(e), -1, roots, 0)
            self.nodes.append(self.root)
            preparation_ms = 1000*(time.perf_counter()-prepare_start)
            for _ in range(cap):
                if time.perf_counter() >= self.deadline-.002:
                    break
                trace, sample, reason = self._simulation(view, min(HORIZON, obs.remaining))
                self.traces.append(trace)
                attempted += 1
                counts[sample] += 1
                terminations[reason] += 1
                if reason == 2:
                    break
                completed += 1
            if completed:
                action = roots[int(np.argmax(self.root.visits))].copy()
        search_ms = 1000*(time.perf_counter()-stamp)
        stamp = time.perf_counter()
        if not safe(obs.position, action, p.speed, p.boxes, np.zeros(2, np.int64)):
            raise RuntimeError('Illegal final pursuit action')
        final_ms = 1000*(time.perf_counter()-stamp)
        total_ms = 1000*(time.perf_counter()-started)
        self.last_stats = dict(
            version=VERSION, feedback=self.feedback, update_ms=update_ms,
            build_ms=e.last_stats['build_ms'], candidate_ms=candidate_ms,
            tree_prepare_ms=preparation_ms, search_including_prepare_ms=search_ms,
            final_check_ms=final_ms, total_ms=total_ms, over_50ms=total_ms > 50.,
            over_200ms=total_ms > 200., simulations=completed,
            attempted_simulations=attempted, budget_aborted_simulations=int(terminations[2]),
            root_counts=counts.tolist(), terminations=terminations.tolist(),
            root_visits=[] if self.root is None else self.root.visits.tolist(),
            root_totals=[] if self.root is None else self.root.totals.tolist(),
            root_candidates=len(roots), planning_depth=min(HORIZON, obs.remaining),
            nodes=len(self.nodes), positive_observation_nodes=self.positive_branches,
            negative_observation_nodes=self.negative_branches,
            exact_conditionings=self.conditionings, geometry_queries=self.queries,
            planning_exports=self.exports, minimum_export_ess=self.minimum_ess,
            exact_geometry_reuse=self.exact_geometry_reuse,
            geometry_preparation_ms=self.geometry_preparation_ms,
            minimum_observation_probability=self.filter_mass_minimum,
            planning_joint='weighted64 root; exact fixed-root-model Bayes updates',
            source='visible_geometry' if view is None else 'belief_feedback_search' if completed else 'weighted_geometry',
        )
        return action
