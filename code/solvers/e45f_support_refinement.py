"""E45-F amendment: refine a genuinely exhausted 0.5 lattice to 0.25.

The frozen estimator and its successful path remain untouched. Recovery uses
only the last identified public measurement and every subsequent team report.
It changes the finite discretization, not the motion sigma or intent dynamics;
it is explicitly a revised estimator, not an exact frozen-D posterior.
"""
import json
import numpy as np
from numba import njit

from e45c_belief_diagnostics import RecordedTeamBelief
from solvers.e45_team_adapter import team_hidden_mask
from solvers.frozen_d_search_interface import PublicMap
from solvers.local_pursuit import clear, legal
from solvers.history_recovery import GEOMETRY_MARGIN
from search_value_model import predict_joint
from l45_corner_shared_20261025.corner_estimator import (
    map_probabilities, geometry_labels, export_regions)
from l45_corner_shared_20261025.multimodal_estimator import SIGMA

SPACING = .25


@njit(cache=True)
def lattice(last, boxes, spacing):
    count = np.zeros(2, np.int64)
    axes = []
    for axis in range(2):
        values = [-10., 10.]
        for k in range(-int(np.ceil(20 / spacing)), int(np.ceil(20 / spacing)) + 1):
            q = last[axis] + k * spacing
            if -10. <= q <= 10.:
                values.append(q)
        axes.append(np.unique(np.asarray(values)))
    nx, ny = len(axes[0]), len(axes[1])
    points = np.empty((nx * ny, 2))
    motion, sensor = boxes.copy(), boxes.copy()
    motion[:, :, 0] -= GEOMETRY_MARGIN
    motion[:, :, 1] += GEOMETRY_MARGIN
    sensor[:, :, 0] += GEOMETRY_MARGIN
    sensor[:, :, 1] -= GEOMETRY_MARGIN
    free = np.zeros(nx * ny, np.bool_)
    for i in range(len(points)):
        points[i] = np.array([axes[0][i // ny], axes[1][i % ny]])
        free[i] = legal(points[i], motion, count)
    # At most two extra arena-edge coordinates per axis.
    radius = int(round(1 / spacing)) + 2
    capacity = (2 * int(round(1 / spacing)) + 3) ** 2
    links = np.full((len(points), capacity), -1, np.int32)
    lengths = np.zeros(len(points), np.int32)
    for i in range(len(points)):
        if not free[i]:
            continue
        x, y = i // ny, i % ny
        for xx in range(max(0, x - radius), min(nx, x + radius + 1)):
            for yy in range(max(0, y - radius), min(ny, y + radius + 1)):
                j = xx * ny + yy
                if (free[j] and np.max(np.abs(points[j] - points[i])) <= 1 + 1e-10
                        and clear(points[i], points[j], motion, count)):
                    if lengths[i] >= capacity:
                        raise RuntimeError('Refined lattice degree exceeds derived bound')
                    links[i, lengths[i]] = j
                    lengths[i] += 1
    root = int(np.argmin(np.sum((points - last) ** 2, axis=1)))
    if np.linalg.norm(points[root] - last) > 1e-8 or not free[root]:
        raise RuntimeError('No lawful refined anchor')
    return points, links, lengths, sensor, root


@njit(cache=True)
def motion_probabilities(points, links, lengths, means):
    probabilities = np.zeros((len(points), links.shape[1], 9))
    for i in range(len(points)):
        for k in range(lengths[i]):
            delta = points[links[i, k]] - points[i]
            for m in range(9):
                probabilities[i, k, m] = np.exp(-.5 * np.sum((delta - means[i, m]) ** 2) / SIGMA ** 2)
        for m in range(9):
            z = probabilities[i, :lengths[i], m].sum()
            if z > 0:
                probabilities[i, :lengths[i], m] /= z
    return probabilities


def refined_posterior(boxes, history, target, spacing=SPACING):
    """Recompute all negative likelihoods since the last positive; no truth input."""
    positives = [(k, np.asarray(q, float)) for k, obs in enumerate(history)
                 for i, q in enumerate(obs['reports'][target])
                 if obs['active_p'][i] and q is not None]
    if not positives:
        return None
    start, last = positives[-1]
    points, links, lengths, sensor, root = lattice(last, boxes, spacing)
    public = PublicMap(boxes, 1.)
    # Frozen route means do not depend on outgoing-edge enumeration. One edge
    # suffices to extract them; normalize the Gaussian over ALL fine edges.
    _, means, queries = map_probabilities(points, links[:, :1].copy(), np.minimum(lengths, 1),
                                         last, boxes, public.vertices, public.dist)
    probabilities = motion_probabilities(points, links, lengths, means)
    graph = (points, links, lengths, probabilities, sensor, root, queries)
    weights = np.zeros((len(points), 9)); weights[root] = 1 / 9.
    parents, observers = [], []
    for obs in history[start + 1:]:
        prediction = predict_joint(weights, links, lengths, probabilities)
        allowed = team_hidden_mask(points, np.asarray(obs['positions']), np.asarray(obs['active_p']), boxes)
        prediction[~allowed] = 0.
        mass = float(prediction.sum())
        if mass <= 0 or not np.isfinite(mass):
            return None
        support = prediction.sum(1) > 0
        parent = np.full(len(points), -1, np.int32)
        for i in np.flatnonzero(weights.sum(1) > 0):
            for k in links[i, :lengths[i]]:
                if support[k] and parent[k] < 0:
                    parent[k] = i
        if np.any(support & (parent < 0)):
            raise RuntimeError('Refined support lacks a legal ancestral edge')
        parents.append(parent); observers.append(np.asarray(obs['positions']).copy())
        weights = prediction / mass
    return graph, weights, parents, observers


class RefinedTeamBelief(RecordedTeamBelief):
    def __init__(self, boxes, seed):
        super().__init__(boxes, seed)
        self.refinements = []
        self.failure_archives = []

    def record_failure(self, exc, obs):
        stem = super().record_failure(exc, obs)
        self.failure_archives.append(stem)
        return stem

    def update(self, obs):
        count = len(self.refinements)
        super().update(obs)
        if len(self.refinements) > count:
            for stem in self.failure_archives:
                path = stem.with_suffix('.json')
                metadata = json.loads(path.read_text())
                if metadata.get('failed_public_step') == obs['step'] and metadata.get('recovered'):
                    metadata.update(repair='E45-F adaptive 0.25 lattice; all public negatives since last measurement',
                                    frozen_discretization_changed=True, refinements=self.refinements[count:])
                    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
            self.last_stats.update(adaptive_grid_refinement=self.refinements[count:])

    def repair_boundary(self, estimator, prior, prior_graph, prior_rng, allowed):
        if super().repair_boundary(estimator, prior, prior_graph, prior_rng, allowed):
            return True
        target = next(j for j, e in enumerate(self.estimators) if e is estimator)
        result = refined_posterior(self.boxes, self.public_history, target)
        if result is None:
            return False
        graph, weights, parents, observers = result
        estimator.graph = graph; estimator.weights = weights
        estimator.reachable = weights.sum(1) > 0
        estimator.parents = parents
        # The frozen estimator stores its selected observer at each step.
        estimator.observers = [o[target] if self.public_history[-len(observers) + k]['active_p'][target]
                               else o[np.flatnonzero(self.public_history[-len(observers) + k]['active_p'])[0]]
                               for k, o in enumerate(observers)]
        estimator.regions = geometry_labels(graph[0], self.boxes)
        estimator.geometry_means = None
        estimator.rng[:] = prior_rng
        estimator.ids, estimator.particle_weights, estimator.labels = export_regions(
            weights, estimator.regions, estimator.rng, 64)
        estimator.particles = graph[0][estimator.ids].copy()
        estimator.last_stats.update(support_points=int(estimator.reachable.sum()),
                                    mode_mass=weights.sum(0).tolist(), refined_spacing=SPACING)
        self.refinements.append(dict(step=self.public_history[-1]['step'], target=target,
                                     spacing=SPACING, support_points=int(estimator.reachable.sum())))
        return True


def install(evidence):
    import e45c_belief_diagnostics as diagnostic
    diagnostic.install(evidence)
    import solvers.e45_team_adapter as adapter
    import solvers.e45b_joint_nte as joint
    import run_e45b_role_free as runner
    adapter.TeamBelief = joint.TeamBelief = runner.TeamBelief = RefinedTeamBelief
