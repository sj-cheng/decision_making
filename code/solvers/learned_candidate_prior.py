"""Research-only root PUCT prior / direct policy; immutable frozen-D and FB.

All search arms share the exact same PUCT formula and uniform deep priors.
Only the root prior is uniform, handcrafted G scores, or learned G imitation.
The policy entry point accepts public observations only.
"""
import time
import numpy as np
from solvers.search_value_belief import Controller as Feedback
from solvers.frozen_d_search_interface import (
    Controller as Geometry, geometric_candidates, safe)
from learning.candidate_prior import Network, features, geometry_scores, mixed_prior


class PublicHistory:
    def __init__(self):
        self.last_positive = None
        self.positive_step = 0
        self.previous_position = None
        self.previous_motion = np.zeros(2)

    def update(self, obs):
        if self.previous_position is not None:
            self.previous_motion = obs.position-self.previous_position
        self.previous_position = obs.position.copy()
        if obs.target is not None:
            self.last_positive = obs.target.copy()
            self.positive_step = obs.step

    def inputs(self, obs):
        if self.last_positive is None:
            raise ValueError('Public positive observation required')
        return (self.last_positive, obs.step-self.positive_step, self.previous_motion)


def candidate_data(controller, obs, roots):
    e, p = controller.estimator, controller.public
    return features(obs.position, roots, e.particles, e.particle_weights,
                    p.boxes, p.speed, obs.remaining, *controller.history.inputs(obs))


class Direct(Geometry):
    def __init__(self, boxes, speed, seed, model, budget_ms=200.):
        super().__init__(boxes, speed, 'G', seed, budget_ms)
        self.network = Network(model)
        self.history = PublicHistory()

    def act(self, obs, fixed_simulations=None):
        started = time.perf_counter()
        self.history.update(obs)
        base = super().act(obs, fixed_simulations=0)
        inference_ms = 0.
        if obs.target is None:
            e, p = self.estimator, self.public
            roots, _ = geometric_candidates(obs.position, e.particles, e.particle_weights,
                e.regions[e.ids], p.speed, p.vertices, p.dist, p.boxes, self.motion_boxes)
            stamp = time.perf_counter()
            logits = self.network(candidate_data(self, obs, roots))
            inference_ms = 1000*(time.perf_counter()-stamp)
            base = roots[int(np.argmax(logits))].copy()
            self.last_stats['root_logits'] = logits.tolist()
        if not safe(obs.position, base, self.public.speed, self.public.boxes, np.zeros(2, np.int64)):
            raise RuntimeError('Illegal direct candidate')
        self.last_stats.update(total_ms=1000*(time.perf_counter()-started),
            inference_ms=inference_ms, source='network_candidate_policy' if obs.target is None else 'visible_geometry')
        return base


class Controller(Feedback):
    def __init__(self, boxes, speed, seed, prior, model=None, budget_ms=200.):
        super().__init__(boxes, speed, 'FB', seed, budget_ms, simulations=4096)
        if prior not in ('uniform', 'geometry', 'learned'):
            raise ValueError(prior)
        self.prior_kind = prior
        self.network = Network(model) if prior == 'learned' else None
        self.history = PublicHistory()
        self.priors = None
        self.prior_ms = 0.
        self.current_obs = None

    def _select(self, node):
        if node is self.root:
            if self.priors is None:
                stamp = time.perf_counter()
                e, p, obs = self.estimator, self.public, self.current_obs
                if self.prior_kind == 'uniform':
                    self.priors = np.ones(len(node.options))/len(node.options)
                elif self.prior_kind == 'geometry':
                    scores = geometry_scores(obs.position, node.options, e.particles,
                        e.particle_weights, p.speed, p.vertices, p.dist, p.boxes)
                    self.priors = mixed_prior(scores)
                else:
                    self.priors = mixed_prior(self.network(candidate_data(self, obs, node.options)))
                self.prior_ms = 1000*(time.perf_counter()-stamp)
                self._allowed()
            prior = self.priors
        else:
            prior = np.ones(len(node.options))/len(node.options)
        q = node.totals/np.maximum(1, node.visits)
        # cpuct=1.5, identical root/deep selection for every search arm.
        return int(np.argmax(q+1.5*prior*np.sqrt(node.n+1)/(1.+node.visits)))

    def act(self, obs, fixed_simulations=None):
        self.history.update(obs)
        self.current_obs = obs
        self.priors, self.prior_ms = None, 0.
        answer = super().act(obs, fixed_simulations)
        self.last_stats.update(prior_kind=self.prior_kind, prior_ms=self.prior_ms,
            root_prior=[] if self.priors is None else self.priors.tolist(), cpuct=1.5,
            prior_scope='root only; uniform PUCT at deeper public-history nodes')
        return answer
