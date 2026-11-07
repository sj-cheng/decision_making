"""Public, fixed-root generative model for the search-value mechanism study.

Online estimation remains frozen D. These are Bayes planning beliefs under
one fixed root motion graph, not a replacement implementation of D's reset.
The matched target is strictly matched only before first rediscovery: after
that online D may reanchor, whereas this diagnostic target keeps its graph.
"""
import numpy as np
from numba import njit
from solvers.frozen_d_search_interface import (
    belief_view, loader, root_sample, target_step, sensor_visible, Observation)


def root_joint(estimator):
    ids, weights, modes = belief_view(estimator)
    result = np.zeros_like(estimator.weights)
    np.add.at(result, ids, weights[:, None]*modes)
    if not np.isclose(result.sum(), 1., rtol=0, atol=1e-12):
        raise ValueError('Unnormalized root distribution')
    return result


@njit(cache=True)
def predict_joint(joint, links, lengths, probabilities):
    result = np.zeros_like(joint)
    for i in range(len(joint)):
        mass = joint[i].sum()
        if mass <= 0.:
            continue
        if lengths[i] <= 0:
            raise RuntimeError('Positive mass has no legal target edge')
        for k in range(lengths[i]):
            j = links[i, k]
            for mode in range(9):
                result[j, mode] += (.9*joint[i, mode]+.1*mass/9.)*probabilities[i, k, mode]
    return result


@njit(cache=True)
def condition_joint(prediction, at, points, boxes, observed_id):
    result = np.zeros_like(prediction)
    for i in range(len(prediction)):
        if prediction[i].sum() <= 0:
            continue
        seen = sensor_visible(at, points[i], boxes)
        allowed = (seen and i == observed_id) if observed_id >= 0 else not seen
        if allowed:
            result[i] = prediction[i]
    mass = result.sum()
    if not np.isfinite(mass) or mass <= 0:
        raise RuntimeError('Impossible simulated public observation')
    return result/mass


class ModelMatchedTarget:
    """Independently drawn hidden world from a public 64-weighted root prior."""
    def __init__(self, packet, seed, draw_index=0):
        self.estimator = loader.load_estimator(packet['map'], int(seed))
        for item in packet['history']:
            obs = Observation(item['step'], np.asarray(item['position'], float),
                None if item['target'] is None else np.asarray(item['target'], float), item['remaining'])
            self.estimator.update(obs)
        self.graph = self.estimator.graph
        self.rng = np.array([(int(seed)+int(draw_index)*104729+84020261026) % 4294967296], np.int64)
        self.root_rng = self.rng.copy()
        _, self.index, self.mode = root_sample(*belief_view(self.estimator), self.rng)
        self.initial_index, self.initial_mode = int(self.index), int(self.mode)
        self.position = self.graph[0][self.index].copy()

    def step(self, own_position_next=None):
        # No opponent response: own position and just-selected own action are
        # intentionally unused. All target motion uses the registered kernel.
        points, links, lengths, probabilities, *_ = self.graph
        self.index, self.mode = target_step(self.index, self.mode, links, lengths, probabilities, self.rng)
        self.position = points[self.index].copy()
        return self.position.copy()
