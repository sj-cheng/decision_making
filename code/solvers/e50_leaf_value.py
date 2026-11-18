"""Leaf-value model for distributed E50 search.

The model consumes only AgentObservation and a candidate endpoint.  It is a
small numpy model so the deployment boundary does not depend on torch; a
checkpoint can be trained offline with :meth:`fit` and saved as an ``npz``.
"""
from pathlib import Path
import numpy as np


class LeafValueNet:
    """Linear leaf-value estimator with a public, versioned feature schema."""
    SCHEMA = 'e50_leaf_value_v2'

    def __init__(self, weights=None, bias=0.0):
        self.weights = None if weights is None else np.asarray(weights, float).copy()
        self.bias = float(bias)
        self.calls = 0
        if self.weights is not None and (self.weights.ndim != 1 or not np.isfinite(self.weights).all()):
            raise ValueError('invalid leaf-value weights')

    @staticmethod
    def features(observation, endpoint):
        endpoint = np.asarray(endpoint, float)
        if endpoint.shape != (2,) or not np.isfinite(endpoint).all():
            raise ValueError('endpoint must have shape (2,)')
        pose = np.asarray(observation.pose, float)
        velocity = np.asarray(observation.velocity, float)
        values = list(pose) + list(velocity) + [np.sin(observation.heading), np.cos(observation.heading)]
        values += list(endpoint - pose)
        peers = tuple(observation.peer_messages)
        if peers:
            peer = peers[0]
            values += list(np.asarray(peer.pose) - pose)
            values += list(np.asarray(peer.velocity))
            if peer.intent is None:
                values += [0., 0.]
            else:
                values += list(np.asarray(peer.intent) - pose)
            values += [float(max(0, observation.cycle - peer.cycle))]
        else:
            values += [0.] * 7
        # Belief is intentionally treated as a public summary.  We do not
        # assume a concrete class here, so only finite means/weights are used.
        belief = getattr(observation, 'belief', None)
        particles = getattr(belief, 'particles', None)
        weights = getattr(belief, 'weights', None)
        if particles is not None and weights is not None:
            p = np.asarray(particles, float).reshape(-1, 2)
            w = np.asarray(weights, float).reshape(-1)
            if len(p) and len(w) == len(p) and np.sum(w) > 0:
                w = w / np.sum(w)
                mean = np.sum(p * w[:, None], axis=0)
                values += list(mean - pose) + [float(np.sum(w * np.linalg.norm(p - mean, axis=1)))]
                values += [float(np.min(np.linalg.norm(p - endpoint, axis=1)))]
            else:
                values += [0., 0., 0., 0.]
        else:
            values += [0., 0., 0., 0.]
        return np.asarray(values, dtype=float)

    def __call__(self, observation, endpoint):
        x = self.features(observation, endpoint)
        self.calls += 1
        if self.weights is None:
            # Deterministic public-information prior used before training.
            # When a target belief exists, prefer endpoints close to its
            # weighted mean.  With no detection, give local exploration a
            # small positive value; a pure ``-distance(endpoint, pose)``
            # prior makes every unobserved vehicle choose a zero action.
            belief = getattr(observation, 'belief', None)
            particles = getattr(belief, 'particles', None)
            weights = getattr(belief, 'weights', None)
            if particles is not None and weights is not None:
                p = np.asarray(particles, float).reshape(-1, 2)
                w = np.asarray(weights, float).reshape(-1)
                if len(p) and len(w) == len(p) and np.sum(w) > 0:
                    w = w / np.sum(w)
                    # For 2v2 pursuit, the weighted mean can lie between
                    # evaders and make both pursuers converge on empty space.
                    # Score the nearest visible belief support instead, while
                    # retaining a small motion regularizer.
                    distances = np.linalg.norm(p - np.asarray(endpoint), axis=1)
                    return float(-np.min(distances)
                                 - .02 * np.linalg.norm(np.asarray(endpoint) - np.asarray(observation.pose)))
            return float(.10 * np.linalg.norm(np.asarray(endpoint) - np.asarray(observation.pose)))
        if len(self.weights) != len(x):
            raise ValueError('leaf-value checkpoint feature dimension mismatch')
        belief = getattr(observation, 'belief', None)
        particles = getattr(belief, 'particles', None)
        weights = getattr(belief, 'weights', None)
        if particles is None or weights is None or len(np.asarray(particles).reshape(-1, 2)) == 0:
            # Learned supervision starts when an observation has target
            # support.  Keep the proven frontier exploration prior otherwise.
            return .10 * float(np.linalg.norm(np.asarray(endpoint) - np.asarray(observation.pose)))
        score = float(x @ self.weights + self.bias)
        # A trained pursuit value is only supervised on observed belief
        # support.  Preserve the deployment exploration behavior when that
        # support is empty, otherwise the linear model can extrapolate a
        # slightly higher value for the zero endpoint and freeze both agents.
        return score

    @classmethod
    def fit(cls, observations, endpoints, labels, ridge=1e-4):
        rows = [cls.features(o, e) for o, e in zip(observations, endpoints)]
        x = np.asarray(rows, float); y = np.asarray(labels, float).reshape(-1)
        if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('invalid value training data')
        reg = float(ridge) * np.eye(x.shape[1]); w = np.linalg.solve(x.T @ x + reg, x.T @ y)
        return cls(w, 0.)

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, schema=np.array(self.SCHEMA), weights=self.weights,
                            bias=np.array(self.bias))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            schema = str(data['schema'])
            if schema != cls.SCHEMA:
                raise ValueError('unsupported leaf-value schema')
            return cls(data['weights'], float(data['bias']))
