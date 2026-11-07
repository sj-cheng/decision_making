"""Targeted feature isolation, PUCT pairing, immutable D and export checks."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learning.candidate_prior import Network, features, geometry_scores, mixed_prior
from solvers.learned_candidate_prior import Controller, Direct, PublicHistory
from solvers.frozen_d_search_interface import Controller as Geometry, Observation, geometric_candidates
from audit_frozen_d_search_interface import observation, independent_traces

BOXES = [[[-1., 1.], [-5., 5.]]]
FIRST = Observation(0, np.array([-3., 6.]), np.array([2., 6.]), 80)
HIDDEN = Observation(1, np.array([-3., 5.]), None, 79)


def signature(c, a):
    return dict(action=a.tolist(), visits=c.last_stats.get('root_visits'), totals=c.last_stats.get('root_totals'),
        rng=c.rng.tolist(), traces=hashlib.sha256(b''.join(t.tobytes() for t in c.traces)).hexdigest())


def write_toy(path):
    rng = np.random.default_rng(19)
    data = dict(mean=np.zeros(64), scale=np.ones(64))
    for i, shape in enumerate(((64, 64), (64, 64), (1, 64))):
        data[f'w{i}'] = rng.normal(0., .1, shape)
        data[f'b{i}'] = np.zeros(shape[0])
    np.savez(path, **data)


class CandidatePriorTests(unittest.TestCase):
    def test_teacher_exactly_matches_strong_G(self):
        c = Geometry(BOXES, 1.1, 'G', 17)
        c.estimator.update(FIRST); expected = c.act(HIDDEN)
        e, p = c.estimator, c.public
        roots, best = geometric_candidates(HIDDEN.position, e.particles, e.particle_weights,
            e.regions[e.ids], p.speed, p.vertices, p.dist, p.boxes, c.motion_boxes)
        scores = geometry_scores(HIDDEN.position, roots, e.particles, e.particle_weights,
            p.speed, p.vertices, p.dist, p.boxes)
        np.testing.assert_array_equal(roots[scores.argmax()], best)
        np.testing.assert_array_equal(best, expected)

    def test_features_are_candidate_and_particle_permutation_equivariant(self):
        samples = np.array([[5., 2.], [5., -2.], [7., 0.]])
        weights = np.array([.2, .3, .5]); actions = np.array([[0., 1.], [1., 0.], [-1., 0.]])
        args = (np.zeros(2), actions, samples, weights, np.asarray(BOXES), 1.1, 60,
                np.array([4., 4.]), 10, np.array([0., -1.]))
        x = features(*args)
        self.assertEqual(x.shape, (3, 64)); self.assertTrue(np.isfinite(x).all())
        reversed_candidates = features(args[0], actions[::-1], *args[2:])
        np.testing.assert_array_equal(x[::-1], reversed_candidates)
        reversed_particles = features(args[0], actions, samples[::-1], weights[::-1], *args[4:])
        np.testing.assert_allclose(x, reversed_particles, atol=1e-15)
        changed_weights = features(args[0], actions, samples, weights[::-1], *args[4:])
        self.assertGreater(np.max(np.abs(x-changed_weights)), .01)

    def test_no_private_truth_or_teacher_score_parameter(self):
        expected = ['at', 'actions', 'samples', 'weights', 'boxes', 'speed', 'remaining',
                    'last_positive', 'positive_age', 'previous_motion']
        self.assertEqual(list(inspect.signature(features).parameters), expected)
        self.assertEqual(list(inspect.signature(Controller.act).parameters), ['self', 'obs', 'fixed_simulations'])
        self.assertEqual(list(inspect.signature(Direct.act).parameters), ['self', 'obs', 'fixed_simulations'])

    def test_prior_floor_survives_extreme_logits(self):
        p = mixed_prior(np.array([1e5, -1e5, 0.]))
        self.assertAlmostEqual(float(p.sum()), 1.)
        self.assertGreaterEqual(float(p.min()), .1/3.)

    def test_public_history_uses_only_observations(self):
        h = PublicHistory(); h.update(FIRST); h.update(HIDDEN)
        last, age, motion = h.inputs(HIDDEN)
        np.testing.assert_array_equal(last, FIRST.target)
        self.assertEqual(age, 1)
        np.testing.assert_array_equal(motion, HIDDEN.position-FIRST.position)

    def test_all_priors_preserve_D_and_fixed_count_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.npz'; write_toy(path)
            baseline = Geometry(BOXES, 1.1, 'G', 17)
            baseline.estimator.update(FIRST); baseline.act(HIDDEN)
            for prior in ('uniform', 'geometry', 'learned'):
                results = []
                for _ in range(2):
                    c = Controller(BOXES, 1.1, 17, prior, path if prior == 'learned' else None)
                    c.estimator.update(FIRST); c.history.update(FIRST)
                    a = c.act(HIDDEN, fixed_simulations=8)
                    results.append(signature(c, a))
                    self.assertEqual(sum(c.root.visits), 8)
                    for name in ('particles', 'particle_weights', 'weights', 'ids', 'rng'):
                        np.testing.assert_array_equal(getattr(c.estimator, name), getattr(baseline.estimator, name))
                    np.testing.assert_array_equal(independent_traces(np.concatenate(c.traces), c.public.boxes, 1.1), np.zeros(4))
                self.assertEqual(results[0], results[1])

    def test_export_inference_does_not_modify_inputs_or_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'model.npz'; write_toy(path)
            net = Network(path); x = np.random.default_rng(8).normal(size=(7, 64)); before = x.copy()
            a = net(x); b = net(x[::-1])
            np.testing.assert_array_equal(x, before)
            np.testing.assert_allclose(a[::-1], b, atol=1e-14)


def run_preflight(output):
    """Train/validation packets only; warm every timed path before testing."""
    from run_candidate_prior_pilot import initialize, METHODS
    manifest = json.loads((output/'bank/manifest.json').read_text())['entries']
    records = []
    for family in ('long_wall', 'wall_corner', 'multi_route'):
        entry = next(e for e in manifest if e['split'] == 'validation' and e['family'] == family)
        packet = json.loads(Path(entry['packet_path']).read_text())
        baseline, _ = initialize(packet, 'G', 719)
        obs = observation(packet['history'][-1]); baseline.act(obs)
        for method in METHODS:
            outputs = []
            for _ in range(2):
                c, _ = initialize(packet, method, 719)
                action = c.act(obs, fixed_simulations=8)
                outputs.append(signature(c, action))
                for name in ('particles', 'particle_weights', 'weights', 'ids', 'rng'):
                    np.testing.assert_array_equal(getattr(c.estimator, name), getattr(baseline.estimator, name))
                errors = independent_traces(np.concatenate(c.traces) if c.traces else np.zeros((0, 10)), c.public.boxes, c.public.speed)
                assert not np.any(errors), errors
            records.append(dict(family=family, method=method, fixed8_replay=outputs[0] == outputs[1], D_exact=True,
                simulated_geometry_passed=True))
    return dict(passed=all(r['fixed8_replay'] and r['D_exact'] for r in records), records=records,
        test_outcomes_read=False, warmed_paths=True)


if __name__ == '__main__':
    unittest.main()
