import copy
import unittest

import numpy as np

from solvers.local_pursuit import Observation, Controller, PublicMap
from solvers.multimodal_estimator import Estimator, export_modes
from solvers.estimator_particle_reference import ParticleEstimator
from verify_l45_occlusion_scenarios import movement, visible


class EstimatorTests(unittest.TestCase):
    def test_instrumented_particle_baseline_exactly_matches_historical_filter(self):
        p = PublicMap([[[-1., 1.], [-5., 5.]]], 1.1)
        old, traced = Controller(p, "G", 91), ParticleEstimator(p.boxes, 91)
        history = [Observation(0, np.array([-3., 6.]), np.array([2., 6.]), 80)]
        history += [Observation(t, np.array([-3., 6.-t]), None, 80-t) for t in range(1, 7)]
        for obs in history:
            old.update(obs, np.zeros(2, np.int64))
            traced.update(obs)
            np.testing.assert_array_equal(old.particles, traced.particles)
            np.testing.assert_array_equal(old.filter_rng, traced.rng)

    def test_modes_keep_probability_mass_when_weak_modes_get_samples(self):
        w = np.zeros((100, 9))
        w[3, 0], w[80, 1], w[50, 2] = .998, .001, .001
        ids, weights, labels = export_modes(w, np.array([1], np.int64), 64)
        self.assertEqual(len(ids), 64)
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertAlmostEqual(weights[labels == 1].sum(), .001)
        self.assertAlmostEqual(weights[labels == 2].sum(), .001)
        self.assertTrue(np.all(w[ids, labels] > 0))

    def test_shared_support_legal_paths_and_observational_invariance(self):
        boxes = [[[-1., 1.], [-5., 5.]]]
        methods = {m: Estimator(boxes, m, 17) for m in ("diffuse", "persistent", "feasible")}
        history = [Observation(0, np.array([-3., 6.]), np.array([2., 6.]), 80)]
        history += [Observation(t, np.array([-3., 6.-t]), None, 80-t) for t in range(1, 7)]
        for obs in history:
            for own in methods.values():
                own.update(copy.deepcopy(obs))
            if obs.target is None:
                np.testing.assert_array_equal(methods['diffuse'].reachable, methods['persistent'].reachable)
                np.testing.assert_array_equal(methods['feasible'].reachable, methods['persistent'].reachable)
        own = methods['persistent']
        saved = own.snapshot()
        obs = Observation(7, np.array([-3., -.5]), None, 73)
        a, aw = own.update(obs)
        first_dist = own.distribution()
        own.restore(saved)
        b, bw = own.update(copy.deepcopy(obs))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(aw, bw)
        np.testing.assert_array_equal(first_dist[1], own.distribution()[1])
        for path in own.witnesses():
            for t, observer in enumerate(own.observers):
                self.assertTrue(movement(path[t].tolist(), path[t+1].tolist(), 1., boxes))
                self.assertFalse(visible(observer.tolist(), path[t+1].tolist(), boxes))

    def test_impossible_history_and_hidden_initialization_are_rejected(self):
        own = Estimator([], "persistent", 3)
        with self.assertRaises(ValueError):
            own.update(Observation(0, np.zeros(2), None, 80))
        own.update(Observation(0, np.zeros(2), np.array([2., 0.]), 80))
        with self.assertRaises(RuntimeError):
            own.update(Observation(1, np.zeros(2), None, 79))


if __name__ == "__main__":
    unittest.main()
