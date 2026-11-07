"""Independent probability checks for the search-value study's public model.

Tree conditioning deliberately uses the exact simulated sensor and Bayesian
mode conditioning.  It is not the frozen D estimator's guarded negative mask
or its positive-measurement mode reset; the real root estimator stays frozen.
"""
import copy
from pathlib import Path
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from search_value_model import (  # noqa: E402
    ModelMatchedTarget, condition_joint, predict_joint, root_joint)
from solvers.frozen_d_search_interface import sensor_visible  # noqa: E402


class SearchValueProbabilityTests(unittest.TestCase):
    def test_root_export_accumulates_duplicate_ids_and_conditional_modes(self):
        # Different export strata may select the same spatial lattice point.
        # The mixture's mass must accumulate, rather than be overwritten.
        ids = np.tile(np.array([0, 0, 1, 2], dtype=np.int64), 16)
        sample_weights = np.arange(1., 65.)
        sample_weights /= sample_weights.sum()
        density = np.zeros((3, 9))
        density[0, :2] = [.1, .3]
        density[1, [2, 8]] = [.2, .3]
        density[2, [1, 7]] = [.08, .02]
        estimator = types.SimpleNamespace(
            graph=(np.zeros((3, 2)),), ids=ids.copy(),
            particle_weights=sample_weights.copy(), weights=density.copy())
        expected = np.zeros_like(density)
        for sample, point in enumerate(ids):
            expected[point] += (sample_weights[sample] * density[point]
                                / density[point].sum())
        actual = root_joint(estimator)
        np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-16)
        self.assertAlmostEqual(float(actual.sum()), 1., places=14)
        np.testing.assert_array_equal(estimator.weights, density)
        np.testing.assert_array_equal(estimator.particle_weights, sample_weights)

    def test_prediction_matches_independent_joint_transition_matrix(self):
        links = np.array([[0, 1], [0, 2], [1, 3], [2, 3]], np.int32)
        lengths = np.full(4, 2, np.int32)
        probabilities = np.empty((4, 2, 9))
        for position in range(4):
            probabilities[position, 0] = .1 + .1 * (np.arange(9) % 5)
            probabilities[position, 1] = 1. - probabilities[position, 0]
        joint = np.arange(1., 37.).reshape(4, 9)
        joint /= joint.sum()
        # Build P((j,n) | (i,m)) directly; this avoids reusing the
        # implementation's predict-modes-then-propagate factorization.
        transition = np.zeros((36, 36))
        for i in range(4):
            for old_mode in range(9):
                for k in range(int(lengths[i])):
                    j = int(links[i, k])
                    for new_mode in range(9):
                        switch = .1 / 9. + .9 * (old_mode == new_mode)
                        transition[9*i+old_mode, 9*j+new_mode] += (
                            switch * probabilities[i, k, new_mode])
        expected = (joint.ravel() @ transition).reshape(4, 9)
        before = joint.copy()
        actual = predict_joint(joint, links, lengths, probabilities)
        np.testing.assert_allclose(transition.sum(axis=1), 1., atol=5e-16)
        np.testing.assert_allclose(actual, expected, rtol=0., atol=5e-16)
        np.testing.assert_array_equal(joint, before)
        self.assertAlmostEqual(float(actual.sum()), 1., places=14)

    def geometry_fixture(self):
        observer = np.array([-3., 0.])
        points = np.array([[-2., 0.], [2., 0.], [-3., 7.], [-3., 1.]])
        boxes = np.array([[[-1., 1.], [-2., 2.]]])
        prediction = np.zeros((4, 9))
        prediction[0, [0, 4]] = [.08, .02]
        prediction[1, [1, 2]] = [.15, .05]
        prediction[2, [3, 5]] = [.06, .24]
        prediction[3, [6, 8]] = [.28, .12]
        return observer, points, boxes, prediction

    def test_negative_observation_conditions_exact_visibility_and_probability(self):
        at, points, boxes, prediction = self.geometry_fixture()
        visible = np.array([sensor_visible(at, p, boxes) for p in points])
        np.testing.assert_array_equal(visible, [True, False, False, True])
        expected = prediction.copy()
        expected[visible] = 0.
        expected /= expected.sum()
        before = prediction.copy()
        actual = condition_joint(prediction, at, points, boxes, -1)
        np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-16)
        np.testing.assert_array_equal(actual[visible], np.zeros((2, 9)))
        np.testing.assert_array_equal(prediction, before)

    def test_positive_observation_reveals_position_but_not_latent_mode(self):
        at, points, boxes, prediction = self.geometry_fixture()
        actual = condition_joint(prediction, at, points, boxes, 0)
        expected = np.zeros_like(prediction)
        expected[0] = prediction[0] / prediction[0].sum()
        np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-16)
        self.assertEqual(int(np.count_nonzero(actual.sum(axis=1))), 1)
        self.assertEqual(int(np.count_nonzero(actual[0])), 2)
        self.assertAlmostEqual(float(actual[0, 0]), .8)
        # Neither the sampled simulator mode nor a one-hot state label is
        # available to this function: both compatible modes must remain.
        self.assertAlmostEqual(float(actual[0, 4]), .2)

    def test_impossible_observations_fail_instead_of_inventing_support(self):
        at, points, boxes, prediction = self.geometry_fixture()
        only_visible = np.zeros_like(prediction)
        only_visible[0, 0] = 1.
        with self.assertRaises((ValueError, RuntimeError)):
            condition_joint(only_visible, at, points, boxes, -1)
        no_mass_at_observation = prediction.copy()
        no_mass_at_observation[0] = 0.
        with self.assertRaises((ValueError, RuntimeError)):
            condition_joint(no_mass_at_observation, at, points, boxes, 0)
        # Position 1 is behind the wall and cannot be a positive reading.
        with self.assertRaises((ValueError, RuntimeError)):
            condition_joint(prediction, at, points, boxes, 1)


class ModelMatchedTargetTests(unittest.TestCase):
    def packet(self):
        return dict(
            map=[[[-1., 1.], [-5., 5.]]],
            motion=dict(pursuer_axis_limit=1.1, evader_axis_limit=1.,
                        dt=1., horizon=80, capture_radius=1.),
            history=[
                dict(step=0, position=[-3., 6.], target=[2., 6.], remaining=80),
                dict(step=1, position=[-3., 5.], target=None, remaining=79)])

    def test_fixed_seed_replay_does_not_use_future_observer_positions(self):
        packet = self.packet()
        before = copy.deepcopy(packet)
        left = ModelMatchedTarget(packet, seed=215, draw_index=3)
        right = ModelMatchedTarget(copy.deepcopy(packet), seed=215, draw_index=3)
        np.testing.assert_array_equal(left.position, right.position)
        previous = np.asarray(left.position).copy()
        for _ in range(16):
            a = np.asarray(left.step(own_position_next=np.array([-3., 4.]))).copy()
            b = np.asarray(right.step(own_position_next=np.array([-4., 3.]))).copy()
            np.testing.assert_array_equal(a, b)
            self.assertLessEqual(float(np.max(np.abs(a-previous))), 1.+1e-10)
            self.assertTrue(np.all(a >= -10.-1e-10))
            self.assertTrue(np.all(a <= 10.+1e-10))
            np.testing.assert_array_equal(left.position, a)
            np.testing.assert_array_equal(right.position, b)
            previous = a
        self.assertEqual(packet, before)


if __name__ == '__main__':
    unittest.main()
