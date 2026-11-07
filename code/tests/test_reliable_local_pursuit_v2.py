"""The v1 contract plus witnessed, observation-only recovery regressions."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import test_reliable_local_pursuit as contract
import solvers.reliable_local_pursuit_v2 as repaired
from solvers import history_recovery
from solvers.local_pursuit import Observation, PublicMap
from verify_l45_occlusion_scenarios import movement, visible


class WitnessedRecoveryTests(contract.ReliabilityTests):
    def setUp(self):
        self.binding = patch.object(contract, "repaired", repaired)
        self.binding.start()

    def tearDown(self):
        self.binding.stop()

    def check_witnesses(self, last, observers, boxes, particles, paths, n):
        self.assertEqual(n, 64)
        np.testing.assert_array_equal(paths[:, 0], np.tile(last, (n, 1)))
        np.testing.assert_array_equal(paths[:, -1], particles)
        for path in paths:
            for t, observer in enumerate(observers):
                self.assertTrue(movement(path[t].tolist(), path[t+1].tolist(), 1., boxes.tolist()))
                self.assertFalse(visible(observer.tolist(), path[t+1].tolist(), boxes.tolist()))

    def test_original_boundary_sliver_has_legal_ancestral_witnesses(self):
        fixture = json.loads((Path(__file__).parent/"fixtures/particle_recovery_boundary_sliver.json").read_text())
        last, observers, boxes = (np.array(fixture[k]) for k in ("last", "observers", "boxes"))
        for spacing in history_recovery.RESOLUTIONS:
            args = (last, observers, boxes, spacing)
            a, pa, n = history_recovery.lattice_paths(*args, np.array([17], np.int64), np.zeros(6, np.int64))
            b, pb, m = history_recovery.lattice_paths(*args, np.array([17], np.int64), np.zeros(6, np.int64))
            self.check_witnesses(last, observers, boxes, a, pa, n)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(pa, pb)
            self.assertEqual(n, m)
            self.assertTrue(np.any(a[:, 1] == -10.), "Exact arena boundary must be retained")

    def test_impossible_history_stays_failure_not_invented_support(self):
        last = np.zeros(2)
        observers = np.zeros((1, 2))
        boxes = np.empty((0, 2, 2))
        for spacing in history_recovery.RESOLUTIONS:
            _, _, n = history_recovery.lattice_paths(last, observers, boxes, spacing,
                np.array([1], np.int64), np.zeros(6, np.int64))
            self.assertEqual(n, 0, "All one-step targets are visible; empty observation is impossible")
        _, n = history_recovery.recover(last, observers, boxes, np.array([1], np.int64), np.zeros(6, np.int64))
        self.assertEqual(n, 0)

    def test_every_sampled_ancestral_path_rejects_corner_grazing(self):
        fixture = json.loads((Path(__file__).parent/"fixtures/particle_recovery_corner_contact.json").read_text())
        last, observers, boxes = (np.array(fixture[k]) for k in ("last", "observers", "boxes"))
        particles, paths, n = history_recovery.lattice_paths(last, observers, boxes, .5,
            np.array([17], np.int64), np.zeros(6, np.int64), number=42*42)
        self.assertEqual(n, 42*42)
        # All endpoints, not a truth-selected example: independently check
        # every complete path in this public-history geometry regression.
        for path in paths:
            for t, observer in enumerate(observers):
                self.assertTrue(movement(path[t].tolist(), path[t+1].tolist(), 1., boxes.tolist()))
                self.assertFalse(visible(observer.tolist(), path[t+1].tolist(), boxes.tolist()))

    def test_full_history_not_just_latest_empty_observation(self):
        last = np.array([2., 6.])
        observers = np.array([[-3., 5.], [-3., 4.], [-3., 3.]])
        boxes = np.array([[[-1., 1.], [-5., 5.]]])
        particles, paths, n = history_recovery.lattice_paths(last, observers, boxes, .5,
            np.array([29], np.int64), np.zeros(6, np.int64))
        self.check_witnesses(last, observers, boxes, particles, paths, n)

    def test_same_recovery_is_used_for_real_G_and_T(self):
        p = PublicMap([[[-1., 1.], [-5., 5.]]], 1.1)
        controllers = [repaired.Controller(p, method, 123, deadline_ms=0.) for method in ("G", "T")]
        packets = [Observation(0, np.array([-3., 6.]), np.array([2., 6.]), 80),
                   Observation(1, np.array([-3., 5.]), None, 79),
                   Observation(2, np.array([-3., 4.]), None, 78)]
        for obs in packets:
            for controller in controllers:
                controller.update(obs, np.zeros(6, np.int64))
            np.testing.assert_array_equal(controllers[0].particles, controllers[1].particles)
            np.testing.assert_array_equal(controllers[0].filter_rng, controllers[1].filter_rng)

    def test_float32_recorded_motion_tolerance_is_not_an_action_limit_change(self):
        p = PublicMap([], 1.3)
        c = repaired.Controller(p, "G", 17, deadline_ms=0.)
        c.update(Observation(0, np.zeros(2), np.array([5., 0.]), 80), np.zeros(6, np.int64))
        position = np.array([float(np.float32(1.3)), 0.])
        c.update(Observation(1, position, np.array([5., 0.]), 79), np.zeros(6, np.int64))
        self.assertFalse(repaired.checked_safe(np.zeros(2), np.array([1.3+2e-6, 0.]),
                         1.3, p.boxes, np.zeros(6, np.int64)))
        with self.assertRaises(ValueError):
            c.update(Observation(2, position+np.array([1.3+8e-6, 0.]), np.array([5., 0.]), 78),
                     np.zeros(6, np.int64))

