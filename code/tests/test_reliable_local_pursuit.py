"""Behavioral regressions for the isolated rollout reliability repair."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

import solvers.reliable_local_pursuit as repaired
from solvers.local_pursuit import Observation, PublicMap, proposal as historical_proposal
from verify_l45_occlusion_scenarios import movement, visible


class ReliabilityTests(unittest.TestCase):
    def test_boundary_and_wall_fallback_is_independently_legal(self):
        examples = [([], [9.95, 0.], [1., 0.]),
                    ([[[-1., 1.], [-8., 8.]]], [-1.01, 0.], [1., 0.]),
                    ([], [-9.99, 9.99], [-1., 1.])]
        old_illegal, repaired_bases = 0, 0
        for boxes, at, base in examples:
            p = PublicMap(boxes, 1.1)
            for seed in range(256):
                args = (np.array(at), np.array(base), 1., p.boxes)
                old, _ = historical_proposal(*args, np.array([seed], np.int64), np.zeros(2, np.int64))
                old_illegal += not movement(at, (np.array(at)+old).tolist(), 1., boxes)
                audit = np.zeros(4, np.int64)
                new, _ = repaired.proposal(*args, np.array([seed], np.int64), np.zeros(2, np.int64), audit)
                self.assertTrue(movement(at, (np.array(at)+new).tolist(), 1., boxes))
                repaired_bases += int(audit[0])
        self.assertGreater(old_illegal, 0, "Regression must exercise the historical defect")
        self.assertGreater(repaired_bases, 0)

    def test_legal_old_draws_keep_action_and_random_stream(self):
        p = PublicMap([], 1.1)
        for seed in range(128):
            a, b = np.array([seed], np.int64), np.array([seed], np.int64)
            args = (np.zeros(2), np.array([.2, -.1]), 1., p.boxes)
            old, guided_old = historical_proposal(*args, a, np.zeros(2, np.int64))
            new, guided_new = repaired.proposal(*args, b, np.zeros(2, np.int64), np.zeros(4, np.int64))
            np.testing.assert_array_equal(old, new)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(guided_old, guided_new)

    def test_world_step_joint_dynamics_reward_and_sensor(self):
        p = PublicMap([[[-1., 1.], [-6., 6.]]], 1.1)
        for seed in range(128):
            at, target, action = np.array([-2., 0.]), np.array([2., 0.]), np.array([0., 1.1])
            count = np.zeros(2, np.int64)
            nxt, end, reward, reason, seen = repaired.world_step(
                at, target, action, p.speed, p.boxes, np.array([seed], np.int64), count, 12,
                np.zeros(4, np.int64))
            self.assertTrue(movement(at.tolist(), nxt.tolist(), p.speed, p.boxes.tolist()))
            self.assertTrue(movement(target.tolist(), end.tolist(), 1., p.boxes.tolist()))
            np.testing.assert_array_equal(nxt, at+action)
            self.assertEqual(reason, -1)
            self.assertEqual(seen, visible(nxt.tolist(), end.tolist(), p.boxes.tolist()))
            expected = .03*np.clip((np.linalg.norm(at-target)-np.linalg.norm(nxt-end))/4., -1., 1.)
            self.assertAlmostEqual(reward, expected)
            self.assertEqual(count[1], 1)

    def test_invalid_transition_is_failure_without_state_advance(self):
        p = PublicMap([], 1.1)
        at, target = np.array([9.9, 0.]), np.array([5., 0.])
        count = np.zeros(2, np.int64)
        nxt, end, _, reason, _ = repaired.world_step(
            at, target, np.array([1.1, 0.]), 1.1, p.boxes,
            np.array([1], np.int64), count, 10, np.zeros(4, np.int64))
        self.assertEqual(reason, repaired.OWN_INVALID)
        self.assertEqual(count[1], 0)
        np.testing.assert_array_equal(at, nxt)
        np.testing.assert_array_equal(target, end)

    def test_horizon_fully_simulated_and_failure_not_backpropagated(self):
        p = PublicMap([[[-1., 1.], [-8., 8.]]], 1.1)

        def attempt(remaining):
            at, target = np.array([-2., 0.]), np.array([2., 0.])
            visits, totals = np.zeros(3, np.int64), np.zeros(3)
            terms, audit, lengths = np.zeros(5, np.int64), np.zeros(4, np.int64), np.zeros(2, np.int64)
            args = (at, np.tile(target, (64, 1)), target, np.empty((0, 2)), 1.1, remaining,
                    p.vertices, p.dist, p.boxes, np.array([3], np.int64), np.zeros(2, np.int64),
                    np.full(3, -1, np.int64), np.zeros((3, 2)), visits, totals, 1, terms, audit, lengths)
            return args, visits, totals, terms, lengths

        args, visits, _, terms, lengths = attempt(1)
        _, reason = repaired.simulation(*args)
        self.assertEqual(reason, repaired.HORIZON)
        self.assertEqual(terms[repaired.HORIZON], 1)
        self.assertEqual(lengths[0], 1)
        self.assertEqual(visits[0], 1)
        # Deliberately exhaust the filter with a stub; exercise the exact
        # Python body of the compiled kernel, without changing production.
        args, visits, totals, terms, _ = attempt(4)
        with patch.object(repaired, "predict", return_value=0), patch.object(
                repaired, "recover", return_value=(np.zeros((64, 2)), 0)):
            _, reason = repaired.simulation.py_func(*args)
        self.assertEqual(reason, repaired.BELIEF_FAILURE)
        self.assertEqual(terms[repaired.BELIEF_FAILURE], 1)
        self.assertFalse(visits.any())
        self.assertFalse(totals.any())

    def test_controller_hidden_world_invariance_visible_gate_and_resume(self):
        p = PublicMap([], 1.1)
        a, b = [repaired.Controller(p, "T", 42, simulations=4, deadline_ms=0.) for _ in range(2)]
        initial = Observation(0, np.zeros(2), np.array([5.9, 0.]), 80)
        for c in (a, b):
            c.act(copy.deepcopy(initial), fixed_simulations=4)
            self.assertEqual(c.last_stats["source"], "visible_geometry")
            self.assertEqual(c.last_stats["simulations"], 0)
        # Two unseen targets generate exactly the same public observation.
        oa = p.observe(1, np.array([-1.1, 0.]), np.array([6.5, 0.]))
        ob = p.observe(1, np.array([-1.1, 0.]), np.array([6.8, .2]))
        self.assertIsNone(oa.target)
        self.assertIsNone(ob.target)
        saved = a.snapshot()
        aa, ab = a.act(oa, fixed_simulations=4), b.act(ob, fixed_simulations=4)
        roots = copy.deepcopy(a.last_stats["root"])
        np.testing.assert_array_equal(aa, ab)
        np.testing.assert_array_equal(a.particles, b.particles)
        np.testing.assert_array_equal(a.rng, b.rng)
        self.assertEqual(a.last_stats["simulations"], 4)
        self.assertEqual(a.last_stats["terminations"]["own_invalid"], 0)
        self.assertEqual(a.last_stats["terminations"]["opponent_invalid"], 0)
        rng = a.rng.copy()
        a.restore(saved)
        np.testing.assert_array_equal(aa, a.act(oa, fixed_simulations=4))
        np.testing.assert_array_equal(rng, a.rng)
        self.assertEqual(roots, a.last_stats["root"])

    def test_failure_stops_controller_and_preserves_failure_evidence(self):
        p = PublicMap([], 1.1)
        c = repaired.Controller(p, "T", 42, simulations=4, deadline_ms=0.)
        c.act(Observation(0, np.zeros(2), np.array([5.9, 0.]), 80))
        with patch.object(c, "update", side_effect=RuntimeError("No lawful belief recovery support")):
            with self.assertRaises(repaired.SimulationReliabilityError):
                c.act(Observation(1, np.zeros(2), None, 79))
        self.assertEqual(c.last_stats["status"], "estimator_failure")
        for pos, base in (([np.nan, 0.], [0., 0.]), ([0., 0.], [np.nan, 0.])):
            if np.isfinite(pos).all():
                action, _ = repaired.proposal(np.array(pos), np.array(base), 1., p.boxes,
                    np.array([1], np.int64), np.zeros(2, np.int64), np.zeros(4, np.int64))
                self.assertTrue(np.isfinite(action).all())
            else:
                with self.assertRaises(ValueError):
                    repaired.proposal(np.array(pos), np.array(base), 1., p.boxes,
                        np.array([1], np.int64), np.zeros(2, np.int64), np.zeros(4, np.int64))
        with self.assertRaises(ValueError):
            c.act(Observation(79, np.zeros(2), None, 80))


if __name__ == "__main__":
    unittest.main()
