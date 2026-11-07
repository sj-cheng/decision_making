"""Independent information, exact replay and interruption checks for belief search."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solvers.search_value_belief import (
    Controller, _Node, _BudgetAbort, prepare_all_tails, export_geometry_inputs)
from solvers.frozen_d_search_interface import Controller as Original, Observation, geometric_candidates
from solvers.rollout_geometry_exact import geometric_candidates_exact
from l45_corner_shared_20261025.corner_estimator import export_regions
from audit_frozen_d_search_interface import independent_traces, observation
from search_value_model import root_joint, predict_joint

BOXES = [[[-1., 1.], [-5., 5.]]]
FIRST = Observation(0, np.array([-3., 6.]), np.array([2., 6.]), 80)
HIDDEN = Observation(1, np.array([-3., 5.]), None, 79)


def signature(c, action):
    return dict(action=action.tolist(), rng=c.rng.tolist(), belief_rng=c.belief_rng.tolist(),
                visits=c.last_stats['root_visits'], totals=c.last_stats['root_totals'],
                counts=c.last_stats['root_counts'], terms=c.last_stats['terminations'],
                traces=hashlib.sha256(b''.join(t.tobytes() for t in c.traces)).hexdigest())


def toy(feedback, visible):
    """Two legal static worlds, with one common prior and no environment API."""
    c = Controller.__new__(Controller)
    points = np.array([[5. if visible else 8., 2.], [5. if visible else 8., -2.]])
    links = np.array([[0], [1]], np.int32)
    lengths = np.ones(2, np.int32)
    probabilities = np.ones((2, 1, 9))
    c.estimator = SimpleNamespace(graph=(points, links, lengths, probabilities), regions=np.array([0, 1]))
    c.public = SimpleNamespace(speed=1.1, boxes=np.empty((0, 2, 2)))
    c.motion_boxes = c.public.boxes
    c.feedback, c.deadline, c.operation_ms = feedback, float('inf'), 2.
    c.rng = np.array([17], np.int64)
    c.belief_rng = np.array([37], np.int64)
    c.conditionings = c.positive_branches = c.negative_branches = 0
    c.filter_mass_minimum, c.minimum_ess = 1., 64.
    c.queries = c.exports = 0
    prior = np.ones((2, 9))/18.
    c.root = _Node(np.zeros(2), prior, -1, np.zeros((1, 2)), 0)
    c.nodes = [c.root]
    view = (np.repeat(np.arange(2), 32), np.ones(64)/64, np.ones((64, 9))/9)
    return c, view


class SearchValueBeliefTests(unittest.TestCase):
    def test_all_point_tail_reuse_has_identical_candidates_and_actions(self):
        bank = Path(__file__).resolve().parents[2]/'current/l45_closed_loop_D_20261026/bank'
        entries = json.loads((bank/'manifest.json').read_text())['entries']
        checked = 0
        for family in ('long_wall', 'wall_corner', 'multi_route'):
            entry = next(e for e in entries if e['family'] == family)
            packet = json.loads((bank/entry['public']).read_text())
            c = Controller(packet['map'], packet['motion']['pursuer_axis_limit'], 'FB', 17)
            for item in packet['history']:
                c.estimator.update(observation(item))
            e, p = c.estimator, c.public
            all_tails = prepare_all_tails(e.graph[0], p.vertices, p.dist, p.boxes, p.speed)
            joint = root_joint(e)
            rng = np.array([17], np.int64)
            at0 = np.asarray(packet['history'][-1]['position'])
            for _ in range(3):
                ids, weights, _ = export_regions(joint, e.regions, rng, 64)
                prepared = export_geometry_inputs(ids, weights, e.regions, e.graph[0], p.vertices, all_tails)
                for delta in (np.zeros(2), np.array([-.1, .1]), np.array([-.1, -.1])):
                    args = (at0+delta, e.graph[0][ids], weights, e.regions[ids], p.speed,
                            p.vertices, p.dist, p.boxes, c.motion_boxes)
                    old, new = geometric_candidates(*args), geometric_candidates_exact(*args, *prepared)
                    self.assertEqual(old[0].tobytes(), new[0].tobytes())
                    self.assertEqual(old[1].tobytes(), new[1].tobytes())
                    checked += 1
                joint = predict_joint(joint, e.graph[1], e.graph[2], e.graph[3])
        self.assertEqual(checked, 27)

    def test_execution_reuse_preserves_fixed_N_search_bits(self):
        for method in ('OL', 'FB'):
            outputs = []
            for reused in (False, True):
                c = Controller(BOXES, 1.1, method, 17)
                c.exact_geometry_reuse = reused
                c.estimator.update(FIRST)
                action = c.act(HIDDEN, fixed_simulations=16)
                outputs.append(signature(c, action))
            self.assertEqual(outputs[0], outputs[1])

    def test_fixed_simulation_replay_legality_and_frozen_D_isolation(self):
        expected = Original(BOXES, 1.1, 'G', 17)
        expected.estimator.update(FIRST)
        expected.estimator.update(HIDDEN)
        for method in ('OL', 'FB'):
            outputs = []
            for _ in range(2):
                c = Controller(BOXES, 1.1, method, 17)
                c.estimator.update(FIRST)
                action = c.act(HIDDEN, fixed_simulations=8)
                outputs.append(signature(c, action))
                for name in ('particles', 'particle_weights', 'weights', 'ids', 'rng'):
                    np.testing.assert_array_equal(getattr(c.estimator, name), getattr(expected.estimator, name))
                np.testing.assert_array_equal(independent_traces(np.concatenate(c.traces), c.public.boxes, 1.1), np.zeros(4))
                self.assertEqual(sum(c.last_stats['root_visits']), 8)
                self.assertEqual(c.last_stats['budget_aborted_simulations'], 0)
            self.assertEqual(outputs[0], outputs[1])

    def test_same_empty_observations_do_not_reveal_current_hidden_world(self):
        outputs = []
        for world in (0, 1):
            c, view = toy(True, False)
            seen_inputs = []
            def geometry(at, joint, observed):
                seen_inputs.append((joint.copy(), observed))
                return np.zeros(2)
            c._geometry = geometry
            with patch('solvers.search_value_belief.root_sample', return_value=(0, world, 0)):
                trace, _, reason = c._simulation(view, 2)
            self.assertEqual(reason, 1)
            self.assertEqual(seen_inputs[0][1], -1)
            outputs.append((trace[:, :4], seen_inputs[0][0]))
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])

    def test_only_FB_conditions_on_public_positive_observation(self):
        for feedback in (False, True):
            c, view = toy(feedback, True)
            received = []
            def geometry(at, joint, observed):
                received.append((joint.copy(), observed))
                return np.zeros(2)
            c._geometry = geometry
            with patch('solvers.search_value_belief.root_sample', return_value=(0, 0, 0)):
                c._simulation(view, 2)
            posterior, observed = received[0]
            if feedback:
                np.testing.assert_allclose(posterior.sum(axis=1), [1., 0.])
                self.assertEqual(observed, 0)
                self.assertEqual(c.positive_branches, 1)
            else:
                np.testing.assert_allclose(posterior.sum(axis=1), [.5, .5])
                self.assertEqual(observed, -1)
                self.assertEqual(c.conditionings, 0)

    def test_budget_interruption_never_backpropagates(self):
        c, view = toy(True, False)
        c._geometry = lambda at, joint, observed: np.zeros(2)
        calls = [0]
        def interrupted():
            calls[0] += 1
            if calls[0] >= 4:
                raise _BudgetAbort()
        with patch.object(c, '_allowed', side_effect=interrupted):
            trace, _, reason = c._simulation(view, 4)
        self.assertEqual(reason, 2)
        self.assertGreater(len(trace), 0)
        self.assertEqual(c.root.n, 0)
        self.assertEqual(int(c.root.visits.sum()), 0)
        self.assertEqual(float(c.root.totals.sum()), 0.)


if __name__ == '__main__':
    unittest.main()
