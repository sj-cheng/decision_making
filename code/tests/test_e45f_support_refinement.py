import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

import e45c_belief_diagnostics as diagnostic
from solvers.e45f_support_refinement import RefinedTeamBelief, refined_posterior, lattice, motion_probabilities
from solvers.e45_team_adapter import team_hidden_mask
from solvers.frozen_d_search_interface import PublicMap
from l45_corner_shared_20261025.multimodal_estimator import make_graph
from l45_corner_shared_20261025.corner_estimator import map_probabilities


class SupportRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((Path(__file__).parent / 'fixtures/e45f_empty_support_public.json').read_text())

    @patch.object(diagnostic, 'EVIDENCE', None)
    def test_real_failure_recovers_public_legal_paths_and_replays_exactly(self):
        d = self.fixture
        coarse = diagnostic.RecordedTeamBelief(d['boxes'], d['seed'])
        coarse.update(d['history'][0])
        with self.assertRaisesRegex(RuntimeError, 'Empty team-conditioned'):
            coarse.update(d['history'][1])
        signatures = []
        for _ in range(2):
            fine = RefinedTeamBelief(d['boxes'], d['seed'])
            for obs in d['history']:
                fine.update(obs)
            self.assertEqual(fine.refinements, [dict(step=79, target=1, spacing=.25, support_points=2)])
            e = fine.estimators[1]
            self.assertAlmostEqual(e.weights.sum(), 1.)
            self.assertTrue(team_hidden_mask(e.particles, np.asarray(obs['positions']),
                                           np.asarray(obs['active_p']), np.asarray(d['boxes'])).all())
            for i in np.flatnonzero(e.reachable):
                parent = e.parents[-1][i]
                self.assertEqual(parent, e.graph[5])
                self.assertIn(i, e.graph[1][parent, :e.graph[2][parent]])
            signatures.append(fine.signature())
        self.assertEqual(*signatures)

    @patch.object(diagnostic, 'EVIDENCE', None)
    def test_successful_path_preserves_weights_rng_and_exports(self):
        from run_e45_team_pilot import make_case
        case = make_case('long_wall', 0)
        old = diagnostic.RecordedTeamBelief(case['map'], 17)
        new = RefinedTeamBelief(case['map'], 17)
        for obs in case['history']:
            old.update(obs); new.update(obs)
            self.assertEqual(old.signature(), new.signature())
            for a, b in zip(old.estimators, new.estimators):
                if a.weights is not None:
                    np.testing.assert_array_equal(a.weights, b.weights)
                np.testing.assert_array_equal(a.rng, b.rng)
        self.assertEqual(new.refinements, [])

    def test_generalized_graph_and_kernel_match_frozen_at_original_spacing(self):
        boxes = np.zeros((0, 2, 2)); last = np.array([.13, -.17])
        old = make_graph(last, boxes)
        points, links, lengths, _, root = lattice(last, boxes, .5)
        np.testing.assert_array_equal(points, old[0])
        np.testing.assert_array_equal(lengths, old[2])
        self.assertEqual(root, old[5])
        public = PublicMap(boxes, 1.)
        expected, means, _ = map_probabilities(*old[:3], last, boxes, public.vertices, public.dist)
        actual = motion_probabilities(points, links, lengths, means)
        for i, n in enumerate(lengths):
            np.testing.assert_array_equal(links[i, :n], old[1][i, :n])
            np.testing.assert_allclose(actual[i, :n], expected[i, :n], rtol=1e-14, atol=0)

    def test_impossible_observations_stay_invalid_and_no_positive_is_invented(self):
        import copy
        d = copy.deepcopy(self.fixture)
        d['history'][1]['positions'] = [d['history'][0]['reports'][1][0]] * 2
        self.assertIsNone(refined_posterior(np.zeros((0, 2, 2)), d['history'], 1))
        self.assertIsNone(refined_posterior(np.zeros((0, 2, 2)), [d['history'][1]], 1))


if __name__ == '__main__':
    unittest.main()
