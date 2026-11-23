"""The state advanced inside the tree must match a feasible executed prefix."""
from types import SimpleNamespace
import unittest

import numpy as np

from solvers.e50_evader_minco import EvaderMinco
from solvers.e53_evader_hmcts import EvaderHMCTS, _Node


class MotionTest(unittest.TestCase):
    def test_tree_predicts_the_first_second_including_derivatives(self):
        boxes = np.empty((0, 2, 2))
        planner = EvaderHMCTS()
        planner.public = SimpleNamespace(boxes=boxes)
        executor = EvaderMinco(boxes, 1.5, 100, max_acceleration=1.5, continuous_speed=True)
        node = _Node(np.zeros((2, 2)), np.array([[-5., -2.], [-6., -3.]]),
                     np.array([True, False]), np.ones(2, bool),
                     np.zeros((2, 2)), np.zeros((2, 2)))
        lengths = []
        for _ in range(5):
            choices = planner._action_candidates(node)[0]
            action, _, track = next(c for c in choices if c[1] == 'escape')
            lengths.append(np.linalg.norm(action))
            predicted, pursuers, active, _, velocity, acceleration = planner._transition(node, (track, None))
            prefix, mode = executor.prefix(0, node.evaders[0], action, waypoint=True)
            self.assertEqual(mode, 'pass_through')
            for field, expected in [('positions', predicted), ('velocities', velocity),
                                    ('accelerations', acceleration)]:
                np.testing.assert_allclose(prefix[field][-1], expected[0], atol=1e-9)
            node = _Node(predicted, pursuers, active, node.active_p, velocity, acceleration)
        self.assertGreater(max(lengths), 1.5)


if __name__ == '__main__':
    unittest.main()
