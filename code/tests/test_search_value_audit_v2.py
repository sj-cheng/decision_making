"""Audit-only regression checks; experimental algorithms stay unchanged."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_search_value_validation_v2 import (  # noqa: E402
    _matched_public_state_valid, _matched_public_template,
    _matched_physical_transition)
from corner_estimator_scenarios import original_model  # noqa: E402


class SearchValueAuditV2Tests(unittest.TestCase):
    def packet(self):
        return dict(map=[[[-1., 1.], [-5., 5.]]],
                    motion=dict(pursuer_axis_limit=1.1),
                    history=[dict(step=11, position=[-3., 4.], target=None,
                                  remaining=69)])

    def test_public_template_covers_full_model_dimension_and_activity_indices(self):
        packet = self.packet()
        model = original_model(packet['map'], packet['motion']['pursuer_axis_limit'])
        state = _matched_public_template(packet, [2., 4.])
        self.assertEqual(state.shape, (model.state_dim,))
        self.assertGreater(model.state_dim, 13)
        self.assertTrue(_matched_public_state_valid(state, packet))
        np.testing.assert_array_equal(state[model.active_idxs], [1., 0., 1., 0.])
        np.testing.assert_array_equal(state[13:], np.zeros(model.state_dim-13))
        self.assertEqual(state[model.time_idx], 11)

    def test_any_extra_state_or_memory_slot_tampering_is_rejected(self):
        packet = self.packet()
        state = _matched_public_template(packet, [2., 4.])
        for index in range(13, len(state)):
            changed = state.copy()
            changed[index] = 1e-8
            self.assertFalse(_matched_public_state_valid(changed, packet), index)
        self.assertFalse(_matched_public_state_valid(state[:-1], packet))
        self.assertFalse(_matched_public_state_valid(np.r_[state, 0.], packet))

    def floating_edge(self):
        current = np.array([-8.806922577118115, -8.919171095033583])
        graph_goal = np.array([-8.806922577118115, -7.919171095033582])
        raw = graph_goal-current
        endpoint = current+np.clip(raw, -1., 1.)
        return current, graph_goal, raw, endpoint

    def test_roundoff_edge_is_clipped_and_exact_physical_endpoint_is_accepted(self):
        current, goal, raw, endpoint = self.floating_edge()
        self.assertGreater(raw[1], 1.)
        self.assertNotEqual(endpoint[1], goal[1])
        self.assertEqual(endpoint[1], np.nextafter(goal[1], -np.inf))
        actual = _matched_physical_transition(current, goal, raw, endpoint)
        np.testing.assert_array_equal(actual, endpoint)

    def test_physical_endpoint_tampering_is_rejected_without_numeric_tolerance(self):
        current, goal, raw, endpoint = self.floating_edge()
        for delta in (1e-8, np.spacing(endpoint[0])):
            changed = endpoint.copy()
            changed[0] += delta
            with self.assertRaises(ValueError):
                _matched_physical_transition(current, goal, raw, changed)

    def test_raw_command_is_checked_exactly_even_if_clipping_hides_the_change(self):
        current, goal, raw, endpoint = self.floating_edge()
        changed = raw.copy()
        changed[1] = np.nextafter(changed[1], np.inf)
        np.testing.assert_array_equal(np.clip(changed, -1., 1.), np.clip(raw, -1., 1.))
        with self.assertRaises(ValueError):
            _matched_physical_transition(current, goal, changed, endpoint)


if __name__ == '__main__':
    unittest.main()
