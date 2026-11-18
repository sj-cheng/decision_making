import unittest
import numpy as np

from solvers.e50_candidate_adapter import plan_action
from solvers.e50_interfaces import JointPlan


class E50CandidateAdapterTests(unittest.TestCase):
    def test_plan_action_matches_legacy_endpoint_semantics(self):
        starts = np.array([[1., 2.], [3., 4.]])
        goals = np.array([[2., 2.], [3., 5.]])
        plan = JointPlan(np.stack((goals, goals), axis=1), 1.)
        np.testing.assert_allclose(plan_action(plan, starts), [[1., 0.], [0., 1.]])


if __name__ == '__main__':
    unittest.main()
