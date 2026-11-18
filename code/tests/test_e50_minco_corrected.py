import unittest
import numpy as np

from problems.example8 import ClosedLoopMincoPlanner
from solvers.e50_minco_corrected import (CorrectedClosedLoopMincoPlanner,
                                         free_terminal_gradient)


class E50CorrectedMincoTests(unittest.TestCase):
    def test_corrected_free_terminal_condition_is_zero(self):
        planner = CorrectedClosedLoopMincoPlanner(.1, .1)
        old = ClosedLoopMincoPlanner(.1, .1)
        q = np.zeros((6, 2)); q[4] = [1., -0.5]
        old_coeff = np.linalg.solve(old.construct_minco_m2(1.), q)
        new_coeff = np.linalg.solve(planner.construct_minco_m2(1.), q)
        row = free_terminal_gradient(planner, 1.)
        self.assertGreater(float(np.max(np.abs(row @ old_coeff))), 1e-6)
        self.assertLess(float(np.max(np.abs(row @ new_coeff))), 1e-10)

    def test_corrected_planner_is_stable_and_distinct_from_legacy(self):
        old = ClosedLoopMincoPlanner(.1, .1)
        new = CorrectedClosedLoopMincoPlanner(.1, .1)
        self.assertLess(max(abs(np.linalg.eigvals(new.mat_F_stab))), 1.)
        self.assertGreater(float(np.max(np.abs(old.mat_F_stab-new.mat_F_stab))), 1e-8)


if __name__ == '__main__': unittest.main()
