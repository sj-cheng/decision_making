import unittest
import numpy as np

from problems.example8 import Example8
from solvers.e50_minco_audit import replay_minco, replay_minco_with_planner
from solvers.e50_minco_corrected import CorrectedClosedLoopMincoPlanner


class E50MincoPlannerSwapTests(unittest.TestCase):
    def test_corrected_planner_can_be_swapped_without_mutating_problem(self):
        p = Example8(); p.obstacles = []
        s = np.zeros((p.state_dim, 1))
        for robot, point in enumerate(([-2., 0.], [2., 0.], [-2., -2.], [2., -2.])):
            s[p.state_idxs[robot], 0] = point
        s[p.active_idxs, 0] = 1.
        a = np.zeros((p.action_dim, 1)); a[p.action_idxs[2], 0] = [1., 0.]
        old = replay_minco(p, s, a, robots=(2, 3), sample_dt=.05)
        new = replay_minco_with_planner(p, s, a, CorrectedClosedLoopMincoPlanner, robots=(2, 3), sample_dt=.05)
        self.assertFalse(old.collision); self.assertFalse(new.collision)
        self.assertGreater(float(np.max(np.abs(old.positions-new.positions))), 1e-6)
        self.assertTrue(hasattr(p, 'get_closed_loop_planner'))


if __name__ == '__main__': unittest.main()
