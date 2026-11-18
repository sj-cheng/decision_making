import unittest
import numpy as np

from solvers.e50_minco_math_audit import (audit_closed_loop, audit_factory_without_cost,
                                          audit_parameter_grid, audit_polytraj_boundary)


class E50MincoMathAuditTests(unittest.TestCase):
    def test_polytraj_boundary_conditions(self):
        row = audit_polytraj_boundary()
        np.testing.assert_allclose(row['position_start'], [0., 0.], atol=1e-12)
        np.testing.assert_allclose(row['position_end'], [1., -2.], atol=1e-12)
        self.assertLess(max(abs(x) for x in row['velocity_end']), 1e-10)

    def test_default_closed_loop_is_stable_but_not_one_step_exact(self):
        row = audit_closed_loop()
        self.assertLess(row['spectral_radius'], 1.)
        self.assertLess(row['endpoint'][0], .5)

    def test_grid_audit_reports_default_and_edge_cases(self):
        rows = audit_parameter_grid()
        self.assertEqual(len(rows), 12)
        self.assertTrue(any(r['matrix_condition'] > 1e6 for r in rows))

    def test_unused_factory_requires_cost_setup(self):
        row = audit_factory_without_cost()
        self.assertEqual(row['status'], 'error')


if __name__ == '__main__': unittest.main()
