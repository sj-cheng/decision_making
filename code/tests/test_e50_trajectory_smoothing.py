import unittest
import numpy as np

from solvers.e50_trajectory_smoothing import rate_limit_displacement


class E50TrajectorySmoothingTests(unittest.TestCase):
    def test_small_changes_are_preserved(self):
        previous = np.zeros((2, 2))
        desired = np.array([[.2, .1], [-.1, .2]])
        np.testing.assert_allclose(rate_limit_displacement(desired, previous, max_delta=.6), desired)

    def test_large_reversal_is_rate_limited_per_robot(self):
        previous = np.array([[1., 0.], [0., 0.]])
        desired = np.array([[-1., 0.], [0., .2]])
        result = rate_limit_displacement(desired, previous, max_delta=.6)
        self.assertAlmostEqual(np.linalg.norm(result[0] - previous[0]), .6)
        np.testing.assert_allclose(result[1], desired[1])


if __name__ == '__main__':
    unittest.main()
