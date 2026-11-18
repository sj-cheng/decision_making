import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'ros2' / 'e50_runtime_bridge'))
from e50_runtime_bridge.node import decode_command, decode_belief


class E50RosBridgeTests(unittest.TestCase):
    def test_command_decode_is_finite_and_two_dimensional(self):
        class M: data = [1., -2.]
        np.testing.assert_allclose(decode_command(M()), [1., -2.])

    def test_command_decode_rejects_bad_shape(self):
        class M: data = [1.]
        with self.assertRaises(ValueError): decode_command(M())

    def test_belief_decode_uses_one_support_particle_per_target(self):
        class M: data = [2., 1., 3., 4., .5, 1., .8]
        belief = decode_belief(M())
        self.assertEqual(belief.particles.shape, (1, 1, 2))
        self.assertTrue(bool(belief.visible[0]))


if __name__ == '__main__': unittest.main()
