import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from learning.e45m_task_adapter import episode_arrays
from learning.e45o_demonstration_replay import DemonstrationReplay
from tests.test_e45m_hasac_task import synthetic_episode


class DemonstrationReplayTests(unittest.TestCase):
    def data(self, marker):
        samples, trace = synthetic_episode()
        data = episode_arrays(samples, trace)
        data['reward'][:] = marker
        return data

    def test_exact_mix_and_protected_demonstrations_after_ring_overwrite(self):
        demo = self.data(7)
        replay = DemonstrationReplay(demo, capacity=5, seed=23)
        for _ in range(6):
            replay.add(self.data(-3))
        batch = replay.sample(32)
        self.assertEqual(int((batch[4] == 7).sum()), 8)
        self.assertEqual(int((batch[4] == -3).sum()), 24)
        np.testing.assert_array_equal(replay.demonstrations['reward'], demo['reward'])
        with self.assertRaises(ValueError):
            replay.demonstrations['reward'][0] = 0

    def test_checkpoint_preserves_both_sampling_sources_and_shuffle(self):
        replay = DemonstrationReplay(self.data(7), capacity=5, seed=23)
        replay.add(self.data(-3)); replay.add(self.data(-3))
        replay.sample(32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'replay.npz'; replay.save(path)
            other = DemonstrationReplay(self.data(7), capacity=5, seed=0)
            other.load(path, copy.deepcopy(replay.rng.bit_generator.state))
            for _ in range(3):
                for a, b in zip(replay.sample(32), other.sample(32)):
                    if a is not None:
                        np.testing.assert_array_equal(a, b)

    def test_retirement_and_terminal_semantics_survive_mixing(self):
        replay = DemonstrationReplay(self.data(7), seed=23)
        replay.add(self.data(-3))
        batch = replay.sample(256)
        np.testing.assert_array_equal(batch[1][:, :, -1:], batch[6])
        self.assertTrue(np.all(batch[2][batch[6][..., 0] == 0] == 0))
        terminal = batch[5][:, 0] == 1
        self.assertTrue(np.all(batch[9][:, terminal] == 0))
        self.assertTrue(np.all(batch[8][terminal] == 0))


if __name__ == '__main__':
    unittest.main()
