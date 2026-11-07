"""Protected public-search demonstrations; unchanged author HASAC batch API."""
import numpy as np
from learning.e45m_task_adapter import Replay


class DemonstrationReplay(Replay):
    """Only online transitions enter the ring. Demonstrations never expire.

    One RNG owns both sampling sources, so the existing checkpoint RNG state
    is sufficient for exact continuation with the same frozen demonstration.
    """
    def __init__(self, demonstrations, capacity=50000, seed=17, fraction=.25):
        super().__init__(capacity, seed)
        if not 0 < fraction < 1:
            raise ValueError('Demonstration fraction must be between zero and one')
        self.demonstrations = {k: np.array(v, dtype=np.float32, copy=True)
                               for k, v in demonstrations.items()}
        self.demo_size = len(self.demonstrations['obs'])
        if not self.demo_size:
            raise ValueError('Empty demonstration set')
        if any(len(v) != self.demo_size for v in self.demonstrations.values()):
            raise ValueError('Inconsistent demonstration lengths')
        for value in self.demonstrations.values():
            value.flags.writeable = False
        self.fraction = fraction

    def sample(self, n):
        if self.size < 1 or n < 2:
            raise ValueError('Mixed replay requires online data and batch size >= 2')
        count = max(1, min(n - 1, int(round(n * self.fraction))))
        for _ in range(100):
            teacher_ids = self.rng.integers(self.demo_size, size=count)
            online_ids = self.rng.integers(self.size, size=n-count)
            order = self.rng.permutation(n)
            d = {key: np.concatenate([self.demonstrations[key][teacher_ids],
                                      value[online_ids]])[order]
                 for key, value in self.arrays.items()}
            if np.all(d['active'].sum(axis=(0, 2)) > 0):
                break
            self.resamples += 1
        else:
            raise RuntimeError('Mixed replay cannot supply active samples for both actors')
        return (d['central'], d['obs'].transpose(1, 0, 2),
                d['action'].transpose(1, 0, 2), None, d['reward'], d['done'],
                d['active'].transpose(1, 0, 2), d['done'].copy(),
                d['next_central'], d['next_obs'].transpose(1, 0, 2), None,
                np.ones((n, 1), np.float32))
