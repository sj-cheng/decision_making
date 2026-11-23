import unittest
from types import SimpleNamespace

import numpy as np

from solvers.e53_capture import capture_samples


class JointCaptureTest(unittest.TestCase):
    def test_competing_for_one_target_leaves_a_second_match(self):
        from dataclasses import make_dataclass
        from solvers.e54_joint_search import JointNode, successor

        poses = np.array([[0., 0.], [.002, 0.]])
        targets = np.array([[0., 0.], [.999, 0.]])
        velocities = np.array([[0., 0.], [1.5, 0.]])
        boxes = np.zeros((0, 2, 2))
        paths = np.broadcast_to(poses, (51, 2, 2)).copy()
        target_paths = targets + np.linspace(0., 1., 51)[:, None, None]*velocities
        Obs = make_dataclass('Obs', ['pose', 'belief', 'local_map'])
        before = SimpleNamespace(particles=targets, velocities=velocities, weights=np.ones(2))
        states = tuple(Obs(p, before, SimpleNamespace(boxes=boxes)) for p in poses)
        moves = []
        for i in range(2):
            prefix = paths[:, i:i+1]
            hits, _ = capture_samples(prefix, target_paths, np.ones(1,bool), np.ones(2,bool), boxes)
            self.assertEqual((hits >= 0).tolist(), [True, False])
            future = SimpleNamespace(predictions=target_paths[-1], captured_mask=hits >= 0)
            moves.append(SimpleNamespace(obs=Obs(poses[i], future, states[i].local_map),
                                         prefix=prefix[:,0], reward=4., prior=1.))
        child = successor(JointNode(states), moves, {})
        self.assertTrue(child.states[0].belief.terminal)
        self.assertEqual(len(child.states[1].belief.particles), 0)
