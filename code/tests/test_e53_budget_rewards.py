from types import SimpleNamespace
from collections import Counter
import unittest

import numpy as np

from e49_rules import capture_pairs
from solvers.e53_capture import capture_samples
from solvers.e53_evader_hmcts import EvaderHMCTS


class BudgetRewardTests(unittest.TestCase):
    def test_discount_and_node_local_returns(self):
        nodes = [SimpleNamespace(reward=r, visits=0, total=0.) for r in (0., -100., 2.)]
        EvaderHMCTS._backup(nodes)
        self.assertAlmostEqual(nodes[1].total, -100. + .98 * 2.)
        self.assertEqual(nodes[2].total, 2.)
        self.assertEqual(nodes[0].total, nodes[1].total)
        self.assertEqual([n.visits for n in nodes], [1, 1, 1])

    def test_three_step_leaf_reward_is_sparse_and_counts_transitions(self):
        policy = EvaderHMCTS(depth=3, window=8)
        policy.events = Counter()
        seen = np.eye(2, dtype=bool)
        def node(depth, parent, visibility, distance):
            return SimpleNamespace(depth=depth, parent=parent, active_e=np.ones(2,bool),
                                   visibility=visibility, pursuers=np.array([[0.,0.],[0.,distance]]))
        root = node(0, None, seen, 3.)
        a = node(1, root, seen, 4.)
        b = node(2, a, seen, 5.)
        c = node(3, b, np.zeros_like(seen), 6.)
        self.assertEqual([policy._reward(n, []) for n in (a,b,c)], [0.,0.,16.])
        self.assertEqual(policy.events, dict(survival=1, lost_visibility=1, separation=1))
        self.assertEqual(policy._reward(c, [(0,0)]), -100.)

    def test_predicted_camera_locks_at_ten_metres_without_mutating_parent(self):
        policy = EvaderHMCTS()
        policy.public = SimpleNamespace(boxes=np.empty((0, 2, 2)))
        parent = SimpleNamespace(cameras=policy.camera_state, headings=np.zeros(2),
                                 pursuers=np.zeros((2, 2)))
        def observed(x):
            child = SimpleNamespace(pursuers=parent.pursuers.copy(), evaders=np.array([[x, 0.], [8., 0.]]),
                                    active_e=np.array([True, False]), active_p=np.array([True, False]))
            policy._advance_camera(parent, child)
            return child.visibility[0, 0]
        self.assertTrue(observed(10.))
        self.assertFalse(observed(-10.))
        policy.public.boxes = np.array([[[4., 5.], [-1., 1.]]])
        self.assertFalse(observed(10.))
        self.assertEqual(parent.cameras[0].locked, [None])
        np.testing.assert_array_equal(parent.cameras[0].yaw, [0.])

    def test_swept_capture_matches_reference_at_intermediate_times_and_walls(self):
        start = np.array([[-2.,0.], [-2.,2.]])
        finish = np.array([[2.,0.], [2.,2.]])
        pp = np.linspace(start, finish, 51)
        ee = np.repeat(np.array([[[0.,.2],[0.,2.3]]]),51,axis=0)
        boxes = np.array([[[-1.,1.],[.05,.15]]])
        active = np.ones(2,bool)
        hits, owners = capture_samples(pp, ee, active, active, boxes)
        expected = [-1,-1]
        for k, positions in enumerate(np.linspace(start, finish, 101)):
            for p,e in capture_pairs(positions,ee[0],active,np.array(expected)<0,boxes):
                expected[e] = k
                self.assertEqual(owners[e],p)
        np.testing.assert_array_equal(hits, expected)
        self.assertEqual(expected[0], -1)
        self.assertTrue(0 < expected[1] < 100)
