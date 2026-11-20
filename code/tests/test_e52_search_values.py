"""Value semantics must remain valid before collecting long E52 teacher runs."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np

from solvers.e50_distributed_runtime import AgentObservation
from solvers.e52_hmcts import HierarchicalMCTS, LocalSnapshot, Node


class _Map:
    def obstacle_boxes(self): return np.zeros((0, 2, 2))
    def free_space_endpoints(self, **kwargs): return [np.array([1., 0.])]


def observation(particles=(), velocities=()):
    particles=np.array(particles, float).reshape(-1, 2)
    velocities=np.array(velocities, float).reshape(-1, 2)
    if len(velocities)!=len(particles): velocities=np.zeros_like(particles)
    belief=SimpleNamespace(particles=particles, velocities=velocities, weights=np.ones(len(particles)))
    return AgentObservation(0,0,0.,np.zeros(2),np.zeros(2),0.,LocalSnapshot(_Map()),belief)


class _ConstantNAE:
    def __init__(self, value): self.value=value; self.calls=0
    def score(self, *args): self.calls+=1; return 0.,self.value


class _Chain(HierarchicalMCTS):
    """A single deterministic branch isolates backups from candidate selection."""
    def __init__(self, **kwargs):
        super().__init__(time_budget_ms=1000., **kwargs)
        self.nodes=[]

    def _expand(self,node,deadline):
        node.expanded=True
        if node.depth>=self.depth: return
        child=Node(replace(node.obs,pose=node.obs.pose+np.array([1.,0.])),node.depth+1,
                   node,np.array([1.,0.]),0,1.,2. if node.depth==0 else 3.)
        node.children=[child];self.nodes.append(child)


class SearchValueTests(unittest.TestCase):
    def test_tracked_strategy_uses_geometry_while_target_unknown(self):
        network = _ConstantNAE(7.)
        controller = _Chain(nae=network, nae_strategy='tracked', depth=1, high_simulations=1)
        self.assertEqual(controller.act(observation()).value, 2.)
        self.assertEqual(network.calls, 0)
        self.assertEqual(controller.act(observation([[4., 0.]])).value, 7.)
        self.assertEqual(network.calls, 1)

    def test_residual_preserves_edge_reward_and_never_scores_shallow_nodes(self):
        class Network:
            def value_scores(self, obs, endpoints, levels):
                return np.array([100., 107.])

        controller = HierarchicalMCTS(nae=Network(), nae_strategy='leaf_residual', depth=2)
        controller.nae_queries = 0
        root = Node(observation([[4., 0.]]))
        first = Node(root.obs, depth=1, parent=root, reward=2.)
        self.assertEqual(controller._leaf_value(first), (0., False))
        left = Node(root.obs, depth=2, parent=first, reward=3.)
        right = Node(root.obs, depth=2, parent=first, reward=4.)
        first.children = [left, right]
        bonus, replaces_edge = controller._leaf_value(right)
        self.assertFalse(replaces_edge)
        self.assertGreater(bonus, 0.)
        self.assertLessEqual(bonus, .15)
        self.assertAlmostEqual(controller._leaf_value(left)[0], -bonus)
        self.assertEqual(controller.nae_queries, 2)

    def test_nae_action_return_does_not_add_immediate_reward_twice(self):
        controller=_Chain(nae=_ConstantNAE(7.),depth=1,high_simulations=1)
        plan=controller.act(observation())
        self.assertEqual(plan.value,7.)
        self.assertEqual(controller.nodes[0].total,7.)

    def test_ancestor_adds_and_discounts_only_its_own_reward(self):
        controller=_Chain(nae=_ConstantNAE(7.),depth=2,high_simulations=3,gamma=.5)
        controller.act(observation())
        root_child=controller.nodes[0]
        leaf=controller.nodes[1]
        self.assertEqual(leaf.visits,1)
        self.assertEqual(leaf.total,7.)
        # First two visits evaluated the first edge's Q; third backed up leaf Q.
        self.assertEqual(root_child.total,7.+7.+2.+.5*7.)

    def test_no_nae_retains_finite_depth_reward_backup(self):
        controller=_Chain(depth=2,high_simulations=3,gamma=.5)
        controller.act(observation())
        self.assertEqual(controller.nodes[1].total,3.)
        self.assertEqual(controller.nodes[0].total,2.+2.+2.+.5*3.)

    def test_predicted_capture_removed_from_all_belief_arrays(self):
        controller=HierarchicalMCTS();controller.midpoint=0
        obs=observation([[.5,0.],[4.,0.]],[[0.,0.],[.25,0.]])
        derivatives=dict(velocities=np.zeros((1,2)),accelerations=np.zeros((1,2)))
        future=controller._imagined(obs,np.array([.5,0.]),derivatives)
        self.assertEqual(future.belief.captured_count,1)
        self.assertEqual(future.belief.particles.shape,(1,2))
        self.assertEqual(future.belief.weights.shape,(1,))
        self.assertEqual(future.belief.velocities.shape,(1,2))
        np.testing.assert_allclose(future.belief.particles,[[4.25,0.]])
        again=controller._imagined(future,np.array([.5,0.]),derivatives)
        self.assertEqual(again.belief.captured_count,0)
        self.assertLess(controller._reward(future,again,0,np.array([.5,0.])),1.)
        # Observation-owned belief is never mutated by imagined capture.
        self.assertEqual(obs.belief.particles.shape,(2,2))

    def test_exhausted_tracking_branch_does_not_receive_exploration_rewards(self):
        controller=HierarchicalMCTS();controller.midpoint=0
        obs=observation([[.5,0.]])
        derivatives=dict(velocities=np.zeros((1,2)),accelerations=np.zeros((1,2)))
        terminal=controller._imagined(obs,np.array([.5,0.]),derivatives)
        self.assertTrue(terminal.belief.terminal)
        again=controller._imagined(terminal,np.array([.5,0.]),derivatives)
        self.assertEqual(controller._reward(terminal,again,0,np.array([.5,0.])),0.)
        node=Node(terminal)
        controller._expand(node,float('inf'))
        self.assertEqual(node.children,[])
        self.assertTrue(node.expanded)
        # No initial target report is an exploration situation, not terminal.
        empty=controller._imagined(observation(),np.array([.5,0.]),derivatives)
        self.assertFalse(empty.belief.terminal)
        self.assertGreater(controller._reward(observation(),empty,1,np.array([.5,0.])),0.)


if __name__=='__main__': unittest.main()
