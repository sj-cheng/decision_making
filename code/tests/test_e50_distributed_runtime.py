import unittest
import numpy as np
from solvers.e50_distributed_runtime import (AgentObservation, DistributedEpisodeRunner,
    FactorisedMCTS, LocalPlan, PeerBus, PeerMailbox, PeerMessage, make_peer_message,
    make_local_geometry_mcts)
from solvers.e50_leaf_value import LeafValueNet

def msg(sender=1, seq=1, cycle=2, ttl=2, intent=None):
    return make_peer_message(sender, seq, cycle, float(cycle), [1.,2.], [.1,0.], .2, intent=intent, ttl=ttl)

def obs(agent=0, peers=()):
    return AgentObservation(agent, 2, 2., [0.,0.], [0.,0.], 0., None, None, tuple(peers))

class DistributedRuntimeTests(unittest.TestCase):
    def test_mailbox_sequence_and_ttl(self):
        box=PeerMailbox(0); self.assertFalse(box.ingest(msg(sender=0))); self.assertTrue(box.ingest(msg(seq=2)))
        self.assertFalse(box.ingest(msg(seq=2))); self.assertFalse(box.ingest(msg(seq=1))); self.assertEqual(len(box.valid(4)),1)
        self.assertFalse(box.valid(5)); self.assertEqual(box.ages(5)[1],3)
    def test_message_rejects_bad_values(self):
        with self.assertRaises(ValueError): msg(intent=[1.,2.,3.])
        with self.assertRaises(ValueError): PeerMessage(1,1,0,0.,1,[np.nan,0.],[0.,0.],0.,True)
    def test_factorised_mcts_is_single_agent_and_value_backed(self):
        calls=[]
        def value(o, endpoint): calls.append(tuple(endpoint)); return float(endpoint[1])
        plan=FactorisedMCTS(lambda o:[o.pose+[1.,0.],o.pose+[0.,1.]], value, simulations=6, seed=3).act(obs())
        self.assertEqual(plan.agent_id,0); np.testing.assert_allclose(plan.endpoint,[0.,1.]); self.assertEqual(plan.action.shape,(2,)); self.assertGreater(plan.visits,0); self.assertEqual(len(calls),6)
    def test_peer_messages_are_explicit_input(self):
        peer=msg(intent=[2.,2.]); seen=[]
        def candidates(o): seen.append(o.peer_messages); return [o.pose+[1.,0.]]
        FactorisedMCTS(candidates, lambda o,a:0., simulations=1).act(obs(peers=(peer,))); self.assertEqual(seen,[(peer,)])

    def test_small_budget_never_returns_unvisited_candidate(self):
        # A negative value for the only simulated child must not make an
        # unvisited child win through its artificial zero score.
        plan = FactorisedMCTS(
            lambda o: [o.pose + [1., 0.], o.pose + [0., 1.]],
            lambda o, endpoint: -1.0 if endpoint[0] > .5 else -2.0,
            simulations=1, seed=2).act(obs())
        self.assertEqual(plan.visits, 1)
        np.testing.assert_allclose(plan.endpoint, [1., 0.])

    def test_peer_bus_delay_and_drop(self):
        bus = PeerBus(delay=1, drop_cycles=(2,))
        boxes = {0: PeerMailbox(0), 1: PeerMailbox(1)}
        self.assertTrue(bus.publish(msg(cycle=1)))
        self.assertEqual(bus.deliver(1, boxes), 0)
        self.assertEqual(bus.deliver(2, boxes), 1)
        self.assertFalse(bus.publish(msg(cycle=2)))

    def test_leaf_value_schema_uses_local_observation_only(self):
        model = LeafValueNet()
        value = model(obs(), np.array([1., 0.]))
        self.assertTrue(np.isfinite(value))
        self.assertEqual(model.calls, 1)
        fitted = LeafValueNet.fit([obs(), obs()], [[1., 0.], [0., 1.]], [2., 1.])
        self.assertTrue(np.isfinite(fitted(obs(), np.array([1., 0.]))))

    def test_episode_runner_keeps_controller_instances_independent(self):
        controllers = {}
        for aid in (0, 1):
            controllers[aid] = FactorisedMCTS(
                lambda o: [o.pose + [1., 0.]], lambda o, e: float(e[0]), simulations=1)
        seen = []
        def make_obs(aid, cycle, peers, previous):
            seen.append((aid, cycle, tuple(p.sender_id for p in peers)))
            return obs(aid, peers)
        logs = DistributedEpisodeRunner(controllers, delay=1).run(3, make_obs)
        self.assertEqual(len(logs), 3)
        self.assertTrue(any(x[2] == (1,) for x in seen if x[0] == 0))
        self.assertTrue(any(x[2] == (0,) for x in seen if x[0] == 1))

    def test_geometry_mcts_factory_is_single_agent(self):
        plan = make_local_geometry_mcts(simulations=2).act(obs())
        self.assertEqual(plan.action.shape, (2,))
        self.assertGreater(plan.visits, 0)

if __name__=='__main__': unittest.main()
