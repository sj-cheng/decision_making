import unittest
from types import SimpleNamespace
import numpy as np

from solvers.e52_hmcts import HierarchicalMCTS
from solvers.e52_nar_nae import observation_features


class _Map:
    def obstacle_boxes(self):
        return np.zeros((0, 2, 2))

    def free_space_endpoints(self, **kwargs):
        return [np.array([1., 0.]), np.array([0., 1.])]

    def segment_safe(self, start, endpoint, **kwargs):
        return True


def _observation():
    return SimpleNamespace(agent_id=0, cycle=0, pose=np.zeros(2), velocity=np.zeros(2),
                           heading=0., local_map=_Map(),
                           belief=SimpleNamespace(particles=np.zeros((0, 2)), weights=np.zeros(0)),
                           peer_messages=(), previous_prefix=np.array([1., 0.]))


class E52HierarchicalTests(unittest.TestCase):
    def test_public_feature_schema_and_action(self):
        observation = _observation()
        self.assertEqual(observation_features(observation, np.array([1., 0.])).shape, (14,))
        plan = HierarchicalMCTS().act(observation)
        self.assertEqual(plan.endpoint.shape, (2,))
        self.assertTrue(plan.minco_feasible)
        self.assertGreater(plan.visits, 0)

    def test_no_global_map_or_joint_action_input(self):
        first = HierarchicalMCTS().act(_observation())
        second = HierarchicalMCTS().act(_observation())
        np.testing.assert_allclose(first.endpoint, second.endpoint)



class E52SearchAndBeliefTests(unittest.TestCase):
    def test_multistep_tree_and_budget(self):
        controller = HierarchicalMCTS(high_simulations=128, time_budget_ms=1000.)
        controller.act(_observation())
        self.assertEqual(controller.last_search['max_depth'], 3)
        self.assertLessEqual(controller.last_search['evaluations'], 128)
        self.assertGreater(sum(x['visits'] for x in controller.last_search['root']), 1)

    def test_network_ablation_independent(self):
        class Network:
            def __init__(self): self.calls = 0
            def score(self, *args): self.calls += 1; return 0., 1.
        network=Network()
        controller=HierarchicalMCTS(nae=network, high_simulations=64,time_budget_ms=1000.)
        controller.act(_observation())
        self.assertGreater(network.calls,0)
        self.assertFalse(controller.last_search['nar_enabled'])
        self.assertTrue(controller.last_search['nae_enabled'])

    def test_curved_trajectory_checked_not_only_endpoint(self):
        from solvers.e52_hmcts import LocalSnapshot
        local=LocalSnapshot(_Map())
        local.boxes=np.array([[[.4,.6],[.4,.6]]])
        self.assertTrue(local.segment_safe([0.,0.],[1.,0.]))
        self.assertFalse(local.path_safe([[0.,0.],[.5,.5],[1.,0.]]))

    def test_identity_stale_relay_capture_and_ttl(self):
        from solvers.e52_belief import TargetBelief
        from solvers.e50_distributed_runtime import make_peer_message
        first=TargetBelief(0); second=TargetBelief(1)
        first.update([(0,[1.,0.]),(1,[2.,0.])],0.)
        message=make_peer_message(0,1,0,0.,np.zeros(2),np.zeros(2),0.,beliefs=first.payload())
        second.merge([message],1.)
        self.assertEqual(set(second.tracks),{0,1})
        second.capture(0); second.merge([message],2.)
        self.assertEqual(set(second.tracks),{1})
        second.merge([message],6.)
        self.assertEqual(len(second.tracks),0)

    def test_visit_labels_and_eight_step_value(self):
        from run_e52_train_nar_nae import records_from_run
        root=[dict(features=[0.]*14,visits=2),dict(features=[1.]*14,visits=6)]
        logs=[]
        for t in range(8):
            logs.append(dict(cycle=t,search={a:dict(root=root) for a in ('0','1')},
                plans={a:dict(candidate_id=1) for a in ('0','1')},captures=[],
                sensor_stats={a:dict(visible_targets=0) for a in ('0','1')},
                beliefs={},pursuers=[[0.,0.],[1.,0.]]))
        records=records_from_run(dict(logs=logs))
        self.assertEqual(len(records),2)
        np.testing.assert_allclose(records[0]['policy'],[.25,.75])
        self.assertEqual(records[0]['value_index'],1)
        self.assertEqual(records_from_run(dict(logs=logs[:7])),[])

class E52TrainingTests(unittest.TestCase):
    def test_checkpoint_roundtrip_and_separate_networks(self):
        import tempfile
        from pathlib import Path
        from solvers.e52_nar_nae import NARNAE
        model=NARNAE(); model.torch.set_num_threads(1)
        record=dict(features=[[0.]*14,[1.]*14],policy=[0.,1.],returns=[0.,2.],value_index=1)
        model.fit_records([record],epochs=3)
        self.assertFalse(set(map(id,model.nar.parameters())) & set(map(id,model.nae.parameters())))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'weights.pt'; model.save(path,metadata={'diagnostic':True})
            restored=NARNAE.load(path)
            self.assertTrue(restored.metadata['diagnostic'])
            np.testing.assert_allclose(model.score(_observation(),[1.,0.]),restored.score(_observation(),[1.,0.]))

    def test_network_can_change_prior(self):
        class Prior:
            def score(self, observation, endpoint, option): return (20. if endpoint[1]>0 else -20.),0.
        model=HierarchicalMCTS(nar=Prior(),high_simulations=64,time_budget_ms=1000.)
        plan=model.act(_observation())
        self.assertGreater(plan.endpoint[1],0.)

if __name__ == '__main__':
    unittest.main()
