import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from learning.e45q_corrective_imitation import cosine_lr,sample_labels,stable_bc_step,factory
from learning.e45p_actor_supervision import public_teacher,initialize_bc_std,fit_metrics


class CorrectiveImitationTests(unittest.TestCase):
    def test_lr_decreases_to_registered_floor(self):
        values=[cosine_lr(i,100,1e-4,1e-5) for i in range(100)]
        self.assertAlmostEqual(values[0],1e-4);self.assertAlmostEqual(values[-1],1e-5)
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))

    def test_half_original_labels_kept_and_sampling_reproducible(self):
        def data(value):return dict(obs=np.zeros((5,2,845),np.float32),action=np.full((5,2,2),value,np.float32),active=np.ones((5,2,1),np.float32))
        original=data(.1);corrections=data(.9)
        a=sample_labels(original,corrections,np.random.default_rng(7));b=sample_labels(original,corrections,np.random.default_rng(7))
        self.assertEqual(int(np.isclose(a['action'][:,0,0],.1).sum()),128)
        for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_stable_updates_preserve_std_and_clip_gradient(self):
        from tests.test_e45p_actor_supervision import ActorSupervisionTests
        torch.set_num_threads(1);helper=ActorSupervisionTests();model=helper.model();teacher=helper.teacher()
        initialize_bc_std(model);before=fit_metrics(model,teacher)
        for i in range(30):stable_bc_step(model,teacher,cosine_lr(i,30,1e-4,1e-5))
        self.assertLess(sum(fit_metrics(model,teacher)['latent_mse']),sum(before['latent_mse']))
        for actor in model.actor:
            norm=sum(float(x.grad.square().sum()) for x in actor.actor.parameters() if x.grad is not None)**.5
            self.assertLessEqual(norm,1.0001)
            self.assertTrue(torch.equal(actor.actor.log_std_layer.weight,torch.zeros_like(actor.actor.log_std_layer.weight)))

    def test_teacher_labels_are_separate_and_teacher_never_actuates(self):
        class Belief:
            def signature(self):return 'same-public-belief'
        class Learner:
            belief=Belief()
            def observe(self,obs):pass
            def act(self,obs,fixed=None):
                self.samples=[dict(obs=np.ones((2,845),np.float32),active=np.ones((2,1),np.float32),action=np.full((2,2),.2,np.float32))]
                self.last_stats={'joint_action_search':False}
                return np.full((2,2),.2)
        class Expert:
            belief=Belief();traces=[];last_stats={'simulations':2,'assignment_used':False,'roles_used':False}
            def __init__(self,*args,**kwargs):pass
            def observe(self,obs):pass
            def act(self,obs,**kwargs):
                assert 'state' not in obs and 'worlds' not in obs
                return np.full((2,2),-.7)
        learner=Learner()
        with patch('learning.e45q_corrective_imitation.o.factory',return_value=learner),patch('learning.e45q_corrective_imitation.GeometryController',Expert),patch('learning.e45q_corrective_imitation.o.j.common.base.audit_simulated',return_value=np.zeros(4)):
            own=factory({'map':[],'speed':1.,'worlds':'not exposed'},dict(seed=7,query_teacher=True))
            actual=own.act({'positions':[[0,0],[1,1]]})
        np.testing.assert_array_equal(actual,np.full((2,2),.2))
        np.testing.assert_allclose(own.labels[0]['action'],-.7)
        np.testing.assert_allclose(own.samples[-1]['action'],.2)
        self.assertFalse(own.oracle_records[-1]['teacher_actuates'])

    def test_evaluation_does_not_construct_teacher(self):
        class Learner:
            def act(self,*args):self.last_stats={};return np.zeros((2,2))
        with patch('learning.e45q_corrective_imitation.o.factory',return_value=Learner()),patch('learning.e45q_corrective_imitation.GeometryController',side_effect=AssertionError('No teacher allowed')):
            own=factory({'map':[],'speed':1.},dict(seed=7));own.act({})

    def test_route_gate_requires_both_seeds_last_two_rounds(self):
        import run_e45q_corrective_hasac as q
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=dict(seeds=[17,29],rounds_stable_required=2)
            for route in q.ROUTES:
                for seed in cfg['seeds']:
                    history=[dict(round=i,deterministic=dict(episodes=6,double=5 if route=='dagger' else 0)) for i in range(3)]
                    q.j.save(root/'imitation'/f'seed_{seed}'/route/'decision.json',dict(history=history))
            self.assertEqual(q.choose_route(root,cfg)[0],'dagger')
            file=root/'imitation/seed_29/dagger/decision.json';data=q.j.read(file);data['history'][-1]['deterministic']['double']=4;q.j.save(file,data)
            self.assertIsNone(q.choose_route(root,cfg)[0])


if __name__=='__main__':unittest.main()
