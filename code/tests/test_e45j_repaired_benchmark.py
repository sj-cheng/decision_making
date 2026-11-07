import unittest
import numpy as np
import torch
from learning.e45j_repaired_mappo import (inputs,clearance,policy,finish_episodes,
    training_rewards,OBS_DIM,CENT_DIM)
from learning.e45i_official_mappo import PublicState
from run_e45_team_pilot import make_case
from run_e45j_repaired_benchmark import curriculum_case,VisibleKernelTargets
from solvers.frozen_d_search_interface import safe
from corner_estimator_scenarios import original_model


class RepairedBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.case=make_case('long_wall',0)
        state=PublicState(cls.case['map'],cls.case['speed'],17)
        for obs in cls.case['history']:state.observe(obs)
        cls.state=state;cls.obs=obs;cls.features=state.features(obs)

    def test_canonical_derived_inputs_are_invariant_and_have_no_privileged_argument(self):
        a,c=inputs(self.features,self.state.public,self.obs)
        other={k:v.copy() for k,v in self.features.items()};rng=np.random.default_rng(17)
        for j in range(2):
            order=rng.permutation(64);other['particles'][j]=other['particles'][j,order];other['weights'][j]=other['weights'][j,order]
        b,d=inputs(other,self.state.public,self.obs)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(c,d)
        self.assertEqual(a.shape,(2,OBS_DIM));self.assertEqual(c.shape,(CENT_DIM-4,))
        self.assertFalse(np.array_equal(a[0],a[1]))

    def test_clearance_reports_blocked_heading_without_moving_state(self):
        positions=np.array([[-1.2,0.],[-3.,0.]])
        before=positions.copy();r=clearance(positions,np.array([True,False]),1.1,np.array([[[-1.,1.],[-5.,5.]]]))
        self.assertGreater(r[0,0],.17);self.assertLess(r[0,0],.19)
        self.assertEqual(r[0,4],1.);np.testing.assert_array_equal(r[1],np.zeros(16));np.testing.assert_array_equal(positions,before)

    def test_curriculum_has_real_visible_sensors_and_paired_retirement(self):
        p=original_model(self.case['map'],self.case['speed'])
        for variant in ('single0','single1','static','moving'):
            c=curriculum_case(self.case,variant);o=c['history'][0]
            s=np.array(c['worlds'][0]['initial'])
            self.assertTrue(p.is_valid(s[:,None]))
            self.assertEqual(sum(o['active_p']),sum(o['active_e']))
            self.assertEqual(sum(o['active_e']),1 if variant.startswith('single') else 2)
            for j,active in enumerate(o['active_e']):
                if active:self.assertTrue(any(q is not None for q in o['reports'][j]))
            self.assertEqual(o['step'],0)

    def test_potential_telescopes_and_intent_cost_is_explicit(self):
        raw=np.array([-.0025,-.0025,.9975]);phi=np.array([-.8,-.3,.15]);blocked=np.array([0.,.5,1.])
        r=training_rewards(raw,phi,blocked)
        self.assertAlmostEqual(float(r.sum()),float(raw.sum()-phi[0]-.05*blocked.sum()),places=6)

    def test_visible_start_d_kernel_anchors_public_measurements_and_is_legal(self):
        c=curriculum_case(self.case,'moving');one=VisibleKernelTargets(c,17);two=VisibleKernelTargets(c,17)
        np.testing.assert_array_equal(one.positions(),np.asarray(c['worlds'][0]['initial'][:4]).reshape(2,2))
        for _ in range(5):
            before=one.positions();a=one.actions(np.ones(2,bool));b=two.actions(np.ones(2,bool))
            np.testing.assert_array_equal(a,b)
            for j in range(2):self.assertTrue(safe(before[j],a[j],1.,np.asarray(c['map']),np.zeros(2,np.int64)))

    def test_new_real_feature_schema_runs_author_recurrent_update(self):
        for kind in ('RMAPPO','MAPPO_APF'):
            model,trainer=policy(kind);x,critic_public=inputs(self.features,self.state.public,self.obs)
            hidden=np.zeros((2,1,64),np.float32);data=[]
            with torch.no_grad():
                for _ in range(13):
                    before=hidden.copy();a,l,h=model.actor(x,before,np.ones((2,1),np.float32));hidden=h.numpy()
                    data.append(dict(obs=x,critic_public=critic_public,actor_hidden=before,raw=a.numpy(),logp=l.numpy(),active=np.ones((2,1),np.float32)))
            ep={k:np.stack([s[k] for s in data]) for k in data[0]}
            ep['privileged']=np.zeros((13,4),np.float32);ep['reward']=np.zeros(13,np.float32);ep['reward'][-1]=1.
            buffer=finish_episodes([ep],model)
            for batch in buffer.recurrent_generator(buffer.returns[:-1]-buffer.value_preds[:-1],2,10):
                with torch.no_grad():_,logp,_=model.evaluate_actions(batch[0],batch[1],batch[2],batch[3],batch[4],batch[7],None,batch[8])
                mask=batch[8][:,0]>0
                np.testing.assert_allclose(logp.numpy()[mask],batch[9][mask],atol=2e-5)
            before={k:v.clone() for k,v in model.actor.state_dict().items()};info=trainer.train(buffer)
            self.assertTrue(all(np.isfinite(float(v.detach())) if torch.is_tensor(v) else np.isfinite(v) for v in info.values()))
            self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.actor.state_dict().items()))


if __name__=='__main__':unittest.main()
