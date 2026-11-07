import unittest
import numpy as np
import torch
from learning.e45i_official_mappo import (policy, EpisodeBuffer, finish_episodes,
    actor_inputs, critic_inputs, PublicState, APFController, apf_action, OBS_DIM, CENT_DIM, HIDDEN)
from learning.e45g_ppo import execution_legal_action
from run_e45_team_pilot import make_case


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.case=make_case('long_wall',0)

    def test_public_actor_input_independent_of_training_privilege(self):
        own=PublicState(self.case['map'],self.case['speed'],17)
        for obs in self.case['history']:own.observe(obs)
        x=actor_inputs(own.features(obs))
        self.assertEqual(x.shape,(2,OBS_DIM))
        a=critic_inputs(x,np.zeros(4));b=critic_inputs(x,np.ones(4))
        self.assertEqual(a.shape,(2,CENT_DIM))
        self.assertFalse(np.array_equal(a,b))
        np.testing.assert_array_equal(x[:,:-2],np.repeat(x[:1,:-2],2,axis=0))
        np.testing.assert_array_equal(a[0],a[1])

    def test_fixed_apf_equals_zero_gain_actor_and_obeys_geometry(self):
        own=APFController(self.case['map'],self.case['speed'],17)
        for obs in self.case['history']:own.observe(obs)
        fixed=apf_action(own,obs)
        np.testing.assert_array_equal(fixed,apf_action(own,obs,np.zeros((2,4))))
        for i,at in enumerate(obs['positions']):
            self.assertTrue(execution_legal_action(np.array(at),fixed[i],own.public.speed,own.public.boxes))
        retired=dict(obs,active_p=[False,True],active_e=[False,True])
        action=apf_action(own,retired)
        np.testing.assert_array_equal(action[0],np.zeros(2))

    def episode(self,kind='RMAPPO',length=13):
        model,trainer=policy(kind)
        rng=np.random.default_rng(17)
        obs=rng.normal(size=(length,2,OBS_DIM)).astype(np.float32)
        # Public part shared; only identity distinguishes actors.
        obs[:,1,:-2]=obs[:,0,:-2]
        hidden=np.zeros((2,1,HIDDEN),np.float32);states=[];actions=[];logp=[]
        with torch.no_grad():
            for x in obs:
                states.append(hidden.copy())
                a,l,h=model.actor(x,hidden,np.ones((2,1),np.float32))
                actions.append(a.numpy());logp.append(l.numpy());hidden=h.numpy()
        e=dict(obs=obs,actor_hidden=np.stack(states),raw=np.stack(actions),logp=np.stack(logp),
            active=np.ones((length,2,1),np.float32),reward=np.zeros(length,np.float32),
            privileged=rng.normal(size=(length,4)).astype(np.float32))
        e['reward'][-1]=1.
        return model,trainer,e

    def test_recurrent_chunk_log_probs_equal_online_policy_with_tail_padding(self):
        model,trainer,e=self.episode()
        e['active'][7:,0]=0
        buffer=finish_episodes([e],model)
        np.testing.assert_allclose(e['values'][:,0],e['values'][:,1],atol=1e-6)
        active_count=0
        for sample in buffer.recurrent_generator(buffer.returns[:-1]-buffer.value_preds[:-1],2,10):
            central,x,h,hc,raw,v,ret,masks,active,old,adv,_=sample
            with torch.no_grad():
                _,fresh,_=model.evaluate_actions(central,x,h,hc,raw,masks,None,active)
            np.testing.assert_allclose(fresh.numpy()[active[:,0]>0],old[active[:,0]>0],rtol=1e-5,atol=2e-5)
            active_count+=int(active.sum())
        self.assertEqual(active_count,int(e['active'].sum()))

    def test_author_update_changes_policy_and_keeps_team_credit_after_retirement(self):
        model,trainer,e=self.episode('MAPPO_APF',20)
        e['active'][5:,0]=0
        buffer=finish_episodes([e],model)
        # Final team success reaches decisions of the retired agent.
        self.assertGreater(float(e['returns'][4,0,0]),0.)
        before={k:v.clone() for k,v in model.actor.state_dict().items()}
        info=trainer.train(buffer)
        self.assertTrue(all(np.isfinite(float(v.detach())) if torch.is_tensor(v) else np.isfinite(v) for v in info.values()))
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.actor.state_dict().items()))


if __name__=='__main__':unittest.main()
