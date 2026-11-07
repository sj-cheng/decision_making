import copy
import unittest
import numpy as np
import torch
from learning.e45m_author_hasac import AuthorHASAC
from learning.e45m_task_adapter import Replay,episode_arrays


def synthetic_episode(n=3):
    samples=[];trace=[]
    for i in range(n):
        active=np.asarray([[float(i==0)],[1]],np.float32)
        obs=np.full((2,845),i/10,np.float32);obs[:,-1:]=active
        samples.append(dict(obs=obs,central_public=np.full(854,i/10,np.float32),
                            action=np.ones((2,2),np.float32)*.1*active,
                            active=active,phi=.3+i*.1,blocked=0.))
        trace.append(dict(state=[1,2,3,4]+[0]*9,reward=.9975 if i==n-1 else -.0025))
    return samples,trace


def synthetic_batch(n=8):
    rng=np.random.default_rng(7);obs=rng.normal(0,.1,(2,n,845)).astype(np.float32);obs[:,:,-1]=1
    center=rng.normal(0,.1,(n,858)).astype(np.float32)
    return (center,obs,rng.uniform(-1,1,(2,n,2)).astype(np.float32),None,
            rng.normal(0,.1,(n,1)).astype(np.float32),np.ones((n,1),np.float32),
            np.ones((2,n,1),np.float32),np.ones((n,1),np.float32),
            np.zeros_like(center),np.zeros_like(obs),None,np.ones((n,1),np.float32))


class HASACTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_retirement_and_full_team_terminal(self):
        samples,trace=synthetic_episode();d=episode_arrays(samples,trace)
        np.testing.assert_array_equal(d['done'][:,0],[0,0,1])
        self.assertEqual(d['active'][1,0,0],0)
        np.testing.assert_array_equal(d['next_obs'][:-1],d['obs'][1:])
        np.testing.assert_array_equal(d['next_obs'][-1],np.zeros((2,845)))
        np.testing.assert_array_equal(d['next_central'][-1],np.zeros(858))
        self.assertAlmostEqual(float(d['reward'].sum()),.9925-.3,places=6)

    def test_privileged_state_never_enters_actor_replay(self):
        samples,trace=synthetic_episode();a=episode_arrays(samples,trace)
        other=copy.deepcopy(trace);other[0]['state'][:4]=[7,8,9,10]
        b=episode_arrays(samples,other)
        np.testing.assert_array_equal(a['obs'],b['obs']);np.testing.assert_array_equal(a['next_obs'],b['next_obs'])
        self.assertFalse(np.array_equal(a['central'],b['central']))

    def test_replay_boundary_and_resume_preserve_transitions(self):
        import tempfile
        from pathlib import Path
        samples,trace=synthetic_episode();d=episode_arrays(samples,trace)
        replay=Replay(capacity=5,seed=19);replay.add(d);replay.add(d)
        self.assertEqual(replay.size,5);self.assertEqual(replay.position,1)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'r.npz';replay.save(p);state=copy.deepcopy(replay.rng.bit_generator.state)
            other=Replay(capacity=5,seed=0);other.load(p,state)
            for a,b in zip(replay.sample(32),other.sample(32)):
                if a is not None:np.testing.assert_array_equal(a,b)

    def test_retired_actor_has_zero_action_and_entropy(self):
        model=AuthorHASAC(obs_dim=845,retirement_mask=True)
        obs=np.zeros((2,845),np.float32);obs[0,-1]=1
        action,logp=model.actor[0].get_actions_with_logprobs(obs,stochastic=False)
        self.assertTrue(torch.isfinite(logp).all())
        torch.testing.assert_close(action[1],torch.zeros(2));torch.testing.assert_close(logp[1],torch.zeros(1))

    def test_terminal_target_does_not_bootstrap_at_time_horizon(self):
        a=AuthorHASAC(obs_dim=845,seed=71,retirement_mask=True)
        b=AuthorHASAC(obs_dim=845,seed=71,retirement_mask=True)
        batch=synthetic_batch();changed=list(copy.deepcopy(batch))
        changed[8][:]=8;changed[9][:]=8;changed[9][:,:,-1]=0
        for model,data in ((a,batch),(b,tuple(changed))):
            np.random.seed(8);torch.manual_seed(8);model.update(data)
        for key,value in a.critic.critic.state_dict().items():
            torch.testing.assert_close(value,b.critic.critic.state_dict()[key],rtol=0,atol=0)
        self.assertFalse(a.config['train']['use_proper_time_limits'])
        self.assertEqual(a.config['algo']['gamma'],1.)

    def test_invalid_retired_action_rejected(self):
        model=AuthorHASAC(obs_dim=845,retirement_mask=True);batch=list(synthetic_batch())
        batch[1][0,-1,-1]=0;batch[6][0,-1,0]=0;batch[2][0,-1]=.4
        with self.assertRaisesRegex(ValueError,'Retired replay actions'):model.update(tuple(batch))


if __name__=='__main__':unittest.main()
