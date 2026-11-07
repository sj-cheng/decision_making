import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import e45c_belief_diagnostics as diagnostic
from learning.e45g_ppo import Actor, Critic, Controller, TeamBelief, gae, latent_log_prob, safe_action, execution_legal_action
from run_e45_team_pilot import make_case
import run_e45g_rl_benchmark as run


class RLBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.case=make_case('long_wall',0)
        own=Controller(cls.case['map'],cls.case['speed'],17,Actor())
        for obs in cls.case['history']:own.observe(obs)
        cls.features=own.features(cls.case['history'][-1])
        cls.x={k:torch.as_tensor(v[None],dtype=torch.float32) for k,v in cls.features.items()}

    def test_actor_has_no_truth_input_and_particle_order_is_irrelevant(self):
        actor=Actor().eval();x={k:v.clone() for k,v in self.x.items()}
        index=torch.arange(63,-1,-1)
        x['particles']=x['particles'][:,:,index];x['weights']=x['weights'][:,:,index]
        with torch.no_grad():a=actor(self.x)[0];b=actor(x)[0]
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)
        self.assertEqual(a.shape,(1,2,2))
        with self.assertRaises(TypeError):actor(self.x,torch.ones(1,4))

    def test_ippo_critic_ignores_privileged_truth_mappo_requires_it(self):
        ippo=Critic('IPPO');mappo=Critic('MAPPO')
        torch.testing.assert_close(ippo(self.x,torch.ones(1,4)),ippo(self.x,-torch.ones(1,4)))
        with self.assertRaises(ValueError):mappo(self.x)
        self.assertEqual(mappo(self.x,torch.zeros(1,4)).shape,(1,2))

    def test_popart_preserves_unnormalized_prediction(self):
        critic=Critic('MAPPO');torch.nn.init.normal_(critic.head[-1].weight)
        privileged=torch.randn(1,4)
        before=critic(self.x,privileged).detach().clone()
        critic.update_normalization(torch.tensor([[10.,20.],[30.,40.]]))
        torch.testing.assert_close(critic(self.x,privileged),before,atol=1e-5,rtol=1e-5)

    def test_complete_episode_gae_preserves_team_credit_after_retirement(self):
        advantage,returns=gae(np.array([0.,0.,1.]),np.zeros((3,2)),lam=1.)
        np.testing.assert_array_equal(returns,np.ones((3,2)))
        mean=torch.zeros((2,2,2),requires_grad=True);std=torch.ones_like(mean);raw=torch.ones_like(mean)
        mask=torch.tensor([[1.,0.],[0.,1.]])
        loss=-(latent_log_prob(mean,std,raw)*mask).sum();loss.backward()
        torch.testing.assert_close(mean.grad[0,1],torch.zeros(2));torch.testing.assert_close(mean.grad[1,0],torch.zeros(2))

    def test_latent_likelihood_is_finite_and_unchanged_by_safety_mapping(self):
        raw=torch.tensor([[[30.,-30.],[0.,0.]]]);mean=torch.zeros_like(raw);std=torch.ones_like(raw)
        lp=latent_log_prob(mean,std,raw)
        self.assertTrue(torch.isfinite(lp).all())
        action,changed=safe_action(np.array([[9.9,0.],[-9.9,0.]]),np.tanh(raw.numpy()[0]),[True,False],1.1,np.zeros((0,2,2)))
        self.assertGreater(changed,0);np.testing.assert_array_equal(action[1],[0.,0.])
        self.assertTrue(run.base.legal_action(np.array([9.9,0.]),action[0],1.1,np.zeros((0,2,2))))

    def test_potential_reward_telescopes_to_raw_minus_initial_phi(self):
        phi=np.array([-30.,-20.,-25.]);raw=np.array([-.0025,-.0025,.9975])
        shaped=raw+.05*(np.r_[phi[1:],0.]-phi)
        self.assertAlmostEqual(shaped.sum(),raw.sum()-.05*phi[0])

    def test_world_boundary_tolerance_cannot_escape_real_environment(self):
        # Exact transition from IPPO29 update40; the tolerant predicate accepts
        # an endpoint 1.09867539e-7 beyond the environment boundary.
        at=np.array([-9.999289425186799,5.99566821466101])
        raw=np.array([-0.0007106846807396714,0.0010837711440694515])
        boxes=np.array([[[-8.17455432190504,8.370077061013378],[-1.0236516689371,.9763483310628999]],
                        [[-8.17455432190504,-6.77455432190504],[.9763483310628999,2.9763483310629]]])
        self.assertTrue(run.base.legal_action(at,raw,1.3,boxes))
        self.assertLess((at+raw)[0],-10.)
        action,changed=safe_action(np.array([at,[0.,5.]]),np.array([raw/1.3,[0.,0.]]),[True,False],1.3,boxes)
        from corner_estimator_scenarios import original_model
        env=original_model(boxes,1.3);state=np.zeros((env.state_dim,1))
        state[:8,0]=[2.2254456780949603,-5.1236516689371,-10.,9.6763483310629,*at,0.,5.]
        state[env.active_idxs,0]=[1,1,1,0]
        self.assertTrue(env.is_valid(env.step(state,np.r_[np.zeros(4),action.ravel()][:,None],1.)))
        self.assertEqual(changed,1)

    def test_safe_action_obeys_all_four_exact_world_faces(self):
        for axis in (0,1):
            for sign in (-1,1):
                at=np.zeros((2,2));at[0,axis]=sign*(10.-1e-8)
                raw=np.zeros((2,2));raw[0,axis]=sign*1e-7
                action,_=safe_action(at,raw,[True,False],1.,np.zeros((0,2,2)))
                self.assertTrue(np.all(abs(at+action)<=10.))
        action,changed=safe_action(np.array([[10.,0.],[0.,0.]]),np.ones((2,2)),[True,False],1.,np.zeros((0,2,2)))
        np.testing.assert_array_equal(action,np.zeros((2,2)))
        self.assertEqual(changed,1)

    def test_float32_wall_rounding_cannot_make_rootpuct_root_terminal(self):
        from corner_estimator_scenarios import original_model
        from solvers.root_audit_puct import RootPUCT,cpp
        boxes=np.array([[[-1.0771405200949744,.9228594799050256],[-6.499732321157042,6.61116210206219]],
                        [[-4.677140520094974,-3.6771405200949743],[-4.499732321157042,4.61116210206219]]])
        bad=np.array([-4.677140834375443,1.375025888366913]);at=bad-np.array([.001,0.]);delta=np.array([.001,0.])
        self.assertTrue(run.base.legal_action(at,delta,1.3,boxes))
        self.assertFalse(execution_legal_action(at,delta,1.3,boxes))
        env=original_model(boxes,1.3);state=np.zeros((env.state_dim,1));state[:8,0]=[-2.113384924410923,3.5034606785499554,
            -2.2698563199034765,.5200199813568096,2.433163714554392,-5.1241597780211965,*bad]
        state[env.active_idxs,0]=1;state[env.time_idx,0]=30
        self.assertTrue(env.is_valid(state))
        other=RootPUCT([None]*4,None,4,0.,0.)
        result=cpp.cpp_search(other._problem_wrapper_for(env),other.solver_wrapper,state,0)
        self.assertFalse(result.success)
        action,_=safe_action(np.array([at,[0.,8.]]),np.array([delta/1.3,[0.,0.]]),[True,False],1.3,boxes)
        state[6:8,0]=at+action[0]
        self.assertTrue(other.search(env,state,turn=0).success)

    def test_validation_cache_distinguishes_checkpoint_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);case=folder/'case.json';case.write_text('{}')
            checkpoint=folder/'model.pt';checkpoint.write_bytes(b'first model')
            entry=dict(id='map',path=str(case),family='long_wall',pattern='shared_entry',split='validation')
            first=run.request(entry,'AB','rootpuct','IPPO',29,'validation',checkpoint)
            checkpoint.write_bytes(b'updated model')
            second=run.request(entry,'AB','rootpuct','IPPO',29,'validation',checkpoint)
            self.assertNotEqual(first['key'],second['key'])
            self.assertNotEqual(first['checkpoint_sha256'],second['checkpoint_sha256'])
            with patch.object(run.base,'episode_job',return_value={'checkpoint_sha256':first['checkpoint_sha256']}):
                with self.assertRaisesRegex(RuntimeError,'checkpoint identity mismatch'):
                    run.eval_job(second)

    @patch.object(diagnostic,'EVIDENCE',None)
    def test_saved_public_failure_is_recovered_without_truth(self):
        d=json.loads((Path(__file__).parent/'fixtures/e45f_empty_support_public.json').read_text())
        first=TeamBelief(d['boxes'],d['seed']);second=TeamBelief(d['boxes'],d['seed'])
        for obs in d['history']:first.update(obs);second.update(obs)
        self.assertEqual(first.signature(),second.signature());self.assertEqual(first.step,79)
        self.assertEqual(first.refinements[0]['support_points'],2)

    def test_factory_retains_search_type_for_independent_trace_audit(self):
        from solvers.e45b_joint_nte import Controller as Search
        task=dict(method='U',seed=17,budget_ms=200.)
        own=run.factory(self.case,task)
        self.assertIsInstance(own,Search)

    def test_statistics_never_drop_invalid_cell_or_treat_seed_as_map(self):
        rows=[]
        for g in range(4):
            for w in ('AB','BA'):
                for o in run.base.OPPONENTS:
                    for arm in run.ARMS:
                        for seed in (run.SEEDS if arm in run.LEARNED else (0,)):
                            rows.append(dict(key=str(len(rows)),suite='benchmark',group=str(g),family=str(g%2),world=w,opponent=o,
                                method=arm+str(seed) if seed else arm,train_seed=seed,utility=.9 if arm=='SINGLE_PRIOR' else .8,
                                double_capture=True,completion_cost=20.,latencies_ms=[10.]))
        stats=run.summarize(rows);self.assertEqual(stats['status'],'complete')
        self.assertEqual(stats['primary']['benchmark']['groups'],4)
        self.assertEqual(len(stats['primary']['benchmark']['comparisons']),14)
        rows[0]['valid_episode']=False
        self.assertEqual(run.summarize(rows)['status'],'incomplete')


if __name__=='__main__':unittest.main()
