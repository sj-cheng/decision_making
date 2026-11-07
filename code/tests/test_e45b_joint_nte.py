import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from learning.e45b_joint_network import Network,Runtime,KINDS,KEYS,log_density,losses,augment
from solvers.e45b_joint_nte import Controller,simultaneous_capture,condition_team,ETA
from run_e45_team_pilot import make_case
from corner_estimator_scenarios import original_model


class RoleFreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.case=make_case('long_wall',0);torch.set_num_threads(1)
    def controller(self):
        c=self.case;x=Controller(c['map'],c['speed'],17)
        for obs in c['history'][:-1]:x.observe(obs)
        x.act(c['history'][-1],fixed_simulations=12);x.deadline=float('inf')
        return x

    def test_no_assignment_and_exact_joint_replay(self):
        with patch('solvers.e45_team_adapter.optimal_assignment',side_effect=AssertionError('assignment forbidden')):
            a,b=self.controller(),self.controller()
        self.assertEqual(a.belief.signature(),b.belief.signature())
        self.assertEqual(a.last_stats['root_visits'],b.last_stats['root_visits'])
        self.assertEqual(a.last_stats['root_actions'],b.last_stats['root_actions'])
        self.assertGreater(a.last_stats['simulations'],0)
        self.assertTrue(a.last_stats['joint_action_search']);self.assertFalse(a.last_stats['assignment_used'])
        self.assertTrue(all(len(v)==4 for v in a.last_stats['root_actions']))
        for f in (Controller.propose,Controller.select,Controller.features):
            self.assertFalse(set(inspect.signature(f).parameters)&{'state','targets','environment','ids','assignment','role'})

    def test_public_proposal_does_not_depend_on_sampled_world(self):
        a=self.controller();state=a.proposal_rng.bit_generator.state
        x=a.propose(a.root);a.proposal_rng.bit_generator.state=state
        for _ in range(10):a.sample_world()
        np.testing.assert_array_equal(x,a.propose(a.root))

    def test_capture_matches_original_simultaneous_environment(self):
        p=original_model([],1.1)
        for pp,ee in (([[0,0],[2,0]],[[.5,0],[2.5,0]]),([[0,0],[0,0]],[[.5,0],[.5,0]]),([[0,0],[8,8]],[[.99,0],[-8,-8]])):
            s=np.zeros((p.state_dim,1));s[:4,0]=np.array(ee).ravel();s[4:8,0]=np.array(pp).ravel();s[p.active_idxs,0]=1
            nxt=p.step(s,np.zeros((8,1)),1.)
            ap,ae,_=simultaneous_capture(np.array(pp,float),np.array(ee,float),np.ones(2,bool),np.ones(2,bool),np.zeros((0,2,2)))
            np.testing.assert_array_equal(ap,nxt[p.active_idxs[2:],0]>.5);np.testing.assert_array_equal(ae,nxt[p.active_idxs[:2],0]>.5)

    def test_team_conditioning_keeps_all_negative_sensors(self):
        points=np.array([[1.,0.],[9.,0.],[-9.,0.]])
        prediction=np.ones((3,9))/27
        result=condition_team(prediction,np.array([[0.,0.],[9.,1.]]),np.ones(2,bool),points,np.zeros((0,2,2)),-1)
        self.assertAlmostEqual(result[2].sum(),1.);self.assertEqual(result[:2].sum(),0.)

    def test_cartesian_density_runtime_gradient_and_inactive_marginal(self):
        c=self.controller();x=c.features(c.root);counts=[]
        with tempfile.TemporaryDirectory() as name:
            for kind in KINDS:
                torch.manual_seed(17);m=Network(kind).double();counts.append(sum(p.numel() for p in m.parameters()))
                path=Path(name)/(kind+'.npz');m.export(path);r=Runtime(path)
                tx={k:torch.from_numpy(v)[None] for k,v in x.items()}
                with torch.no_grad():params,value=m(tx)
                pp=np.exp(params[0].numpy()-params[0].numpy().max(-1,keepdims=True));pp/=pp.sum(-1,keepdims=True)
                expected=(pp[0],params[1].numpy()[0],params[2].numpy()[0]);actual,v=r(x)
                for a,b in zip(expected,actual):np.testing.assert_allclose(a,b,atol=1e-11,rtol=0)
                self.assertAlmostEqual(float(value[0]),v,places=10)
                actions=torch.tensor([[[-1.,1.,.1,.2],[1.,-1.,-.8,.7]]],dtype=torch.double)
                active=torch.tensor([[1.,0.]],dtype=torch.double)
                first=log_density(params,actions,active,kind)
                altered=actions.clone();altered[:,:,2:]=.99
                torch.testing.assert_close(first,log_density(params,altered,active,kind))
                tx.update(label_actions=actions,label_weights=torch.tensor([[.5,.5]],dtype=torch.double),label_value=torch.tensor([.7],dtype=torch.double))
                loss,_,_=losses(m,tx);loss.backward()
                self.assertTrue(torch.isfinite(loss));self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
                twice=augment(augment(tx,True,True),True,True)
                for k in tx:torch.testing.assert_close(tx[k],twice[k])
        self.assertLess(max(counts)/min(counts),1.025)

    def test_potential_changes_no_complete_episode_objective(self):
        raw=np.array([-.0025,-.0025,1-.0025]);phi=np.array([-25.,-21.,-9.,0.])
        shaped=raw+ETA*np.diff(phi)
        self.assertAlmostEqual(float(shaped.sum()),float(raw.sum()-ETA*phi[0]))


if __name__=='__main__':unittest.main()
