"""Behavioral tests for information isolation, belief planning and PPO."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark_baselines import configure_problem, build_solver
from local_observation import (BASE_DIM, OBS_DIM, TEAMS, TeamBelief, EpisodeSpec,
    observe, make_episode, random_map, connected_map, legal_points, visible_points,
    first_collision_point)
from learning.local_rl import PolicyBundle, LocalActor, advantages, joint_log_prob, critic_features
from solvers.belief_mcts import LocalController, BeliefMCTS, public_model, enable_fast_model
from train_search_rl import episode_seed, rollout_task, ppo_update

ROOT=Path(__file__).resolve().parents[2]


class LocalProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.bundle=PolicyBundle()

    def hidden_scene(self):
        p=configure_problem('none');p.swept_collision=True
        p.obstacles=[np.array([[-1.,1.],[-3.,3.]])]
        state=p.initialize()
        for robot,point in enumerate(((4.,-1.),(4.,1.),(-4.,-1.),(-4.,1.))):
            state[p.state_idxs[robot],0]=point
            state[p.active_idxs[robot],0]=1.
        state[p.time_idx,0]=0.
        return p,state

    def test_hidden_truth_does_not_change_sensor_belief_policy_or_search(self):
        p,state=self.hidden_scene();other=state.copy()
        other[p.state_idxs[0],0]=(5.,-.5);other[p.state_idxs[1],0]=(6.,2.)
        a,b=observe(p,state,1),observe(p,other,1)
        self.assertEqual(len(a.detections),0)
        np.testing.assert_array_equal(a.base,b.base)
        for cls,kwargs in [(LocalController,{}),(BeliefMCTS,dict(simulations=3,depth=2))]:
            left=cls(public_model(p.obstacles),self.bundle,1,99,**kwargs)
            right=cls(public_model(p.obstacles),self.bundle,1,99,**kwargs)
            la,li=left.act(a);ra,ri=right.act(b)
            np.testing.assert_array_equal(left.belief.particles,right.belief.particles)
            np.testing.assert_array_equal(la,ra)
            np.testing.assert_array_equal(li['features'],ri['features'])

    def test_particles_legal_negative_observation_and_recovery(self):
        p,state=self.hidden_scene();obs=observe(p,state,1)
        belief=TeamBelief(p.obstacles,p.state_lims[:2],1,4).update(obs)
        for t in (1.,2.,4.):
            belief.update(replace(obs,time=t))
            points=belief.particles[:,:,:2].reshape(-1,2)
            self.assertTrue(legal_points(points,p.obstacles,p.state_lims[:2]).all())
            self.assertFalse(belief.negative_visible(points,obs).any())
        belief.particles[:,:,:2]=obs.positions[0]
        belief.update(replace(obs,time=5.))
        self.assertGreater(belief.recovery_count,0)
        self.assertFalse(belief.negative_visible(belief.particles[:,:,:2].reshape(-1,2),obs).any())

    def test_reappearance_crossing_and_team_fusion(self):
        p,state=self.hidden_scene();obs=observe(p,state,1)
        belief=TeamBelief(p.obstacles,p.state_lims[:2],1,4)
        # Anonymous detections retain no simulator identity; reversed input
        # ordering after crossing must still produce a valid assignment.
        belief.update(replace(obs,detections=np.array([[-3.,6.],[-2.,6.]])))
        belief.update(replace(obs,time=1.,detections=np.empty((0,2))))
        belief.update(replace(obs,time=2.,detections=np.array([[-2.,6.],[-3.,6.]])))
        self.assertTrue(belief.visible.all());self.assertTrue((belief.age==0.).all())
        self.assertEqual(len(belief.reacquisition_times),2)
        self.assertTrue(np.isfinite(belief.batch(replace(obs,time=2.))).all())

    def test_inactive_and_reset_contract(self):
        p,state=self.hidden_scene();obs=observe(p,state,1)
        belief=TeamBelief(p.obstacles,p.state_lims[:2],1,4).update(obs)
        belief.update(replace(obs,time=1.,remaining=1,active=np.array([True,False])))
        self.assertEqual(int(belief.existence.sum()),1)
        with self.assertRaises(ValueError):belief.update(obs)
        with self.assertRaises(ValueError):belief.update(replace(obs,team=0,time=2.))
        fresh=TeamBelief(p.obstacles,p.state_lims[:2],1,4).update(obs)
        self.assertTrue((fresh.existence==1).all())

    def test_partial_team_initialization_and_public_speed_bounds(self):
        p,state=self.hidden_scene()
        state[p.state_idxs[0],0]=(-3.,6.)
        state[p.active_idxs[1],0]=0.;state[p.active_idxs[3],0]=0.
        obs=observe(p,state,1)
        self.assertEqual(len(obs.detections),1)
        ctrl=BeliefMCTS(public_model(p.obstacles),self.bundle,1,7,simulations=1,depth=1)
        ctrl.prepare(obs);hypothesis=ctrl.hypothesis(obs)
        self.assertTrue(ctrl.model.is_valid(hypothesis))
        p,state=self.hidden_scene();obs=observe(p,state,1)
        belief=TeamBelief(p.obstacles,p.state_lims[:2],1,9).update(obs)
        belief.particles[:,:,2:]=10.
        belief.update(replace(obs,time=1.))
        self.assertTrue((np.abs(belief.particles[:,:,2:])<=1.).all())

    def test_belief_rng_round_trip(self):
        p,state=self.hidden_scene();obs=observe(p,state,1)
        belief=TeamBelief(p.obstacles,p.state_lims[:2],1,9).update(obs)
        saved=belief.state_dict()
        belief.update(replace(obs,time=1.))
        restored=TeamBelief(p.obstacles,p.state_lims[:2],1,0);restored.load_state_dict(saved)
        restored.update(replace(obs,time=1.))
        np.testing.assert_array_equal(belief.particles,restored.particles)

    def test_episode_speed_map_and_state_round_trip(self):
        spec=make_episode(901,'random');p,state=spec.restore()
        rebuilt=EpisodeSpec(**json.loads(json.dumps(spec.__dict__)))
        q,other=rebuilt.restore()
        self.assertEqual(spec.fingerprint,rebuilt.fingerprint)
        np.testing.assert_array_equal(p.action_lims,q.action_lims)
        np.testing.assert_array_equal(p.control_lims,q.control_lims)
        np.testing.assert_array_equal(state,other)
        a=np.zeros((8,1));np.testing.assert_array_equal(p.step(state,a,1.),q.step(other,a,1.))

    def test_random_maps_connected_repeatable_and_disjoint(self):
        p=configure_problem()
        for seed in range(4):
            a=random_map(np.random.default_rng(seed),p.state_lims[:2])
            b=random_map(np.random.default_rng(seed),p.state_lims[:2])
            np.testing.assert_array_equal(a,b)
            self.assertTrue(connected_map(a,p.state_lims[:2]))
            self.assertTrue(2<=len(a)<=4)
        self.assertNotEqual(episode_seed(0,1,0,0),episode_seed(0,100,0,0))

    def test_native_geometry_matches_numpy_reference(self):
        from local_observation import clear_segments
        p,_=self.hidden_scene();rng=np.random.default_rng(72)
        points=rng.uniform(-9.,9.,(200,2));ends=rng.uniform(-9.,9.,(200,2));origins=np.array([[-5.,0.],[-5.,2.]])
        native=[legal_points(points,p.obstacles,p.state_lims[:2]),clear_segments(points,ends,p.obstacles),visible_points(points,origins,p.obstacles,28.3)]
        with patch('local_observation.native_geometry',return_value=None):
            reference=[legal_points(points,p.obstacles,p.state_lims[:2]),clear_segments(points,ends,p.obstacles),visible_points(points,origins,p.obstacles,28.3)]
        for a,b in zip(native,reference):np.testing.assert_array_equal(a,b)

    def test_corner_sweep_python_cpp_and_legacy(self):
        p=configure_problem('two');state=p.initialize()
        for i,point in enumerate(((-4.2,-1.8),(8.,7.),(-8.,-7.),(8.,-7.))):
            state[p.state_idxs[i],0]=point;state[p.active_idxs[i],0]=1.
        action=np.zeros((8,1));action[p.action_idxs[0],0]=(.4,.4)
        legacy=p.step(state,action,1.);self.assertTrue(p.is_valid(legacy))
        p.swept_collision=True;nxt=p.step(state,action,1.)
        self.assertFalse(p.is_valid(nxt));self.assertTrue(p.is_terminal(nxt))
        search=build_solver(p,'Search Only',str(ROOT),5)
        cpp=search._problem_wrapper_for(p)
        native=np.asarray(cpp.step(state,action,1.)).reshape(-1,1)
        np.testing.assert_allclose(native,nxt,atol=3e-5)
        np.testing.assert_allclose(np.asarray(cpp.transition_reward(state,action,native)).reshape(-1,1),p.transition_reward(state,nxt),atol=3e-5)
        for robot in range(4):
            np.testing.assert_allclose(np.asarray(cpp.policy_encoding(state,robot)).reshape(-1),p.policy_encoding(state,robot).reshape(-1),atol=3e-5)

    def test_cpp_cache_invalidates_map_and_speeds(self):
        p=configure_problem('two');solver=build_solver(p,'Search Only',str(ROOT),2)
        first=solver._problem_wrapper_for(p)
        p.obstacles=[];second=solver._problem_wrapper_for(p)
        self.assertIsNot(first,second)
        p.action_lims=p.action_lims*.9;third=solver._problem_wrapper_for(p)
        self.assertIsNot(second,third)

    def test_invalid_robot_is_not_credited_as_capture(self):
        p=configure_problem('two');p.swept_collision=True;state=p.initialize()
        for i,point in enumerate(((-4.,-2.5),(8.,7.),(-4.2,-2.5),(-8.,-7.))):
            state[p.state_idxs[i],0]=point;state[p.active_idxs[i],0]=1.
        self.assertFalse(p.is_valid(state));self.assertEqual(p.get_capture_pairs(state),[])
        cpp=build_solver(p,'Search Only',str(ROOT),1)._problem_wrapper_for(p)
        native=np.asarray(cpp.step(state,np.zeros((8,1)),1.)).reshape(-1,1)
        self.assertEqual(float(native[p.active_idxs[0],0]),1.)

    def test_nearest_semantic_return_does_not_rule_out_farther_object(self):
        from local_observation import TeamObservation
        p=configure_problem('none')
        base=np.ones((2,270),np.float32);base[:,264:]=0.
        base[0,9]=4./p.lidar_max_range()
        obs=TeamObservation(1,0.,np.array([[0.,0.],[0.,1.]]),np.array([True,False]),
                np.tile([[-1.,1.],[-1.,1.]],(2,1,1)),base,np.array([[4.,0.]]),2)
        belief=TeamBelief([],p.state_lims[:2],1,5)
        self.assertFalse(belief.negative_visible(np.array([[6.,0.]]),obs)[0])
        self.assertTrue(belief.negative_visible(np.array([[2.,0.]]),obs)[0])

    def test_quantized_positive_return_near_boundary_is_not_dropped(self):
        p=configure_problem('none');state=p.initialize()
        for robot,point in enumerate(((9.9,8.),(8.,-8.),(-9.9,-9.9),(-8.,-7.))):
            state[p.state_idxs[robot],0]=point
            state[p.active_idxs[robot],0]=float(robot in (0,2))
        obs=observe(p,state,1)
        self.assertEqual(len(obs.detections),1)
        self.assertTrue(legal_points(obs.detections,[],p.state_lims[:2]).all())
        belief=TeamBelief([],p.state_lims[:2],1,5).update(obs)
        self.assertTrue(np.isfinite(belief.batch(obs)).all())

    def test_negative_observation_matches_finite_beam_corner_occlusion(self):
        p=configure_problem('none');p.obstacles=[np.array([[-2.,-1.],[.25,1.]])]
        state=p.initialize()
        for robot,point in enumerate(((2.,.4),(8.,-8.),(-5.,0.),(-8.,-7.))):
            state[p.state_idxs[robot],0]=point;state[p.active_idxs[robot],0]=float(robot in (0,2))
        obs=observe(p,state,1)
        self.assertEqual(len(obs.detections),0)
        point=np.array([[2.,.4]])
        self.assertTrue(visible_points(point,[[-5.,0.]],p.obstacles,p.lidar_max_range())[0])
        belief=TeamBelief(p.obstacles,p.state_lims[:2],1,8)
        self.assertFalse(belief.negative_visible(point,obs)[0])
        with patch('local_observation.native_geometry',return_value=None):
            self.assertFalse(belief.negative_visible(point,obs)[0])

    def test_local_beam_boundaries_have_no_gaps_or_duplicate_returns(self):
        p=public_model([]);enable_fast_model(p)
        state=np.zeros((p.state_dim,1));state[p.active_idxs[0],0]=1.;state[p.active_idxs[2],0]=1.
        for origin in (np.zeros(2),np.array([-3.6058995723724365,-4.206657409667969])):
            state[p.state_idxs[2],0]=origin
            for beam in range(64):
                for epsilon in (-1e-8,0.,1e-8):
                    angle=(beam+.5)*2*np.pi/64+epsilon
                    point=origin+3.*np.array([np.cos(angle),np.sin(angle)])
                    state[p.state_idxs[0],0]=point
                    native=np.asarray(p._local_cpp.policy_encoding(state,2)).reshape(-1)
                    reference=p.policy_encoding(state,2).reshape(-1)
                    hits=np.flatnonzero(native[9:264:4]<1.-1e-6)
                    self.assertEqual(len(hits),1,(origin,beam,epsilon))
                    np.testing.assert_array_equal(hits,np.flatnonzero(reference[9:264:4]<1.-1e-6))
                    obs=observe(p,state,1)
                    self.assertEqual(len(obs.detections),1)
                    belief=TeamBelief([],p.state_lims[:2],1,2).update(obs)
                    self.assertFalse(belief.negative_visible(point[None],obs)[0])

    def test_warm_start_zero_added_weights_preserves_mean(self):
        from learning.gaussian_policy_network import GaussianPolicyNetwork
        p=configure_problem('two');s=p.initialize()
        legacy=GaussianPolicyNetwork(p,0,path=str(ROOT/'current/memory_observation_v3/models/model_policy_l45_i0.pt'))
        x=np.random.default_rng(8).normal(size=(5,OBS_DIM)).astype(np.float32)
        with torch.no_grad():
            old,_=legacy(torch.from_numpy(x[:,:BASE_DIM]),training=True)
            raw=self.bundle.actors[0].psi(torch.from_numpy(x))[:,:2]
        np.testing.assert_allclose(old.numpy(),raw.numpy(),atol=2e-6)

    def test_numpy_gaussian_matches_torch_and_raw_logprob(self):
        actor=self.bundle.actors[0]
        rng=np.random.default_rng(3);obs=rng.normal(size=(5,OBS_DIM)).astype(np.float32)
        limits=np.tile(np.array([[-.8,.8],[-.9,.9]],np.float32),(5,1,1))
        mean,std=actor.numpy_parameters(obs,limits)
        dist=actor.distribution(torch.from_numpy(obs),torch.from_numpy(limits))
        np.testing.assert_allclose(dist.mean.detach().numpy(),mean,atol=2e-6)
        np.testing.assert_allclose(dist.scale.detach().numpy(),std,atol=2e-6)
        clipped,raw,logp=actor.action(obs,limits,rng)
        np.testing.assert_allclose(dist.log_prob(torch.as_tensor(raw,dtype=torch.float32)).sum(-1).detach().numpy(),logp,atol=3e-5)
        self.assertTrue((clipped<=limits[...,1]).all())

    def test_native_features_and_gaussian_match_python(self):
        from cpp.build import bindings
        p,state=self.hidden_scene();obs=observe(p,state,1)
        belief=TeamBelief(p.obstacles,p.state_lims[:2],1,5).update(obs)
        native=bindings.local_belief_features(obs.base,obs.positions,obs.active.astype(np.float32),
              obs.limits.reshape(4,2),belief.particles.reshape(64,8),belief.age,belief.visible.astype(np.float32),belief.existence)
        reference=np.array([belief.features(obs,i) for i in range(2)])
        np.testing.assert_allclose(native,reference,atol=2e-6)
        x=torch.randn(4,OBS_DIM);actor=self.bundle.actors[0]
        limits=np.array([[-.8,.8],[-.9,.9]],np.float32)
        raw=actor.psi(x).detach().numpy()
        native=bindings.physical_gaussian_parameters(raw,limits)
        dist=actor.distribution(x,torch.tensor(np.tile(limits,(4,1,1))))
        np.testing.assert_allclose(native[:,:2],dist.mean.detach().numpy(),atol=2e-6)
        np.testing.assert_allclose(native[:,2:],dist.scale.detach().numpy(),atol=2e-6)

    def test_joint_ratio_one_and_inactive_mask(self):
        rng=np.random.default_rng(5)
        obs=torch.tensor(rng.normal(size=(4,2,OBS_DIM)),dtype=torch.float32)
        raw=torch.tensor(rng.normal(size=(4,2,2)),dtype=torch.float32)
        limits=torch.tensor(np.tile([[-1.,1.],[-1.,1.]],(4,2,1,1)),dtype=torch.float32)
        active=torch.tensor([[1.,0.]]*4)
        original=joint_log_prob(self.bundle.actors,obs,raw,limits,active,0)
        changed=raw.clone();changed[:,1]=100.
        altered=joint_log_prob(self.bundle.actors,obs,changed,limits,active,0)
        torch.testing.assert_close(original,altered)
        torch.testing.assert_close((original-original.detach()).exp(),torch.ones(4))

    def test_gae_no_cross_episode_bootstrap(self):
        adv,ret=advantages([1.,2.,3.],[0.,0.,0.],[False,True,True],gamma=1.,lam=1.,bootstrap=999.)
        np.testing.assert_array_equal(ret,[3.,2.,3.])
        _,truncated=advantages([1.],[0.],[False],bootstrap=2.)
        self.assertEqual(float(truncated[0]),3.)

    def test_search_budget_shared_and_action_legal(self):
        spec=make_episode(889,'two');p,state=spec.restore()
        ctrl=BeliefMCTS(public_model(spec.obstacles),self.bundle,1,7,simulations=5,depth=2)
        obs=observe(p,state,1);action,_=ctrl.act(obs)
        self.assertEqual(ctrl.last_stats['simulations'],5)
        self.assertLessEqual(ctrl.last_stats['transitions'],10)
        self.assertTrue(ctrl.safe(action,obs))

    def test_unvalidated_value_rejected(self):
        with self.assertRaises(ValueError):
            BeliefMCTS(public_model([]),self.bundle,0,beta_value=.1)

    def test_time_budget_bounds_completed_simulations(self):
        spec=make_episode(3,'two');p,state=spec.restore()
        ctrl=BeliefMCTS(public_model(spec.obstacles),self.bundle,1,7,simulations=100,depth=1,time_budget_ms=.001)
        ctrl.act(observe(p,state,1))
        self.assertEqual(ctrl.last_stats['simulations'],1)

    def test_policy_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.pt';self.bundle.save(path);restored=PolicyBundle.load(path)
            obs=np.zeros(OBS_DIM,np.float32);limits=np.array([[-1.,1.]]*2)
            np.testing.assert_array_equal(self.bundle.actors[0].action(obs,limits)[0],restored.actors[0].action(obs,limits)[0])

    def test_rollout_and_real_ppo_update_changes_parameters(self):
        seed=episode_seed(17,2,1,0)
        state=self.bundle.state_dict()
        result=rollout_task(dict(seed=seed,role=1,progress=0.,kind='ppo',bundle=state,opponent=state,budget=2,depth=1))
        self.assertTrue(result['rows'][-1]['done'])
        fresh=self.bundle.copy();anchor=fresh.copy()
        before=[p.detach().clone() for p in fresh.actors[2].parameters()]
        actor_opt=torch.optim.Adam([p for r in TEAMS[1] for p in fresh.actors[r].parameters()],lr=3e-5)
        critic_opt=torch.optim.Adam(fresh.critics[1].parameters(),lr=3e-4)
        args=type('Args',(),dict(anchor=.1,ppo_epochs=1,minibatch=64,clip=.1,entropy=.001))()
        metrics=ppo_update(fresh,anchor,1,[result],actor_opt,critic_opt,args,0.)
        self.assertTrue(np.isfinite(metrics['actor_loss']))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,fresh.actors[2].parameters())))
        self.assertEqual(critic_features(make_episode(1).restore()[0],np.asarray(result['spec']['initial_state']),result['rows'][0]['obs']).shape[0],635)


if __name__=='__main__':unittest.main()
