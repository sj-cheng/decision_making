"""Behavioral/mathematical regressions for the ten audited blockers."""
import copy
import itertools
import json
import pickle
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from large_pursuit.core import Config,World,State,PublicMap,Belief,clear,move,visibility
from large_pursuit.methods import (Network,Controller,features,tensors,stack_features,evader_features,
    candidate_pool,simulated_step,METHODS,METHOD_BRANCH)
from large_pursuit.geometry import bottleneck_assignment,continuous_certificate
from large_pursuit.motion import shared_model,MotionKernel
from large_pursuit.semantics import team_reward,backed_return,STEP_COST

class Semantics(unittest.TestCase):
    def test_rare_lawful_negative_report_recovers_from_public_history(self):
        import json
        from pathlib import Path
        from large_pursuit.core import visibility
        fixture=json.loads((Path(__file__).parent/'fixtures/large_pursuit_rare_negative.json').read_text())
        public=PublicMap(Config(**fixture['cfg']),fixture['boxes'])
        belief=Belief(public,17)
        belief.rng.bit_generator.state=fixture['rng_state']
        belief.history=[{k:np.asarray(v) if isinstance(v,list) else v for k,v in row.items()}
                        for row in fixture['history']]
        target=fixture['target'];anchor=belief.history[0]['e'][target]
        samples,modes=belief._replay(target)
        self.assertGreater(len(samples),0)
        self.assertLessEqual(float(np.abs(samples-anchor).max()),1.)
        self.assertFalse(visibility(belief.history[-1]['p'],samples,public.boxes,public.cfg.sensing).any())
        self.assertTrue(((modes>=0)&(modes<9)).all())

    def test_impossible_negative_report_returns_without_hanging(self):
        public=PublicMap(Config(5,40,160),np.zeros((0,2,2)))
        belief=Belief(public,3,8)
        origin=np.zeros((5,2));seen=np.zeros(5,bool);seen[0]=True
        hidden=dict(p=origin.copy(),e=origin.copy(),alive=np.ones(5,bool),seen=seen.copy(),t=0)
        later=dict(hidden,seen=np.zeros(5,bool),t=1)
        belief.history=[hidden,later];belief.t=1
        started=__import__('time').perf_counter()
        samples,modes=belief._replay(0)
        self.assertLess(__import__('time').perf_counter()-started,30.)
        self.assertLessEqual(float(np.max(np.abs(samples))),1.)
        self.assertTrue(((modes>=0)&(modes<9)).all())

    def test_thin_sensor_exit_uses_a_fine_public_lattice(self):
        public=PublicMap(Config(5,40,160),np.zeros((0,2,2)))
        belief=Belief(public,5,8)
        anchor=np.array([5.05,0.])
        pursuers=np.zeros((5,2));pursuers[1:]=30.
        seen=np.zeros(5,bool);seen[0]=True
        positions=np.repeat(anchor.reshape(1,2),5,axis=0)
        belief.history=[
            dict(p=pursuers.copy(),e=positions.copy(),alive=np.ones(5,bool),seen=seen.copy(),t=0),
            dict(p=pursuers.copy(),e=positions.copy(),alive=np.ones(5,bool),seen=np.zeros(5,bool),t=1)]
        samples,modes=belief._replay(0)
        self.assertFalse(visibility(pursuers,samples[:1],public.boxes,public.cfg.sensing).any())
        self.assertLessEqual(float(np.max(np.abs(samples-anchor))),1.)
        self.assertTrue(((modes>=0)&(modes<9)).all())

    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(17)
        self.cfg=Config(5,40,160);self.public=PublicMap(self.cfg,[[[5.,6.],[5.,6.]]])
        rng=np.random.default_rng(17)
        self.state=State(rng.uniform(-4,4,(5,2)),rng.uniform(10,15,(5,2)),np.ones(5,bool))
        self.world=World(self.cfg,self.public.boxes);self.obs=self.world.observe(self.state)
        self.x=features(self.public,self.obs,self.state.e);self.model=Network().eval()

    def test_actor_receives_obstacles_and_gradient(self):
        a=tensors(self.x);b={k:v.clone() for k,v in a.items()};b['boxes']+=.4
        before=self.model(a)[0];after=self.model(b)[0]
        self.assertGreater(float((before-after).detach().abs().max()),1e-6)
        before.sum().backward()
        self.assertTrue(any(p.grad is not None and (p.grad!=0).any() for p in self.model.be.parameters()))

    def test_dead_target_mask_and_empty_terminal(self):
        a=tensors(self.x);a['alive'][0,0]=0.;a['target'][0,0,-1]=0.
        b={k:v.clone() for k,v in a.items()};b['target'][0,0,:2]+=100.
        torch.testing.assert_close(self.model(a)[0],self.model(b)[0],rtol=0,atol=0)
        b['alive'][:]=0
        self.assertTrue(all(torch.isfinite(t).all() for t in self.model(b)))

    def test_set_permutations_and_masked_padding(self):
        a=tensors(self.x);b={k:v.clone() for k,v in a.items()};order=torch.tensor([2,0,4,1,3])
        b['target']=b['target'][:,order];b['alive']=b['alive'][:,order];b['particles']=b['particles'][:,order];b['particle_mask']=b['particle_mask'][:,order]
        torch.testing.assert_close(self.model(a)[0],self.model(b)[0],atol=1e-6,rtol=1e-6)
        b={k:v.clone() for k,v in a.items()};b['boxes']=torch.cat((b['boxes'],torch.full((1,3,4),999.)),1)
        b['box_mask']=torch.cat((b['box_mask'],torch.zeros((1,3))),1)
        torch.testing.assert_close(self.model(a)[0],self.model(b)[0],atol=1e-6,rtol=1e-6)
        b={k:v.clone() for k,v in a.items()};b['p']=b['p'][:,order];b['actor_alive']=b['actor_alive'][:,order]
        torch.testing.assert_close(self.model(a)[0][:,order],self.model(b)[0],atol=1e-6,rtol=1e-6)

    def test_clock_reaches_evader_actor(self):
        earlier=evader_features(self.public,self.state);later=self.state.copy();later.t=100
        later=evader_features(self.public,later)
        self.assertNotEqual(earlier['time'],later['time'])
        self.assertGreater(float((self.model(tensors(earlier))[0]-self.model(tensors(later))[0]).detach().abs().max()),1e-6)

    def test_rectangular_bottleneck_before_sum(self):
        cost=np.array([[1.,6.],[6.,10.]])
        rows,cols=bottleneck_assignment(cost);self.assertEqual(cols.tolist(),[1,0])
        cost=np.array([[1,9],[9,3],[2,2]],float);rows,cols=bottleneck_assignment(cost)
        self.assertEqual(len(rows),2);self.assertEqual(len(set(rows)),2);self.assertEqual(len(set(cols)),2)
        self.assertEqual(float(cost[rows,cols].max()),2.)

    def test_continuous_component_detects_grid_invisible_pocket(self):
        c=Config(5,40,160)
        # Four closed walls enclose a .1x.1 pocket containing no 1m grid center.
        boxes=np.array([[[0,.1],[0,.4]],[[.2,.3],[0,.4]],[[0,.3],[0,.1]],[[0,.3],[.2,.3]]],float)
        with self.assertRaises(ValueError):continuous_certificate(c,boxes)
        good=np.array([[[0,.1],[0,.4]],[[0,.4],[.2005,.4]],[[0,.4],[0,.1995]]])
        self.assertEqual(continuous_certificate(c,good,[[.2,.2]])['components'],1)

    def test_reference_D_factorization_matches_legacy(self):
        from solvers.multimodal_estimator import make_graph
        from solvers.corner_estimator import map_probabilities
        from solvers.local_pursuit import PublicMap as LegacyMap
        boxes=np.array([[[-1.,1.],[-3.,3.]]]);anchor=np.array([-2.3,-.2])
        old=LegacyMap(boxes,1.);graph=make_graph(anchor,boxes)
        probabilities,_,_=map_probabilities(graph[0],graph[1],graph[2],anchor,boxes,old.vertices,old.dist)
        public=PublicMap(Config(2,20,80),boxes);model=shared_model(public)
        root=int(graph[5]);indices=[root]+list(graph[1][root,:min(3,graph[2][root])])
        for index in indices:
            for mode in (0,1,4,8):
                support,weight=model.distribution(tuple(graph[0][index]),tuple(anchor),mode)
                reference={tuple(np.round(graph[0][graph[1][index,k]],9)):probabilities[index,k,mode] for k in range(graph[2][index])}
                self.assertEqual(set(map(lambda p:tuple(np.round(p,9)),support)),set(reference))
                for q,w in zip(support,weight):self.assertAlmostEqual(float(w),float(reference[tuple(np.round(q,9))]),places=9)

    def test_posterior_joint_sampling_and_tree_transition(self):
        belief=Belief(self.public,17,16);belief.update(self.obs)
        for j in range(5):
            belief.samples[j,:,0]=np.arange(16)*.01+10
            belief.modes[j]=np.arange(16)%9
        before=pickle.dumps({k:v for k,v in belief.__dict__.items() if k!='public'})
        branch=belief.fork(29);targets,modes=branch.sample_joint(np.random.default_rng(1))
        for j in range(5):self.assertEqual(modes[j],int(round((targets[j,0]-10)*100))%9)
        state=State(self.state.p.copy(),targets,np.ones(5,bool),0)
        pa=np.zeros((5,2));nxt,_=simulated_step(self.public,branch,state,modes,pa,np.random.default_rng(2))
        replay=self.world.transition(state,pa,nxt.e-state.e)
        np.testing.assert_array_equal(replay.e,nxt.e);np.testing.assert_array_equal(replay.alive,nxt.alive)
        self.assertEqual(before,pickle.dumps({k:v for k,v in belief.__dict__.items() if k!='public'}))
        self.assertEqual(branch.t,1);np.testing.assert_array_equal(branch.ages(),np.ones(5,int))

    def test_hidden_worlds_produce_identical_actions(self):
        left=Controller(self.public,'JOINT',17,self.model,simulations=2,depth=2)
        right=Controller(self.public,'JOINT',17,self.model,simulations=2,depth=2)
        np.testing.assert_array_equal(left.act(self.obs),right.act(self.obs))
        a=self.state.copy();a.t=1;b=a.copy();b.e+=.2
        oa=self.world.observe(a);ob=self.world.observe(b)
        for key in oa:np.testing.assert_equal(oa[key],ob[key])
        np.testing.assert_array_equal(left.act(oa),right.act(ob))

    def test_team_and_episode_route_caches_are_isolated(self):
        left=Controller(self.public,'AG',17)
        right=Controller(self.public,'AG',17)
        motion=MotionKernel(self.public,np.random.default_rng(29))
        self.assertIs(left.public._route_dist,right.public._route_dist)
        self.assertIsNot(left.public._route_heads,right.public._route_heads)
        self.assertIsNot(left.public._route_tail,motion.public._route_tail)
        left.public.kernel_mean(self.state.p[0],self.state.e[0])
        before=(left.public._route_heads.cache_info(),left.public._route_tail.cache_info())
        motion.actions(self.state)
        self.assertEqual(before,(left.public._route_heads.cache_info(),left.public._route_tail.cache_info()))
        self.assertEqual(right.public._route_heads.cache_info().currsize,0)
        self.assertEqual(right.public._route_tail.cache_info().currsize,0)
        self.assertIsNot(shared_model(left.public),shared_model(motion.public))
        np.testing.assert_array_equal(left.belief.samples,right.belief.samples)

    def test_random_MCTS_has_no_geometry_proposal(self):
        import large_pursuit.methods as methods
        previous=methods.geometry_choices
        def fail(*a,**k):raise AssertionError('Geometry proposal in random MCTS')
        methods.geometry_choices=fail
        try:
            ctrl=Controller(self.public,'RANDOM_MCTS',17,simulations=8,depth=3);ctrl.act(self.obs)
            self.assertEqual(ctrl.last_stats['simulations'],8)
        finally:methods.geometry_choices=previous

    def test_same_pool_ordering_and_geom_not_alias(self):
        rng=np.random.default_rng(17);b=Belief(self.public,17,16);b.update(self.obs)
        pool=candidate_pool(self.public,self.state.p,b.estimates(),self.state.alive,rng,belief=b)
        arrays=[];orders=[]
        for name in ('POOL_UNIFORM','POOL_GEOM','POOL_LEARNED'):
            c=Controller(self.public,name,17,self.model,simulations=2,depth=2);c.act(self.obs,candidates=pool)
            arrays.append(c.last_candidates);orders.append(np.array([n.action for n in c.root.children]))
        for a in arrays:np.testing.assert_array_equal(a,pool)
        self.assertFalse(np.array_equal(orders[0],orders[1]))
        geom=Controller(self.public,'GEOM',17,simulations=2,depth=2);geom.act(self.obs)
        self.assertFalse(np.array_equal(geom.last_candidates,pool))

    def test_reward_value_units_and_no_teacher_critic_target(self):
        before=self.state.copy();before.alive[:]=[1,1,0,0,0];after=before.copy();after.alive[0]=False
        self.assertAlmostEqual(team_reward(self.cfg,before,after),1/5-STEP_COST)
        self.assertAlmostEqual(backed_return([.199,.199],.1),.498)
        from large_pursuit.training import supervised
        record=dict(x=self.x,pool=np.zeros((32,5,2),np.float32),scores=np.zeros(32,np.float32),action=np.zeros((5,2),np.float32))
        opt=torch.optim.Adam(self.model.parameters(),lr=1e-3)
        initial=copy.deepcopy(self.model.critic.state_dict());supervised(self.model,opt,[record],1,np.random.default_rng(1),'cpu')
        for k,v in self.model.critic.state_dict().items():torch.testing.assert_close(initial[k],v,rtol=0,atol=0)

    def test_validation_matrix_covers_all_maps_and_pool(self):
        from large_pursuit.training import validation_plan
        specs=[]
        for method in METHODS:
            for seed in ((17,29,43) if method in METHOD_BRANCH else (17,)):specs.append(dict(method=method,seed=seed))
        plan=validation_plan([str(i) for i in range(56)],specs)
        self.assertEqual(len(plan),112);self.assertEqual(len({(x['case'],x['initial']) for x in plan}),112)
        from collections import Counter
        counts=Counter(specs[x['opponent_index']]['method'] for x in plan)
        self.assertEqual(set(counts),set(METHODS));self.assertLessEqual(max(counts.values())-min(counts.values()),1)
        self.assertEqual({x['opponent_index'] for x in plan},set(range(len(specs))))

    def test_potential_telescopes_and_terminal_zero(self):
        from large_pursuit.semantics import shaped_reward,pursuit_potential,ETA
        raw=[.199,-.001,.199];phi=[-4.,-3.,-1.,0.]
        shaped=[shaped_reward(r,phi[i],phi[i+1]) for i,r in enumerate(raw)]
        self.assertAlmostEqual(backed_return(shaped),backed_return(raw)-ETA*phi[0])
        b=Belief(self.public,17,16);b.update(self.obs)
        terminal=self.state.copy();terminal.t=self.cfg.horizon
        self.assertEqual(pursuit_potential(self.public,terminal,b),0.)

    def test_rootpuct_expands_both_teams_with_legacy_parameters(self):
        from large_pursuit.rootpuct import rootpuct_evader
        action,stats=rootpuct_evader(self.public,self.state,np.random.default_rng(17),simulations=64,depth=8,return_stats=True)
        self.assertEqual(action.shape,(5,2));self.assertGreater(stats['evader_expansions'],0);self.assertGreater(stats['pursuer_expansions'],0)
        self.assertEqual((stats['C_pw'],stats['alpha_pw'],stats['C_exp'],stats['alpha_exp']),(2.,.5,1.,.25))

    def test_sensor_matches_existing_boundary_oracle(self):
        from e45b_sensor_audit import visible_batch
        path=Path(__file__).parent/'fixtures/e45b_sensor_boundary_v1.npz'
        with np.load(path) as data:pairs=data['pairs'];boxes=data['boxes']
        expected=visible_batch(pairs[:,:2],pairs[:,2:4],boxes)
        actual=np.array([visibility(a[None],b[None],boxes,6.)[0] for a,b in zip(pairs[:,:2],pairs[:,2:4])])
        np.testing.assert_array_equal(actual,expected)

    def test_packet_snapshots_do_not_alias_live_posterior(self):
        from large_pursuit.training import belief_state
        b=Belief(self.public,17,16);b.update(self.obs);saved=belief_state(b)
        expected=saved['samples'].copy();b.samples[:]=0.;b.last_seen[:]=90
        np.testing.assert_array_equal(saved['samples'],expected)
        self.assertFalse((saved['last_seen']==90).any())

    def test_validation_uses_actual_branch_and_all_opponent_methods(self):
        import large_pursuit.training as training
        from unittest.mock import patch
        cfg=Config(5,40,1);public=PublicMap(cfg,self.public.boxes)
        initial=dict(p=self.state.p.tolist(),e=self.state.e.tolist(),alive=[True]*5,t=0)
        case=dict(n=5,id='functional',initials=[initial,initial])
        calls=[];original=training.Controller
        def controller(*args,**kwargs):
            calls.append(args[1]);return original(*args,**kwargs)
        with tempfile.TemporaryDirectory() as folder,patch.object(training,'entries',return_value=['fixture']),patch.object(training,'read_case',return_value=(case,public)),patch.object(training,'Controller',side_effect=controller):
            job=dict(n=5,bank='unused',validation_split='validation',validation_maps=1,verification=True,validation_simulations=1,search_depth=1,folder=folder,role='pursuer',branch='fusion',opponents=[])
            result=training.validation(self.model,job,'cpu');self.assertEqual(calls,['FUSION','FUSION']);self.assertEqual(len(result['episodes']),2)
            calls.clear();job.update(role='evader',opponents=[dict(method='IG'),dict(method='AG')])
            training.validation(self.model,job,'cpu');self.assertEqual(set(calls),{'IG','AG'})

    def test_same_checkpoint_direct_aliases_do_not_change_pool_weights(self):
        from large_pursuit.protocol import POOL_METHODS,DIRECT_ALIASES
        from run_large_pursuit_full import checkpoint_specs
        results=[dict(n=5,seed=17,branch=b,checkpoint=b+'.pt') for b in ('prior','joint','fusion','rank')]
        specs=checkpoint_specs(results,5)
        self.assertEqual({s['method'] for s in specs},set(POOL_METHODS))
        self.assertFalse(set(DIRECT_ALIASES)&{s['method'] for s in specs})
        self.assertEqual(METHOD_BRANCH['NET_PRIOR'],METHOD_BRANCH['PRIOR']);self.assertEqual(METHOD_BRANCH['NET_FUSION'],METHOD_BRANCH['FUSION'])

    def test_legacy_cache_rejected_and_source_bound(self):
        from large_pursuit.provenance import bind_job
        from large_pursuit.training import load_model
        with tempfile.TemporaryDirectory() as folder:
            job=dict(kind='verification',team_steps=10)
            first=bind_job(folder,job);self.assertEqual(first,bind_job(folder,dict(job,gpu=3,cpus=[10])))
            with self.assertRaises(RuntimeError):bind_job(folder,dict(job,team_steps=11))
            old=Path(folder)/'old.pt';torch.save(dict(model=self.model.state_dict()),old)
            with self.assertRaises(ValueError):load_model(old)

if __name__=='__main__':unittest.main(verbosity=2)
