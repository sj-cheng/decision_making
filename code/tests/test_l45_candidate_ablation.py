"""Targeted tests for candidate controls, frozen suffixes and paired statistics."""
import copy
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_l45_candidate_ablation as m

class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg={'O':dict(models=str(m.SOURCE/'models/O'),model_hash=m.o.model_fingerprint(m.SOURCE/'models/O'))}

    def task(self,uid='G',layout='none',budget=3,time=76.):
        spec=m.o.make_scenario(56261028,layout);p,state=m.o.restore_scenario(spec)
        state[[p.active_idxs[0],p.active_idxs[2]],0]=0.;state[p.time_idx,0]=time
        state[p.state_idxs[1],0]=[-5.,0.];state[p.state_idxs[3],0]=[5.,0.]
        p._update_observation_memory(state,state,0.)
        requested=dict(search_budget=budget,search_only_budget=3,search_depth=100,beta_policy=.5,beta_value=0.)
        task=dict(key='test-'+uid,uid=uid,spec=spec,registry=self.reg,own_settings=requested,
            opponent_settings=requested,scene_id=spec['episode_hash'],stream=0,
            policy_id=uid,effective=m.effective(uid,self.reg),role=1,opponent='Search Only',timing_context='parallel',mode='search',historical=None)
        _,_,own,other=m.make_pair(task)
        stats=m.old.previous.initial_stats();stats.update(steps=int(time),first=time,last=time,capture_count=1)
        snap=m.old.previous.snapshot(p,state,stats,own,other)
        return task,p,state,own,other,snap

    def test_F_mean_scale_and_rng(self):
        task,p,state,own,_,_=self.task('F')
        paths,_=m.old.previous.research.h.oracle_paths(str(m.ROOT),iteration=45,model_dir=self.reg['O']['models'])
        from learning.gaussian_policy_network import GaussianPolicyNetwork
        actor=GaussianPolicyNetwork(p,3);actor.load_state_dict(m.o.load_torch(paths[3]))
        obs=torch.tensor(p.policy_encoding(state,3).reshape(-1),dtype=torch.float32)
        limits=torch.tensor(p.action_lims[p.action_idxs[3]],dtype=torch.float32)
        expected=m.o.distribution(actor,obs,limits).mean.detach().numpy()
        before=own.solver.get_rng_state(p)
        got=np.asarray(m.cpp.candidate_mean_debug(own.solver._problem_wrapper_for(p),own.solver.solver_wrapper,state,3)).reshape(-1)
        np.testing.assert_allclose(got,expected,atol=3e-6,rtol=0)
        self.assertEqual(before,own.solver.get_rng_state(p))
        probe=np.array([.2,-.4],dtype=np.float32);result=m.cpp.candidate_debug(own.solver._problem_wrapper_for(p),state,1,probe)
        upper=p.action_lims[p.action_idxs[3],1]
        np.testing.assert_allclose(np.asarray(result['action']).reshape(-1),probe/np.max(abs(probe)/upper),atol=1e-6)
        zero=m.cpp.candidate_debug(own.solver._problem_wrapper_for(p),state,1,[0.,0.])
        np.testing.assert_array_equal(zero['action'],0)

    def test_D_G_geometry_and_invalid_endpoint(self):
        _,p,state,own,_,_=self.task('G',layout='four')
        state[p.state_idxs[3],0]=[-4.5,-2.5];state[p.state_idxs[1],0]=[5.,-2.5]
        direct=m.cpp.candidate_debug(own.solver._problem_wrapper_for(p),state,2,[0.,0.])
        around=m.cpp.candidate_debug(own.solver._problem_wrapper_for(p),state,3,[0.,0.])
        self.assertFalse(direct['valid']);self.assertTrue(around['valid'])
        self.assertGreater(abs(np.asarray(around['action']).reshape(-1)[1]),.1)
        state[p.active_idxs[0],0]=1.
        inactive=m.cpp.candidate_debug(own.solver._problem_wrapper_for(p),state,3,[0.,0.])
        self.assertFalse(inactive['active'])

    def test_S_P_PR_frozen_suffix_and_rng_exact(self):
        for layout in ('none','four'):
            for u in ('S','P','PR'):
                task,p,state,own,other,snap=self.task(u,layout,budget=11)
                row,record,_,_=m.old.simulate(p,state,own,other,snap['stats'])
                p2,_,_,other2=m.make_pair(task)
                frozen=m.old.make_own(p2,self.reg,u,task['own_settings'],task['spec']['search_seeds'][1])
                frozen.solver.set_rng_state(p2,snap['own_rng']);other2.solver.set_rng_state(p2,snap['opponent_rng']);m.o.restore_rng(snap['python_rng'])
                row2,record2,_,_=m.old.simulate(p2,state.copy(),frozen,other2,snap['stats'])
                self.assertEqual(m.old.previous.science(row),m.old.previous.science(row2))
                self.assertEqual(m.old.previous.trajectory_digest(record['trajectory']),m.old.previous.trajectory_digest(record2['trajectory']))
                for side in ('own_rng','opponent_rng'):self.assertEqual(record['final'][side],record2['final'][side])

    def test_formal_1000_and_original_rollout_controls(self):
        for u in ('S','F','D','G','P','PR'):
            task,p,state,own,other,snap=self.task(u,budget=1000,time=30.)
            before=own.solver.get_rng_state(p);r=own.solver.search(p,state,2)
            self.assertEqual(r.num_visits,1000);self.assertEqual(r.root_audit[:,1].sum(),1000)
            np.testing.assert_array_equal(r.best_action[:6],0)
            if u in ('F','D','G'):
                self.assertGreater(r.guidance_stats.get('control_proposals',0),0)
                self.assertEqual(r.guidance_stats.get('guided_steps',0),0)
            if u in ('S','P','PR'):
                frozen=m.old.make_own(p,self.reg,u,task['own_settings'],task['spec']['search_seeds'][1])
                frozen.solver.set_rng_state(p,before);rr=frozen.solver.search(p,state,2)
                np.testing.assert_array_equal(r.best_action,rr.best_action)
                np.testing.assert_array_equal(r.root_audit,rr.root_audit)
                self.assertEqual(own.solver.get_rng_state(p),frozen.solver.get_rng_state(p))

    def test_no_obstacle_D_G_identical(self):
        records=[]
        for u in ('D','G'):
            task,p,state,own,other,snap=self.task(u,budget=30)
            row,d,_,_=m.old.simulate(p,state,own,other,snap['stats']);records.append((row,d))
        self.assertEqual(m.old.previous.science(records[0][0]),m.old.previous.science(records[1][0]))
        self.assertEqual(m.old.previous.trajectory_digest(records[0][1]['trajectory']),m.old.previous.trajectory_digest(records[1][1]['trajectory']))

    def test_precapture_all_match_S(self):
        task,p,state,own,other,_=self.task('S',budget=30)
        state[p.active_idxs[0],0]=1.;state[p.active_idxs[2],0]=1.;state[p.time_idx,0]=0.
        expected=own.solver.search(p,state,2);rng=own.solver.get_rng_state(p)
        for u in ('F','D','G','P','PR'):
            other=m.make_own(p,self.reg,u,task['own_settings'],task['spec']['search_seeds'][1])
            actual=other.solver.search(p,state,2)
            np.testing.assert_array_equal(expected.best_action,actual.best_action)
            self.assertEqual(rng,other.solver.get_rng_state(p))

    def test_restore_workers_cache_and_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks=[]
            for u in ('F','G'):
                task,_,_,_,_,snap=self.task(u,time=77.)
                path=Path(tmp)/f'{u}.pt';m.o.save_torch(path,snap)
                task.update(snapshot_path=str(path),snapshot_sha256=m.o.sha(path));tasks.append(task)
            serial=[m.evaluate_one(t) for t in tasks]
            with mp.get_context('spawn').Pool(2) as pool:parallel=pool.map(m.evaluate_one,tasks)
            for a,b in zip(serial,parallel):
                self.assertEqual(a['trajectory_sha256'],b['trajectory_sha256']);self.assertLessEqual(len(a['_record']['trajectory']),3)
            args=m.parse_args(['--smoke','--output',str(Path(tmp)/'output')]);item=dict(key=tasks[0]['key'],uid='F',task=tasks[0])
            m.commit_row(args,serial[0]);row,d=m.read_episode(args,item)
            self.assertEqual(row['trajectory_sha256'],parallel[0]['trajectory_sha256'])
            with mock.patch.object(m,'evaluate_one',side_effect=AssertionError('Must reuse committed task')):
                m.run(args,[item],lambda *a,**k:None)
            Path(row['episode_record']['path']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):m.read_cached(args,item)

    def test_statistics_cluster_streams_and_missing_pairs(self):
        rows=[]
        for scene in range(20):
            for stream in (0,1):
                for u in m.UIDS:
                    success=int(scene<(10 if u=='S' else 12))
                    rows.append(dict(uid=u,scene_id=str(scene),stream=stream,search_seeds=[scene,stream],
                        legal_double_capture=success,pursuer_illegal=0,evader_illegal=0,completion_cost=40 if success else 80,
                        first_capture_time=20.,capture_time=40. if success else float('nan'),
                        reused_historical=False,one_capture_timeout=1-success))
        all_cases=[dict(scene_id=str(i),eligible=True) for i in range(20)]
        a=m.analyze(rows,all_cases);b=m.analyze(list(reversed(rows)),all_cases)
        self.assertEqual(a,b);self.assertEqual(a['physical_scene_count'],20)
        self.assertAlmostEqual(a['comparisons']['F-S']['delta'],.1)
        self.assertEqual(a['comparisons']['D-F']['delta'],0.)
        with self.assertRaises(ValueError):m.analyze(rows[:-1],all_cases)
        with self.assertRaises(ValueError):m.analyze(rows+[rows[0]],all_cases)

if __name__=='__main__':unittest.main()
