"""Geometry, frozen-search parity, phase activation and paired development tests."""
import copy
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_l45_interception_guidance as m


class InterceptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg={'O':dict(models=str(m.SOURCE/'models/O'),model_hash=m.o.model_fingerprint(m.SOURCE/'models/O'))}

    def task(self,uid='PR',layout='none',late=False,budget=3):
        spec=m.o.make_scenario(55261024,layout)
        if late:
            p,state=m.o.restore_scenario(spec)
            state[p.time_idx,0]=77.;state[[p.active_idxs[0],p.active_idxs[2]],0]=0.
            state[p.state_idxs[1],0]=[-5.,0.];state[p.state_idxs[3],0]=[5.,0.]
            spec['initial_state']=state.tolist();spec['episode_hash']=m.o.json_hash({k:v for k,v in spec.items() if k!='episode_hash'})
        args=m.parse_args(['--smoke']);own=m.settings(args);own['search_budget']=budget
        return dict(key='test-'+uid,uid=uid,registry=self.reg,spec=spec,role=1,opponent='Search Only',
            policy_id='test-'+uid,timing_context='parallel',mode='search',own_settings=own,
            opponent_settings=m.settings(args,0),effective=m.effective(uid,self.reg))

    def geometry(self,layout='none',e=(-5.,0.),pursuer=(5.,0.),time=10.):
        task=self.task(layout=layout);p,state,own,other=m.make_pair(task)
        state[[p.active_idxs[0],p.active_idxs[2]],0]=0.
        state[p.state_idxs[1],0]=e;state[p.state_idxs[3],0]=pursuer;state[p.time_idx,0]=time
        return p,state,own

    def test_axis_limited_time_and_first_action(self):
        p,state,own=self.geometry()
        goal=np.array([-1.3,3.7]);start=state[p.state_idxs[3],0]
        upper=p.action_lims[p.action_idxs[3],1]
        data=m.cpp.interception_debug(own.solver._problem_wrapper_for(p),state,goal,3)
        expected=int(np.ceil(np.max(np.abs(goal-start)/upper)-1e-6))
        self.assertEqual(data['steps'],expected)
        action=np.asarray(data['action']).reshape(-1)
        self.assertTrue(np.all(np.abs(action)<=upper+1e-6))
        self.assertTrue(data['valid'])
        self.assertEqual(len(data['path']),1)

    def test_obstacle_detour_path_segments_legal(self):
        p,state,own=self.geometry('four',e=(5.,-2.5),pursuer=(-6.,-2.5))
        goal=np.array([5.,-2.5]);data=m.cpp.interception_debug(own.solver._problem_wrapper_for(p),state,goal,3)
        self.assertTrue(data['valid']);self.assertGreater(len(data['path']),1)
        from local_observation import segment_interval
        start=state[p.state_idxs[3],0]
        for point in data['path']:
            self.assertTrue(all(segment_interval(start,point,box) is None for box in p.obstacles))
            start=point

    def test_remaining_time_inactive_and_arrived_goal(self):
        p,state,own=self.geometry(time=79.)
        data=m.cpp.interception_debug(own.solver._problem_wrapper_for(p),state,[5.,0.],3)
        self.assertEqual(data['steps'],0);np.testing.assert_array_equal(data['action'],0)
        self.assertEqual(data['routes'].shape[0],0)
        state[p.active_idxs[0],0]=1.
        data=m.cpp.interception_debug(own.solver._problem_wrapper_for(p),state,[0.,0.],3)
        self.assertFalse(data['active'])

    def test_private_B_S_match_frozen_whole_episodes(self):
        for layout in ('none','four'):
            for uid in ('B','S'):
                task=self.task(uid,layout);m.previous.seed_task(task)
                p,state,own,other=m.make_pair(task)
                reg=dict(O=self.reg['O'],L=self.reg['O'])
                frozen=m.previous.make_own(p,reg,'B' if uid=='B' else 'D',1,
                    task['own_settings'],task['spec']['search_seeds'][1])
                row,detail,_,_=m.simulate(p,state,own,other)
                m.previous.seed_task(task);p,state,_,other=m.make_pair(task)
                row2,detail2,_,_=m.simulate(p,state,frozen,other)
                self.assertEqual(m.previous.science(row),m.previous.science(row2))
                self.assertEqual(m.previous.trajectory_digest(detail['trajectory']),m.previous.trajectory_digest(detail2['trajectory']))
                self.assertTrue(m.s.equal_tree(detail['final']['own_rng'],detail2['final']['own_rng']))
                self.assertTrue(m.s.equal_tree(detail['final']['opponent_rng'],detail2['final']['opponent_rng']))

    def test_switches_off_and_pre_capture_calls_are_exact(self):
        for late in (False,True):
            task=self.task('S',late=late,budget=30);p,state,base,other=m.make_pair(task)
            expected=base.solver.search(p,state,2)
            expected_rng=base.solver.get_rng_state(p)
            for uid in (('OFF',) if late else ('OFF','P','R','PR')):
                p2,state2,_,_=m.make_pair(task)
                raw=m.make_own(p2,self.reg,uid,task['own_settings'],task['spec']['search_seeds'][1]).solver
                actual=raw.search(p2,state2,2)
                np.testing.assert_array_equal(expected.best_action,actual.best_action)
                np.testing.assert_array_equal(expected.root_audit,actual.root_audit)
                self.assertEqual(expected_rng,raw.get_rng_state(p2))
                self.assertEqual(expected.step_calls,actual.step_calls)

    def test_formal_budget_proposal_sources_and_rollout_prefix(self):
        task=self.task('PR',late=True,budget=1000);p,state,own,other=m.make_pair(task)
        state[p.time_idx,0]=20.
        result=own.solver.search(p,state,2);g=result.guidance_stats
        self.assertEqual(result.num_visits,1000);self.assertEqual(result.root_audit[:,1].sum(),1000)
        self.assertEqual(g['active'],1);self.assertGreater(g['routes'],0)
        self.assertGreater(g.get('root_source_2',0),0)
        self.assertGreater(g.get('guided_steps',0),0);self.assertLessEqual(g['max_guided_prefix'],8)
        self.assertTrue(np.all(result.best_action[p.action_idxs[2]]==0))
        self.assertTrue(np.all(result.best_action[:4]==0))
        ratio=g['guided_rollouts']/g['rollouts'];self.assertGreater(ratio,.72);self.assertLess(ratio,.88)
        self.assertLess(abs(g['root_source_1']/g['root_candidates']-.5),.2)
        self.assertLess(abs(g['root_source_2']/g['root_candidates']-.25),.2)

    def test_rng_roundtrip_and_stats_do_not_consume_randomness(self):
        task=self.task('PR',late=True);p,state,own,other=m.make_pair(task)
        before=own.solver.get_rng_state(p);own.solver.set_rng_state(p,before)
        self.assertEqual(before,own.solver.get_rng_state(p))
        a=own.solver.search(p,state,2);rng=own.solver.get_rng_state(p)
        own.solver.set_rng_state(p,before)
        b=own.solver.search(p,state,2)
        np.testing.assert_array_equal(a.root_audit,b.root_audit)
        np.testing.assert_array_equal(a.guidance_routes,b.guidance_routes)
        self.assertEqual(rng,own.solver.get_rng_state(p))
        self.assertEqual(a.guidance_stats.get('guided_steps'),b.guidance_stats.get('guided_steps'))

    def test_worker_count_and_snapshot_resume(self):
        tasks=[self.task(u,late=True) for u in ('P','R','PR')]
        serial=[m.evaluate_one(t) for t in tasks]
        with mp.get_context('spawn').Pool(2) as pool:parallel=pool.map(m.evaluate_one,tasks)
        for a,b in zip(serial,parallel):
            self.assertEqual(m.previous.science(a),m.previous.science(b))
            self.assertEqual(a['trajectory_sha256'],b['trajectory_sha256'])
        task=tasks[2];row=serial[2];d=row['_record'];snap=d['initial']
        p,state,own,other=m.make_pair(task)
        own.solver.set_rng_state(p,snap['own_rng']);other.solver.set_rng_state(p,snap['opponent_rng']);m.o.restore_rng(snap['python_rng'])
        restored,detail,_,_=m.simulate(p,snap['state'].copy(),own,other,snap['stats'])
        self.assertEqual(m.previous.science(row),m.previous.science(restored))
        self.assertEqual(row['trajectory_sha256'],m.previous.trajectory_digest(detail['trajectory']))

    @staticmethod
    def rows(n=100):
        return [dict(uid=u,episode_hash=str(i),search_seeds=[i,i+1],legal_double_capture=int(i<(60 if u in ('B','S') else 66)),
            pursuer_illegal=0,capture_count=2 if i<60 else 1,one_capture_timeout=int(i>=60),completion_cost=40.)
            for u in m.UIDS for i in range(n)]

    def test_development_gate_selection_and_statistics(self):
        rows=self.rows();result=m.selection(rows)
        self.assertTrue(result['passed']);self.assertEqual(result['selected'],'P')
        self.assertEqual(m.statistics(rows),m.statistics(rows))
        self.assertEqual(m.statistics(rows)['comparisons']['interaction']['delta'],-.06)
        changed=copy.deepcopy(rows)
        for r in changed:
            if r['uid'] in ('P','R','PR') and int(r['episode_hash'])==65:r['legal_double_capture']=0
        self.assertTrue(m.selection(changed)['passed'])
        for r in changed:
            if r['uid'] in ('P','R','PR') and int(r['episode_hash'])==64:r['legal_double_capture']=0
        self.assertFalse(m.selection(changed)['passed'])
        self.assertFalse(m.selection(rows,True)['passed'])
        with self.assertRaises(ValueError):m.selection(rows[:-1])
        with self.assertRaises(ValueError):m.selection(rows+[rows[0]])

    def test_cache_commit_integrity_and_reuse(self):
        task=self.task('PR',late=True);row=m.evaluate_one(task)
        with tempfile.TemporaryDirectory() as tmp:
            args=m.parse_args(['--smoke','--output',tmp]);item=dict(key=task['key'],uid=task['uid'],task=task)
            m.commit_row(args,row);loaded,d=m.read_episode(args,item)
            self.assertEqual(loaded['trajectory_sha256'],m.previous.trajectory_digest(d['trajectory']))
            with mock.patch.object(m,'evaluate_one',side_effect=AssertionError('Unexpected new execution')):
                m.run_evaluation(args,'pilot',[item],None,lambda *a,**kw:None)
            artifact=Path(loaded['episode_record']['path']);artifact.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):m.read_cached(args,item)

if __name__=='__main__':unittest.main()
