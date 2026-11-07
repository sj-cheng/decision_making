"""Meaningful phase/RNG/parity, cache recovery, statistics and adoption checks."""
import copy
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_l45_phase_handoff as m

quiet = lambda *a,**k: None


class PhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='l45_phase_tests_')
        cls.args = m.parse_args(['--smoke','--workers','2','--output',cls.temp.name])
        cls.banks = m.prepare(cls.args)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def own(self,uid='H'):
        spec = self.banks['audit_none'][0]
        p,state = m.o.restore_scenario(spec)
        own = m.make_own(p,m.registry(self.args),uid,1,m.settings(self.args),spec['search_seeds'][1])
        return p,state,own

    def late_items(self,uids=('H','O','L'),role=1):
        spec = copy.deepcopy(self.banks['audit_none'][0])
        p,state = m.o.restore_scenario(spec)
        state[p.time_idx,0] = p.tf-p.dt
        spec['initial_state'] = state.tolist()
        spec['episode_hash'] = m.o.json_hash({k:v for k,v in spec.items() if k != 'episode_hash'})
        return m.requests(self.args,uids,[spec],role=role)

    def test_fixed_protocol_and_original_opponent_configuration(self):
        args = m.parse_args([])
        cfg = m.config(args)
        self.assertEqual((cfg['budget'],cfg['search_only_budget'],cfg['depth']),(1000,3000,100))
        self.assertEqual((cfg['training_episodes'],cfg['teacher_queries'],cfg['optimizer_updates']),(0,0,0))
        self.assertEqual(cfg['candidate_phases'],['O','S'])
        self.assertEqual(cfg['bootstrap_seed'],48020261023)
        self.assertEqual(len(cfg['comparisons']),2)
        self.assertEqual(m.phase_pair('H',0),('O','O'))
        self.assertEqual(self.late_items(role=0)[0]['task']['opponent_settings']['beta_value'],.1)

    def test_scene_disjointness_including_archives_unused_banks(self):
        record = m.verify(self.args)
        scenes = [s for bank in self.banks.values() for s in bank]
        ids = [m.s.previous.physical_hash(s) for s in scenes]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertFalse(set(ids)&m.s.history_scenes([Path(p) for p in record['history']]))
        self.assertTrue(any('guarded_supervision' in p for p in record['history']))

    def test_live_original_action_episode_and_segment_audit(self):
        result = m.audit(self.args,self.banks)
        self.assertTrue(result['passed'])
        self.assertEqual(sum(r.get('episode_exact',False) for r in result['original_parity']),12)
        self.assertEqual(len(result['segmented_parity']),3)

    def test_rng_roundtrip_invalid_restore_atomic_and_switch_preserves_stream(self):
        p,state,own = self.own()
        before = own.get_rng_state(p)
        action = own.policy(p,state)
        own.set_rng_state(p,before)
        np.testing.assert_array_equal(action,own.policy(p,state))
        old_rng = own.get_rng_state(p)
        wrapper = own.solvers['O'].solver_wrapper
        for invalid in ('','bad','42 trailing'):
            with self.assertRaises(ValueError):
                wrapper.set_rng_state(invalid)
            self.assertEqual(old_rng,own.get_rng_state(p))
        state[[p.active_idxs[0],p.active_idxs[2]],0] = 0.
        # A separately restored S must receive exactly the old stream.
        expected = own.solvers['S']
        expected.set_rng_state(p,old_rng['rng'])
        expected_action = expected.policy(p,state,team=m.o.TEAMS[1])
        expected_rng = expected.get_rng_state(p)
        actual_action = own.policy(p,state)
        np.testing.assert_array_equal(actual_action,expected_action)
        self.assertEqual(own.get_rng_state(p)['rng'],expected_rng)
        self.assertTrue(own.last_switched)
        self.assertEqual(own.switches,1)
        own.policy(p,state)
        self.assertFalse(own.last_switched)
        self.assertEqual(own.switches,1)
        np.testing.assert_array_equal(actual_action[p.action_idxs[2]],0.)

    def test_prefix_H_equals_O_and_complete_evader_alias(self):
        p,state,h = self.own('H')
        _,_,o = self.own('O')
        for _ in range(3):
            np.testing.assert_array_equal(h.policy(p,state),o.policy(p,state))
        self.assertEqual(h.get_rng_state(p),o.get_rng_state(p))
        items = self.late_items(role=0)
        self.assertEqual(items[0]['key'],items[3]['key'])
        self.assertNotEqual(items[0]['key'],items[6]['key'])
        rows = m.run_evaluation(self.args,'evader_alias_test',items,None,quiet)
        self.assertEqual(m.science(rows[0]),m.science(rows[3]))
        self.assertEqual([r['uid'] for r in rows],['H']*3+['O']*3+['L']*3)

    def test_real_capture_next_decision_and_direct_double_no_suffix(self):
        p,state,h = self.own()
        positions = ([-3.,0.],[3.,0.],[-3.1,0.],[0.,3.])
        for r,xy in enumerate(positions):
            state[p.state_idxs[r],0] = xy
        stats = m.initial_stats()
        nxt = m.advance(p,state,np.zeros((p.action_dim,1)),stats)
        self.assertEqual(stats['capture_count'],1)
        self.assertEqual(h.active,'O')
        action = h.policy(p,nxt)
        self.assertEqual(h.active,'S')
        np.testing.assert_array_equal(action[p.action_idxs[2]],0.)
        nxt[p.state_idxs[3],0] = nxt[p.state_idxs[1],0]+[.05,0.]
        final = m.advance(p,nxt,np.zeros((p.action_dim,1)),stats)
        self.assertTrue(p.is_terminal(final))
        self.assertEqual(m.metrics(p,final,stats)['legal_double_capture'],1)
        p,state,h = self.own()
        for r,xy in enumerate(([-3.,0.],[3.,0.],[-3.1,0.],[3.1,0.])):
            state[p.state_idxs[r],0] = xy
        stats = m.initial_stats()
        final = m.advance(p,state,np.zeros((p.action_dim,1)),stats)
        self.assertTrue(p.is_terminal(final))
        self.assertEqual(stats['capture_count'],2)
        self.assertEqual(h.switches,0)

    def test_terminal_and_remaining_time_and_illegal_metrics(self):
        p,state,_ = self.own()
        state[p.time_idx,0] = p.tf-p.dt
        stats = m.initial_stats()
        final = m.advance(p,state,np.zeros((p.action_dim,1)),stats)
        self.assertTrue(p.is_terminal(final))
        self.assertEqual(m.metrics(p,final,stats)['survival_time'],80.)
        state[p.state_idxs[3][0],0] = 1000.
        stats = m.initial_stats()
        final = m.advance(p,state,np.zeros((p.action_dim,1)),stats)
        self.assertEqual(m.metrics(p,final,stats)['pursuer_illegal'],1)
        self.assertEqual(m.metrics(p,final,stats)['legal_double_capture'],0)

    def test_parallel_reproducibility_and_cache_resume_corruption(self):
        items = self.late_items()[:3]
        serial = [m.evaluate_one(i['task']) for i in items]
        with mp.get_context('spawn').Pool(2) as pool:
            parallel = pool.map(m.evaluate_one,[i['task'] for i in items])
        self.assertEqual([m.science(r) for r in serial],[m.science(r) for r in parallel])
        m.run_evaluation(self.args,'resume_test',items,None,quiet)
        with mock.patch.object(m,'evaluate_one',side_effect=AssertionError('must reuse cache')):
            m.run_evaluation(self.args,'resume_test',items,None,quiet)
        path = self.args.output/'cache'/(items[0]['key']+'.json')
        original = path.read_text()
        changed = m.o.read_json(path);changed['legal_double_capture']=9
        m.o.write_json(path,changed)
        try:
            with self.assertRaises(ValueError):
                m.read_cached(self.args,items[0])
        finally:
            path.write_text(original)

    def test_partial_validation_confirmation_commits_replay_only_missing(self):
        worker=m.evaluate_one
        for stage in ('partial_validation','partial_confirmation'):
            items=self.late_items()
            for item in items:
                item['task']['timing_context']=stage
                identity={k:item['task'][k] for k in ('protocol','policy_id','episode_hash','role','opponent',
                    'timing_context','own_settings','opponent_settings','mode')}
                item['key']=item['task']['key']=m.o.json_hash(identity)
            calls=[]
            def fail_second(task):
                calls.append(task['key'])
                if len(calls)==2:raise RuntimeError('injected after first atomic commit')
                return worker(task)
            with mock.patch.object(m,'evaluate_one',side_effect=fail_second):
                with self.assertRaises(RuntimeError):m.run_evaluation(self.args,stage,items,None,quiet)
            committed=items[0]['key']
            self.assertTrue((self.args.output/'cache_checksums'/(committed+'.lock.json')).exists())
            resumed=[]
            def remaining(task):
                self.assertNotEqual(task['key'],committed)
                resumed.append(task['key']);return worker(task)
            with mock.patch.object(m,'evaluate_one',side_effect=remaining):
                rows=m.run_evaluation(self.args,stage,items,None,quiet)
            self.assertEqual(len(resumed),len(items)-1)
            self.assertEqual(len(rows),len(items))

    def test_private_failure_nonfinite_does_not_return_an_action(self):
        p,state,own=self.own('O')
        from solvers import phase_puct
        for result in (SimpleNamespace(success=False,best_action=np.zeros(8)),
                       SimpleNamespace(success=True,best_action=np.full(8,np.nan))):
            with mock.patch.object(phase_puct.cpp,'cpp_search',return_value=result):
                with self.assertRaises(RuntimeError):own.policy(p,state)

    def test_prefix_suffix_resume_and_tamper(self):
        tasks = m.diagnostic_tasks(self.args)[:2]
        m.run_segments(self.args,'prefix_test',tasks,m.prefix_one,None,quiet)
        with mock.patch.object(m,'prefix_one',side_effect=AssertionError('cache')):
            m.run_segments(self.args,'prefix_test',tasks,m.prefix_one,None,quiet)
        suffixes = m.suffix_tasks(self.args,tasks)
        with mock.patch.object(m,'suffix_one',side_effect=RuntimeError('injected interruption')):
            with self.assertRaises(RuntimeError):
                m.run_segments(self.args,'suffix_test',suffixes,m.suffix_one,None,quiet)
        rows = m.run_segments(self.args,'suffix_test',suffixes,m.suffix_one,None,quiet)
        self.assertEqual(len(rows),4)
        path = Path(tasks[0]['result']);original=path.read_bytes()
        path.write_bytes(b'broken')
        try:
            with self.assertRaises(ValueError):
                m.read_segment(tasks[0])
        finally:
            path.write_bytes(original)

    def synthetic(self,role=1):
        rows=[]
        for uid in ('H','O','L'):
            for layout in ('four','none'):
                for i in range(100):
                    success=int(i<(80 if uid=='H' else 50))
                    for op in m.o.OPPONENTS:
                        rows.append(dict(uid=uid,role=role,layout=layout,episode_hash=f'{layout}{i}',opponent=op,
                            search_seeds=[i,i+1],legal_double_capture=success,legal_survival=success,
                            pursuer_illegal=0,evader_illegal=0,completion_cost=40 if success else 80,one_capture_timeout=1-success))
        return rows

    def test_statistics_two_predeclared_physical_scene_comparisons(self):
        rows=self.synthetic()
        stats=m.statistics(rows,draws=2000)
        self.assertEqual(stats['bootstrap_seed'],48020261023)
        self.assertEqual([r['id'] for r in stats['comparisons']],['H-O','H-L'])
        self.assertGreater(stats['comparisons'][0]['simultaneous_lower95'],0.)
        self.assertTrue(m.comparison(rows,'O',stats['comparisons'][0])['passed'])
        self.assertFalse(m.comparison(rows,'O',dict(simultaneous_lower95=0.))['passed'])
        with self.assertRaises(ValueError):
            m.statistics(rows[1:],draws=20)
        next(r for r in rows if r['uid']=='L' and r['layout']=='four')['search_seeds']=[999,999]
        with self.assertRaises(ValueError):
            m.statistics(rows,draws=20)

    def test_each_reference_opponent_and_map_constraints(self):
        rows=self.synthetic()
        for row in rows:
            if row['uid']=='H' and row['layout']=='none':
                row['pursuer_illegal']=1
        result=m.comparison(rows,'O')
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['no_obstacle_own_illegal'])

    def test_source_protocol_scene_tamper_stops(self):
        changed=copy.copy(self.args);changed.smoke=False
        with self.assertRaises(ValueError):m.verify(changed)
        path=self.args.output/'scenarios.json';original=path.read_text();path.write_text('{}')
        try:
            with self.assertRaises(ValueError):m.verify(self.args)
        finally:path.write_text(original)

    def test_confirmation_and_regression_failure_fallback_and_unauthorized_stages(self):
        for failure in ('confirmation','guard'):
            with tempfile.TemporaryDirectory(prefix='l45_phase_gate_failure_') as path:
                args=copy.copy(self.args);args.output=Path(path);args.smoke=False
                shutil.copytree(self.args.output/'models',args.output/'models')
                m.lock_json(args.output/'experiment.json',dict(test=True))
                m.lock_json(args.output/'diagnostic_result.json',dict(passed=True,summary={},pairs={}))
                val=dict(passed=True,aggregates={uid:m.v.aggregate(self.synthetic(),uid) for uid in ('H','O','L')},
                    comparisons={ref:m.comparison(self.synthetic(),ref) for ref in ('O','L')})
                m.lock_json(args.output/'validation_result.json',val)
                m.lock_json(args.output/'confirmation_result.json',dict(passed=failure!='confirmation'))
                if failure=='guard':m.lock_json(args.output/'guard_result.json',dict(passed=False))
                with mock.patch.object(m,'verify',return_value={}),mock.patch.object(m,'replay',return_value=dict(passed=True)):
                    result=m.finish(args,self.banks)
                self.assertEqual(result['recommended'],'O')
                self.assertEqual(result['reason'],failure+'_failed')

    def test_downstream_refuses_missing_gates(self):
        for function in (m.validate,m.confirm,m.guard):
            with self.assertRaises(FileNotFoundError):function(self.args,self.banks,None,quiet)

    def test_controlled_confirm_guard_timing_export_and_fallback(self):
        # Controlled outcomes exercise otherwise conditional branches without
        # reading any real unused confirmation scenario.
        with tempfile.TemporaryDirectory(prefix='l45_phase_branch_') as path:
            args=copy.copy(self.args);args.output=Path(path);args.smoke=False
            shutil.copytree(self.args.output/'models',args.output/'models')
            m.lock_json(args.output/'experiment.json',dict(test=True))
            m.lock_json(args.output/'diagnostic_result.json',dict(passed=True,summary={},pairs={}))
            with mock.patch.object(m,'run_evaluation',return_value=self.synthetic()):
                self.assertTrue(m.validate(args,self.banks,None,quiet)['passed'])
            with mock.patch.object(m,'run_evaluation',return_value=self.synthetic()):
                confirmed=m.confirm(args,self.banks,None,quiet)
            self.assertTrue(confirmed['passed'])
            with mock.patch.object(m,'run_evaluation',return_value=self.synthetic(role=0)):
                guarded=m.guard(args,self.banks,None,quiet)
            self.assertTrue(guarded['passed'])
            timing_rows=[dict(uid=uid,opponent=op,decision_latency_ms=[[1.,2.,3.],[3.,2.,5.]],
                switch_decision_ms=[3.] if uid=='H' else [],episode_wall_ms=20.) for uid in ('H','O','L') for op in m.o.OPPONENTS]
            with mock.patch.object(m,'run_evaluation',return_value=timing_rows),mock.patch.object(m,'evaluate_one',return_value={}):
                timed=m.timing(args,self.banks,quiet)
            self.assertEqual(timed['H']['pursuer_mean_ms'],2.)
            self.assertEqual(timed['H']['switch_decisions'],3)
            with mock.patch.object(m,'verify',return_value={}),mock.patch.object(m,'replay',return_value=dict(passed=True)):
                result=m.finish(args,self.banks)
            self.assertEqual(result['recommended'],'H')
            self.assertTrue(result['quality_gate_passed'])
            self.assertEqual(m.read_recommended(args)['pursuer_phases'],['O','S'])
            self.assertTrue(m.replay(args,self.banks)['passed'])
        with tempfile.TemporaryDirectory(prefix='l45_phase_fallback_') as path:
            args=copy.copy(self.args);args.output=Path(path)
            shutil.copytree(self.args.output/'models',args.output/'models')
            m.lock_json(args.output/'experiment.json',dict(test=True))
            m.lock_json(args.output/'diagnostic_result.json',dict(passed=False,summary={},pairs={}))
            with mock.patch.object(m,'verify',return_value={}),mock.patch.object(m,'replay',return_value=dict(passed=True)):
                result=m.finish(args,self.banks)
            self.assertEqual(result['recommended'],'O')
            self.assertFalse(result['quality_gate_passed'])
            self.assertEqual(m.read_recommended(args)['pursuer_phases'],['O','O'])


if __name__=='__main__':unittest.main()
