"""AQ phase semantics, RNG/parity, corruption, gates and end-to-end export."""
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
import run_l45_aq_once as m
from solvers.aq_once import select_aq
quiet=lambda *a,**k:None

class AQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
            cls.temp = tempfile.TemporaryDirectory(prefix='l45_aq_tests_')
            cls.args = m.parse_args(['--smoke','--workers','2','--output',cls.temp.name])
            cls.banks = m.prepare(cls.args)

    @classmethod
    def tearDownClass(cls):
            cls.temp.cleanup()

    def late_items(self,uids=('AQ','O','L'),role=1):
            spec = copy.deepcopy(self.banks['audit_none'][0])
            p,state = m.o.restore_scenario(spec)
            state[p.time_idx,0] = p.tf-p.dt
            spec['initial_state'] = state.tolist()
            spec['episode_hash'] = m.o.json_hash({k:v for k,v in spec.items() if k != 'episode_hash'})
            return m.requests(self.args,uids,[spec],role=role)

    def test_scene_disjointness_including_archives_unused_banks(self):
            record = m.verify(self.args)
            scenes = [s for bank in self.banks.values() for s in bank]
            ids = [m.s.previous.physical_hash(s) for s in scenes]
            self.assertEqual(len(ids),len(set(ids)))
            self.assertFalse(set(ids)&m.s.history_scenes([Path(p) for p in record['history']]))
            self.assertTrue(any('guarded_supervision' in p for p in record['history']))

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

    def synthetic(self,role=1):
            rows=[]
            for uid in ('AQ','O','L'):
                for layout in ('four','none'):
                    for i in range(100):
                        success=int(i<(80 if uid=='AQ' else 50))
                        for op in m.o.OPPONENTS:
                            rows.append(dict(uid=uid,role=role,layout=layout,episode_hash=f'{layout}{i}',opponent=op,
                                search_seeds=[i,i+1],legal_double_capture=success,legal_survival=success,
                                pursuer_illegal=0,evader_illegal=0,completion_cost=40 if success else 80,one_capture_timeout=1-success))
            return rows

    def test_statistics_two_predeclared_physical_scene_comparisons(self):
            rows=self.synthetic()
            stats=m.statistics(rows,draws=2000)
            self.assertEqual(stats['bootstrap_seed'],50020261024)
            self.assertEqual([r['id'] for r in stats['comparisons']],['AQ-O','AQ-L'])
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
                if row['uid']=='AQ' and row['layout']=='none':
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

    def own(self,uid='AQ',role=1):
        spec=self.banks['audit_none'][0]
        p,state=m.o.restore_scenario(spec)
        own=m.make_own(p,m.registry(self.args),uid,role,m.settings(self.args,role),spec['search_seeds'][role])
        return p,state,own

    def test_fixed_protocol(self):
        c=m.config(m.parse_args([]))
        self.assertEqual((c['budget'],c['search_only_budget'],c['depth']),(1000,3000,100))
        self.assertEqual((c['training_episodes'],c['teacher_queries'],c['optimizer_updates']),(0,0,0))
        self.assertEqual(c['candidate'],'AQ')
        self.assertEqual(c['bootstrap_seed'],50020261024)
        self.assertEqual(c['request_caps'],dict(validation=1350,confirmation=4500,evader_regression=1800,serial_timing=270))
        self.assertEqual(self.late_items(role=0)[0]['task']['opponent_settings']['beta_value'],.1)

    def test_all_historical_roots_match_frozen_AQ_selection_without_new_search(self):
        files=list((m.SOURCE/'roots').glob('*.pt'))
        self.assertEqual(len(files),99)
        for path in files:
            root=m.a.read_pt(path)
            p,_=m.o.restore_scenario(root['task']['row']['spec'])
            expected=m.a.selected_candidates(root['root_table'],p,root['state'])
            chosen=expected[min(1,len(expected)-1)]
            actual,event=select_aq(root['root_table'],p,root['state'])
            np.testing.assert_array_equal(actual,chosen['action'])
            self.assertEqual(event['aq_ordinal'],chosen['ordinal'])

    def test_rng_single_search_single_override_inactive_and_restore(self):
        p,state,own=self.own()
        _,_,ordinary=self.own('O')
        # Both pre-capture actions and streams must match, even with root export enabled.
        for _ in range(3):
            np.testing.assert_array_equal(own.policy(p,state),ordinary.policy(p,state))
            self.assertEqual(own.get_rng_state(p)['rng'],ordinary.get_rng_state(p)['rng'])
        state[[p.active_idxs[0],p.active_idxs[2]],0]=0.
        before=own.get_rng_state(p)
        action=own.policy(p,state)
        self.assertEqual(own.solver.calls,4)
        expected,_=select_aq(own.solver.last_result.root_audit,p,state)
        np.testing.assert_array_equal(action,expected)
        ordinary.policy(p,state)
        self.assertEqual(own.get_rng_state(p)['rng'],ordinary.get_rng_state(p)['rng'])
        self.assertEqual(own.switches,1)
        self.assertTrue(own.last_switched)
        after=own.get_rng_state(p)
        own.set_rng_state(p,before)
        np.testing.assert_array_equal(action,own.policy(p,state))
        self.assertEqual(after,own.get_rng_state(p))
        np.testing.assert_array_equal(own.policy(p,state),ordinary.policy(p,state))
        self.assertFalse(own.last_switched)
        self.assertEqual(own.switches,1)
        np.testing.assert_array_equal(action[p.action_idxs[2]],0.)
        original=own.get_rng_state(p)
        bad=copy.deepcopy(original);bad['rng']['problem']='invalid'
        with self.assertRaises(ValueError):own.set_rng_state(p,bad)
        self.assertEqual(original,own.get_rng_state(p))
        own.set_rng_state(p,original)
        self.assertEqual(original,own.get_rng_state(p))

    def test_signed_zero_dedup_Q_ties_fallback_and_nonfinite(self):
        p,state,own=self.own()
        state[[p.active_idxs[0],p.active_idxs[2]],0]=0.
        own.policy(p,state)
        table=np.asarray(own.solver.last_result.root_audit).copy()
        one=table[:1].copy();one[0,0]=0.;one[0,5]=1.
        duplicate=np.repeat(one,3,axis=0);duplicate[:,0]=np.arange(3);duplicate[1:,5]=0.
        duplicate[1,p.action_idxs[0][0]+6]=-0.0
        action,event=select_aq(duplicate,p,state)
        self.assertTrue(event['fallback']);self.assertEqual(event['distinct_actions'],1)
        # An opportunity without a distinct action is still consumed.
        fresh=m.AQOnceSolver(own.solver,m.o.TEAMS[1],True)
        with mock.patch.object(own.solver,'policy',return_value=action),mock.patch.object(own.solver,'last_result',SimpleNamespace(root_audit=duplicate)):
            fresh.policy(p,state);fresh.policy(p,state)
        self.assertEqual(fresh.switches,1);self.assertTrue(fresh.events[0]['fallback'])
        duplicate[1,6+p.action_idxs[3][0]]+=.01
        duplicate[2,6+p.action_idxs[3][0]]+=.02
        _,event=select_aq(duplicate,p,state)
        self.assertEqual(event['aq_ordinal'],1)
        bad=table.copy();bad[0,4]+=1.
        with self.assertRaises(ValueError):select_aq(bad,p,state)
        bad=table.copy();bad[0,6]=np.nan
        with self.assertRaises(ValueError):select_aq(bad,p,state)

    def test_real_capture_next_decision_direct_double_terminal_and_remaining_time(self):
        p,state,own=self.own()
        for r,xy in enumerate(([-3.,0.],[3.,0.],[-3.1,0.],[0.,3.])):
            state[p.state_idxs[r],0]=xy
        stats=m.initial_stats()
        nxt=m.advance(p,state,np.zeros((p.action_dim,1)),stats)
        self.assertEqual(stats['capture_count'],1);self.assertFalse(own.consumed)
        own.policy(p,nxt);self.assertTrue(own.consumed)
        p,state,own=self.own()
        for r,xy in enumerate(([-3.,0.],[3.,0.],[-3.1,0.],[3.1,0.])):
            state[p.state_idxs[r],0]=xy
        stats=m.initial_stats();final=m.advance(p,state,np.zeros((p.action_dim,1)),stats)
        self.assertTrue(p.is_terminal(final));self.assertEqual(stats['capture_count'],2)
        with self.assertRaises(ValueError):own.policy(p,final)
        self.assertEqual(own.switches,0);self.assertEqual(own.solver.calls,0)
        p,state,own=self.own()
        state[p.time_idx,0]=79.
        other=m.private_solver(p,m.registry(self.args)['L'],'Search Only',0,m.settings(self.args,0),11)
        final,stats,_,_=m.simulate(p,state,own,other,1,stats=dict(m.initial_stats(),steps=79))
        self.assertEqual(stats['steps'],80);self.assertEqual(float(final[p.time_idx,0]),80.)
        p,state,own=self.own()
        state[p.state_idxs[3][0],0]=1000.
        with self.assertRaises(ValueError):own.policy(p,state)
        self.assertEqual(own.switches,0)

    def test_complete_evader_alias_and_cache(self):
        items=self.late_items(role=0)
        self.assertEqual(items[0]['key'],items[3]['key'])
        self.assertNotEqual(items[0]['key'],items[6]['key'])
        rows=m.run_evaluation(self.args,'evader_alias_test',items,None,quiet)
        self.assertEqual(m.science(rows[0]),m.science(rows[3]))
        self.assertEqual([r['uid'] for r in rows],['AQ']*3+['O']*3+['L']*3)
        self.assertTrue(all(r['switches']==0 for r in rows))

    def test_live_original_complete_trace_and_snapshot_audit(self):
        result=m.audit(self.args,self.banks)
        self.assertTrue(result['passed']);self.assertEqual(len(result['original_parity']),12)
        self.assertEqual(len(result['segmented_parity']),4)
        # A missing audit result after atomic snapshots permits exact retry.
        for filename in ('audit_result.json','audit_result.lock.json'):
            (self.args.output/filename).unlink()
        self.assertTrue(m.audit(self.args,self.banks)['passed'])

    def test_every_effect_constraint(self):
        for failure in ('minimum_gain','search_only_improves','other_opponents','no_obstacle_regression',
                        'own_illegal','no_obstacle_own_illegal'):
            rows=self.synthetic()
            for r in rows:
                if r['uid']!='AQ':continue
                idx=int(r['episode_hash'].replace(r['layout'],''))
                if failure=='minimum_gain':r['legal_double_capture']=int(idx<51)
                if failure=='search_only_improves' and r['layout']=='four' and r['opponent']=='Search Only':r['legal_double_capture']=int(idx<50)
                if failure=='other_opponents' and r['layout']=='four' and r['opponent']=='Policy Only':r['legal_double_capture']=int(idx<46)
                if failure=='no_obstacle_regression' and r['layout']=='none':r['legal_double_capture']=int(idx<46)
                if failure=='own_illegal' and r['layout']=='four':r['pursuer_illegal']=int(idx<3)
                if failure=='no_obstacle_own_illegal' and r['layout']=='none':r['pursuer_illegal']=int(idx<3)
            result=m.comparison(rows,'O')
            self.assertFalse(result['checks'][failure],failure);self.assertFalse(result['passed'])

    def test_downstream_requires_gates(self):
        for function in (m.confirm,m.guard):
            with self.assertRaises(FileNotFoundError):function(self.args,self.banks,None,quiet)
        with self.assertRaises(FileNotFoundError):m.timing(self.args,self.banks,quiet)

    def test_controlled_confirmation_regression_timing_adoption_export_replay_and_fallbacks(self):
        for failure in (None,'validation','confirmation','guard'):
            with tempfile.TemporaryDirectory(prefix='l45_aq_controlled_') as path:
                args=copy.copy(self.args);args.output=Path(path);args.smoke=False
                shutil.copytree(self.args.output/'models',args.output/'models')
                m.lock_json(args.output/'experiment.json',dict(controlled_test=True))
                if failure=='validation':
                    m.lock_json(args.output/'validation_result.json',dict(passed=False))
                else:
                    with mock.patch.object(m,'run_evaluation',return_value=self.synthetic()):
                        self.assertTrue(m.validate(args,self.banks,None,quiet)['passed'])
                    if failure=='confirmation':
                        m.lock_json(args.output/'confirmation_result.json',dict(passed=False))
                    else:
                        with mock.patch.object(m,'run_evaluation',return_value=self.synthetic()):
                            self.assertTrue(m.confirm(args,self.banks,None,quiet)['passed'])
                        if failure=='guard':
                            bad=self.synthetic(role=0)
                            for r in bad:
                                if r['uid']=='AQ':r['legal_survival']=0
                            with mock.patch.object(m,'run_evaluation',return_value=bad):
                                self.assertFalse(m.guard(args,self.banks,None,quiet)['passed'])
                        else:
                            with mock.patch.object(m,'run_evaluation',return_value=self.synthetic(role=0)):
                                self.assertTrue(m.guard(args,self.banks,None,quiet)['passed'])
                            rows=[dict(uid=uid,opponent=op,decision_latency_ms=[[1.,2.,3.],[3.,2.,5.]],
                                switch_decision_ms=[3.] if uid=='AQ' else [],episode_wall_ms=20.)
                                for uid in ('AQ','O','L') for op in m.o.OPPONENTS]
                            with mock.patch.object(m,'run_evaluation',return_value=rows),mock.patch.object(m,'evaluate_one',return_value={}):
                                self.assertEqual(m.timing(args,self.banks,quiet)['AQ']['switch_p95_ms'],3.)
                with mock.patch.object(m,'verify',return_value={}),mock.patch.object(m,'replay',return_value=dict(passed=True)),mock.patch.object(m,'report'):
                    result=m.finish(args,self.banks)
                self.assertEqual(result['recommended'],'AQ' if failure is None else 'O')
                self.assertEqual(m.read_recommended(args)['aq_once'],failure is None)
                # Exercise the actual loader and complete-game replay for adopted and retained models.
                if failure in (None,'validation'):
                    self.assertTrue(m.replay(args,self.banks)['passed'])

    def test_model_implementation_and_snapshot_tamper(self):
        path=Path(m.registry(self.args)['O']['models'])/m.o.MODEL_NAMES[2]
        original=path.read_bytes()
        path.write_bytes(original+b'corruption')
        try:
            with self.assertRaises(ValueError):m.verify(self.args)
        finally:path.write_bytes(original)
        with mock.patch.object(m,'fingerprints',return_value={}):
            with self.assertRaises(ValueError):m.verify(self.args)
        path=self.args.output/'controlled_snapshot.pt'
        m.a.commit_pt(path,dict(state=np.zeros((53,1)),remaining_time=7.))
        path.write_bytes(b'broken')
        with self.assertRaises(ValueError):m.a.read_pt(path)

if __name__=='__main__':unittest.main()
