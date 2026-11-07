"""Scientific pairing, bounded stage gates, record/replay and frozen composition."""
import copy
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_l45_search_factorial as m


class FactorialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = {n:dict(models=str(m.SOURCE/'models'/n),
            model_hash=m.o.model_fingerprint(m.SOURCE/'models'/n)) for n in ('O','L')}

    @staticmethod
    def rows(n=10):
        return [dict(uid=uid,layout=layout,role=1,episode_hash=f'{layout}-{i}',opponent=op,
            search_seeds=[i,i+10],legal_double_capture=int(uid=='D'),legal_survival=0,
            pursuer_illegal=0,evader_illegal=0,capture_count=2 if uid=='D' else 1,
            first_capture_time=20.,capture_time=30. if uid=='D' else float('nan'),
            one_capture_timeout=int(uid!='D'),completion_cost=30. if uid=='D' else 80.)
            for layout in ('four','none') for i in range(n) for op in m.o.OPPONENTS for uid in m.UIDS]

    def task(self,uid='D',role=1,opponent='Search Only',late=True):
        spec = m.o.make_scenario(520202611,'none')
        if late:
            p,state = m.o.restore_scenario(spec)
            state[p.time_idx,0] = 77.
            state[[p.active_idxs[0],p.active_idxs[2]],0] = 0.
            spec['initial_state'] = state.tolist()
            spec['episode_hash'] = m.o.json_hash({k:v for k,v in spec.items() if k!='episode_hash'})
        args = m.parse_args(['--smoke'])
        return dict(key='controlled-'+uid,uid=uid,registry=self.reg,spec=spec,policy_id='controlled-'+uid,
            role=role,opponent=opponent,timing_context='parallel',mode='search',
            own_settings=m.settings(args,role),opponent_settings=m.settings(args,0),
            effective=m.effective(uid,role,self.reg))

    def test_fixed_factor_mapping_and_evader_aliases(self):
        self.assertEqual([m.phase_pair(u,1)[0] for u in m.UIDS],['L','O','L','O'])
        self.assertEqual([m.effective(u,1,self.reg)['algorithm'] for u in m.UIDS],
                         ['legacy_PUCT','legacy_PUCT','synchronous_DUCT','synchronous_DUCT'])
        for u in ('B','C','D'):
            self.assertEqual(m.effective(u,0,self.reg),m.effective('B',0,self.reg))
        cfg = m.config(m.parse_args([]))
        self.assertEqual(cfg['candidate'],'D')
        self.assertEqual(cfg['request_caps'],dict(validation=1800,confirmation=6000,evader_regression=2400,serial_timing=360))
        self.assertEqual((cfg['budget'],cfg['search_only_budget'],cfg['bootstrap_seed']),(1000,3000,52020261024))
        self.assertEqual((cfg['training_episodes'],cfg['teacher_queries'],cfg['optimizer_updates']),(0,0,0))

    def test_all_three_references_and_regression_gates(self):
        for ref in m.REFS:
            self.assertTrue(m.comparison(self.rows(),ref)['passed'])
            for kind in ('gain','Search Only','Policy Only','Search + Learning','four_illegal','none_illegal','none_regression'):
                changed = self.rows()
                for row in changed:
                    if row['uid']==ref and (row['opponent']==kind or kind=='none_regression' and row['layout']=='none'):
                        row['legal_double_capture'] = 1
                    if row['uid'] != 'D':
                        continue
                    if kind=='gain' or row['opponent']==kind or (kind=='none_regression' and row['layout']=='none'):
                        row['legal_double_capture'] = 0
                    if kind=='four_illegal' and row['layout']=='four' or kind=='none_illegal' and row['layout']=='none':
                        row['pursuer_illegal'] = 1
                self.assertFalse(m.comparison(changed,ref)['passed'],(ref,kind))
            conf = dict(simultaneous_lower95=0.)
            self.assertFalse(m.comparison(self.rows(),ref,conf)['passed'])

    def test_exact_three_point_boundary(self):
        rows = self.rows(100)
        for row in rows:
            row['legal_double_capture'] = int(row['uid']=='D' and int(row['episode_hash'].split('-')[1])<3)
        for ref in m.REFS:
            self.assertTrue(m.comparison(rows,ref)['passed'])
        for row in rows:
            if row['uid']=='D' and row['episode_hash']=='four-2':
                row['legal_double_capture'] = 0
        self.assertFalse(m.comparison(rows,'B')['passed'])

    def test_three_comparisons_scene_bootstrap_preserves_opponents(self):
        rows = self.rows()
        for row in rows:
            row['legal_double_capture'] = int(row['uid']=='D' and int(row['episode_hash'].split('-')[1])<5)
        result = m.statistics(rows)
        self.assertEqual(result,m.statistics(rows))
        self.assertEqual([x['id'] for x in result['comparisons']],['D-A','D-B','D-C'])
        for row in result['comparisons']:
            self.assertEqual((row['n_scenes'],row['n_games'],row['delta']),(10,30,.5))
            np.testing.assert_allclose(row['ci95'],[.2,.8])
            self.assertLess(row['simultaneous_lower95'],.5)
        with self.assertRaises(ValueError):
            m.statistics(rows[1:],draws=10)
        with self.assertRaises(ValueError):
            m.statistics(rows+[rows[0]],draws=10)
        changed = copy.deepcopy(rows)
        changed[0]['search_seeds'] = [999,999]
        with self.assertRaises(ValueError):
            m.statistics(changed,draws=10)

    def test_interaction_is_not_automatically_positive(self):
        rows = self.rows()
        for row in rows:
            row['legal_double_capture'] = int(row['uid']!='A')
        result = m.factorial_statistics(rows,draws=100)
        interaction = next(x for x in result['comparisons'] if x['id']=='interaction')
        self.assertEqual(interaction['delta'],-1.)
        self.assertEqual(interaction['ci95'],[-1.,-1.])
        self.assertTrue(result['exploratory_only'])

    def test_failed_validation_and_smoke_cannot_open_confirmation(self):
        args = m.parse_args([])
        with mock.patch.object(m,'locked_read',return_value=dict(passed=False)):
            with self.assertRaisesRegex(ValueError,'confirmation forbidden'):
                m.check_candidate(args)
        args.smoke = True
        with mock.patch.object(m,'locked_read',return_value=dict(passed=True)):
            with self.assertRaises(ValueError):
                m.check_candidate(args)

    def test_worker_count_does_not_change_science_or_trace(self):
        tasks = [self.task(u) for u in m.UIDS]
        serial = [m.evaluate_one(t) for t in tasks]
        with mp.get_context('spawn').Pool(2) as pool:
            parallel = pool.map(m.evaluate_one,tasks)
        for a,b in zip(serial,parallel):
            self.assertEqual(m.science(a),m.science(b))
            self.assertEqual(a['trajectory_sha256'],b['trajectory_sha256'])
            self.assertEqual(a['own_cost'],b['own_cost'])

    def test_snapshot_roundtrip_inactive_mask_and_remaining_horizon(self):
        task = self.task()
        row = m.evaluate_one(task)
        detail = row['_record']
        p,state,own,other,stats = m.restore_snapshot(task,detail['initial'])
        self.assertEqual(float(state[p.time_idx,0]),77.)
        self.assertEqual(detail['initial']['remaining_time'],3.)
        before = m.snapshot(p,state,stats,own,other)
        own.solver.set_rng_state(p,own.solver.get_rng_state(p))
        self.assertTrue(m.s.equal_tree(before,m.snapshot(p,state,stats,own,other)))
        action = own.policy(p,state)
        self.assertTrue(np.all(action[p.action_idxs[2]]==0.))
        p,state,own,other,stats = m.restore_snapshot(task,detail['initial'])
        resumed,detail2,_,_ = m.simulate_record(p,state,own,other,1,stats)
        self.assertEqual(m.science(row),m.science(resumed))
        self.assertEqual(row['trajectory_sha256'],m.trajectory_digest(detail2['trajectory']))
        self.assertLessEqual(resumed['steps'],3)
        self.assertTrue(m.s.equal_tree(detail['final']['own_rng'],detail2['final']['own_rng']))
        self.assertTrue(m.s.equal_tree(detail['final']['opponent_rng'],detail2['final']['opponent_rng']))

    def test_actual_environment_direct_double_capture_and_zero_timeout(self):
        for double in (True,False):
            task = self.task(late=False)
            p,state,own,other = m.make_pair(task)
            for robot,pos in enumerate(([-2.,0.],[2.,0.],[-1.8,0.] if double else [-8.,0.],
                                       [2.2,0.] if double else [8.,0.])):
                state[p.state_idxs[robot],0] = pos
            state[p.time_idx,0] = 79.
            own.solver.last_result = SimpleNamespace(root_audit=np.zeros((1,p.action_dim+6)))
            with mock.patch.object(own,'policy',return_value=np.zeros((p.action_dim,1))), \
                 mock.patch.object(other,'policy',return_value=np.zeros((p.action_dim,1))):
                row,detail,_,_ = m.simulate_record(p,state,own,other,1)
            self.assertEqual(row['capture_count'],2 if double else 0)
            if double:
                self.assertFalse(detail['first_capture']['branch_available'])
                self.assertEqual(detail['first_capture']['next_decision'],1)
            else:
                self.assertIsNone(detail['first_capture'])
                self.assertEqual(row['completion_cost'],80.)

    def test_full_frozen_composition_and_snapshot_parity(self):
        for uid in m.UIDS:
            result = m.audit_one(self.task(uid,late=False))
            self.assertTrue(result['full_episode_exact'])

    def test_atomic_commit_corruption_and_uncommitted_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            args = m.parse_args(['--smoke','--output',directory])
            task = self.task()
            row = m.evaluate_one(task)
            # Orphaned artifacts without a row commit may safely be replaced.
            artifact = args.output/'episodes'/(task['key']+'.pt')
            m.o.save_torch(artifact,dict(orphaned=True))
            m.commit_row(args,row)
            item = dict(key=task['key'],uid=task['uid'],task=task)
            self.assertEqual(m.read_episode(args,item)[0]['uid'],'D')
            data = artifact.read_bytes()
            artifact.write_bytes(data+b'corrupt')
            with self.assertRaisesRegex(ValueError,'trajectory or snapshot'):
                m.read_cached(args,item)
            artifact.write_bytes(data)
            path = args.output/'cache'/(task['key']+'.json')
            path.write_text(path.read_text()+' ')
            with self.assertRaises(ValueError):
                m.read_cached(args,item)

    def test_interrupted_stage_reuses_committed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            args = m.parse_args(['--smoke','--output',directory])
            tasks = [self.task(uid,opponent=op) for uid in m.UIDS for op in m.o.OPPONENTS]
            for i,t in enumerate(tasks):
                t['key'] += '-'+str(i)
            items = [dict(key=t['key'],uid=t['uid'],task=t) for t in tasks]
            class InterruptedPool:
                _pool = []
                def imap_unordered(self,fn,tasks,chunksize):
                    yield fn(tasks[0])
                    raise InterruptedError('controlled stage interruption')
            with self.assertRaises(InterruptedError):
                m.run_evaluation(args,'validation',items,InterruptedPool(),lambda *a,**k:None)
            self.assertEqual(len(list((args.output/'cache_checksums').glob('*.lock.json'))),1)
            with mock.patch.object(m,'evaluate_one',wraps=m.evaluate_one) as evaluate:
                rows = m.run_evaluation(args,'validation',items,None,lambda *a,**k:None)
                self.assertEqual(evaluate.call_count,11)
            self.assertEqual(len(rows),12)
            with mock.patch.object(m,'evaluate_one',side_effect=AssertionError('must use cache')):
                m.run_evaluation(args,'validation',items,None,lambda *a,**k:None)

    def test_guard_alias_requests_and_failure_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            args = m.parse_args(['--smoke','--output',directory])
            m.o.write_json(args.output/'experiment.json',dict(controlled=True))
            spec = self.task()['spec']
            with mock.patch.object(m,'registry',return_value=self.reg):
                items = m.requests(args,m.UIDS,[spec],role=0)
            self.assertEqual(len(items),12)
            self.assertEqual(len({x['key'] for x in items}),6)
            with mock.patch.object(m,'check_candidate'),mock.patch.object(m,'locked_read',return_value=dict(passed=False)):
                with self.assertRaisesRegex(ValueError,'Confirmation failed'):
                    m.guard(args,{},None,lambda *a,**k:None)
            with mock.patch.object(m,'locked_read',return_value=dict(passed=False)):
                with self.assertRaisesRegex(ValueError,'Regression failed'):
                    m.timing(args,{},lambda *a,**k:None)

    def test_controlled_confirmation_and_guard_decisions(self):
        for passed in (True,False):
            with tempfile.TemporaryDirectory() as directory:
                args = m.parse_args(['--output',directory])
                m.o.write_json(args.output/'experiment.json',dict(controlled=True))
                banks = {key:[m.o.make_scenario(520202613+i,layout)] for i,(key,layout) in
                    enumerate((('confirm_four','four'),('confirm_none','none'),
                               ('guard_four','four'),('guard_none','none')))}
                rows = self.rows()
                if not passed:
                    for row in rows:
                        row['legal_double_capture'] = 0
                with mock.patch.object(m,'check_candidate'),mock.patch.object(m,'registry',return_value=self.reg), \
                     mock.patch.object(m,'run_evaluation',return_value=rows):
                    result = m.confirm(args,banks,None,lambda *a,**k:None)
                self.assertEqual(result['passed'],passed)
                self.assertEqual(len(result['statistics']['comparisons']),3)
                if not passed:
                    continue
                for row in rows:
                    row['role'] = 0
                    row['legal_survival'] = int(row['uid']!='A')
                with mock.patch.object(m,'check_candidate'),mock.patch.object(m,'registry',return_value=self.reg), \
                     mock.patch.object(m,'run_evaluation',return_value=rows):
                    result = m.guard(args,banks,None,lambda *a,**k:None)
                self.assertTrue(result['passed'])
                self.assertEqual((result['logical_requests'],result['unique_requests']),(24,12))
                with mock.patch.object(m.mp,'active_children',return_value=[object()]):
                    with self.assertRaisesRegex(ValueError,'Exit parallel pool'):
                        m.timing(args,banks,lambda *a,**k:None)

    def test_report_complete_four_groups_and_smoke_has_no_effect_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            args = m.parse_args(['--output',directory])
            rows = self.rows()
            items = [dict(row=row) for row in rows]
            result = dict(reason='validation_no_gain',recommended='B',quality_gate_passed=False,
                validation=dict(comparisons={ref:m.comparison(rows,ref) for ref in m.REFS}),
                confirmation=None,guard=None,timing=None)
            with mock.patch.object(m,'requests',return_value=items), \
                 mock.patch.object(m,'read_cached',side_effect=lambda args,item:item['row']), \
                 mock.patch.object(m,'diagnostic_summary',return_value={}):
                m.report(args,dict(validation_four=[],validation_none=[]),result)
            self.assertIn('|D|100.00%|100.00%|100.00%|',(args.output/'REPORT.md').read_text())
            self.assertIn('old0:novalue',(args.output/'CONCLUSION.md').read_text())
            args.smoke = True
            with mock.patch.object(m,'requests',return_value=items), \
                 mock.patch.object(m,'read_cached',side_effect=lambda args,item:item['row']), \
                 mock.patch.object(m,'read_episode'), \
                 mock.patch.object(m,'factorial_statistics',side_effect=AssertionError('no smoke inference')):
                m.report(args,dict(validation_four=[],validation_none=[]),result)
            self.assertIn('无效果结论',(args.output/'REPORT.md').read_text())

    def test_prepare_rejects_fingerprint_model_and_physical_scene_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            args = m.parse_args(['--smoke','--output',directory])
            banks = m.prepare(args)
            with mock.patch.object(m,'fingerprints',return_value={}):
                with self.assertRaisesRegex(ValueError,'Protocol, implementation'):
                    m.verify(args)
            with mock.patch.object(m,'registry',return_value={}):
                with self.assertRaisesRegex(ValueError,'Frozen model'):
                    m.verify(args)
            scene_hash = m.s.previous.physical_hash(banks['validation_four'][0])
            with mock.patch.object(m.s,'history_scenes',return_value={scene_hash}):
                with self.assertRaisesRegex(ValueError,'Scene leakage'):
                    m.verify(args)
            changed = copy.deepcopy(banks)
            changed['audit_four'] = changed['validation_four']
            original = m.locked_read
            with mock.patch.object(m,'locked_read',side_effect=lambda path:
                    changed if path.name=='scenarios.json' else original(path)):
                with self.assertRaisesRegex(ValueError,'Duplicate or missing'):
                    m.verify(args)
            args.smoke = False
            with self.assertRaisesRegex(ValueError,'Protocol, implementation'):
                m.verify(args)

    def test_controlled_adoption_and_each_failure_export_use_copied_runtime(self):
        for failure in ('validation','confirmation','guard',None):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                args = m.parse_args(['--output',directory])
                m.o.write_json(args.output/'experiment.json',dict(controlled=True))
                for name,model in self.reg.items():
                    dest = args.output/'models'/name
                    dest.mkdir(parents=True)
                    for filename in m.o.MODEL_NAMES:
                        m.shutil.copyfile(Path(model['models'])/filename,dest/filename)
                stages = ('validation','confirmation','guard')
                for stage in stages:
                    m.lock_json(args.output/(stage+'_result.json'),dict(passed=stage!=failure))
                    if stage==failure:
                        break
                if failure is None:
                    m.lock_json(args.output/'timing_result.json',dict(D=dict(pursuer_mean_ms=1.)))
                low = m.settings(m.parse_args(['--smoke']))
                banks = dict(warmup=[self.task()['spec']])
                with mock.patch.object(m,'verify'),mock.patch.object(m,'report'), \
                     mock.patch.object(m,'settings',side_effect=lambda args,role=1:dict(low,beta_value=0. if role else .1)):
                    result = m.finish(args,banks)
                self.assertEqual(result['recommended'],'D' if failure is None else 'B')
                self.assertTrue(result['replay']['exported_binary_loaded'])
                self.assertEqual(result['reason'], 'adopted' if failure is None else
                                 'validation_no_gain' if failure=='validation' else failure+'_failed')
                rec = m.locked_read(args.output/'recommended.json')
                self.assertEqual(rec['pursuer_model'],'O')
                self.assertEqual(rec['evader_search'],'legacy_PUCT')
                self.assertEqual(rec['pursuer_search'],'synchronous_DUCT' if failure is None else 'legacy_PUCT')


if __name__=='__main__':
    unittest.main()
