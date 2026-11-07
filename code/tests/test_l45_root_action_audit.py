"""Root readout, counterfactual timing/RNG, selection and commit integrity."""
import copy
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_l45_root_action_audit as m

quiet=lambda *a,**kw:None


class RootTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='l45_root_tests_')
        cls.args=m.parse_args(['--smoke','--workers','2','--output',cls.temp.name])
        cls.bank=m.prepare(cls.args)
        cls.roots=m.tasks(cls.args,cls.bank)
        m.run_tasks(cls.args,'extract',cls.roots,m.extract_one,None,quiet)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_protocol_and_disjointness(self):
        cfg=m.config(m.parse_args([]))
        self.assertEqual((cfg['budget'],cfg['opponent_budget'],cfg['depth']),(1000,3000,100))
        self.assertEqual((cfg['training_episodes'],cfg['optimizer_updates'],cfg['model_exports']),(0,0,0))
        self.assertEqual(cfg['bootstrap_seed'],49020261024)
        self.assertEqual(cfg['formal_suffix_cap'],1188)
        self.assertTrue(m.verify(self.args))

    def test_real_original_private_and_export_rng_parity(self):
        out=m.audit(self.args,self.bank,None,quiet)
        self.assertEqual(len(out['original_pairs']),12)
        self.assertEqual(len(out['rng_action_probes']),8)

    def test_root_candidate_execution_Q_scale_order(self):
        root=m.read_task(self.roots[0]);p,_=m.o.restore_scenario(self.bank['rows'][0]['spec'])
        prefix=m.read_pt(self.bank['rows'][0]['prefix_path'])
        self.assertTrue(m.s.equal_tree(root['python_rng'],prefix['python_rng']))
        picks=m.selected_candidates(root['root_table'],p,root['state'])
        self.assertTrue(picks[0]['is_A0'])
        self.assertLessEqual(len(picks),4)
        for c in picks:
            for robot in (0,1,2):np.testing.assert_array_equal(c['action'][p.action_idxs[robot]],0.)
        bad=root['root_table'].copy();bad[0,4]+=1.
        with self.assertRaisesRegex(ValueError,'Q formula'):m.selected_candidates(bad,p,root['state'])
        bad=root['root_table'].copy();bad[0,6]=np.nan
        with self.assertRaises(ValueError):m.selected_candidates(bad,p,root['state'])

    def test_float32_dedup_signed_zero_and_single_action(self):
        root=m.read_task(self.roots[0]);p,_=m.o.restore_scenario(self.bank['rows'][0]['spec'])
        table=root['root_table'][:1].copy();table[0,0]=0;table[0,5]=1
        table=np.repeat(table,2,axis=0);table[1,0]=1;table[1,5]=0
        picks=m.selected_candidates(table,p,root['state'])
        self.assertEqual(len(picks),1)

    def test_rng_restore_and_new_stream_independent_of_candidate(self):
        root=m.read_task(self.roots[0]);p,_=m.o.restore_scenario(self.bank['rows'][0]['spec'])
        own,other=m.make_pair(p,m.registry(self.args),m.config(self.args))
        records=[]
        for _ in range(2):
            m.initialize_stream(p,own,other,root,1)
            before=(own.get_rng_state(p),other.get_rng_state(p))
            action=own.policy(p,root['state'],team=m.o.TEAMS[1]);records.append((before,action))
        self.assertEqual(records[0][0],records[1][0]);np.testing.assert_array_equal(records[0][1],records[1][1])
        wrapper=own.solver_wrapper;before=wrapper.get_rng_state()
        for invalid in ('','bad','42 trailing'):
            with self.assertRaises(ValueError):wrapper.set_rng_state(invalid)
            self.assertEqual(wrapper.get_rng_state(),before)

    def test_only_first_action_simultaneous_and_remaining_time(self):
        root=m.read_task(self.roots[0]);p,_=m.o.restore_scenario(self.bank['rows'][0]['spec'])
        state=root['state'].copy();state[p.time_idx,0]=p.tf-p.dt
        stats=copy.deepcopy(root['stats']);stats['steps']=79
        own,other=m.make_pair(p,m.registry(self.args),m.config(self.args))
        seen=[]
        original=other.policy
        def observe(problem,current,team=None):
            seen.append(current.copy());return original(problem,current,team=team)
        with mock.patch.object(other,'policy',side_effect=observe):
            final,trace,_=m.closed_loop(p,state,stats,own,other,root['candidates'][0]['action'])
        self.assertEqual(own.calls,0);self.assertEqual(other.calls,1)
        np.testing.assert_array_equal(seen[0],state)
        self.assertEqual(float(final[p.time_idx,0]),80.)
        self.assertEqual(len(trace),1)

    def test_direct_terminal_illegal_and_inactive(self):
        root=m.read_task(self.roots[0]);p,_=m.o.restore_scenario(self.bank['rows'][0]['spec'])
        state=root['state'].copy();state[p.state_idxs[3],0]=state[p.state_idxs[1],0]+[.01,0]
        stats=copy.deepcopy(root['stats'])
        final=m.h.advance(p,state,np.zeros((p.action_dim,1)),stats)
        self.assertTrue(p.is_terminal(final));self.assertEqual(stats['capture_count'],2)
        state=root['state'].copy();state[p.state_idxs[3][0],0]=1000.
        stats=copy.deepcopy(root['stats']);final=m.h.advance(p,state,np.zeros((p.action_dim,1)),stats)
        self.assertEqual(m.h.metrics(p,final,stats)['pursuer_illegal'],1)
        self.assertEqual(m.h.metrics(p,final,stats)['legal_double_capture'],0)

    def test_worker_invariance_and_suffix_commit_resume(self):
        items=m.branch_tasks(self.args,self.roots)[:3]
        serial=[m.branch_one(t) for t in items]
        with mp.get_context('spawn').Pool(2) as pool:parallel=pool.map(m.branch_one,items)
        for a,b in zip(serial,parallel):
            self.assertEqual(m.h.science(a['metrics']),m.h.science(b['metrics']))
            self.assertEqual(m.o.json_hash(a['trajectory']),m.o.json_hash(b['trajectory']))
            self.assertTrue(m.s.equal_tree(a['initial_rng'],b['initial_rng']))
        m.run_tasks(self.args,'test_resume',items,m.branch_one,None,quiet)
        with mock.patch.object(m,'branch_one',side_effect=AssertionError('cache required')):
            m.run_tasks(self.args,'test_resume',items,m.branch_one,None,quiet)

    def test_saved_formal_rng_historical_parity(self):
        original=next(t for t in m.h.diagnostic_tasks(m.h.parse_args([]))
            if t['prefix']=='O' and m.h.read_segment(t)['branch'])
        suffix=next(t for t in m.h.suffix_tasks(m.h.parse_args([]),[original]) if t['after']=='O')
        task=dict(kind='root',protocol='engineering-history-probe',row=dict(spec=original['spec'],
            prefix_path=original['result'],prefix_sha=m.o.sha(original['result']),historical_suffix=suffix['result']),
            registry=m.registry(self.args),config=m.config(m.parse_args([])))
        task['key']=m.o.json_hash(task);task['result']=str(self.args.output/'history_probe.pt')
        m.commit_pt(task['result'],m.extract_one(task))
        branch=dict(kind='suffix',protocol='engineering-history-probe',root_task=task,
            root_sha=m.o.sha(task['result']),candidate=0,stream=0,key='engineering-history-probe')
        self.assertTrue(m.branch_one(branch)['historical_exact'])

    def test_interrupt_root_and_suffix_only_replay_uncommitted(self):
        for kind,original,worker in [('root',self.roots[0],m.extract_one),
                ('suffix',m.branch_tasks(self.args,self.roots)[0],m.branch_one)]:
            task=copy.deepcopy(original);task['key']='interrupt-'+kind
            task['result']=str(self.args.output/('interrupt-'+kind+'.pt'))
            calls=[]
            def fail_before_commit(t):
                calls.append(t['key']);raise RuntimeError('interrupted before commit')
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                m.run_tasks(self.args,'interruption-'+kind,[task],fail_before_commit,None,quiet)
            m.run_tasks(self.args,'interruption-'+kind,[task],worker,None,quiet)
            m.run_tasks(self.args,'interruption-'+kind,[task],fail_before_commit,None,quiet)
            self.assertEqual(calls,[task['key']])

    def test_partial_commit_recovery_and_corruption(self):
        task=copy.deepcopy(self.roots[0]);task['key']='controlled-root';task['result']=str(self.args.output/'controlled.pt')
        value=m.extract_one(task);m.commit_pt(task['result'],value)
        path=Path(task['result']);projection=path.with_suffix('.json');projection.unlink()
        self.assertEqual(m.read_task(task)['key'],task['key'])
        original=path.read_bytes();path.write_bytes(original+b'corrupt')
        with self.assertRaisesRegex(ValueError,'checksum'):m.read_task(task)
        path.write_bytes(original)
        changed=copy.deepcopy(task);changed['kind']='different'
        with self.assertRaisesRegex(ValueError,'identity'):m.read_task(changed)

    @staticmethod
    def group():
        return {i:{r:dict(legal_double_capture=int(i==1),pursuer_illegal=0,completion_cost=60. if i==1 else 80.,
            ordinal=i,q=float(1-i),discounted_suffix_return=float(i)) for r in range(3)} for i in range(2)}

    def test_selection_does_not_read_heldout_and_prefers_A0_ties(self):
        g=self.group();self.assertEqual(m.choose_without_heldout(g,2),1)
        g[1][2]['legal_double_capture']=0;g[1][2]['pursuer_illegal']=1
        self.assertEqual(m.choose_without_heldout(g,2),1)
        for r in range(3):g[1][r]['legal_double_capture']=0;g[1][r]['pursuer_illegal']=0
        self.assertEqual(m.choose_without_heldout(g,2),0)
        g=self.group();g[1][0]['pursuer_illegal']=1
        self.assertEqual(m.choose_without_heldout(g,2),0)

    def test_missing_pair_stops_and_bootstrap_physical_clusters(self):
        g=self.group();del g[1][2]
        with self.assertRaisesRegex(ValueError,'pairing'):m.choose_without_heldout(g,0)
        stats=m.compute_statistics({str(i):self.group() for i in range(10)},11,draws=200)
        self.assertEqual(stats['delta'],1.);self.assertEqual(stats['ci95'],[1.,1.])
        self.assertEqual(stats['all_scene_delta'],10/11)
        self.assertTrue(stats['passed']);self.assertEqual(stats['per_stream_delta'],[1.,1.,1.])

    def test_every_gate_and_fallback(self):
        self.assertTrue(m.gate(.05,.001,[.1,.1,-.1],.02))
        for args in ((.049,.001,[1,1,1],0),(.1,0,[1,1,1],0),(.1,.01,[1,0,0],0),(.1,.01,[1,1,1],.021)):
            self.assertFalse(m.gate(*args))

    def test_watch_finished_process_counts_committed_files(self):
        with tempfile.TemporaryDirectory() as temp:
            args=m.parse_args(['watch','--once','--output',temp])
            m.o.write_json(args.output/'status.json',dict(phase='complete',pid=999999999))
            with mock.patch('builtins.print'):
                row=m.watch(args)
            self.assertFalse(row['pid_alive']);self.assertEqual(row['live_workers'],0)
            self.assertEqual(row['committed_roots'],0);self.assertEqual(row['committed_suffixes'],0)

    def test_fingerprint_and_manifest_tamper_stop(self):
        path=self.args.output/'bank.json';original=path.read_text()
        path.write_text('{}')
        try:
            with self.assertRaises(ValueError):m.verify(self.args)
        finally:path.write_text(original)
        with mock.patch.object(m,'fingerprints',return_value={}):
            with self.assertRaisesRegex(ValueError,'Protocol'):m.verify(self.args)

    def test_full_smoke_statistics_replay_and_report(self):
        suffixes=m.branch_tasks(self.args,self.roots)
        m.run_tasks(self.args,'branch',suffixes,m.branch_one,None,quiet)
        stats=m.analyze(self.args,self.bank,self.roots,suffixes)
        self.assertFalse(stats['passed']);self.assertEqual(stats['reason'],'smoke_only')
        m.report(self.args,self.bank,stats)
        self.assertEqual(m.replay(self.args,self.bank)['suffixes'],len(suffixes))
        with self.assertRaisesRegex(ValueError,'pairing'):m.analyze(self.args,self.bank,self.roots,suffixes[:-1])


if __name__=='__main__':unittest.main()
