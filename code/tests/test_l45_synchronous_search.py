"""Fresh-scene gates, shared-role caching and export branches."""
import copy
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_l45_synchronous_search as m


class SynchronousTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg={n:dict(models=str(m.SOURCE/'models'/n),model_hash=m.o.model_fingerprint(m.SOURCE/'models'/n)) for n in ('O','L')}

    @staticmethod
    def synthetic_rows():
        return [dict(uid=uid,layout=layout,role=1,episode_hash=f'{layout}-{i}',opponent=op,
            search_seeds=[i,i+10],legal_double_capture=int(uid=='C'),legal_survival=0,
            pursuer_illegal=0,evader_illegal=0,one_capture_timeout=int(uid!='C'),completion_cost=20. if uid=='C' else 80.)
            for layout in ('four','none') for i in range(8) for op in m.o.OPPONENTS for uid in ('C','O','L')]

    def test_unique_no_PPO_pursuit_and_effective_evader_sharing(self):
        self.assertEqual(m.phase_pair('C',1),('L','L'))
        self.assertEqual(m.phase_pair('C',0),('O','O'))
        self.assertEqual(m.effective('C',0,self.reg),m.effective('O',0,self.reg))
        self.assertNotEqual(m.effective('C',1,self.reg),m.effective('L',1,self.reg))
        cfg=m.config(m.parse_args([]));self.assertEqual(cfg['bootstrap_seed'],51020261024)
        self.assertEqual((cfg['budget'],cfg['search_only_budget']),(1000,3000))
        self.assertEqual(cfg['optimizer_updates'],0)

    def test_gates_pass_and_reject_opponent_or_illegal_regression(self):
        rows=self.synthetic_rows();self.assertTrue(m.comparison(rows,'O')['passed'])
        changed=copy.deepcopy(rows)
        for row in changed:
            if row['uid']=='C' and row['opponent']=='Policy Only':row['legal_double_capture']=0
            if row['uid']=='O' and row['opponent']=='Policy Only':row['legal_double_capture']=1
        self.assertFalse(m.comparison(changed,'O')['passed'])
        changed=copy.deepcopy(rows)
        for row in changed:
            if row['uid']=='C' and row['layout']=='none':row['pursuer_illegal']=1
        self.assertFalse(m.comparison(changed,'O')['passed'])

    def test_physical_scene_statistics_and_missing_pairs_stop(self):
        rows=self.synthetic_rows();out=m.statistics(rows,draws=1000)
        self.assertEqual(out['bootstrap_seed'],51020261024)
        for comparison in out['comparisons']:
            self.assertEqual(comparison['n_scenes'],8);self.assertEqual(comparison['n_games'],24)
            self.assertEqual(comparison['ci95'],[1.,1.]);self.assertEqual(comparison['simultaneous_lower95'],1.)
        with self.assertRaises(ValueError):m.statistics(rows[1:],draws=10)
        changed=copy.deepcopy(rows);changed[0]['search_seeds']=[999,999]
        with self.assertRaises(ValueError):m.statistics(changed,draws=10)

    def test_confirmation_cannot_open_after_failed_validation(self):
        args=m.parse_args(['--output','/tmp/unused_sync_gate'])
        with mock.patch.object(m,'locked_read',return_value=dict(passed=False)):
            with self.assertRaisesRegex(ValueError,'Validation failed'):m.check_candidate(args)

    def test_worker_independent_scientific_results(self):
        spec=m.o.make_scenario(510202610,'none')
        args=m.parse_args(['--smoke']);requested=m.settings(args)
        task=dict(key='controlled',uid='C',registry=self.reg,spec=spec,policy_id='controlled',role=1,
            opponent='Search Only',timing_context='parallel',mode='search',own_settings=requested,
            opponent_settings=m.settings(args,0),effective=m.effective('C',1,self.reg))
        serial=m.evaluate_one(task)
        with mp.get_context('spawn').Pool(2) as pool:parallel=pool.map(m.evaluate_one,[task])[0]
        self.assertEqual(m.science(serial),m.science(parallel));self.assertEqual(serial['own_cost'],parallel['own_cost'])

    def test_cache_corruption_and_identity_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            args=m.parse_args(['--smoke','--output',directory]);spec=m.o.make_scenario(510202611,'none')
            task=dict(key='controlled',uid='O',registry=self.reg,spec=spec,policy_id='controlled',role=1,
                opponent='Search Only',timing_context='parallel',mode='search',own_settings=m.settings(args),
                opponent_settings=m.settings(args,0),effective=m.effective('O',1,self.reg))
            row=m.evaluate_one(task);row['content_sha256']=m.o.json_hash(row)
            path=args.output/'cache/controlled.json';m.o.write_json(path,row)
            m.lock_json(args.output/'cache_checksums/controlled.json',dict(sha256=m.o.sha(path)))
            item=dict(key='controlled',uid='O',task=task);self.assertEqual(m.read_cached(args,item)['uid'],'O')
            path.write_text(path.read_text()+' ')
            with self.assertRaises(ValueError):m.read_cached(args,item)

    def test_controlled_adoption_and_failure_export(self):
        for accepted in (False,True):
            with self.subTest(accepted=accepted),tempfile.TemporaryDirectory() as directory:
                args=m.parse_args(['--output',directory])
                for name,model in self.reg.items():
                    dest=args.output/'models'/name;dest.mkdir(parents=True)
                    for filename in m.o.MODEL_NAMES:m.shutil.copyfile(Path(model['models'])/filename,dest/filename)
                m.lock_json(args.output/'validation_result.json',dict(passed=accepted))
                if accepted:
                    for name in ('confirmation_result','guard_result'):m.lock_json(args.output/(name+'.json'),dict(passed=True))
                    m.lock_json(args.output/'timing_result.json',dict(C=dict(pursuer_mean_ms=1.)))
                with mock.patch.object(m,'verify'),mock.patch.object(m,'replay',return_value=dict(passed=True)),mock.patch.object(m,'report'):
                    result=m.finish(args,{})
                self.assertEqual(result['recommended'],'C' if accepted else 'O')
                rec=m.locked_read(args.output/'recommended.json')
                self.assertEqual(rec['pursuer_model'],'L' if accepted else 'O')
                self.assertEqual(rec['evader_model'],'O')
                p,state=m.o.restore_scenario(m.o.make_scenario(510202612,'none'))
                # Exported role settings are fixed, independently of CLI flags.
                args.smoke=True
                a=m.load_recommended_solver(args,p,1,17)
                b=m.make_own(p,m.registry(args),rec['uid'],1,rec['pursuer_settings'],17)
                np.testing.assert_array_equal(a.policy(p,state),b.policy(p,state))


if __name__=='__main__':unittest.main()
