"""Semantic, RNG, resume and leakage checks for the bounded research experiment."""
import copy
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_l45_deployment_consistency as m
from learning import deployment_scorer as s


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg={name:dict(models=str(m.SOURCE/'models'/name),
            model_hash=m.o.model_fingerprint(m.SOURCE/'models'/name)) for name in ('O','L')}
        cls.cfg=m.config(m.parse_args(['--smoke']))
        cls.root_tasks=m.read_locked(m.SOURCE/'extract/manifest.json')

    def test_protocol_budget_and_frozen_actors(self):
        cfg=m.config(m.parse_args([]))
        self.assertEqual((cfg['budget'],cfg['opponent_budget'],cfg['continuation_requests']),(1000,3000,396))
        self.assertEqual((cfg['actor_updates'],cfg['ppo_updates'],cfg['new_teacher_queries']),(0,0,0))

    def test_joint_transition_capture_counterexample(self):
        p,state,joint,pursuit=m.fixture()
        self.assertTrue(p.is_valid(state))
        solver=m.make_solver(p,self.reg['O'],self.cfg,True,1)
        wrapper=solver._problem_wrapper_for(p)
        simultaneous=np.asarray(wrapper.step(state,joint,p.dt)).reshape(-1,1)
        alternating=np.asarray(wrapper.step(state,pursuit,p.dt)).reshape(-1,1)
        self.assertEqual(p.active_evader_count(simultaneous),1)
        self.assertEqual(p.active_evader_count(alternating),0)
        np.testing.assert_allclose(simultaneous,p.step(state,joint,p.dt),atol=5e-5)
        self.assertEqual(float(simultaneous[p.time_idx,0]),21.)

    def test_legacy_instrumentation_exact_and_nonconsuming(self):
        root=m.r.read_task(self.root_tasks[0]);p,_=m.o.restore_scenario(root['task']['row']['spec'])
        a=m.r.make_solver(p,self.reg['O'],'Search + Learning',1,self.cfg,True,27)
        b=m.make_solver(p,self.reg['O'],self.cfg,False,1,27)
        self.assertEqual(a.get_rng_state(p),b.get_rng_state(p))
        aa=a.policy(p,root['state'],team=(2,3));bb=b.policy(p,root['state'],team=(2,3))
        np.testing.assert_array_equal(aa,bb)
        self.assertEqual(a.get_rng_state(p),b.get_rng_state(p))
        self.assertGreater(b.last_result.step_calls,0)

    def test_sync_root_role_does_not_change_tree_or_random_stream(self):
        p,state,_,_=m.fixture();solver=m.make_solver(p,self.reg['O'],self.cfg,True,1,17)
        before=solver.get_rng_state(p)
        a=solver.search(p,state,0);after=solver.get_rng_state(p)
        solver.set_rng_state(p,before);b=solver.search(p,state,2)
        self.assertEqual(after,solver.get_rng_state(p))
        np.testing.assert_array_equal(a.value,b.value)
        self.assertEqual(a.step_calls,b.step_calls)
        np.testing.assert_array_equal(a.best_action[4:],0.)
        np.testing.assert_array_equal(b.best_action[:4],0.)

    def test_remaining_horizon_and_pure_network_call_count(self):
        p,state,_,_=m.fixture();state[p.time_idx,0]=79.
        solver=m.make_solver(p,self.reg['O'],self.cfg,True,1,1,True)
        result=solver.search(p,state,2)
        self.assertLessEqual(result.step_calls,self.cfg['opponent_budget'])
        self.assertEqual(result.policy_evaluations,0)
        self.assertEqual(result.value_evaluations,0)
        self.assertEqual(int(np.asarray(result.root_audit)[:,1].sum()),self.cfg['opponent_budget'])

    def test_rng_invalid_state_transactional(self):
        p,state,_,_=m.fixture();solver=m.make_solver(p,self.reg['O'],self.cfg,True,1)
        before=solver.get_rng_state(p)
        for invalid in ('','invalid','42 trailing'):
            with self.assertRaises(ValueError):solver.solver_wrapper.set_rng_state(invalid)
            self.assertEqual(before,solver.get_rng_state(p))

    def test_atomic_cache_tamper_and_idempotence(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cache.pt';value=dict(state=np.array([1.,2.]),metrics=dict(success=1))
            m.r.commit_pt(path,value);m.r.commit_pt(path,value)
            with self.assertRaises(ValueError):m.r.commit_pt(path,dict(other=True))
            path.write_bytes(path.read_bytes()+b'tampered')
            with self.assertRaises(ValueError):m.r.read_pt(path)

    def test_continuation_worker_count_and_snapshot_replay(self):
        original=self.root_tasks[0];root=m.r.read_task(original)
        root=copy.deepcopy(root);root['state'][8,0]=79.;root['stats']['steps']=79
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'root.pt';m.r.commit_pt(path,root)
            task=dict(kind='continuation',variant='L_sync',root=dict(task=dict(result=str(path)),sha256=m.o.sha(path)),
                models=self.reg,config=self.cfg,key='controlled')
            serial=m.continuation_one(task)
            with mp.get_context('spawn').Pool(2) as pool:parallel=pool.map(m.continuation_one,[task])[0]
            self.assertEqual(m.h.science(serial['metrics']),m.h.science(parallel['metrics']))
            for key in ('state','trajectory','own_rng','opponent_rng','own_cost','other_cost'):
                self.assertTrue(m.s.equal_tree(serial[key],parallel[key]),key)
            self.assertEqual(serial['metrics']['steps'],80)

    @staticmethod
    def synthetic_dataset(count=50):
        result=[]
        for i in range(count):
            sign=1. if i%2 else -1.
            result.append(dict(physical_hash=f'physical-{i:03}',compact=[sign,i/count],
                observation=[sign,i/count,0.],paired_delta=[sign]*3,illegal_delta=[0.]*3))
        return result

    def test_nested_selection_learns_only_available_signal(self):
        result=s.evaluate(self.synthetic_dataset())
        self.assertGreater(result['delta_vs_A0'],.3)
        self.assertGreater(result['delta_vs_AQ'],.3)
        self.assertTrue(result['passed'])

    def test_outer_test_labels_cannot_affect_own_predictions(self):
        data=self.synthetic_dataset();before=s.evaluate(data)
        ids=[x['physical_hash'] for x in data];fold=s.folds(ids)
        changed=copy.deepcopy(data)
        for i in np.flatnonzero(fold==0):changed[i]['paired_delta']=[-v for v in changed[i]['paired_delta']]
        after=s.evaluate(changed)
        a=[x for x in before['predictions'] if x['fold']==0]
        b=[x for x in after['predictions'] if x['fold']==0]
        self.assertEqual([(x['prediction'],x['choose_AQ']) for x in a],[(x['prediction'],x['choose_AQ']) for x in b])

    def test_missing_pairs_duplicate_scenes_and_nonfinite_stop(self):
        data=self.synthetic_dataset();data[0]['paired_delta']=[0.,1.]
        with self.assertRaises((ValueError,TypeError)):s.evaluate(data)
        data=self.synthetic_dataset();data[1]['physical_hash']=data[0]['physical_hash']
        with self.assertRaises(ValueError):s.evaluate(data)
        with self.assertRaises(ValueError):s.fit('compact_ridge',np.array([[np.nan]]),[0.])

    def test_gate_rejects_no_gain_vs_constant_and_uncertain_gain(self):
        result=dict(delta_vs_A0=.1,ci95_vs_A0=[.01,.2],delta_vs_AQ=.05,ci95_vs_AQ=[.01,.1],
            outer_fold_delta=[.1]*5,illegal_delta=0.)
        self.assertTrue(s.gate(result))
        for key,value in [('ci95_vs_A0',[0.,.2]),('ci95_vs_AQ',[-.01,.1]),('illegal_delta',.021),('delta_vs_A0',.029)]:
            changed=dict(result);changed[key]=value;self.assertFalse(s.gate(changed))


if __name__=='__main__':unittest.main()
