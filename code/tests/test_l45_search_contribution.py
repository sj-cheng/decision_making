"""Isolation, real deadlines, fixed compatibility, geometry and physical pairing."""
from pathlib import Path
import sys
import unittest
import time
import copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import run_l45_search_contribution as m

class ContributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args=m.parse_args(['--smoke'])
        cls.reg={u:dict(models=str(m.SOURCE/'models'/u),model_hash=m.o.model_fingerprint(m.SOURCE/'models'/u)) for u in ('O','L')}

    def problem(self,map_id='none'):
        spec=m.prior.make_scenario(59261024,map_id);p,state=m.prior.restore_scenario(spec)
        state[[p.active_idxs[0],p.active_idxs[2]],0]=0.;state[p.time_idx,0]=30.
        state[p.state_idxs[1],0]=[-5.,0.];state[p.state_idxs[3],0]=[5.,0.]
        return p,state,spec

    def test_fixed_1000_matches_original_and_replay(self):
        for map_id in ('none','asymmetric'):
            p,state,spec=self.problem(map_id)
            for uid in ('S','F','G'):
                setting=m.settings(self.args,uid);setting['search_budget']=1000
                a=m.make_own(p,self.reg,uid,setting,123);b=m.prior.make_own(p,self.reg,uid,setting,123)
                before=a.solver.get_rng_state(p);aa=a.policy(p,state);bb=b.policy(p,state)
                np.testing.assert_array_equal(aa,bb)
                np.testing.assert_array_equal(a.solver.last_result.root_audit,b.solver.last_result.root_audit)
                self.assertEqual(a.solver.get_rng_state(p),b.solver.get_rng_state(p))
                self.assertEqual(a.solver.last_result.num_visits,1000)
                a.solver.set_rng_state(p,before);np.testing.assert_array_equal(aa,a.policy(p,state))

    def test_no_neural_only_pursuer_removed_and_sampling_draws_match(self):
        p,state,_=self.problem();setting=m.settings(self.args);setting['search_budget']=1000
        n=m.make_own(p,self.reg,'G-no-neural',setting,123);g=m.make_own(p,self.reg,'G',setting,123)
        n.policy(p,state);audit=n.solver.last_result.guidance_stats
        self.assertEqual(audit['policy_pursuer'],0);self.assertGreater(audit['policy_evader'],0)
        self.assertGreater(audit['discarded_neural_noise_pairs'],0)
        self.assertEqual(audit.get('root_source_1',0),0)
        for seed in range(15):
            for team in (0,1):
                n.solver.solver_wrapper.set_seed(seed);g.solver.solver_wrapper.set_seed(seed)
                wrapper=n.solver._problem_wrapper_for(p)
                m.cpp.contribution_sample_probe(wrapper,n.solver.solver_wrapper,state,team)
                m.cpp.contribution_sample_probe(g.solver._problem_wrapper_for(p),g.solver.solver_wrapper,state,team)
                self.assertEqual(n.solver.get_rng_state(p),g.solver.get_rng_state(p))

    def test_geometry_no_tree_network_rng_and_G_base(self):
        p,state,_=self.problem('asymmetric')
        geometry=m.make_own(p,self.reg,'Geometry-only',m.settings(self.args,'Geometry-only'),123)
        before=geometry.solver.get_rng_state(p);action=geometry.policy(p,state);r=geometry.solver.last_result
        self.assertEqual(r.num_visits,0);self.assertEqual(r.step_calls,0);self.assertEqual(r.policy_evaluations,0)
        self.assertEqual(before,geometry.solver.get_rng_state(p))
        data=m.cpp.candidate_debug(geometry.solver._problem_wrapper_for(p),state,3,[0.,0.])
        np.testing.assert_array_equal(action[6:8,0],np.asarray(data['action']).reshape(-1))
        np.testing.assert_array_equal(action[:6],0)

    def test_deadline_measures_real_work_and_stops_before_cap(self):
        p,state,_=self.problem();setting=m.settings(self.args,'G','timed');setting['deadline_ms']=5.
        solver=m.make_own(p,self.reg,'G',setting,123)
        solver.solver._problem_wrapper_for(p);started=time.perf_counter();solver.policy(p,state);elapsed=(time.perf_counter()-started)*1000
        r=solver.solver.last_result
        self.assertGreaterEqual(r.num_visits,1);self.assertLess(r.num_visits,100000)
        self.assertGreaterEqual(elapsed,4.5);self.assertLess(elapsed,100.)
        self.assertFalse(r.guidance_stats['simulation_cap_hit'])

    def rows(self):
        rows=[]
        for scene in range(9):
            layout=('original_four','unseen_four','none')[scene//3]
            for phase in ('fixed','timed'):
                for opponent in m.prior.OPPONENTS:
                    for uid in m.UIDS:
                        success=int(scene%3<(1 if uid=='S' else 2))
                        rows.append(dict(phase=phase,uid=uid,episode_hash=str(scene),layout=layout,opponent=opponent,
                            search_seeds=[scene,scene+1],legal_double_capture=success,completion_cost=40 if success else 80,
                            pursuer_illegal=0,evader_illegal=0,capture_count=2 if success else 1,one_capture_timeout=1-success,
                            own_cost=dict(step_calls=1000,policy_evaluations=100,value_evaluations=0),decision_latency_ms=[[10.,20.,30.]]))
        return rows

    def test_physical_opponent_phase_pairing(self):
        rows=self.rows();a=m.analyze(rows);b=m.analyze(list(reversed(rows)))
        self.assertEqual(a,b);self.assertEqual(a['physical_scenes'],9)
        self.assertAlmostEqual(a['sections']['all']['fixed']['mean']['comparisons']['G minus S']['delta'],1/3)
        self.assertEqual(a['sections']['all']['fixed']['mean']['comparisons']['G-no-neural minus G']['delta'],0)
        with self.assertRaises(ValueError):m.analyze(rows[:-1])
        with self.assertRaises(ValueError):m.analyze(rows+[rows[0]])
        rows[0]['search_seeds']=[999,999]
        with self.assertRaises(ValueError):m.analyze(rows)

if __name__=='__main__':unittest.main()
