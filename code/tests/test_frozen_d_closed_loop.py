"""Meaningful closed-loop boundary, timing contract and grouped inference tests."""
from pathlib import Path
import sys,unittest,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_frozen_d_closed_loop import legal_action,statistics,original_model,OUT,ROOT,BUDGET
from solvers.frozen_d_search_interface import Controller,loader


class ClosedLoopTests(unittest.TestCase):
    def test_independent_motion_checks_reject_corner_and_nonfinite(self):
        boxes=np.array([[[-1.,1.],[-1.,1.]]])
        self.assertFalse(legal_action([-2.,0.],[2.,2.],2.,boxes))
        self.assertFalse(legal_action([-2.,2.],[np.nan,0.],2.,boxes))
        self.assertTrue(legal_action([-2.,2.],[.5,0.],1.,boxes))

    def test_original_environment_applies_simultaneous_actions_once(self):
        p=original_model([],1.3)
        state=np.zeros((p.state_dim,1));state[:2,0]=[-4.,0.];state[4:6,0]=[4.,0.]
        state[2:4,0]=[-9.8,-9.8];state[6:8,0]=[9.8,9.8];state[p.active_idxs,0]=[1,0,1,0]
        action=np.zeros((8,1));action[:2,0]=[.3,.4];action[4:6,0]=[-.7,.8]
        nxt=p.step(state,action,1.)
        np.testing.assert_allclose(nxt[:2,0],[-3.7,.4])
        np.testing.assert_allclose(nxt[4:6,0],[3.3,.8])
        self.assertEqual(nxt[8,0],1.)
        self.assertTrue(nxt[9,0] and nxt[11,0])

    def test_simultaneous_capture_intervals_and_cost_guard(self):
        rows=[]
        for h in range(12):
            for world in ('A','B'):
                for method in ('G','T','T+Geo'):
                    capture=int(method!='G')
                    rows.append(dict(history_id=f'group{h}',world=world,method=method,family=('long_wall','wall_corner','multi_route')[h%3],
                        capture=capture,rediscovered=capture,completion_cost=3 if capture else 30,
                        executed_steps=3 if capture else 30,first_rediscovery=1 if capture else None,
                        reason='capture' if capture else 'timeout',latencies_ms=[2.],cycle_ms=[10.],opponent_ms=[8.],
                        cold_start_first_decision_ms=100.,over_200ms=0,over_high_level_period=0,shield_count=0,
                        stationary_last8=not capture,hidden_covered=1,hidden_steps=2,simulated_transitions=8,failure=None))
        s=statistics(rows)
        self.assertEqual(s['groups'],12)
        self.assertTrue(s['comparisons']['T-G']['reliable_extra_capture_value'])
        self.assertFalse(s['comparisons']['T+Geo-T']['reliable_extra_capture_value'])
        self.assertEqual(s['comparisons']['T+Geo-T']['capture']['simultaneous_ci95'],[0.,0.])

    def test_budget_override_does_not_rewrite_old_failure_or_estimator(self):
        before=loader.verify()
        c=Controller([],1.1,'G',17,budget_ms=BUDGET)
        self.assertEqual(c.budget_ms,200.)
        self.assertEqual(before,loader.verify())
        old=json.loads((ROOT/'current/l45_weighted_search_interface_v2_20261026/results.json').read_text())
        self.assertFalse(old['passed'])
        self.assertTrue(all(not v for v in old['budget_checks'].values()))


if __name__=='__main__':unittest.main()
