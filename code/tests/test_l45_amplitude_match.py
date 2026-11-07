"""Statistical pairing and the exact scientific diff; no closed-loop M suffixes."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_l45_amplitude_match as m

class AmplitudeMatchTests(unittest.TestCase):
    def test_only_mean_scaling_and_dispatch_change(self):
        parent=m.CODE/'cpp/build_candidate_ablation/source'
        new=m.CODE/'cpp/build_amplitude_match/source'
        p=(parent/'solvers/root_candidate_ablation.hpp').read_text()
        n=(new/'solvers/root_candidate_ablation.hpp').read_text()
        self.assertEqual(n.replace('if(mode==4) action=mean;\n        else if(mode==1) {','if(mode==1) {'),p)
        p=(parent/'solvers/interception_puct.hpp').read_text()
        n=(new/'solvers/interception_puct.hpp').read_text()
        self.assertEqual(n.replace('if(candidate_mode==1||candidate_mode==4) {','if(candidate_mode==1) {'),p)
        self.assertEqual((parent/'problems/example8.hpp').read_bytes(),(new/'problems/example8.hpp').read_bytes())

    def rows(self):
        rows=[]
        for scene in range(99):
            for stream in (0,1):
                for u,limit in [('S',50),('M',60),('F',80)]:
                    success=int(scene<limit)
                    rows.append(dict(key=f'{u}-{scene}-{stream}',uid=u,scene_id=str(scene),stream=stream,search_seeds=[scene,stream],
                        legal_double_capture=success,completion_cost=40. if success else 80.,first_capture_time=20.,
                        capture_time=40. if success else float('nan'),pursuer_illegal=0,evader_illegal=0,
                        own_mcts_calls=20,own_cost=dict(step_calls=10000,policy_evaluations=100,value_evaluations=0)))
        return rows

    def test_scene_bootstrap_direction_and_cost(self):
        rows=self.rows();a,p=m.analyze(rows);b,q=m.analyze(list(reversed(rows)))
        self.assertEqual(a,b);self.assertEqual(p,q);self.assertEqual(a['physical_states'],99)
        self.assertAlmostEqual(a['comparisons']['M-F']['delta'],-20/99)
        self.assertAlmostEqual(a['comparisons']['M-S']['delta'],10/99)
        self.assertLess(a['comparisons']['M-F']['ci95'][1],0)
        self.assertGreater(a['comparisons']['M-S']['ci95'][0],0)
        self.assertEqual(a['comparisons']['M-F']['lost_state_streams'],40)
        self.assertEqual(a['comparisons']['M-F']['worsened_physical_states'],20)
        self.assertGreater(a['comparisons']['M-F']['mean_post_cost_delta'],0)

    def test_missing_duplicate_or_unpaired_stream_stops(self):
        rows=self.rows()
        with self.assertRaises(ValueError):m.analyze(rows[:-1])
        with self.assertRaises(ValueError):m.analyze(rows+[rows[0]])
        rows[0]['search_seeds']=[999,999]
        with self.assertRaises(ValueError):m.analyze(rows)

    def test_budget_cap_before_any_execution(self):
        args=m.parse_args([])
        with self.assertRaises(ValueError):m.run(args,[],lambda *a,**kw:None)

if __name__=='__main__':unittest.main()
