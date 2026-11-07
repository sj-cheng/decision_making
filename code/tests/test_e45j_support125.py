import json
import unittest
from pathlib import Path
import numpy as np
from learning.e45j_support125 import TeamBelief125,PreviousBelief
from solvers.e45f_support_refinement import refined_posterior
from solvers.e45_team_adapter import team_hidden_mask

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'current/e45j_repaired_benchmark_20261031/belief_failures/public_support_2643505_1793430909283741180.json'


class FineRecoveryTests(unittest.TestCase):
    def test_real_failure_has_public_fine_support_and_exact_replay(self):
        d=json.loads(EVIDENCE.read_text());boxes=np.array(d['boxes']);h=d['history']
        self.assertIsNone(refined_posterior(boxes,h,0,.25))
        beliefs=[TeamBelief125(boxes,d['seed']) for _ in range(2)]
        for b in beliefs:
            for obs in h:b.update(obs)
            e=b.estimators[0]
            self.assertEqual(b.step,22);self.assertEqual(int(e.reachable.sum()),4)
            self.assertAlmostEqual(float(e.particle_weights.sum()),1.)
            self.assertTrue(team_hidden_mask(e.particles,np.array(h[-1]['positions']),np.array(h[-1]['active_p']),boxes).all())
            np.testing.assert_array_equal(b.known[1],h[-1]['reports'][1][0])
            # Last sighting was exactly one step ago: every revived point
            # needs an actual lawful graph edge from that measured anchor.
            graph=e.graph;support=np.flatnonzero(e.reachable)
            self.assertTrue(set(support).issubset(set(graph[1][graph[5],:graph[2][graph[5]]])))
        self.assertEqual(beliefs[0].signature(),beliefs[1].signature())

    def test_successful_public_histories_keep_original_draws_and_posteriors(self):
        entries=json.loads((ROOT/'current/e45j_repaired_benchmark_20261031/bank/manifest.json').read_text())['validation']
        for family in ('long_wall','wall_corner','multi_route'):
            item=next(e for e in entries if e['family']==family);c=json.loads(Path(item['path']).read_text())
            old=PreviousBelief(c['map'],17);new=TeamBelief125(c['map'],17)
            for obs in c['history']:
                old.update(obs);new.update(obs);self.assertEqual(old.signature(),new.signature())


if __name__=='__main__':unittest.main()
