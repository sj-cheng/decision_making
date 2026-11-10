"""Scientific protocol checks: map pairing, calibration and independent capture."""
import unittest
import numpy as np
from e49_audit import replay_motion, audit_frame
from run_e49_benchmark import saturation, method_rates, adaptation_decisions
from e49_report import ARMS, map_matrix, weights_for, statistics


class CoordinatorTests(unittest.TestCase):
    def test_saturation_requires_ten_methods_and_keeps_mixed_outcomes(self):
        rates={arm:1. for arm in ARMS}
        self.assertEqual(saturation(rates),'easy')
        for arm in ARMS[:3]:rates[arm]=0.
        self.assertEqual(saturation(rates),'usable')
        rates={arm:0. for arm in ARMS}
        self.assertEqual(saturation(rates),'hard')
        rates[ARMS[0]]=.5;rates[ARMS[1]]=.5
        self.assertEqual(saturation(rates),'hard')

    def test_model_seeds_do_not_overweight_method_or_map(self):
        rows=[]
        for tier in ('sparse','16','24'):
            for arm in ARMS:
                for seed in ((17,29,43) if arm=='A3' else (17,)):
                    rows.append(dict(task=dict(arm=arm,group=tier,obstacle_tier=tier),
                                     summary=dict(double_capture=seed==17)))
        rates=method_rates(rows)
        self.assertAlmostEqual(rates['A3'],1/3)
        self.assertEqual(rates['A2'],1)

    def test_suite_and_tier_weighting(self):
        names=['a','b','c','d']
        metadata={'a':('16','ID','wall'),'b':('16','OOD','gate'),
                  'c':('16','OOD','gate'),'d':('24','ID','wall')}
        w=weights_for(names,metadata)
        # Equal available tier/suite cells; within-cell maps are equal.
        np.testing.assert_allclose(w,[1/3,1/6,1/6,1/3])

    def test_independent_crossing_stop_and_single_sample_matching(self):
        boxes=np.empty((0,2,2));p=np.array([[0.,0.],[-7.,-7.]])
        e=np.array([[1.2,0.],[7.,7.]])
        pp,ee,ap,ae,times,captors=replay_motion(p,e,np.array([[1.,0.],[0.,0.]]),
            np.array([[-1.5,0.],[0.,0.]]),np.ones(2,bool),np.ones(2,bool),boxes)
        self.assertFalse(ae[0]);self.assertTrue(ap.all())
        self.assertAlmostEqual(times[0],.09)
        self.assertAlmostEqual(ee[0,0],1.065)
        self.assertEqual(captors[0],0)
        self.assertAlmostEqual(pp[0,0],1.)

    def test_paired_bootstrap_identical_methods_have_zero_contrast(self):
        rows=[]
        for tier in ('sparse','16','24'):
            for suite in ('ID','OOD'):
                for index in range(2):
                    for arm in ARMS:
                        rows.append(dict(task=dict(group=f'{tier}{suite}{index}',arm=arm,
                            obstacle_tier=tier,suite=suite,family=suite),summary=dict(
                            double_capture=bool(index),completion_cost=40,utility=float(index)-.1,
                            at_least_one=True,latencies_ms=[10.],online_overruns=0)))
        result=statistics(rows,repetitions=100)
        self.assertEqual(result['contrasts'][0]['simultaneous_ci95'],[0.,0.])
        self.assertAlmostEqual(result['methods']['A3']['double_capture'],.5)


if __name__=='__main__':unittest.main()
