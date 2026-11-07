import unittest
from unittest.mock import patch
import numpy as np
import run_e45d_prior_correlation as d


class FakeRuntime:
    kind='joint'
    def __call__(self,x):return x


class CorrelationTests(unittest.TestCase):
    def test_product_has_same_marginals_and_removes_shared_mode(self):
        params=(np.array([.4,.6]),np.array([[-3.,-3.,-3.,-3.],[3.,3.,3.,3.]]),np.full((2,4),-3.))
        product=d.ProductRuntime(FakeRuntime());rng=np.random.default_rng(123)
        draws=np.array([product.sample(params,rng) for _ in range(12000)])
        positive=draws[:, :, 0]>0
        np.testing.assert_allclose(positive.mean(0),[.6,.6],atol=.015)
        self.assertLess(abs(np.corrcoef(positive.T)[0,1]),.035)
        self.assertTrue(np.all(np.abs(draws)<1))

    def test_factory_preserves_prior_and_disables_value_in_both_arms(self):
        case=dict(map=[],speed=1.1)
        for arm,cls in (('PRIOR','Controller'),('PRODUCT','ProductController')):
            with patch.object(d,cls) as controller:
                d.factory(case,dict(method=arm+'17',seed=9,checkpoint='frozen.npz',budget_ms=200))
                self.assertEqual(controller.call_args.kwargs,dict(use_policy=True,use_value=False,budget_ms=200))

    def test_two_opponent_paired_inference_and_incomplete_family(self):
        rows=[]
        for g in range(5):
            for op in d.base.OPPONENTS:
                for a in d.ARMS:
                    for s in d.SEEDS:
                        rows.append(dict(group=str(g),opponent=op,method=a+str(s),utility=.9 if a=='PRIOR' else .8,
                            double_capture=True,completion_cost=20,latencies_ms=[195.]))
        result=d.summarize(rows)
        self.assertEqual(result['primary_status'],'complete')
        for comparison in result['primary']:self.assertAlmostEqual(comparison['utility_difference'],.1)
        rows[0]['valid_episode']=False
        self.assertIsNone(d.summarize(rows)['primary'])


if __name__=='__main__':unittest.main()
