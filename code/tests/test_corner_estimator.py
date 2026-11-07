import copy
import unittest
import numpy as np

from solvers.local_pursuit import Observation
from solvers.multimodal_estimator import Estimator as Old
from solvers.corner_estimator import Estimator,export_regions
from estimator_audit_geometry import validate_paths


class CornerEstimatorTests(unittest.TestCase):
    def test_old_A_exact_and_export_ablation_preserves_full_distribution(self):
        boxes=[[[-1.,1.],[-5.,5.]]]
        models={m:Estimator(boxes,m,123) for m in ('A','B','C','D','U0','U1')}
        old=Old(boxes,'persistent',123)
        history=[Observation(0,np.array([-3.,6.]),np.array([2.,6.]),80)]
        history += [Observation(t,np.array([-3.,6.-t]),None,80-t) for t in range(1,6)]
        for obs in history:
            old.update(copy.deepcopy(obs))
            for own in models.values():own.update(copy.deepcopy(obs))
            np.testing.assert_array_equal(old.particles,models['A'].particles)
            np.testing.assert_array_equal(old.rng,models['A'].rng)
            if old.weights is not None:
                np.testing.assert_array_equal(old.weights,models['A'].weights)
                for a,b in [('A','B'),('C','D'),('U0','U1')]:np.testing.assert_array_equal(models[a].weights,models[b].weights)
                for own in models.values():np.testing.assert_array_equal(old.reachable,own.reachable)

    def test_blocked_downward_intent_routes_around_wing(self):
        boxes=[[[-1.,1.],[-6.,6.]],[[1.,3.],[4.6,6.]]]
        own=Estimator(boxes,'D',17)
        own.update(Observation(0,np.array([-2.8,6.6]),np.array([2.4,6.6]),80))
        own.update(Observation(1,np.array([-2.8,5.6]),None,79))
        root=own.graph[5]
        down=own.geometry_means[root,4]
        self.assertGreater(down[0],0.)
        self.assertLess(down[1],0.)
        self.assertTrue(np.all(np.abs(down)<=1.))
        errors,_=validate_paths(own.witnesses(),np.array(own.observers),np.array(boxes))
        self.assertEqual(errors,0)

    def test_geometric_export_reserves_important_regions_without_inflating_mass(self):
        weights=np.zeros((100,9));regions=np.repeat(np.arange(4),25)
        weights[3,0]=.96;weights[30,0]=.02;weights[60,0]=.015;weights[90,0]=.005
        ids,w,labels=export_regions(weights,regions,np.array([3],np.int64),64)
        self.assertEqual(len(ids),64)
        self.assertAlmostEqual(w.sum(),1.)
        full=np.bincount(regions,weights=weights.sum(axis=1))
        empirical=np.bincount(regions[ids],weights=w,minlength=4)
        np.testing.assert_allclose(full,empirical,atol=1e-13)
        self.assertTrue(set([0,1,2]).issubset(set(labels.tolist())))
        self.assertTrue(np.all(weights.sum(axis=1)[ids]>0))

    def test_snapshot_and_identical_public_input_invariance(self):
        boxes=[[[-1.,1.],[-5.,5.]]]
        own=Estimator(boxes,'D',93)
        own.update(Observation(0,np.array([-3.,6.]),np.array([2.,6.]),80))
        own.update(Observation(1,np.array([-3.,5.]),None,79))
        saved=own.snapshot();obs=Observation(2,np.array([-3.,4.]),None,78)
        a,w=own.update(obs);distribution=own.weights.copy();rng=own.rng.copy()
        own.restore(saved);b,v=own.update(copy.deepcopy(obs))
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(w,v)
        np.testing.assert_array_equal(distribution,own.weights);np.testing.assert_array_equal(rng,own.rng)
        errors,_=validate_paths(own.witnesses(),np.array(own.observers),np.array(boxes))
        self.assertEqual(errors,0)


if __name__=='__main__':unittest.main()
