"""Meaningful acceptance checks for the new speed-bound experiment adapter."""
import importlib
import tempfile
import unittest
from pathlib import Path
import numpy as np

from speed_difficulty_adapter import install, speed_lattice, speed_map_probabilities
from solvers.frozen_d_search_interface import loader, PublicMap, Observation, target_step


class SpeedDifficultyTests(unittest.TestCase):
    def test_scaled_lattice_can_represent_full_speed_and_probabilities(self):
        boxes=np.zeros((0,2,2));last=np.array([0.,0.])
        for speed in (1.3,1.5,1.8):
            points,links,lengths,sensor,root,_=speed_lattice(last,boxes,.5*speed,speed)
            delta=points[links[root,:lengths[root]]]-last
            self.assertAlmostEqual(float(np.max(np.abs(delta))),speed)
            self.assertTrue(np.any(np.max(np.abs(delta),axis=1)>1.))
            public=PublicMap(boxes,speed)
            probabilities,means,_=speed_map_probabilities(points,links,lengths,last,boxes,public.vertices,public.dist,speed)
            np.testing.assert_allclose(probabilities.sum(1),1.,atol=1e-12)
            self.assertLessEqual(float(np.max(np.abs(means))),speed+1e-10)
            rng=np.array([17001],np.int64);node=root;mode=8
            for _ in range(50):
                nxt,mode=target_step(node,mode,links,lengths,probabilities,rng)
                self.assertLessEqual(float(np.max(np.abs(points[nxt]-points[node]))),speed+1e-10)
                node=nxt

    def test_process_hooks_restore_original_one_speed(self):
        import run_l5_followup
        fc=importlib.import_module('l45_corner_shared_20261025.corner_estimator')
        original=(fc.make_graph,fc.map_probabilities,fc.PublicMap)
        with tempfile.TemporaryDirectory() as folder:
            install(1.3,Path(folder))
            self.assertIsNot(fc.make_graph,original[0])
            install(1.,Path(folder))
            self.assertEqual((fc.make_graph,fc.map_probabilities,fc.PublicMap),original)

    def test_hidden_fast_target_is_not_deleted_by_old_speed_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            install(1.3,Path(folder))
            est=loader.load_estimator([],17)
            est.update(Observation(0,np.array([0.,0.]),np.array([5.5,0.]),80))
            est.update(Observation(1,np.array([0.,0.]),None,79))
            points=est.graph[0]
            at=int(np.argmin(np.sum((points-np.array([6.8,0.]))**2,axis=1)))
            self.assertLess(float(np.linalg.norm(points[at]-[6.8,0.])),1e-8)
            self.assertGreater(float(est.weights[at].sum()),0.)
            install(1.,Path(folder))


if __name__=='__main__':
    unittest.main()
