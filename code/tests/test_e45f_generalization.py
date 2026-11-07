import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import run_e45f_generalization_benchmark as f
from e45f_scenarios import OOD_FAMILIES,shifted_case,geometry_key
from solvers.e45f_baselines import best_voronoi_mask,VoronoiController


class GeneralizationTests(unittest.TestCase):
    def test_new_layout_histories_are_causal_shared_and_within_encoder_capacity(self):
        for layout in OOD_FAMILIES:
            for index in (0,1):
                c=shifted_case(layout,index,'test',945720261029)
                self.assertTrue(3<=len(c['map'])<=6)
                self.assertEqual(c['suite'],'generalization')
                self.assertTrue(all(q is None for report in c['history'][-1]['reports'] for q in report))
                for w in c['worlds']:
                    self.assertEqual(w['initial'],w['scripted_states'][-1])

    def test_geometry_key_removes_translation_and_quarter_turn(self):
        boxes=np.array([[[-1,1],[-5,5]],[[4,5],[2,3]]],float)
        changed=[]
        for box in boxes:
            corners=np.array([[x,y] for x in box[0] for y in box[1]])
            points=corners[:,[1,0]]*[-1,1]+[.7,-1.2]
            changed.append(np.stack((points.min(0),points.max(0)),axis=1))
        self.assertEqual(geometry_key(boxes),geometry_key(changed))

    def test_voronoi_metric_ignores_inactive_robot(self):
        actions=np.array([[[-.8,0],[100,0]],[[.8,0],[-100,0]]])
        candidates=np.array([[[.6,0],[1000,0]],[[-.6,0],[-1000,0]]])
        np.testing.assert_array_equal(best_voronoi_mask(candidates,actions,1,[True,False]),[True,False])

    def test_voronoi_sampler_concentrates_in_best_cell_without_forbidden_actions(self):
        v=VoronoiController.__new__(VoronoiController)
        v.public=SimpleNamespace(speed=1.);v.motion_boxes=np.empty((0,2,2));v.allowed=lambda:None
        v.proposal_rng=np.random.default_rng(90);v.rejections=v.fallbacks=0
        node=SimpleNamespace(visits=[20,20],totals=[0.,20.],ap=np.array([True,False]),
            positions=np.zeros((2,2)),actions=[np.array([[-.8,0],[0,0]]),np.array([[.8,0],[0,0]])])
        samples=np.array([v.propose(node) for _ in range(800)])
        self.assertGreater(np.mean(samples[:,0,0]>0),.91)
        self.assertTrue(np.all(samples[:,1]==0));self.assertTrue(np.max(abs(samples))<=1.)

    def test_factory_controls_same_checkpoint_value_and_direct_flags(self):
        case=dict(map=[],speed=1.1)
        for arm in f.LEARNED:
            with patch.object(f,'Controller') as cls:
                f.factory(case,dict(method=arm+'17',seed=42,checkpoint='frozen.npz',budget_ms=200.))
                self.assertEqual(cls.call_args.args,([],1.1,42,'frozen.npz'))
                self.assertEqual(cls.call_args.kwargs,dict(use_policy=True,use_value=arm=='SINGLE_VALUE',direct=arm=='SINGLE_DIRECT',budget_ms=200.))

    def test_suite_inference_keeps_all_worlds_seeds_and_separates_failures(self):
        rows=[]
        for suite in ('benchmark','generalization'):
            for group in range(4):
                for world in ('AB','BA'):
                    for op in f.base.OPPONENTS:
                        for arm in f.ARMS:
                            for seed in (f.SEEDS if arm in f.LEARNED else (0,)):
                                rows.append(dict(suite=suite,group=suite+str(group),family='family'+str(group%2),world=world,
                                    opponent=op,method=arm+str(seed) if seed else arm,train_seed=seed,
                                    utility=.9 if arm=='SINGLE_PRIOR' else .8,double_capture=True,completion_cost=20,latencies_ms=[195.]))
        with patch.object(f,'SMOKE',False):
            result=f.summarize(rows)
            self.assertEqual(result['primary_status'],'complete')
            for data in result['primary'].values():
                self.assertEqual(len(data['comparisons']),18)
                for comparison in data['comparisons']:
                    np.testing.assert_allclose(comparison['simultaneous95'],[.1,.1],atol=1e-14)
            rows[0]['valid_episode']=False
            result=f.summarize(rows)
            self.assertEqual(result['primary']['benchmark']['status'],'withheld_incomplete_cells')
            self.assertEqual(result['primary']['generalization']['status'],'complete')


if __name__=='__main__':unittest.main()
