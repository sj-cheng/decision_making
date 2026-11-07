"""Reward intervention must preserve trajectories and forbid illegal credit."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_search_rl_reward_experiment as study


class RewardStudyTests(unittest.TestCase):
    def test_completion_bonus_only_on_legal_second_capture(self):
        f=study.reward_increment
        self.assertEqual(f('completion_bonus_v1',2,1,30,80,True),0.)
        self.assertEqual(f('completion_bonus_v1',1,0,79,80,True),1.)
        self.assertEqual(f('completion_bonus_v1',2,0,80,80,True),1.)
        self.assertEqual(f('completion_bonus_v1',0,0,80,80,True),0.)
        self.assertEqual(f('completion_bonus_v1',1,0,30,80,False),0.)

    def test_capture_floor_preserves_late_event_credit(self):
        for t in (0,1,40,79,80):
            for before,after in ((2,1),(1,0),(2,0)):
                legacy=.5*(before-after)*(1-t/80)
                added=study.reward_increment('capture_floor_v1',before,after,t,80,True)
                self.assertAlmostEqual(legacy+added,.5*(before-after))
        self.assertEqual(study.reward_increment('capture_floor_v1',1,0,80,80,False),0.)

    def test_relabel_does_not_modify_behavior_or_original_rewards(self):
        row=dict(reward=.2,legacy_reward=.2,raw=np.array([1.,2.]),old_logp=-.3,
                 reward_increments={'completion_bonus_v1':1.,'legacy':0.})
        data=[dict(rows=[row],stats={})]
        changed=study.relabel(data,'completion_bonus_v1')
        self.assertEqual(changed[0]['rows'][0]['reward'],1.2)
        self.assertEqual(data[0]['rows'][0]['reward'],.2)
        np.testing.assert_array_equal(changed[0]['rows'][0]['raw'],row['raw'])
        self.assertEqual(changed[0]['rows'][0]['old_logp'],row['old_logp'])

    def test_instrumented_legacy_rollout_is_identical(self):
        torch.set_num_threads(1);torch.manual_seed(12)
        bundle=study.PolicyBundle(scratch=True)
        wire=study.trainer.wire_weights(bundle.state_dict())
        task=dict(seed=study.trainer.episode_seed(41,502,1,1024),role=1,kind='ppo',
                  progress=1024/5000,bundle=wire,opponent=wire,search_opponent=False,
                  opponent_budget=1,budget=1,depth=1)
        original=study.trainer.rollout_task(task)
        instrumented=study.reward_rollout(task)
        self.assertEqual(original['spec'],instrumented['spec'])
        self.assertEqual(len(original['rows']),len(instrumented['rows']))
        for a,b in zip(original['rows'],instrumented['rows']):
            for key in a:np.testing.assert_array_equal(a[key],b[key])
        for key in ('steps','double','survival','illegal','returns','transitions'):
            self.assertEqual(original['stats'][key],instrumented['stats'][key])

    def test_unknown_reward_id_fails(self):
        with self.assertRaises(ValueError):study.reward_increment('typo',1,0,30,80,True)


if __name__=='__main__':unittest.main()
