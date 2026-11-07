"""The diagnostic must preserve the intended causal comparison."""
import argparse
from pathlib import Path
import sys
import unittest

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_search_rl_ablation as ablation


class AblationTests(unittest.TestCase):
    def test_short_run_does_not_compress_original_curriculum(self):
        options = argparse.Namespace(batch=64,start_index=1024,horizon=5000,opponent_budget=200,depth=8,episodes=128)
        first = ablation.task_plan(options,17,0,0,b'initial',b'anchor')
        options.episodes = 64
        second = ablation.task_plan(options,17,0,0,b'initial',b'anchor')
        self.assertEqual(first,second)
        self.assertEqual(first[0]['progress'],1024/5000)
        self.assertEqual(first[-1]['training_index'],1087)
        self.assertTrue(all(t['opponent_budget']==200 and t['depth']==8 for t in first))
        self.assertTrue(any(t['search_opponent'] for t in first))

    def test_anchor_schedule_is_explicit_and_preserves_original_cutoff(self):
        self.assertAlmostEqual(ablation.actual_anchor('original_linear',512,5000),.0488)
        self.assertEqual(ablation.actual_anchor('original_linear',1000,5000),0.)
        self.assertEqual(ablation.actual_anchor('original_linear',1024,5000),0.)
        self.assertEqual(ablation.actual_anchor('constant',1024,5000),.1)

    def test_exact_kl_matches_torch_and_masks_inactive_agent(self):
        mean0=torch.tensor([[0.,1.],[1.,2.]])
        mean1=mean0+.2
        std0=torch.ones_like(mean0)
        std1=std0*1.2
        mask=torch.tensor([1.,0.])
        before={'logp':torch.zeros(2),'distributions':[(mean0,std0,mask)]}
        after={'distributions':[(mean1,std1,mask)]}
        expected=(torch.distributions.kl_divergence(torch.distributions.Normal(mean0,std0),torch.distributions.Normal(mean1,std1)).sum(-1)*mask).mean()
        self.assertAlmostEqual(ablation.gaussian_kl(before,after),float(expected),places=7)

    def test_composed_team_keeps_its_own_critic_and_normalizer(self):
        source=ablation.trainer.ROOT/'current/search_rl_v3_overlap/pilot_seed0'
        if not source.exists():
            self.skipTest('Pilot artifacts unavailable')
        merged=ablation.compose_initial(source)
        for role in range(2):
            expected=ablation.PolicyBundle.load(source/f'best_role{role}.pt')
            self.assertEqual(ablation.state_hash(merged,role),ablation.state_hash(expected,role))

    def test_sharing_requires_own_actor_critic_and_norm_equality(self):
        original=ablation.PolicyBundle(scratch=True)
        modified=original.copy()
        before=ablation.state_hash(original,1)
        with torch.no_grad():
            next(modified.actors[0].parameters()).add_(.1)
        self.assertEqual(before,ablation.state_hash(modified,1))
        with torch.no_grad():
            next(modified.critics[1].parameters()).add_(.1)
        self.assertNotEqual(before,ablation.state_hash(modified,1))
        modified=original.copy()
        modified.norms[1].count+=1
        self.assertNotEqual(before,ablation.state_hash(modified,1))


if __name__=='__main__':
    unittest.main()
