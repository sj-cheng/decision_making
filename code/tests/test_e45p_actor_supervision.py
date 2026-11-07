import copy
import unittest
import numpy as np
import torch
from learning.e45m_author_hasac import AuthorHASAC
from learning.e45m_task_adapter import episode_arrays
from learning.e45p_actor_supervision import public_teacher,latent_bc_loss,initialize_bc_std,bc_step,install_anchor,fit_metrics
from tests.test_e45m_hasac_task import synthetic_episode,synthetic_batch


class ActorSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def teacher(self,device='cpu'):
        samples,trace=synthetic_episode()
        return public_teacher(episode_arrays(samples,trace),device)

    def model(self):return AuthorHASAC(obs_dim=845,seed=17,retirement_mask=True)

    def test_public_teacher_excludes_privileged_state(self):
        self.assertEqual(set(self.teacher()),{'obs','action','active'})

    def test_retired_teacher_targets_have_no_effect(self):
        model=self.model();teacher=self.teacher()
        a=latent_bc_loss(model.actor[0],teacher['obs'][:,0],teacher['action'][:,0],teacher['active'][:,0])
        changed=teacher['action'][:,0].clone();changed[teacher['active'][:,0,0]==0]=1
        b=latent_bc_loss(model.actor[0],teacher['obs'][:,0],changed,teacher['active'][:,0])
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_bc_improves_fit_without_updating_critic_or_std(self):
        model=self.model();teacher=self.teacher();initialize_bc_std(model)
        before=copy.deepcopy(model.critic.critic.state_dict());initial=fit_metrics(model,teacher)
        for _ in range(30):bc_step(model,teacher)
        after=fit_metrics(model,teacher)
        self.assertLess(sum(after['latent_mse']),sum(initial['latent_mse']))
        for key,value in before.items():torch.testing.assert_close(value,model.critic.critic.state_dict()[key],rtol=0,atol=0)
        for actor in model.actor:
            torch.testing.assert_close(actor.actor.log_std_layer.weight,torch.zeros_like(actor.actor.log_std_layer.weight),rtol=0,atol=0)
            torch.testing.assert_close(actor.actor.log_std_layer.bias,torch.full_like(actor.actor.log_std_layer.bias,float(np.log(.15))),rtol=0,atol=0)

    def test_zero_weight_is_bit_exact_original_author_update(self):
        a=self.model();b=self.model();install_anchor(b,self.teacher(),0)
        for model in (a,b):
            torch.manual_seed(91);np.random.seed(91);model.update(synthetic_batch())
        for x,y in zip(a.actor,b.actor):
            for key,value in x.actor.state_dict().items():torch.testing.assert_close(value,y.actor.state_dict()[key],rtol=0,atol=0)
        for key,value in a.critic.critic.state_dict().items():torch.testing.assert_close(value,b.critic.critic.state_dict()[key],rtol=0,atol=0)

    def test_hook_matches_combined_loss_single_adam_step(self):
        a=self.model();b=self.model();teacher=self.teacher();_,stats=install_anchor(a,teacher,.1)
        for model in (a,b):
            actor=model.actor[0];actor.turn_on_grad();actor.actor_optimizer.zero_grad()
            objective=actor.get_actions(teacher['obs'][:,0],stochastic=False).square().mean()
            if model is b:
                objective=objective+.1*latent_bc_loss(actor,teacher['obs'][:,0],teacher['action'][:,0],teacher['active'][:,0])
            objective.backward();actor.actor_optimizer.step()
        for key,value in a.actor[0].actor.state_dict().items():
            torch.testing.assert_close(value,b.actor[0].actor.state_dict()[key],rtol=1e-6,atol=1e-7)
        self.assertEqual(stats['calls'],1)

    def test_real_author_update_uses_anchor_once_per_actor_same_critic(self):
        a=self.model();b=self.model();_,stats=install_anchor(b,self.teacher(),.1)
        for model in (a,b):
            torch.manual_seed(19);np.random.seed(19);model.update(synthetic_batch())
        self.assertEqual(stats['calls'],2)
        self.assertTrue(any(not torch.equal(v,b.actor[0].actor.state_dict()[k]) for k,v in a.actor[0].actor.state_dict().items()))
        for key,value in a.critic.critic.state_dict().items():torch.testing.assert_close(value,b.critic.critic.state_dict()[key],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
