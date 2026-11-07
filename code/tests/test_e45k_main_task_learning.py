import unittest
import numpy as np
from run_e45k_main_task_learning import rewards, schedule
from learning.e45g_ppo import gae


class MainTaskLearningTests(unittest.TestCase):
    def test_complete_capture_preserves_total_but_single_capture_gets_credit(self):
        for events, expected in [([0,1,0,1],.99),([0,1,0,0],.49),([0,0,0,0],-.01),([0,0,0,2],.99)]:
            raw=np.full(4,-.0025)
            if sum(events)==2:raw[-1]+=1
            r=rewards(raw,np.zeros(4),np.zeros(4),events,'capture_credit')
            self.assertAlmostEqual(float(r.sum()),expected,places=6)
            original=rewards(raw,np.zeros(4),np.zeros(4),events,'original_reward')
            np.testing.assert_allclose(original,raw,atol=1e-7)

    def test_terminal_potential_and_intent_cost_remain_explicit(self):
        raw=np.array([-.0025,.9975]);phi=np.array([-.8,.2]);blocked=np.array([.5,0.])
        for variant in ['original_reward','capture_credit']:
            r=rewards(raw,phi,blocked,[1,1],variant)
            self.assertAlmostEqual(float(r.sum()),.995-phi[0]-.05*blocked.sum(),places=6)

    def test_partial_reward_credits_both_team_members_before_retirement(self):
        r=rewards(np.full(12,-.0025),np.zeros(12),np.zeros(12),[0]*10+[1,0],'capture_credit')
        _,ret=gae(r[:,None],np.zeros((12,2)),gamma=1.,lam=.95)
        self.assertGreater(ret[0,0],0);np.testing.assert_array_equal(ret[:,0],ret[:,1])

    def test_paired_episode_schedule_is_policy_independent(self):
        entries=[{'id':str(i)} for i in range(36)]
        a=schedule(entries,17,0);b=schedule(entries,17,0)
        self.assertEqual(a,b)
        self.assertEqual(sum(x[2]=='rootpuct' for x in a),18)
        self.assertNotEqual(a,schedule(entries,17,1))
        self.assertNotEqual(a,schedule(entries,29,0))


if __name__=='__main__':unittest.main()
