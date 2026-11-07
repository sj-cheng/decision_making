import unittest
from unittest.mock import patch
import torch
import run_e45l_marl_diagnosis as l


class DiagnosisTests(unittest.TestCase):
    def test_environment_seed_fixed_action_seed_separate(self):
        entries=[dict(id='a'),dict(id='b')]
        self.assertEqual(l.situations(entries),l.situations(entries))
        self.assertEqual(len({s for _,_,s in l.situations(entries)}),4)
        env=l.situations(entries)[0][2]
        self.assertNotEqual(env,l.seed_for('train_actions',17,0,0))
        self.assertNotEqual(l.seed_for('train_actions',17,0,0),l.seed_for('train_actions',17,1,0))

    def test_execution_mode_only_changes_policy_draws(self):
        class Own:pass
        def fake(case,task):return Own()
        with patch.object(l,'OLD_FACTORY',fake):
            task=dict(arm='RMAPPO',execution_mode='stochastic',policy_seed=13,seed=471)
            own=l.factory({},task);a=torch.rand(4)
            own2=l.factory({},task);b=torch.rand(4)
            self.assertFalse(own.deterministic);self.assertFalse(own2.deterministic)
            torch.testing.assert_close(a,b)
            task['execution_mode']='deterministic'
            self.assertTrue(l.factory({},task).deterministic)
            self.assertEqual(task['seed'],471)


if __name__=='__main__':unittest.main()
