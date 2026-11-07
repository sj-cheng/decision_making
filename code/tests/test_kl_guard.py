import copy
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.kl_guard import accepted_actor_step


class KLGuardTests(unittest.TestCase):
    def fixture(self):
        parameter = torch.nn.Parameter(torch.tensor([0.]))
        optimizer = torch.optim.Adam([parameter], lr=1.)
        return parameter, optimizer

    def test_overshoot_backtracks_and_commits_one_adam_step(self):
        p, opt = self.fixture()
        result = accepted_actor_step([p], opt, lambda: (p-10.).square().sum(), lambda: p.square().sum()/2.)
        self.assertTrue(result['accepted'])
        self.assertGreater(len(result['attempts']), 1)
        self.assertLessEqual(float((p.square()/2.).detach()), .02)
        self.assertEqual(int(opt.state[p]['step']), 1)
        self.assertEqual(opt.param_groups[0]['lr'], 1.)

    def test_cumulative_kl_uses_original_batch_not_previous_step(self):
        p, opt = self.fixture()
        for _ in range(12):
            result = accepted_actor_step([p], opt, lambda: (p-10.).square().sum(), lambda: p.square().sum()/2.)
            self.assertLessEqual(float((p.square()/2.).detach()), .02)
            if not result['accepted']:
                break

    def test_rejected_trials_restore_parameters_and_populated_adam(self):
        p, opt = self.fixture()
        accepted_actor_step([p], opt, lambda: (p-10.).square().sum(), lambda: p.square().sum()/2.)
        before, state = p.detach().clone(), copy.deepcopy(opt.state_dict())
        calls = [0]
        def reject():
            calls[0] += 1
            return torch.tensor(0. if calls[0] == 1 else 1.)
        result = accepted_actor_step([p], opt, lambda: (p-10.).square().sum(), reject, maximum_backtracks=3)
        self.assertFalse(result['accepted'])
        self.assertTrue(torch.equal(p, before))
        self.assertEqual(opt.param_groups, [dict(group, params=[p]) for group in state['param_groups']])
        for key, value in state['state'][0].items():
            self.assertTrue(torch.equal(opt.state[p][key], value))

    def test_nonfinite_trial_is_rolled_back(self):
        p, opt = self.fixture()
        calls = [0]
        def broken():
            calls[0] += 1
            return torch.tensor(0. if calls[0] == 1 else float('nan'))
        result = accepted_actor_step([p], opt, lambda: (p-10.).square().sum(), broken, maximum_backtracks=2)
        self.assertFalse(result['accepted'])
        self.assertEqual(float(p.detach()), 0.)
        self.assertFalse(opt.state)

    def test_callback_exception_and_nonfinite_loss_restore_state(self):
        p, opt = self.fixture()
        calls = [0]
        def exploding():
            calls[0] += 1
            if calls[0] > 1:
                raise RuntimeError('diagnostic interrupted')
            return torch.tensor(0.)
        with self.assertRaises(RuntimeError):
            accepted_actor_step([p], opt, lambda: p.square().sum(), exploding)
        self.assertEqual(float(p.detach()), 0.)
        self.assertFalse(opt.state)
        with self.assertRaises(FloatingPointError):
            accepted_actor_step([p], opt, lambda: p.sum()*float('nan'), lambda: p.square().sum())
        self.assertEqual(float(p.detach()), 0.)

    def test_invalid_start_or_partial_optimizer_cannot_bypass_constraint(self):
        p, opt = self.fixture()
        with self.assertRaises(ValueError):
            accepted_actor_step([p], opt, lambda: p.sum(), lambda: torch.tensor(.03))
        other = torch.nn.Parameter(torch.tensor([0.]))
        with self.assertRaises(ValueError):
            accepted_actor_step([other], opt, lambda: other.sum(), lambda: torch.tensor(0.))
        self.assertFalse(opt.state)


if __name__ == '__main__':
    unittest.main()
