"""Lifecycle semantics that would otherwise silently corrupt the HASAC update."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gymnasium.spaces import Box

from learning.e45v_retirement import (
    FROZEN, HASAC, OffPolicyHARunner, RetirementHASAC, RetirementReplayEP,
    RetirementRunnerMixin, masked_actor_alpha_loss, retirement_target_entropy,
    retirement_train, train_source_audit,
)
from harl.algorithms.critics.soft_twin_continuous_q_critic import SoftTwinContinuousQCritic
from harl.common.buffers.off_policy_buffer_ep import OffPolicyBufferEP


def settings():
    cfg = json.loads((FROZEN / "tuned_configs/pettingzoo_mpe/simple_spread_v2-continuous/hasac/config.json").read_text())["algo_args"]
    cfg["model"]["hidden_sizes"] = [16, 16]
    cfg["train"].update(n_rollout_threads=2, use_proper_time_limits=False, use_valuenorm=False)
    cfg["algo"].update(batch_size=8, buffer_size=32, n_step=1, gamma=1.)
    return cfg


def arrays(batch=8, partial=False):
    rng = np.random.default_rng(123)
    obs = rng.normal(size=(2, batch, 7)).astype(np.float32)
    obs[..., -1] = 1
    if partial:
        obs[0, 2:4, -1] = 0
        obs[1, 4:6, -1] = 0
    active = obs[..., -1:].copy()
    action = rng.uniform(-1, 1, (2, batch, 2)).astype(np.float32) * active
    central = rng.normal(size=(batch, 9)).astype(np.float32)
    return (central, obs, action, None, rng.normal(size=(batch, 1)).astype(np.float32),
            np.zeros((batch, 1), np.float32), active, np.zeros((batch, 1), np.float32),
            central.copy(), obs.copy(), None, np.ones((batch, 1), np.float32))


def update_fixture(retirement=False, partial=False):
    # Unit fixture exercises actual author neural networks and Adam updates;
    # formal production uses the full author constructor and run loop.
    torch.manual_seed(9)
    cfg = settings(); args = {**cfg["model"], **cfg["train"], **cfg["algo"]}
    obs = Box(-np.inf, np.inf, (7,), np.float32)
    cent = Box(-np.inf, np.inf, (9,), np.float32)
    spaces = [Box(-1, 1, (2,), np.float32) for _ in range(2)]
    actor_cls = RetirementHASAC if retirement else HASAC
    actors = [actor_cls(args, obs, a) for a in spaces]
    critic = SoftTwinContinuousQCritic(args, cent, spaces, 2, "EP")
    data = arrays(partial=partial)
    runner = SimpleNamespace(args={"algo": "hasac"}, algo_args=cfg, actor=actors, critic=critic,
                             buffer=SimpleNamespace(sample=lambda: data), device=torch.device("cpu"),
                             num_agents=2, state_type="EP", total_it=0, policy_freq=1,
                             fixed_order=False, value_normalizer=None, target_entropy=[-2, -2])
    runner.log_alpha = [torch.zeros(1, requires_grad=True) for _ in range(2)]
    runner.alpha_optimizer = [torch.optim.Adam([a], lr=3e-4) for a in runner.log_alpha]
    runner.alpha = [torch.exp(a.detach()) for a in runner.log_alpha]
    return runner


class OriginalInsert:
    """The actual source insert, with only constructor-independent state set."""
    insert = OffPolicyHARunner.insert


class LifecycleInsert(RetirementRunnerMixin, OriginalInsert):
    def __init__(self):
        # This fixture tests insert without opening native vector workers.
        pass


def insertion_fixture():
    obj = LifecycleInsert()
    cfg = settings(); args = {**cfg["model"], **cfg["train"], **cfg["algo"]}
    obs = Box(-np.inf, np.inf, (7,), np.float32)
    cent = Box(-np.inf, np.inf, (9,), np.float32)
    spaces = [Box(-1, 1, (2,), np.float32) for _ in range(2)]
    obj.buffer = RetirementReplayEP(args, cent, 2, [obs, obs], spaces)
    obj.args = {"algo": "hasac"}; obj.algo_args = cfg
    obj.state_type = "EP"; obj.num_agents = 2
    obj.agent_deaths = np.zeros((2, 2, 1), np.float32)
    obj.train_episode_rewards = np.zeros(2); obj.done_episodes_rewards = []
    return obj


def insertion_data(active, done=(False, False)):
    obs = np.zeros((2, 2, 7), np.float32); obs[..., -1] = np.asarray(active)
    return (np.zeros((2, 2, 9), np.float32), obs,
            np.ones((2, 2, 2), np.float32), None, np.zeros((2, 2, 1), np.float32),
            np.repeat(np.asarray(done)[:, None], 2, axis=1),
            [[{}, {}], [{}, {}]], np.zeros((2, 2, 9), np.float32),
            obs.transpose(1, 0, 2).copy(), None)


class RetirementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_retired_action_and_entropy(self):
        r = update_fixture(retirement=True)
        obs = arrays(partial=True)[1][0]
        action, logp = r.actor[0].get_actions_with_logprobs(obs)
        retired = obs[:, -1] == 0
        self.assertTrue(torch.equal(action[retired], torch.zeros_like(action[retired])))
        self.assertTrue(torch.equal(logp[retired], torch.zeros_like(logp[retired])))
        self.assertTrue(torch.isfinite(logp).all())

    def test_actor_alpha_uses_only_original_active_rows(self):
        alpha = torch.tensor([.4], requires_grad=True)
        log_prob = torch.tensor([[-3.], [1000.], [-5.], [-1000.]])
        mask = torch.tensor([[1.], [0.], [1.], [0.]])
        value = masked_actor_alpha_loss(alpha, log_prob, mask)
        value.backward()
        self.assertAlmostEqual(float(value.detach()), 1.6, places=6)
        self.assertAlmostEqual(float(alpha.grad), 4., places=6)
        with self.assertRaisesRegex(RuntimeError, "No active actor rows"):
            masked_actor_alpha_loss(alpha, log_prob, torch.zeros_like(mask))

    def test_critic_entropy_target_tracks_active_dimensions(self):
        valid = torch.tensor([[[1.], [0.], [1.]], [[1.], [1.], [0.]]])
        actual = retirement_target_entropy([-2, -2], valid)
        torch.testing.assert_close(actual, torch.tensor([[-4.], [-2.], [-2.]]), rtol=0, atol=0)

    def test_all_active_actual_update_matches_frozen_original(self):
        original = update_fixture()
        adapted = update_fixture(retirement=True)
        for runner, method in ((original, OffPolicyHARunner.train), (adapted, retirement_train)):
            torch.manual_seed(88); np.random.seed(88); method(runner)
        for a, b in zip(original.actor, adapted.actor):
            for key, tensor in a.actor.state_dict().items():
                torch.testing.assert_close(tensor, b.actor.state_dict()[key], rtol=0, atol=0)
        for name in ("critic", "critic2", "target_critic", "target_critic2"):
            for key, tensor in getattr(original.critic, name).state_dict().items():
                torch.testing.assert_close(tensor, getattr(adapted.critic, name).state_dict()[key], rtol=0, atol=0)
        for a, b in zip(original.log_alpha + [original.critic.log_alpha], adapted.log_alpha + [adapted.critic.log_alpha]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_partial_retirement_actual_update_is_finite(self):
        runner = update_fixture(retirement=True, partial=True)
        retirement_train(runner)
        self.assertEqual(runner.total_it, 1)
        for module in [a.actor for a in runner.actor] + [runner.critic.critic, runner.critic.critic2]:
            self.assertTrue(all(torch.isfinite(p).all() for p in module.parameters()))

    def test_current_mask_survives_other_vector_environment_end(self):
        obj = insertion_fixture()
        # env0 ends while env1 has retired actor0. Source clears all deaths;
        # explicit current-observation masks preserve env1's retired status.
        obj.insert(insertion_data([[1, 0], [1, 1]], done=(True, False)))
        self.assertEqual(obj.buffer.valid_transitions[0][1, 0], 0)
        self.assertEqual(obj.buffer.valid_transitions[1][1, 0], 1)
        np.testing.assert_array_equal(obj.buffer.actions[0][1], np.zeros(2))
        self.assertTrue(obj.buffer.terms[0, 0])
        self.assertFalse(obj.buffer.terms[1, 0])
        obj.insert(insertion_data([[1, 0], [1, 1]]))
        self.assertEqual(obj.buffer.valid_transitions[0][3, 0], 0)
        self.assertEqual(obj.buffer.valid_transitions[1][3, 0], 1)

    def test_first_capture_prestate_remains_valid_and_next_retirement_invalid(self):
        obj = insertion_fixture()
        obj.insert(insertion_data([[1, 1], [1, 1]]))
        obj.insert(insertion_data([[0, 1], [1, 1]]))
        self.assertEqual(obj.buffer.valid_transitions[0][0, 0], 1)
        self.assertEqual(obj.buffer.valid_transitions[0][2, 0], 0)
        self.assertEqual(obj.retirement_insert_audit["warmup_nonzero_retired_intents_masked"], 1)

    def test_all_retired_actor_minibatch_has_bounded_redraw(self):
        obj = insertion_fixture()
        batch = list(arrays())
        batch[1][0, :, -1] = 0; batch[2][0] = 0; batch[6][0] = 0
        with patch.object(OffPolicyBufferEP, "sample", return_value=tuple(batch)) as source:
            with self.assertRaisesRegex(RuntimeError, "after 100 draws"):
                obj.buffer.sample()
            self.assertEqual(source.call_count, 100)
            self.assertEqual(obj.buffer.retirement_resamples, 100)

    def test_source_derivation_has_exactly_two_declared_replacements(self):
        audit = train_source_audit()
        self.assertEqual(audit["replacement_count"], 2)
        self.assertEqual(audit["diff"].count("@@"), 4)
        self.assertIn("masked_actor_alpha_loss", audit["diff"])
        self.assertIn("retirement_target_entropy", audit["diff"])


if __name__ == "__main__":
    unittest.main()
