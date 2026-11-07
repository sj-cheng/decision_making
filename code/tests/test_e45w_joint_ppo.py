"""Meaningful density/lifecycle/provider-update checks for the joint PPO adapter."""
from pathlib import Path
import json
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from learning.e45b_joint_network import Network, Runtime
from learning.e45w_joint_ppo import (StructuredJointPolicy, MaskedJointGaussian,
    structured_observation_space, latent_action_space, provider_audit)

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "current/e45c_budget_scaling_20261028/learning/x20/round_12/models/single_17.pt"
NPZ = MODEL.with_suffix(".npz")


def observation(active=(1, 1), seed=4):
    rng = np.random.default_rng(seed)
    x = {k: rng.normal(0, .1, space.shape).astype(np.float32)
         for k, space in structured_observation_space().spaces.items()}
    x["map_mask"][:] = 0; x["map_mask"][:2] = 1
    x["history_mask"][:] = 0; x["history_mask"][:4] = 1
    x["weights"][:] = 1 / 64
    x["particles"][:, :, 2] = 1 / 64
    x["agents"][:, 4] = active
    x["public"][4:6] = active
    return x


def policy(pretrained=True, seed=17):
    return StructuredJointPolicy(structured_observation_space(), latent_action_space(), lambda _: 3e-4,
                                 pretrained_path=MODEL if pretrained else None, initialization_seed=seed)


class SmallTask(gym.Env):
    """Short smooth team task with retirement and true terminal, for real PPO update."""
    def __init__(self):
        self.observation_space = structured_observation_space(); self.action_space = latent_action_space()
        self.t = 0; self.clip_changed = False
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self.t = 0
        return observation(seed=0), {}
    def step(self, action):
        action = np.asarray(action); assert np.all(np.abs(action) < 1e6)
        self.clip_changed |= not np.array_equal(action, np.clip(action, self.action_space.low, self.action_space.high))
        self.t += 1
        active = (1, 1) if self.t < 4 else (1, 0)
        reward = -float(np.square(np.tanh(action[:2]) - .3).sum())
        return observation(active=active, seed=self.t), reward, self.t >= 8, False, {}


class JointPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_pretrained_torch_and_numpy_forward_same_law(self):
        model = policy(); x = observation(); mean, logs, active = model.latent_parameters(x)
        (probs, old_mu, old_logs), _ = Runtime(NPZ)(x)
        np.testing.assert_allclose(mean.detach().numpy()[0], old_mu[0], atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(logs.detach().numpy()[0], old_logs[0], atol=2e-5, rtol=2e-5)
        eps = np.array([-.5, .2, .7, -.1])
        expected = np.tanh(old_mu[0] + np.exp(old_logs[0]) * eps)
        actual = torch.tanh(mean + logs.exp() * torch.as_tensor(eps)).detach().numpy()[0]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    def test_critic_fresh_and_equal_between_random_and_pretrained(self):
        b, c = policy(False), policy(True)
        for k, value in b.critic_network.state_dict().items():
            torch.testing.assert_close(value, c.critic_network.state_dict()[k], atol=0, rtol=0)
        torch.testing.assert_close(c.predict_values(observation()), torch.zeros((1, 1)), atol=0, rtol=0)
        # A value-only gradient does not reach actor representation/heads.
        c.zero_grad(); c.predict_values(observation()).sum().backward()
        self.assertTrue(all(p.grad is None for p in c.actor_network.parameters()))

    def test_retired_coords_zero_and_excluded_from_density_entropy(self):
        model = policy(); obs = observation((1, 0))
        actions, values, logp = model(obs)
        self.assertTrue(torch.equal(actions[:, 2:], torch.zeros_like(actions[:, 2:])))
        changed = actions.clone(); changed[:, 2:] = torch.tensor([123., -456.])
        _, other, entropy = model.evaluate_actions(obs, changed)
        torch.testing.assert_close(logp, other)
        d = model.get_distribution(obs)
        torch.testing.assert_close(entropy, d.distribution.entropy()[:, :2].sum(-1))

    def test_identical_policy_ratio_and_kl(self):
        model = policy(); obs = observation((0, 1))
        actions, _, old = model(obs); _, new, _ = model.evaluate_actions(obs, actions)
        torch.testing.assert_close((new - old).exp(), torch.ones_like(new), atol=1e-6, rtol=1e-6)
        dist = model.get_distribution(obs)
        torch.testing.assert_close(dist.kl_to(model.get_distribution(obs)), torch.zeros_like(old), atol=1e-7, rtol=0)

    def test_latent_and_tanh_ratios_agree_without_inverse_tanh(self):
        mu = torch.tensor([[.2, -.3, .1, .4]])
        logs = torch.tensor([[-.3, -.2, -.5, -.1]])
        active = torch.tensor([[1., 1., 0., 0.]])
        old = MaskedJointGaussian().proba_distribution(mu, logs, active)
        new = MaskedJointGaussian().proba_distribution(mu + .1, logs - .02, active)
        z = torch.tensor([[.4, -.5, 0., 0.]])
        torch.testing.assert_close(new.log_prob(z) - old.log_prob(z),
                                   new.tanh_log_prob_from_latent(z) - old.tanh_log_prob_from_latent(z))
        # Saturated float32 tanh cannot be inverted; direct latent scoring stays finite.
        extreme = torch.tensor([[30., -30., 0., 0.]])
        self.assertTrue(bool(torch.isfinite(old.tanh_log_prob_from_latent(extreme)).all()))

    def test_guard_forbids_silent_latent_clipping(self):
        model = policy()
        with self.assertRaises(RuntimeError): model._check_latents(torch.full((1, 4), 1e6))
        with self.assertRaises(ValueError):
            StructuredJointPolicy(structured_observation_space(), gym.spaces.Box(-1., 1., (4,), np.float32), lambda _: .001)

    def test_real_sb3_ppo_update_and_save_load(self):
        env = SmallTask()
        model = PPO(StructuredJointPolicy, env, n_steps=16, batch_size=8, n_epochs=2,
                    learning_rate=3e-4, gamma=1., gae_lambda=.95, seed=9, device="cpu",
                    policy_kwargs={"pretrained_path": str(MODEL), "initialization_seed": 17}, verbose=0)
        initial = {k: v.clone() for k, v in model.policy.actor_network.state_dict().items()}
        model.learn(total_timesteps=32)
        self.assertEqual(model.num_timesteps, 32); self.assertGreater(model._n_updates, 0)
        self.assertFalse(env.clip_changed)
        self.assertTrue(any(not torch.equal(initial[k], v) for k, v in model.policy.actor_network.state_dict().items()))
        self.assertTrue(all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()))
        obs = observation((1, 0)); expected = model.predict(obs, deterministic=True)[0]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model"; model.save(path)
            loaded = PPO.load(path, env=SmallTask(), device="cpu")
            np.testing.assert_allclose(expected, loaded.predict(obs, deterministic=True)[0], atol=0, rtol=0)
        self.assertTrue(provider_audit()["PPO_train_is_unmodified_provider"])

    def test_real_pursuit_rollout_density_and_ppo_update(self):
        from learning.e45w_joint_env import PublicJointPursuitEnv
        cfg = json.loads((ROOT / "current/e45m_hasac_task_20261101/experiment.json").read_text())
        env = PublicJointPursuitEnv({"case_paths": [cfg["entries"][0]["path"]],
                                    "opponents": ["fixed_D_kernel"]})
        checks = []
        class DensityAudit(BaseCallback):
            def _on_step(self): return True
            def _on_rollout_end(self):
                b = self.model.rollout_buffer
                obs = {key: torch.as_tensor(value.reshape((-1,) + value.shape[2:]), device=self.model.device)
                       for key, value in b.observations.items()}
                actions = torch.as_tensor(b.actions.reshape(-1, 4), device=self.model.device)
                with torch.no_grad(): _, lp, _ = self.model.policy.evaluate_actions(obs, actions)
                np.testing.assert_allclose(lp.cpu().numpy(), b.log_probs.ravel(), atol=2e-5, rtol=2e-5)
                self.assert_bound = bool(np.all(np.abs(b.actions) < 1e6))
                checks.append(self.assert_bound)
        model = PPO(StructuredJointPolicy, env, n_steps=16, batch_size=8, n_epochs=2,
                    learning_rate=3e-4, gamma=1., gae_lambda=.95, seed=31, device="cpu",
                    policy_kwargs={"pretrained_path": str(MODEL), "initialization_seed": 17}, verbose=0)
        old = model.policy.actor_state_numpy()
        model.learn(total_timesteps=80, callback=DensityAudit())
        self.assertEqual(model.num_timesteps, 80); self.assertEqual(len(checks), 5)
        self.assertTrue(all(checks))
        self.assertTrue(any(not np.array_equal(old[k], v) for k, v in model.policy.actor_state_numpy().items()))
        self.assertTrue(all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()))
        model.get_env().close()


if __name__ == "__main__": unittest.main()
