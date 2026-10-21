"""Fast structural checks for the MAPPO baseline observation contract."""
import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from problems.problem import get_problem
from learning.mappo import GaussianActor, agent_observation, agent_observation_dim
from train_mappo import (
    ValueNorm, active_mean_reward, entropy_coefficient, episode_mode,
    executed_normalized_action, expert_normalized_actions, masked_joint_log_prob,
    sample_teams, scripted_normalized_action,
)


class MAPPOObservationTest(unittest.TestCase):
    def setUp(self):
        self.problem = get_problem("example8")
        self.problem.obstacles = []
        self.problem.evader_spawn_near_obstacles = False
        self.problem.use_minco_dynamics = False
        self.state = self.problem.initialize()

    def test_fixed_dimension_and_identity(self):
        a = agent_observation(self.problem, self.state, 0)
        b = agent_observation(self.problem, self.state, 1)
        self.assertEqual(a.shape, (agent_observation_dim(self.problem),))
        self.assertFalse(np.allclose(a, b))

    def test_observation_is_finite(self):
        for robot in range(self.problem.num_robots):
            self.assertTrue(np.isfinite(agent_observation(self.problem, self.state, robot)).all())

    def test_transition_reward_matches_reward_api(self):
        other = get_problem("example8")
        other.obstacles = []
        other.evader_spawn_near_obstacles = False
        other.use_minco_dynamics = False
        state = np.array(self.state, copy=True)
        action = np.zeros((self.problem.action_dim, 1))
        combined = self.problem.reward(state, action)
        nxt = other.step(state, action, other.dt)
        direct = other.transition_reward(state, nxt)
        self.assertTrue(np.allclose(combined, direct))

    def test_capture_reward_keeps_environment_sign(self):
        state = np.array(self.state, copy=True)
        state[self.problem.state_idxs[2], :] = state[self.problem.state_idxs[0], :]
        action = np.zeros((self.problem.action_dim, 1))
        reward = self.problem.reward(state, action)
        self.assertLess(float(reward[0, 0]), 0.0)
        self.assertGreater(float(reward[2, 0]), 0.0)

    def test_inactive_robot_is_omitted_from_the_joint_action(self):
        state = np.array(self.state, copy=True)
        state[self.problem.active_idxs[1], 0] = 0.0
        actors = [GaussianActor(agent_observation_dim(self.problem)) for _ in range(2)]
        _, per_team = sample_teams(self.problem, actors, state, torch.device("cpu"))
        self.assertEqual(per_team[0]["active"].tolist(), [1.0, 0.0])
        self.assertTrue(np.all(per_team[0]["actions"][1] == 0.0))
        self.assertEqual(per_team[1]["active"].tolist(), [1.0, 1.0])

    def test_masked_log_prob_ignores_inactive_action(self):
        actor = GaussianActor(4, action_dim=2)
        obs = torch.randn(3, 2, 4)
        action = torch.tanh(torch.randn(3, 2, 2))
        active = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        changed = action.clone()
        changed[0, 1, :] = 0.3
        changed[2, 0, :] = -0.4
        original = masked_joint_log_prob(actor, obs, action, active)
        altered = masked_joint_log_prob(actor, obs, changed, active)
        self.assertTrue(torch.allclose(original, altered))

    def test_active_mean_reward_drops_inactive_robot(self):
        self.assertAlmostEqual(active_mean_reward([1.0, 5.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(active_mean_reward([1.0, 5.0], [1.0, 1.0]), 3.0)

    def test_expert_assigns_distinct_evaders(self):
        state = np.array(self.state, copy=True)
        state[self.problem.state_idxs[0], 0] = np.array([-6.0, 0.0])
        state[self.problem.state_idxs[1], 0] = np.array([6.0, 0.0])
        state[self.problem.state_idxs[2], 0] = np.array([-6.0, 3.0])
        state[self.problem.state_idxs[3], 0] = np.array([6.0, 3.0])
        actions = expert_normalized_actions(self.problem, state)
        self.assertLess(float(actions[2][0]), 0.3)
        self.assertGreater(float(actions[3][0]), -0.3)
        self.assertLess(float(actions[2][1]), 0.0)
        self.assertLess(float(actions[3][1]), 0.0)

    def test_shield_removes_outward_residual_at_the_wall(self):
        state = np.array(self.state, copy=True)
        state[self.problem.state_idxs[0], 0] = np.array([9.5, 0.0])
        action = executed_normalized_action(
            np.array([0.0, 0.2], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
            self.problem, state, 0)
        self.assertEqual(float(action[0]), 0.0)
        self.assertGreater(float(action[1]), 0.0)

    def test_scripted_evader_does_not_step_outward_at_the_wall(self):
        state = np.array(self.state, copy=True)
        state[self.problem.state_idxs[0], 0] = np.array([9.5, 0.0])
        state[self.problem.state_idxs[2], 0] = np.array([5.0, 0.0])
        action = scripted_normalized_action(self.problem, state, 0)
        self.assertLessEqual(float(action[0]), 0.0)

    def test_episode_mode_reserves_self_play(self):
        modes = [episode_mode(seed, 0.5) for seed in range(200)]
        self.assertIn("self", modes)
        self.assertIn("script_evader", modes)
        self.assertIn("script_pursuer", modes)
        self.assertEqual(episode_mode(3, 0.0), "self")

    def test_value_norm_roundtrip(self):
        normalizer = ValueNorm(beta=0.0)
        values = np.array([0.0, 2.0, 4.0], dtype=np.float32)
        normalizer.update(values)
        restored = normalizer.denormalize(normalizer.normalize(values))
        self.assertTrue(np.allclose(restored, values, atol=1e-5))

    def test_entropy_coefficient_anneals(self):
        args = type("Args", (), {"entropy_coef": 0.01, "entropy_coef_final": 0.001})()
        self.assertAlmostEqual(entropy_coefficient(args, 0, 5), 0.01)
        self.assertAlmostEqual(entropy_coefficient(args, 4, 5), 0.001)


if __name__ == "__main__":
    unittest.main()
