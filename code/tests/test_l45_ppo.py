"""Behavioral checks for PPO continuation of the unchanged original l45 stack.

All outputs are temporary. The subprocess test uses the explicitly isolated
smoke configuration, never the scientific experiment directory.
"""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch.distributions import Normal, kl_divergence

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import train_l45_ppo as study
from problems.problem import get_problem


class L45PPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def actors(self):
        return study.load_actors(study.original_problem(), directory=study.SOURCE_MODELS)

    def smoke_args(self, output):
        return study.parse_args(['--smoke', '--output', str(output), '--workers', '1'])

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                with self.subTest(key=key):
                    self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def synthetic_batch(self, actors):
        """Actual samples from the current Gaussian, including a masked actor."""
        generator = np.random.default_rng(190)
        episodes = []
        for episode_index in range(2):
            rows = []
            for step in range(4):
                obs = generator.normal(size=(2, 270)).astype(np.float32)
                limits = np.array([[[-.9, .9], [-.9, .9]], [[-1.1, 1.1], [-1.1, 1.1]]], np.float32)
                active = np.array([float(step % 2 == 0), 1.], np.float32)
                raw = np.zeros((2, 2), np.float32)
                mean = np.zeros_like(raw)
                std = np.ones_like(raw)
                logp = 0.
                for local, robot in enumerate(study.TEAMS[1]):
                    with torch.no_grad():
                        dist = study.distribution(actors[robot], torch.tensor(obs[local]), torch.tensor(limits[local]))
                        value = dist.sample()
                    raw[local] = value.numpy()
                    mean[local] = dist.mean.numpy()
                    std[local] = dist.stddev.numpy()
                    logp += float(dist.log_prob(value).sum()) * active[local]
                rows.append(dict(obs=obs, limits=limits, active=active, raw=raw,
                                 old_logp=logp, old_mean=mean, old_std=std,
                                 critic=generator.normal(size=69).astype(np.float32),
                                 reward=float((step - 1) * (episode_index + 1)), done=step == 3))
            episodes.append(dict(rows=rows, stats=dict(environment_return=sum(r['reward'] for r in rows))))
        return episodes

    def optimizers(self, actors, critic, lr=3e-5):
        actor_opt = torch.optim.Adam([v for robot in study.TEAMS[1] for v in actors[robot].parameters()], lr=lr)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
        return actor_opt, critic_opt

    def config(self, **changes):
        return dict(ppo_epochs=4, minibatch=8, clip=.1, entropy=.001, target_kl=.02, **changes)

    def test_zero_update_gaussian_matches_original_mean_std_and_sampling(self):
        p, state = study.restore_scenario(study.make_scenario(330))
        actors = study.load_actors(p, directory=study.SOURCE_MODELS)
        self.assertEqual(p.policy_encoding_dim, 270)
        for robot, actor in enumerate(actors):
            obs = torch.tensor(p.policy_encoding(state, robot).reshape(-1), dtype=torch.float32)
            limits = torch.tensor(p.action_lims[p.action_idxs[robot]], dtype=torch.float32)
            with torch.no_grad():
                dist = study.distribution(actor, obs, limits)
                _, original_logvar = actor(obs[None], training=True)
            np.testing.assert_array_equal(dist.mean.numpy(), actor.eval(p, state, robot, stochastic=False).reshape(-1))
            torch.testing.assert_close(dist.stddev, torch.exp(.5 * original_logvar[0].clamp(-8., 2.)), rtol=0, atol=0)
            torch.manual_seed(410 + robot)
            expected = actor.eval(p, state, robot, stochastic=True).reshape(-1)
            torch.manual_seed(410 + robot)
            sampled = dist.sample().numpy()
            sampled = np.clip(sampled, limits[:, 0].numpy(), limits[:, 1].numpy())
            np.testing.assert_array_equal(sampled, expected)

    def test_export_roundtrip_preserves_original_cpp_search_and_frozen_files(self):
        with tempfile.TemporaryDirectory(prefix='l45_export_test_') as temporary:
            args = self.smoke_args(temporary)
            manifest, _ = study.prepare(args, temporary)
            result = study.correctness_check(args, temporary, manifest)
            self.assertTrue(result['original_cpp_search_exact_match'])
            self.assertEqual(result['actor_mean_max_error'], 0.)
            original = study.model_fingerprint(study.SOURCE_MODELS)
            exported = study.model_fingerprint(Path(temporary) / 'zero_update_models')
            for name in (*study.MODEL_NAMES[:2], study.MODEL_NAMES[-1]):
                self.assertEqual(original[name], exported[name])
            self.assertFalse(study.read_json(Path(temporary) / 'zero_update_models/export.json')['critic_exported'])

    def test_scenarios_restore_original_initialization_without_consuming_global_rng(self):
        np.random.seed(96)
        before = np.random.get_state()
        spec = study.make_scenario(1421)
        self.assert_nested_equal(before, np.random.get_state())
        restored, state = study.restore_scenario(spec)
        np.random.seed(1421)
        original = get_problem('example8')
        original.use_minco_dynamics = False
        expected = original.initialize()
        np.testing.assert_array_equal(state, expected)
        np.testing.assert_array_equal(restored.action_lims, original.action_lims)
        np.testing.assert_array_equal(restored.control_lims, original.control_lims)
        self.assertEqual(restored.current_evader_speed_lim, original.current_evader_speed_lim)
        self.assertEqual(restored.current_pursuer_speed_lim, original.current_pursuer_speed_lim)
        action = np.zeros((original.action_dim, 1))
        np.testing.assert_array_equal(restored.reward(state, action), original.reward(expected, action))
        np.testing.assert_array_equal(restored.step(state, action, restored.dt), original.step(expected, action, original.dt))
        self.assertFalse(restored.swept_collision)
        self.assertFalse(restored.use_minco_dynamics)
        self.assertEqual(len(restored.obstacles), 4)
        tampered = copy.deepcopy(spec)
        tampered['pursuer_speed'] += .1
        with self.assertRaisesRegex(ValueError, 'Scenario fingerprint'):
            study.restore_scenario(tampered)

    def test_endpoint_collision_and_original_illegal_reward_remain_original(self):
        p, state = study.restore_scenario(study.make_scenario(560))
        robot = 2
        obstacle = p.obstacles[0]
        center_y = float(obstacle[1].mean())
        state[p.state_idxs[robot], 0] = [obstacle[0, 0] - 1., center_y]
        velocity = np.array([[obstacle[0, 1] - obstacle[0, 0] + 2.], [0.]])
        endpoint = p.propagate_robot_state(state, velocity, 1., robot)
        np.testing.assert_array_equal(endpoint.reshape(-1), [obstacle[0, 1] + 1., center_y])
        crossed = state.copy()
        crossed[p.state_idxs[robot], :] = endpoint
        self.assertFalse(p.check_obstacle_collision(crossed, robot))
        collided = state.copy()
        collided[p.state_idxs[robot], 0] = obstacle.mean(axis=1)
        collided[p.time_idx, 0] += p.dt
        self.assertTrue(p.check_obstacle_collision(collided, robot))
        self.assertTrue(p.is_terminal(collided))
        np.testing.assert_array_equal(p.transition_reward(state, collided)[list(study.TEAMS[1]), 0], [-1., -1.])

    def test_real_rollout_uses_original_reward_and_records_unclipped_on_policy_action(self):
        with tempfile.TemporaryDirectory(prefix='l45_rollout_test_') as temporary:
            actors = self.actors()
            snapshot = Path(temporary) / 'snapshot.pt'
            study.save_torch(snapshot, dict(actors=study.actor_states(actors), episode=0))
            spec = study.make_scenario(731)
            p, state = study.restore_scenario(spec)
            state[p.time_idx, 0] = p.tf - p.dt
            state[p.active_idxs[0], 0] = state[p.active_idxs[2], 0] = 0.
            spec['initial_state'] = state.tolist()
            spec['episode_hash'] = study.json_hash({k: v for k, v in spec.items() if k != 'episode_hash'})
            args = self.smoke_args(temporary)
            task = dict(spec=spec, snapshot=str(snapshot), snapshot_episode=0,
                        baseline=str(study.SOURCE_MODELS), opponent='Policy Only',
                        settings=study.settings(args), sampling_seed=14)
            result = study.rollout(task)
            self.assertEqual(len(result['rows']), 1)
            row = result['rows'][0]
            self.assertTrue(row['done'])
            np.testing.assert_array_equal(row['active'], [0., 1.])
            np.testing.assert_array_equal(row['raw'][0], [0., 0.])
            p, state = study.restore_scenario(spec)
            opponent = study.make_solver(p, study.SOURCE_MODELS, 'Policy Only', 0,
                                         study.settings(args), spec['search_seeds'][0])
            action = opponent.policy(p, state)
            action[p.action_idxs[3], 0] = np.clip(row['raw'][1], row['limits'][1, :, 0], row['limits'][1, :, 1])
            expected_reward = float(p.reward(state, action)[list(study.TEAMS[1]), 0].mean())
            self.assertEqual(row['reward'], expected_reward)
            self.assertEqual(result['stats']['environment_return'], expected_reward)
            computed = study.joint_log_prob(actors, torch.tensor(row['obs'][None]),
                                            torch.tensor(row['raw'][None]), torch.tensor(row['limits'][None]),
                                            torch.tensor(row['active'][None]))
            self.assertAlmostEqual(float(computed.detach()), row['old_logp'], places=5)
            expected_critic = study.critic_features(p, state)
            np.testing.assert_array_equal(row['critic'], expected_critic)

    def test_gaussian_logprob_masks_inactive_actor_and_gae_masks_terminal(self):
        torch.manual_seed(102)
        actors = self.actors()
        episodes = self.synthetic_batch(actors)
        row = episodes[0]['rows'][1]
        obs = torch.tensor(row['obs'][None])
        limits = torch.tensor(row['limits'][None])
        active = torch.tensor(row['active'][None])
        raw = torch.tensor(row['raw'][None])
        before = study.joint_log_prob(actors, obs, raw, limits, active)
        altered = raw.clone()
        altered[:, 0] += 100.
        after = study.joint_log_prob(actors, obs, altered, limits, active)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        adv, target = study.advantages([1., 2., 100.], [10., 20., 30.], [False, True, True], gamma=.9, lam=.8)
        np.testing.assert_allclose(adv, [9. + .72 * -18., -18., 70.], atol=2e-6)
        np.testing.assert_allclose(target, adv + [10., 20., 30.], atol=2e-6)

    def test_ppo_updates_pursuers_and_critic_but_never_evaders(self):
        torch.manual_seed(12)
        actors = self.actors()
        critic = study.TrainingCritic()
        episodes = self.synthetic_batch(actors)
        actors_before = study.actor_states(actors)
        critic_before = copy.deepcopy(critic.state_dict())
        actor_opt, critic_opt = self.optimizers(actors, critic)
        result = study.update_ppo(actors, critic, actor_opt, critic_opt, episodes, self.config())
        self.assertLess(result['old_logp_max_error'], 3e-4)
        for robot in study.TEAMS[0]:
            self.assert_nested_equal(actors_before[robot], actors[robot].state_dict())
        for robot in study.TEAMS[1]:
            self.assertTrue(any(not torch.equal(v, actors_before[robot][k]) for k, v in actors[robot].state_dict().items()))
        self.assertTrue(any(not torch.equal(v, critic_before[k]) for k, v in critic.state_dict().items()))
        invalid = copy.deepcopy(episodes)
        invalid[0]['rows'][0]['old_logp'] += 1.
        fresh = self.actors()
        fresh_critic = study.TrainingCritic()
        fresh_opts = self.optimizers(fresh, fresh_critic)
        with self.assertRaisesRegex(RuntimeError, 'On-policy logprob mismatch'):
            study.update_ppo(fresh, fresh_critic, *fresh_opts, invalid, self.config())

    def test_exact_full_batch_kl_stops_actor_epochs_while_critic_continues(self):
        torch.manual_seed(15)
        actors = self.actors()
        critic = study.TrainingCritic()
        episodes = self.synthetic_batch(actors)
        actor_opt, critic_opt = self.optimizers(actors, critic, lr=.01)
        config = self.config()
        config['target_kl'] = 0.
        with mock.patch.object(actor_opt, 'step', wraps=actor_opt.step) as actor_steps, mock.patch.object(critic_opt, 'step', wraps=critic_opt.step) as critic_steps:
            result = study.update_ppo(actors, critic, actor_opt, critic_opt, episodes, config)
        self.assertTrue(result['kl_stopped'])
        self.assertEqual(result['actor_epochs'], 1)
        self.assertEqual(actor_steps.call_count, 1)
        self.assertEqual(critic_steps.call_count, 4)
        rows = [row for episode in episodes for row in episode['rows']]
        with torch.no_grad():
            per_row = []
            for row in rows:
                kl = 0.
                for local, robot in enumerate(study.TEAMS[1]):
                    old = Normal(torch.tensor(row['old_mean'][local]), torch.tensor(row['old_std'][local]))
                    new = study.distribution(actors[robot], torch.tensor(row['obs'][local]), torch.tensor(row['limits'][local]))
                    kl += float(kl_divergence(old, new).sum()) * row['active'][local]
                per_row.append(kl)
        self.assertAlmostEqual(result['exact_kl_by_epoch'][0], float(np.mean(per_row)), places=5)

    def test_nonfinite_on_policy_input_or_reward_rejected_before_any_update(self):
        torch.manual_seed(819)
        actors = self.actors()
        critic = study.TrainingCritic()
        source = self.synthetic_batch(actors)
        original = study.actor_states(actors)
        actor_opt, critic_opt = self.optimizers(actors, critic)
        for field in ('obs', 'old_logp', 'reward'):
            with self.subTest(field=field):
                invalid = copy.deepcopy(source)
                row = invalid[0]['rows'][0]
                if field == 'obs':
                    row[field][0, 0] = np.nan
                else:
                    row[field] = float('nan')
                with self.assertRaises(FloatingPointError):
                    study.update_ppo(actors, critic, actor_opt, critic_opt, invalid, self.config())
                self.assert_nested_equal(original, study.actor_states(actors))
                self.assertFalse(actor_opt.state)
                self.assertFalse(critic_opt.state)

    def test_prepare_is_repeatable_disjoint_and_rejects_source_or_scenario_changes(self):
        with tempfile.TemporaryDirectory(prefix='l45_prepare_test_') as temporary:
            args = self.smoke_args(temporary)
            implementation = {'test_copy': 'original'}
            with mock.patch.object(study, 'implementation_files', return_value=implementation):
                manifest, signature = study.prepare(args, temporary)
                again, next_signature = study.prepare(args, temporary)
                self.assertEqual(manifest, again)
                self.assertEqual(signature, next_signature)
                hashes = [spec['episode_hash'] for split in manifest.values() for spec in split]
                self.assertEqual(len(hashes), len(set(hashes)))
                study.verify_experiment(args, temporary)
                with mock.patch.object(study, 'implementation_files', return_value={'test_copy': 'changed'}):
                    with self.assertRaisesRegex(ValueError, 'fingerprint/configuration'):
                        study.prepare(args, temporary)
                    with self.assertRaisesRegex(ValueError, 'implementation changed'):
                        study.verify_experiment(args, temporary)
                scenarios = Path(temporary) / 'scenarios.json'
                changed = study.read_json(scenarios)
                changed['validation'][0]['seed'] += 1
                study.write_json(scenarios, changed)
                with self.assertRaisesRegex(ValueError, 'Frozen scenarios changed'):
                    study.prepare(args, temporary)

    def test_prepare_rejects_changed_frozen_model_without_touching_original_models(self):
        with tempfile.TemporaryDirectory(prefix='l45_model_integrity_test_') as temporary:
            args = self.smoke_args(temporary)
            study.prepare(args, temporary)
            frozen = Path(temporary) / 'baseline' / study.MODEL_NAMES[0]
            frozen.write_bytes(frozen.read_bytes() + b'changed-test-copy')
            with self.assertRaisesRegex(ValueError, 'Frozen baseline models changed'):
                study.prepare(args, temporary)
            with self.assertRaisesRegex(ValueError, 'Original models or implementation changed'):
                study.verify_experiment(args, temporary)

    def test_interrupted_subprocess_smoke_resumes_bitwise_identically(self):
        python = Path('/home/sjc/miniconda3/envs/dm_env_py311/bin/python')
        if not python.exists():
            python = Path(sys.executable)
        with tempfile.TemporaryDirectory(prefix='l45_resume_test_') as temporary:
            continuous = Path(temporary) / 'continuous'
            resumed = Path(temporary) / 'resumed'
            def execute(output, max_updates=0):
                command = [str(python), str(CODE / 'train_l45_ppo.py'), '--smoke', '--stage', 'train',
                           '--output', str(output), '--workers', '1', '--max-updates', str(max_updates)]
                result = subprocess.run(command, cwd=CODE, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout[-12000:])
            execute(continuous)
            execute(resumed, max_updates=1)
            paused = study.load_torch(resumed / 'seed0/latest.pt')
            self.assertEqual(paused['episode'], 2)
            self.assertFalse((resumed / 'seed0/final.pt').exists())
            self.assertEqual(study.read_json(resumed / 'status.json')['phase'], 'stopped_at_checkpoint')
            execute(resumed)
            left = study.load_torch(continuous / 'seed0/final.pt')
            right = study.load_torch(resumed / 'seed0/final.pt')
            for key in ('actors', 'critic', 'actor_optimizer', 'critic_optimizer', 'rng', 'episode',
                        'best_episode', 'best_score', 'pending_validation', 'signature'):
                with self.subTest(field=key):
                    self.assert_nested_equal(left[key], right[key])
            self.assertEqual(left['episode'], 4)
            self.assertEqual(study.read_json(resumed / 'status.json')['phase'], 'training_complete')
            source = study.model_fingerprint(study.SOURCE_MODELS)
            exported = study.model_fingerprint(resumed / 'seed0/final_models')
            for name in (*study.MODEL_NAMES[:2], study.MODEL_NAMES[-1]):
                self.assertEqual(source[name], exported[name])


if __name__ == '__main__':
    unittest.main()
