"""Single-S hard-KL integration, scientific gates and resumable batch checks."""
import copy
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_l45_guarded_supervision as s


class GuardedSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix='l45_guarded_tests_')
        cls.args = s.parse_args(['--smoke', '--output', cls.temp.name, '--workers', '4'])
        cls.banks = s.prepare(cls.args)
        cls.runtime = s.load_runtime(cls.args, 'S', 0)
        cls.tasks = s.sampling_tasks(cls.args, cls.banks, cls.runtime)
        with mp.get_context('spawn').Pool(4) as pool:
            s.sample_batch(cls.args, cls.tasks, pool, lambda *a, **k: None, 'fixture_sampling')
            cls.teaching = s.teacher_tasks(cls.args, cls.runtime, cls.tasks)
            s.teach_batch(cls.args, cls.teaching, pool, lambda *a, **k: None, 'fixture_teacher')
        cls.episodes = [s.read_rollout(task) for task in cls.tasks]
        cls.labels = [label for task in cls.teaching for label in s.read_teacher(task)['labels']]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def fresh(self):
        return s.load_runtime(self.args, 'S', 0)

    def update(self, runtime):
        s.o.restore_rng(runtime['saved']['rng'])
        return s.update_policy(runtime['actors'], runtime['critic'], runtime['actor_opt'], runtime['critic_opt'],
                               self.episodes, s.config(self.args), self.labels, 'S')

    def synthetic(self, role=1, gain=True):
        rows = []
        for uid in (s.CONTROL, s.BASELINE, 'S0'):
            for layout in ('four', 'none'):
                for index in range(100):
                    cutoff = 80 if layout == 'none' else {s.CONTROL: 50, s.BASELINE: 45, 'S0': 75 if gain else 50}[uid]
                    for op in s.o.OPPONENTS:
                        success = int(index < cutoff)
                        rows.append(dict(uid=uid, role=role, layout=layout, opponent=op,
                            episode_hash=f'{layout}_{index}', search_seeds=[index, index+1],
                            legal_double_capture=success, legal_survival=success, pursuer_illegal=0,
                            evader_illegal=0, one_capture_timeout=1-success, completion_cost=40. if success else 80.))
        return rows

    def test_protocol_single_arm_seed_budget_and_two_comparisons(self):
        cfg = s.config(s.parse_args([]))
        self.assertEqual((s.ARMS, s.SEEDS), (('S',), (0,)))
        self.assertEqual(cfg['max_training_episodes'], 512)
        self.assertEqual(cfg['max_teacher_queries'], 8192)
        self.assertEqual(cfg['ppo_weights'], {'S': 0.})
        self.assertEqual(len(s.comparison_specs()), 2)
        self.assertEqual(cfg['bootstrap_seed'], 47020261023)
        with self.assertRaises(ValueError):
            s.load_runtime(self.args, 'J', 0)
        with self.assertRaises(ValueError):
            s.load_runtime(self.args, 'S', 1)

    def test_complete_initialization_and_frozen_weights(self):
        source = s.o.load_torch(self.args.output/'initial/source.pt')
        runtime = self.fresh()
        for key in ('actors', 'critic', 'actor_optimizer', 'critic_optimizer'):
            self.assertTrue(s.equal_tree(runtime['saved'][key], source[key]))
        self.assertTrue(s.equal_tree(runtime['actor_opt'].state_dict(), source['actor_optimizer']))
        self.assertEqual(runtime['saved']['source_episode'], 2048)
        for actor in runtime['actors'][:2]:
            self.assertFalse(any(p.requires_grad for p in actor.parameters()))

    def test_disjoint_scenes_include_completion_unused_banks_and_smoke(self):
        files = [Path(path) for path in s.locked_read(self.args.output/'experiment.json')['history']]
        self.assertTrue(any('l45_completion_search_20261023/scenarios.json' in str(p) for p in files))
        self.assertTrue(any('l45_completion_search_smoke_20261023/scenarios.json' in str(p) for p in files))
        used = s.history_scenes(files)
        identities = [s.previous.physical_hash(spec) for bank in self.banks.values() for spec in bank]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertFalse(used.intersection(identities))

    def test_exact_opponent_mix_and_teacher_phase_markers(self):
        schedule = s.opponent_schedule(self.args, 0, 0)
        self.assertEqual(set(schedule), {'S'})
        self.assertEqual([schedule['S'].count(op) for op in s.o.OPPONENTS], [16, 16, 32])
        selected, allocation = s.select_teacher_rows(self.args, self.runtime, self.episodes)
        self.assertEqual(len(selected), 32)
        self.assertEqual(allocation['before_first'], 16)
        self.assertEqual(allocation['after_first'], 16)
        broken = copy.deepcopy(self.episodes[:1])
        broken[0]['rows'][0]['state'][9, 0] = 0.
        with self.assertRaisesRegex(ValueError, 'phase'):
            s.select_teacher_rows(self.args, self.runtime, broken)

    def test_actual_teacher_deployment_and_zero_export_audit(self):
        audit = s.audit(self.args, self.banks)
        self.assertTrue(audit['passed'])
        self.assertTrue(all(p['joint_action_exact_match'] for p in audit['teacher_joint_action_probes']))
        for actual in audit['actual_cpp_settings']:
            if actual['method'] == 'Search + Learning':
                self.assertAlmostEqual(actual['beta_value'], .1 if actual['role'] == 0 else 0.)

    def test_hard_kl_every_step_actor_only_and_critic_finishes(self):
        runtime = self.fresh()
        before = s.o.actor_states(runtime['actors'])
        critic = copy.deepcopy(runtime['critic'].state_dict())
        metric = self.update(runtime)
        self.assertLessEqual(metric['max_accepted_kl'], .02)
        self.assertLessEqual(metric['final_exact_kl'], .02)
        self.assertTrue(all(step['final_kl'] <= .02 for step in metric['accepted_step_trace']))
        self.assertEqual(metric['critic_steps'], 4*((metric['samples']+255)//256))
        self.assertTrue(s.equal_tree(before[:2], s.o.actor_states(runtime['actors'])[:2]))
        self.assertFalse(s.equal_tree(critic, runtime['critic'].state_dict()))
        self.assertEqual(metric['ppo_weight'], 0.)
        self.assertIsNone(metric['ppo_loss'])

    def test_rejected_actor_batch_keeps_actor_and_updates_critic(self):
        runtime = self.fresh()
        before = s.o.actor_states(runtime['actors'])
        adam = copy.deepcopy(runtime['actor_opt'].state_dict())
        reject = dict(accepted=False, initial_kl=0., final_kl=0., loss=0., gradient_norm=0.,
                      attempts=[dict(scale=1., cumulative_kl=.1, accepted=False)])
        with mock.patch.object(s, 'accepted_actor_step', return_value=reject) as step:
            metric = self.update(runtime)
        self.assertEqual(step.call_count, 1)
        self.assertEqual(metric['actor_steps'], 0)
        self.assertEqual(metric['actor_stop_reason'], 'no_acceptable_step')
        self.assertEqual(metric['critic_steps'], 4*((metric['samples']+255)//256))
        self.assertTrue(s.equal_tree(before, s.o.actor_states(runtime['actors'])))
        self.assertTrue(s.equal_tree(adam, runtime['actor_opt'].state_dict()))

    def test_duplicate_or_cross_state_labels_rejected(self):
        runtime = self.fresh()
        for labels in (self.labels+[self.labels[0]], [dict(self.labels[0], state_sha256='changed')]):
            with self.assertRaises(ValueError):
                s.update_policy(runtime['actors'], runtime['critic'], runtime['actor_opt'], runtime['critic_opt'],
                                self.episodes, s.config(self.args), labels, 'S')

    def test_teacher_calls_do_not_change_sampled_trajectories_and_workers_match(self):
        task = self.tasks[0]
        before = s.rollout(task)
        s.teacher_one(self.teaching[0])
        after = s.rollout(task)
        self.assertTrue(s.equal_tree(before['rows'], after['rows']))
        with mp.get_context('spawn').Pool(2) as pool:
            parallel = pool.map(s.rollout, [task])[0]
        self.assertTrue(s.equal_tree(before['rows'], parallel['rows']))

    def late_spec(self):
        spec = copy.deepcopy(self.banks['validation_four'][0])
        p, state = s.o.restore_scenario(spec)
        state[p.time_idx, 0] = p.tf-p.dt
        spec['initial_state'] = state.tolist()
        spec['episode_hash'] = s.o.json_hash({k: v for k, v in spec.items() if k != 'episode_hash'})
        return spec

    def test_alias_cache_resume_and_tampering(self):
        current = s.controls(self.args)[0]
        items = s.requests(self.args, [current, dict(current, uid='S0')], [self.late_spec()], opponents=('Policy Only',))
        self.assertEqual(items[0]['key'], items[1]['key'])
        with mock.patch.object(s, 'evaluate_one', wraps=s.evaluate_one) as evaluator:
            s.run_evaluation(self.args, 'fixture_alias', items, None, lambda *a, **k: None)
            s.run_evaluation(self.args, 'fixture_alias', items, None, lambda *a, **k: None)
            self.assertEqual(evaluator.call_count, 1)
        target = self.args.output/'cache'/(items[0]['key']+'.json')
        original = target.read_bytes()
        broken = s.o.read_json(target); broken['legal_double_capture'] = 9
        s.o.write_json(target, broken)
        try:
            with self.assertRaises(ValueError):
                s.read_cached(self.args, items[0])
        finally:
            target.write_bytes(original)

    def test_two_comparison_statistics_and_pair_checks(self):
        rows = self.synthetic()
        result = s.confirmation_result(self.args, rows)
        self.assertEqual(result['eligible'], ['S0'])
        self.assertEqual(result['four']['simultaneous_comparisons'], 2)
        self.assertEqual(result['four']['bootstrap_seed'], 47020261023)
        self.assertNotIn('ppo_contribution', result)
        with self.assertRaises(ValueError):
            s.paired_statistics(rows[1:], s.comparison_specs(), draws=10)
        bad = copy.deepcopy(rows); bad[0]['search_seeds'] = [9, 9]
        with self.assertRaises(ValueError):
            s.paired_statistics(bad, s.comparison_specs(), draws=10)

    def test_regression_gates_and_zero_gain_cannot_pass(self):
        result = s.confirmation_result(self.args, self.synthetic(gain=False))
        self.assertFalse(result['eligible'])
        four = dict(delta=.04, own_illegal_delta=0., per_opponent_delta=dict(zip(s.o.OPPONENTS, [-.04, 0., .16])))
        self.assertFalse(all(s.performance_checks(four, dict(delta=0.), gain=True).values()))
        with tempfile.TemporaryDirectory() as path:
            args = s.parse_args(['--smoke', '--output', path])
            s.lock_json(args.output/'pilot_result.json', dict(passed=False))
            with self.assertRaisesRegex(ValueError, 'authorize'):
                s.confirm(args, {}, None, lambda *a, **k: None)

    def test_protocol_source_and_scenario_fingerprints_reject_changes(self):
        for rel in ('initial/source.pt', 'models/current/'+s.o.MODEL_NAMES[2], 'scenarios.json'):
            path = self.args.output/rel
            original = path.read_bytes()
            path.write_bytes(original+b'changed')
            try:
                with self.assertRaises(ValueError):
                    s.verify(self.args)
            finally:
                path.write_bytes(original)

    def test_atomic_update_interruption_and_pending_validation_recover(self):
        with tempfile.TemporaryDirectory(prefix='l45_guarded_resume_') as path:
            args = copy.copy(self.args); args.output = Path(path)/'run'
            shutil.copytree(self.args.output, args.output)
            shutil.rmtree(args.output/'training')
            runtime = s.load_runtime(args, 'S', 0)
            s.lock_json(runtime['directory']/'teachers/batch000000.json',
                s.locked_read(self.runtime['directory']/'teachers/batch000000.json'))
            s.lock_json(runtime['directory']/'batches/000000.json',
                s.locked_read(self.runtime['directory']/'batches/000000.json'))
            original = s.update_policy
            def interrupted(*a, **k):
                original(*a, **k)
                raise RuntimeError('interrupted after in-memory update')
            with mock.patch.object(s, 'update_policy', side_effect=interrupted):
                with self.assertRaises(RuntimeError):
                    s.update_runtime(args, runtime, self.tasks, lambda *a, **k: None)
            recovered = s.load_runtime(args, 'S', 0)
            self.assertEqual(recovered['saved']['episode'], 0)
            s.update_runtime(args, recovered, self.tasks, lambda *a, **k: None)
            committed = s.load_runtime(args, 'S', 0)
            self.assertEqual(committed['saved']['episode'], 64)
            self.assertTrue(committed['saved']['pending_validation'])
            expected = self.fresh()
            metric = self.update(expected)
            self.assertTrue(s.equal_tree(s.o.actor_states(expected['actors']), committed['saved']['actors']))
            self.assertTrue(s.equal_tree(expected['actor_opt'].state_dict(), committed['saved']['actor_optimizer']))
            self.assertEqual(metric['final_exact_kl'], committed['saved']['last_metrics']['final_exact_kl'])
            synthetic = self.synthetic()
            with mock.patch.object(s, 'run_evaluation', return_value=[r for r in synthetic if r['uid'] == 'S0']):
                s.validate_runtime(args, self.banks, committed,
                    [r for r in synthetic if r['uid'] != 'S0'], None, lambda *a, **k: None)
            self.assertFalse(s.load_runtime(args, 'S', 0)['saved']['pending_validation'])

    def test_sampling_and_teacher_interruption_reuses_completed_items(self):
        task = self.tasks[0]
        result = Path(task['result'])
        artifacts = {p: p.read_bytes() for p in (result, result.with_suffix('.json'), result.with_suffix('.lock.json'))}
        for path in artifacts:
            path.unlink()
        try:
            with mock.patch.object(s, 'rollout', side_effect=RuntimeError('sample interrupted')):
                with self.assertRaises(RuntimeError):
                    s.sample_batch(self.args, self.tasks[:2], None, lambda *a, **k: None, 'interrupted')
            s.sample_batch(self.args, self.tasks[:2], None, lambda *a, **k: None, 'resumed')
            self.assertTrue(s.equal_tree(s.read_rollout(task)['rows'], self.episodes[0]['rows']))
        finally:
            for path, data in artifacts.items():
                path.write_bytes(data)
        teaching = self.teaching[0]
        path = Path(teaching['result']); original = path.read_bytes(); path.unlink()
        try:
            with mock.patch.object(s, 'teacher_one', side_effect=RuntimeError('teacher interrupted')):
                with self.assertRaises(RuntimeError):
                    s.teach_batch(self.args, self.teaching, None, lambda *a, **k: None, 'interrupted')
            with mock.patch.object(s, 'teacher_one', wraps=s.teacher_one) as teacher:
                s.teach_batch(self.args, self.teaching, None, lambda *a, **k: None, 'resumed')
                self.assertEqual(teacher.call_count, 1)
        finally:
            path.write_bytes(original)

    def test_controlled_confirm_regression_timing_export_and_fallback(self):
        for passed in (False, True):
            with tempfile.TemporaryDirectory(prefix='l45_guarded_branches_') as path:
                args = copy.copy(self.args); args.output = Path(path)/'run'
                shutil.copytree(self.args.output, args.output)
                shutil.rmtree(args.output/'training')
                runtime = s.load_runtime(args, 'S', 0)
                exported = s.f.export_checkpoint(args, runtime['directory'], 64, runtime['saved']['actors'])
                checkpoint = s.o.read_json(runtime['directory']/'latest.json')['path']
                descriptor = s.model('S0', exported, 0, 64, 'S', checkpoint)
                rows = self.synthetic(gain=passed)
                entry = dict(model=descriptor, episode=64, eligible=True,
                    four=s.aggregate(rows, 'S0'), none=s.aggregate(rows, 'S0', 'none'),
                    comparison=s.point_comparison(rows, 'S0'), none_comparison=s.point_comparison(rows, 'S0', layout='none'),
                    baseline_comparison=s.point_comparison(rows, 'S0', reference=s.BASELINE),
                    baseline_none_comparison=s.point_comparison(rows, 'S0', reference=s.BASELINE, layout='none'))
                selection = dict(records=[entry], best=entry)
                s.o.write_json(runtime['directory']/'selection.json', selection)
                s.o.write_json(runtime['directory']/'progress.json', dict(episode=64))
                gate = dict(passed=passed, smoke=True, selected=[entry], gates=[dict(uid='S0', passed=passed)])
                s.lock_json(args.output/'pilot_result.json', gate)
                quiet = lambda *a, **k: None
                if passed:
                    s.lock_selection(args, gate)
                    with mock.patch.object(s, 'run_evaluation', return_value=rows):
                        confirmed = s.confirm(args, self.banks, None, quiet)
                    self.assertEqual(confirmed['eligible'], ['S0'])
                    with mock.patch.object(s, 'run_evaluation', return_value=self.synthetic(role=0)):
                        regressed = s.guard(args, self.banks, None, quiet)
                    self.assertEqual(regressed['accepted'], ['S0'])
                    serial = [dict(uid=uid, decision_latency_ms=[[1., 1., 2.], [3., 1., 4.]],
                        episode_wall_ms=10., timing_machine={'before': {'pid': 1}, 'after': {'pid': 1}})
                        for uid in (s.CONTROL, s.BASELINE, 'S0')]
                    with mock.patch.object(s, 'run_evaluation', return_value=serial), \
                            mock.patch.object(s, 'evaluate_one', return_value={}):
                        timings = s.timing(args, self.banks, quiet)
                    self.assertEqual(timings['S0']['pursuer_mean_ms'], 2.)
                with mock.patch.object(s, 'replay', return_value=dict(scientific_result_exact_match=True)):
                    final = s.finish(args, self.banks)
                self.assertEqual(final['recommended'], 'S0' if passed else s.CONTROL)
                self.assertFalse(final['quality_gate_passed'])  # Controlled smoke never claims a scientific gain.
                self.assertEqual(s.read_recommended(args)['uid'], final['recommended'])
                self.assertEqual('confirm' in final['actual_stages'], passed)


if __name__ == '__main__':
    unittest.main()
