"""Scientific controls for paired continuation; temporary, independent fixtures."""
import copy
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import run_l45_strong_opponent as s


class StrongOpponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix='l45_strong_tests_')
        cls.args = s.parse_args(['--smoke', '--output', cls.temp.name, '--workers', '2'])
        cls.banks = s.prepare(cls.args)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def late_spec(self):
        spec = copy.deepcopy(self.banks['validation_four'][0])
        p, state = s.o.restore_scenario(spec)
        state[p.time_idx, 0] = p.tf-p.dt
        spec['initial_state'] = state.tolist()
        spec['episode_hash'] = s.o.json_hash({k: v for k, v in spec.items() if k != 'episode_hash'})
        return spec

    def fake_models(self):
        current, baseline = s.controls(self.args)
        return [current, baseline]+[dict(current, uid=f'{a}{n}', label=f'{a}{n}', arm=a, family=a, seed=n)
                                    for a in s.ARMS for n in s.SEEDS]

    def synthetic_rows(self, role=1):
        rows = []
        for m in self.fake_models():
            for layout in ('four', 'none'):
                for index in range(100):
                    cutoff = 80 if layout == 'none' else {s.CONTROL: 50, s.BASELINE: 45, 'A0': 75,
                              'A1': 40, 'A2': 52, 'B0': 85, 'B1': 60, 'B2': 55}[m['uid']]
                    success = int(index < cutoff)
                    for op in s.o.OPPONENTS:
                        rows.append(dict(uid=m['uid'], role=role, layout=layout, opponent=op,
                                         episode_hash=f'{layout}_{index}', search_seeds=[index, index+1],
                                         legal_double_capture=success, legal_survival=success,
                                         pursuer_illegal=0, evader_illegal=0,
                                         one_capture_timeout=1-success, completion_cost=40. if success else 80.))
        return rows

    def test_protocol_matrix_and_fifteen_comparisons(self):
        formal = s.parse_args(['--output', str(self.args.output)])
        scenes = [dict(episode_hash=f's{i}', layout='four' if i < 200 else 'none') for i in range(300)]
        items = s.requests(formal, self.fake_models(), scenes)
        self.assertEqual(len(items), 7200)
        self.assertEqual(len(s.comparison_specs()), 15)
        self.assertEqual(len({x['id'] for x in s.comparison_specs()}), 15)
        self.assertEqual(s.config(formal)['max_training_episodes'], 3072)
        for item in items:
            task = item['task']
            self.assertEqual(task['own_settings']['search_budget'], 1000)
            self.assertEqual(task['own_settings']['beta_value'], 0.)
            self.assertEqual(task['opponent_settings']['beta_value'], .1)
            self.assertEqual(task['opponent_settings']['search_only_budget'], 3000)

    def test_exact_paired_opponent_mix_and_fresh_stream(self):
        for seed in s.SEEDS:
            for completed in range(0, 512, 64):
                arrangement = s.opponent_schedule(self.args, seed, completed)
                for arm, expected in [('A', [32, 32, 0]), ('B', [16, 16, 32])]:
                    self.assertEqual([arrangement[arm].count(op) for op in s.o.OPPONENTS], expected)
                common = arrangement['common_positions']
                self.assertEqual(len(common), 32)
                self.assertTrue(all(arrangement['A'][i] == arrangement['B'][i] for i in common))
                self.assertTrue(all(arrangement['B'][i] == 'Search Only' for i in range(64) if i not in common))
                self.assertEqual(arrangement, s.opponent_schedule(self.args, seed, completed))
        self.assertNotEqual(s.config(self.args)['scenario_namespace'], s.config(s.parse_args([]))['scenario_namespace'])

    def test_complete_initialization_and_adam_preserved_for_all_six_runs(self):
        initial = s.o.load_torch(self.args.output / 'initial/source.pt')
        states = []
        for arm in s.ARMS:
            for seed in s.SEEDS:
                runtime = s.load_runtime(self.args, arm, seed)
                saved = runtime['saved']
                self.assertEqual(saved['episode'], 0)
                self.assertEqual(saved['source_episode'], 2048)
                for key in ('actors', 'critic', 'actor_optimizer', 'critic_optimizer'):
                    self.assertTrue(s.equal_tree(saved[key], initial[key]))
                self.assertTrue(s.equal_tree(runtime['actor_opt'].state_dict(), initial['actor_optimizer']))
                self.assertTrue(s.equal_tree(runtime['critic_opt'].state_dict(), initial['critic_optimizer']))
                states.append(saved['rng'])
        self.assertTrue(s.equal_tree(states[0], states[3]))
        self.assertFalse(s.equal_tree(states[0], states[1]))

    def test_history_contains_recent_smoke_archive_and_actual_rollouts(self):
        files = [Path(p) for p in s.o.read_json(self.args.output / 'experiment.json')['history']]
        used = s.history_scenes(files)
        for name in ('l45_value_off_confirm_20261023', 'l45_value_off_smoke_20261023',
                     'l45_value_off_startup_archive_20261023/formal'):
            banks = s.o.read_json(s.ROOT / 'current' / name / 'scenarios.json')
            self.assertIn(s.previous.physical_hash(next(iter(banks.values()))[0]), used)
        self.assertTrue(any('/rollouts/' in str(p) for p in files))
        identities = [s.previous.physical_hash(x) for group in self.banks.values() for x in group]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertFalse(set(identities).intersection(used))

    def test_actual_cpp_roles_are_independent(self):
        item = s.controls(self.args)[0]
        for role in (0, 1):
            for opponent in s.o.OPPONENTS:
                task = s.requests(self.args, [item], [self.late_spec()], role=role, opponents=(opponent,))[0]['task']
                row = s.evaluate_one(task)
                self.assertAlmostEqual(row['actual_own']['beta_value'], .1 if role == 0 else 0.)
                self.assertEqual(row['actual_own']['depth'], 100)
                if opponent == 'Search + Learning':
                    self.assertAlmostEqual(row['actual_opponent']['beta_value'], .1)
                if opponent == 'Search Only':
                    self.assertEqual(row['actual_opponent']['beta_policy'], 0.)
                    self.assertEqual(row['actual_opponent']['beta_value'], 0.)

    def test_original_export_replay(self):
        result = s.audit(self.args, self.banks)
        self.assertTrue(result['zero_update_export_and_replay_exact_match'])

    def test_alias_cache_resume_and_tampering(self):
        current = s.controls(self.args)[0]
        alias = dict(current, uid='alias', label='alias')
        items = s.requests(self.args, [current, alias], [self.late_spec()], opponents=('Policy Only',))
        self.assertEqual(items[0]['key'], items[1]['key'])
        with mock.patch.object(s, 'evaluate_one', wraps=s.evaluate_one) as evaluator:
            left = s.run_evaluation(self.args, 'fixture_alias', items, None, lambda *a, **k: None)
            right = s.run_evaluation(self.args, 'fixture_alias', items, None, lambda *a, **k: None)
            self.assertEqual(evaluator.call_count, 1)
        self.assertEqual([s.scientific_digest(r) for r in left], [s.scientific_digest(r) for r in right])
        path = self.args.output / 'cache' / (items[0]['key']+'.json')
        original = path.read_bytes()
        try:
            row = s.o.read_json(path)
            row['legal_double_capture'] = 1-row['legal_double_capture']
            s.o.write_json(path, row)
            with self.assertRaisesRegex(ValueError, 'cache content'):
                s.run_evaluation(self.args, 'fixture_alias', items, None, lambda *a, **k: None)
        finally:
            path.write_bytes(original)

    def test_single_checkpoint_statistics_and_full_simultaneous_family(self):
        rows = self.synthetic_rows()
        spec = [dict(id='one', candidate='A0', reference=s.CONTROL)]
        single = s.paired_statistics(rows, spec, draws=2000)
        self.assertAlmostEqual(single['comparisons'][0]['delta'], .25)
        self.assertGreater(single['comparisons'][0]['simultaneous_lower95'], 0.)
        self.assertEqual(single['n_scenes'], 100)
        self.assertEqual(single['comparisons'][0]['n_games'], 300)
        joint = s.paired_statistics(rows, s.comparison_specs(), draws=2000)
        self.assertEqual(joint['simultaneous_comparisons'], 15)
        self.assertGreaterEqual(joint['max_centered_correction'], single['max_centered_correction']-1e-12)
        same = s.paired_statistics(rows, [dict(id='same', candidate=s.CONTROL, reference=s.CONTROL)], draws=100)
        self.assertEqual(same['comparisons'][0]['simultaneous_lower95'], 0.)

    def test_missing_pairs_and_search_seeds_rejected(self):
        rows = self.synthetic_rows()
        spec = [dict(id='one', candidate='A0', reference=s.CONTROL)]
        bad = [r for r in rows if not (r['uid'] == 'A0' and r['episode_hash'] == 'four_0' and r['opponent'] == 'Search Only')]
        with self.assertRaisesRegex(ValueError, 'Missing opponent'):
            s.paired_statistics(bad, spec, draws=100)
        bad = copy.deepcopy(rows)
        next(r for r in bad if r['uid'] == 'A0' and r['layout'] == 'four')['search_seeds'] = [2, 9]
        with self.assertRaisesRegex(ValueError, 'Unpaired search seeds'):
            s.paired_statistics(bad, spec, draws=100)

    def test_adoption_is_individual_and_B_over_A_is_not_extra_gate(self):
        result = s.confirmation_result(self.args, self.synthetic_rows())
        self.assertIn('A0', result['eligible'])
        self.assertNotIn('A1', result['eligible'])
        self.assertEqual(len(result['candidates']), 6)
        self.assertEqual(len(result['strong_opponent_contribution']), 3)
        a = next(x for x in result['candidates'] if x['uid'] == 'A0')
        self.assertNotIn('strong_opponent_contribution', a['checks'])

    def test_selection_regressions_and_ties(self):
        four = dict(delta=.04, own_illegal_delta=0., per_opponent_delta=dict(zip(s.o.OPPONENTS, [0., 0., .12])))
        none = dict(delta=-.02)
        self.assertTrue(all(s.performance_checks(four, none, gain=True).values()))
        four['per_opponent_delta']['Policy Only'] = -.04
        self.assertFalse(all(s.performance_checks(four, none, gain=True).values()))
        models = [dict(uid='A0', arm='A', seed=0, episode=256), dict(uid='B0', arm='B', seed=0, episode=256)]
        candidates = [dict(uid=m['uid'], four=dict(success_rate=.9, own_illegal_rate=0., completion_cost=50.)) for m in models]
        timings = {m['uid']: dict(pursuer_mean_ms=100.) for m in models}
        self.assertLess(s.selection_key(candidates[0], timings, models), s.selection_key(candidates[1], timings, models))
        candidates[1]['four']['success_rate'] = .91
        timings['B0']['pursuer_mean_ms'] = 900.
        self.assertLess(s.selection_key(candidates[1], timings, models), s.selection_key(candidates[0], timings, models))

    def test_protocol_model_and_scene_tampering_rejected(self):
        for rel in ('initial/source.pt', 'models/current/'+s.o.MODEL_NAMES[2], 'scenarios.json'):
            path = self.args.output / rel
            original = path.read_bytes()
            try:
                path.write_bytes(original+b' ' if rel.endswith('.json') else original+b'changed')
                if rel.endswith('.json'):
                    data = s.o.read_json(path)
                    data['validation_four'][0]['search_seeds'][0] += 1
                    s.o.write_json(path, data)
                with self.assertRaises(ValueError):
                    s.verify(self.args)
            finally:
                path.write_bytes(original)

    def test_locked_gates_and_projections(self):
        with tempfile.TemporaryDirectory(prefix='l45_strong_gate_') as out:
            args = s.parse_args(['--smoke', '--output', out])
            s.lock_json(args.output / 'pilot_result.json', dict(passed=False, expansion_authorized=False))
            with self.assertRaisesRegex(ValueError, 'did not authorize'):
                s.expand(args, {}, None, lambda *a, **k: None)
            with self.assertRaises(FileNotFoundError):
                s.selected_models(args)
            path = args.output / 'atomic.json'
            s.lock_json(path, dict(value=1))
            path.unlink()
            self.assertEqual(s.locked_read(path), dict(value=1))
            s.o.write_json(path, dict(value=2))
            with self.assertRaisesRegex(ValueError, 'projection changed'):
                s.locked_read(path)

    def test_snapshot_and_rollout_identity_resume_across_worker_counts(self):
        runtime = s.load_runtime(self.args, 'B', 2)
        banks = dict(self.banks, train2=[self.late_spec() for _ in range(64)])
        tasks = s.sampling_tasks(self.args, banks, runtime)
        task = tasks[0]
        serial = s.rollout(task)
        for workers in (1, 2):
            with mp.get_context('spawn').Pool(workers) as pool:
                actual = pool.map(s.rollout, [task])[0]
            self.assertTrue(s.equal_tree(serial, actual))
        s.sample_batch(self.args, [task], None, lambda *a, **k: None, 'fixture_sample')
        with mock.patch.object(s, 'rollout', side_effect=AssertionError('must reuse cache')):
            s.sample_batch(self.args, [task], None, lambda *a, **k: None, 'fixture_sample')
        bad = copy.deepcopy(serial)
        bad['arm'] = 'A'
        with self.assertRaisesRegex(ValueError, 'Cross-group'):
            s.validate_rollout(task, bad)
        path = Path(task['result'])
        original = path.read_bytes()
        try:
            path.write_bytes(original+b'changed')
            with self.assertRaisesRegex(ValueError, 'cache content'):
                s.read_rollout(task)
        finally:
            path.write_bytes(original)
        snapshot = Path(task['snapshot'])
        original = snapshot.read_bytes()
        try:
            snapshot.write_bytes(original+b'changed')
            with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                s.sampling_tasks(self.args, banks, runtime)
        finally:
            snapshot.write_bytes(original)

    def test_interrupted_update_and_pending_validation_restore(self):
        runtime = s.load_runtime(self.args, 'A', 2)
        baseline = s.baseline_validation(self.args, self.banks, None, lambda *a, **k: None)
        s.validate_runtime(self.args, self.banks, runtime, baseline, None, lambda *a, **k: None)
        banks = dict(self.banks, train2=[self.late_spec() for _ in range(64)])
        tasks = s.sampling_tasks(self.args, banks, runtime)
        s.sample_batch(self.args, tasks, None, lambda *a, **k: None, 'fixture_update')
        original_commit = s.f.commit_checkpoint
        def interrupted_commit(*args, **kwargs):
            if args[1]['episode'] == 64:
                raise RuntimeError('controlled interruption after update')
            return original_commit(*args, **kwargs)
        with mock.patch.object(s.f, 'commit_checkpoint', side_effect=interrupted_commit):
            with self.assertRaisesRegex(RuntimeError, 'controlled interruption'):
                s.update_runtime(self.args, runtime, tasks, lambda *a, **k: None)
        expected = s.o.actor_states(runtime['actors'])
        restored = s.load_runtime(self.args, 'A', 2)
        self.assertEqual(restored['saved']['episode'], 0)
        s.update_runtime(self.args, restored, tasks, lambda *a, **k: None)
        self.assertTrue(s.equal_tree(expected, restored['saved']['actors']))
        committed = s.load_runtime(self.args, 'A', 2)
        self.assertEqual(committed['saved']['episode'], 64)
        self.assertTrue(committed['saved']['pending_validation'])
        self.assertTrue(s.equal_tree(committed['actor_opt'].state_dict(), restored['actor_opt'].state_dict()))
        self.assertTrue(s.equal_tree(committed['critic_opt'].state_dict(), restored['critic_opt'].state_dict()))
        s.assert_frozen(self.args, committed['saved']['actors'])
        with mock.patch.object(s, 'run_evaluation', side_effect=RuntimeError('controlled validation interruption')):
            with self.assertRaisesRegex(RuntimeError, 'controlled validation interruption'):
                s.validate_runtime(self.args, self.banks, committed, baseline, None, lambda *a, **k: None)
        self.assertTrue(s.load_runtime(self.args, 'A', 2)['saved']['pending_validation'])
        s.validate_runtime(self.args, self.banks, committed, baseline, None, lambda *a, **k: None)
        self.assertFalse(s.load_runtime(self.args, 'A', 2)['saved']['pending_validation'])

    def test_active_mask_and_exact_logprob(self):
        p, state = s.o.restore_scenario(self.late_spec())
        actors = s.o.load_actors(p, directory=self.args.output / 'models/current')
        obs = torch.tensor(np.array([[p.policy_encoding(state, r).reshape(-1) for r in s.o.TEAMS[1]]]), dtype=torch.float32)
        lim = torch.tensor(np.array([[p.action_lims[p.action_idxs[r]] for r in s.o.TEAMS[1]]]), dtype=torch.float32)
        raw = torch.tensor([[[10., -10.], [20., -20.]]])
        mask = torch.tensor([[1., 0.]])
        logp = s.o.joint_log_prob(actors, obs, raw, lim, mask)
        raw[:, 1] = 10000.
        self.assertTrue(torch.equal(logp, s.o.joint_log_prob(actors, obs, raw, lim, mask)))

    def test_guard_configuration_and_both_layouts(self):
        models = self.fake_models()
        with tempfile.TemporaryDirectory(prefix='l45_strong_guard_') as out:
            args = s.parse_args(['--smoke', '--output', out])
            s.lock_json(args.output / 'confirmation_result.json', dict(eligible=['A0', 'A1']))
            rows = self.synthetic_rows(role=0)
            with mock.patch.object(s, 'selected_models', return_value=models), \
                 mock.patch.object(s, 'run_evaluation', return_value=rows), \
                 mock.patch.object(s, 'requests', return_value=[]):
                result = s.guard(args, self.banks, None, lambda *a, **k: None)
            self.assertIn('A0', result['accepted'])
            self.assertNotIn('A1', result['accepted'])
            self.assertEqual(result['evader_settings']['beta_value'], .1)
            self.assertEqual(set(result['candidates'][0]['comparisons']), {'four', 'none'})

    def test_protected_paths_and_dashboard(self):
        with self.assertRaises(SystemExit):
            s.parse_args(['--output', str(s.SOURCE / 'nested')])
        with self.assertRaises(SystemExit):
            s.parse_args(['--smoke'])
        with mock.patch.object(s, 'live_pids', return_value=[]):
            self.assertIn('未运行', s.dashboard(self.args))

    def test_controlled_confirmation_guard_timing_export_and_fallback(self):
        for branch in ('accept', 'fallback'):
            with tempfile.TemporaryDirectory(prefix=f'l45_strong_{branch}_') as out:
                args = s.parse_args(['--smoke', '--output', out])
                banks = s.prepare(args)
                current, baseline = s.controls(args)
                models = [dict(current, uid=f'{a}{n}', label=f'{a}{n}', arm=a, family=a, seed=n)
                          for a in s.ARMS for n in s.SEEDS]
                entries = [dict(model=m, episode=0, eligible=True, four=dict(success_rate=.8),
                                comparison=dict(delta=0., per_opponent_delta={'Search Only': 0.}),
                                none_comparison=dict(delta=0.)) for m in models]
                # Explicit controlled gates exist only inside this temporary test.
                gate = dict(passed=branch == 'accept', expansion_authorized=branch == 'accept',
                            selected=[entries[0], entries[3]],
                            gates=[dict(uid='A0', passed=False), dict(uid='B0', passed=False)])
                s.lock_json(args.output / 'pilot_result.json', gate)
                if branch == 'accept':
                    for entry in entries:
                        m = entry['model']
                        s.o.write_json(s.directory(args, m['arm'], m['seed']) / 'selection.json', dict(best=entry, records=[entry]))
                    s.lock_json(args.output / 'checkpoint_selection.json',
                                dict(models=models, controls=[current, baseline], selected=entries,
                                     experiment_sha256=s.o.sha(args.output / 'experiment.json')))
                    actual_items = s.requests(args, [current, baseline]+models, banks['confirm_four']+banks['confirm_none'])
                    s.run_evaluation(args, 'fixture_actual_search', actual_items, None, lambda *a, **k: None)
                    with mock.patch.object(s, 'run_evaluation', return_value=self.synthetic_rows()):
                        pursuit = s.confirm(args, banks, None, lambda *a, **k: None)
                    self.assertIn('A0', pursuit['eligible'])
                    # All eligible fixed-checkpoint aliases must enter real regression.
                    regression = s.guard(args, banks, None, lambda *a, **k: None)
                    self.assertEqual(regression['accepted'], pursuit['eligible'])
                    measured = s.timing(args, banks, lambda *a, **k: None)
                    self.assertEqual(set(measured), set(pursuit['eligible']) | {s.CONTROL})
                    self.assertGreater(measured['A0']['decisions'], 0)
                else:
                    s.baseline_validation(args, banks, None, lambda *a, **k: None)
                result = s.finish(args, banks)
                self.assertFalse(result['quality_gate_passed'])  # Smoke never makes scientific claims.
                self.assertEqual(result['recommended'], 'B0' if branch == 'accept' else s.CONTROL)
                self.assertTrue(result['replay']['scientific_result_exact_match'])
                rec = s.read_recommended(args)
                p, _ = s.o.restore_scenario(banks['validation_four'][0])
                solver = s.load_recommended_solver(args, p, 0, 123)
                self.assertAlmostEqual(solver.actual_settings['beta_value'], .1)
                self.assertEqual(s.o.model_fingerprint(rec['models']), rec['model_hash'])
                frozen = (args.output / 'results.json').read_bytes()
                s.finish(args, banks)
                self.assertEqual(frozen, (args.output / 'results.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
