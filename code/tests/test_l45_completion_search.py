"""Completion utility isolation, exact parity, inference and recovery checks."""
import copy
import multiprocessing as mp
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_l45_completion_search as c


class CompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='l45_completion_tests_')
        cls.args = c.parse_args(['--smoke', '--output', cls.temp.name, '--workers', '2'])
        cls.banks = c.prepare(cls.args)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_protocol_fixed_no_training_and_five_comparisons(self):
        args = c.parse_args([])
        cfg = c.config(args)
        self.assertTrue(cfg['no_training'])
        self.assertEqual(cfg['budget'], 1000)
        self.assertEqual(cfg['search_only_budget'], 3000)
        self.assertEqual(len(c.comparisons()), 5)
        self.assertEqual(sum(n for name, (_, n) in cfg['splits'].items() if name.startswith('screen')), 90)
        self.assertEqual(c.settings(args, 0, 1.)['completion_bonus'], 0.)
        self.assertEqual(c.settings(args)['beta_value'], 0.)
        self.assertEqual(c.settings(args, 0)['beta_value'], .1)

    def test_physical_scenes_new_and_disjoint_with_unused_banks_excluded(self):
        identities = [c.s.previous.physical_hash(s) for bank in self.banks.values() for s in bank]
        history = c.locked_read(self.args.output/'experiment.json')['history']
        self.assertEqual(len(identities), len(set(identities)))
        self.assertFalse(set(identities) & c.s.history_scenes([Path(p) for p in history]))
        self.assertTrue(any('l45_search_feedback_20261023/scenarios.json' in p for p in history))

    def solver(self, layout='none', bonus=1., role=1):
        spec = self.banks['audit_'+layout][0]
        p, state = c.o.restore_scenario(spec)
        m = c.models(self.args)[0]
        solver = c.make_own(p, m['models'], role, c.settings(self.args, role, bonus), spec['search_seeds'][role])
        return p, state, solver.solver

    def test_bonus_legal_completion_once_both_reward_paths_evaders_unchanged(self):
        p, state, solver = self.solver()
        wrapper = solver._problem_wrapper_for(p)
        action = np.zeros((p.action_dim, 1))
        next_state = state.copy()
        next_state[p.active_idxs[:2], 0] = 0.
        next_state[p.time_idx, 0] += p.dt
        original = wrapper.transition_reward(state, action, next_state)
        for rollout in (False, True):
            utility = solver.solver_wrapper.transition_utility(wrapper, state, action, next_state, rollout)
            np.testing.assert_allclose(utility-original, [0., 0., 1., 1.], atol=1e-7)
            repeated = solver.solver_wrapper.transition_utility(wrapper, next_state, action, next_state, rollout)
            repeated_base = wrapper.transition_reward(next_state, action, next_state)
            np.testing.assert_array_equal(repeated, repeated_base)

    def test_no_bonus_first_capture_illegal_timeout_or_terminal_root(self):
        p, state, solver = self.solver()
        wrapper = solver._problem_wrapper_for(p)
        action = np.zeros((p.action_dim, 1))
        cases = []
        first = state.copy(); first[p.active_idxs[0], 0] = 0.
        cases.append((state, first))
        illegal = state.copy(); illegal[p.active_idxs[:2], 0] = 0.
        illegal[p.state_idxs[2][0], 0] = 1000.
        cases.append((state, illegal))
        timeout = state.copy(); timeout[p.time_idx, 0] = p.tf
        cases.append((state, timeout))
        completed = state.copy(); completed[p.active_idxs[:2], 0] = 0.
        cases.append((completed, completed))
        ended = state.copy(); ended[p.time_idx, 0] = p.tf
        cases.append((ended, completed))
        for before, after in cases:
            for rollout in (False, True):
                np.testing.assert_array_equal(
                    solver.solver_wrapper.transition_utility(wrapper, before, action, after, rollout),
                    wrapper.transition_reward(before, action, after))

    def test_bonus_changes_only_solver_instance(self):
        p, state, positive = self.solver()
        _, _, zero = self.solver(bonus=0.)
        self.assertEqual(positive.solver_wrapper.completion_bonus, 1.)
        self.assertEqual(zero.solver_wrapper.completion_bonus, 0.)
        other = c.f.make_solver(p, c.models(self.args)[1]['models'], 'Search Only', 0,
                               c.settings(self.args, 0), 42)
        self.assertFalse(hasattr(other.solver.solver_settings, 'completion_bonus'))
        self.assertFalse(hasattr(p, 'completion_bonus'))

    def test_tree_real_search_counts_and_budget(self):
        p, state, solver = self.solver()
        for robot, position in enumerate(([-4., 0.], [4., 0.], [-4.1, 0.], [4.1, 0.])):
            state[p.state_idxs[robot], 0] = position
        result = solver.search(p, state, turn=2)
        self.assertEqual(result.num_visits, c.config(self.args)['budget'])
        self.assertGreater(result.completion_tree_events+result.completion_rollout_events, 0)
        self.assertEqual(len(solver.diagnostics), 1)

    def test_zero_bonus_full_parity_audit(self):
        result = c.audit(self.args, self.banks)
        self.assertTrue(result['passed'])
        self.assertEqual(sum('exact_episode' in x for x in result['zero_bonus_parity']), 6)
        self.assertEqual(sum('exact_action' in x for x in result['zero_bonus_parity']), 8)

    def late_items(self):
        spec = copy.deepcopy(self.banks['audit_none'][0])
        p, state = c.o.restore_scenario(spec)
        state[p.time_idx, 0] = p.tf-p.dt
        spec['initial_state'] = state.tolist()
        spec['episode_hash'] = c.o.json_hash({k: v for k, v in spec.items() if k != 'episode_hash'})
        return c.requests(self.args, c.models(self.args), [spec])

    def test_cache_identity_masks_actual_utility_and_tampering(self):
        items = self.late_items()
        self.assertEqual(len({i['key'] for i in items}), 12)
        rows = c.run_evaluation(self.args, 'test_cache', items, None, lambda *a, **k: None)
        for row in rows:
            self.assertEqual(row['actual_own']['completion_bonus'], float(row['uid'].endswith('1')))
            self.assertEqual(row['actual_opponent']['completion_bonus'], 0.)
        target = self.args.output/'cache'/(items[0]['key']+'.json')
        original = target.read_text()
        broken = c.o.read_json(target); broken['legal_double_capture'] = 9
        c.o.write_json(target, broken)
        try:
            with self.assertRaises(ValueError):
                c.run_evaluation(self.args, 'test_cache', items, None, lambda *a, **k: None)
        finally:
            target.write_text(original)

    def test_parallel_worker_change_and_cached_resume(self):
        items = self.late_items()[:3]
        serial = [c.evaluate_one(i['task']) for i in items]
        with mp.get_context('spawn').Pool(2) as pool:
            parallel = pool.map(c.evaluate_one, [i['task'] for i in items])
        for a, b in zip(serial, parallel):
            self.assertEqual(c.v.scientific_digest(a), c.v.scientific_digest(b))
        c.run_evaluation(self.args, 'test_resume', items, None, lambda *a, **k: None)
        with mock.patch.object(c, 'evaluate_one', side_effect=AssertionError('must use committed cache')):
            c.run_evaluation(self.args, 'test_resume', items, None, lambda *a, **k: None)

    def synthetic(self, ids=('R0', 'L0', 'R1', 'L1'), role=1):
        rows = []
        for uid in ids:
            for layout in ('four', 'none'):
                for i in range(100):
                    success = int(i < (50 if uid in c.REFERENCES else 80))
                    for op in c.o.OPPONENTS:
                        rows.append(dict(uid=uid, role=role, layout=layout, opponent=op, episode_hash=f'{layout}{i}',
                            search_seeds=[i, i+1], legal_double_capture=success, legal_survival=success,
                            pursuer_illegal=0, evader_illegal=0, completion_cost=40 if success else 80,
                            one_capture_timeout=1-success))
        return rows

    def test_full_and_partial_statistics_and_missing_pairs(self):
        rows = self.synthetic()
        result = c.statistics(rows, ['R1', 'L1'], draws=2000)
        self.assertEqual(result['observed_comparison_count'], 5)
        self.assertEqual(result['correction'], 'max_centered_joint_95')
        self.assertEqual(result['bootstrap_seed'], 46020261023)
        self.assertGreater(result['comparisons'][0]['formal_lower95'], 0.)
        partial = c.statistics([r for r in rows if r['uid'] != 'L1'], ['R1'], draws=2000)
        self.assertEqual(partial['observed_comparison_count'], 2)
        self.assertEqual(partial['preregistered_comparison_count'], 5)
        self.assertEqual(partial['correction'], 'fixed_five_bonferroni_99')
        with self.assertRaises(ValueError):
            c.statistics(rows[1:], ['R1', 'L1'], draws=20)

    def test_both_reference_gates_and_unrun_stages_rejected(self):
        rows = self.synthetic()
        for candidate in c.CANDIDATES:
            for reference in c.REFERENCES:
                self.assertTrue(c.checks(rows, candidate, reference)['passed'])
                self.assertFalse(c.checks(rows, candidate, reference, {'formal_lower95': 0.})['passed'])
        with self.assertRaises(FileNotFoundError):
            c.confirm(self.args, self.banks, None, lambda *a, **k: None)

    def test_tampered_signature_and_changed_protocol_rejected(self):
        altered = copy.copy(self.args); altered.smoke = False
        with self.assertRaises(ValueError):
            c.verify(altered)
        path = self.args.output/'scenarios.json'
        original = path.read_text()
        path.write_text('{}')
        try:
            with self.assertRaises(ValueError):
                c.verify(self.args)
        finally:
            path.write_text(original)

    def sandbox(self, path):
        args = copy.copy(self.args)
        args.output = Path(path)
        shutil.copytree(self.args.output/'models', args.output/'models')
        for name in ('experiment', 'scenarios', 'signature'):
            for suffix in ('.json', '.lock.json'):
                shutil.copyfile(self.args.output/(name+suffix), args.output/(name+suffix))
        return args

    def report_row(self, args, item):
        spec, uid = item['task']['spec'], item['uid']
        return dict(uid=uid, role=1, layout=spec['layout'], opponent=item['task']['opponent'],
                    episode_hash=spec['episode_hash'], search_seeds=spec['search_seeds'],
                    legal_double_capture=1, legal_survival=0, pursuer_illegal=0, evader_illegal=0,
                    completion_cost=40., one_capture_timeout=0)

    def test_confirm_guard_timing_export_full_controlled_branch(self):
        with tempfile.TemporaryDirectory(prefix='l45_completion_branch_') as path:
            args = self.sandbox(path)
            c.lock_json(args.output/'screen_result.json', dict(eligible=list(c.CANDIDATES),
                selected_models=[m for m in c.models(args) if m['uid'] in c.CANDIDATES]))
            quiet = lambda *a, **k: None
            with mock.patch.object(c, 'run_evaluation', return_value=self.synthetic()):
                confirmed = c.confirm(args, self.banks, None, quiet)
            self.assertEqual(confirmed['eligible'], list(c.CANDIDATES))
            with mock.patch.object(c, 'run_evaluation', return_value=self.synthetic(role=0)):
                guarded = c.guard(args, self.banks, None, quiet)
            self.assertEqual(guarded['accepted'], list(c.CANDIDATES))
            timing_rows = [dict(uid=uid, decision_latency_ms=[[1., 1., 2.], [3., 1., 4.]],
                                episode_wall_ms=10., timing_machine={'before': {'pid': 1}, 'after': {'pid': 1}})
                           for uid in (*c.REFERENCES, *c.CANDIDATES)]
            with mock.patch.object(c, 'run_evaluation', return_value=timing_rows), \
                    mock.patch.object(c, 'evaluate_one', return_value={}):
                timed = c.timing(args, self.banks, quiet)
            self.assertEqual(timed['R1']['pursuer_mean_ms'], 2.)
            with mock.patch.object(c, 'read_cached', side_effect=self.report_row):
                result = c.finish(args, self.banks)
            self.assertEqual(result['reason'], 'adopted')
            self.assertIn(result['recommended'], c.CANDIDATES)
            self.assertTrue(result['replay']['passed'])
            self.assertFalse(result['quality_gate_passed'])  # Smoke cannot promote a scientific result.
            self.assertEqual(c.read_recommended(args)['pursuer_settings']['completion_bonus'], 1.)

    def test_no_gain_fallback_exports_original_replays_and_skips_confirmation(self):
        with tempfile.TemporaryDirectory(prefix='l45_completion_fallback_') as path:
            args = self.sandbox(path)
            rows = self.synthetic()
            for row in rows:
                if row['uid'] in c.CANDIDATES:
                    i = int(row['episode_hash'].replace(row['layout'], ''))
                    row['legal_double_capture'] = int(i < 50)
            with mock.patch.object(c, 'run_evaluation', return_value=rows):
                gate = c.screen(args, self.banks, None, lambda *a, **k: None)
            self.assertFalse(gate['eligible'])
            with mock.patch.object(c, 'read_cached', side_effect=self.report_row):
                result = c.finish(args, self.banks)
            self.assertEqual(result['recommended'], 'R0')
            self.assertEqual(result['reason'], 'screen_no_gain')
            self.assertIsNone(result['confirmation'])
            self.assertEqual(c.read_recommended(args)['pursuer_settings']['completion_bonus'], 0.)
            self.assertTrue(result['replay']['passed'])


if __name__ == '__main__':
    unittest.main()
