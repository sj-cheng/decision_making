"""Meaningful controls for frozen model confirmation; all outputs are temporary."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import confirm_l45_value_off as s


class ValueOffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix='l45_value_off_tests_')
        cls.args = s.parse_args(['--smoke', '--output', cls.temp.name, '--workers', '2'])
        cls.banks = s.prepare(cls.args)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def late_spec(self):
        spec = copy.deepcopy(self.banks['confirm_four'][0])
        p, state = s.o.restore_scenario(spec)
        state[p.time_idx, 0] = p.tf-p.dt
        spec['initial_state'] = state.tolist()
        spec['episode_hash'] = s.o.json_hash({k: v for k, v in spec.items() if k != 'episode_hash'})
        return spec

    def synthetic_rows(self, n=100):
        rows = []
        for model in s.models(self.args):
            for layout in ('four', 'none'):
                for scene in range(n):
                    if layout == 'none':
                        cutoff = 90
                    elif model['label'] == 'baseline':
                        cutoff = 40 if model['config'] == 'base' else 70
                    elif model['label'] == 'A0':
                        cutoff = 85
                    elif model['label'] == 'A1':
                        cutoff = 35
                    else:
                        cutoff = 71
                    success = int(scene < cutoff)
                    for opponent in s.o.OPPONENTS:
                        rows.append(dict(uid=model['uid'], label=model['label'], config=model['config'],
                                         role=1, layout=layout, opponent=opponent,
                                         episode_hash=f'{layout}_{scene}', search_seeds=[scene, scene+1],
                                         legal_double_capture=success, legal_survival=1-success,
                                         pursuer_illegal=0, evader_illegal=0, one_capture_timeout=1-success,
                                         completion_cost=40. if success else 80.))
        return rows

    def test_formal_matrix_and_nineteen_comparisons(self):
        args = s.parse_args(['--output', str(self.args.output)])
        scenes = [dict(episode_hash=f'case{i}', layout='four' if i < 200 else 'none') for i in range(300)]
        items = s.requests(args, s.models(self.args), scenes)
        self.assertEqual(len(items), 12600)
        self.assertEqual(len({item['key'] for item in items}), 12600)
        self.assertEqual(len(s.comparison_specs()), 19)
        self.assertEqual(len({item['id'] for item in s.comparison_specs()}), 19)
        for item in items:
            task = item['task']
            self.assertEqual(task['own_settings']['search_budget'], 1000)
            self.assertEqual(task['own_settings']['search_depth'], 100)
            self.assertEqual(task['opponent_settings']['beta_value'], .1)
            self.assertEqual(task['opponent_settings']['search_only_budget'], 3000)

    def test_actual_cpp_parameters_and_evader_role_stays_original(self):
        for cfg in s.CONFIGS:
            model = next(m for m in s.models(self.args) if m['uid'] == f'old0:{cfg}')
            request = s.requests(self.args, [model], [self.late_spec()], opponents=('Search + Learning',))[0]
            result = s.evaluate_one(request['task'])
            self.assertAlmostEqual(result['actual_own']['beta_value'], .1 if cfg == 'base' else 0.)
            self.assertAlmostEqual(result['actual_opponent']['beta_value'], .1)
            self.assertEqual(result['actual_own']['depth'], 100)
        model = next(m for m in s.models(self.args) if m['uid'] == 'old0:novalue')
        request = s.requests(self.args, [model], [self.late_spec()], role=0, opponents=('Search Only',))[0]
        result = s.evaluate_one(request['task'])
        self.assertAlmostEqual(result['actual_own']['beta_value'], .1)
        self.assertEqual(result['actual_opponent']['beta_policy'], 0.)
        self.assertEqual(result['actual_opponent']['beta_value'], 0.)

    def test_original_search_reproduction_and_frozen_model_sources(self):
        result = s.audit(self.args, self.banks)
        self.assertTrue(result['original_configuration_exact_match'])
        registry = s.o.read_json(self.args.output / 'experiment.json')['registry']
        self.assertEqual(len(registry), 7)
        baseline = registry[0]['model_hash']
        for model in registry:
            for name in (*s.o.MODEL_NAMES[:2], s.o.MODEL_NAMES[-1]):
                self.assertEqual(model['model_hash'][name], baseline[name])
            if model['family'] == 'A':
                self.assertEqual(model['episode'], 512)

    def test_latest_followup_and_all_new_splits_are_excluded(self):
        identities = [s.previous.physical_hash(x) for group in self.banks.values() for x in group]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertFalse(set(identities).intersection(s.history_scenes()))
        recent = s.o.read_json(s.SOURCE / 'scenarios.json')
        self.assertTrue(all(s.previous.physical_hash(x) in s.history_scenes()
                            for x in recent['train0'][:2]))

    def test_smoke_and_formal_use_disjoint_scenario_namespaces(self):
        formal = s.parse_args(['--output', str(self.args.output)])
        self.assertNotEqual(s.config(formal)['scenario_namespace'], s.config(self.args)['scenario_namespace'])
        smoke = s.o.make_scenario(s.o.scenario_seed(750, s.config(self.args)['scenario_namespace'], 0))
        confirm = s.o.make_scenario(s.o.scenario_seed(750, s.config(formal)['scenario_namespace'], 0))
        self.assertNotEqual(s.previous.physical_hash(smoke), s.previous.physical_hash(confirm))
        for path in s.additional_history_files():
            first = next(iter(s.o.read_json(path).values()))[0]
            self.assertIn(s.previous.physical_hash(first), s.history_scenes())

    def test_single_checkpoint_bootstrap_is_not_three_seed_indexing(self):
        spec = [dict(id='value', candidate='baseline:novalue', reference='baseline:base')]
        result = s.paired_statistics(self.synthetic_rows(), spec, draws=2000)
        self.assertEqual(result['n_scenes'], 100)
        self.assertAlmostEqual(result['comparisons'][0]['delta'], .3)
        self.assertGreater(result['comparisons'][0]['simultaneous_lower95'], 0.)

    def test_clustered_opponents_and_zero_difference_do_not_create_evidence(self):
        rows = self.synthetic_rows()
        spec = [dict(id='same', candidate='baseline:base', reference='baseline:base')]
        result = s.paired_statistics(rows, spec, draws=2000)
        self.assertEqual(result['comparisons'][0]['simultaneous_lower95'], 0.)
        self.assertEqual(result['comparisons'][0]['ci95'], [0., 0.])
        # Each physical scene contributes one cluster despite having three opponents.
        self.assertEqual(result['n_scenes'], 100)
        self.assertEqual(result['comparisons'][0]['n_games'], 300)

    def test_simultaneous_correction_cannot_improve_individual_one_sided_evidence(self):
        rows = self.synthetic_rows()
        specs = s.comparison_specs()
        joint = s.paired_statistics(rows, specs, draws=2000)
        for spec in specs:
            single = s.paired_statistics(rows, [spec], draws=2000)
            self.assertGreaterEqual(joint['max_centered_correction']+1e-12, single['max_centered_correction'])
        self.assertEqual(joint['simultaneous_comparisons'], 19)

    def test_missing_opponent_and_mismatched_search_seed_refuse_pairing(self):
        rows = self.synthetic_rows()
        spec = [dict(id='value', candidate='baseline:novalue', reference='baseline:base')]
        bad = [r for r in rows if not (r['uid'] == 'baseline:novalue' and r['layout'] == 'four'
                                      and r['episode_hash'] == 'four_0' and r['opponent'] == 'Search Only')]
        with self.assertRaisesRegex(ValueError, 'Missing opponent'):
            s.paired_statistics(bad, spec, draws=100)
        bad = copy.deepcopy(rows)
        next(r for r in bad if r['uid'] == 'baseline:novalue' and r['layout'] == 'four')['search_seeds'] = [91, 92]
        with self.assertRaisesRegex(ValueError, 'Unpaired search seeds'):
            s.paired_statistics(bad, spec, draws=100)

    def test_individual_checkpoint_can_pass_when_another_seed_does_not(self):
        result = s.confirmation_result(self.args, self.synthetic_rows())
        self.assertIn('A0:novalue', result['eligible'])
        self.assertNotIn('A1:novalue', result['eligible'])
        self.assertEqual(len(result['families']), 8)
        a = next(f for f in result['families'] if f['family'] == 'A'
                 and f['config'] == 'novalue' and f['layout'] == 'four')
        self.assertFalse(a['all_seeds_nonnegative'])

    def test_partial_cache_resume_aliases_and_tampering(self):
        models = s.models(self.args)
        base = next(m for m in models if m['uid'] == 'baseline:base')
        alias = dict(base, uid='alias:base', label='alias')
        first = s.requests(self.args, [base, alias], [self.late_spec()], opponents=('Policy Only',))
        self.assertEqual(first[0]['key'], first[1]['key'])
        row = s.evaluate_one(first[0]['task'])
        row['content_sha256'] = s.o.json_hash(row)
        path = self.args.output / 'cache' / (row['key']+'.json')
        s.o.write_json(path, row)
        original = path.read_bytes()
        other = next(m for m in models if m['uid'] == 'old1:novalue')
        items = first+s.requests(self.args, [other], [self.late_spec()], opponents=('Policy Only',))
        with mock.patch.object(s, 'evaluate_one', wraps=s.evaluate_one) as evaluate:
            left = s.run_evaluation(self.args, 'partial_test', items, None, lambda *a, **kw: None)
            self.assertEqual(evaluate.call_count, 1)
            right = s.run_evaluation(self.args, 'partial_test', items, None, lambda *a, **kw: None)
            self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual([s.scientific_digest(r) for r in left], [s.scientific_digest(r) for r in right])
        changed = json.loads(original)
        changed['legal_double_capture'] = 1-changed['legal_double_capture']
        s.o.write_json(path, changed)
        with self.assertRaisesRegex(ValueError, 'cache content'):
            s.read_cached(self.args, first[0])
        path.write_bytes(original)

    def test_guard_aliases_share_actual_original_evader_configuration(self):
        base = next(m for m in s.models(self.args) if m['uid'] == 'A0:base')
        off = next(m for m in s.models(self.args) if m['uid'] == 'A0:novalue')
        items = s.requests(self.args, [base, off], [self.late_spec()], role=0, opponents=('Policy Only',))
        self.assertEqual(items[0]['key'], items[1]['key'])
        self.assertAlmostEqual(items[1]['task']['own_settings']['beta_value'], .1)

    def test_selection_order_and_latency_is_only_a_tie_breaker(self):
        def candidate(uid, success, cost=50.):
            label, cfg = uid.split(':')
            return dict(uid=uid, label=label, config=cfg,
                        four=dict(success_rate=success, own_illegal_rate=0., completion_cost=cost))
        first = candidate('A0:novalue', .8)
        second = candidate('baseline:novalue', .7)
        timings = {first['uid']: dict(pursuer_mean_ms=900.), second['uid']: dict(pursuer_mean_ms=10.)}
        self.assertLess(s.selection_key(first, timings), s.selection_key(second, timings))

    def test_protected_sources_cannot_be_output(self):
        with self.assertRaises(SystemExit):
            s.parse_args(['--output', str(s.SOURCE / 'nested')])
        with self.assertRaises(SystemExit):
            s.parse_args(['--smoke'])

    def test_atomic_artifact_recovers_missing_projection_and_rejects_mutation(self):
        path = self.args.output / 'atomic_fixture.json'
        s.lock_json(path, dict(accepted=['baseline:novalue']))
        path.unlink()
        self.assertEqual(s.locked_read(path), dict(accepted=['baseline:novalue']))
        self.assertTrue(path.exists())
        s.o.write_json(path, dict(accepted=[]))
        with self.assertRaisesRegex(ValueError, 'projection changed'):
            s.locked_read(path)

    def test_model_and_scenario_fingerprint_changes_refuse_resume(self):
        path = self.args.output / 'models/baseline' / s.o.MODEL_NAMES[2]
        original = path.read_bytes()
        try:
            path.write_bytes(original+b'changed')
            with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                s.verify(self.args)
        finally:
            path.write_bytes(original)
        path = self.args.output / 'scenarios.json'
        original = path.read_bytes()
        try:
            changed = json.loads(original)
            changed['confirm_four'][0]['search_seeds'][0] += 1
            s.o.write_json(path, changed)
            with self.assertRaisesRegex(ValueError, 'scenario fingerprint'):
                s.verify(self.args)
        finally:
            path.write_bytes(original)

    def test_live_dashboard_does_not_confuse_watch_with_runner(self):
        with mock.patch.object(s, 'live_pids', return_value=[]):
            text = s.dashboard(self.args)
        self.assertIn('未运行', text)
        self.assertIn('只评测', text)


class ValueOffIntegration(unittest.TestCase):
    def test_real_smoke_resume_with_different_workers_and_replay(self):
        with tempfile.TemporaryDirectory(prefix='l45_value_off_integration_') as directory:
            out = Path(directory)
            script = CODE / 'confirm_l45_value_off.py'
            for workers in (2, 3):
                with (out / f'run{workers}.log').open('w') as log:
                    subprocess.run([sys.executable, str(script), '--smoke', '--workers', str(workers),
                                    '--output', str(out)], cwd=CODE.parent, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=300)
                if workers == 2:
                    frozen = {p.name: p.read_bytes() for p in (out / 'cache').glob('*.json')}
                    results = (out / 'results.json').read_bytes()
                else:
                    self.assertEqual(frozen, {p.name: p.read_bytes() for p in (out / 'cache').glob('*.json')})
                    self.assertEqual(results, (out / 'results.json').read_bytes())
            result = s.o.read_json(out / 'results.json')
            self.assertFalse(result['quality_gate_passed'])
            self.assertFalse(result['continue_formal'])
            self.assertTrue(result['replay']['scientific_result_exact_match'])
            args = s.parse_args(['--smoke', '--output', str(out)])
            value = s.read_recommended(args)
            p, _ = s.o.restore_scenario(s.o.read_json(out / 'scenarios.json')['confirm_four'][0])
            solver = s.load_recommended_solver(args, p, 0, 91)
            self.assertAlmostEqual(solver.actual_settings['beta_value'], .1)
            self.assertEqual(s.o.model_fingerprint(Path(value['models'])), value['model_hash'])
            # Also exercise the non-improvement fallback with the same real cache.
            # Only the test's eligibility fixture changes; no fake search result is produced.
            with tempfile.TemporaryDirectory(prefix='l45_value_off_fallback_') as fallback_directory:
                fallback = s.parse_args(['--smoke', '--output', fallback_directory])
                banks = s.prepare(fallback)
                (fallback.output / 'cache').mkdir()
                for path in (out / 'cache').glob('*.json'):
                    (fallback.output / 'cache' / path.name).write_bytes(path.read_bytes())
                pursuit = copy.deepcopy(s.locked_read(out / 'confirmation_result.json'))
                pursuit['eligible'] = []
                s.lock_json(fallback.output / 'confirmation_result.json', pursuit)
                failure = s.finish(fallback, banks)
                self.assertEqual(failure['recommended'], 'baseline:base')
                self.assertEqual(failure['reason'], 'no_candidate_confirmed')
                self.assertTrue(failure['replay']['scientific_result_exact_match'])
                self.assertFalse(failure['quality_gate_passed'])
            # Force only the smoke eligibility branch to exercise real regression,
            # serial timing and export regardless of the random smoke case outcomes.
            with tempfile.TemporaryDirectory(prefix='l45_value_off_forced_path_') as forced_directory:
                forced = s.parse_args(['--smoke', '--output', forced_directory])
                banks = s.prepare(forced)
                (forced.output / 'cache').mkdir()
                for path in (out / 'cache').glob('*.json'):
                    (forced.output / 'cache' / path.name).write_bytes(path.read_bytes())
                pursuit = copy.deepcopy(s.locked_read(out / 'confirmation_result.json'))
                pursuit['eligible'] = ['baseline:novalue']
                candidate = next(c for c in pursuit['candidates'] if c['uid'] == 'baseline:novalue')
                candidate['pursuit_pass'] = True
                candidate['checks'] = {k: True for k in candidate['checks']}
                s.lock_json(forced.output / 'confirmation_result.json', pursuit)
                regression = s.guard(forced, banks, None, lambda *a, **kw: None)
                self.assertEqual(regression['accepted'], ['baseline:novalue'])
                measured = s.timing(forced, banks, lambda *a, **kw: None)
                self.assertEqual(set(measured), {'baseline:base', 'baseline:novalue'})
                self.assertGreater(measured['baseline:novalue']['decisions'], 0)
                exported = s.finish(forced, banks)
                self.assertTrue(exported['replay']['scientific_result_exact_match'])
                self.assertFalse(exported['quality_gate_passed'])


if __name__ == '__main__':
    unittest.main()
