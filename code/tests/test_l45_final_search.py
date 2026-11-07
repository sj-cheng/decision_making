"""Checks for the evaluation-only, frozen final-checkpoint search study."""
import copy
from collections import Counter
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import confirm_l45_final_search as study


class FinalSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def registry(self, root, fingerprint=None):
        models = [dict(label='baseline', seed=None)]
        models += [dict(label=f'seed{seed}_{kind}', seed=seed)
                   for kind in ('best', 'final') for seed in study.SEEDS]
        for model in models:
            model.update(models=str(Path(root)/model['label']), model_hash=fingerprint or {'frozen': 'hash'})
        return models

    def rows(self):
        rows = []
        models = self.registry('/unused')
        for model in models:
            modes = ('search', 'policy') if model['label'] == 'baseline' or model['label'].endswith('_final') else ('search',)
            for mode in modes:
                for scene in range(2):
                    for opponent in study.original.OPPONENTS:
                        rows.append(dict(label=model['label'], training_seed=model['seed'], mode=mode, role=1,
                            layout='four', episode_hash=f'scene-{scene}', opponent=opponent,
                            search_seeds=[13+scene, 31+scene], legal_double_capture=0,
                            legal_survival=0, one_capture_timeout=0, pursuer_illegal=0, evader_illegal=0,
                            completion_cost=80., capture_time=None, pursuer_return=0., evader_return=0.,
                            pursuer_latency_sum_ms=3., pursuer_latency_count=1))
        return rows

    def test_complete_matrix_is_9900_with_seven_search_four_policy_models(self):
        args = SimpleNamespace(output=Path('/tmp/l45_final_matrix_only'), smoke=False)
        registry = self.registry('/unused')
        record = dict(source_contract=dict(registry=registry))
        scenes = [dict(layout='four' if i < 200 else 'none', episode_hash=f'scene-{i}') for i in range(300)]
        with mock.patch.object(study.original, 'model_fingerprint', return_value={'frozen': 'hash'}):
            tasks = study.build_tasks(args, scenes, record)
        self.assertEqual(len(tasks), 9900)
        self.assertEqual(len({t['key'] for t in tasks}), 9900)
        modes = Counter(t['mode'] for t in tasks)
        self.assertEqual(modes, {'search': 6300, 'policy': 3600})
        self.assertEqual({t['label'] for t in tasks if t['mode'] == 'search'}, {m['label'] for m in registry})
        self.assertEqual({t['label'] for t in tasks if t['mode'] == 'policy'}, {'baseline', 'seed0_final', 'seed1_final', 'seed2_final'})
        self.assertEqual(Counter(t['opponent'] for t in tasks), {op:3300 for op in study.original.OPPONENTS})
        expected = dict(search_budget=1000, search_only_budget=3000, search_depth=100, beta_policy=.5, beta_value=.1)
        for task in tasks:
            self.assertEqual(task['role'], 1)
            self.assertEqual(task['settings'], expected)
            self.assertEqual(Path(task['baseline']), args.output/'models/baseline')
            self.assertEqual(Path(task['models']), args.output/'models'/task['label'])

    def test_physical_identity_ignores_rng_metadata_but_keeps_physical_state(self):
        spec = study.original.make_scenario(560)
        changed = copy.deepcopy(spec)
        changed['seed'] += 1
        changed['search_seeds'] = [1, 2]
        changed['episode_hash'] = 'metadata-only'
        self.assertEqual(study.physical_hash(spec), study.physical_hash(changed))
        for field in ('initial_state', 'action_lims', 'control_lims', 'evader_speed', 'pursuer_speed', 'layout'):
            changed = copy.deepcopy(spec)
            if field in ('initial_state', 'action_lims', 'control_lims'):
                changed[field][0][0] += .001
            elif field == 'layout':
                changed[field] = 'none'
            else:
                changed[field] += .001
            with self.subTest(field=field):
                self.assertNotEqual(study.physical_hash(spec), study.physical_hash(changed))

    def test_strict_pairs_reject_missing_duplicate_or_different_search_rng(self):
        rows = self.rows()
        study.strict_pairs(rows, 'final', 'baseline', 'search')
        missing = copy.deepcopy(rows)
        missing.remove(next(r for r in missing if r['label'] == 'seed1_final' and r['mode'] == 'search'))
        with self.assertRaisesRegex(ValueError, 'Missing or unpaired'):
            study.strict_pairs(missing, 'final', 'baseline', 'search')
        duplicated = copy.deepcopy(rows)
        duplicated.append(copy.deepcopy(next(r for r in rows if r['label'] == 'seed2_final' and r['mode'] == 'search')))
        with self.assertRaisesRegex(ValueError, 'Duplicate paired'):
            study.strict_pairs(duplicated, 'final', 'baseline', 'search')
        rng_changed = copy.deepcopy(rows)
        row = next(r for r in rng_changed if r['label'] == 'seed0_final' and r['mode'] == 'search')
        row['search_seeds'][0] += 1
        with self.assertRaisesRegex(ValueError, 'Unpaired search random seeds'):
            study.strict_pairs(rng_changed, 'final', 'baseline', 'search')

    def test_final_best_pairing_is_same_seed_and_statistics_cluster_opponents(self):
        rows = self.rows()
        for row in rows:
            if row['label'].endswith('_final'):
                row['legal_double_capture'] = 1
            elif row['label'] == 'seed1_best':
                row['legal_double_capture'] = 1
        keys, candidates, references = study.strict_pairs(rows, 'final', 'best', 'search')
        self.assertEqual(len(keys), 6)
        for seed, candidate, reference in zip(study.SEEDS, candidates, references):
            self.assertEqual({r['label'] for r in candidate.values()}, {f'seed{seed}_final'})
            self.assertEqual({r['label'] for r in reference.values()}, {f'seed{seed}_best'})
        result = study.comparison(rows, reference='best')
        self.assertEqual(result['per_seed_delta'], [1., 0., 1.])
        self.assertAlmostEqual(result['delta'], 2/3)
        self.assertEqual(result['n_scenes'], 2)
        self.assertEqual(result['observed_discordances'], 12)
        self.assertEqual(result['per_opponent_delta'], {op:2/3 for op in study.original.OPPONENTS})
        self.assertEqual(result, study.comparison(rows, reference='best'))
        clustered = study.paired_statistics(keys, [[1., 1., 1., 0., 0., 0.]] * 3)
        self.assertEqual(clustered['n_scenes'], 2)
        self.assertEqual(clustered['delta'], .5)
        self.assertEqual(clustered['ci95'], [0., 1.])
        # With no observed discordances, report the degenerate interval without
        # calling it evidence of equivalence.
        equal = study.comparison(self.rows())
        self.assertEqual(equal['ci95'], [0., 0.])
        self.assertIn('do not prove equivalence', equal['note'])

    def test_transfer_reports_search_gain_minus_policy_gain_and_summary_groups(self):
        rows = self.rows()
        for row in rows:
            if row['label'].endswith('_final'):
                row['legal_double_capture'] = int(row['mode'] == 'policy' or row['training_seed'] < 2)
        transfer = study.transfer_comparison(rows)
        self.assertEqual(transfer['per_seed_delta'], [0., 0., -1.])
        self.assertAlmostEqual(transfer['delta'], -1/3)
        self.assertEqual(transfer['n_scenes'], 2)
        grouped = {(r['label'], r['mode']):r for r in study.model_summary(rows)}
        self.assertEqual(len(grouped), 11)
        self.assertEqual(grouped[('seed2_final', 'policy')]['success_rate'], 1.)
        self.assertEqual(grouped[('seed2_final', 'search')]['success_rate'], 0.)
        self.assertEqual(grouped[('baseline', 'search')]['n'], 6)
        self.assertEqual(grouped[('baseline', 'search')]['decision_ms'], 3.)

    def test_parent_output_guard_rejects_before_creating_any_child_directory(self):
        with tempfile.TemporaryDirectory(prefix='l45_final_guard_test_') as temporary:
            parent = Path(temporary)/'parent'
            parent.mkdir()
            marker = parent/'untouched.txt'
            marker.write_text('original')
            snapshot = set(parent.iterdir())
            child = parent/'forbidden_child'
            args = SimpleNamespace(source=parent, output=child)
            with self.assertRaisesRegex(ValueError, 'never modify the parent'):
                study.prepare(args)
            self.assertEqual(set(parent.iterdir()), snapshot)
            self.assertEqual(marker.read_text(), 'original')
            with self.assertRaises(SystemExit):
                study.parse_args(['--source', str(parent), '--output', str(child)])
            self.assertEqual(set(parent.iterdir()), snapshot)

    def test_snapshots_and_scene_fingerprints_reject_tampering(self):
        with tempfile.TemporaryDirectory(prefix='l45_final_integrity_test_') as temporary:
            parent = Path(temporary)/'parent'
            parent.mkdir()
            previous = study.original.make_scenario(910)
            study.original.write_json(parent/'scenarios.json', dict(validation=[previous]))
            args = study.parse_args(['--source', str(parent), '--output', str(Path(temporary)/'output'), '--smoke'])
            fingerprint = study.original.model_fingerprint(study.original.SOURCE_MODELS)
            registry = self.registry('/unused', fingerprint)
            for model in registry:
                model['models'] = str(study.original.SOURCE_MODELS)
            record = dict(source=str(parent), source_contract=dict(registry=registry), config=dict(no_training=True))
            with mock.patch.object(study, 'signed_record', return_value=record):
                scenarios, _ = study.prepare(args)
                study.verify(args)
                scenes_again, _ = study.prepare(args)
                self.assertEqual(scenarios, scenes_again)
                new = {study.physical_hash(s) for cases in scenarios.values() for s in cases}
                self.assertEqual(len(new), 2)
                self.assertNotIn(study.physical_hash(previous), new)
                target = args.output/'models/seed0_final'/study.original.MODEL_NAMES[0]
                original_bytes = target.read_bytes()
                target.write_bytes(original_bytes+b'tampered-copy')
                with self.assertRaisesRegex(ValueError, 'Frozen model snapshot changed'):
                    study.verify(args)
                target.write_bytes(original_bytes)
                changed_record = copy.deepcopy(record)
                changed_record['config']['no_training'] = False
                with mock.patch.object(study, 'signed_record', return_value=changed_record):
                    with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
                        study.verify(args)
                scenes = args.output/'scenarios.json'
                changed = study.original.read_json(scenes)
                changed['confirmation'][0]['search_seeds'][0] += 1
                study.original.write_json(scenes, changed)
                with self.assertRaisesRegex(ValueError, 'Frozen evaluation scenarios changed'):
                    study.verify(args)
            self.assertEqual(set(parent.iterdir()), {parent/'scenarios.json'})
            self.assertEqual(study.original.read_json(parent/'scenarios.json'), dict(validation=[previous]))

    def test_real_passive_cpp_replay_has_identical_scientific_metrics(self):
        # One real final-time decision exercises the C++ bindings and tree
        # observation, without running the expensive scientific evaluation.
        spec = study.original.make_scenario(560)
        p, state = study.original.restore_scenario(spec)
        state[p.time_idx, 0] = p.tf-p.dt
        spec['initial_state'] = state.tolist()
        spec['episode_hash'] = study.original.json_hash({k:v for k,v in spec.items() if k != 'episode_hash'})
        settings = study.search_settings(SimpleNamespace(smoke=True))
        baseline = study.DEFAULT_SOURCE/'baseline'
        models = study.DEFAULT_SOURCE/'seed0/final_models'
        task = study.original.evaluation_tasks(models, 'seed0_final', 0, baseline, [spec], settings,
                    modes=('search',), roles=(1,), opponents=('Search Only',))[0]
        row = study.original.evaluate_one(task)
        original_run = study.original.run_episode
        task['expected_scientific_digest'] = study.scientific_digest(row)
        task['timing_context'] = 'diagnostic_replay'
        task['key'] = 'diagnostic-key'
        replay = study.trace_evaluation(task)
        self.assertIs(study.original.run_episode, original_run)
        self.assertEqual(study.scientific_digest(row), study.scientific_digest(replay))
        self.assertEqual(len(replay['trace']), 1)
        step = replay['trace'][0]
        self.assertEqual(len(step['state']), p.state_dim)
        self.assertEqual(len(step['action']), p.action_dim)
        self.assertEqual(len(step['policies']), 2)
        self.assertTrue(step['roots'])
        self.assertTrue(all(root['child_visit_total'] > 0 for root in step['roots']))
        self.assertTrue(all(policy['mean_shift'] >= 0 for policy in step['policies']))
        with self.assertRaisesRegex(ValueError, 'Passive diagnostic replay changed'):
            study.trace_evaluation(dict(task, expected_scientific_digest='wrong'))
        self.assertIs(study.original.run_episode, original_run)


if __name__ == '__main__':
    unittest.main()
