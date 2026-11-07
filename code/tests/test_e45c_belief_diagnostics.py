import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import e45c_belief_diagnostics as diagnostic
from run_e45_team_pilot import make_case


class BeliefDiagnosticTests(unittest.TestCase):
    def test_recording_keeps_frozen_public_posterior_and_rng_exact(self):
        case = make_case('long_wall', 0)
        a = diagnostic.FrozenTeamBelief(case['map'], 17)
        b = diagnostic.RecordedTeamBelief(case['map'], 17)
        for obs in case['history']:
            a.update(obs)
            b.update(obs)
            self.assertEqual(a.signature(), b.signature())
            for ea, eb in zip(a.estimators, b.estimators):
                np.testing.assert_array_equal(ea.rng, eb.rng)
                np.testing.assert_array_equal(ea.weights, eb.weights)
                np.testing.assert_array_equal(ea.reachable, eb.reachable)
        self.assertEqual(b.public_history, case['history'])

    def test_support_failure_preserves_public_history_and_arrays(self):
        case = make_case('long_wall', 0)
        belief = diagnostic.RecordedTeamBelief(case['map'], 17)
        for obs in case['history'][:-1]:belief.update(obs)
        with tempfile.TemporaryDirectory() as temp, patch.object(diagnostic, 'EVIDENCE', Path(temp)):
            with patch.object(diagnostic.RecordedTeamBelief, 'update_team', side_effect=RuntimeError(diagnostic.SUPPORT_ERRORS[0])):
                with self.assertRaisesRegex(RuntimeError, 'public_history_evidence='):
                    belief.update(case['history'][-1])
            path, = Path(temp).glob('*.json')
            saved = json.loads(path.read_text())
            self.assertEqual(saved['history'], case['history'])
            self.assertEqual(saved['seed'], 17)
            self.assertEqual(saved['error'], diagnostic.SUPPORT_ERRORS[0])
            with np.load(saved['arrays'], allow_pickle=False) as data:
                np.testing.assert_array_equal(data['boxes'], case['map'])
                self.assertIn('target_0_rng', data.files)

    def test_other_errors_still_raise_without_a_support_archive(self):
        case = make_case('long_wall', 0)
        belief = diagnostic.RecordedTeamBelief(case['map'], 17)
        with tempfile.TemporaryDirectory() as temp, patch.object(diagnostic, 'EVIDENCE', Path(temp)):
            with patch.object(diagnostic.RecordedTeamBelief, 'update_team', side_effect=ValueError('Bad clock')):
                with self.assertRaisesRegex(ValueError, 'Bad clock'):belief.update(case['history'][0])
            self.assertFalse(list(Path(temp).iterdir()))

    @patch.object(diagnostic,'EVIDENCE',None)
    def test_real_sensor_boundary_exhaustion_repaired_without_changing_d_kernel(self):
        from solvers.e45_team_adapter import public_observation,team_hidden_mask
        boxes=np.zeros((0,2,2));active=np.ones(2,bool)
        first=public_observation(0,np.array([[-5.,-1.],[1.,5.]]),np.zeros((2,2)),active,active,boxes)
        second=public_observation(1,np.array([[-5.0000001,-1.],[1.,5.0000001]]),np.tile([1.,-1.],(2,1)),active,active,boxes)
        self.assertEqual(second['reports'],[[None,None],[None,None]])
        frozen=diagnostic.FrozenTeamBelief(boxes,17);frozen.update(first)
        with self.assertRaisesRegex(RuntimeError,'Empty team-conditioned'):frozen.update(second)
        answers=[]
        for _ in range(2):
            belief=diagnostic.RecordedTeamBelief(boxes,17);belief.update(first);belief.update(second)
            self.assertEqual(belief.step,1)
            self.assertEqual(belief.last_stats['repaired_targets'],[0,1])
            for e in belief.estimators:
                self.assertEqual(e.step,1);self.assertEqual(len(e.parents),1)
                self.assertAlmostEqual(e.weights.sum(),1.)
                np.testing.assert_array_equal(e.particles,np.tile([1.,-1.],(64,1)))
                self.assertTrue(team_hidden_mask(e.particles,np.asarray(second['positions']),active,boxes).all())
                ids=np.flatnonzero(e.weights.sum(1)>0)
                for k in ids:
                    parent=e.parents[-1][k];self.assertEqual(parent,e.graph[5])
                    self.assertIn(k,e.graph[1][parent,:e.graph[2][parent]])
            answers.append(belief.signature())
        self.assertEqual(*answers)
        from solvers.e45b_joint_nte import Controller
        from e45c_capture_audit import audit_simulated
        with patch('solvers.e45b_joint_nte.TeamBelief',diagnostic.RecordedTeamBelief):
            controller=Controller(boxes,1.1,17);controller.observe(first)
            controller.act(second,fixed_simulations=12)
            self.assertEqual(controller.last_stats['simulations'],12)
            self.assertFalse(audit_simulated(np.concatenate(controller.traces),boxes,1.1).any())

    def test_genuinely_empty_motion_support_is_not_filled(self):
        case=make_case('long_wall',0);belief=diagnostic.RecordedTeamBelief(case['map'],17)
        for obs in case['history']:belief.update(obs)
        e=belief.estimators[0];weights=e.weights.copy();rng=e.rng.copy()
        self.assertFalse(belief.repair_boundary(e,weights,e.graph,rng,np.zeros(len(weights),bool)))
        np.testing.assert_array_equal(e.weights,weights);np.testing.assert_array_equal(e.rng,rng)

    def test_chained_amendments_keep_previous_evidence_and_hash_checks(self):
        import run_e45c_budget as runner
        with tempfile.TemporaryDirectory() as temp, patch.object(runner, 'OUT', Path(temp)):
            root = Path(temp)
            source = root/'source.py';source.write_text('version 3')
            snapshot = root/'previous_snapshot.py';snapshot.write_text('version 2')
            original = root/'original.py';original.write_text('version 1')
            config = dict(source_hashes={str(source):runner.sha(original)}, protected_hashes={}, bank_hashes={})
            runner.save(root/'experiment.json', config)
            first = root/'REPAIR_CAPTURE_AUDIT_20261028.json'
            runner.save(first, dict(original_experiment_sha256=runner.sha(root/'experiment.json'),
                overrides={str(source):dict(original_sha256=runner.sha(original),repaired_sha256=runner.sha(snapshot))},
                extra_hashes={str(snapshot):runner.sha(snapshot)}))
            second = root/'REPAIR_SUPPORT_DIAGNOSTICS_20261028.json'
            amendment = dict(original_experiment_sha256=runner.sha(root/'experiment.json'),
                previous_amendment=str(first), previous_amendment_sha256=runner.sha(first),
                overrides={str(source):dict(original_sha256=runner.sha(snapshot),repaired_sha256=runner.sha(source))},extra_hashes={})
            runner.save(second, amendment)
            self.assertEqual(runner.verify(), config)
            snapshot.write_text('tampered')
            with self.assertRaisesRegex(RuntimeError, 'repair artifact changed'):runner.verify()
            snapshot.write_text('version 2')
            amendment['overrides'][str(source)]['original_sha256']='incorrect'
            runner.save(second, amendment)
            with self.assertRaises(AssertionError):runner.verify()


if __name__ == '__main__':unittest.main()
