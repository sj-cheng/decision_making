import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_e45c_budget as r
import run_e45c_evaluation_audit as audit
from e45c_belief_diagnostics import RecordedTeamBelief,FrozenTeamBelief
from solvers.e45_team_adapter import team_hidden_mask
from search_value_model import predict_joint
from run_frozen_d_closed_loop import legal_action
from e45b_sensor_audit import visible_batch


class EvaluationAuditTests(unittest.TestCase):
    def test_exact_real_failure_is_a_coarse_grid_coverage_gap(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/e45c_grid_exhaustion_20261028.json').read_text())
        boxes=np.asarray(fixture['boxes']);obs=fixture['history'][-1]
        for cls in (FrozenTeamBelief,RecordedTeamBelief):
            belief=cls(boxes,fixture['seed'])
            for item in fixture['history'][:-1]:belief.update(item)
            with self.assertRaisesRegex(RuntimeError,'Empty team-conditioned'):belief.update(obs)
        estimator=belief.estimators[0];g=estimator.graph
        prior=np.zeros_like(estimator.weights);prior[g[5]]=1/9
        prediction=predict_joint(prior,g[1],g[2],g[3])
        allowed=team_hidden_mask(g[0],np.asarray(obs['positions']),np.asarray(obs['active_p']),boxes)
        self.assertTrue(np.isfinite(prediction).all());self.assertAlmostEqual(prediction.sum(),1.)
        self.assertEqual((prediction.sum(1)>0).sum(),25);self.assertEqual(prediction[allowed].sum(),0.)
        witness=estimator.last+np.array([-.25,1.])
        self.assertTrue(legal_action(estimator.last,witness-estimator.last,1.,boxes))
        self.assertFalse(visible_batch(np.asarray(obs['positions']),np.tile(witness,(2,1)),boxes).any())

    def task(self,root,method='DIRECT43'):
        return dict(key='case_'+method,case=str(root/'case.json'),_output=str(root/'x5'),stage='evaluation',method=method,seed=17)

    def public_failure(self,root,task):
        case=dict(map=[],history=[dict(remaining=74)]);r.save(root/'case.json',case)
        r.save(root/'models_frozen.json',dict(files={}))
        stem=root/'belief_failures/public_support_test';stem.parent.mkdir()
        np.savez_compressed(stem.with_suffix('.npz'),weights=np.zeros(2))
        r.save(stem.with_suffix('.json'),dict(seed=17,boxes=[],recovered=False,failed_public_step=54,arrays=str(stem.with_suffix('.npz'))))
        return dict(task=task,error="RuntimeError('Empty team-conditioned finite-grid support; public_history_evidence="+str(stem)+"')")

    def test_failed_episode_has_null_outcomes_and_is_not_a_loss(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(r,'OUT',Path(temp)):
            root=Path(temp);task=self.task(root);failure=self.public_failure(root,task)
            row=audit.invalid_result(task,failure)
            self.assertFalse(row['valid_episode'])
            for key in ('utility','completion_cost','double_capture','captures'):self.assertIsNone(row[key])
            self.assertTrue((root/'x5/evaluation/engineering_failures/case_DIRECT43.json').exists())
            with self.assertRaisesRegex(RuntimeError,'task mismatch'):audit.invalid_result(task,failure|dict(task=task|dict(seed=99)))
            with self.assertRaisesRegex(RuntimeError,'eligible'):audit.invalid_result(task,failure|dict(error='RuntimeError(Illegal simulated transition)'))

    def test_successful_jobs_pass_through_and_unexpected_errors_remain_fatal(self):
        with tempfile.TemporaryDirectory() as temp:
            task=self.task(Path(temp));row=dict(utility=.5,reason='timeout')
            with patch.object(r,'configure'),patch.object(r.base,'paths',return_value=(Path(temp)/'episode.json',Path(temp)/'cache.json',Path(temp)/'label.npz')):
                with patch.object(audit,'ORIGINAL_EPISODE_JOB',return_value=row):self.assertIs(audit.evaluation_job(task),row)
                with patch.object(audit,'ORIGINAL_EPISODE_JOB',side_effect=RuntimeError('Independent joint simulation audit failed')):
                    with self.assertRaisesRegex(RuntimeError,'simulation audit'):audit.evaluation_job(task)

    def test_auxiliary_failure_does_not_remove_complete_primary_cells(self):
        tasks=[dict(_output='x1',key=m,method=m) for m in ('JOINT17','PRIOR17','DIRECT43')]
        rows=[t|dict(utility=.5) for t in tasks[:2]]+[tasks[-1]|dict(valid_episode=False,utility=None)]
        with patch.object(r,'compare',return_value=dict(comparisons=[])) as compare:
            result=audit.primary_inference(rows,tasks);self.assertEqual(result['status'],'complete')
            self.assertTrue(all(x['utility'] is not None for x in compare.call_args.args[0]))
        rows[0]['valid_episode']=False;rows[0]['utility']=None
        with patch.object(r,'compare') as compare:
            result=audit.primary_inference(rows,tasks);self.assertEqual(result['status'],'incomplete_due_to_engineering_failures')
            compare.assert_not_called();self.assertEqual(len(result['missing_primary_cells']),1)

    def test_failure_bounds_do_not_impute_scores_into_means(self):
        good=dict(_output='x5',key='a',opponent='rootpuct',method='DIRECT43',utility=.5,double_capture=True,completion_cost=10,latencies_ms=[2])
        bad=dict(_output='x5',key='b',opponent='rootpuct',method='DIRECT43',valid_episode=False,utility=None,remaining_at_takeover=74)
        with tempfile.TemporaryDirectory() as temp,patch.object(r,'OUT',Path(temp)),patch.object(audit,'primary_inference',return_value=dict(status='complete')),patch.object(r,'compare',return_value={}):
            result=audit.statistics([good,bad],[],[]);arm=result['arms'][0]
            self.assertEqual(arm['capture_rate_bounds'],[.5,1.])
            np.testing.assert_allclose(arm['utility_bounds'],[(.5-.185)/2,(.5+.9975)/2])
            self.assertEqual(arm['valid_only_utility'],.5);self.assertFalse(result['invalid_outcomes_imputed'])
            self.assertIsNone(bad['utility'])

if __name__=='__main__':unittest.main()
