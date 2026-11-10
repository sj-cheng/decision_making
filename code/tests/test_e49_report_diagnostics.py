"""Independent exposure/count and map-pairing invariants; no scientific runs."""
import copy
import unittest

from e49_report import ARMS,method_diagnostics,paired_table,paired_breakdowns,statistics


def row(arm='A2',group='one',complete=False,first=70.,tier='24',suite='ID',family='long_wall'):
    return dict(task=dict(arm=arm,group=group,horizon=80,obstacle_tier=tier,suite=suite,family=family),
        summary=dict(double_capture=complete,at_least_one=first is not None,completion_cost=2 if complete else 80,
            utility=float(complete)-.005,decisions=2,latencies_ms=[10.,210.],cpu_ms=[9.,20.],
            online_overruns=1,zero_simulation_decisions=1,stop_decisions=1,simulated_transitions=3,
            hidden_target_decisions=3,visible_target_decisions=1,post_capture_visible_decisions=1,
            post_capture_active_decisions=1,first_capture_time=first,final_capture_time=71. if complete else None,
            post_capture_seconds=1. if complete else None,capture_events=[dict(pursuer=0,time=first)] if first is not None else [],
            rediscoveries=[dict(target=0,steps=2)],censored_occlusions=[dict(target=1,steps=5,reason='end_of_episode')]),
        trace=[dict(stats=dict(simulations=3),observation=dict(active_e=[True,True],reports=[[None,None],[None,None]])),
               dict(stats=dict(simulations=0),observation=dict(active_e=[True,False],reports=[[[1.,2.],None],[None,None]]))])


class ReportDiagnosticsTests(unittest.TestCase):
    def test_censoring_late_capture_and_simulation_exposure(self):
        rows=[row(),row(group='two',complete=True,first=10.)]
        result=method_diagnostics(rows)['A2']
        self.assertEqual(result['raw']['late_first_capture'],1)
        self.assertEqual(result['raw']['late_first_only'],1)
        self.assertEqual(result['raw']['rediscovered_runs'],2)
        self.assertEqual(result['raw']['censored_runs'],2)
        self.assertEqual(result['exposure']['censored_run_steps']['mean'],5.)
        self.assertEqual(result['weighted']['post_first_completion'],.5)
        self.assertEqual(result['weighted']['visible_fraction'],.25)
        self.assertEqual(result['computation']['total_completed_simulations'],6)
        self.assertEqual(result['computation']['trace_decisions_with_simulation_count'],4)
        self.assertTrue(result['computation']['simulation_evidence_complete'])
        self.assertEqual(result['exposure']['trace_visibility_counts'],dict(hidden=4,visible=2))

    def test_missing_trace_does_not_claim_simulation_evidence(self):
        item=row();item.pop('trace')
        result=method_diagnostics([item])['A2']
        self.assertFalse(result['computation']['simulation_evidence_complete'])
        self.assertEqual(result['computation']['completed_simulations']['count'],0)

    def test_map_pairing_ignores_extra_model_copies_and_keeps_four_map_denominator(self):
        rows=[]
        for tier in ('sparse','16','24'):
            for suite in ('ID','OOD'):
                for index in range(4):
                    for arm in ARMS:
                        item=row(arm=arm,group=f'{tier}/{suite}/{index}',complete=bool(index%2),
                                 first=10.,tier=tier,suite=suite,family=suite)
                        rows.append(item)
                        if arm=='A3':rows.extend([copy.deepcopy(item),copy.deepcopy(item)])
        main=statistics(rows,repetitions=100)
        table=paired_table(rows,repetitions=100)
        self.assertEqual(main['methods']['A3']['double_capture'],main['methods']['A2']['double_capture'])
        self.assertEqual(table['contrasts'][0]['simultaneous_ci95'],[0.,0.])
        breakdown=paired_breakdowns(rows,repetitions=100)
        self.assertEqual(breakdown['tier_families']['24/ID']['maps'],4)
        self.assertEqual(breakdown['tier_families']['24/ID']['methods']['A3']['raw_episodes'],12)
        self.assertEqual(breakdown['tier_families']['24/ID']['methods']['A2']['raw_episodes'],4)


if __name__=='__main__':unittest.main()
