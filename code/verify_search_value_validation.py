"""Read-only scientific delivery verification for the bounded search study."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'current/l45_search_value_20261026'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(output):
    output=Path(output).resolve()
    errors=[];checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:errors.append(name)
    config=json.loads((output/'experiment.json').read_text())
    check('all_algorithm_input_and_protected_hashes_unchanged',all(sha(p)==h for section in ('source_hashes','protected_sources','prior_results') for p,h in config[section].items()))
    for p,h in config['source_hashes'].items():
        relative=Path(p).relative_to(ROOT/'code') if Path(p).is_relative_to(ROOT/'code') else None
        if relative is not None and (output/'source_snapshot'/relative).exists():
            check('snapshot:'+str(relative),sha(output/'source_snapshot'/relative)==h)
    bank=json.loads((output/'bank_frozen.json').read_text())
    check('unseen_bank_still_exact',all(sha(p)==h for p,h in bank['files'].items()))
    check('bank_was_frozen_before_outcomes',bank['outcomes_read'] is False and bank['groups']==12)
    tests=json.loads((output/'unit_tests.json').read_text())
    check('twelve_unit_tests_passed',tests['passed'] and tests['tests']==12)
    check('unit_test_sources_unchanged',all(sha(ROOT/p)==h for p,h in tests['source_hashes'].items()))
    if (output/'resume_record.json').exists():
        resume=json.loads((output/'resume_record.json').read_text())
        check('interruption_cached_results_preserved',all(sha(p)==h for p,h in resume['cached_hashes'].items()))
        check('resume_unchanged_parameters',resume['parameters_changed'] is False)
        check('resume_launcher_preserved',sha(ROOT/'code/resume_search_value_validation.py')==resume['launcher_sha256'])
    for record_path in sorted(output.glob('confirmation_resume_record*.json')):
        resume=json.loads(record_path.read_text())
        prefix=record_path.stem+':'
        check(prefix+'committed_results_preserved',all(sha(p)==h for p,h in resume['cached_hashes'].items()))
        check(prefix+'unchanged_algorithm_and_parameters',resume['algorithm_changed'] is False and resume['parameters_changed'] is False)
        check(prefix+'launcher_preserved',sha(ROOT/resume['launcher'])==resume['launcher_sha256'])
    preflight=json.loads((output/'preflight.json').read_text())
    check('independent_preflight_passed',preflight['passed'] and len(preflight['records'])==15)
    check('high_budget_all_preflight_roots_visited',all(r['visited_roots']==r['root_candidates'] for r in preflight['records'] if r['method'].endswith('H')))
    groups={};counts={}
    for stage,expected in (('development',120),('confirmation',240)):
        p=output/stage/'results.json'
        if not p.exists():continue
        rows=json.loads(p.read_text());counts[stage]=len(rows)
        check(stage+':complete_count',len(rows)==expected)
        seen=set()
        for row in rows:
            key=(row['group'],row['world'],row['opponent'],row['method'])
            check(stage+':unique:'+row['key'],key not in seen);seen.add(key)
            check(stage+':episode_hash:'+row['key'],sha(row['episode'])==row['episode_sha256'])
        groups[stage]={r['group'] for r in rows}
        audit_path=output/stage/'postrun_audit_v2.json'
        if not audit_path.exists():audit_path=output/stage/'postrun_audit.json'
        audit=json.loads(audit_path.read_text())
        check(stage+':independent_postrun_audit',audit['passed'] and not audit['errors'])
    gate=json.loads((output/'development_gate.json').read_text())
    dev_stats=json.loads((output/'development/statistics.json').read_text())
    check('development_gate_matches_frozen_analysis',gate['decision']==dev_stats['decision'] and gate['continue_to_confirmation']==dev_stats['decision']['development_continue'])
    check('confirmation_not_consumed_if_development_stops',gate['continue_to_confirmation'] or not list((output/'confirmation').glob('episodes/*.json')))
    check('confirmation_complete_if_started',not gate['continue_to_confirmation'] or counts.get('confirmation')==240)
    if 'confirmation' in groups:
        check('development_confirmation_groups_disjoint',not(groups['development']&groups['confirmation']))
    if (output/'corrective_audit_registration.json').exists():
        corrected=json.loads((output/'corrective_audit_registration.json').read_text())
        check('audit_correction_sources_frozen_before_confirmation',all(sha(p)==h for p,h in corrected['source_hashes'].items()) and corrected['confirmation_games_previously_evaluated']==0)
        check('original_failed_audit_preserved',sha(output/'development/postrun_audit.json')==corrected['original_audit_sha256'])
        check('original_stop_decision_preserved',sha(output/'decision.json')==corrected['original_decision_sha256'])
        check('correction_did_not_change_algorithm_or_gate',corrected['algorithm_changed'] is False and corrected['parameters_changed'] is False and corrected['gate_unchanged'] is True)
        extra=json.loads((output/'audit_unit_tests.json').read_text())
        check('five_auditor_regression_tests_passed',extra['passed'] and extra['tests']==5)
    decision_path=output/'final_decision.json'
    if not decision_path.exists():decision_path=output/'decision.json'
    decision=json.loads(decision_path.read_text())
    check('no_training_or_deployment_change',decision.get('neural_training') is False and decision.get('deployment_changed') is False)
    if gate['continue_to_confirmation']:
        confirm_stats=json.loads((output/'confirmation/statistics.json').read_text())
        check('authoritative_decision_matches_confirmation',decision.get('statistical_decision')==confirm_stats['decision'])
        check('only_rootpuct_both_primary_passes_allow_followup',decision.get('stop')==(not confirm_stats['decision']['rootpuct_followup']))
        check('confirmation_and_corrected_audit_recorded',decision.get('confirmation_run') is True and decision.get('authoritative_audit_version')==2)
    analysis=json.loads((output/'analysis_decision.json').read_text())
    check('analysis_and_execution_stop_decisions_agree',analysis['stop_modification_direction']==decision['stop'])
    prior=ROOT/'current/l45_rollout_replan_exact_20261026/DELIVERY_SHA256.json'
    old=json.loads(prior.read_text())['files']
    check('all_previous_delivery_files_preserved',all(sha(p)==h for p,h in old.items()))
    check('report_delivered',(output/'REPORT.md').exists())
    manifest=output/'DELIVERY_SHA256.json'
    if manifest.exists():
        delivered=json.loads(manifest.read_text())
        check('delivery_count',len(delivered['files'])==delivered['count'])
        check('delivery_hashes',all(sha(p)==h for p,h in delivered['files'].items()))
    return dict(passed=not errors,errors=errors,checks=checks,counts=counts,
        previous_delivery_files=len(old),neural_training=False,deployment_changed=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=DEFAULT);args=ap.parse_args()
    result=verify(args.output)
    p=args.output/'delivery_checks.json'
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],errors=result['errors'],counts=result['counts'],previous_delivery_files=result['previous_delivery_files'])))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
