"""Read-only delivery checks for the bounded candidate-prior experiment."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'current/l45_candidate_prior_train_20261026'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    config = json.loads((OUT/'experiment.json').read_text())
    manifest = json.loads((OUT/'bank/manifest.json').read_text())
    rows = json.loads((OUT/'evaluation/results.json').read_text())
    checks = {}
    checks['registered_source_inputs_and_prior_work_unchanged'] = all(sha(p) == h
        for key in ('source_hashes', 'protected_hashes', 'bank_hashes') for p, h in config[key].items())
    for name in ('models_frozen.json', 'data/manifest.json'):
        checks[name+'_hashes'] = all(sha(p) == h for p, h in json.loads((OUT/name).read_text())['files'].items())
    entries = manifest['entries']
    checks['entire_maps_disjoint'] = len({e['map_sha256'] for e in entries}) == len(entries) == 36
    checks['entire_histories_disjoint'] = len({e['geometry_history_sha256'] for e in entries}) == 36
    expected = {(e['id'], w, m) for e in entries if e['split'] == 'test'
                for w in ('A', 'B') for m in config['methods']}
    actual = [(r['group'], r['world'], r['method']) for r in rows]
    checks['all108_unique_paired_evaluation_games'] = len(actual) == len(set(actual)) == 108 and set(actual) == expected
    checks['all_episode_hashes'] = all(sha(r['episode']) == r['episode_sha256'] for r in rows)
    checks['no_recorded_engineering_failures'] = all(r['failure'] is None for r in rows)
    audit = json.loads((OUT/'evaluation/postrun_audit_v2.json').read_text())
    checks['complete_independent_audit'] = audit['passed'] and audit['episodes'] == 108 and not audit['errors']
    old_audit = json.loads((ROOT/'current/l45_search_value_20261026/corrective_audit_registration.json').read_text())
    audit_name = str(ROOT/'code/audit_search_value_validation_v2.py')
    checks['auditor_matches_original_prior_registration'] = sha(audit_name) == old_audit['source_hashes'][audit_name]
    training = json.loads((OUT/'training.json').read_text())['records']
    checks['three_fixed80_epoch_runs_validation_only_selection'] = len(training) == 3 and all(
        len(r['logs']) == r['epochs'] == 80 and not r['test_read'] and
        r['selected_epoch'] == min(r['logs'], key=lambda x: x['validation_loss'])['epoch'] and
        r['numpy_export_max_logit_difference'] < .001 for r in training)
    report = (OUT/'REPORT.md').read_text()
    links = [target for target in re.findall(r'\]\(([^)]+)\)', report) if '://' not in target]
    checks['report_local_links_exist'] = all((OUT/link.split('#')[0]).exists() for link in links)
    checks['plots_png_pdf_present'] = all((OUT/'figures'/f'{name}.{suffix}').exists()
        for name in ('training', 'closed_loop') for suffix in ('png', 'pdf'))
    checks['no_deployment_or_value_network'] = config['deployment_changed'] is False and config['value_network'] is False
    decision = json.loads((OUT/'decision.json').read_text())
    result = dict(passed=all(checks.values()), checks=checks, evaluation_games=len(rows), independent_history_groups=6,
        audit_frames=audit['frames'], audited_simulated_transitions=audit['simulated_transitions'],
        audit_check_counts=audit['checks_performed'], continuation_decision=decision)
    (OUT/'delivery_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (OUT/'source_snapshot/verify_candidate_prior_pilot.py').write_bytes(Path(__file__).read_bytes())
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ('DELIVERY_SHA256.json', 'status.json', 'pipeline.log')]
    hashes = {str(p): sha(p) for p in files}
    (OUT/'DELIVERY_SHA256.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2)+'\n')
    assert all(sha(p) == h for p, h in hashes.items())
    print(json.dumps(result, ensure_ascii=False))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
