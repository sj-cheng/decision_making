"""Final integrity checks and archive manifest for the completed capture study."""
from pathlib import Path
import hashlib,json,re,fcntl
from run_frozen_d_closed_loop import ROOT,OUT,FROZEN,save,sha,tasks,loader


def main():
    experiment=json.loads((OUT/'experiment.json').read_text())
    assert (OUT/'status.json').exists() and json.loads((OUT/'status.json').read_text())['stage']=='complete'
    assert all(sha(p)==h for p,h in (experiment['sources_and_inputs']|experiment['protected_before']).items())
    bank=json.loads((OUT/'bank_frozen.json').read_text())
    assert all(sha(p)==h for p,h in bank['files'].items())
    frozen=loader.verify()
    checks=[]
    for stage,groups,games in (('development',12,72),('confirmation',24,144)):
        rows=json.loads((OUT/stage/'results.json').read_text())
        stat=json.loads((OUT/stage/'statistics.json').read_text())
        audit=json.loads((OUT/stage/'postrun_audit.json').read_text())
        assert len(rows)==games and stat['groups']==groups and stat['games']==games
        assert set(r['key'] for r in rows)==set(t['key'] for t in tasks(stage))
        assert len(set((r['history_id'],r['world'],r['method']) for r in rows))==games
        assert all(sha(r['episode'])==r['episode_sha256'] for r in rows)
        assert all(audit['checks'].values())
        assert all(not any(r['simulation_errors']) for r in rows)
        assert len(audit['errors'])==0
        checks.append(dict(stage=stage,groups=groups,games=games,passed=True))
    preflight=json.loads((OUT/'preflight.json').read_text())
    assert preflight['passed'] and all(r['exact'] for r in preflight['replays'])
    old=ROOT/'current/l45_weighted_search_interface_v2_20261026'
    prior=json.loads((old/'DELIVERY_SHA256.json').read_text())
    # Top-level chronological summaries may now contain this later stage;
    # all prior experiment/data/frozen-code artifacts themselves must match.
    prior_entries={p:h for p,h in prior['files'].items() if Path(p).is_relative_to(ROOT/'current') or Path(p).is_relative_to(ROOT/'code')
        and Path(p).name!='L45_RESEARCH_DIRECTIONS_20261025.md'}
    assert all(sha(p)==h for p,h in prior_entries.items())
    links=[]
    for target in re.findall(r'\]\(([^)]+)\)',(OUT/'REPORT.md').read_text()):
        if '://' not in target and not target.startswith('#'):
            assert (OUT/target.split('#')[0]).resolve().exists(),target
            links.append(target)
    archive=json.loads((OUT/'source_snapshot.json').read_text())
    assert all(sha(v['snapshot'])==v['sha256'] for v in archive.values())
    status=dict(passed=True,stages=checks,engineering_replays=len(preflight['replays']),
        frozen_estimator_valid=bool(frozen),scientific_sources_and_inputs_unchanged=True,
        protected_deployment_files_unchanged=True,prior_experiment_artifacts_unchanged=len(prior_entries),
        source_snapshot_valid=True,report_links_checked=len(links),network_training=False,
        confirmation_retuned=False)
    save(OUT/'delivery_checks.json',status)
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='DELIVERY_SHA256.json']
    files += [Path(p) for p in experiment['sources_and_inputs']]
    files += [ROOT/'code'/name for name in ('analyze_frozen_d_closed_loop.py','report_frozen_d_closed_loop.py',
        'verify_frozen_d_closed_loop_delivery.py','diagnose_frozen_d_geometry_failure.py','L45_RESEARCH_DIRECTIONS_20261025.md')]
    files += [ROOT/'README.md',ROOT/'Git提交后实验完整汇总_20261025.md']
    files += [FROZEN/p for p in frozen['files']]
    manifest={str(p.resolve()):sha(p) for p in files}
    save(OUT/'DELIVERY_SHA256.json',dict(files=manifest,count=len(manifest)))
    assert all(sha(p)==h for p,h in manifest.items())
    print(json.dumps(status|dict(delivery_files=len(manifest)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    with (ROOT/'current/.l45_local_pursuit_serial.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
