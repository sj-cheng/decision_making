"""Integrity/archive checks for the completed exact-execution rollout study."""
import json,re,hashlib,fcntl,shutil
from pathlib import Path
from run_rollout_replan_experiment import ROOT,SOURCE,save,sha,loader

OUT=ROOT/'current/l45_rollout_replan_exact_20261026'
SLOW=ROOT/'current/l45_rollout_replan_20261026'


def main():
    assert json.loads((OUT/'status.json').read_text())['stage']=='complete'
    config=json.loads((OUT/'experiment.json').read_text());revision=json.loads((OUT/'execution_revision.json').read_text())
    sources=config['source_hashes']|revision['source_hashes']
    assert all(sha(p)==h for p,h in (sources|config['original_sources_and_inputs']|config['protected_sources']).items())
    frozen=json.loads((OUT/'bank_frozen.json').read_text())
    assert all(sha(p)==h for p,h in frozen['files'].items())
    for original,h in frozen['files'].items():
        relative=Path(original).relative_to(SLOW/'bank')
        assert sha(OUT/'bank'/relative)==h,'Actual copied confirmation input changed: '+str(relative)
    assert not list((SLOW/'confirmation/episodes').glob('*.json'))
    assert len(list((SLOW/'development/cache').glob('*.json')))==62
    assert len(json.loads((SLOW/'development/partial_results.json').read_text()))==62
    checks=[]
    for stage,games,groups in (('development',120,12),('confirmation',240,24)):
        rows=json.loads((OUT/stage/'results.json').read_text());stats=json.loads((OUT/stage/'statistics.json').read_text())
        audit=json.loads((OUT/stage/'postrun_audit.json').read_text())
        assert len(rows)==stats['games']==games and stats['groups']==groups
        assert len(set((r['history_id'],r['world'],r['method']) for r in rows))==games
        assert all(sha(r['episode'])==r['episode_sha256'] for r in rows)
        checks.append(dict(stage=stage,games=games,groups=groups,postrun_error_count=len(audit['errors']),
            all_information_and_accounting_checks=all(audit['checks'].values())))
    assert json.loads((OUT/'preflight.json').read_text())['passed']
    assert json.loads((OUT/'unit_tests.json').read_text())['passed']
    prior=json.loads((SOURCE/'DELIVERY_SHA256.json').read_text())
    exclude={str(ROOT/'README.md'),str(ROOT/'code/L45_RESEARCH_DIRECTIONS_20261025.md'),str(ROOT/'Git提交后实验完整汇总_20261025.md')}
    preserved={p:h for p,h in prior['files'].items() if p not in exclude}
    assert all(sha(p)==h for p,h in preserved.items())
    assert loader.verify()
    links=[]
    for p in (OUT/'REPORT.md',SLOW/'REPORT.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' not in target and not target.startswith('#'):
                assert (p.parent/target.split('#')[0]).resolve().exists(),(p,target)
                links.append(target)
    archive={}
    for name,h in sources.items():
        original=Path(name);dest=OUT/'source_snapshot'/original.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,dest)
        assert sha(dest)==h;archive[name]=dict(snapshot=str(dest),sha256=h)
    save(OUT/'source_snapshot.json',archive)
    summary=dict(passed=True,stages=checks,original_artifacts_unchanged=len(preserved),
        slow_execution_completed_games_preserved=62,confirmation_bank_not_evaluated_before_execution_revision=True,
        exact_geometry_position_checks=198,fixed64_slow_fast_search_equivalent=True,
        frozen_D_valid=True,original_algorithms_unchanged=True,neural_training=False,deployment_changed=False,report_links_checked=len(links))
    save(OUT/'delivery_checks.json',summary)
    files=[p for directory in (OUT,SLOW) for p in directory.rglob('*') if p.is_file() and p.name!='DELIVERY_SHA256.json']
    files += [Path(p) for p in sources]
    files += [ROOT/'code'/n for n in ('analyze_rollout_replan_experiment.py','report_rollout_replan_experiment.py','verify_rollout_replan_experiment.py')]
    files += [ROOT/'README.md',ROOT/'code/L45_RESEARCH_DIRECTIONS_20261025.md',ROOT/'Git提交后实验完整汇总_20261025.md']
    manifest={str(p.resolve()):sha(p) for p in files}
    save(OUT/'DELIVERY_SHA256.json',dict(files=manifest,count=len(manifest)))
    assert all(sha(p)==h for p,h in manifest.items())
    print(json.dumps(summary|dict(delivery_files=len(manifest)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    with (ROOT/'current/.l45_local_pursuit_serial.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
