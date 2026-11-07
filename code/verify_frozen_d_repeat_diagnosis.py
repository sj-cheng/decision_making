"""Archive integrity and semantic checks for exact-repeat/diagnosis results."""
import json,hashlib,re,fcntl,shutil
from pathlib import Path
from repeat_frozen_d_closed_loop import ROOT,SOURCE,OUT,scientific_trace
from run_frozen_d_closed_loop import sha,save,loader


def main():
    config=json.loads((OUT/'experiment.json').read_text())
    assert all(sha(p)==h for p,h in (config['sources']|config['baseline_source_and_input_hashes']|config['protected_before']).items())
    assert sha(SOURCE/'experiment.json')==config['baseline_experiment_sha256']
    assert sha(SOURCE/'confirmation/results.json')==config['baseline_results_sha256']
    manifest=json.loads((SOURCE/'DELIVERY_SHA256.json').read_text())
    # Chronological user-facing summaries now include the later experiment.
    historic={p:h for p,h in manifest['files'].items() if p not in [str(ROOT/'README.md'),
        str(ROOT/'code/L45_RESEARCH_DIRECTIONS_20261025.md'),str(ROOT/'Git提交后实验完整汇总_20261025.md')]}
    assert all(sha(p)==h for p,h in historic.items())
    repeats=json.loads((OUT/'repeat_results.json').read_text())
    contrasts=json.loads((OUT/'repeat_comparisons.json').read_text())
    assert len(repeats)==len(contrasts['rows'])==144
    assert len({r['history_id'] for r in repeats})==24
    exact=[]
    for row in repeats:
        assert sha(row['episode'])==row['episode_sha256']
        actual=json.loads(Path(row['episode']).read_text())
        prior=json.loads((SOURCE/'confirmation/episodes'/(row['key']+'.json')).read_text())
        exact.append(scientific_trace(actual['trace'])==scientific_trace(prior['trace']))
    assert all(exact)==contrasts['all_exact']
    data=json.loads((OUT/'root_diagnostics.json').read_text())
    assert len(data['rows'])==24 and len({r['group'] for r in data['rows']})==12
    assert all(r['common_public_estimator_identical'] and r['G_in_TGeo_candidates'] for r in data['rows'])
    assert all(len(r['cases'])==12 and len(r['paired_model_differences'])==12 for r in data['rows'])
    assert all(len(c['own_path'])==25 for r in data['rows'] for c in r['cases'])
    tests=json.loads((OUT/'unit_tests.json').read_text());assert tests['passed']
    links=[]
    for target in re.findall(r'\]\(([^)]+)\)',(OUT/'REPORT.md').read_text()):
        if '://' not in target and not target.startswith('#'):
            assert (OUT/target.split('#')[0]).resolve().exists(),target
            links.append(target)
    archive={}
    for name,h in config['sources'].items():
        p=Path(name);dest=OUT/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest);assert sha(dest)==h
        archive[name]=dict(snapshot=str(dest),sha256=h)
    save(OUT/'source_snapshot.json',archive)
    result=dict(passed=True,repeated_games=144,scientifically_exact_games=sum(exact),effective_history_groups=24,
        diagnostic_roots=24,diagnostic_history_groups=12,old_artifacts_unchanged=len(historic),
        frozen_D_valid=bool(loader.verify()),protected_sources_unchanged=True,training=False,deployment_changed=False,
        report_links=len(links),no_new_independent_confirmation_claimed=True)
    save(OUT/'delivery_checks.json',result)
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='DELIVERY_SHA256.json']
    files += [ROOT/'code'/name for name in ('report_frozen_d_repeat_diagnosis.py','verify_frozen_d_repeat_diagnosis.py',
        'tests/test_frozen_d_repeat_diagnosis.py')]
    files += [Path(p) for p in config['sources']]
    hashed={str(p.resolve()):sha(p) for p in files}
    save(OUT/'DELIVERY_SHA256.json',dict(files=hashed,count=len(hashed)))
    assert all(sha(p)==h for p,h in hashed.items())
    print(json.dumps(result|dict(delivery_files=len(hashed)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    with (ROOT/'current/.l45_local_pursuit_serial.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
