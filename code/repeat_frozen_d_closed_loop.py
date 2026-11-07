"""Actually repeat all 144 previous confirmation games, retaining original code."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import fcntl,json,time,hashlib
from pathlib import Path
import run_frozen_d_closed_loop as original
from audit_frozen_d_search_interface import warmup

ROOT=original.ROOT
SOURCE=original.OUT
OUT=ROOT/'current/l45_search_diagnosis_repeat_20261026'


def scientific_trace(trace):
    fields=('step','state','next_state','observation','own_action','opponent_raw','opponent_action',
        'opponent_shield','particles','weights','estimator_sha256','search_rng_before','search_rng_after',
        'post_seen','capture','own_legal','opponent_legal','simulated_transitions','simulation_errors','simulation_sha256')
    stats=('simulations','root_counts','terminations','root_visits','root_totals','root_candidates','source','planning_depth')
    return [{k:row[k] for k in fields}|dict(statistics={k:row['statistics'][k] for k in stats}) for row in trace]


def inputs(task):
    entry=task['entry']
    return (json.loads((SOURCE/'bank'/entry['public']).read_text()),
        json.loads((SOURCE/'bank'/entry['environment']).read_text()))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((SOURCE/'experiment.json').read_text())
    assert all(original.sha(p)==h for p,h in (baseline['sources_and_inputs']|baseline['protected_before']).items())
    requests=original.tasks('confirmation')
    sources=[Path(__file__),ROOT/'code/L45_SEARCH_DIAGNOSIS_REPEAT_20261026.md']
    sources += [ROOT/'code/diagnose_frozen_d_root_rollout.py']
    config=dict(role='exact repeat of previous closed-loop confirmation; not a new independent test',
        games=144,public_groups=24,seeds_changed=False,controller_changed=False,estimator_changed=False,
        budget_ms=200.,baseline_experiment_sha256=original.sha(SOURCE/'experiment.json'),
        baseline_results_sha256=original.sha(SOURCE/'confirmation/results.json'),
        sources={str(p):original.sha(p) for p in sources},
        baseline_source_and_input_hashes=baseline['sources_and_inputs'],
        protected_before=baseline['protected_before'],training=False,deployment_changed=False)
    path=OUT/'experiment.json'
    if path.exists():assert json.loads(path.read_text())==config,'Frozen repeat protocol changed'
    else:original.save(path,config)
    if (OUT/'repeat_results.json').exists():raise RuntimeError('Completed repeat is immutable')
    original.OUT=OUT;original.inputs=inputs
    started=time.perf_counter();warmup()
    original.save(OUT/'warmup.json',dict(seconds=time.perf_counter()-started))
    rows=[];comparisons=[];started=time.perf_counter()
    for task in requests:
        row=original.cached(task)
        if row is None:row=original.commit(task,original.episode(task))
        actual=json.loads(Path(row['episode']).read_text())
        prior=json.loads((SOURCE/'confirmation/episodes'/(task['key']+'.json')).read_text())
        a,b=scientific_trace(actual['trace']),scientific_trace(prior['trace'])
        mismatch=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
        same=a==b and actual['summary']['failure']==prior['summary']['failure']
        comparisons.append(dict(key=task['key'],exact=same,first_mismatch=mismatch,old_steps=len(b),new_steps=len(a),
            new_scientific_sha256=hashlib.sha256(json.dumps(a,sort_keys=True).encode()).hexdigest(),
            old_scientific_sha256=hashlib.sha256(json.dumps(b,sort_keys=True).encode()).hexdigest()))
        rows.append(row)
        status=dict(stage='repeat',completed=len(rows),total=144,exact=sum(c['exact'] for c in comparisons),
            elapsed_seconds=time.perf_counter()-started,pid=os.getpid(),training=False)
        original.save(OUT/'status.json',status);print(json.dumps(status),flush=True)
    original.save(OUT/'repeat_results.json',rows)
    original.save(OUT/'repeat_statistics.json',original.statistics(rows))
    original.save(OUT/'repeat_comparisons.json',dict(all_exact=all(c['exact'] for c in comparisons),rows=comparisons,
        effective_public_groups=24,not_additional_independent_samples=True))
    assert all(original.sha(p)==h for p,h in (baseline['sources_and_inputs']|baseline['protected_before']).items())
    original.save(OUT/'repeat_integrity.json',dict(passed=True,estimator_valid=bool(original.loader.verify()),
        original_sources_and_data_unchanged=True,training=False,deployment_changed=False))
    original.save(OUT/'status.json',dict(stage='repeat_complete_diagnosis_pending',completed=144,total=144,pid=None))


if __name__=='__main__':
    with (ROOT/'current/.l45_local_pursuit_serial.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
