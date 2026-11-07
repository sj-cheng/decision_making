"""Verify frozen audit evidence and measure complete call-return overhead.

This is an additional timing check of the same development inputs, not an
independent efficacy confirmation. Both timing runs are retained.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import hashlib,json,time,fcntl
from audit_frozen_d_search_interface import ROOT,SOURCE,OUTPUT,observation,quantiles,save,warmup
from solvers.frozen_d_search_interface import Controller,METHODS,loader


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=OUTPUT/'external_timing.json'
    if p.exists():raise RuntimeError('Complete-return timing evidence cannot be overwritten')
    experiment=json.loads((OUTPUT/'experiment.json').read_text())
    for name,h in (experiment['sources']|experiment['inputs']|experiment['deployment_before_sha256']).items():
        if sha(name)!=h:raise RuntimeError('Source or input changed: '+name)
    archive=ROOT/'current/l45_weighted_search_interface_20261026/source_snapshot.json'
    for name,entry in json.loads(archive.read_text()).items():
        if sha(entry['snapshot'])!=entry['sha256']:raise RuntimeError('Original audit source archive changed: '+name)
    loader.verify()
    save(OUTPUT/'external_timing_protocol.json',dict(purpose='account for return and diagnostic-statistics assembly',
        source_sha256=sha(__file__),original_results_sha256=sha(OUTPUT/'results.json'),
        input_protocol_sha256=sha(OUTPUT/'experiment.json'),estimator_changed=False,
        no_efficacy_games=True,independent_confirmation=False,groups=24,methods=list(METHODS),
        retain_original_timing=True,read_results_before_additional_timing=True))
    warmup()
    entries=json.loads((SOURCE/'bank/manifest.json').read_text())['entries'];rows=[]
    for i,entry in enumerate(entries):
        packet=json.loads((SOURCE/'bank'/entry['public']).read_text())
        for method in METHODS[i%3:]+METHODS[:i%3]:
            c=Controller(packet['map'],packet['motion']['pursuer_axis_limit'],method,73020261025+i)
            for item in packet['history']:
                obs=observation(item)
                started=time.perf_counter()
                c.act(obs)
                elapsed=1000*(time.perf_counter()-started)
                s=c.last_stats
                kind='visible' if obs.target is not None else 'occlusion_build' if s['build_ms']>0 else 'ordinary_hidden'
                rows.append(dict(group=entry['id'],method=method,step=obs.step,kind=kind,
                    complete_return_ms=elapsed,internal_action_ready_ms=s['total_ms'],
                    diagnostic_return_overhead_ms=elapsed-s['total_ms']))
        print(json.dumps(dict(stage='external_return_timing',groups=i+1,total=24)),flush=True)
    summary={method:{kind:quantiles([r['complete_return_ms'] for r in rows if r['method']==method and r['kind']==kind])
        for kind in ('visible','ordinary_hidden','occlusion_build')} for method in METHODS}
    save(p,dict(decisions=len(rows),independent_confirmation=False,summary=summary,rows=rows,
        return_overhead=quantiles([r['diagnostic_return_overhead_ms'] for r in rows]),
        budget_passed=all(summary[m][k]['p95_ms']<=50 for m in METHODS for k in ('ordinary_hidden','occlusion_build'))))
    save(OUTPUT/'source_integrity_checks.json',dict(passed=True,frozen_estimator=bool(loader.verify()),
        audit_sources_and_inputs_unchanged=all(sha(n)==h for n,h in (experiment['sources']|experiment['inputs']).items()),
        deployment_unchanged=all(sha(n)==h for n,h in experiment['deployment_before_sha256'].items()),
        original_v1_source_snapshots_valid=True))


if __name__=='__main__':
    with (ROOT/'current/.l45_local_pursuit_serial.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
