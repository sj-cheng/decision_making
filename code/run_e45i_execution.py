"""Execution-only calibrated evaluation concurrency amendment for E45-I."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import run_e45i_baselines as run
from run_e44_parallel import choose_cpus
from run_frozen_d_closed_loop import save,sha


def calibrate(args):
    run.configure(args.output);run.verify();cpus=choose_cpus(24)
    entries=run.read(run.OUT/'bank/manifest.json')['validation'][::2]
    def tasks(stage):
        result=[]
        for e in entries:
            for arm in ('SINGLE_PRIOR','U','GEOM_SEARCH'):
                checkpoint=run.CROOT/'learning/x20/round_12/models/single_17.npz' if arm=='SINGLE_PRIOR' else None
                result.append(run.request(e,'AB','fixed_D_kernel',arm,17 if checkpoint else 0,stage,checkpoint)|dict(limit=1))
        return result
    serial=run.evaluate(tasks('calibration_serial'),cpus[:1],'calibration_serial')
    selected=None;measurements=[]
    for workers in (24,16,12,8):
        parallel=run.evaluate(tasks(f'calibration_parallel_{workers}'),cpus[:workers],f'calibration_parallel_{workers}')
        ratio=sum(r['simulated_transitions'] for r in parallel)/max(1,sum(r['simulated_transitions'] for r in serial))
        by_arm={arm:sum(r['simulated_transitions'] for r in parallel if r['arm']==arm)/max(1,sum(r['simulated_transitions'] for r in serial if r['arm']==arm)) for arm in ('SINGLE_PRIOR','U','GEOM_SEARCH')}
        cpu_ratio=float(np.mean([sum(r['cpu_ms'])/max(1e-9,sum(r['latencies_ms'])) for r in parallel]))
        measure=dict(workers=workers,total_simulation_ratio=ratio,by_arm=by_arm,cpu_wall_ratio=cpu_ratio,
            serial_keys=[r['key'] for r in serial],parallel_keys=[r['key'] for r in parallel])
        measurements.append(measure)
        if ratio>=.75 and min(by_arm.values())>=.60 and cpu_ratio>=.80:
            selected=workers;break
    if selected is None:raise RuntimeError('No calibrated concurrency passed')
    cached={str(p):sha(p) for p in (run.OUT/'static_test/cache').glob('*.json')}
    record=dict(type='execution-only',original_experiment_sha256=sha(run.OUT/'experiment.json'),
        wrapper_sha256=sha(Path(__file__)),evaluation_workers=selected,
        calibration='validation maps only, first decision, same seeds and 200ms budget; no wins used',
        measurements=measurements,existing_static_caches=cached,
        algorithm_reward_models_seeds_tasks_and_budget_unchanged=True,
        note='reuse completed 8-worker episodes; remaining episodes use calibrated concurrency; timing is execution-specific')
    save(run.OUT/'execution_amendment.json',record)
    print(json.dumps(record|dict(existing_static_caches=len(cached)),ensure_ascii=False),flush=True)


def pipeline(args):
    run.configure(args.output);run.verify();amend=run.read(run.OUT/'execution_amendment.json')
    assert amend['original_experiment_sha256']==sha(run.OUT/'experiment.json')
    assert amend['wrapper_sha256']==sha(Path(__file__))
    for path,digest in amend['existing_static_caches'].items():
        assert sha(path)==digest,('Existing episode cache changed',path)
    original=run.evaluate
    def evaluate(tasks,cpus,label):
        actual=choose_cpus(amend['evaluation_workers'])
        save(run.OUT/f'{label}_execution.json',dict(workers=len(actual),cpus=actual,
            execution_amendment_sha256=sha(run.OUT/'execution_amendment.json')))
        return original(tasks,actual,label)
    run.evaluate=evaluate
    run.pipeline(argparse.Namespace(steps=run.read(run.OUT/'experiment.json')['training_steps_per_run']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['calibrate','pipeline'])
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.command=='calibrate':calibrate(args)
    else:pipeline(args)


if __name__=='__main__':main()
