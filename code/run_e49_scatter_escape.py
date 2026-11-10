"""Same 16 scatter cases and 11 methods, opponent swapped to geometry escape.

Diagnostic only. Cases, method checkpoints and the fixed 64 simulations come
from the no-budget scatter run; the only change is the evader policy. The
environment keeps its 1.5 evader limit; --speeds caps the escape steps only,
so the pursuers' 1.5 target-motion model still covers every evader move.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import json
from pathlib import Path

import numpy as np

import run_e49_benchmark as bench
import run_e49_scatter_left as scatter

OUT = scatter.ROOT/'current/e49_scatter_escape_20261110'
SPEED_KEY = 'E49_SCATTER_ESCAPE_SPEED'


def stage_for(speed):
    return 'scatter' if speed == 1.5 else f'speed_{speed:g}'


def make_escape(env, seed, simulations=3000):
    from solvers.scatter_escape_evader import ScatterEscapeEvader
    speed = float(os.environ[SPEED_KEY])
    if speed > env.current_evader_speed_lim:
        raise RuntimeError('Escape speed exceeds the environment evader limit')
    return ScatterEscapeEvader(speed)


def worker_init(queue):
    import e49_rules
    e49_rules.make_opponent = make_escape
    _original_init(queue)


_original_init = bench.worker_init


def tasks(cases, speed):
    for task in scatter.tasks(cases):
        task.update(stage=stage_for(speed), opponent=f'geometry_escape_{speed:g}')
        yield task


def batch(tasks, cpus):
    """Like bench.batch, but an engineering failure is kept and reported, not fatal."""
    import concurrent.futures as cf
    import multiprocessing as mp
    pending = [task for task in tasks
               if not (OUT/'results'/task['stage']/(bench.identity(task)+'.json')).exists()]
    if not pending:
        return
    stage = pending[0]['stage']
    context = mp.get_context('spawn')
    queue = context.Queue()
    for cpu in cpus:
        queue.put(cpu)
    done_count = 0
    with cf.ProcessPoolExecutor(len(cpus), mp_context=context, initializer=worker_init, initargs=(queue,)) as pool:
        futures = {pool.submit(bench.execute, task, str(OUT)): task for task in pending}
        for future in cf.as_completed(futures):
            row = future.result(); done_count += 1
            bench.event(OUT, stage, scheduled=len(pending), completed=done_count,
                        last_method=futures[future]['arm'], last_valid=row['valid'], last_seconds=row['seconds'])


def summarize(speed):
    rows = [bench.read(path) for path in (OUT/'results'/stage_for(speed)).glob('*.json')]
    ok = [row for row in rows if row.get('valid')]
    table = {}
    for arm in sorted({row['task']['arm'] for row in ok}):
        own = [row for row in ok if row['task']['arm'] == arm]

        def seeded(key, keep=lambda row: True):
            means = []
            for seed in sorted({row['task']['train_seed'] for row in own}):
                part = [row for row in own if row['task']['train_seed'] == seed and keep(row)]
                means.append(float(np.mean([key(row) for row in part])))
            return float(np.mean(means))
        table[arm] = dict(episodes=len(own),
                          double_capture=seeded(lambda r: float(r['summary']['double_capture'])),
                          at_least_one=seeded(lambda r: float(r['summary']['at_least_one'])),
                          decisions=seeded(lambda r: r['summary']['decisions']),
                          open=seeded(lambda r: float(r['summary']['double_capture']), lambda r: 'open' in r['task']['group']),
                          dense=seeded(lambda r: float(r['summary']['double_capture']), lambda r: 'dense' in r['task']['group']))
    failures = [dict(arm=row['task']['arm'], group=row['task']['group'], train_seed=row['task']['train_seed'],
                     error=row['error']) for row in rows if not row.get('valid')]
    bench.write(OUT/f'summary_{stage_for(speed)}.json', dict(speed=speed, valid=len(ok), total=len(rows),
                                                            engineering_failures=failures, methods=table))
    return len(ok), len(rows), table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--speeds', type=float, nargs='+', default=[1.5])
    args = parser.parse_args()
    index = bench.read(scatter.OUT/'map_index.json')
    OUT.mkdir(parents=True, exist_ok=True)
    for speed in args.speeds:
        cpus = scatter.free_cpus(args.workers)
        bench.write(OUT/f'workers_{stage_for(speed)}.json', dict(cpus=cpus, cases=str(scatter.OUT/'map_index.json')))
        os.environ[SPEED_KEY] = repr(speed)
        batch(list(tasks(index['cases'], speed)), cpus)
        valid, total, table = summarize(speed)
        print(f'speed {speed:g}: valid {valid} / {total}, engineering failures {total-valid}')
        for arm, row in sorted(table.items(), key=lambda item: -item[1]['double_capture']):
            print(f"  {arm:16s} double {row['double_capture']:6.1%}  one {row['at_least_one']:6.1%}  steps {row['decisions']:5.1f}")


if __name__ == '__main__':
    main()
