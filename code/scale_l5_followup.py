"""Engineering-only concurrency probe/launcher; frozen scientific code stays intact."""
import argparse
import copy
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import run_l5_followup as run

_BARRIER = None


def probe_init(out, queue, barrier):
    global _BARRIER
    run.worker_init(out, queue)
    _BARRIER = barrier


def probe_one(cfg):
    _BARRIER.wait(timeout=180)
    start = time.time()
    result = run.calibration_job({}, cfg)
    return dict(cpu=list(os.sched_getaffinity(0)), started=start,
                finished=time.time(), rows=result['rows'])


def expanded_layout(original, timing_cpus):
    layout = copy.deepcopy(original)
    timing_cpus = list(timing_cpus)
    # Resolve actual SMT siblings instead of assuming CPU numbering.
    sibling_cpus = set()
    core_keys = set()
    for cpu in timing_cpus:
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = ((base/'physical_package_id').read_text(), (base/'core_id').read_text())
        if key in core_keys:
            raise ValueError('Timing workers must use distinct physical cores')
        core_keys.add(key)
        for part in (base/'thread_siblings_list').read_text().strip().split(','):
            bounds = [int(x) for x in part.split('-')]
            sibling_cpus.update(range(bounds[0], bounds[-1]+1))
    if len(layout['reserved']) < 20 or sibling_cpus.intersection(layout['reserved']+layout['training']):
        raise ValueError('Timing CPUs overlap reserved/coordinator physical cores')
    layout['timing'] = timing_cpus
    layout['timing_siblings_idle'] = sorted(sibling_cpus-set(timing_cpus))
    layout['bulk'] = sorted(set(layout['bulk'])-sibling_cpus)
    return layout


def probe(out):
    cfg = run.read(out/'experiment.json'); run.verify(cfg)
    reference = run.read(out/'timing_serial_reference.json')
    ctx = mp.get_context('fork')
    report = dict(started=run.stamp(), scientific_config_sha256=run.sha(out/'experiment.json'),
                  source_sha256=run.sha(__file__), attempts=[], selected_workers=8,
                  scope='Fixed-work engineering calibration only; no new scientific episodes',
                  existing_workers_continue=8)
    for count in (16, 32):
        # Additional disjoint physical cores; the original eight keep running.
        cpus = list(range(16, 16+count))
        queue = ctx.Queue()
        for cpu in cpus: queue.put(cpu)
        barrier = ctx.Barrier(count)
        with ProcessPoolExecutor(count, mp_context=ctx, initializer=probe_init,
                                 initargs=(str(out), queue, barrier)) as pool:
            rows = list(pool.map(probe_one, [cfg]*count))
        samples = [v for r in rows for group in r['rows'] for v in group['samples_ms']]
        median = float(run.np.median(samples))/reference['median']
        p95 = float(run.np.percentile(samples, 95))/reference['p95']
        good = median <= 1.2 and p95 <= 1.3
        report['attempts'].append(dict(workers=count, probe_cpus=cpus,
             concurrent_scientific_workers=8, median_ratio=median, p95_ratio=p95,
             passed=good, rows=rows))
        if good: report['selected_workers'] = count
        report['updated'] = run.stamp()
        run.write(out/'concurrency_expansion_probe.json', report)
        print(dict(workers=count, median_ratio=median, p95_ratio=p95, passed=good), flush=True)
        if not good: break
    selected = report['selected_workers']
    report['selected_timing_cpus'] = list(range(48, 56))+list(range(16, 16+selected-8))
    expanded_layout(run.e46.resource_layout(), report['selected_timing_cpus'])
    report['finished'] = run.stamp()
    run.write(out/'concurrency_expansion_probe.json', report)


def launch(out):
    report = run.read(out/'concurrency_expansion_probe.json')
    if report['source_sha256'] != run.sha(__file__):
        raise RuntimeError('Expansion launcher changed since probe')
    if report['scientific_config_sha256'] != run.sha(out/'experiment.json'):
        raise RuntimeError('Scientific configuration changed since probe')
    gate = run.read(out/'timing_calibration.json')
    if not gate['passed'] or gate['workers'] != report['selected_workers']:
        raise RuntimeError('Expansion not recorded in timing gate')
    original_layout = run.e46.resource_layout
    run.e46.resource_layout = lambda: expanded_layout(original_layout(), report['selected_timing_cpus'])
    sys.argv = [str(Path(run.__file__)), '--output', str(out)]
    run.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('probe', 'launch'))
    parser.add_argument('--output', type=Path, default=run.DEFAULT)
    args = parser.parse_args()
    if args.mode == 'probe': probe(args.output)
    else: launch(args.output)
