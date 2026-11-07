"""Explicitly authorized SMT concurrency calibration, separate from science."""
import argparse
import copy
import multiprocessing as mp
import os
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor

import scale_l5_followup as scale

run = scale.run


def timing_cpus(count):
    if count == 32:
        return list(range(48, 56))+list(range(16, 40))
    if count not in (64, 80):
        raise ValueError('Only explicitly requested 64/80 and existing 32 control')
    return list(range(16, 56))+list(range(72, 72+count-40))


def resource_layout(original, count):
    cpus = timing_cpus(count)
    # First isolate all physical cores, then explicitly enable calibrated SMT.
    physical = sorted(set(c % 56 for c in cpus))
    result = scale.expanded_layout(original, physical)
    result['timing'] = cpus
    occupied = set(physical+[c+56 for c in physical])
    result['timing_siblings_idle'] = sorted(occupied-set(cpus))
    result['smt_enabled'] = count > len(physical)
    assert not occupied.intersection(result['reserved']+result['training'])
    assert len(result['reserved']) >= 20 and len(cpus) == count
    return result


def probe(out):
    cfg = run.read(out/'experiment.json'); run.verify(cfg)
    reference = run.read(out/'timing_serial_reference.json')
    report = dict(started=run.stamp(), source_sha256=run.sha(__file__),
                  helper_sha256=run.sha(scale.__file__),
                  scientific_config_sha256=run.sha(out/'experiment.json'),
                  selected_workers=32, attempts=[],
                  scope='Engineering only, scientific coordinator drained and stopped; no scientific results inspected')
    ctx = mp.get_context('fork')
    for count in (32, 64, 80):
        cpus = timing_cpus(count)
        resource_layout(run.e46.resource_layout(), count)
        queue = ctx.Queue()
        for cpu in cpus: queue.put(cpu)
        barrier = ctx.Barrier(count)
        with ProcessPoolExecutor(count, mp_context=ctx, initializer=scale.probe_init,
                                 initargs=(str(out), queue, barrier)) as pool:
            rows = list(pool.map(scale.probe_one, [cfg]*count))
        samples = [v for r in rows for g in r['rows'] for v in g['samples_ms']]
        median = float(run.np.median(samples))/reference['median']
        p95 = float(run.np.percentile(samples, 95))/reference['p95']
        good = median <= 1.2 and p95 <= 1.3
        report['attempts'].append(dict(workers=count, cpus=cpus, median_ratio=median,
                                      p95_ratio=p95, passed=good, rows=rows))
        if good: report['selected_workers'] = count
        report['updated'] = run.stamp()
        run.write(out/'smt_scaling_probe.json', report)
        print(dict(workers=count, median_ratio=median, p95_ratio=p95, passed=good), flush=True)
    report['finished'] = run.stamp()
    run.write(out/'smt_scaling_probe.json', report)


def launch(out):
    report = run.read(out/'smt_scaling_probe.json')
    assert report['source_sha256'] == run.sha(__file__)
    assert report['helper_sha256'] == run.sha(scale.__file__)
    assert report['scientific_config_sha256'] == run.sha(out/'experiment.json')
    count = report['selected_workers']
    assert run.read(out/'timing_calibration.json')['workers'] == count
    original = run.e46.resource_layout
    run.e46.resource_layout = lambda: resource_layout(original(), count)
    sys.argv = [run.__file__, '--output', str(out)]
    run.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('probe', 'launch'))
    parser.add_argument('--output', type=Path, default=run.DEFAULT)
    args = parser.parse_args()
    if args.mode == 'probe': probe(args.output)
    else: launch(args.output)
