"""Execution entry for E45-F's separately recorded geometry-buffer amendment."""
from pathlib import Path
import run_e45f_generalization_benchmark as f
import solvers.e45_team_adapter as adapter
from solvers.e45f_geometry_capacity import geometric_candidates_capacity

FORMAL=f.ROOT/'current/e45f_generalization_benchmark_20261029'
AMENDMENT=FORMAL/'GEOMETRY_CAPACITY_COMPAT_20261029.json'
ORIGINAL_VERIFY=f.verify
ORIGINAL_REPORT=f.report


def verify():
    config=ORIGINAL_VERIFY();repair=f.read(AMENDMENT)
    if repair['original_experiment_sha256']!=f.sha(FORMAL/'experiment.json'):
        raise RuntimeError('Capacity amendment has wrong parent')
    for path,digest in repair['source_hashes'].items():
        if f.sha(path)!=digest:raise RuntimeError('Capacity amendment changed: '+path)
    for path,digest in repair['evidence_hashes'].items():
        if f.sha(path)!=digest:raise RuntimeError('Capacity regression evidence changed: '+path)
    return config


def report(rows=None,stats=None):
    ORIGINAL_REPORT(rows,stats)
    path=f.OUT/'REPORT.md'
    with path.open('a') as output:
        output.write('\n[几何候选容量兼容修订]('+str(FORMAL/'GEOMETRY_CAPACITY_COMPAT_20261029.json')+')：原64候选数组在密集校准地图不足，正式对局前改为9＋样本数＋地图角点数的足够容量，枚举/评分/动作规则保持原实现。原始预热失败现场与回归核对均保留。\n')


# Spawned workers import this entry too, before invoking f.worker_init.
adapter.geometric_candidates=geometric_candidates_capacity
f.verify=verify
f.report=report

if __name__=='__main__':f.main()
