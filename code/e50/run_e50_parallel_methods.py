"""Detached parallel launcher for the E50 per-method rolling runs."""
import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
import repo_paths
repo_paths.install()
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


METHODS = ('A2', 'A3', 'A4', 'AG', 'GEOM_SEARCH', 'RouteAPF',
           'IG', 'SINGLE_PRIOR', 'A5', 'MAPPO_RouteAPF', 'PPO_C')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--cycles', type=int, default=68)
    ap.add_argument('--fixed-simulations', type=int, default=16)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    py = os.environ.get('E50_PYTHON', '/home/sjc/miniconda3/envs/dm_env_py311/bin/python')
    env = dict(os.environ, PYTHONPATH='code', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
    jobs = []
    for method in METHODS:
        output = outdir / f'{method}.json'
        log_path = outdir / f'{method}.log'
        log = log_path.open('w')
        cmd = [py, 'code/e50/run_e50_rolling_smoke.py', '--case', str(args.case),
               '--output', str(output), '--cycles', str(args.cycles),
               '--unknown-map', '--geometry-evader', '--moving-evader',
               '--lidar-rays', '256', '--fixed-simulations', str(args.fixed_simulations),
               '--methods', method]
        proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        jobs.append((method, output, proc, log))

    status_path = outdir / 'parallel_status.json'
    status_path.write_text(json.dumps(dict(state='running', methods=list(METHODS),
                                           started=time.time()), indent=2))
    while jobs:
        time.sleep(30)
        left = []
        statuses = {}
        for method, output, proc, log in jobs:
            rc = proc.poll()
            statuses[method] = dict(pid=proc.pid, returncode=rc,
                                    output_exists=output.exists())
            if rc is None:
                left.append((method, output, proc, log))
            else:
                log.close()
        status_path.write_text(json.dumps(dict(state='running', statuses=statuses,
                                               updated=time.time()), indent=2))
        jobs = left
    rows = []
    for method in METHODS:
        path = outdir / f'{method}.json'
        if path.exists():
            payload = json.loads(path.read_text())
            rows.extend(payload.get('rows', []))
    merged = dict(stage='rolling_endpoint_lidar_planning_parallel', case=str(args.case),
                  cycles=int(args.cycles), fixed_simulations=int(args.fixed_simulations),
                  lidar_rays=256, methods=list(METHODS), rows=rows)
    (outdir / 'merged.json').write_text(json.dumps(merged, ensure_ascii=False, indent=2) + '\n')
    status_path.write_text(json.dumps(dict(state='complete', methods=list(METHODS),
                                           finished=time.time(), rows=len(rows)), indent=2))


if __name__ == '__main__':
    main()
