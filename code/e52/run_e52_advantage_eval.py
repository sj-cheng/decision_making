"""Four-arm paired evaluation on test map families in two communication profiles."""
import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
import repo_paths
repo_paths.install()
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from solvers.e50_distributed_geometry import SHARED_SPEED_LIMIT, SHARED_ACCELERATION_LIMIT
from e52_nae_diagnostic_cases import DEFAULT_PRODUCTION_DIR
from run_e52_nae_diagnostics import aggregate, evaluate, write_json


ARMS = ('geometry', 'nar', 'both_residual', 'branch_advantage')
PROFILES = {'nominal': (0, ()), 'degraded': (2, (2, 3))}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    result = {}
    for profile in PROFILES:
        group = [r for r in rows if r['profile']==profile]
        baseline = {r['case']: r for r in group if r['arm']=='nar'}
        pairs = [(r, baseline[r['case']]) for r in group
                 if r['arm']=='branch_advantage' and r['case'] in baseline]
        delta = [a['capture_cycles'][1]-b['capture_cycles'][1] for a,b in pairs
                 if a['captures']==b['captures']==2]
        result[profile] = dict(
            by_arm=aggregate(group), complete_pairs=len(pairs),
            paired_double_capture_wins=sum(a['captures']==2 and b['captures']!=2 for a,b in pairs),
            paired_double_capture_losses=sum(b['captures']==2 and a['captures']!=2 for a,b in pairs),
            both_double_capture_pairs=len(delta), median_cycle_delta_vs_nar=median(delta) if delta else None,
            faster=sum(x<0 for x in delta), tied=sum(x==0 for x in delta), slower=sum(x>0 for x in delta),
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-manifest', type=Path, required=True)
    p.add_argument('--advantage', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=36)
    opponent = p.add_mutually_exclusive_group()
    opponent.add_argument('--rootpuct-simulations', type=int, default=20,
                   help='evader tree-search simulations per decision')
    opponent.add_argument('--geometric-evader-speed', type=float,
                          help='geometric evader speed cap, per axis for straight-line motion')
    p.add_argument('--evader-minco', action='store_true',
                   help='execute the geometric evader with MINCO and a Euclidean speed cap')
    opponent.add_argument('--shared-max-speed', type=float, help='planar speed limit for both MINCO teams (m/s)')
    p.add_argument('--max-acceleration', type=float,
                   help='shared continuous planar acceleration cap in m/s² (requires --evader-minco)')
    p.add_argument('--fixed-budget', action='store_true',
                   help='use the pursuer search count limits without a wall-clock cutoff')
    p.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    a = p.parse_args()
    if a.rootpuct_simulations < 1:
        p.error('--rootpuct-simulations must be positive')
    speed = a.geometric_evader_speed
    if a.evader_minco and speed is None and a.shared_max_speed is None:
        a.shared_max_speed = SHARED_SPEED_LIMIT
    if a.shared_max_speed is not None:
        if not a.evader_minco:
            p.error('--shared-max-speed requires --evader-minco')
        speed = a.shared_max_speed
        if a.max_acceleration is None:
            a.max_acceleration = SHARED_ACCELERATION_LIMIT
    if speed is not None and (not math.isfinite(speed) or speed <= 0):
        p.error('--geometric-evader-speed must be positive and finite')
    if a.max_acceleration is not None:
        if not a.evader_minco:
            p.error('--max-acceleration requires --evader-minco')
        if not math.isfinite(a.max_acceleration) or a.max_acceleration <= 0:
            p.error('--max-acceleration must be positive and finite')
    manifest = json.loads(a.dataset_manifest.read_text())
    test_cases = [Path(r['path']) for r in manifest['cases'] if r['split']=='test']
    if not test_cases:
        raise SystemExit('no held-out test cases')
    old = DEFAULT_PRODUCTION_DIR/'frozen_nar_nae.pt'
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output/'manifest.json').exists():
        raise SystemExit('use a fresh evaluation directory')
    write_json(a.output/'manifest.json',dict(test_cases=list(map(str,test_cases)),
        profiles=PROFILES, arms=list(dict.fromkeys(a.arms)), cycles=160, budget=256, depth=3,
        rootpuct_simulations=a.rootpuct_simulations if speed is None else None,
        opponent_policy='rootpuct' if speed is None else 'geometric',
        evader_speed_per_axis=None if a.evader_minco else (1.5 if speed is None else speed),
        evader_minco=a.evader_minco, evader_max_speed=speed if a.evader_minco else None,
        shared_max_speed=a.shared_max_speed, max_acceleration=a.max_acceleration,
        fixed_budget=a.fixed_budget,
        deadline_ms=None if a.fixed_budget else 200,
        checkpoint_sha256=digest(a.advantage), old_checkpoint_sha256=digest(old)))
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs = {}
        for profile,(delay, drops) in PROFILES.items():
            for case in test_cases:
                for arm in dict.fromkeys(a.arms):
                    target = a.output/profile/f'{case.stem}_{arm}.json'
                    job = pool.submit(evaluate, arm, case, target, old, 160,
                                      advantage=a.advantage if arm=='branch_advantage' else None,
                                      delay=delay, drops=drops,
                                      rootpuct_simulations=a.rootpuct_simulations, fixed_budget=a.fixed_budget,
                                      geometric_evader_speed=speed, evader_minco=a.evader_minco,
                                      max_acceleration=a.max_acceleration, shared_max_speed=a.shared_max_speed)
                    jobs[job] = profile
        for job in as_completed(jobs):
            row = job.result()
            row['profile'] = jobs[job]
            rows.append(row)
            write_json(a.output/'summary.json',dict(completed=len(rows), total=len(jobs),
                                                      results=summarize(rows), rows=rows))
            print(len(rows), len(jobs), row['profile'], row['arm'], row['case'],
                  'captures',row['captures'],flush=True)


if __name__=='__main__':
    main()
