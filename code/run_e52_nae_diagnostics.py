"""Paired fresh-start experiment isolating how the frozen NAE affects H-MCTS."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np

from e52_nae_diagnostic_cases import DEFAULT_PRODUCTION_DIR, generate_diagnostic_cases
from run_e50_distributed_dynamic_smoke import run


ARMS = {
    'geometry': ('geometry', 'legacy'),
    'nar': ('nar', 'legacy'),
    'both_legacy': ('both', 'legacy'),
    'both_tracked': ('both', 'tracked'),
    'both_residual': ('both', 'leaf_residual'),
    'branch_advantage': ('both', 'leaf_residual'),
}
DIAGNOSTIC_ARMS = tuple(arm for arm in ARMS if arm != 'branch_advantage')


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def evaluate(arm, case, output, checkpoint, cycles, *, advantage=None, delay=0, drops=(),
             rootpuct_simulations=20, fixed_budget=False, geometric_evader_speed=None, evader_minco=False,
             max_acceleration=None, shared_max_speed=None):
    mode, strategy = ARMS[arm]
    result = run(case, output, cycles=cycles, e52=True, lidar_rays=256,
                 e52_mode=mode, e52_checkpoint=checkpoint if mode != 'geometry' else None,
                 e52_nae_strategy=strategy, seed=17, rootpuct=geometric_evader_speed is None and not evader_minco,
                 rootpuct_simulations=rootpuct_simulations,
                 geometric_evader_speed=geometric_evader_speed,
                 evader_minco=evader_minco, max_acceleration=max_acceleration, shared_max_speed=shared_max_speed,
                 e52_time_ms=1e9 if fixed_budget else 200.,
                 delay=delay, drop_cycles=drops, e52_advantage_checkpoint=advantage)
    logs = result['logs']
    initial = json.loads(case.read_text())['worlds'][0]['initial'][4:8]
    positions = np.vstack([initial, [np.ravel(row['pursuers']) for row in logs]]).reshape(-1, 2, 2)
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    no_belief = []
    queries = []
    decision_ms = []
    for row in logs:
        for aid, search in row['search'].items():
            queries.append(search['nae_queries'])
            decision_ms.append(search['elapsed_ms'])
            chosen = row['plans'][aid]['candidate_id']
            if chosen >= 0:
                features = search['root'][chosen]['features']
                if features[8] == 0:
                    no_belief.append(features[2] < .05)
    capture_cycles = sorted(capture['cycle'] + 1 for capture in result['captures'])
    return {
        'case': case.stem, 'scene': case.parent.name, 'arm': arm, 'trace': str(output),
        'captures': len(capture_cycles), 'capture_cycles': capture_cycles,
        'executed_cycles': len(logs), 'agent_steps': distances.size,
        'nearly_stationary_steps': int((distances < .03).sum()),
        'no_belief_decisions': len(no_belief), 'no_belief_stops': sum(no_belief),
        'nae_queried_decisions': sum(q > 0 for q in queries),
        'decision_p95_ms': float(np.percentile(decision_ms, 95)),
        'collisions': sum(row['pursuer_collision'] for row in logs),
        'minco_infeasible': sum(not m['feasible'] for row in logs for m in row['minco'].values()),
        'rootpuct_failures': result['rootpuct_failures'],
        'terminal_opponent_skips': result['terminal_opponent_skips'],
    }


def aggregate(rows):
    summary = {}
    for arm in ARMS:
        group = [row for row in rows if row['arm'] == arm]
        if not group:
            continue
        doubles = [r['capture_cycles'][1] for r in group if r['captures'] == 2]
        summary[arm] = {
            'episodes': len(group), 'double_captures': len(doubles),
            'single_captures': sum(r['captures'] == 1 for r in group),
            'median_double_capture_cycle': median(doubles) if doubles else None,
            'stall_fraction': sum(r['nearly_stationary_steps'] for r in group) / sum(r['agent_steps'] for r in group),
            'no_belief_stops': sum(r['no_belief_stops'] for r in group),
            'no_belief_decisions': sum(r['no_belief_decisions'] for r in group),
            'nae_queried_fraction': sum(r['nae_queried_decisions'] for r in group) / sum(r['agent_steps'] for r in group),
            'max_episode_p95_ms': max(r['decision_p95_ms'] for r in group),
            **{key: sum(r[key] for r in group) for key in
               ('collisions', 'minco_infeasible', 'rootpuct_failures', 'terminal_opponent_skips')},
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--production', type=Path, default=DEFAULT_PRODUCTION_DIR)
    parser.add_argument('--workers', type=int, default=30)
    parser.add_argument('--cycles', type=int, default=160)
    parser.add_argument('--case-seed', type=int, default=521119)
    parser.add_argument('--arms', nargs='+', choices=DIAGNOSTIC_ARMS, default=DIAGNOSTIC_ARMS)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise SystemExit(f'Use a fresh output directory: {output}')
    checkpoint = (args.production / 'frozen_nar_nae.pt').resolve()
    arms = dict.fromkeys(args.arms)
    cases, case_manifest = generate_diagnostic_cases(
        output / 'cases', production_dir=args.production, seed=args.case_seed)
    manifest = dict(cases=case_manifest, arms={arm: ARMS[arm] for arm in arms}, checkpoint=str(checkpoint),
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    cycles=args.cycles, seed=17, workers=args.workers)
    write_json(output / 'manifest.json', manifest)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [
            pool.submit(evaluate, arm, case, output/'traces'/f'{case.stem}_{arm}.json', checkpoint, args.cycles)
            for case in cases for arm in arms
        ]
        for job in as_completed(jobs):
            result = job.result()
            rows.append(result)
            summary = dict(completed=len(rows), total=len(jobs), by_arm=aggregate(rows), rows=rows)
            write_json(output/'summary.json', summary)
            print(json.dumps(result), flush=True)
    print(json.dumps(summary['by_arm'], indent=2), flush=True)


if __name__ == '__main__':
    main()
