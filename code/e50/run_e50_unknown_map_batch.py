"""Batch strict unknown-map rolling audits over frozen E49 validation cases."""
import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
import repo_paths
repo_paths.install()
import argparse
import json
from pathlib import Path

from run_e50_rolling_smoke import run


def batch(cases, output, cycles=4, speed=1.5, acceleration=6.0,
          command_delay=0, drop_cycles=(), pose_delay=0, moving_evader=False,
          rootpuct_evader=True, rootpuct_simulations=20, methods=None):
    rows = []
    for case in cases:
        case = Path(case)
        result = run(case, Path(output).parent / (case.stem + '_unknown.json'),
                     cycles=cycles, pursuer_speed=speed,
                     pursuer_acceleration=acceleration, unknown_map=True,
                     command_delay=command_delay, drop_cycles=drop_cycles,
                     pose_delay=pose_delay, moving_evader=moving_evader,
                     rootpuct_evader=rootpuct_evader,
                     rootpuct_simulations=rootpuct_simulations,
                     methods=methods)
        for row in result['rows']:
            cycles_out = row.get('cycles', [])
            rows.append(dict(case=case.stem, method=row['method'], valid=row['valid'],
                             error=row.get('error'), cycles=len(cycles_out),
                             executed_collisions=sum(c.get('collision', False) for c in cycles_out),
                             planned_collisions=sum(c.get('planned_trajectory_collision', False) for c in cycles_out),
                             safety_overrides=sum(c.get('safety_override') is not None for c in cycles_out),
                             local_projections=sum(c.get('safety_override') == 'local_action_projection' for c in cycles_out),
                             future_stops=sum(c.get('safety_override') == 'stop_future_collision' for c in cycles_out),
                             stop_degraded=sum(c.get('communication_event') == 'stop_degraded' for c in cycles_out),
                             packet_dropped=sum(c.get('communication_event') == 'packet_dropped' for c in cycles_out),
                             infeasible=sum(not c.get('feasible', False) for c in cycles_out),
                             captures=sum(len(c.get('capture_events', [])) for c in cycles_out),
                             planning_ms_mean=(sum(c.get('planning_ms', 0.) for c in cycles_out) / len(cycles_out)
                                               if cycles_out else 0.),
                             minco_max_speed_peak=max((c.get('minco_max_speed', 0.) for c in cycles_out), default=0.),
                             minco_max_acceleration_peak=max((c.get('minco_max_acceleration', 0.) for c in cycles_out), default=0.),
                             executed_distance=sum(c.get('executed_distance', 0.) for c in cycles_out)))
    result = dict(stage='unknown_map_batch', cases=[str(x) for x in cases], cycles=cycles,
                  pursuer_speed=speed, pursuer_acceleration=acceleration,
                  command_delay=command_delay, drop_cycles=list(drop_cycles),
                  pose_delay=pose_delay,
                  moving_evader=bool(moving_evader),
                  rootpuct_evader=bool(rootpuct_evader),
                  rootpuct_simulations=int(rootpuct_simulations),
                  methods=methods,
                  unknown_map_claim=True, rows=rows)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cycles', type=int, default=4)
    parser.add_argument('--pursuer-speed', type=float, default=1.5)
    parser.add_argument('--pursuer-acceleration', type=float, default=6.)
    parser.add_argument('--command-delay', type=int, default=0)
    parser.add_argument('--drop-cycles', type=int, nargs='*', default=[])
    parser.add_argument('--pose-delay', type=int, default=0)
    parser.add_argument('--moving-evader', action='store_true')
    parser.add_argument('--no-rootpuct-evader', action='store_true')
    parser.add_argument('--rootpuct-simulations', type=int, default=20)
    parser.add_argument('--methods', nargs='*', default=None)
    args = parser.parse_args()
    print(json.dumps(batch(args.cases, args.output, args.cycles,
                           args.pursuer_speed, args.pursuer_acceleration,
                           args.command_delay, args.drop_cycles, args.pose_delay,
                           args.moving_evader, not args.no_rootpuct_evader,
                           args.rootpuct_simulations, args.methods),
                     ensure_ascii=False, indent=2))
