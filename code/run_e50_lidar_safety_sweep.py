"""Sweep LiDAR/safety parameters for the strict unknown-map rolling audit."""
import argparse
import json
from pathlib import Path

from run_e50_rolling_smoke import run


def sweep(case, output, cycles=12):
    rows = []
    for rays in (32, 64, 128):
        for inflation in (.02, .05, .10):
            path = Path(output).with_name(f'{Path(output).stem}_r{rays}_i{int(inflation*100):02d}.json')
            result = run(case, path, cycles=cycles, pursuer_speed=1.5,
                         pursuer_acceleration=6., unknown_map=True,
                         command_delay=1, drop_cycles=(1, 5), pose_delay=1,
                         moving_evader=True, lidar_rays=rays,
                         lidar_inflation=inflation, grid_inflation=inflation)
            for row in result['rows']:
                cycles_out = row.get('cycles', [])
                rows.append(dict(rays=rays, inflation=inflation, method=row['method'],
                                 valid=row['valid'], executed_collisions=sum(c['collision'] for c in cycles_out),
                                 safety_overrides=sum(c.get('safety_override') is not None for c in cycles_out),
                                 executed_distance=sum(c.get('executed_distance', 0.) for c in cycles_out),
                                 captures=sum(len(c.get('capture_events', [])) for c in cycles_out)))
    result = dict(stage='lidar_safety_parameter_sweep', case=str(case), cycles=cycles,
                  unknown_map_claim=True, rows=rows)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cycles', type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(sweep(args.case, args.output, args.cycles), ensure_ascii=False, indent=2))
