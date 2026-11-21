"""Apply deterministic communication/pose perturbations to a rolling smoke log."""
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
import numpy as np

from solvers.e50_distributed_audit import (DistributedAuditConfig,
                                           simulate_distributed_execution)


def run(rolling_json, output):
    source = json.loads(Path(rolling_json).read_text())
    rows = []
    scenarios = {
        'nominal': DistributedAuditConfig(),
        'delay_1': DistributedAuditConfig(command_delay=1, pose_delay=1),
        'drop_cycle_1': DistributedAuditConfig(drop_cycles=(1,)),
        'delay_drop': DistributedAuditConfig(command_delay=1, pose_delay=1,
                                             drop_cycles=(1,)),
    }
    for row in source.get('rows', []):
        if not row.get('valid'):
            rows.append(dict(method=row.get('method'), valid=False,
                             error=row.get('error', 'rolling smoke invalid')))
            continue
        cycles = row.get('cycles', [])
        commands = np.asarray([c['action'] for c in cycles], dtype=float)
        poses = np.asarray([c['midpoint_position'] for c in cycles], dtype=float)
        method = dict(method=row['method'], valid=True, cycles=len(cycles), scenarios={})
        for name, config in scenarios.items():
            result = simulate_distributed_execution(commands, poses, config)
            events = list(result.events)
            method['scenarios'][name] = dict(
                command_delay=config.command_delay,
                pose_delay=config.pose_delay,
                drop_cycles=list(config.drop_cycles),
                stop_degraded=sum(e['event'] == 'stop_degraded' for e in events),
                packet_dropped=sum(e['event'] == 'packet_dropped' for e in events),
                applied_commands=sum(e['event'] == 'command_applied' for e in events),
                max_command=float(np.max(np.abs(result.commands))) if len(result.commands) else 0.,
            )
        rows.append(method)
    result = dict(stage='distributed_execution_smoke', source=str(rolling_json),
                  unknown_map_claim=False, rows=rows)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rolling-json', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.rolling_json, args.output), ensure_ascii=False, indent=2))
