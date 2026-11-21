"""Receipt for the hardware-neutral runtime input/degradation boundary."""
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

from solvers.e50_interfaces import BeliefState, LidarScan
from solvers.e50_runtime_adapter import E50RuntimeAdapter


def make_belief(stamp):
    return BeliefState(np.zeros((2, 2, 2)), np.full((2, 2), .5),
                       np.array([False, False]), np.array([.2, .2]), stamp)


def run(output):
    angles = np.linspace(-np.pi, np.pi, 16, endpoint=False)
    cases = []
    adapter = E50RuntimeAdapter()
    scan = LidarScan('base', 1., angles, np.full(16, 5.), 5.)
    adapter.update_poses(1., np.zeros((2, 2)))
    adapter.update_scan(0, scan); adapter.update_scan(1, scan); adapter.update_belief(make_belief(1.))
    cases.append(dict(case='live', snapshot=adapter.snapshot(1.05)))
    cases.append(dict(case='stale', snapshot=adapter.snapshot(1.5)))
    adapter.update_belief(make_belief(1.3))
    cases.append(dict(case='timestamp_skew', snapshot=adapter.snapshot(1.)))
    fresh = E50RuntimeAdapter(); cases.append(dict(case='missing_input', snapshot=fresh.snapshot(1.)))
    rows = []
    command = np.ones((2, 2))
    for item in cases:
        snap = item['snapshot']; rows.append(dict(case=item['case'], degraded=snap.degraded,
                                                  reason=snap.reason, stamp=snap.stamp,
                                                  requested_command=command.tolist(),
                                                  gated_command=adapter.gate_command(snap, command).tolist(),
                                                  record=adapter.record(snap, command,
                                                                        adapter.gate_command(snap, command))))
    result = dict(stage='runtime_adapter_smoke', rows=rows)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n'); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
