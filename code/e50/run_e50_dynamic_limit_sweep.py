"""Sweep E50 endpoint-trajectory feasibility under stronger chassis limits."""
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


def run(actions_json, output):
    from e49_rules import create_environment
    from solvers.e50_compat_smoke import replay_endpoint_semantic
    data = json.loads(Path(actions_json).read_text())
    case_path = Path(data['case']); case = json.loads(case_path.read_text())
    boxes = np.asarray(case['map'], float).reshape(-1, 2, 2)
    env = create_environment(boxes, horizon=int(case.get('horizon', 80)))
    state = np.asarray(case['worlds'][0]['initial'], float).reshape(-1, 1)
    rows = []
    for row in data['rows']:
        if not row.get('valid'):
            continue
        action = np.zeros((env.action_dim, 1)); a = np.asarray(row['action'], float)
        action[env.action_idxs[2], 0] = a[0]; action[env.action_idxs[3], 0] = a[1]
        for speed in (1., 1.5, 2., 2.5):
            for acceleration in (1.2, 2., 4., 6., 8., 10.):
                env.current_pursuer_speed_lim = speed
                env.update_action_lims()
                for robot in env.pursuers:
                    env.state_lims[env.acc_idxs[robot], :] = [[-acceleration, acceleration], [-acceleration, acceleration]]
                audit, feasible = replay_endpoint_semantic(env, state, action, dt=1., robots=(2, 3), sample_dt=.02)
                rows.append(dict(method=row['method'], speed=speed, acceleration=acceleration,
                                 feasible=bool(feasible), collision=bool(audit.collision),
                                 max_speed=audit.max_speed, max_acceleration=audit.max_acceleration))
    result = dict(stage='endpoint_dynamic_limit_sweep', source=str(actions_json), rows=rows,
                  note='Simulation parameter sweep only; values require real chassis measurement before deployment.')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--actions-json', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(json.dumps(run(args.actions_json, args.output), indent=2))
