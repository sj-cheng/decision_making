"""Convert a full E50 rolling run into the existing trajectory-atlas format."""
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


METHOD_LABELS = {
    'A2': 'A2 — geometry ranking + search',
    'A3': 'A3 — L5 learned ranking + search',
    'A4': 'A4 — geometry ranking, direct',
    'AG': 'AG — assigned geometry',
    'GEOM_SEARCH': 'GEOM_SEARCH — E46 reference',
    'RouteAPF': 'RouteAPF',
    'IG': 'IG — independent geometry',
    'SINGLE_PRIOR': 'SINGLE_PRIOR',
    'A5': 'A5 — L5 learned ranking, direct',
    'MAPPO_RouteAPF': 'MAPPO + RouteAPF',
    'PPO_C': 'PPO-C',
}


def convert(case_path, result_path, output_dir):
    case = json.loads(Path(case_path).read_text())
    result = json.loads(Path(result_path).read_text())
    output_dir = Path(output_dir)
    episodes = output_dir / 'episodes'; episodes.mkdir(parents=True, exist_ok=True)
    initial = np.asarray(case['worlds'][0]['initial'], dtype=float)
    family = 'e50_lidar_minco_dense00_AB'
    methods = []
    for row in result['rows']:
        if not row.get('valid'):
            continue
        method = row['method']; methods.append(method)
        state = initial.copy()
        states = [state.tolist()]
        observations = []
        trace = []
        for cycle in row.get('cycles', []):
            nxt = state.copy()
            midpoint = np.asarray(cycle['midpoint_position'], dtype=float)
            nxt[4:8] = midpoint.reshape(-1)
            target = np.asarray(cycle['evader_position'], dtype=float)
            nxt[:4] = target.reshape(-1)
            nxt[9:13] = np.r_[np.asarray(cycle.get('active_e', [1., 1.])),
                               np.asarray(cycle.get('active_p', [1., 1.]))]
            if 'midpoint_velocity' in cycle:
                nxt[17:21] = np.asarray(cycle['midpoint_velocity'], dtype=float).reshape(-1)
            if 'midpoint_acceleration' in cycle:
                nxt[25:29] = np.asarray(cycle['midpoint_acceleration'], dtype=float).reshape(-1)
            nxt[8] = state[8] + 1.
            states.append(nxt.tolist())
            observations.append(cycle.get('observation', {}))
            trace.append(dict(state=state.tolist(), next_state=nxt.tolist(),
                              action=cycle.get('action'),
                              controller_action=cycle.get('controller_action'),
                              requested_action=cycle.get('requested_action'),
                              capture_events=cycle.get('capture_events', []),
                              safety_override=cycle.get('safety_override'),
                              minco_max_speed=cycle.get('minco_max_speed'),
                              minco_max_acceleration=cycle.get('minco_max_acceleration'),
                              executed_trajectory_positions=cycle.get('executed_trajectory_positions'),
                              lidar_scans=cycle.get('lidar_scans')))
            state = nxt
        captures = sum(len(c.get('capture_events', [])) for c in row.get('cycles', []))
        payload = dict(method=method, label=METHOD_LABELS.get(method, method),
                       case_id=case['id'], family=family, suite='E50_LiDAR_MINCO',
                       map=case['map'], states=states, observations=observations,
                       summary=dict(captures=captures, double_capture=bool(np.all(np.asarray(state[9:11]) <= .5)),
                                    decisions=len(trace), safety_overrides=sum(t['safety_override'] is not None for t in trace),
                                    executed_collisions=sum(c.get('collision', False) for c in row.get('cycles', []))),
                       task=dict(method=method, source='E50 rolling LiDAR/MINCO',
                                  geometry_evader=bool(result.get('geometry_evader')),
                                  unknown_map=bool(result.get('unknown_map_claim')),
                                  lidar_rays=result.get('lidar_rays'),
                                  cross_cycle_smoothing=result.get('cross_cycle_smoothing')),
                       scope='Qualitative full 68-cycle E50 trajectory, converted for the existing atlas renderer',
                       trace=trace)
        (episodes / f'{family}__{method}.json').write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    (output_dir / 'selection.json').write_text(json.dumps(
        dict(entries=[dict(id=family, family=family, path=str(case_path), world='AB')],
             methods=methods, selected_before_outcomes=True,
             note='E50 full 68-cycle LiDAR/MINCO run converted to the existing atlas format'),
        ensure_ascii=False, indent=2) + '\n')
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(convert(args.case, args.result, args.output_dir))
