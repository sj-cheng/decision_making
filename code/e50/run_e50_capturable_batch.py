"""Diagnostic paired-state batch with history-consistent capturable initials."""
import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
import repo_paths
repo_paths.install()
import argparse
import copy
import json
from pathlib import Path
import numpy as np

from run_e50_rolling_smoke import run


METHODS = ('A2', 'A3', 'A4', 'AG', 'GEOM_SEARCH', 'RouteAPF',
           'IG', 'SINGLE_PRIOR', 'A5', 'MAPPO_RouteAPF', 'PPO_C')


def make_case(base, offset):
    data = copy.deepcopy(json.loads(Path(base).read_text()))
    state = np.asarray(data['worlds'][0]['initial'], dtype=float)
    targets = state[:4].reshape(2, 2).copy()
    pursuers = targets.copy(); pursuers[:, 0] -= float(offset)
    state[4:8] = pursuers.ravel()
    for world in data['worlds']:
        world['initial'] = state.tolist()
    for frame in data['history']:
        frame['positions'] = pursuers.tolist()
        frame['active_p'] = [True, True]; frame['active_e'] = [True, True]
        frame['reports'] = [[targets[j].tolist(), targets[j].tolist()] for j in range(2)]
    data['id'] = f'e50_capturable_offset_{offset:.2f}'
    return data


def batch(base, output, cycles=4):
    output = Path(output); rows = []
    for offset in (.25, .5, .75):
        case_path = output.parent / f'_capturable_{offset:.2f}.json'
        case_path.write_text(json.dumps(make_case(base, offset)))
        result = run(case_path, output.parent / f'_capturable_{offset:.2f}_run.json',
                     cycles=cycles, unknown_map=True, moving_evader=True,
                     rootpuct_evader=True, rootpuct_simulations=20, methods=METHODS)
        for row in result['rows']:
            cycles_out = row.get('cycles', [])
            captures = sum(len(c.get('capture_events', [])) for c in cycles_out)
            rows.append(dict(offset=offset, method=row['method'], valid=row['valid'],
                             error=row.get('error'), captures=captures,
                             double_capture=bool(captures >= 2),
                             collisions=sum(c.get('collision', False) for c in cycles_out),
                             projections=sum(c.get('safety_override') == 'local_action_projection' for c in cycles_out),
                             communication_stops=sum(c.get('communication_event') == 'stop_degraded' for c in cycles_out)))
    result = dict(stage='capturable_consistent_batch', base=str(base), cycles=cycles,
                  offsets=[.25, .5, .75], methods=list(METHODS), rows=rows,
                  diagnostic_initials=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--base', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--cycles', type=int, default=4)
    args = parser.parse_args(); print(json.dumps(batch(args.base, args.output, args.cycles), ensure_ascii=False, indent=2))
