"""History-consistent capturable-initial batch for the distributed E50 runner."""
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

from e49_rules import _legal
from run_e50_distributed_dynamic_smoke import run


def make_case(base, offset):
    data = copy.deepcopy(json.loads(Path(base).read_text()))
    state = np.asarray(data['worlds'][0]['initial'], dtype=float)
    targets = state[:4].reshape(2, 2).copy()
    boxes = np.asarray(data['map'], dtype=float).reshape(-1, 2, 2)
    pursuers = np.zeros_like(targets)
    for i, target in enumerate(targets):
        candidates = [target + np.array([-float(offset), 0.]),
                      target + np.array([float(offset), 0.]),
                      target + np.array([0., -float(offset)]),
                      target + np.array([0., float(offset)])]
        legal = [p for p in candidates if _legal(p, boxes)]
        pursuers[i] = legal[0] if legal else target.copy()
    state[4:8] = pursuers.ravel()
    for world in data['worlds']:
        world['initial'] = state.tolist()
    data['id'] = f'e50_distributed_capturable_offset_{offset:.2f}'
    return data


def batch(base, output, cycles=8, simulations=5):
    base = Path(base); output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset in (.25, .5, .75):
        case_path = output.parent / f'_{base.stem}_capturable_{offset:.2f}.json'
        case_path.write_text(json.dumps(make_case(base, offset)))
        run_path = output.parent / f'_{base.stem}_capturable_{offset:.2f}_run.json'
        result = run(case_path, run_path, cycles=cycles, lidar_rays=256,
                     delay=0, drop_cycles=(), rootpuct=True,
                     rootpuct_simulations=simulations)
        captures = result.get('captures', [])
        rows.append(dict(offset=offset, valid=(result.get('rootpuct_failures', 1) == 0),
                         captures=len(captures), double_capture=(len(captures) >= 2),
                         collision=sum(bool(x.get('pursuer_collision')) for x in result.get('logs', [])),
                         minco_infeasible=sum(not z.get('feasible', True)
                             for x in result.get('logs', []) for z in x.get('minco', {}).values()),
                         rootpuct_failures=result.get('rootpuct_failures', 0),
                         case=str(case_path), run=str(run_path)))
    out = dict(stage='e50_distributed_capturable_batch', base=str(base), cycles=cycles,
               simulations=simulations, offsets=[.25, .5, .75], rows=rows,
               diagnostic_initials=True)
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--base', required=True)
    p.add_argument('--output', required=True); p.add_argument('--cycles', type=int, default=8)
    p.add_argument('--simulations', type=int, default=5); a = p.parse_args()
    print(json.dumps(batch(a.base, a.output, a.cycles, a.simulations), ensure_ascii=False, indent=2))
