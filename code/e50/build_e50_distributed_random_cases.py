"""Build legal random-translation initial states for E50 distributed runs."""
import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
import repo_paths
repo_paths.install()
import argparse, copy, json
from pathlib import Path
import numpy as np
from e49_rules import _legal


def build(base, output, count=3, seed=20261116):
    base = Path(base); data = json.loads(base.read_text()); boxes = np.asarray(data['map'], float).reshape(-1, 2, 2)
    state0 = np.asarray(data['worlds'][0]['initial'], float)
    positions = state0[:8].reshape(4, 2)
    rng = np.random.default_rng(seed); rows=[]; tries=0
    while len(rows) < int(count) and tries < 10000:
        tries += 1
        shift = rng.uniform(-1.25, 1.25, 2)
        moved = positions + shift
        if not all(_legal(p, boxes) for p in moved): continue
        item = copy.deepcopy(data); state = state0.copy(); state[:8] = moved.ravel()
        def shift_pair(pair):
            return [float(pair[0] + shift[0]), float(pair[1] + shift[1])]
        for world in item['worlds']:
            world['initial'] = state.tolist()
            # Keep every public scripted state in the same translated frame.
            for scripted in world.get('scripted_states', []):
                if len(scripted) >= 8:
                    scripted[:8] = (np.asarray(scripted[:8], float).reshape(4, 2) + shift).ravel().tolist()
            for row in world.get('history', []):
                row['positions'] = [shift_pair(p) for p in row.get('positions', [])]
                reports = row.get('reports', [])
                for observer in reports:
                    for idx, report in enumerate(observer):
                        if report is not None:
                            observer[idx] = shift_pair(report)
        item['id'] = f"{data.get('id','case')}_random_shift_{len(rows):02d}"
        path = Path(output) / f"{base.stem}_random_{len(rows):02d}.json"; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(item, ensure_ascii=False, indent=2) + '\n')
        rows.append(dict(index=len(rows), seed=int(seed), shift=shift.tolist(), case=str(path)))
    if len(rows) < count: raise RuntimeError(f"could not build {count} legal cases after {tries} tries")
    return rows


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--base',required=True); p.add_argument('--output',required=True); p.add_argument('--count',type=int,default=3); p.add_argument('--seed',type=int,default=20261116); a=p.parse_args()
    print(json.dumps(build(a.base,a.output,a.count,a.seed),ensure_ascii=False,indent=2))
