"""Screen E50 methods from the separate unknown-map and capturable receipts."""
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


def load(path): return json.loads(Path(path).read_text())


def run(unknown_batch, long_first, capturable, output):
    unknown = load(unknown_batch); long_data = load(long_first); cap = load(capturable)
    methods = unknown.get('methods', [])
    rows = []
    for method in methods:
        u = [r for r in unknown['rows'] if r['method'] == method]
        l = [r for r in long_data['rows'] if r['method'] == method]
        c = [r for r in cap['rows'] if r['method'] == method]
        valid = bool(u) and all(r['valid'] for r in u) and all(r['valid'] for r in l + c)
        long_covered = bool(l)
        collisions = sum(r.get('executed_collisions', 0) for r in u) + sum(r.get('execution_collisions', 0) for r in l)
        captures = sum(r.get('captures', 0) for r in c)
        double = sum(r.get('double_capture', False) for r in c)
        rows.append(dict(method=method, valid=valid, unknown_scene_count=len(u),
                         execution_collisions=collisions, capturable_captures=captures,
                         capturable_double_capture_cases=double,
                         local_projections=sum(r.get('local_projections', 0) for r in u),
                         communication_stops=sum(r.get('stop_degraded', 0) for r in u),
                         long_horizon_covered=long_covered,
                         stable_under_audit=bool(valid and long_covered and collisions == 0),
                         retain_for_next_stage=bool(valid and long_covered and collisions == 0 and double >= 2)))
    result = dict(stage='e50_method_screening', sources=[str(unknown_batch), str(long_first), str(capturable)], rows=rows,
                  policy='retain requires valid receipts, zero execution collisions, and at least 2/3 capturable double-capture cases')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n'); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unknown-batch', required=True); parser.add_argument('--long-first', required=True)
    parser.add_argument('--capturable', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(json.dumps(run(args.unknown_batch, args.long_first, args.capturable, args.output), ensure_ascii=False, indent=2))
