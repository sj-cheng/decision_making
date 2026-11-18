"""Independent capture-event gate smoke for the E50 task metric layer."""
import argparse
import json
from pathlib import Path
import numpy as np

from e49_rules import capture_pairs


def run(output):
    boxes = np.asarray([[[8., 9.], [8., 9.]]], dtype=float)
    scenarios = {
        'double_capture': (np.array([[0., 0.], [1., 0.]]),
                           np.array([[.25, 0.], [1.25, 0.]])),
        'single_capture': (np.array([[0., 0.], [4., 0.]]),
                           np.array([[.25, 0.], [4., 3.]])),
        'blocked_line_no_capture': (np.array([[0., 0.], [4., 0.]]),
                                    np.array([[.25, 0.], [4., 3.]])),
    }
    rows = []
    for name, (pursuers, evaders) in scenarios.items():
        local_boxes = boxes if name != 'blocked_line_no_capture' else np.asarray(
            [[[.1, .2], [-.2, .2]], [[8., 9.], [8., 9.]]], dtype=float)
        pairs = capture_pairs(pursuers, evaders, [True, True], [True, True], local_boxes)
        rows.append(dict(scenario=name, pairs=[list(x) for x in pairs],
                         captured_evaders=sorted(int(x[1]) for x in pairs),
                         double_capture=len(pairs) == 2))
    result = dict(stage='capture_gate_smoke', rows=rows,
                  note='Metric-layer acceptance only; no controller or benchmark claim.')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(json.dumps(run(args.output), ensure_ascii=False, indent=2))
