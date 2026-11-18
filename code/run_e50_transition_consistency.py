"""Independent E50 receipt for Python accelerated/reference transition parity."""
import argparse
import json
from pathlib import Path
import numpy as np


def run(output, cases=100, seed=20261111):
    from e49_rules import joint_transition, reference_joint_transition
    rng = np.random.default_rng(seed)
    max_position_error = max_active_error = 0.
    failures = []
    for index in range(int(cases)):
        boxes = np.asarray([[[3., 3.5], [3., 4.]], [[-5., -4.], [4., 5.]]])
        positions = rng.uniform(-2., 2., (2, 2)); targets = rng.uniform(-2., 2., (2, 2))
        pa = rng.uniform(-1., 1., (2, 2)); ea = rng.uniform(-1.5, 1.5, (2, 2))
        ap = rng.random(2) > .1; ae = rng.random(2) > .1
        args = (positions, targets, pa, ea, ap, ae, boxes, .01, 1.)
        actual = joint_transition(*args); expected = reference_joint_transition(*args)
        pos_error = float(max(np.max(np.abs(actual[i] - expected[i])) for i in (0, 1)))
        active_error = int(np.any(actual[2] != expected[2]) or np.any(actual[3] != expected[3]))
        max_position_error = max(max_position_error, pos_error); max_active_error = max(max_active_error, active_error)
        if pos_error > 1e-9 or active_error:
            failures.append(dict(index=index, position_error=pos_error, active_error=active_error))
    result = dict(stage='e50_transition_consistency', cases=int(cases), seed=int(seed),
                  max_position_error=max_position_error, max_active_error=max_active_error,
                  failures=failures, passed=not failures)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    parser.add_argument('--cases', type=int, default=100); args = parser.parse_args()
    print(json.dumps(run(args.output, args.cases), indent=2))
