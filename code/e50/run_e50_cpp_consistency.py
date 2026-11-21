"""E50 receipt for native C++ Example8 transition parity."""
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


def _state(problem, rng):
    s = np.zeros((problem.state_dim, 1)); s[:8, 0] = np.r_[rng.uniform(3, 8, (2, 2)).ravel(),
                                                            rng.uniform(-8, -3, (2, 2)).ravel()]
    s[problem.active_idxs, 0] = 1.; return s


def run(output, cases=20, seed=20261111):
    from e49_rules import create_environment, make_opponent
    rng = np.random.default_rng(seed); failures = []; max_error = 0.
    for index in range(int(cases)):
        boxes = [] if index % 2 else [[[0., .4], [-2., 2.]]]
        problem = create_environment(boxes, 40); opponent = make_opponent(problem, 19, simulations=3)
        wrapper = opponent._problem_wrapper_for(problem)
        state = _state(problem, rng); action = rng.uniform(-2., 2., (problem.action_dim, 1))
        py = problem.step(state, action); native = np.asarray(wrapper.step(state, action, 1.)).reshape(-1, 1)
        error = float(np.max(np.abs(py - native))); max_error = max(max_error, error)
        if error > 3e-5: failures.append(dict(index=index, error=error))
    result = dict(stage='e50_cpp_transition_consistency', cases=int(cases), seed=int(seed),
                  max_state_error=max_error, failures=failures, passed=not failures)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n'); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    parser.add_argument('--cases', type=int, default=20); args = parser.parse_args()
    print(json.dumps(run(args.output, args.cases), indent=2))
