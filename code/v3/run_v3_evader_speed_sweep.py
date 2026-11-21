"""Self-play of V3 search+policy+value while sweeping the evader speed."""
import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
import repo_paths
repo_paths.install()
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from benchmark_baselines import configure_problem, generate_initial_states, run_episode
from run_v3_component_ablation import build
from solvers.baselines import TeamSolver

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "current", "benchmarks", "v3_evader_speed_sweep_20261113")
PURSUER_SPEED = 1.2
EVADER_SPEEDS = (1.0, 1.3, 1.6, 2.0, 2.5, 3.0)
EPISODES = 40
SEED = 1000


def run_one(task):
    torch.set_num_threads(1)
    problem = configure_problem("none", evader_speed=task["evader_speed"], pursuer_speed=PURSUER_SPEED)
    pursuer = build(problem, "full", problem.pursuers)
    evader = build(problem, "full", problem.evaders)
    metrics = run_episode(problem, TeamSolver(evader, pursuer), np.asarray(task["state"]))
    return {
        "evader_speed": float(task["evader_speed"]),
        "episode": int(task["episode"]),
        "double_capture": int(metrics["double_capture"]),
        "total_capture": int(metrics["total_capture"]),
        "capture_count": int(metrics["capture_count"]),
        "survival_time": float(metrics["survival_time"]),
        "illegal_terminal": int(metrics["illegal_terminal"]),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    problem = configure_problem("none", evader_speed=1.0, pursuer_speed=PURSUER_SPEED)
    states = generate_initial_states(problem, [SEED], EPISODES)[0]
    tasks = []
    for speed in EVADER_SPEEDS:
        for episode, state in enumerate(states):
            tasks.append({"evader_speed": speed, "episode": episode, "state": state})
    workers = int(os.environ.get("ABLATION_WORKERS", "12"))
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, record in enumerate(pool.map(run_one, tasks), start=1):
            records.append(record)
            if done % 40 == 0 or done == len(tasks):
                print("done {}/{}".format(done, len(tasks)), flush=True)
    summary = []
    for speed in EVADER_SPEEDS:
        rows = [r for r in records if r["evader_speed"] == speed]
        double = float(np.mean([r["double_capture"] for r in rows]))
        summary.append({
            "evader_speed": speed,
            "pursuer_speed": PURSUER_SPEED,
            "pursuit_double_capture": double,
            "escape_rate": 1.0 - double,
            "any_capture": float(np.mean([r["total_capture"] for r in rows])),
            "mean_length": float(np.mean([r["survival_time"] for r in rows])),
            "illegal": float(np.mean([r["illegal_terminal"] for r in rows])),
            "episodes": len(rows),
        })
    with open(os.path.join(OUT, "result.json"), "w") as handle:
        json.dump({"summary": summary, "records": records}, handle)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
