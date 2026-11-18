"""Evader-side ablation: evader speed 1.2, pursuer speed 0.6.

The learned method plays evader.  The pursuer is Search Only, 3000
simulations, no policy and no value.
Decision time is the evader solver alone, excluding the fixed pursuer.
The first call of each episode is dropped.
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from benchmark_baselines import configure_problem, generate_initial_states, run_episode
from run_v3_component_ablation import METHODS, build
from run_v3_half_match_ablation import fixed_search
from solvers.baselines import TeamSolver

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "current", "benchmarks", "v3_evader_ablation_pursuer06_20261114")
EVADER_SPEED = 1.2
PURSUER_SPEED = 0.6
EPISODES = 40
SEED = 1000


class TimedSolver(object):
    def __init__(self, solver):
        self.solver = solver
        self.times_ms = []

    def policy(self, problem, state, team=None):
        started = time.perf_counter()
        if team is None:
            action = self.solver.policy(problem, state)
        else:
            action = self.solver.policy(problem, state, team=team)
        self.times_ms.append((time.perf_counter() - started) * 1000.0)
        return action


def run_one(task):
    torch.set_num_threads(1)
    problem = configure_problem("none", evader_speed=EVADER_SPEED, pursuer_speed=PURSUER_SPEED)
    pursuer = fixed_search(problem, problem.pursuers)
    evader = TimedSolver(build(problem, task["method"], problem.evaders))
    metrics = run_episode(problem, TeamSolver(evader, pursuer), np.asarray(task["state"]))
    samples = evader.times_ms[1:]
    return {
        "method": task["method"],
        "episode": int(task["episode"]),
        "double_capture": int(metrics["double_capture"]),
        "total_capture": int(metrics["total_capture"]),
        "survival_time": float(metrics["survival_time"]),
        "illegal_terminal": int(metrics["illegal_terminal"]),
        "decision_ms": [float(x) for x in samples],
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    problem = configure_problem("none", evader_speed=EVADER_SPEED, pursuer_speed=PURSUER_SPEED)
    states = generate_initial_states(problem, [SEED], EPISODES)[0]
    tasks = []
    for method in METHODS:
        for episode, state in enumerate(states):
            tasks.append({"method": method, "episode": episode, "state": state})
    # Few workers so a search step is not waiting on other searches.
    workers = int(os.environ.get("ABLATION_WORKERS", "4"))
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, record in enumerate(pool.map(run_one, tasks), start=1):
            records.append(record)
            if done % 40 == 0 or done == len(tasks):
                print("done {}/{}".format(done, len(tasks)), flush=True)
    summary = []
    for method in METHODS:
        rows = [r for r in records if r["method"] == method]
        captured = float(np.mean([r["double_capture"] for r in rows]))
        samples = np.asarray([t for r in rows for t in r["decision_ms"]], dtype=float)
        summary.append({
            "method": method,
            "evader_speed": EVADER_SPEED,
            "pursuer_speed": PURSUER_SPEED,
            "fixed_pursuer": "Search Only 3000",
            "escape_rate": 1.0 - captured,
            "double_capture": captured,
            "any_capture": float(np.mean([r["total_capture"] for r in rows])),
            "mean_length": float(np.mean([r["survival_time"] for r in rows])),
            "illegal": float(np.mean([r["illegal_terminal"] for r in rows])),
            "episodes": len(rows),
            "decision_count": int(samples.size),
            "decision_mean_ms": float(np.mean(samples)) if samples.size else None,
            "decision_p50_ms": float(np.percentile(samples, 50)) if samples.size else None,
            "decision_p95_ms": float(np.percentile(samples, 95)) if samples.size else None,
        })
    with open(os.path.join(OUT, "result.json"), "w") as handle:
        json.dump({"summary": summary, "records": records}, handle)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
