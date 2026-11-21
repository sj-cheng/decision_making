"""Pursuer decision time for the V3 ablation.

Learned method is the pursuer.  Evader is Search Only, 3000 simulations.
Evader speed 1.0, pursuer speed 1.2.  Time is the pursuer solver alone.
The first call of each episode is dropped.
"""
import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
import repo_paths
repo_paths.install()
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

import run_v3_component_ablation as ablation
from benchmark_baselines import configure_problem, generate_initial_states, run_episode
from run_v3_component_ablation import build
from run_v3_half_match_ablation import fixed_search
from solvers.baselines import TeamSolver

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "current", "benchmarks", "v3_pursuer_ablation_betav05_rates_20261114")
EVADER_SPEED = 1.0
PURSUER_SPEED = 1.2
EPISODES = 10
METHODS = ("full", "no_policy")
BETA_VALUE = 0.5
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
    ablation.BETA_VALUE = BETA_VALUE
    problem = configure_problem("none", evader_speed=EVADER_SPEED, pursuer_speed=PURSUER_SPEED)
    pursuer = TimedSolver(build(problem, task["method"], problem.pursuers))
    evader = fixed_search(problem, problem.evaders)
    metrics = run_episode(problem, TeamSolver(evader, pursuer), np.asarray(task["state"]))
    samples = pursuer.times_ms[1:]
    return {
        "method": task["method"],
        "episode": int(task["episode"]),
        "double_capture": int(metrics["double_capture"]),
        "total_capture": int(metrics["total_capture"]),
        "survival_time": float(metrics["survival_time"]),
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
    workers = int(os.environ.get("ABLATION_WORKERS", "4"))
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, record in enumerate(pool.map(run_one, tasks), start=1):
            records.append(record)
            if done % 20 == 0 or done == len(tasks):
                print("done {}/{}".format(done, len(tasks)), flush=True)
    summary = []
    for method in METHODS:
        rows = [r for r in records if r["method"] == method]
        samples = np.asarray([t for r in rows for t in r["decision_ms"]], dtype=float)
        beta_policy, beta_value = {
            "full": (0.5, BETA_VALUE),
            "no_search": (0.0, 0.0),
            "no_value": (0.5, 0.0),
            "no_policy": (0.0, BETA_VALUE),
        }[method]
        summary.append({
            "method": method,
            "beta_policy": beta_policy,
            "beta_value": beta_value,
            "episodes": len(rows),
            "pursuit_double_capture": float(np.mean([r["double_capture"] for r in rows])),
            "mean_length": float(np.mean([r["survival_time"] for r in rows])),
            "decision_count": int(samples.size),
            "decision_mean_ms": float(np.mean(samples)),
            "decision_p50_ms": float(np.percentile(samples, 50)),
            "decision_p95_ms": float(np.percentile(samples, 95)),
        })
    with open(os.path.join(OUT, "result.json"), "w") as handle:
        json.dump({"summary": summary, "records": records}, handle)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
