"""Four-method ablation against a fixed Search Only opponent.

Two separate speed settings, each chosen so the full method matches one of
the surviving numbers:

- evader 1.0, pursuer 1.2: full method's pursuit rate is about 78
- evader 1.8, pursuer 1.2: full method's escape rate is about 48

The learned side is never on both ends of a game.  The other side is Search
Only with 3000 simulations.
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
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from benchmark_baselines import configure_problem, generate_initial_states, run_episode
from run_v3_component_ablation import METHODS, build
from solvers.baselines import RoleMaskedSolver, TeamSolver
from solvers.solver import get_solver

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "current", "benchmarks", "v3_half_match_ablation_20261113")
PURSUER_SPEED = 1.2
EPISODES = 40
SEED = 1000
FIXED_SIMULATIONS = 3000
BLOCKS = (
    {"name": "pursuit", "evader_speed": 1.0, "learned_role": "pursuer"},
    {"name": "escape", "evader_speed": 1.8, "learned_role": "evader"},
)


def fixed_search(problem, team):
    solver = get_solver(
        "C_PUCT_V1", policy_oracle=[None] * problem.num_robots, value_oracle=None,
        search_depth=100, number_simulations=FIXED_SIMULATIONS,
        C_pw=2.0, alpha_pw=0.5, C_exp=1.0, alpha_exp=0.25,
        beta_policy=0.0, beta_value=0.0, vis_on=False,
    )
    return RoleMaskedSolver(solver, team)


def run_one(task):
    torch.set_num_threads(1)
    problem = configure_problem(
        "none", evader_speed=task["evader_speed"], pursuer_speed=PURSUER_SPEED,
    )
    if task["learned_role"] == "pursuer":
        pursuer = build(problem, task["method"], problem.pursuers)
        evader = fixed_search(problem, problem.evaders)
    else:
        pursuer = fixed_search(problem, problem.pursuers)
        evader = build(problem, task["method"], problem.evaders)
    metrics = run_episode(problem, TeamSolver(evader, pursuer), np.asarray(task["state"]))
    return {
        "block": task["block"],
        "method": task["method"],
        "evader_speed": float(task["evader_speed"]),
        "learned_role": task["learned_role"],
        "episode": int(task["episode"]),
        "double_capture": int(metrics["double_capture"]),
        "total_capture": int(metrics["total_capture"]),
        "survival_time": float(metrics["survival_time"]),
        "illegal_terminal": int(metrics["illegal_terminal"]),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    tasks = []
    for block in BLOCKS:
        problem = configure_problem(
            "none", evader_speed=block["evader_speed"], pursuer_speed=PURSUER_SPEED,
        )
        states = generate_initial_states(problem, [SEED], EPISODES)[0]
        for method in METHODS:
            for episode, state in enumerate(states):
                tasks.append({
                    "block": block["name"],
                    "method": method,
                    "evader_speed": block["evader_speed"],
                    "learned_role": block["learned_role"],
                    "episode": episode,
                    "state": state,
                })
    workers = int(os.environ.get("ABLATION_WORKERS", "10"))
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, record in enumerate(pool.map(run_one, tasks), start=1):
            records.append(record)
            if done % 40 == 0 or done == len(tasks):
                print("done {}/{}".format(done, len(tasks)), flush=True)
    summary = []
    for block in BLOCKS:
        for method in METHODS:
            rows = [r for r in records if r["block"] == block["name"] and r["method"] == method]
            double = float(np.mean([r["double_capture"] for r in rows]))
            if block["learned_role"] == "pursuer":
                pursuit, escape = double, None
            else:
                pursuit, escape = None, 1.0 - double
            summary.append({
                "block": block["name"],
                "method": method,
                "evader_speed": block["evader_speed"],
                "pursuer_speed": PURSUER_SPEED,
                "fixed_opponent": "Search Only {}".format(FIXED_SIMULATIONS),
                "pursuit_double_capture": pursuit,
                "escape_rate": escape,
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
