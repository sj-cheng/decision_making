"""V3 search+policy+value against a fixed Search Only opponent.

Pursuit rate: learned pursuer vs fixed Search Only evader.
Escape rate: learned evader vs fixed Search Only pursuer.
The two sides are never the same agent.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from benchmark_baselines import configure_problem, generate_initial_states, run_episode
from run_v3_component_ablation import build
from solvers.baselines import RoleMaskedSolver, TeamSolver
from solvers.solver import get_solver

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "current", "benchmarks", "v3_fixed_search_speed_sweep_20261113")
PURSUER_SPEED = 1.2
EVADER_SPEEDS = (1.0, 1.4, 1.8, 2.2, 2.8)
EPISODES = 30
SEED = 1000
FIXED_SIMULATIONS = 3000


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
    problem = configure_problem("none", evader_speed=task["evader_speed"], pursuer_speed=PURSUER_SPEED)
    if task["learned_role"] == "pursuer":
        pursuer = build(problem, "full", problem.pursuers)
        evader = fixed_search(problem, problem.evaders)
    else:
        pursuer = fixed_search(problem, problem.pursuers)
        evader = build(problem, "full", problem.evaders)
    metrics = run_episode(problem, TeamSolver(evader, pursuer), np.asarray(task["state"]))
    return {
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
    problem = configure_problem("none", evader_speed=1.0, pursuer_speed=PURSUER_SPEED)
    states = generate_initial_states(problem, [SEED], EPISODES)[0]
    tasks = []
    for speed in EVADER_SPEEDS:
        for episode, state in enumerate(states):
            for role in ("pursuer", "evader"):
                tasks.append({
                    "evader_speed": speed, "learned_role": role,
                    "episode": episode, "state": state,
                })
    workers = int(os.environ.get("ABLATION_WORKERS", "10"))
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, record in enumerate(pool.map(run_one, tasks), start=1):
            records.append(record)
            if done % 30 == 0 or done == len(tasks):
                print("done {}/{}".format(done, len(tasks)), flush=True)
    summary = []
    for speed in EVADER_SPEEDS:
        as_pursuer = [r for r in records if r["evader_speed"] == speed and r["learned_role"] == "pursuer"]
        as_evader = [r for r in records if r["evader_speed"] == speed and r["learned_role"] == "evader"]
        pursuit = float(np.mean([r["double_capture"] for r in as_pursuer]))
        captured = float(np.mean([r["double_capture"] for r in as_evader]))
        summary.append({
            "evader_speed": speed,
            "pursuer_speed": PURSUER_SPEED,
            "fixed_opponent": "Search Only {}".format(FIXED_SIMULATIONS),
            "pursuit_double_capture": pursuit,
            "escape_rate": 1.0 - captured,
            "pursuit_any_capture": float(np.mean([r["total_capture"] for r in as_pursuer])),
            "escape_mean_length": float(np.mean([r["survival_time"] for r in as_evader])),
            "pursuit_mean_length": float(np.mean([r["survival_time"] for r in as_pursuer])),
            "episodes": EPISODES,
        })
    with open(os.path.join(OUT, "result.json"), "w") as handle:
        json.dump({"summary": summary, "records": records}, handle)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
