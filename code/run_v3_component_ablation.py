"""Four-way ablation of the V3 iteration-45 search agent.

The other side is always the full agent (search + policy + value).  Pursuit
rate is that method's double-capture rate as pursuer.  Escape rate is one
minus the opponent's double-capture rate when the method plays evader.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from benchmark_baselines import configure_problem, generate_initial_states, run_episode
from learning.oracles import get_oracles
from solvers.baselines import RoleMaskedSolver, TeamSolver
from solvers.solver import get_solver

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "current", "memory_observation_v3", "models")
OUT = os.path.join(ROOT, "current", "benchmarks", "v3_component_ablation_20261113")
ITERATION = 45
EPISODES = 100
SEED = 1000
SIMULATIONS = 1000
DEPTH = 100
BETA_POLICY = 0.5
BETA_VALUE = 0.1

METHODS = ("full", "no_search", "no_value", "no_policy")


def build(problem, name, team):
    paths = [os.path.join(MODEL_DIR, "model_policy_l{}_i{}.pt".format(ITERATION, i)) for i in range(4)]
    value_path = os.path.join(MODEL_DIR, "model_value_l{}.pt".format(ITERATION))
    if name == "no_search":
        policy, _ = get_oracles(
            problem, policy_oracle_name="gaussian", policy_oracle_paths=paths,
            value_oracle_name=None, value_oracle_path=None,
        )
        solver = get_solver("NeuralNetwork", policy_oracle=policy, policy_stochastic=False)
        return RoleMaskedSolver(solver, team)
    use_policy = name in ("full", "no_value")
    use_value = name in ("full", "no_policy")
    policy, value = (None, None)
    if use_policy or use_value:
        policy, value = get_oracles(
            problem,
            policy_oracle_name="gaussian" if use_policy else None,
            policy_oracle_paths=paths if use_policy else None,
            value_oracle_name="deterministic" if use_value else None,
            value_oracle_path=value_path if use_value else None,
        )
    if not use_policy:
        policy = [None] * problem.num_robots
    solver = get_solver(
        "C_PUCT_V1", policy_oracle=policy, value_oracle=value,
        search_depth=DEPTH, number_simulations=SIMULATIONS,
        C_pw=2.0, alpha_pw=0.5, C_exp=1.0, alpha_exp=0.25,
        beta_policy=BETA_POLICY if use_policy else 0.0,
        beta_value=BETA_VALUE if use_value else 0.0, vis_on=False,
    )
    return RoleMaskedSolver(solver, team)


def run_one(task):
    torch.set_num_threads(1)
    problem = configure_problem("none")
    pursuer = build(problem, task["pursuer"], problem.pursuers)
    evader = build(problem, task["evader"], problem.evaders)
    metrics = run_episode(
        problem, TeamSolver(evader, pursuer), np.asarray(task["state"]),
        timing_warmup_steps=0,
    )
    return {
        "pursuer": task["pursuer"],
        "evader": task["evader"],
        "episode": int(task["episode"]),
        "double_capture": int(metrics["double_capture"]),
        "total_capture": int(metrics["total_capture"]),
        "capture_count": int(metrics["capture_count"]),
        "survival_time": float(metrics["survival_time"]),
        "illegal_terminal": int(metrics["illegal_terminal"]),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    problem = configure_problem("none")
    states = generate_initial_states(problem, [SEED], EPISODES)[0]
    tasks = []
    for method in METHODS:
        for episode, state in enumerate(states):
            tasks.append({"pursuer": method, "evader": "full", "episode": episode, "state": state})
            if method != "full":
                tasks.append({"pursuer": "full", "evader": method, "episode": episode, "state": state})
    workers = int(os.environ.get("ABLATION_WORKERS", "16"))
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, record in enumerate(pool.map(run_one, tasks), start=1):
            records.append(record)
            if done % 50 == 0 or done == len(tasks):
                print("done {}/{}".format(done, len(tasks)), flush=True)
    summary = []
    for method in METHODS:
        as_pursuer = [r for r in records if r["pursuer"] == method and r["evader"] == "full"]
        as_evader = [r for r in records if r["evader"] == method and r["pursuer"] == "full"]
        pursuit = float(np.mean([r["double_capture"] for r in as_pursuer]))
        captured = float(np.mean([r["double_capture"] for r in as_evader]))
        summary.append({
            "method": method,
            "pursuit_double_capture": pursuit,
            "escape_rate": 1.0 - captured,
            "as_pursuer_any_capture": float(np.mean([r["total_capture"] for r in as_pursuer])),
            "as_evader_any_capture_against": float(np.mean([r["total_capture"] for r in as_evader])),
            "as_pursuer_mean_length": float(np.mean([r["survival_time"] for r in as_pursuer])),
            "as_evader_mean_length": float(np.mean([r["survival_time"] for r in as_evader])),
            "as_pursuer_illegal": float(np.mean([r["illegal_terminal"] for r in as_pursuer])),
            "as_evader_illegal": float(np.mean([r["illegal_terminal"] for r in as_evader])),
            "episodes": EPISODES,
        })
    payload = {
        "model_dir": MODEL_DIR,
        "model_iteration": ITERATION,
        "obstacle_layout": "none",
        "speeds": "V7 ranges, evader [0.8,1.0], pursuer [1.0,1.2]",
        "simulations": SIMULATIONS,
        "depth": DEPTH,
        "beta_policy": BETA_POLICY,
        "beta_value": BETA_VALUE,
        "opponent": "full",
        "seed": SEED,
        "episodes": EPISODES,
        "summary": summary,
        "records": records,
    }
    with open(os.path.join(OUT, "result.json"), "w") as handle:
        json.dump(payload, handle)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
