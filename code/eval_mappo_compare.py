"""Compare MAPPO checkpoints (5k vs 50k) head-to-head and vs fixed references.

Usage (from code/):
    ~/miniconda3/envs/dm_env/bin/python eval_mappo_compare.py --episodes 100
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from problems.problem import get_problem
from learning.mappo import MAPPOPolicySolver
from solvers.baselines import RandomBaseline, GreedyPursuitEvasion, TeamSolver


def make_problem():
    problem = get_problem("example8")
    problem.obstacles = []
    problem.evader_spawn_near_obstacles = False
    problem.use_minco_dynamics = False
    return problem


def run_episode(problem, pursuer_solver, evader_solver, seed):
    np.random.seed(seed)
    state = problem.initialize()
    steps = 0
    for _ in range(len(problem.times) - 1):
        if problem.is_terminal(state):
            break
        action = np.zeros((problem.action_dim, 1))
        a_p = pursuer_solver.policy(problem, state, team=problem.pursuers) \
            if getattr(pursuer_solver, "supports_team_policy", False) else pursuer_solver.policy(problem, state)
        a_e = evader_solver.policy(problem, state, team=problem.evaders) \
            if getattr(evader_solver, "supports_team_policy", False) else evader_solver.policy(problem, state)
        for robot in range(problem.num_robots):
            src = a_e if robot in problem.evaders else a_p
            action[problem.action_idxs[robot], :] = src[problem.action_idxs[robot], :]
        state = problem.step(state, action, problem.dt)
        steps += 1
    caught = 2 - problem.active_evader_count(state)
    oob = not problem.is_valid(state)
    # team returns over the episode are recomputed cheaply: use final-outcome proxies
    return {
        "steps": steps,
        "caught": caught,
        "double": float(caught == 2),
        "timeout_survive": float(caught < 2 and state[problem.time_idx, 0] >= problem.tf),
        "oob": float(oob),
    }


def evaluate(name, pursuer, evader, episodes, seed0=2000):
    problem = make_problem()
    rows = [run_episode(problem, pursuer, evader, seed0 + i) for i in range(episodes)]
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print("{:42s} steps={:5.1f} caught={:4.2f} double={:5.1%} survive={:5.1%} oob={:5.1%}".format(
        name, agg["steps"], agg["caught"], agg["double"], agg["timeout_survive"], agg["oob"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    m5 = MAPPOPolicySolver(make_problem(), path="../current/baselines/mappo_global.pt")
    m50 = MAPPOPolicySolver(make_problem(), path="../current/baselines/mappo_50k.pt")
    probe = make_problem()
    rnd_p = RandomBaseline(team=probe.pursuers)
    rnd_e = RandomBaseline(team=probe.evaders)
    greedy_p = GreedyPursuitEvasion(team=probe.pursuers)
    greedy_e = GreedyPursuitEvasion(team=probe.evaders)

    n = args.episodes
    print("=== MAPPO-5k vs MAPPO-50k cross matches (n={}) ===".format(n))
    evaluate("P=50k  vs E=50k (self)", m50, m50, n)
    evaluate("P=5k   vs E=5k  (self)", m5, m5, n)
    evaluate("P=50k  vs E=5k", m50, m5, n)
    evaluate("P=5k   vs E=50k", m5, m50, n)
    print("=== vs fixed references ===")
    evaluate("P=50k  vs E=Random", m50, rnd_e, n)
    evaluate("P=50k  vs E=Greedy", m50, greedy_e, n)
    evaluate("P=5k   vs E=Random", m5, rnd_e, n)
    evaluate("P=5k   vs E=Greedy", m5, greedy_e, n)
    evaluate("P=Random vs E=50k", rnd_p, m50, n)
    evaluate("P=Greedy vs E=50k", greedy_p, m50, n)
    evaluate("P=Random vs E=5k", rnd_p, m5, n)
    evaluate("P=Greedy vs E=5k", greedy_p, m5, n)


if __name__ == "__main__":
    main()
