"""Reproducible benchmark for the multi-robot pursuit-evasion setting.

Run from ``code``.  The default benchmark evaluates every method against the
large-budget MCTS opponent and writes per-episode records:

Each record includes role-specific latency fields (``evader_*`` and
``pursuer_*``).  The legacy ``mean_latency_ms`` fields measure the complete
joint action call and are retained only for compatibility.
``*_step_mean_latency_ms`` in the summary files is the pooled mean over
all measured solver calls after the configured warm-up steps.

    MPLCONFIGDIR=/tmp/mpl python benchmark_baselines.py \
        --seeds 5 --large-seeds 1 --episodes 200 \
        --search-only-budget 3000 --search-learning-budget 1000 \
        --large-budget 50000 --beta-policy 0.5 --beta-value 0.5 \
        --output ../current/benchmarks

Use ``--round-robin`` for the full method-by-method matrix.  MAPPO is included
when ``--mappo-checkpoint`` points to a trained checkpoint.
Use ``--obstacle-layout two`` or ``four`` for fixed obstacle maps; ``none``
retains the original no-obstacle protocol. See ``run_benchmark_obstacles.sh``.
"""

import argparse
import contextlib
import csv
import io
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from collections import Counter, OrderedDict
from queue import Empty

import numpy as np
import torch
from tqdm import tqdm

from problems.problem import get_problem
from learning.oracles import get_oracles
from learning.mappo import MAPPOPolicySolver
from solvers.solver import get_solver
from solvers.baselines import (
    GreedyPursuitEvasion,
    RandomBaseline,
    RoleMaskedSolver,
    TeamSolver,
)
from util import contains


METHODS = ("Random", "Greedy", "Policy Only", "Search Only", "Search + Learning", "MAPPO", "Large-budget MCTS")

# Fixed maps, independent of future changes to Example8's training defaults.
# Each rectangle is [[xmin, xmax], [ymin, ymax]].
OBSTACLE_LAYOUTS = {
    "none": (),
    "two": (((-4.0, -2.0), (-3.5, -1.5)),
            ((2.0, 4.0), (1.5, 3.5))),
    "four": (((-4.0, -2.0), (-3.5, -1.5)),
             ((2.0, 4.0), (1.5, 3.5)),
             ((-4.0, -2.0), (1.5, 3.5)),
             ((2.0, 4.0), (-3.5, -1.5))),
}

_PROGRESS_QUEUE = None
_PROGRESS_STEP_INTERVAL = 5


def init_worker_progress(progress_queue, step_interval):
    global _PROGRESS_QUEUE, _PROGRESS_STEP_INTERVAL
    _PROGRESS_QUEUE = progress_queue
    _PROGRESS_STEP_INTERVAL = max(1, int(step_interval))


def configure_problem(obstacle_layout="none"):
    problem = get_problem("example8")
    # Retain the benchmark dynamics and spawn rules, changing only the map.
    # The parent and every spawned worker must use the same explicit layout.
    problem.obstacles = [np.array(rectangle, dtype=float)
                         for rectangle in OBSTACLE_LAYOUTS[obstacle_layout]]
    problem.evader_spawn_near_obstacles = False
    problem.use_minco_dynamics = False
    return problem


def problem_metadata(problem, obstacle_layout):
    protocol = ("example8_global_no_obstacle_direct_v1" if obstacle_layout == "none"
                else "example8_global_{}_obstacles_direct_v1".format(obstacle_layout))
    return {
        "benchmark_protocol": protocol,
        "obstacle_layout": obstacle_layout,
        "obstacles": [obstacle.tolist() for obstacle in problem.obstacles],
        "use_minco_dynamics": problem.use_minco_dynamics,
        "evader_spawn_near_obstacles": problem.evader_spawn_near_obstacles,
        "split_spawn_by_team": problem.split_spawn_by_team,
        "observation_contract": {
            "search": "full simulator state and fixed obstacle map",
            "policy": "semantic lidar with obstacle occlusion and last-seen memory",
            "value": "physical state encoding",
            "mappo": "checkpoint observation schema and action decoder",
        },
        "collision_rule": "endpoint inside rectangle (inclusive), following Example8",
    }


def oracle_paths(root, iteration=79, model_dir=None):
    # This checkpoint family matches the current Gaussian policy wrapper
    # (64-hidden-unit network, 264-D lidar encoding) and the 29-D state used
    # by the current Example8 implementation.  Older 16_* checkpoints use a
    # legacy architecture and cannot be loaded by the current wrapper.
    dirname = model_dir or os.path.join(root, "current", "historymodels", "18_lidar_2obstacle")
    policy_paths = [os.path.join(dirname, "model_policy_l{}_i{}.pt".format(iteration, i)) for i in range(4)]
    value_path = os.path.join(dirname, "model_value_l{}.pt".format(iteration))
    if not all(os.path.exists(p) for p in policy_paths + [value_path]):
        raise FileNotFoundError("missing policy/value checkpoint under {}".format(dirname))
    return policy_paths, value_path


def resolve_mappo_checkpoint(root, requested_path):
    """Resolve the MAPPO checkpoint independently of the launch directory."""
    default_path = os.path.join(root, "current", "baselines", "mappo_ego_50k.pt")
    if not requested_path:
        if os.path.exists(default_path):
            return default_path
        return None

    requested_path = os.path.expanduser(str(requested_path))
    candidates = [requested_path]
    if not os.path.isabs(requested_path):
        candidates.append(os.path.join(root, requested_path))
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "MAPPO checkpoint not found: {} (default checked: {})".format(
            requested_path, default_path
        )
    )


def build_solver(
    problem,
    name,
    root,
    search_budget,
    mappo_checkpoint=None,
    model_dir=None,
    model_iteration=79,
    beta_policy=0.0,
    beta_value=0.0,
    mappo_device="cpu",
    mappo_legacy=False,
):
    if name == "Random":
        return RandomBaseline()
    if name == "Greedy":
        return GreedyPursuitEvasion()
    if name == "MAPPO":
        if not mappo_checkpoint:
            raise ValueError("MAPPO requested but --mappo-checkpoint was not supplied")
        return MAPPOPolicySolver(problem, path=mappo_checkpoint, device=mappo_device, legacy=mappo_legacy)

    if name == "Policy Only":
        policy_paths, value_path = oracle_paths(root, iteration=model_iteration, model_dir=model_dir)
        policy, _ = get_oracles(
            problem,
            value_oracle_name=None,
            value_oracle_path=None,
            policy_oracle_name="gaussian",
            policy_oracle_paths=policy_paths,
        )
        return get_solver("NeuralNetwork", policy_oracle=policy)

    if name == "Search Only":
        return get_solver(
            "C_PUCT_V1", policy_oracle=[None] * problem.num_robots,
            value_oracle=None, search_depth=100,
            number_simulations=int(search_budget), C_pw=2.0,
            alpha_pw=0.5, C_exp=1.0, alpha_exp=0.25,
            beta_policy=0.0, beta_value=0.0, vis_on=False,
        )

    if name == "Search + Learning":
        policy_paths, value_path = oracle_paths(root, iteration=model_iteration, model_dir=model_dir)
        policy, value = get_oracles(
            problem,
            value_oracle_name="deterministic",
            value_oracle_path=value_path,
            policy_oracle_name="gaussian",
            policy_oracle_paths=policy_paths,
        )
        return get_solver(
            "C_PUCT_V1", policy_oracle=policy, value_oracle=value,
            search_depth=100, number_simulations=int(search_budget),
            C_pw=2.0, alpha_pw=0.5, C_exp=1.0, alpha_exp=0.25,
            beta_policy=float(beta_policy), beta_value=float(beta_value), vis_on=False,
        )

    if name == "Large-budget MCTS":
        return get_solver(
            "C_PUCT_V1", policy_oracle=[None] * problem.num_robots,
            value_oracle=None, search_depth=100,
            number_simulations=int(search_budget), C_pw=2.0,
            alpha_pw=0.5, C_exp=1.0, alpha_exp=0.25,
            beta_policy=0.0, beta_value=0.0, vis_on=False,
        )
    raise ValueError("unknown method {}".format(name))


def make_team_solver(
    problem,
    method,
    team,
    root,
    search_budget,
    mappo_checkpoint,
    model_dir,
    model_iteration,
    beta_policy=0.0,
    beta_value=0.0,
    mappo_device="cpu",
    mappo_legacy=False,
):
    solver = build_solver(
        problem,
        method,
        root,
        search_budget,
        mappo_checkpoint,
        model_dir,
        model_iteration,
        beta_policy,
        beta_value,
        mappo_device,
        mappo_legacy,
    )
    # Random and Greedy already support role masking.  All other solvers are
    # joint solvers and are masked only when used against a different policy.
    if method == "Random":
        return RandomBaseline(team=team)
    if method == "Greedy":
        return GreedyPursuitEvasion(team=team)
    return RoleMaskedSolver(solver, team)


def generate_initial_states(problem, seeds, episodes):
    all_states = []
    for seed in seeds:
        np.random.seed(int(seed))
        states = []
        for _ in range(episodes):
            states.append(np.array(problem.initialize(), copy=True))
        all_states.append(states)
    return all_states


def run_episode(problem, solver, initial_state, progress_info=None, timing_warmup_steps=0):
    problem.clear_closed_loop_cache()
    state = np.array(initial_state, copy=True)
    total_reward = np.zeros((problem.num_robots,), dtype=float)
    latencies = []
    evader_latencies = []
    pursuer_latencies = []
    invalid = False
    violations = {role + "_" + kind: 0
                  for role in ("evader", "pursuer")
                  for kind in ("obstacle_collision", "out_of_bounds")}
    first_capture_time = np.nan
    final_capture_time = np.nan
    capture_count = 0
    step_count = 0
    timing_warmup_steps = max(0, int(timing_warmup_steps))
    for _ in range(len(problem.times) - 1):
        if problem.is_terminal(state):
            break
        t0 = time.perf_counter()
        action = solver.policy(problem, state)
        # TeamSolver exposes per-side timings.  Keep the old joint timing as
        # well for backwards compatibility, but do not use it as the latency
        # of either method: it includes both solvers in the matchup.
        joint_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if step_count >= timing_warmup_steps:
            latencies.append(float(getattr(solver, "last_joint_latency_ms", joint_elapsed_ms)))
            evader_latencies.append(float(getattr(solver, "last_evader_latency_ms", np.nan)))
            pursuer_latencies.append(float(getattr(solver, "last_pursuer_latency_ms", np.nan)))
        step_count += 1
        reward = np.asarray(problem.reward(state, action), dtype=float).reshape(-1)
        next_state = problem.step(state, action, problem.dt)
        total_reward += reward
        current_captures = len(problem.evaders) - problem.active_evader_count(next_state)
        if current_captures > capture_count:
            if np.isnan(first_capture_time):
                first_capture_time = float(next_state[problem.time_idx, 0])
            capture_count = current_captures
            final_capture_time = float(next_state[problem.time_idx, 0])
        if not problem.is_valid(next_state):
            invalid = True
            for role, robots in (("evader", problem.evaders), ("pursuer", problem.pursuers)):
                for robot in robots:
                    if problem.check_obstacle_collision(next_state, robot):
                        violations[role + "_obstacle_collision"] = 1
                    idx = problem.state_idxs[robot]
                    if not contains(next_state[idx, :], problem.state_lims[idx, :]):
                        violations[role + "_out_of_bounds"] = 1
        state = next_state
        if progress_info is not None and _PROGRESS_QUEUE is not None:
            stem, seed, episode, max_steps = progress_info
            step_number = step_count
            if step_number % _PROGRESS_STEP_INTERVAL == 0:
                _PROGRESS_QUEUE.put((
                    "step", stem, int(seed), int(episode),
                    int(_PROGRESS_STEP_INTERVAL), int(step_number), os.getpid()
                ))
        if problem.is_terminal(state):
            break

    latencies = np.asarray(latencies, dtype=float)
    if progress_info is not None and _PROGRESS_QUEUE is not None:
        stem, seed, episode, max_steps = progress_info
        reported_steps = (step_count // _PROGRESS_STEP_INTERVAL) * _PROGRESS_STEP_INTERVAL
        # Pad the final episode to the fixed horizon so the global progress
        # bar has a deterministic total even when an episode terminates early.
        _PROGRESS_QUEUE.put((
            "episode_done", stem, int(seed), int(episode),
            int(max_steps - reported_steps), int(step_count), os.getpid()
        ))
    final_time = float(state[problem.time_idx, 0])

    def latency_stats(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return {"mean": np.nan, "p50": np.nan, "p95": np.nan, "max": np.nan}
        return {
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    joint_latency = latency_stats(latencies)
    evader_latency = latency_stats(evader_latencies)
    pursuer_latency = latency_stats(pursuer_latencies)
    return {
        "capture_count": int(capture_count),
        "total_capture": int(capture_count > 0),
        "double_capture": int(capture_count == 2),
        # First capture is defined for every episode with at least one
        # captured evader.  ``capture_time`` remains the legacy final
        # (second) capture time and is only defined for double captures.
        "first_capture_time": float(first_capture_time) if capture_count > 0 else np.nan,
        "capture_time": float(final_capture_time) if capture_count == 2 else np.nan,
        "survival_time": final_time,
        "pursuer_return": float(np.sum(total_reward[problem.pursuers])),
        "evader_return": float(np.sum(total_reward[problem.evaders])),
        "collision_or_out_of_bounds": int(invalid),
        "illegal_terminal": int(invalid),
        "obstacle_collision": max(violations["evader_obstacle_collision"],
                                  violations["pursuer_obstacle_collision"]),
        "out_of_bounds": max(violations["evader_out_of_bounds"],
                             violations["pursuer_out_of_bounds"]),
        **violations,
        # Legacy fields are explicitly the whole joint action call.  The
        # side-specific fields below are the values to use for per-method
        # latency tables (pursuer_* for a pursuer row, evader_* for an evader
        # row).
        "mean_latency_ms": joint_latency["mean"],
        "p50_latency_ms": joint_latency["p50"],
        "p95_latency_ms": joint_latency["p95"],
        "max_latency_ms": joint_latency["max"],
        "evader_mean_latency_ms": evader_latency["mean"],
        "evader_p50_latency_ms": evader_latency["p50"],
        "evader_p95_latency_ms": evader_latency["p95"],
        "evader_max_latency_ms": evader_latency["max"],
        "pursuer_mean_latency_ms": pursuer_latency["mean"],
        "pursuer_p50_latency_ms": pursuer_latency["p50"],
        "pursuer_p95_latency_ms": pursuer_latency["p95"],
        "pursuer_max_latency_ms": pursuer_latency["max"],
        # Sums and counts allow the summary writer to compute the exact
        # pooled mean over all measured decision calls, rather than an
        # unweighted mean of episode-level means.
        "joint_latency_sum_ms": float(np.nansum(latencies)),
        "joint_latency_count": int(np.isfinite(latencies).sum()),
        "evader_latency_sum_ms": float(np.nansum(evader_latencies)),
        "evader_latency_count": int(np.isfinite(evader_latencies).sum()),
        "pursuer_latency_sum_ms": float(np.nansum(pursuer_latencies)),
        "pursuer_latency_count": int(np.isfinite(pursuer_latencies).sum()),
        "steps": int(step_count),
    }


def method_budget(method, args, role=None):
    role_override = None
    if role == "pursuer":
        role_override = getattr(args, "pursuer_budget", None)
    elif role == "evader":
        role_override = getattr(args, "evader_budget", None)
    if role_override is not None and method in {"Search Only", "Search + Learning", "Large-budget MCTS"}:
        return int(role_override)
    if method == "Large-budget MCTS":
        return int(args.large_budget)
    if method == "Search Only":
        return int(args.search_only_budget)
    if method == "Search + Learning":
        return int(args.search_learning_budget)
    return 0


def method_betas(method, args):
    if method == "Search + Learning":
        return float(args.beta_policy), float(args.beta_value)
    return 0.0, 0.0


def evaluate_matchup(problem, pursuer_name, evader_name, initial_states, seeds, args, episode_offsets=None):
    pursuer = make_team_solver(
        problem,
        pursuer_name,
        problem.pursuers,
        args.root,
        method_budget(pursuer_name, args, role="pursuer"),
        args.mappo_checkpoint,
        args.model_dir,
        args.model_iteration,
        *method_betas(pursuer_name, args),
        args.mappo_device,
        args.mappo_legacy,
    )
    evader = make_team_solver(
        problem,
        evader_name,
        problem.evaders,
        args.root,
        method_budget(evader_name, args, role="evader"),
        args.mappo_checkpoint,
        args.model_dir,
        args.model_iteration,
        *method_betas(evader_name, args),
        args.mappo_device,
        args.mappo_legacy,
    )
    solver = TeamSolver(evader, pursuer)
    records = []
    if episode_offsets is None:
        episode_offsets = [0] * len(initial_states)
    for seed, states, episode_offset in zip(seeds, initial_states, episode_offsets):
        for episode, initial_state in enumerate(states):
            started = time.perf_counter()
            global_episode = int(episode_offset + episode)
            stem = "pursuer-{}_evader-{}".format(pursuer_name, evader_name).replace(" ", "_")
            metrics = run_episode(
                problem,
                solver,
                initial_state,
                progress_info=(stem, seed, global_episode, len(problem.times) - 1),
                timing_warmup_steps=args.timing_warmup_steps,
            )
            metrics.update({
                "seed": int(seed),
                "episode": global_episode,
                "pursuer_method": pursuer_name,
                "evader_method": evader_name,
                "wall_time_s": time.perf_counter() - started,
            })
            records.append(metrics)
            if args.limit and len(records) >= args.limit:
                return records
    return records


def summarize(records):
    numeric = [
        "total_capture", "double_capture", "capture_count", "first_capture_time", "capture_time",
        "survival_time", "pursuer_return", "evader_return",
        "collision_or_out_of_bounds", "illegal_terminal", "mean_latency_ms",
        "p50_latency_ms", "p95_latency_ms", "max_latency_ms",
        "evader_mean_latency_ms", "evader_p50_latency_ms",
        "evader_p95_latency_ms", "evader_max_latency_ms",
        "pursuer_mean_latency_ms", "pursuer_p50_latency_ms",
        "pursuer_p95_latency_ms", "pursuer_max_latency_ms", "wall_time_s",
    ]
    # Older saved chunks do not contain the detailed collision diagnostics.
    numeric.extend(key for key in (
        "obstacle_collision", "out_of_bounds", "evader_obstacle_collision",
        "pursuer_obstacle_collision", "evader_out_of_bounds", "pursuer_out_of_bounds",
    ) if any(key in record for record in records))
    summary = OrderedDict()
    for key in numeric:
        values = np.asarray([r.get(key, np.nan) for r in records], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            summary[key] = {"mean": np.nan, "std": np.nan, "ci95": np.nan, "n": 0}
            continue
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": std,
            "ci95": float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0,
            "n": int(len(values)),
        }
    for role in ("joint", "evader", "pursuer"):
        total = float(sum(r[role + "_latency_sum_ms"] for r in records))
        count = int(sum(r[role + "_latency_count"] for r in records))
        summary[role + "_step_mean_latency_ms"] = {
            "mean": total / count if count else np.nan,
            "std": np.nan,
            "ci95": np.nan,
            "n": count,
        }
    return summary


def write_records(path, records):
    if not records:
        return
    # A resumed legacy no-obstacle run may mix old chunks with the new,
    # optional collision columns. Preserve both schemas when merging.
    fieldnames = list(dict.fromkeys(key for record in records for key in record))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_summary(path, records):
    summary = summarize(records)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std", "ci95", "n"])
        for metric, values in summary.items():
            writer.writerow([metric, values["mean"], values["std"], values["ci95"], values["n"]])


def matchup_stem(pursuer_name, evader_name):
    return "pursuer-{}_evader-{}".format(pursuer_name, evader_name).replace(" ", "_")


def chunk_filename(stem, seed, start):
    return "{}.seed{}.ep{}.csv".format(stem, int(seed), int(start))


def parse_chunk_filename(name):
    if not name.endswith(".csv"):
        return None
    body = name[:-4]
    episode_at = body.rfind(".ep")
    seed_at = body.rfind(".seed", 0, episode_at if episode_at >= 0 else 0)
    if episode_at < 0 or seed_at < 0:
        return None
    try:
        seed = int(body[seed_at + 5:episode_at])
        start = int(body[episode_at + 3:])
    except ValueError:
        return None
    return body[:seed_at], seed, start


def completed_chunks(output):
    directory = os.path.join(output, "chunks")
    done = {}
    if not os.path.isdir(directory):
        return done
    for name in os.listdir(directory):
        parsed = parse_chunk_filename(name)
        if parsed is None:
            continue
        stem, seed, start = parsed
        done[(stem, seed, start)] = os.path.join(directory, name)
    return done


def write_chunk_file(output, stem, seed, start, records):
    if not records:
        return None
    directory = os.path.join(output, "chunks")
    os.makedirs(directory, exist_ok=True)
    final_path = os.path.join(directory, chunk_filename(stem, seed, start))
    temporary_path = final_path + ".tmp"
    write_records(temporary_path, records)
    os.replace(temporary_path, final_path)
    return final_path


def load_chunk_records(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def merge_chunks(output):
    """Rewrite each matchup CSV and summary from the chunks saved so far."""
    grouped = {}
    for (stem, _seed, _start), path in completed_chunks(output).items():
        grouped.setdefault(stem, []).append(path)
    for stem, paths in grouped.items():
        records = []
        for path in paths:
            records.extend(load_chunk_records(path))
        if not records:
            continue
        for record in records:
            for key, value in list(record.items()):
                if key in ("pursuer_method", "evader_method") or not isinstance(value, str):
                    continue
                if value == "":
                    record[key] = float("nan")
                    continue
                try:
                    record[key] = float(value)
                except ValueError:
                    continue
            record["seed"] = int(record["seed"])
            record["episode"] = int(record["episode"])
        records.sort(key=lambda row: (row["seed"], row["episode"]))
        write_records(os.path.join(output, stem + ".csv"), records)
        write_summary(os.path.join(output, stem + "_summary.csv"), records)
    return len(grouped)


RESUME_KEYS = (
    "seeds", "large_seeds", "episodes", "search_only_budget", "search_learning_budget",
    "large_budget", "beta_policy", "beta_value", "methods", "round_robin",
    "chunk_episodes", "model_iteration", "mappo_checkpoint", "pursuer_method",
    "evader_method", "pursuer_budget", "evader_budget", "limit",
    "benchmark_protocol", "obstacle_layout", "obstacles", "model_dir",
    "mappo_legacy", "timing_warmup_steps",
)


def resume_signature(metadata):
    signature = {key: metadata.get(key) for key in RESUME_KEYS}
    # Old no-obstacle runs predate explicit scene metadata.
    if signature["obstacle_layout"] is None:
        signature["obstacle_layout"] = "none"
    if signature["obstacles"] is None and signature["obstacle_layout"] == "none":
        signature["obstacles"] = []
    return signature


def check_resume_signature(output, metadata):
    path = os.path.join(output, "config.json")
    if not os.path.exists(path) or not completed_chunks(output):
        return
    with open(path) as handle:
        previous = json.load(handle)
    current = resume_signature(metadata)
    saved = resume_signature(previous)
    if current != saved:
        mismatched = [key for key in RESUME_KEYS if current.get(key) != saved.get(key)]
        raise SystemExit(
            "resume refused: {} already contains chunks from a different command ({}). "
            "Use the same arguments, or write to a new --output directory.".format(
                output, ", ".join(mismatched)
            )
        )


def run_matchup_task(task):
    """Spawn-safe worker for one independent matchup."""
    pursuer_name, evader_name, initial_states, seeds, episode_offsets, config = task
    args = argparse.Namespace(**config)
    problem = configure_problem(getattr(args, "obstacle_layout", "none"))
    torch.set_num_threads(args.cpu_threads)
    # Network constructors print one line per checkpoint.  With dozens of
    # workers those messages make the parent progress display unreadable.
    # Exceptions still propagate normally through the pool.
    with contextlib.redirect_stdout(io.StringIO()):
        records = evaluate_matchup(problem, pursuer_name, evader_name, initial_states, seeds, args, episode_offsets)
    stem = "pursuer-{}_evader-{}".format(pursuer_name, evader_name).replace(" ", "_")
    return stem, records


def main():
    # A process started in the background inherits SIGINT ignored.  Restore the
    # default so Ctrl+C stops the run and keeps every chunk already written.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--large-seeds", type=int, default=1,
                        help="number of seeds for any matchup containing Large-budget MCTS")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--search-only-budget", type=int, default=3000)
    parser.add_argument("--search-learning-budget", type=int, default=1000)
    parser.add_argument("--large-budget", type=int, default=50000)
    parser.add_argument("--beta-policy", type=float, default=0.5)
    parser.add_argument("--beta-value", type=float, default=0.5)
    parser.add_argument("--obstacle-layout", choices=list(OBSTACLE_LAYOUTS), default="none",
                        help="fixed obstacle map; none preserves the original benchmark")
    parser.add_argument("--output", default="../current/benchmarks")
    parser.add_argument("--root", default=None, help="repository root; defaults to the parent of code/")
    parser.add_argument("--mappo-checkpoint", default=None)
    parser.add_argument(
        "--mappo-device",
        default="cpu",
        help="device for MAPPO only, e.g. auto, cpu, cuda, or cuda:1; PUCT remains CPU/C++",
    )
    parser.add_argument("--mappo-legacy", action="store_true",
                        help="explicitly evaluate a pre-ego legacy checkpoint")
    parser.add_argument("--model-dir", default=None, help="policy/value checkpoint directory")
    parser.add_argument("--model-iteration", type=int, default=79)
    parser.add_argument("--round-robin", action="store_true")
    parser.add_argument("--parallel", action="store_true", help="run independent matchups in CPU worker processes")
    parser.add_argument("--workers", type=int, default=4, help="number of matchup worker processes")
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="PyTorch CPU threads per process; use 1 for parallel search sweeps")
    parser.add_argument(
        "--mappo-gpu-workers",
        type=int,
        default=8,
        help="maximum concurrent matchup workers that contain MAPPO (limits CUDA contexts)",
    )
    parser.add_argument("--chunk-episodes", type=int, default=0,
                        help="split each seed into episode chunks for higher CPU parallelism; 0 keeps matchup-level parallelism")
    parser.add_argument("--pursuer-method", default=None, choices=list(METHODS),
                        help="run one matchup with this pursuer; use with --evader-method")
    parser.add_argument("--evader-method", default=None, choices=list(METHODS),
                        help="run one matchup with this evader; use with --pursuer-method")
    parser.add_argument("--pursuer-budget", type=int, default=None,
                        help="override the search budget for the pursuer")
    parser.add_argument("--evader-budget", type=int, default=None,
                        help="override the search budget for the evader")
    parser.add_argument("--timing-warmup-steps", type=int, default=0,
                        help="exclude initial decision calls from latency statistics")
    parser.add_argument("--progress-step-interval", type=int, default=5,
                        help="worker progress heartbeat interval in environment steps")
    parser.add_argument("--no-progress", action="store_true", help="disable terminal progress bars")
    parser.add_argument("--limit", type=int, default=0, help="limit records per matchup for smoke tests")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    args = parser.parse_args()
    if args.workers < 1 or args.cpu_threads < 1 or args.chunk_episodes < 0:
        parser.error("workers/cpu-threads must be positive; chunk-episodes must be nonnegative")
    if args.seeds < 1 or args.large_seeds < 1 or args.episodes < 1:
        parser.error("seeds/large-seeds/episodes must be positive")
    unknown_methods = set(args.methods) - set(METHODS)
    if unknown_methods:
        parser.error("unknown methods: {}".format(", ".join(sorted(unknown_methods))))
    torch.set_num_threads(args.cpu_threads)
    args.root = args.root or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    args.mappo_checkpoint = resolve_mappo_checkpoint(args.root, args.mappo_checkpoint)
    os.makedirs(args.output, exist_ok=True)
    problem = configure_problem(args.obstacle_layout)
    scene = problem_metadata(problem, args.obstacle_layout)
    print("protocol={} obstacles={}".format(scene["benchmark_protocol"], len(problem.obstacles)), flush=True)
    seeds = list(range(1000, 1000 + args.seeds))
    large_seeds = list(range(1000, 1000 + args.large_seeds))
    initial_states = generate_initial_states(problem, seeds, args.episodes)
    large_initial_states = generate_initial_states(problem, large_seeds, args.episodes)

    # Large-budget MCTS gets its own budget; all ordinary search methods share
    # the declared benchmark budget.
    matchups = []
    if (args.pursuer_method is None) != (args.evader_method is None):
        parser.error("--pursuer-method and --evader-method must be provided together")
    if args.pursuer_method is not None:
        matchups.append((args.pursuer_method, args.evader_method))
    elif args.round_robin:
        for pursuer in args.methods:
            for evader in args.methods:
                matchups.append((pursuer, evader))
    else:
        strong = "Large-budget MCTS"
        for method in args.methods:
            matchups.append((method, strong))
            if method != strong:
                matchups.append((strong, method))

    config = {
        "root": args.root,
        "search_only_budget": args.search_only_budget,
        "search_learning_budget": args.search_learning_budget,
        "large_budget": args.large_budget,
        "beta_policy": args.beta_policy,
        "beta_value": args.beta_value,
        "mappo_checkpoint": args.mappo_checkpoint,
        "mappo_device": args.mappo_device,
        "mappo_legacy": args.mappo_legacy,
        "model_dir": args.model_dir,
        "model_iteration": args.model_iteration,
        "limit": args.limit,
        **scene,
        "latency_mode": "team_solver_calls_v2",
        "role_mask_mode": "team_aware_root_policy",
        "pursuer_budget": args.pursuer_budget,
        "evader_budget": args.evader_budget,
        "timing_warmup_steps": args.timing_warmup_steps,
        "cpu_threads": args.cpu_threads,
    }
    metadata = vars(args).copy()
    metadata.update(scene)
    metadata.update({
        "root": args.root,
        "regular_seeds": seeds,
        "large_seeds_used": large_seeds,
        "matchup_seed_rule": "matchups containing Large-budget MCTS use large_seeds; other matchups use seeds",
        "benchmark_protocol": config["benchmark_protocol"],
        "latency_mode": config["latency_mode"],
        "role_mask_mode": config["role_mask_mode"],
        "timing_context": "concurrent_workers" if args.parallel else "serial_solver_calls",
    })
    check_resume_signature(args.output, metadata)
    with open(os.path.join(args.output, "config.json"), "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    already_done = completed_chunks(args.output)
    pending = []
    skipped_chunks = 0
    skipped_episodes = Counter()
    for pursuer_name, evader_name in matchups:
        if ("MAPPO" in (pursuer_name, evader_name)) and not args.mappo_checkpoint:
            raise FileNotFoundError(
                "MAPPO is included in --methods, but no checkpoint was found. "
                "Expected current/baselines/mappo_ego_50k.pt or pass --mappo-checkpoint."
            )
        contains_large = "Large-budget MCTS" in (pursuer_name, evader_name)
        matchup_seeds = large_seeds if contains_large else seeds
        matchup_states = large_initial_states if contains_large else initial_states
        if args.parallel and args.chunk_episodes > 0:
            chunk_size = max(1, int(args.chunk_episodes))
            stem = matchup_stem(pursuer_name, evader_name)
            for seed, states in zip(matchup_seeds, matchup_states):
                for start in range(0, len(states), chunk_size):
                    if (stem, int(seed), int(start)) in already_done:
                        skipped_chunks += 1
                        skipped_episodes[stem] += len(states[start:start + chunk_size])
                        continue
                    pending.append((
                        pursuer_name,
                        evader_name,
                        [states[start:start + chunk_size]],
                        [seed],
                        [start],
                        config,
                    ))
        else:
            pending.append((pursuer_name, evader_name, matchup_states, matchup_seeds,
                            [0] * len(matchup_states), config))

    if args.parallel and (args.chunk_episodes > 0 or (len(pending) + skipped_chunks) > 1):
        if not pending:
            written = merge_chunks(args.output)
            print("all {} chunks already complete; wrote {} matchup files".format(
                skipped_chunks, written), flush=True)
            return
        if skipped_chunks:
            print("resuming: {} chunks already saved, {} remaining".format(
                skipped_chunks, len(pending)), flush=True)
        workers = max(1, min(int(args.workers), len(pending), os.cpu_count() or 1))
        contains_mappo = any("MAPPO" in (task[0], task[1]) for task in pending)
        # Each MAPPO worker may create one CUDA context.  Keep the total
        # number of workers bounded when MAPPO is on an accelerator.  CPU-only
        # runs can still use the full requested worker count.
        if contains_mappo and str(args.mappo_device).lower() not in ("cpu", "none"):
            workers = min(workers, max(1, int(args.mappo_gpu_workers)))
        unit = "episode chunks" if args.chunk_episodes > 0 else "matchups"
        print("running {} {} with {} CPU workers".format(len(pending), unit, workers), flush=True)
        context = mp.get_context("spawn")
        aggregated = {}
        task_totals = Counter(matchup_stem(task[0], task[1]) for task in pending)
        task_done = Counter()
        record_done = Counter(skipped_episodes)
        task_episode_totals = Counter(skipped_episodes)
        for task in pending:
            stem = matchup_stem(task[0], task[1])
            task_episode_totals[stem] += sum(len(states) for states in task[2])
        saved_chunks = Counter(stem for stem, _seed, _start in already_done)
        for stem, count in saved_chunks.items():
            task_totals[stem] += count
            task_done[stem] += count
        total_episodes = sum(task_episode_totals.values())
        total_steps = total_episodes * (len(problem.times) - 1)
        saved_steps = sum(skipped_episodes.values()) * (len(problem.times) - 1)
        # A plain context.Queue is enough for worker heartbeats and avoids
        # multiprocessing.Manager's socket server (which is unavailable in
        # some restricted/containerized environments).
        progress_queue = context.Queue()
        with context.Pool(
            processes=workers,
            initializer=init_worker_progress,
            initargs=(progress_queue, args.progress_step_interval),
        ) as pool:
            async_results = [pool.apply_async(run_matchup_task, (task,)) for task in pending]
            remaining_results = set(range(len(async_results)))
            active_episodes = {}
            episode_done = Counter()
            last_status = {"text": "starting"}

            def handle_progress_message(message, progress):
                kind, stem, seed, episode, amount, step, pid = message
                if kind == "step":
                    progress.update(int(amount))
                    active_episodes[int(pid)] = (stem, seed, episode, step)
                    last_status["text"] = (
                        "{} seed={} ep={}/{} step={}/{} task_ep={}/{} active={}".format(
                            stem, seed, episode, args.episodes, step,
                            len(problem.times) - 1, episode_done[stem],
                            task_episode_totals[stem], len(active_episodes)
                        )
                    )
                elif kind == "episode_done":
                    progress.update(int(amount))
                    episode_done[stem] += 1
                    active_episodes.pop(int(pid), None)
                    last_status["text"] = (
                        "{} seed={} ep={} done task_ep={}/{} active={}".format(
                            stem, seed, episode, episode_done[stem],
                            task_episode_totals[stem], len(active_episodes)
                        )
                    )

            with tqdm(
                total=total_steps,
                desc="benchmark steps",
                unit="step",
                dynamic_ncols=True,
                mininterval=0.5,
                disable=args.no_progress,
            ) as progress:
                if saved_steps:
                    progress.update(saved_steps)
                try:
                    while remaining_results:
                        drained = False
                        while True:
                            try:
                                message = progress_queue.get_nowait()
                            except Empty:
                                break
                            drained = True
                            handle_progress_message(message, progress)

                        for index in list(remaining_results):
                            result = async_results[index]
                            if not result.ready():
                                continue
                            stem, records = result.get()
                            remaining_results.remove(index)
                            seed = int(pending[index][3][0])
                            start = int(pending[index][4][0])
                            write_chunk_file(args.output, stem, seed, start, records)
                            aggregated.setdefault(stem, []).extend(records)
                            task_done[stem] += 1
                            record_done[stem] += len(records)
                            tqdm.write(
                                "completed chunk for {}: chunks {}/{}, episodes {}/{}".format(
                                    stem, task_done[stem], task_totals[stem],
                                    record_done[stem], task_episode_totals[stem]
                                ),
                                file=sys.stdout,
                            )
                            drained = True
                            if task_done[stem] == task_totals[stem]:
                                tqdm.write(
                                    "finished {}: {} episodes".format(stem, record_done[stem]),
                                    file=sys.stdout,
                                )

                        progress.set_postfix_str(
                            "{} chunks={}/{} episodes={}/{}".format(
                                last_status["text"],
                                sum(task_done.values()), sum(task_totals.values()),
                                sum(record_done.values()), total_episodes,
                            ),
                            refresh=False,
                        )
                        if remaining_results and not drained:
                            time.sleep(0.05)

                    # Worker results can reach the pool result pipe before all
                    # heartbeat messages have been flushed through the progress
                    # queue.  At this point every task has completed, so force
                    # the display to its known deterministic total instead of
                    # leaving a misleading partial percentage (for example 34%).
                    if progress.n < total_steps:
                        progress.update(total_steps - progress.n)
                    elif progress.n > total_steps:
                        progress.n = total_steps
                    progress.set_postfix_str(
                        "all tasks complete chunks={}/{} episodes={}/{}".format(
                            sum(task_totals.values()), sum(task_totals.values()),
                            total_episodes, total_episodes
                        ),
                        refresh=True,
                    )
                except KeyboardInterrupt:
                    print(
                        "\nStopped. Completed episodes are in {}/chunks. "
                        "Run the same command to continue.".format(args.output),
                        flush=True,
                    )
                    pool.terminate()
                    merge_chunks(args.output)
                    raise SystemExit(130)
        merge_chunks(args.output)
        return

    for pursuer_name, evader_name, matchup_states, matchup_seeds, episode_offsets, _ in pending:
        print(
            "running pursuer={} evader={} budgets(pursuer={}, evader={}) seeds={}".format(
                pursuer_name,
                evader_name,
                method_budget(pursuer_name, args, role="pursuer"),
                method_budget(evader_name, args, role="evader"),
                len(matchup_seeds),
            ),
            flush=True,
        )
        records = evaluate_matchup(problem, pursuer_name, evader_name, matchup_states, matchup_seeds, args, episode_offsets)
        stem = "pursuer-{}_evader-{}".format(pursuer_name, evader_name).replace(" ", "_")
        write_records(os.path.join(args.output, stem + ".csv"), records)
        write_summary(os.path.join(args.output, stem + "_summary.csv"), records)
        print("  wrote {} records".format(len(records)), flush=True)


if __name__ == "__main__":
    main()
