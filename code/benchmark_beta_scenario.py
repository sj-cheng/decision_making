"""Factorial component ablation of policy/value guidance at fixed budget.

Single-scenario ablation (v5 protocol: no obstacles, direct propagation) of
the two guidance components of the fused team-alternating MCTS controller:

    beta_policy (bpi): probability that node expansion proposes the distilled
                       policy action instead of a uniform sample
    beta_value  (bv):  probability that a leaf is evaluated by the value
                       network instead of a random rollout

The tested side plays with one (bpi, bv) cell at B=1000; the opponent is a
fixed reference (bpi=0.5, bv=0.5, B=1000) in every cell.  Both roles are
evaluated: ``pursuit`` (tested pursuer vs reference evader) and ``evasion``
(tested evader vs reference pursuer).  The (0, 0) cell is built as "Search
Only" (no networks loaded), which is semantically identical to bpi=bv=0.

All simulator/solver machinery is imported from ``benchmark_baselines``; this
file only adds per-side beta assignment, fixed-size chunk tasks, chunked
checkpointing, and table aggregation.  No existing file is modified.

Usage (from ``code``; the C++ bindings require the CPython-3.11 env):

    python benchmark_beta_scenario.py --mode selftest
    python benchmark_beta_scenario.py --mode run --workers 80
    python benchmark_beta_scenario.py --mode aggregate

Outputs under ``<output>`` (default ``../current/benchmarks/beta_ablation_v1``):

    chunks/<role>/<cell>/chunk_s<seed>_e<offset>.csv   per-chunk records
    <role>/<cell>.csv                                  merged per-episode records
    <role>/<cell>_summary.csv                          mean/std/ci95 per metric
    tables/ablation_<role>.md / .csv                   paper-ready ablation table
    config.json                                        full protocol record
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import argparse
import contextlib
import csv
import glob
import io
import json
import multiprocessing as mp
import sys
import tempfile
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_baselines import (
    build_solver,
    configure_problem,
    generate_initial_states,
    run_episode,
    summarize,
    write_records,
    write_summary,
)
from solvers.baselines import RoleMaskedSolver, TeamSolver

SEARCH_BUDGET = 1000
REFERENCE_BETAS = (0.5, 0.5)
DEFAULT_CELLS = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
ROLES = ("pursuit", "evasion")
PROTOCOL = "example8_global_no_obstacle_direct_v1__beta_ablation"


def cell_stem(cell):
    return "bpi{:g}_bv{:g}".format(float(cell[0]), float(cell[1]))


def parse_cell(token):
    parts = str(token).split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "cell must be 'bpi,bv', got {!r}".format(token))
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "cell must be numeric 'bpi,bv', got {!r}".format(token))


def method_label(cell):
    if tuple(cell) == (0.0, 0.0):
        return "Search Only (B={})".format(SEARCH_BUDGET)
    return "S+L(bpi={:g},bv={:g},B={})".format(cell[0], cell[1], SEARCH_BUDGET)


def build_side(problem, betas, root, model_dir=None, model_iteration=79):
    """Build the joint solver for one side with explicit guidance mixing.

    Unlike ``benchmark_baselines.evaluate_matchup`` (which assigns the same
    betas to both sides of a matchup), here each side gets its own
    (beta_policy, beta_value), so the tested side can vary while the
    reference opponent stays fixed at ``REFERENCE_BETAS``.
    """
    bpi, bv = float(betas[0]), float(betas[1])
    if bpi == 0.0 and bv == 0.0:
        return build_solver(
            problem, "Search Only", root, SEARCH_BUDGET,
            model_dir=model_dir, model_iteration=model_iteration)
    return build_solver(
        problem, "Search + Learning", root, SEARCH_BUDGET,
        model_dir=model_dir, model_iteration=model_iteration,
        beta_policy=bpi, beta_value=bv)


def chunk_path(output, role, cell, seed, offset):
    directory = os.path.join(output, "chunks", role, cell_stem(cell))
    return os.path.join(directory, "chunk_s{}_e{}.csv".format(int(seed), int(offset)))


def run_chunk_task(task):
    """Spawn-safe worker for one episode chunk of one (cell, role).

    Solvers are constructed inside the worker: C_PUCT instances hold pybind
    objects that cannot cross process boundaries.  The chunk CSV is written
    by the worker itself so completed chunks survive an interrupted run.
    """
    cell, role, seed, episode_offset, initial_states, config = task
    problem = configure_problem()
    path = chunk_path(config["output"], role, cell, seed, episode_offset)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Network constructors print one line per checkpoint; with dozens of
    # workers that output makes the parent progress display unreadable.
    with contextlib.redirect_stdout(io.StringIO()):
        tested = build_side(
            problem, cell, config["root"],
            model_dir=config["model_dir"], model_iteration=config["model_iteration"])
        reference = build_side(
            problem, REFERENCE_BETAS, config["root"],
            model_dir=config["model_dir"], model_iteration=config["model_iteration"])
        if role == "pursuit":
            pursuer = RoleMaskedSolver(tested, problem.pursuers)
            evader = RoleMaskedSolver(reference, problem.evaders)
        else:
            pursuer = RoleMaskedSolver(reference, problem.pursuers)
            evader = RoleMaskedSolver(tested, problem.evaders)
        solver = TeamSolver(evader, pursuer)
        records = []
        for index, initial_state in enumerate(initial_states):
            started = time.perf_counter()
            metrics = run_episode(
                problem, solver, initial_state,
                timing_warmup_steps=config["timing_warmup_steps"])
            metrics.update({
                "seed": int(seed),
                "episode": int(episode_offset + index),
                "beta_policy": float(cell[0]),
                "beta_value": float(cell[1]),
                "role": role,
                "pursuer_method": method_label(cell) if role == "pursuit"
                                  else method_label(REFERENCE_BETAS) + " [ref]",
                "evader_method": method_label(REFERENCE_BETAS) + " [ref]" if role == "pursuit"
                                 else method_label(cell),
                "wall_time_s": time.perf_counter() - started,
            })
            records.append(metrics)
        write_records(path, records)
    return role, cell, path, len(records)


def read_records(path):
    """Read a records CSV back with numeric type restoration."""
    records = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            record = {}
            for key, value in row.items():
                try:
                    record[key] = int(value)
                    continue
                except (TypeError, ValueError):
                    pass
                try:
                    record[key] = float(value)
                except (TypeError, ValueError):
                    record[key] = value
            records.append(record)
    return records


def merge_cell(output, role, cell):
    """Concatenate all chunk CSVs of one (role, cell) into the merged CSVs."""
    pattern = os.path.join(output, "chunks", role, cell_stem(cell), "chunk_s*_e*.csv")
    paths = sorted(glob.glob(pattern))
    records = []
    for path in paths:
        records.extend(read_records(path))
    if not records:
        return 0, 0
    records.sort(key=lambda row: (int(row["seed"]), int(row["episode"])))
    directory = os.path.join(output, role)
    os.makedirs(directory, exist_ok=True)
    stem = cell_stem(cell)
    write_records(os.path.join(directory, stem + ".csv"), records)
    write_summary(os.path.join(directory, stem + "_summary.csv"), records)
    return len(records), len(paths)


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return center - half, center + half


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, 0
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    ci = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return float(np.mean(values)), ci, len(values)


def aggregate(output, roles, cells):
    """Build the pursuit/evasion ablation tables from merged cell CSVs."""
    tables_dir = os.path.join(output, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    for role in roles:
        rows = []
        for cell in cells:
            path = os.path.join(output, role, cell_stem(cell) + ".csv")
            if not os.path.exists(path):
                count, _ = merge_cell(output, role, cell)
                if count == 0:
                    print("aggregate: skipping {} {} (no records)".format(role, cell_stem(cell)))
                    continue
            records = read_records(path)
            n = len(records)
            doubles = sum(int(r["double_capture"]) for r in records)
            invalids = sum(int(r["collision_or_out_of_bounds"]) for r in records)
            return_key = "pursuer_return" if role == "pursuit" else "evader_return"
            dur_mean, dur_ci, _ = mean_ci([r["survival_time"] for r in records])
            ret_mean, ret_ci, _ = mean_ci([r[return_key] for r in records])
            lo_d, hi_d = wilson_interval(doubles, n)
            lo_i, hi_i = wilson_interval(invalids, n)
            rows.append({
                "beta_policy": cell[0],
                "beta_value": cell[1],
                "episodes": n,
                "double_capture_mean": doubles / n,
                "double_capture_wilson_lo": lo_d,
                "double_capture_wilson_hi": hi_d,
                "duration_mean": dur_mean,
                "duration_ci95": dur_ci,
                "return_mean": ret_mean,
                "return_ci95": ret_ci,
                "invalid_mean": invalids / n,
                "invalid_wilson_lo": lo_i,
                "invalid_wilson_hi": hi_i,
            })
        if not rows:
            continue
        capture_title = "Double capture (%)" if role == "pursuit" else "Double caught (%)"
        return_title = "Pursuer return" if role == "pursuit" else "Evader return"
        lines = [
            "### {} ablation at B={} (reference: {} )".format(
                role, SEARCH_BUDGET, method_label(REFERENCE_BETAS)),
            "",
            "| βπ | βv | {} | Duration (steps) | {} | Invalid (%) | n |".format(
                capture_title, return_title),
            "|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            lines.append(
                "| {:g} | {:g} | {:.1f} [{:.1f}, {:.1f}] | {:.1f} ± {:.1f} | "
                "{:.3f} ± {:.3f} | {:.1f} [{:.1f}, {:.1f}] | {} |".format(
                    row["beta_policy"], row["beta_value"],
                    100 * row["double_capture_mean"],
                    100 * row["double_capture_wilson_lo"],
                    100 * row["double_capture_wilson_hi"],
                    row["duration_mean"], row["duration_ci95"],
                    row["return_mean"], row["return_ci95"],
                    100 * row["invalid_mean"],
                    100 * row["invalid_wilson_lo"],
                    100 * row["invalid_wilson_hi"],
                    row["episodes"],
                ))
        text = "\n".join(lines)
        with open(os.path.join(tables_dir, "ablation_{}.md".format(role)), "w") as handle:
            handle.write(text + "\n")
        with open(os.path.join(tables_dir, "ablation_{}.csv".format(role)), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("\n" + text + "\n")
        print("wrote tables to {}".format(tables_dir))


def mode_run(args, cells, roles, seeds):
    problem = configure_problem()
    initial_states = generate_initial_states(problem, seeds, args.episodes)
    os.makedirs(args.output, exist_ok=True)
    config = {
        "output": args.output,
        "root": args.root,
        "model_dir": args.model_dir,
        "model_iteration": args.model_iteration,
        "timing_warmup_steps": args.timing_warmup_steps,
    }
    metadata = {
        "protocol": PROTOCOL,
        "benchmark_protocol": "example8_global_no_obstacle_direct_v1",
        "latency_mode": "team_solver_calls_v2",
        "role_mask_mode": "team_aware_root_policy",
        "search_budget": SEARCH_BUDGET,
        "reference_betas": list(REFERENCE_BETAS),
        "cells": [list(cell) for cell in cells],
        "roles": list(roles),
        "seeds": seeds,
        "episodes_per_seed": args.episodes,
        "chunk_episodes": args.chunk_episodes,
        "model_dir": args.model_dir,
        "model_iteration": args.model_iteration,
        "initial_state_rule": "same pre-generated batch shared by all cells (paired comparison)",
    }
    with open(os.path.join(args.output, "config.json"), "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    pending = []
    skipped = 0
    for role in roles:
        for cell in cells:
            for seed_index, seed in enumerate(seeds):
                states = initial_states[seed_index]
                for start in range(0, len(states), args.chunk_episodes):
                    if os.path.exists(chunk_path(args.output, role, cell, seed, start)):
                        skipped += 1
                        continue
                    pending.append((
                        cell, role, seed, start,
                        states[start:start + args.chunk_episodes],
                        config,
                    ))
                    if args.limit and len(pending) >= args.limit:
                        break
                if args.limit and len(pending) >= args.limit:
                    break
            if args.limit and len(pending) >= args.limit:
                break
        if args.limit and len(pending) >= args.limit:
            break

    expected = sum(
        len(range(0, args.episodes, args.chunk_episodes)) * len(seeds)
        for _ in roles for _ in cells
    )
    print("chunk tasks: {} pending, {} already complete (expected total {})".format(
        len(pending), skipped, expected), flush=True)

    if pending:
        workers = max(1, min(int(args.workers), len(pending), os.cpu_count() or 1))
        print("running {} chunk tasks with {} CPU workers".format(len(pending), workers), flush=True)
        context = mp.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            with tqdm(total=len(pending), desc="chunks", unit="chunk",
                      dynamic_ncols=True, disable=args.no_progress) as bar:
                for role, cell, path, count in pool.imap_unordered(run_chunk_task, pending):
                    bar.update(1)
                    tqdm.write(
                        "completed {} {}: {} episodes -> {}".format(
                            role, cell_stem(cell), count, path),
                        file=sys.stdout)

    for role in roles:
        for cell in cells:
            n_records, n_chunks = merge_cell(args.output, role, cell)
            if n_records == 0:
                print("merged {} {}: no records yet".format(role, cell_stem(cell)))
            else:
                print("merged {} {}: {} episodes from {} chunks".format(
                    role, cell_stem(cell), n_records, n_chunks), flush=True)


def mode_selftest(args):
    output = tempfile.mkdtemp(prefix="beta_ablation_selftest_")
    problem = configure_problem()
    assert problem.obstacles == [], "v5 protocol expects no obstacles"
    assert problem.evader_spawn_near_obstacles is False
    assert problem.use_minco_dynamics is False, "v5 protocol expects direct propagation"

    # The (0, 0) cell must be a pure-search solver at the ablation budget,
    # and a guided cell must forward its betas into the C++ settings.
    # (C_PUCT wraps oracles into cpp objects, so oracle presence is not a
    # meaningful Python-side assertion; the forwarded settings are what the
    # search actually consumes.)
    search_only = build_side(problem, (0.0, 0.0), args.root,
                             model_dir=args.model_dir, model_iteration=args.model_iteration)
    assert search_only.solver_settings.beta_policy == 0.0
    assert search_only.solver_settings.beta_value == 0.0
    assert search_only.solver_settings.number_simulations == SEARCH_BUDGET
    guided = build_side(problem, (0.5, 0.5), args.root,
                        model_dir=args.model_dir, model_iteration=args.model_iteration)
    assert guided.solver_settings.beta_policy == 0.5
    assert guided.solver_settings.beta_value == 0.5
    assert guided.solver_settings.number_simulations == SEARCH_BUDGET
    del search_only, guided

    np.random.seed(1000)
    states = [np.array(problem.initialize(), copy=True) for _ in range(2)]
    config = {
        "output": output,
        "root": args.root,
        "model_dir": args.model_dir,
        "model_iteration": args.model_iteration,
        "timing_warmup_steps": 0,
    }
    print("selftest: running 2 cells x 2 episodes at B={} (a few minutes) ...".format(
        SEARCH_BUDGET), flush=True)
    for cell in ((0.0, 0.0), (0.5, 0.5)):
        role, _, path, count = run_chunk_task((cell, "pursuit", 1000, 0, states, config))
        assert count == 2 and os.path.exists(path)
        records = read_records(path)
        assert len(records) == 2
        summary = summarize(records)
        assert summary["double_capture"]["n"] == 2
        for record in records:
            assert record["double_capture"] in (0, 1)
            assert record["steps"] >= 1
            assert record["seed"] == 1000
        n_records, n_chunks = merge_cell(output, role, cell)
        assert n_records == 2 and n_chunks == 1
    print("selftest OK (scratch output: {})".format(output))


def main():
    parser = argparse.ArgumentParser(
        description="Factorial (beta_policy, beta_value) component ablation at fixed budget.")
    parser.add_argument("--mode", choices=("run", "aggregate", "selftest"), default="run")
    parser.add_argument("--cells", nargs="+", type=parse_cell, default=None,
                        help="cells as 'bpi,bv', e.g. --cells 0,0 0.5,0 0,0.5 0.5,0.5 1,1")
    parser.add_argument("--roles", nargs="+", choices=list(ROLES), default=list(ROLES))
    parser.add_argument("--seeds", type=int, default=5,
                        help="number of evaluation seeds (1000..1000+seeds-1)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="episodes per seed per cell")
    parser.add_argument("--chunk-episodes", type=int, default=20,
                        help="episodes per parallel task")
    parser.add_argument("--workers", type=int, default=80)
    parser.add_argument("--output", default="../current/benchmarks/beta_ablation_v1")
    parser.add_argument("--root", default=None,
                        help="repository root; defaults to the parent of code/")
    parser.add_argument("--model-dir", default=None, help="policy/value checkpoint directory")
    parser.add_argument("--model-iteration", type=int, default=79)
    parser.add_argument("--timing-warmup-steps", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0,
                        help="cap the number of chunk tasks (smoke tests)")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    args.root = args.root or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    cells = [tuple(cell) for cell in (args.cells or DEFAULT_CELLS)]
    roles = list(args.roles)
    seeds = list(range(1000, 1000 + args.seeds))

    if args.mode == "selftest":
        mode_selftest(args)
    elif args.mode == "aggregate":
        aggregate(args.output, roles, cells)
    else:
        mode_run(args, cells, roles, seeds)


if __name__ == "__main__":
    main()
