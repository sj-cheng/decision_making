#!/usr/bin/env python3
"""Generate paper-ready evaluation and dataset diagnostic figures.

This script is intentionally separate from train.py: it reads the completed
checkpoints and datasets under current/ and writes publication figures under
current/document/.
"""

from __future__ import annotations

import argparse
from copy import copy
import csv
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
from queue import Queue

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
CURRENT_DIR = ROOT_DIR / "current"
MODEL_DIR = CURRENT_DIR / "models"
DATA_DIR = CURRENT_DIR / "data"
DOCUMENT_DIR = CURRENT_DIR / "document"
PREVIEW_DIR = DOCUMENT_DIR / "preview"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from learning.oracles import get_oracles  # noqa: E402
from problems.problem import get_problem  # noqa: E402
from run import run_instance  # noqa: E402
from solvers.solver import get_solver  # noqa: E402


DEFAULT_ROUNDS = list(range(1, 50, 2))
EVADER_COLOR = "#1b9e77"
PURSUER_COLOR = "#d95f02"
ACCENT_COLOR = "#386cb0"
SECONDARY_COLOR = "#7570b3"
GRID_COLOR = "#d9d9d9"
_WORKER_CONTEXT = None


ROLLOUT_FIELDS = [
    "round_l",
    "trial",
    "episode_steps",
    "terminal_time",
    "capture_any",
    "capture_all_evaders",
    "first_capture_time",
    "full_capture_time",
    "surviving_evaders",
    "active_pursuers",
    "boundary_violation",
    "evader_return_mean",
    "pursuer_return_mean",
    "total_return_robot0",
    "total_return_robot1",
    "total_return_robot2",
    "total_return_robot3",
]


SUMMARY_FIELDS = [
    "round_l",
    "num_trials",
    "capture_rate_mean",
    "capture_any_rate_mean",
    "episode_steps_mean",
    "episode_steps_se",
    "terminal_time_mean",
    "terminal_time_se",
    "first_capture_time_mean",
    "first_capture_time_se",
    "full_capture_time_mean",
    "full_capture_time_se",
    "evader_return_mean",
    "evader_return_se",
    "pursuer_return_mean",
    "pursuer_return_se",
    "surviving_evaders_mean",
    "surviving_evaders_se",
    "active_pursuers_mean",
    "active_pursuers_se",
    "boundary_violation_rate_mean",
]


def parse_rounds(value: str) -> list[int]:
    if value.strip().lower() in {"default", "paper"}:
        return DEFAULT_ROUNDS[:]
    rounds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            rounds.extend(range(int(lo), int(hi) + 1))
        else:
            rounds.append(int(part))
    return sorted(set(rounds))


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Noto Serif CJK JP",
                "Noto Sans CJK JP",
                "DejaVu Serif",
            ],
            "font.sans-serif": [
                "Noto Sans CJK JP",
                "Noto Serif CJK JP",
                "DejaVu Sans",
            ],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2.2,
            "axes.linewidth": 0.8,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def ensure_dirs() -> None:
    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def solver_slug(solver_name: str) -> str:
    return solver_name.lower().replace("c_puct", "cpuct")


def summary_name(num_trials: int, solver_name: str) -> str:
    return f"performance_summary_{solver_slug(solver_name)}_n{num_trials}"


def cache_stem(num_trials: int, solver_name: str) -> str:
    return f"performance_rollouts_{solver_slug(solver_name)}_n{num_trials}"


def model_paths(round_l: int, num_robots: int) -> tuple[list[str], str]:
    policies = [
        str(MODEL_DIR / f"model_policy_l{round_l}_i{robot}.pt")
        for robot in range(num_robots)
    ]
    value = str(MODEL_DIR / f"model_value_l{round_l}.pt")
    return policies, value


def require_file(path: str | Path) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")


def build_solver(problem, round_l: int, args):
    policy_paths, value_path = model_paths(round_l, problem.num_robots)
    for path in policy_paths:
        require_file(path)
    require_file(value_path)
    policy_oracle, value_oracle = get_oracles(
        problem,
        value_oracle_name="deterministic",
        value_oracle_path=value_path,
        policy_oracle_name="gaussian",
        policy_oracle_paths=policy_paths,
    )
    return get_solver(
        args.solver_name,
        policy_oracle=policy_oracle,
        value_oracle=value_oracle,
        search_depth=args.search_depth,
        number_simulations=args.number_simulations,
        C_pw=args.C_pw,
        alpha_pw=args.alpha_pw,
        C_exp=args.C_exp,
        alpha_exp=args.alpha_exp,
        beta_policy=args.beta_policy,
        beta_value=args.beta_value,
        vis_on=False,
    ), policy_oracle, value_oracle


def total_return(problem, sim_result: dict) -> np.ndarray:
    returns = np.zeros(problem.num_robots, dtype=float)
    rewards = np.asarray(sim_result["rewards"]).squeeze(axis=2)
    if rewards.size == 0:
        return returns
    if rewards.ndim == 1:
        rewards = rewards.reshape(1, -1)
    for step, reward in enumerate(rewards):
        returns += (problem.gamma ** step) * reward
    return returns


def capture_times(problem, states: np.ndarray) -> tuple[float, float]:
    active_evader_counts = [
        problem.active_evader_count(state.reshape(problem.state_dim, 1))
        for state in states
    ]
    initial = active_evader_counts[0] if active_evader_counts else len(problem.evaders)
    first_capture = math.nan
    full_capture = math.nan
    for idx, count in enumerate(active_evader_counts):
        if math.isnan(first_capture) and count < initial:
            first_capture = float(states[idx, problem.time_idx, 0])
        if count == 0:
            full_capture = float(states[idx, problem.time_idx, 0])
            break
    return first_capture, full_capture


def rollout_metrics(problem, sim_result: dict, round_l: int, trial: int) -> dict[str, float]:
    states = np.asarray(sim_result["states"])
    returns = total_return(problem, sim_result)
    final_state = states[-1]
    final_col = final_state.reshape(problem.state_dim, 1)
    surviving_evaders = problem.active_evader_count(final_col)
    active_pursuers = problem.active_pursuer_count(final_col)
    first_capture, full_capture = capture_times(problem, states)
    terminal_time = float(final_col[problem.time_idx, 0])
    capture_any = 0 if math.isnan(first_capture) else 1
    capture_all = 1 if surviving_evaders == 0 else 0
    boundary_violation = 0 if problem.is_valid(final_col) else 1
    metrics: dict[str, float] = {
        "round_l": round_l,
        "trial": trial,
        "episode_steps": max(0, len(states) - 1),
        "terminal_time": terminal_time,
        "capture_any": capture_any,
        "capture_all_evaders": capture_all,
        "first_capture_time": first_capture,
        "full_capture_time": full_capture,
        "surviving_evaders": surviving_evaders,
        "active_pursuers": active_pursuers,
        "boundary_violation": boundary_violation,
        "evader_return_mean": float(np.mean(returns[problem.evaders])),
        "pursuer_return_mean": float(np.mean(returns[problem.pursuers])),
    }
    for robot in range(problem.num_robots):
        metrics[f"total_return_robot{robot}"] = float(returns[robot])
    return metrics


def evaluate_round(round_l: int, args) -> list[dict[str, float]]:
    if args.parallel and args.num_trials > 1:
        return evaluate_round_parallel(round_l, args)

    problem = get_problem("example8")
    solver, policy_oracle, value_oracle = build_solver(problem, round_l, args)
    instance = {
        "problem": problem,
        "solver": solver,
        "policy_oracle": policy_oracle,
        "value_oracle": value_oracle,
    }
    rows: list[dict[str, float]] = []
    print(
        f"Evaluating l={round_l} with {args.solver_name}, "
        f"{args.num_trials} trials, {args.number_simulations} simulations/search..."
    )
    for trial in range(args.num_trials):
        np.random.seed(args.seed + round_l * 100_000 + trial)
        instance["initial_state"] = problem.initialize()
        sim_result = run_instance(
            0,
            Queue(),
            len(problem.times),
            instance,
            verbose=False,
            tqdm_on=False,
        )
        rows.append(rollout_metrics(problem, sim_result, round_l, trial))
        if (trial + 1) % max(1, args.progress_every) == 0:
            print(f"  l={round_l}: completed {trial + 1}/{args.num_trials} trials")
    return rows


def _init_eval_worker(round_l: int, args) -> None:
    global _WORKER_CONTEXT
    problem = get_problem("example8")
    solver, policy_oracle, value_oracle = build_solver(problem, round_l, args)
    _WORKER_CONTEXT = {
        "round_l": round_l,
        "args": args,
        "problem": problem,
        "instance": {
            "problem": problem,
            "solver": solver,
            "policy_oracle": policy_oracle,
            "value_oracle": value_oracle,
        },
    }


def _eval_trial_worker(trial: int) -> dict[str, float]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Evaluation worker was not initialized")
    round_l = _WORKER_CONTEXT["round_l"]
    args = _WORKER_CONTEXT["args"]
    problem = _WORKER_CONTEXT["problem"]
    instance = _WORKER_CONTEXT["instance"]
    np.random.seed(args.seed + round_l * 100_000 + trial)
    instance["initial_state"] = problem.initialize()
    sim_result = run_instance(
        0,
        Queue(),
        len(problem.times),
        instance,
        verbose=False,
        tqdm_on=False,
    )
    return rollout_metrics(problem, sim_result, round_l, trial)


def evaluate_round_parallel(round_l: int, args) -> list[dict[str, float]]:
    workers = min(args.num_workers, args.num_trials)
    print(
        f"Evaluating l={round_l} with {args.solver_name}, "
        f"{args.num_trials} trials, {args.number_simulations} simulations/search, "
        f"{workers} workers..."
    )
    rows: list[dict[str, float]] = []
    ctx = mp.get_context(args.mp_context)
    with ctx.Pool(
        processes=workers,
        initializer=_init_eval_worker,
        initargs=(round_l, args),
    ) as pool:
        for idx, row in enumerate(pool.imap_unordered(_eval_trial_worker, range(args.num_trials)), start=1):
            rows.append(row)
            if idx % max(1, args.progress_every) == 0 or idx == args.num_trials:
                print(f"  l={round_l}: completed {idx}/{args.num_trials} trials")
    return sorted(rows, key=lambda row: int(row["trial"]))


def write_csv(path: Path, rows: list[dict[str, float]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, math.nan) for field in fields})


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                key: (float(value) if value not in {"", "nan", "NaN"} else math.nan)
                for key, value in row.items()
            }
            for row in reader
        ]


def write_npz(path: Path, rows: list[dict[str, float]]) -> None:
    arrays = {
        field: np.array([row.get(field, math.nan) for row in rows], dtype=float)
        for field in ROLLOUT_FIELDS
    }
    np.savez(path, **arrays)


def mean_and_se(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1:
        return float(values[0]), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(values.size))


def summarize_rollouts(rows: list[dict[str, float]], rounds: list[int]) -> list[dict[str, float]]:
    summary: list[dict[str, float]] = []
    for round_l in rounds:
        group = [row for row in rows if int(row["round_l"]) == round_l]
        if not group:
            continue
        arr = {field: np.array([row.get(field, math.nan) for row in group], dtype=float) for field in ROLLOUT_FIELDS}
        episode_mean, episode_se = mean_and_se(arr["episode_steps"])
        terminal_mean, terminal_se = mean_and_se(arr["terminal_time"])
        first_mean, first_se = mean_and_se(arr["first_capture_time"])
        full_mean, full_se = mean_and_se(arr["full_capture_time"])
        evader_mean, evader_se = mean_and_se(arr["evader_return_mean"])
        pursuer_mean, pursuer_se = mean_and_se(arr["pursuer_return_mean"])
        surv_mean, surv_se = mean_and_se(arr["surviving_evaders"])
        purs_active_mean, purs_active_se = mean_and_se(arr["active_pursuers"])
        summary.append(
            {
                "round_l": round_l,
                "num_trials": len(group),
                "capture_rate_mean": float(np.nanmean(arr["capture_all_evaders"])),
                "capture_any_rate_mean": float(np.nanmean(arr["capture_any"])),
                "episode_steps_mean": episode_mean,
                "episode_steps_se": episode_se,
                "terminal_time_mean": terminal_mean,
                "terminal_time_se": terminal_se,
                "first_capture_time_mean": first_mean,
                "first_capture_time_se": first_se,
                "full_capture_time_mean": full_mean,
                "full_capture_time_se": full_se,
                "evader_return_mean": evader_mean,
                "evader_return_se": evader_se,
                "pursuer_return_mean": pursuer_mean,
                "pursuer_return_se": pursuer_se,
                "surviving_evaders_mean": surv_mean,
                "surviving_evaders_se": surv_se,
                "active_pursuers_mean": purs_active_mean,
                "active_pursuers_se": purs_active_se,
                "boundary_violation_rate_mean": float(np.nanmean(arr["boundary_violation"])),
            }
        )
    return summary


def load_or_evaluate(args, rounds: list[int]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    rollouts_csv = DOCUMENT_DIR / f"{cache_stem(args.num_trials, args.solver_name)}.csv"
    rollouts_npz = DOCUMENT_DIR / f"{cache_stem(args.num_trials, args.solver_name)}.npz"
    summary_csv = DOCUMENT_DIR / f"{summary_name(args.num_trials, args.solver_name)}.csv"
    if rollouts_csv.is_file() and summary_csv.is_file() and not args.force_eval:
        print(f"Using cached rollout metrics: {rollouts_csv}")
        rows = read_csv(rollouts_csv)
        summary = read_csv(summary_csv)
        return rows, summary

    rows: list[dict[str, float]] = []
    for round_l in rounds:
        rows.extend(evaluate_round(round_l, args))
    summary = summarize_rollouts(rows, rounds)
    write_csv(rollouts_csv, rows, ROLLOUT_FIELDS)
    write_csv(summary_csv, summary, SUMMARY_FIELDS)
    write_npz(rollouts_npz, rows)
    return rows, summary


def style_axis(ax) -> None:
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def style_paper_axis(ax) -> None:
    ax.grid(True, color="#e7e7e7", linewidth=0.65, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3.0, width=0.7)


def set_round_ticks(ax, x: np.ndarray) -> None:
    if x.size == 0:
        return
    ticks = np.linspace(float(np.min(x)), float(np.max(x)), 6)
    ticks = np.unique(np.rint(ticks).astype(int))
    ax.set_xticks(ticks)


def panel_label(ax, label: str) -> None:
    ax.text(
        0.0,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
    )


def save_figure(fig, filename: str) -> Path:
    pdf_path = DOCUMENT_DIR / filename
    png_path = PREVIEW_DIR / filename.replace(".pdf", ".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return pdf_path


def summary_arrays(summary: list[dict[str, float]]) -> dict[str, np.ndarray]:
    return {
        field: np.array([row.get(field, math.nan) for row in summary], dtype=float)
        for field in SUMMARY_FIELDS
    }


def line_with_se(ax, x, y, se, label, color, marker="o") -> None:
    ax.plot(x, y, marker=marker, markersize=4.2, color=color, label=label)
    if se is not None:
        se = np.asarray(se, dtype=float)
        if np.isfinite(se).any():
            ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=color, alpha=0.16, linewidth=0)


def plot_performance_main(summary: list[dict[str, float]]) -> Path:
    data = summary_arrays(summary)
    x = data["round_l"]
    fig, axs = plt.subplots(2, 2, figsize=(7.35, 5.25), constrained_layout=True)
    axs = axs.ravel()

    axs[0].plot(x, data["capture_rate_mean"], marker="o", markersize=4.2, color=PURSUER_COLOR, label="全部捕获")
    axs[0].plot(x, data["capture_any_rate_mean"], marker="s", markersize=4.2, color=ACCENT_COLOR, label="任意捕获")
    axs[0].set_ylabel("捕获率")
    axs[0].set_ylim(0.75, 1.02)
    axs[0].legend(frameon=False, loc="lower right", fontsize=8)
    panel_label(axs[0], "a")

    line_with_se(axs[1], x, data["first_capture_time_mean"], data["first_capture_time_se"], "首次捕获", SECONDARY_COLOR)
    line_with_se(axs[1], x, data["full_capture_time_mean"], data["full_capture_time_se"], "全部捕获", PURSUER_COLOR, marker="s")
    axs[1].set_ylabel("捕获时间")
    axs[1].legend(frameon=False, loc="upper right", fontsize=8)
    panel_label(axs[1], "b")

    line_with_se(axs[2], x, data["evader_return_mean"], data["evader_return_se"], "逃跑方", EVADER_COLOR)
    line_with_se(axs[2], x, data["pursuer_return_mean"], data["pursuer_return_se"], "追捕方", PURSUER_COLOR)
    axs[2].set_ylabel("累计回报")
    axs[2].legend(frameon=False, loc="center right", fontsize=8)
    panel_label(axs[2], "c")

    line_with_se(axs[3], x, data["episode_steps_mean"], data["episode_steps_se"], "回合长度", ACCENT_COLOR)
    axs[3].set_ylabel("步数")
    panel_label(axs[3], "d")

    for ax in axs:
        ax.set_xlabel("训练轮次 $l$")
        set_round_ticks(ax, x)
        style_paper_axis(ax)
    return save_figure(fig, "paper_performance_main.pdf")


def plot_performance(summary: list[dict[str, float]], args) -> list[Path]:
    data = summary_arrays(summary)
    x = data["round_l"]
    outputs: list[Path] = []
    title_suffix = f"{args.solver_name}，样本数 n={args.num_trials}"
    outputs.append(plot_performance_main(summary))

    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    ax.plot(x, data["capture_rate_mean"], marker="o", color=PURSUER_COLOR, label="全部逃跑者被捕获")
    ax.plot(x, data["capture_any_rate_mean"], marker="s", color=ACCENT_COLOR, label="至少一次捕获")
    ax.set_title(f"捕获率（{title_suffix}）")
    ax.set_xlabel("训练轮次 l")
    ax.set_ylabel("比例")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False)
    style_axis(ax)
    outputs.append(save_figure(fig, "paper_performance_capture_rate.pdf"))

    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    line_with_se(ax, x, data["evader_return_mean"], data["evader_return_se"], "逃跑方", EVADER_COLOR)
    line_with_se(ax, x, data["pursuer_return_mean"], data["pursuer_return_se"], "追捕方", PURSUER_COLOR)
    ax.set_title(f"平均累计回报（{title_suffix}）")
    ax.set_xlabel("训练轮次 l")
    ax.set_ylabel("折扣累计回报")
    ax.legend(frameon=False)
    style_axis(ax)
    outputs.append(save_figure(fig, "paper_performance_mean_return.pdf"))

    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    line_with_se(ax, x, data["episode_steps_mean"], data["episode_steps_se"], "回合长度", ACCENT_COLOR)
    ax.set_title(f"平均回合长度（{title_suffix}）")
    ax.set_xlabel("训练轮次 l")
    ax.set_ylabel("步数")
    ax.legend(frameon=False)
    style_axis(ax)
    outputs.append(save_figure(fig, "paper_performance_episode_length.pdf"))

    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    line_with_se(ax, x, data["first_capture_time_mean"], data["first_capture_time_se"], "首次捕获", SECONDARY_COLOR)
    line_with_se(ax, x, data["full_capture_time_mean"], data["full_capture_time_se"], "全部捕获", PURSUER_COLOR, marker="s")
    ax.set_title(f"捕获时间（{title_suffix}）")
    ax.set_xlabel("训练轮次 l")
    ax.set_ylabel("时间")
    ax.legend(frameon=False)
    style_axis(ax)
    outputs.append(save_figure(fig, "paper_performance_capture_time.pdf"))

    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    line_with_se(ax, x, data["surviving_evaders_mean"], data["surviving_evaders_se"], "终局存活逃跑者", EVADER_COLOR)
    line_with_se(ax, x, data["active_pursuers_mean"], data["active_pursuers_se"], "终局有效追捕者", PURSUER_COLOR, marker="s")
    ax.set_title(f"终局有效智能体数量（{title_suffix}）")
    ax.set_xlabel("训练轮次 l")
    ax.set_ylabel("数量")
    ax.legend(frameon=False)
    style_axis(ax)
    outputs.append(save_figure(fig, "paper_performance_survival.pdf"))

    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    ax.plot(x, data["boundary_violation_rate_mean"], marker="o", color="#666666", label="越界终止")
    ax.set_title(f"越界率（{title_suffix}）")
    ax.set_xlabel("训练轮次 l")
    ax.set_ylabel("比例")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False)
    style_axis(ax)
    outputs.append(save_figure(fig, "paper_performance_boundary_violation.pdf"))

    fig, axs = plt.subplots(2, 2, figsize=(8.2, 5.8), constrained_layout=True)
    axs = axs.ravel()
    axs[0].plot(x, data["capture_rate_mean"], marker="o", color=PURSUER_COLOR, label="全部捕获率")
    axs[0].plot(x, data["capture_any_rate_mean"], marker="s", color=ACCENT_COLOR, label="任意捕获率")
    axs[0].set_ylabel("比例")
    axs[1].plot(x, data["episode_steps_mean"], marker="o", color=ACCENT_COLOR, label="回合长度")
    axs[1].set_ylabel("步数")
    axs[2].plot(x, data["evader_return_mean"], marker="o", color=EVADER_COLOR, label="逃跑方")
    axs[2].plot(x, data["pursuer_return_mean"], marker="s", color=PURSUER_COLOR, label="追捕方")
    axs[2].set_ylabel("回报")
    axs[3].plot(x, data["surviving_evaders_mean"], marker="o", color=EVADER_COLOR, label="存活逃跑者")
    axs[3].plot(x, data["boundary_violation_rate_mean"], marker="s", color="#666666", label="越界率")
    axs[3].set_ylabel("数量 / 比例")
    for ax in axs:
        ax.set_xlabel("训练轮次 l")
        ax.legend(frameon=False)
        style_axis(ax)
    fig.suptitle(f"性能指标汇总（{title_suffix}）")
    outputs.append(save_figure(fig, "paper_performance_summary_panel.pdf"))
    return outputs


def load_dataset(kind: str, split: str, round_l: int, robot: int = 0) -> np.ndarray:
    path = DATA_DIR / f"{split}_{kind}_l{round_l}_i{robot}.npy"
    require_file(path)
    return np.load(path, allow_pickle=True)


def dataset_sizes(rounds: list[int], kind: str) -> dict[str, np.ndarray]:
    train_counts = []
    test_counts = []
    round_values = []
    for round_l in rounds:
        if kind == "policy":
            train_total = 0
            test_total = 0
            for robot in range(4):
                train_total += load_dataset("policy", "train", round_l, robot).shape[0]
                test_total += load_dataset("policy", "test", round_l, robot).shape[0]
        else:
            train_total = load_dataset("value", "train", round_l, 0).shape[0]
            test_total = load_dataset("value", "test", round_l, 0).shape[0]
        round_values.append(round_l)
        train_counts.append(train_total)
        test_counts.append(test_total)
    return {
        "round_l": np.array(round_values, dtype=float),
        "train": np.array(train_counts, dtype=float),
        "test": np.array(test_counts, dtype=float),
    }


def sample_rows(array: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if array.shape[0] <= max_rows:
        return array
    idx = rng.choice(array.shape[0], size=max_rows, replace=False)
    return array[idx]


def policy_action_stats(rounds: list[int], rng: np.random.Generator, max_rows: int) -> dict[str, np.ndarray]:
    means = np.zeros((len(rounds), 4), dtype=float)
    stds = np.zeros((len(rounds), 4), dtype=float)
    for i, round_l in enumerate(rounds):
        for robot in range(4):
            chunks = []
            for split in ["train", "test"]:
                data = sample_rows(load_dataset("policy", split, round_l, robot), max_rows, rng)
                chunks.append(data[:, 13:15])
            actions = np.vstack(chunks)
            norms = np.linalg.norm(actions, axis=1)
            means[i, robot] = float(np.mean(norms))
            stds[i, robot] = float(np.std(norms))
    return {"round_l": np.array(rounds, dtype=float), "mean": means, "std": stds}


def value_target_samples(rounds: list[int], rng: np.random.Generator, max_rows: int) -> dict[int, np.ndarray]:
    samples: dict[int, np.ndarray] = {}
    for round_l in rounds:
        chunks = []
        for split in ["train", "test"]:
            data = sample_rows(load_dataset("value", split, round_l, 0), max_rows, rng)
            chunks.append(data[:, 13:17])
        samples[round_l] = np.vstack(chunks)
    return samples


def state_coverage(rounds: list[int], rng: np.random.Generator, max_rows: int) -> dict[str, np.ndarray]:
    coverage = []
    mean_radius = []
    for round_l in rounds:
        chunks = []
        for robot in range(4):
            data = sample_rows(load_dataset("policy", "train", round_l, robot), max_rows, rng)
            chunks.append(data[:, :13])
        states = np.vstack(chunks)
        positions = states[:, :8].reshape(-1, 2)
        mins = np.min(positions, axis=0)
        maxs = np.max(positions, axis=0)
        coverage.append(float((maxs[0] - mins[0]) * (maxs[1] - mins[1])))
        mean_radius.append(float(np.mean(np.linalg.norm(positions, axis=1))))
    return {
        "round_l": np.array(rounds, dtype=float),
        "coverage": np.array(coverage, dtype=float),
        "mean_radius": np.array(mean_radius, dtype=float),
    }


def train_test_shift(rounds: list[int], rng: np.random.Generator, max_rows: int) -> dict[str, np.ndarray]:
    policy_shift = []
    value_shift = []
    for round_l in rounds:
        p_shifts = []
        for robot in range(4):
            train = sample_rows(load_dataset("policy", "train", round_l, robot), max_rows, rng)
            test = sample_rows(load_dataset("policy", "test", round_l, robot), max_rows, rng)
            p_shifts.append(float(np.linalg.norm(np.mean(train[:, :13], axis=0) - np.mean(test[:, :13], axis=0))))
        policy_shift.append(float(np.mean(p_shifts)))
        train_v = sample_rows(load_dataset("value", "train", round_l, 0), max_rows, rng)
        test_v = sample_rows(load_dataset("value", "test", round_l, 0), max_rows, rng)
        value_shift.append(float(np.linalg.norm(np.mean(train_v[:, :13], axis=0) - np.mean(test_v[:, :13], axis=0))))
    return {
        "round_l": np.array(rounds, dtype=float),
        "policy": np.array(policy_shift, dtype=float),
        "value": np.array(value_shift, dtype=float),
    }


def plot_dataset_sizes(rounds: list[int], kind: str, filename: str, title: str) -> Path:
    sizes = dataset_sizes(rounds, kind)
    fig, ax = plt.subplots(figsize=(5.4, 3.4), constrained_layout=True)
    x = sizes["round_l"]
    if len(x) > 1:
        min_spacing = float(np.min(np.diff(np.sort(x))))
    else:
        min_spacing = 2.0
    bar_width = min(0.75, max(0.25, min_spacing * 0.32))
    ax.bar(x - bar_width / 2, sizes["train"], width=bar_width, color=ACCENT_COLOR, alpha=0.85, label="训练集")
    ax.bar(x + bar_width / 2, sizes["test"], width=bar_width, color=SECONDARY_COLOR, alpha=0.85, label="测试集")
    ax.set_title(title)
    ax.set_xlabel("训练轮次 l")
    ax.set_ylabel("样本数")
    ax.set_xticks(x)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False)
    style_axis(ax)
    return save_figure(fig, filename)


def plot_data_diagnostics(rounds: list[int], args) -> list[Path]:
    rng = np.random.default_rng(args.seed)
    outputs: list[Path] = []
    outputs.append(plot_dataset_sizes(rounds, "policy", "paper_data_policy_dataset_size.pdf", "策略网络数据集规模"))
    outputs.append(plot_dataset_sizes(rounds, "value", "paper_data_value_dataset_size.pdf", "价值网络数据集规模"))

    action_stats = policy_action_stats(rounds, rng, args.max_diagnostic_samples)
    fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    colors = [EVADER_COLOR, "#66c2a5", PURSUER_COLOR, "#e6ab02"]
    labels = ["逃跑者 0", "逃跑者 1", "追捕者 0", "追捕者 1"]
    for robot in range(4):
        ax.plot(action_stats["round_l"], action_stats["mean"][:, robot], marker="o", color=colors[robot], label=labels[robot])
    ax.set_title("策略标签动作幅值随训练轮次变化")
    ax.set_xlabel("训练轮次 l")
    ax.set_ylabel("平均动作范数 ||a||")
    ax.legend(frameon=False, ncol=2)
    style_axis(ax)
    outputs.append(save_figure(fig, "paper_data_action_magnitude_by_round.pdf"))

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    selected_rounds = [rounds[0], rounds[len(rounds) // 2], rounds[-1]]
    bins = np.linspace(0.0, 4.0, 36)
    for robot, ax in enumerate(axs.ravel()):
        for round_l, color in zip(selected_rounds, [GRID_COLOR, ACCENT_COLOR, PURSUER_COLOR]):
            train = sample_rows(load_dataset("policy", "train", round_l, robot), args.max_diagnostic_samples, rng)
            norms = np.linalg.norm(train[:, 13:15], axis=1)
            ax.hist(norms, bins=bins, histtype="step", density=True, linewidth=2.0, color=color, label=f"第 {round_l} 轮")
        ax.set_title(labels[robot])
        ax.set_xlabel("动作范数 ||a||")
        ax.set_ylabel("概率密度")
        ax.legend(frameon=False)
        style_axis(ax)
    fig.suptitle("策略标签动作分布")
    outputs.append(save_figure(fig, "paper_data_action_distribution_by_robot.pdf"))

    value_samples = value_target_samples(selected_rounds, rng, args.max_diagnostic_samples)
    fig, axs = plt.subplots(1, len(selected_rounds), figsize=(9.2, 3.2), sharey=True, constrained_layout=True)
    for ax, round_l in zip(np.ravel(axs), selected_rounds):
        vals = value_samples[round_l]
        ax.boxplot(
            [vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3]],
            tick_labels=["逃0", "逃1", "追0", "追1"],
            patch_artist=True,
            boxprops={"facecolor": "#f2f2f2", "edgecolor": "#555555"},
            medianprops={"color": PURSUER_COLOR, "linewidth": 2},
        )
        ax.set_title(f"第 {round_l} 轮")
        ax.set_xlabel("智能体")
        style_axis(ax)
    axs[0].set_ylabel("价值标签")
    fig.suptitle("价值网络标签分布")
    outputs.append(save_figure(fig, "paper_data_value_target_distribution.pdf"))

    coverage = state_coverage(rounds, rng, args.max_diagnostic_samples)
    fig, ax1 = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    ax1.plot(coverage["round_l"], coverage["coverage"], marker="o", color=ACCENT_COLOR, label="位置覆盖面积")
    ax1.set_xlabel("训练轮次 l")
    ax1.set_ylabel("覆盖面积")
    ax2 = ax1.twinx()
    ax2.plot(coverage["round_l"], coverage["mean_radius"], marker="s", color=SECONDARY_COLOR, label="平均位置半径")
    ax2.set_ylabel("平均位置半径")
    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels1 + labels2, frameon=False)
    ax1.set_title("状态空间覆盖情况")
    style_axis(ax1)
    ax2.spines["top"].set_visible(False)
    outputs.append(save_figure(fig, "paper_data_state_coverage_by_round.pdf"))

    shifts = train_test_shift(rounds, rng, args.max_diagnostic_samples)
    fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    ax.plot(shifts["round_l"], shifts["policy"], marker="o", color=ACCENT_COLOR, label="策略数据")
    ax.plot(shifts["round_l"], shifts["value"], marker="s", color=SECONDARY_COLOR, label="价值数据")
    ax.set_title("训练集与测试集状态分布差异")
    ax.set_xlabel("训练轮次 l")
    ax.set_ylabel("均值状态距离")
    ax.legend(frameon=False)
    style_axis(ax)
    outputs.append(save_figure(fig, "paper_data_train_test_shift.pdf"))
    return outputs


def fixed_initial_state(problem) -> np.ndarray:
    state = np.zeros((problem.state_dim, 1), dtype=float)
    layout = np.array(
        [
            [-4.2, 2.5],
            [-4.2, -2.5],
            [4.4, 2.3],
            [4.4, -2.3],
        ],
        dtype=float,
    )
    for robot, idxs in enumerate(problem.state_idxs):
        state[idxs, 0] = layout[robot]
    state[problem.time_idx, 0] = 0.0
    for idx in problem.active_idxs:
        state[idx, 0] = 1.0
    if problem.is_terminal(state):
        raise RuntimeError("Fixed qualitative initial state is terminal")
    return state


def set_fixed_speed_limits(problem, evader_speed: float, pursuer_speed: float) -> None:
    problem.current_evader_speed_lim = evader_speed
    problem.current_pursuer_speed_lim = pursuer_speed
    problem.update_action_lims()


def prepare_fixed_problem(problem, args) -> None:
    problem.randomize_speed_limits()
    set_fixed_speed_limits(problem, args.qualitative_evader_speed, args.qualitative_pursuer_speed)


def run_fixed_rollout(round_l: int, args) -> dict:
    local_args = copy(args)
    local_args.number_simulations = args.qualitative_simulations
    np.random.seed(args.seed + round_l * 100_000 + 17)
    problem = get_problem("example8")
    prepare_fixed_problem(problem, args)
    solver, policy_oracle, value_oracle = build_solver(problem, round_l, local_args)
    prepare_fixed_problem(problem, args)
    instance = {
        "problem": problem,
        "solver": solver,
        "policy_oracle": policy_oracle,
        "value_oracle": value_oracle,
        "initial_state": fixed_initial_state(problem),
    }
    sim_result = run_instance(
        0,
        Queue(),
        len(problem.times),
        instance,
        verbose=False,
        tqdm_on=False,
    )
    sim_result["round_l"] = round_l
    return sim_result


def square_limits(results: list[dict], margin: float = 1.0) -> tuple[tuple[float, float], tuple[float, float]]:
    xy = []
    for result in results:
        states = np.asarray(result["states"]).squeeze(axis=2)
        xy.append(states[:, :8].reshape(-1, 2))
    points = np.vstack(xy)
    lo = np.nanmin(points, axis=0) - margin
    hi = np.nanmax(points, axis=0) + margin
    center = (lo + hi) / 2.0
    half = max(float(np.max(hi - lo)) / 2.0, 5.0)
    xlim = (max(-10.0, center[0] - half), min(10.0, center[0] + half))
    ylim = (max(-10.0, center[1] - half), min(10.0, center[1] + half))
    return xlim, ylim


def plot_trajectory_panel(ax, problem, states: np.ndarray, round_l: int, xlim, ylim, show_legend: bool) -> None:
    states = states.squeeze(axis=2)
    colors = [EVADER_COLOR, "#55b6a3", PURSUER_COLOR, "#c44e52"]
    labels = ["逃跑者 0", "逃跑者 1", "追捕者 0", "追捕者 1"]
    for robot, idxs in enumerate(problem.state_idxs):
        xy = states[:, idxs]
        ax.plot(xy[:, 0], xy[:, 1], color=colors[robot], linewidth=2.0, label=labels[robot])
        ax.scatter(xy[0, 0], xy[0, 1], s=32, facecolor="white", edgecolor=colors[robot], linewidth=1.5, zorder=3)
        ax.scatter(xy[-1, 0], xy[-1, 1], s=34, marker="s", color=colors[robot], zorder=3)
        if len(xy) > 4:
            mid = len(xy) // 2
            delta = xy[min(mid + 1, len(xy) - 1)] - xy[max(mid - 1, 0)]
            ax.annotate(
                "",
                xy=xy[mid] + 0.001 * delta,
                xytext=xy[mid] - 0.24 * delta,
                arrowprops={"arrowstyle": "->", "color": colors[robot], "lw": 1.2},
            )
    for e in problem.evaders:
        idxs = problem.state_idxs[e]
        circle = plt.Circle(
            (states[-1, idxs[0]], states[-1, idxs[1]]),
            problem.desired_distance,
            edgecolor=EVADER_COLOR,
            facecolor=EVADER_COLOR,
            alpha=0.08,
            linewidth=1.0,
        )
        ax.add_patch(circle)
    ax.set_title(f"$l$ = {round_l}")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    if show_legend:
        ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.18))
        ax.set_ylabel("y")
    style_paper_axis(ax)


def plot_qualitative_fixed_rollouts(rounds: list[int], args) -> list[Path]:
    outputs: list[Path] = []
    results = [run_fixed_rollout(round_l, args) for round_l in rounds]
    xlim, ylim = square_limits(results)
    fig, axs = plt.subplots(1, len(results), figsize=(3.05 * len(results), 3.25), constrained_layout=True)
    axs = np.ravel(axs)
    for idx, (ax, result) in enumerate(zip(axs, results)):
        plot_trajectory_panel(
            ax,
            result["instance"]["problem"],
            result["states"],
            int(result["round_l"]),
            xlim,
            ylim,
            show_legend=idx == 0,
        )
    outputs.append(save_figure(fig, "paper_qualitative_fixed_rollouts.pdf"))
    return outputs


def validate_outputs(paths: list[Path]) -> None:
    missing = []
    for path in paths:
        if not path.is_file() or path.stat().st_size <= 1024:
            missing.append(str(path))
    if missing:
        raise RuntimeError("Some figure outputs are missing or too small:\n" + "\n".join(missing))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate paper figures from current/models and current/data.")
    parser.add_argument("--rounds", default="default", help="Comma/range list, e.g. 0,10,20 or 0-5. Default: paper rounds.")
    parser.add_argument("--diagnostic-rounds", default=None, help="Optional separate rounds for data diagnostics.")
    parser.add_argument("--num-trials", type=int, default=200, help="Rollout trials per checkpoint.")
    parser.add_argument("--force-eval", action="store_true", help="Re-run rollouts even if cached metrics exist.")
    parser.add_argument("--seed", type=int, default=20260503, help="Base random seed.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print rollout progress every N trials.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, mp.cpu_count() - 25),
        help="Parallel rollout workers per checkpoint.",
    )
    parser.add_argument("--no-parallel", dest="parallel", action="store_false", help="Disable parallel rollout evaluation.")
    parser.add_argument(
        "--mp-context",
        choices=["fork", "spawn", "forkserver"],
        default="fork",
        help="Multiprocessing start method for rollout workers.",
    )
    parser.set_defaults(parallel=True)
    parser.add_argument("--max-diagnostic-samples", type=int, default=5000, help="Max rows sampled from each npy for diagnostics.")
    parser.add_argument("--solver-name", default="C_PUCT_V1")
    parser.add_argument("--number-simulations", type=int, default=1000)
    parser.add_argument("--search-depth", type=int, default=100)
    parser.add_argument("--C-pw", dest="C_pw", type=float, default=2.0)
    parser.add_argument("--alpha-pw", dest="alpha_pw", type=float, default=0.5)
    parser.add_argument("--C-exp", dest="C_exp", type=float, default=1.0)
    parser.add_argument("--alpha-exp", dest="alpha_exp", type=float, default=0.25)
    parser.add_argument("--beta-policy", type=float, default=1.0)
    parser.add_argument("--beta-value", type=float, default=1.0)
    parser.add_argument("--skip-performance", action="store_true", help="Only generate data diagnostics.")
    parser.add_argument("--skip-data", action="store_true", help="Only generate performance figures.")
    parser.add_argument("--with-qualitative", action="store_true", help="Generate fixed-initial-state trajectory panels.")
    parser.add_argument("--qualitative-rounds", default="1,11,49", help="Rounds for the fixed-rollout trajectory panel.")
    parser.add_argument("--qualitative-simulations", type=int, default=300, help="Simulations/search for qualitative rollouts.")
    parser.add_argument("--qualitative-evader-speed", type=float, default=2.0, help="Fixed evader speed limit for qualitative rollouts.")
    parser.add_argument("--qualitative-pursuer-speed", type=float, default=2.0, help="Fixed pursuer speed limit for qualitative rollouts.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    rounds = parse_rounds(args.rounds)
    diagnostic_rounds = parse_rounds(args.diagnostic_rounds) if args.diagnostic_rounds else rounds
    qualitative_rounds = parse_rounds(args.qualitative_rounds)
    configure_matplotlib()
    ensure_dirs()

    outputs: list[Path] = []
    if not args.skip_performance:
        _, summary = load_or_evaluate(args, rounds)
        outputs.extend(plot_performance(summary, args))
    if not args.skip_data:
        outputs.extend(plot_data_diagnostics(diagnostic_rounds, args))
    if args.with_qualitative:
        outputs.extend(plot_qualitative_fixed_rollouts(qualitative_rounds, args))
    validate_outputs(outputs)
    print("Generated paper figures:")
    for path in outputs:
        print(f"  {path.relative_to(ROOT_DIR)}")
    print(f"Preview PNGs are in {PREVIEW_DIR.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
