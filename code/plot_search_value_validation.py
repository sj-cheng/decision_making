"""Plot completed search-value stages without changing scientific inputs.

Each panel uses an ordinary 95% public-history bootstrap interval for its
method mean.  These descriptive intervals are not the four-comparison
simultaneous intervals used by the frozen analysis to decide continuation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "current/l45_search_value_20261026"
METHODS = ("G", "OL200", "FB200", "OLH", "FBH")
PANELS = (("matched", "rediscovery_cost", "Model-matched target: first rediscovery",
           "Failure-penalized rediscovery cost (steps)"),
          ("rootpuct", "cost", "Actual RootPUCT opponent: capture",
           "Failure-penalized completion cost (steps)"))
COLORS = ("#737373", "#9ecae1", "#fdae6b", "#3182bd", "#e6550d")
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 84120261026


def _stage_inputs(output: Path, stage: str) -> tuple[Path, Path] | None:
    candidates = (output / "stages" / stage, output / stage)
    complete = [p for p in candidates if (p / "results.json").exists()
                and (p / "statistics.json").exists()]
    incomplete = [p for p in candidates if ((p / "results.json").exists()
                  != (p / "statistics.json").exists())]
    if not complete:
        if incomplete:
            raise FileNotFoundError(f"Completed results and statistics required: {incomplete[0]}")
        return None
    if len(complete) > 1:
        for name in ("results.json", "statistics.json"):
            if (complete[0] / name).read_bytes() != (complete[1] / name).read_bytes():
                raise ValueError(f"Ambiguous distinct {stage}/{name} files")
    return complete[0] / "results.json", complete[0] / "statistics.json"


def grouped_means_and_intervals(rows: list[dict], draws: int = BOOTSTRAP_DRAWS) -> dict:
    """Bootstrap whole public histories; never sample individual worlds."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty completed result list")
    groups = sorted({str(r["group"]) for r in rows})
    index = {}
    worlds = {g: set() for g in groups}
    for row in rows:
        if row["method"] not in METHODS or row["opponent"] not in ("matched", "rootpuct"):
            raise ValueError("Unknown method or opponent")
        key = (str(row["group"]), str(row["world"]), row["opponent"], row["method"])
        if key in index:
            raise ValueError(f"Duplicate paired result {key}")
        index[key] = row
        worlds[key[0]].add(key[1])
    blocks = np.empty((len(groups), len(PANELS), len(METHODS)), dtype=float)
    counts = {}
    for pi, (opponent, metric, _, _) in enumerate(PANELS):
        counts[opponent] = {}
        for mi, method in enumerate(METHODS):
            method_rows = []
            for gi, group in enumerate(groups):
                group_rows = []
                for world in sorted(worlds[group]):
                    key = (group, world, opponent, method)
                    if key not in index:
                        raise ValueError(f"Incomplete paired stage: {key}")
                    group_rows.append(index[key])
                values = np.asarray([float(r[metric]) for r in group_rows], dtype=float)
                if not np.all(np.isfinite(values)) or np.any(values < 0.):
                    raise ValueError("Invalid failure-penalized cost")
                blocks[gi, pi, mi] = values.mean()
                method_rows.extend(group_rows)
            counts[opponent][method] = dict(
                games=len(method_rows), capture=sum(bool(r["capture"]) for r in method_rows),
                rediscovered=sum(r["first_seen"] is not None for r in method_rows))
    if draws < 100:
        raise ValueError("At least 100 bootstrap draws required")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty((draws, len(PANELS), len(METHODS)), dtype=float)
    for start in range(0, draws, 1000):
        stop = min(start + 1000, draws)
        ids = rng.integers(0, len(groups), size=(stop-start, len(groups)))
        bootstrap[start:stop] = blocks[ids].mean(axis=1)
    means = blocks.mean(axis=0)
    intervals = np.quantile(bootstrap, [.025, .975], axis=0)
    result = dict(groups=len(groups), draws=draws, seed=BOOTSTRAP_SEED, panels={})
    for pi, (opponent, _, _, _) in enumerate(PANELS):
        result["panels"][opponent] = {
            method: dict(mean=float(means[pi, mi]),
                         ci95=[float(intervals[0, pi, mi]), float(intervals[1, pi, mi])],
                         **counts[opponent][method])
            for mi, method in enumerate(METHODS)}
    return result


def plot_stage(output: Path | str, stage: str) -> dict | None:
    output = Path(output).resolve()
    paths = _stage_inputs(output, stage)
    if paths is None:
        return None
    results_path, statistics_path = paths
    rows = json.loads(results_path.read_text())
    stats = json.loads(statistics_path.read_text())
    plotted = grouped_means_and_intervals(rows)
    if int(stats["groups"]) != plotted["groups"] or int(stats["games"]) != len(rows):
        raise ValueError("Statistics and completed results disagree")
    for opponent, _, _, _ in PANELS:
        for method in METHODS:
            existing = stats["summary"][opponent][method]
            actual = plotted["panels"][opponent][method]
            if any(int(existing[k]) != actual[k] for k in ("games", "capture", "rediscovered")):
                raise ValueError("Statistics and plotted count annotations disagree")

    # Import plotting only after complete input validation; this script is not
    # part of the controller timing process.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), sharey=False)
    title = "Development screening" if stage == "development" else "Independent confirmation"
    figure.suptitle(f"{title} | {plotted['groups']} public-history groups", fontsize=14, y=.98)
    x = np.arange(len(METHODS))
    for ax, (opponent, _, panel_title, ylabel) in zip(axes, PANELS):
        panel = plotted["panels"][opponent]
        means = np.asarray([panel[m]["mean"] for m in METHODS])
        lower = np.asarray([panel[m]["ci95"][0] for m in METHODS])
        upper = np.asarray([panel[m]["ci95"][1] for m in METHODS])
        ax.bar(x, means, width=.67, color=COLORS, edgecolor="white", linewidth=.6, zorder=2)
        # Use direct interval segments.  A percentile interval need not contain
        # the observed mean, so forcing nonnegative error-bar lengths is wrong.
        ax.vlines(x, lower, upper, colors="#202020", linewidth=1.4, zorder=3)
        ax.hlines(lower, x-.095, x+.095, colors="#202020", linewidth=1.4, zorder=3)
        ax.hlines(upper, x-.095, x+.095, colors="#202020", linewidth=1.4, zorder=3)
        extent = max(float(np.max(upper)), float(np.max(means)), 1.)
        for xi, method in zip(x, METHODS):
            value = panel[method]
            ax.text(xi, max(value["mean"], value["ci95"][1]) + .025*extent,
                    f"{value['mean']:.1f}\nC {value['capture']}/{value['games']}\n"
                    f"R {value['rediscovered']}/{value['games']}",
                    ha="center", va="bottom", fontsize=8, linespacing=1.2)
        ax.set_ylim(0., extent*1.31)
        ax.set_xticks(x, METHODS)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title, pad=13)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dddddd", linewidth=.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=7)
    figure.text(.5, .095,
                "Bars: public-history mean; whiskers: ordinary 95% grouped bootstrap CI "
                "(10,000 draws). C: captures; R: rediscoveries.",
                ha="center", fontsize=8.4)
    figure.text(.5, .055,
                "Matched motion is exact only before first positive observation. "
                "High-budget methods are diagnostic; plotted CIs are not decision-test CIs.",
                ha="center", fontsize=8.4)
    figure.subplots_adjust(left=.075, right=.985, bottom=.205, top=.84, wspace=.26)
    destination = output / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    png = destination / f"{stage}_costs.png"
    pdf = destination / f"{stage}_costs.pdf"
    figure.savefig(png, dpi=220, facecolor="white")
    figure.savefig(pdf, facecolor="white", metadata={"Title": f"Search-value validation: {title}",
                   "Creator": "plot_search_value_validation.py", "CreationDate": None})
    plt.close(figure)
    return dict(stage=stage, groups=plotted["groups"], games=len(rows), png=str(png), pdf=str(pdf))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    results = []
    for stage in ("development", "confirmation"):
        result = plot_stage(args.output, stage)
        if result is not None:
            results.append(result)
    print(json.dumps(dict(figures=results), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
