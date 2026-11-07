"""Grouped analysis of the frozen-D observation-feedback search-value study.

This module imports no controller or environment and reads no confirmation
bank.  Its only scientific inputs are completed stage result rows.  Confidence
intervals preserve the shared public-history block, all worlds, all methods,
and both target conditions.  Capture is an observed screening guard, never a
claim of non-inferiority or equivalence.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "current/l45_search_value_20261026"
METHODS = ("G", "OL200", "FB200", "OLH", "FBH")
OPPONENTS = ("matched", "rootpuct")
MAIN_PAIRS = (("FBH", "G"), ("FBH", "OLH"))
SECONDARY_PAIRS = (("FB200", "G"), ("FB200", "OL200"),
                   ("FBH", "FB200"), ("OLH", "OL200"))
METRICS = ("completion_cost", "rediscovery_cost", "capture", "rediscovery")
BOOTSTRAP_SEED = 84120261026
DEVELOPMENT_MINIMUM_USEFUL_STEPS = 1.0


def _number(value: Any, label: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Non-finite {label}: {value!r}")
    return result


def _metric(row: dict, name: str) -> float:
    if name == "completion_cost":
        return _number(row["cost"], name)
    if name == "rediscovery_cost":
        return _number(row["rediscovery_cost"], name)
    if name == "capture":
        return float(bool(row["capture"]))
    if name == "rediscovery":
        return float(row["first_seen"] is not None)
    raise KeyError(name)


def _family(row: dict) -> str:
    if row.get("family"):
        return str(row["family"])
    name = str(row["group"])
    for family in ("long_wall", "wall_corner", "multi_route"):
        if name.startswith(family + "_") or name == family:
            return family
    return "unspecified"


def _quantiles(values: list[float]) -> dict:
    if not values:
        return dict(count=0, mean_ms=None, p50_ms=None, p95_ms=None,
                    maximum_ms=None, over_50ms=0, over_200ms=0,
                    over_1000ms=0)
    a = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("Invalid recorded latency")
    return dict(count=int(len(a)), mean_ms=float(a.mean()),
                p50_ms=float(np.quantile(a, .5)),
                p95_ms=float(np.quantile(a, .95)), maximum_ms=float(a.max()),
                over_50ms=int(np.sum(a > 50.)),
                over_200ms=int(np.sum(a > 200.)),
                over_1000ms=int(np.sum(a > 1000.)))


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return dict(games=0)
    latencies = [_number(x, "latency") for r in rows
                 for x in r.get("latencies_ms", [])]
    initial = [_number(r["initial_total_ms"], "initial_total_ms")
               for r in rows if r.get("initial_total_ms") is not None]
    simulations = [_number(x, "simulations") for r in rows
                   for x in r.get("simulations", [])]
    if any(x < 0 or not x.is_integer() for x in simulations):
        raise ValueError("Simulation counts must be nonnegative integers")
    found = [float(r["first_seen"]) for r in rows if r["first_seen"] is not None]
    successes = [float(r["cost"]) for r in rows if r["capture"]]
    capture = sum(bool(r["capture"]) for r in rows)
    return dict(
        games=len(rows), groups=len({r["group"] for r in rows}),
        capture=int(capture), capture_rate=float(capture / len(rows)),
        rediscovered=len(found), rediscovery_rate=float(len(found) / len(rows)),
        completion_cost=float(np.mean([_metric(r, "completion_cost") for r in rows])),
        rediscovery_cost=float(np.mean([_metric(r, "rediscovery_cost") for r in rows])),
        success_steps=None if not successes else float(np.mean(successes)),
        first_seen_conditional=None if not found else float(np.mean(found)),
        engineering_failures=sum(r.get("failure") is not None for r in rows),
        not_rediscovered=sum(not r["capture"] and r["first_seen"] is None for r in rows),
        rediscovered_not_captured=sum(not r["capture"] and r["first_seen"] is not None for r in rows),
        decisions=len(latencies), decision_timing=_quantiles(latencies),
        initial_timing=_quantiles(initial),
        simulations=dict(decisions=len(simulations),
                         total=int(sum(simulations)),
                         mean=None if not simulations else float(np.mean(simulations)),
                         minimum=None if not simulations else int(min(simulations)),
                         maximum=None if not simulations else int(max(simulations)),
                         positive_decisions=sum(x > 0 for x in simulations),
                         positive_mean=None if not any(x > 0 for x in simulations)
                         else float(np.mean([x for x in simulations if x > 0]))))


def _validated_index(rows: list[dict]) -> tuple[list[str], dict, dict]:
    index = {}
    worlds: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        for name in ("group", "world", "opponent", "method", "capture",
                     "first_seen", "original_remaining", "cost", "rediscovery_cost"):
            if name not in row:
                raise ValueError(f"Missing field {name}: {row.get('key', '<unnamed>')}")
        if row["opponent"] not in OPPONENTS or row["method"] not in METHODS:
            raise ValueError("Unknown opponent or method")
        key = (str(row["group"]), str(row["world"]), row["opponent"], row["method"])
        if key in index:
            raise ValueError(f"Duplicate paired result: {key}")
        index[key] = row
        worlds[key[0]].add(key[1])
        for metric in METRICS:
            _metric(row, metric)
    groups = sorted(worlds)
    missing = []
    for group in groups:
        for world in sorted(worlds[group]):
            for opponent in OPPONENTS:
                remaining = []
                for method in METHODS:
                    key = (group, world, opponent, method)
                    if key not in index:
                        missing.append(key)
                    else:
                        remaining.append(_number(index[key]["original_remaining"], "remaining"))
                if remaining and len(set(remaining)) != 1:
                    raise ValueError(f"Methods have different initial clocks: {group}, {world}, {opponent}")
    if missing:
        raise ValueError(f"Incomplete paired stage, {len(missing)} missing cells; first={missing[0]}")
    return groups, index, worlds


def statistics(rows: list[dict], draws: int = 10000) -> dict:
    """Return JSON-compatible, public-history paired summaries and intervals.

    A stage must contain the complete five-method/two-opponent design for
    every recorded history and world.  Missing cells are an error, never
    silently removed.  The same bootstrap group draws are used by every
    endpoint and every comparison.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError("A nonempty list of completed result rows is required")
    if draws < 100:
        raise ValueError("At least 100 bootstrap draws are required")
    groups, index, worlds = _validated_index(rows)
    comparisons = [(opponent, a, b) for opponent in OPPONENTS
                   for a, b in MAIN_PAIRS + SECONDARY_PAIRS]
    blocks = np.empty((len(groups), len(comparisons), len(METRICS)), dtype=float)
    for gi, group in enumerate(groups):
        world_ids = sorted(worlds[group])
        for ci, (opponent, a, b) in enumerate(comparisons):
            for mi, metric in enumerate(METRICS):
                blocks[gi, ci, mi] = np.mean([
                    _metric(index[(group, w, opponent, a)], metric)
                    - _metric(index[(group, w, opponent, b)], metric)
                    for w in world_ids])
    means = blocks.mean(axis=0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.empty((draws, len(comparisons), len(METRICS)), dtype=float)
    for start in range(0, draws, 1000):
        end = min(draws, start + 1000)
        selected = rng.integers(0, len(groups), size=(end-start, len(groups)))
        boot[start:end] = blocks[selected].mean(axis=1)

    main_indices = [i for i, (_, a, b) in enumerate(comparisons) if (a, b) in MAIN_PAIRS]
    primary_metrics = {"matched": "rediscovery_cost", "rootpuct": "completion_cost"}
    centered = np.stack([
        boot[:, i, METRICS.index(primary_metrics[comparisons[i][0]])]
        - means[i, METRICS.index(primary_metrics[comparisons[i][0]])]
        for i in main_indices], axis=1)
    radius = float(np.quantile(np.max(np.abs(centered), axis=1), .95))
    no_engineering_failure = all(r.get("failure") is None for r in rows)
    output_comparisons = {}
    for i, (opponent, a, b) in enumerate(comparisons):
        key = f"{opponent}:{a}__{b}"
        c = dict(opponent=opponent, method=a, baseline=b,
                 main=(a, b) in MAIN_PAIRS, groups=len(groups),
                 primary_endpoint=primary_metrics[opponent])
        for j, metric in enumerate(METRICS):
            c[metric] = dict(mean=float(means[i, j]),
                             ci95=[float(x) for x in np.quantile(boot[:, i, j], [.025, .975])],
                             interval_type="ordinary_paired_public_history_bootstrap")
        if c["main"]:
            primary = c[c["primary_endpoint"]]
            primary["simultaneous_ci95"] = [primary["mean"] - radius, primary["mean"] + radius]
            primary["simultaneous_interval_type"] = "max_absolute_centered_four_primary_endpoints"
            success_metric = "rediscovery" if opponent == "matched" else "capture"
            c["reliable_cost_signal"] = bool(primary["simultaneous_ci95"][1] < 0.
                and c[success_metric]["mean"] >= 0. and no_engineering_failure)
        else:
            c["reliable_cost_signal"] = None
        c["capture_guard"] = dict(observed_change_nonnegative=c["capture"]["mean"] >= 0.,
                                  inferential_noninferiority_claim=False)
        success_metric = "rediscovery" if opponent == "matched" else "capture"
        c["screening_success_guard"] = dict(metric=success_metric,
            observed_change_nonnegative=c[success_metric]["mean"] >= 0.,
            inferential_noninferiority_claim=False)
        output_comparisons[key] = c

    summary = {o: {m: _summarize([r for r in rows if r["opponent"] == o and r["method"] == m])
                   for m in METHODS} for o in OPPONENTS}
    families = sorted({_family(r) for r in rows})
    by_family = {f: {o: {m: _summarize([r for r in rows if _family(r) == f
                                      and r["opponent"] == o and r["method"] == m])
                         for m in METHODS} for o in OPPONENTS} for f in families}
    main_values = [c for c in output_comparisons.values() if c["main"]]
    legacy_development_stop = bool(all(c[c["primary_endpoint"]]["mean"] >= 0. for c in main_values)
                                   and all(c["capture"]["mean"] <= 0. for c in main_values))
    development_checks = {}
    for opponent in OPPONENTS:
        condition_checks = []
        for a, b in MAIN_PAIRS:
            c = output_comparisons[f"{opponent}:{a}__{b}"]
            difference = c[c["primary_endpoint"]]["mean"]
            condition_checks.append(dict(
                comparison=f"{opponent}:{a}__{b}",
                primary_endpoint=c["primary_endpoint"], primary_difference=difference,
                minimum_useful_steps=DEVELOPMENT_MINIMUM_USEFUL_STEPS,
                useful_cost_signal=bool(difference <= -DEVELOPMENT_MINIMUM_USEFUL_STEPS),
                success_metric=c["screening_success_guard"]["metric"],
                observed_success_difference=c[c["screening_success_guard"]["metric"]]["mean"],
                observed_success_nondecrease=c["screening_success_guard"]["observed_change_nonnegative"]))
        development_checks[opponent] = dict(
            comparisons=condition_checks,
            passes=bool(no_engineering_failure and all(
                c["useful_cost_signal"] and c["observed_success_nondecrease"]
                for c in condition_checks)))
    development_continue = bool(any(c["passes"] for c in development_checks.values()))
    root_signal = bool(all(output_comparisons[f"rootpuct:{a}__{b}"]["reliable_cost_signal"]
                           for a, b in MAIN_PAIRS))
    matched_signal = bool(all(output_comparisons[f"matched:{a}__{b}"]["reliable_cost_signal"]
                              for a, b in MAIN_PAIRS))
    failure_rows = []
    for r in rows:
        if r["capture"]:
            continue
        baseline = index[(str(r["group"]), str(r["world"]), r["opponent"], "G")]
        failure_rows.append(dict(
            group=str(r["group"]), family=_family(r), world=str(r["world"]),
            opponent=r["opponent"], method=r["method"], first_seen=r["first_seen"],
            cost=float(r["cost"]), G_cost=float(baseline["cost"]),
            difference_from_G=float(r["cost"] - baseline["cost"]),
            reason="engineering_failure" if r.get("failure") is not None else
                   "not_rediscovered" if r["first_seen"] is None else "rediscovered_not_captured",
            episode=r.get("episode"), failure=r.get("failure")))
    return dict(
        games=len(rows), groups=len(groups), group_ids=groups,
        summary=summary, by_family=by_family, comparisons=output_comparisons,
        bootstrap=dict(draws=int(draws), seed=BOOTSTRAP_SEED,
                       unit="public_history", preserve="worlds,methods,both_opponents",
                       simultaneous_family_size=4, simultaneous_radius=radius,
                       primary_endpoints=primary_metrics),
        decision=dict(development_continue=development_continue,
                      development_stop_rule=not development_continue,
                      development_rule="At least one opponent: FBH improves both G and OLH by >=1 step, observed success does not decrease in either comparison, no engineering failures.",
                      development_checks=development_checks,
                      development_minimum_useful_steps=DEVELOPMENT_MINIMUM_USEFUL_STEPS,
                      legacy_development_stop_rule_superseded=legacy_development_stop,
                      legacy_development_stop_rule_is_active=False,
                      engineering_gate_passed=no_engineering_failure,
                      rootpuct_followup=root_signal,
                      matched_mechanism_signal=matched_signal,
                      capture_guard_is_exploratory=True,
                      no_capture_noninferiority_claim=True,
                      neural_training_authorized=False,
                      deployment_change_authorized=False),
        failure_cases=failure_rows,
        interpretation=dict(
            matched="Matched only until first positive observation; use rediscovery_cost as primary. Post-rediscovery capture is not a strictly model-matched test.",
            rootpuct="Use failure-penalized completion cost as primary; strong actual opponent unchanged.",
            capture="Observed capture change >=0 is an exploratory continuation screen, not an inferential non-inferiority/equivalence result.",
            confirmation="Only an independently frozen confirmation stage can provide confirmation evidence; development cannot be relabeled."))


def _stage_path(output: Path, stage: str) -> Path | None:
    candidates = (output / "stages" / stage / "results.json",
                  output / stage / "results.json")
    existing = [p for p in candidates if p.exists()]
    if len(existing) > 1:
        payloads = [p.read_bytes() for p in existing]
        if payloads[0] != payloads[1]:
            raise ValueError(f"Ambiguous distinct {stage} result files")
    return existing[0] if existing else None


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _format(value: Any, digits: int = 2) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _interval(metric: dict, simultaneous: bool = False, scale: float = 1.) -> str:
    interval = metric.get("simultaneous_ci95") if simultaneous else metric["ci95"]
    if interval is None:
        return "N/A"
    return (f"{scale*metric['mean']:+.2f} "
            f"[{scale*interval[0]:+.2f}, {scale*interval[1]:+.2f}]")


def _episode_link(path: str | None, output: Path) -> str:
    if not path:
        return "N/A"
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.relative_to(output)
        except ValueError:
            pass
    return f"[轨迹]({p.as_posix()})"


def write_report(output: Path | str) -> dict:
    """Write per-stage statistics, a transparent decision and REPORT.md.

    Existing experiment protocols and runner decisions are never changed.
    No absent confirmation is invented or inferred from the bank.
    """
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stages = {}
    for stage in ("development", "confirmation"):
        path = _stage_path(output, stage)
        if path is None:
            continue
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError(f"Expected list of rows in {path}")
        if not rows:
            continue
        stage_stats = statistics(rows)
        stage_stats["stage"] = stage
        stage_stats["evidence_status"] = "independent_confirmation" if stage == "confirmation" else "development_screen_only"
        _save(path.with_name("statistics.json"), stage_stats)
        stages[stage] = stage_stats

    confirm = stages.get("confirmation")
    development = stages.get("development")
    if confirm:
        decision = dict(stage="confirmation", **confirm["decision"])
        decision["stop_modification_direction"] = not confirm["decision"]["rootpuct_followup"]
        conclusion = ("独立确认出现预登记的真实对手成本收益信号；只支持进一步研究，尚不授权训练或部署。"
                      if confirm["decision"]["rootpuct_followup"] else
                      "独立确认未同时通过真实对手下的两项成本收益门槛，停止这一登记修改方向。")
    elif development:
        decision = dict(stage="development", **development["decision"])
        decision["stop_modification_direction"] = bool(
            development["decision"]["development_stop_rule"]
            or not development["decision"]["engineering_gate_passed"])
        conclusion = ("开发筛选触发停止条件，确认未执行；这是开发阶段未确认，不是搜索普遍无效的证明。"
                      if decision["stop_modification_direction"] else
                      "开发筛选未触发预登记停止条件，尚无独立确认结果；不能声称搜索收益已经成立。")
    else:
        decision = dict(stage="not_started", stop_modification_direction=None,
                        no_capture_noninferiority_claim=True,
                        neural_training_authorized=False, deployment_change_authorized=False)
        conclusion = "尚无完整开发或确认结果；本报告不作效果判断。"
    _save(output / "analysis_decision.json", decision)

    lines = ["# 搜索价值验证：模拟观测反馈与信念搜索", "",
             "2026-10-26，Asia/Shanghai。冻结 D 与强 G；本轮研究无神经网络、无价值网络，不改部署。", "",
             f"**当前结论：{conclusion}**", "",
             "## 研究问题与对照", "",
             "OL 与 FB 在相同物理规划时长、公共信息、运动模型和候选条件下，对照未来模拟观测是否进入树内信念与后续策略。",
             "G 为强几何基线；OL200/FB200 为200ms实际上限版本，OLH/FBH为另列实际耗时的高预算诊断，不称实时部署版本。",
             "模型匹配目标只在首次正观测前严格匹配，故主要终点为失败惩罚首次重发现成本。重发现后的捕获只作描述，不称严格模型匹配捕获。",
             "真实 RootPUCT 对手条件以失败惩罚完成成本为主要终点。两种对手的绝对胜率不可直接当成观测反馈的因果效果。", "",
             "## 终点、分组与预登记门槛", "",
             "完成成本：成功用接管后完成步数，失败用接管时剩余步数。重发现成本：首次发现用发现步数，未发现用原剩余步数。失败不从均值删除。",
             "四项主要比较为FBH−G、FBH−OLH分别在matched与rootpuct条件下的主要成本终点。公共历史为独立统计组，组内全部世界、五方法及两种对手共同保留；配对bootstrap10000次，种子84120261026。",
             "四项主要成本区间采用共同max-absolute-centered半径给出同时95%区间；捕获与重发现率差仅给普通描述性区间。其余预算对照不进入通过决定。",
             "成本信号通过条件：主要成本差同时95%上界<0、观测成功率不下降、无工程失败；matched使用重发现率，RootPUCT使用捕获率。成功率条件只是探索性筛选，**不构成捕获率或重发现率非劣、等效证明**；天花板下bootstrap退化亦不能支持此类结论。",
             "只有RootPUCT条件下FBH同时胜过G与OLH，才考虑后续研究。匹配诊断单独成功不足以授权教师生成、神经训练或部署。",
             "开发继续门槛：至少一种对手下，FBH相对G与OLH的主要成本点估计均≤−1.0步，且两项比较的观测成功率均不下降，全部无工程失败，才运行完整新确认；matched成功率为重发现率，RootPUCT为捕获率。否则开发停止。门槛在读取结果前冻结；开发无需先取得显著性，也不把通过筛选称为确认收益。",
             "早期建议的‘四项成本均无改善且捕获无改善才停止’规则已被上述最小有用信号门槛替代；统计保留其描述值但不用于本轮决定。", ""]
    if not confirm:
        lines += ["**确认状态：未读取到完整确认结果，不把预先冻结的确认银行当作已执行实验。**", ""]

    for stage, stats in stages.items():
        name = "开发筛选" if stage == "development" else "独立确认"
        lines += [f"## {name}：{stats['groups']}组公共历史、{stats['games']}局", "",
                  "|目标条件|方法|合法捕获|重发现|完成成本|重发现成本|工程失败|",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for opponent in OPPONENTS:
            for method in METHODS:
                s = stats["summary"][opponent][method]
                lines.append(f"|{opponent}|{method}|{s['capture']}/{s['games']}|{s['rediscovered']}/{s['games']}|"
                             f"{s['completion_cost']:.2f}|{s['rediscovery_cost']:.2f}|{s['engineering_failures']}|")
        lines += ["", "|主要比较|主要终点|成本差步 同时95% CI|捕获差pp 普通95% CI|重发现差pp 普通95% CI|成本信号门槛|",
                  "|---|---|---:|---:|---:|---|"]
        for c in stats["comparisons"].values():
            if c["main"]:
                lines.append(f"|{c['opponent']} {c['method']}−{c['baseline']}|{c['primary_endpoint']}|"
                             f"{_interval(c[c['primary_endpoint']], True)}|{_interval(c['capture'], scale=100.)}|"
                             f"{_interval(c['rediscovery'], scale=100.)}|{'通过' if c['reliable_cost_signal'] else '未通过'}|")
        lines += ["", "区间跨零表示证据不足，不表示等效。开发表中的‘通过’只指计算出的筛选信号，不替代独立确认。", "",
                  "### 预算差异与200ms结果（辅助）", "",
                  "|目标条件|比较|完成成本差步 普通95% CI|重发现成本差步 普通95% CI|捕获差pp 普通95% CI|",
                  "|---|---|---:|---:|---:|"]
        for c in stats["comparisons"].values():
            if not c["main"]:
                lines.append(f"|{c['opponent']}|{c['method']}−{c['baseline']}|{_interval(c['completion_cost'])}|"
                             f"{_interval(c['rediscovery_cost'])}|{_interval(c['capture'], scale=100.)}|")
        lines += ["", "高预算优于低预算只说明该登记策略存在计算代价；若未同时优于G，不认定有值得训练网络压缩的真实任务收益。", "",
                  "### 实际耗时与完成模拟数", "",
                  "|目标条件|方法|决策P95/最大ms|初始化+首决策P95/最大ms|决策>200ms|首次>200ms|每决策平均完成模拟|有搜索帧平均模拟|",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for opponent in OPPONENTS:
            for method in METHODS:
                s = stats["summary"][opponent][method]
                t, initial, sims = s["decision_timing"], s["initial_timing"], s["simulations"]
                lines.append(f"|{opponent}|{method}|{_format(t['p95_ms'])}/{_format(t['maximum_ms'])}|"
                             f"{_format(initial['p95_ms'])}/{_format(initial['maximum_ms'])}|{t['over_200ms']}|"
                             f"{initial['over_200ms']}|{_format(sims['mean'], 1)}|{_format(sims['positive_mean'], 1)}|")
        lines += ["", "决策计时使用记录的实际入口到返回耗时；首次总计另含初始化。200ms是实际上限，G不等待。高预算的>200ms只作描述，不视为200ms工程失败。",
                  "模拟均值按记录全部决策统计；‘有搜索帧’仅筛选完成模拟数>0的帧，不等于独立的隐藏帧定义，也不能掩盖隐藏帧零模拟。", "",
                  "### 拓扑分层（描述性）", "",
                  "|拓扑|目标条件|方法|合法捕获|重发现|完成成本|重发现成本|",
                  "|---|---|---|---:|---:|---:|---:|"]
        for family, families in stats["by_family"].items():
            for opponent in OPPONENTS:
                for method in METHODS:
                    s = families[opponent][method]
                    if not s["games"]:
                        continue
                    lines.append(f"|{family}|{opponent}|{method}|{s['capture']}/{s['games']}|"
                                 f"{s['rediscovered']}/{s['games']}|{s['completion_cost']:.2f}|{s['rediscovery_cost']:.2f}|")
        lines += ["", "拓扑分层不用于挑选有利结果；多通路正例不能替代总体成组比较。", "",
                  "### 失败与代表轨迹", "",
                  "|失败类型|局数|", "|---|---:|"]
        failure_counts = collections.Counter(r["reason"] for r in stats["failure_cases"])
        for reason in ("engineering_failure", "not_rediscovered", "rediscovered_not_captured"):
            lines.append(f"|{reason}|{failure_counts[reason]}|")
        lines += ["", "|目标条件|方法|历史/世界|首次发现|成本/G成本|失败类型|轨迹|",
                  "|---|---|---|---:|---:|---|---|"]
        for opponent in OPPONENTS:
            for method in ("FB200", "FBH"):
                failures = sorted((r for r in stats["failure_cases"] if r["opponent"] == opponent
                                   and r["method"] == method),
                                  key=lambda r: (-r["difference_from_G"], r["group"], r["world"]))[:3]
                for r in failures:
                    lines.append(f"|{opponent}|{method}|{r['group']} / {r['world']}|{r['first_seen']}|"
                                 f"{r['cost']:.0f}/{r['G_cost']:.0f}|{r['reason']}|{_episode_link(r['episode'], output)}|")
        lines += ["", "代表轨迹按相对G完成成本差选取，仅用于解释失败，不回流修改算法或重新筛选确认场景。", ""]

    lines += ["## 证据边界与交付", "",
              "本分析只处理已完成逐局结果，不验证隐藏真值隔离、动作/传感器合法性或冻结D散列；这些工程结论必须由独立审计记录支持。",
              "匹配目标的后段捕获不构成全程模型匹配证据。相同三类拓扑的新实例不等于未见拓扑或真实机器人泛化。",
              "本轮信号仅用于判断是否值得进一步研究，不授权自动生成教师、训练网络、添加价值模块或修改部署。", "",
              "- [本分析结论](analysis_decision.json)。"]
    for stage in stages:
        path = _stage_path(output, stage)
        assert path is not None
        relative = path.relative_to(output)
        lines.append(f"- [{stage}逐局结果]({relative.as_posix()})、"
                     f"[{stage}成组统计]({relative.with_name('statistics.json').as_posix()})。")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return dict(stages={s: dict(games=v["games"], groups=v["groups"]) for s, v in stages.items()},
                decision=decision, report=str(output / "REPORT.md"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(write_report(args.output), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
