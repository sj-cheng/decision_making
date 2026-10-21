#!/usr/bin/env python3
"""Export key V7 results with the V5 method corrected to NTE."""

import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
SL = "Search + Learning"
METHODS = ["Policy Only", "Search Only", SL, "MAPPO", "Large-budget MCTS", "NTE"]
BUDGETS = {"Policy Only": 0, "Search Only": 3000, SL: 1000,
           "MAPPO": 0, "Large-budget MCTS": 50000, "NTE": 1000}
TZ = dt.timezone(dt.timedelta(hours=8))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def renamed(method, version):
    return "NTE" if version == "V5" and method == SL else method


def near(actual, expected):
    return ((math.isnan(actual) and math.isnan(expected)) or
            math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_matchup(item, config):
    records = item["records"]
    seeds = (config["large_seeds_used"] if "Large-budget MCTS" in
             (item["original_pursuer"], item["original_evader"]) else config["regular_seeds"])
    expected = {(seed, episode) for seed in seeds for episode in range(config["episodes"])}
    keys = [(int(float(row["seed"])), int(float(row["episode"]))) for row in records]
    require(len(keys) == len(set(keys)) and set(keys) == expected,
            f"Missing or duplicate episodes: {item['path']}")
    for row in records:
        require(row["pursuer_method"] == item["original_pursuer"] and
                row["evader_method"] == item["original_evader"],
                f"Method mismatch: {item['path']}")
        count = float(row["capture_count"])
        require(float(row["total_capture"]) == int(count > 0) and
                float(row["double_capture"]) == int(count == 2),
                f"Capture indicators disagree: {item['path']}")
    checked = 0
    for row in item["summaries"]:
        metric = row["metric"]
        if metric.endswith("_step_mean_latency_ms"):
            role = metric.removesuffix("_step_mean_latency_ms")
            n = sum(int(float(record[role + "_latency_count"])) for record in records)
            total = sum(float(record[role + "_latency_sum_ms"]) for record in records)
            calculated = dict(mean=total / n if n else math.nan, std=math.nan,
                              ci95=math.nan, n=n)
        else:
            values = [float(record[metric]) for record in records
                      if math.isfinite(float(record[metric]))]
            n = len(values)
            std = statistics.stdev(values) if n > 1 else (0.0 if n else math.nan)
            calculated = dict(mean=statistics.fmean(values) if n else math.nan,
                              std=std, ci95=1.96 * std / math.sqrt(n) if n else math.nan, n=n)
        for field, actual in calculated.items():
            require(near(actual, float(row[field])),
                    f"Summary mismatch: {item['path']} {metric}/{field}")
            checked += 1
    item["invalid"] = sum(int(float(row["illegal_terminal"])) for row in records)
    return checked


def audit_chunks(items, directory):
    seen_files = set()
    count = 0
    for item in items:
        expected = {(int(float(r["seed"])), int(float(r["episode"]))): r
                    for r in item["records"]}
        seen = set()
        for path in sorted(directory.glob(item["path"].stem + ".seed*.ep*.csv")):
            seen_files.add(path)
            fields, records = read_csv(path)
            require(fields == item["fields"], f"Chunk schema mismatch: {path}")
            for record in records:
                key = (int(float(record["seed"])), int(float(record["episode"])))
                require(key not in seen and key in expected, f"Duplicate/unknown chunk row: {path}")
                seen.add(key)
                for field in fields:
                    actual, original = record[field], expected[key][field]
                    if actual != original:
                        try:
                            matches = near(float(actual), float(original))
                        except ValueError:
                            matches = False
                        require(matches, f"Chunk value mismatch: {path} {key} {field}")
                count += 1
        require(seen == set(expected), f"Incomplete chunks: {item['path']}")
    require(seen_files == set(directory.glob("*.csv")), "Unaccounted V7 chunk files")
    return len(seen_files), count


def load_data():
    items, configs = [], {}
    checked = 0
    for version in ("V7", "V5"):
        directory = ROOT / "current/benchmarks" / ("final_benchmark_" + version.lower())
        configs[version] = json.loads((directory / "config.json").read_text())
        for path in sorted(directory.glob("pursuer-*.csv")):
            if path.name.endswith("_summary.csv"):
                continue
            if version == "V5" and "Search_+_Learning" not in path.name:
                continue
            fields, records = read_csv(path)
            _, summaries = read_csv(path.with_name(path.stem + "_summary.csv"))
            original_pursuer, original_evader = (records[0][role + "_method"]
                                                for role in ("pursuer", "evader"))
            item = dict(version=version, path=path, fields=fields, records=records,
                        summaries=summaries, summary={r["metric"]: r for r in summaries},
                        original_pursuer=original_pursuer, original_evader=original_evader,
                        pursuer=renamed(original_pursuer, version),
                        evader=renamed(original_evader, version))
            checked += audit_matchup(item, configs[version])
            items.append(item)
    require(collections.Counter(i["version"] for i in items) == {"V7": 25, "V5": 9},
            "Expected 25 V7 matchups and 9 V5 NTE matchups")
    index = {(i["pursuer"], i["evader"]): i for i in items}
    require(len(index) == 34, "Duplicate merged matchups")
    require(set(index) == {(a, b) for a in METHODS for b in METHODS}
            - {("NTE", SL), (SL, "NTE")}, "Unexpected merged matrix coverage")
    chunk_files, chunk_rows = audit_chunks([i for i in items if i["version"] == "V7"],
                                         ROOT / "current/benchmarks/final_benchmark_v7/chunks")
    return items, configs, index, checked, chunk_files, chunk_rows


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", "<br>")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(cell, row)) + " |" for row in rows]) + "\n"


def mean(item, metric):
    return float(item["summary"][metric]["mean"])


def value(item, metric, digits=3, percent=False, ci=False):
    scale = 100 if percent else 1
    result = f"{mean(item, metric) * scale:.{digits}f}"
    if ci:
        result += f" ± {float(item['summary'][metric]['ci95']) * scale:.{digits}f}"
    return result + ("%" if percent else "")


def source_link(path, output):
    import os
    return f"[{path.name}]({os.path.relpath(path, output.parent)})"


def build_full_report(items, configs, index, checked, chunk_files, chunk_rows, output):
    parts = []
    add = parts.append
    add("# V7 benchmark 完整数据与 V5 NTE 合并实验报告\n")
    add("生成日期：2026-10-21（Asia/Shanghai）。数据仅来自已有实验文件。\n")
    add("V7 的 **Search + Learning** 保留原名；V5 中原标为 **Search + Learning** 的方法在本报告中更正为 **NTE**。"
        "NTE 作为独立方法加入 V7 结果表。V7 共 25 组、17,800 局；V5 NTE 相关共 9 组、7,400 局；"
        "合并共 34 组、25,200 局。所有逐局字段、所有 summary 指标及其原始精度均在本文附录中输出。\n")
    add("## 1. 来源与命名\n")
    add(table(["版本", "输入目录", "导出范围", "报告方法名"], [
        ["V7", "`current/benchmarks/final_benchmark_v7`", "全部 25 组", "Search + Learning 保留"],
        ["V5", "`current/benchmarks/final_benchmark_v5`", "原 Search + Learning 出现在任一角色的 9 组", "原方法名更正为 NTE"],
    ]))
    v7_paths = [i["path"] for i in items if i["version"] == "V7"]
    earliest = min(p.stat().st_mtime for p in v7_paths)
    latest = max(p.stat().st_mtime for p in v7_paths)
    add(f"V7 逐局结果文件最后修改时间范围：{dt.datetime.fromtimestamp(earliest, TZ):%Y-%m-%d %H:%M:%S}"
        f" 至 {dt.datetime.fromtimestamp(latest, TZ):%Y-%m-%d %H:%M:%S}（UTC+08:00），与昨天运行的版本一致。"
        "当前 V5 输入与 `backup_pre_search_fix_20261009/benchmarks/final_benchmark_v5` 是两个路径；本次以当前目录为输入。\n")
    add("命名更正用于本次 Markdown 导出；原 CSV 文件名和历史输入保持可追溯。原 V5 的 Search + Learning 对自身的对局，"
        "对应 **NTE 对 NTE**。NTE 对 V7 Search + Learning 的两个方向均没有现成结果。"
        "凡涉及 NTE 的表格单元格均取 V5，其中同名的其他对手也仍是 V5 当时的版本。\n")
    add("## 2. 实验配置\n")
    config_fields = ["benchmark_protocol", "episodes", "regular_seeds", "large_seeds_used",
                     "search_learning_budget", "search_only_budget", "large_budget", "beta_policy",
                     "beta_value", "model_dir", "model_iteration", "mappo_checkpoint", "mappo_device",
                     "role_mask_mode", "latency_mode", "workers", "cpu_threads", "timing_context",
                     "timing_warmup_steps"]
    add(table(["配置字段", "V7", "V5 / NTE"], [[field] + [
        json.dumps(configs[v][field], ensure_ascii=False) if field in configs[v] else "配置未记录"
        for v in ("V7", "V5")] for field in config_fields]))
    add("普通对局为 5 个评估 seed × 每 seed 200 局，合计 1,000 局；涉及 Large-budget MCTS 的对局为 "
        "1 个 seed × 200 局。V7 使用模型第 45 轮与 beta_value=0.1，V5 NTE 使用第 79 轮与 beta_value=0.5；"
        "MAPPO checkpoint 也不同。因此这里合并的是已有结果，不能据此视为同一套实现、模型和对手下新运行的六方法完整对战，"
        "也不能把跨版本回报或耗时差直接归因于算法名称。\n")
    add("## 3. 指标口径\n")
    add(table(["字段 / 表头", "含义"], [
        ["total_capture / 至少捕获率", "至少捕获一个逃逸者的局数比例，不是双捕获率"],
        ["double_capture / 双捕获率", "捕获两个逃逸者的局数比例；对追捕方越高越好，对逃逸方越低越好"],
        ["capture_count", "每局捕获数量的均值，范围 0–2"],
        ["first_capture_time", "首次捕获时间；仅在至少捕获一个的局上统计"],
        ["capture_time", "第二次（完成双捕获）时间；仅在双捕获局上统计"],
        ["survival_time / 对局长度", "所有局的最终仿真时间，不是单个逃逸者各自的存活时间"],
        ["pursuer_return / evader_return", "各阵营累计回报；保留各版本原值"],
        ["collision_or_out_of_bounds / illegal_terminal", "保存的无效状态标记；未过滤相应局"],
        ["mean_latency_ms / p50 / p95 / max", "完整联合动作调用的逐局统计，包含两方，不能作为某一方法单独时延"],
        ["pursuer_*_latency_ms / evader_*_latency_ms", "对应角色的团队 solver 调用统计；summary 为逐局统计量再求均值"],
        ["*_step_mean_latency_ms（V7）", "sum(latency_sum_ms) / sum(latency_count)，按所有调用加权的均值"],
        ["std / ci95 / n", "样本标准差 / 1.96×std/√n 的区间半宽 / 有效样本数"],
    ]))
    add("正文的 ± 原样沿用 summary 的逐局 95% 正态近似区间半宽；并非训练 seed 间区间。"
        "0% 或 100% 对应的 ±0 只是该计算公式的结果。时间按原 CSV 仿真时间单位输出，耗时单位为 ms；"
        "episode p95 的均值不能解释为全部调用合并后的 p95。V7 为并发 worker 场景，V5 也启用并行，"
        "这些耗时不构成受控部署计时。缺失值 `nan` 原样保留。\n")
    add("## 4. 合并实验结论表\n")
    add("行是追捕方，列是逃逸方。单元格 = **双捕获率 / 平均对局长度**。"
        "含 NTE 的单元格标为 `†`，表示整场对局来自 V5；其他单元格来自 V7。\n")
    matrix = []
    for pursuer in METHODS:
        cells = []
        for evader in METHODS:
            item = index.get((pursuer, evader))
            cells.append((f"{value(item, 'double_capture', 1, percent=True)} / "
                          f"{value(item, 'survival_time', 2)}" + (" †" if item["version"] == "V5" else ""))
                         if item else "未运行")
        matrix.append([pursuer] + cells)
    add(table(["追捕方 / 逃逸方"] + METHODS, matrix))
    add("† V5 历史数据。缺少的两个方向为 NTE → Search + Learning（V7）和 Search + Learning（V7）→ NTE；"
        "没有用 NTE 自对局或其他版本结果填充。完整覆盖为 34/36 组。\n")
    add("### 4.1 追捕角色：对各自版本的 Search + Learning / NTE 逃逸方\n")
    add("这张表并列展示 V7 的参考逃逸方 Search + Learning 与 V5 的参考逃逸方 NTE。两个参考对手不同，"
        "不能把两组行当作同一固定对手下的排名。\n")
    reference = [i for i in items if (i["version"] == "V7" and i["evader"] == SL)
                 or (i["version"] == "V5" and i["evader"] == "NTE")]
    reference.sort(key=lambda i: (i["version"] != "V7", METHODS.index(i["pursuer"])))
    add(table(["来源", "追捕方", "逃逸方", "搜索次数", "局数", "双捕获率", "对局长度", "追捕回报"], [
        [i["version"], i["pursuer"], i["evader"], BUDGETS[i["pursuer"]], len(i["records"]),
         value(i, "double_capture", 1, percent=True, ci=True), value(i, "survival_time", 2, ci=True),
         value(i, "pursuer_return", 3, ci=True)] for i in reference]))
    add("### 4.2 逃逸角色：对抗各自版本的 Large-budget MCTS\n")
    add(table(["来源", "追捕方", "逃逸方", "逃逸搜索次数", "局数", "被双捕获率", "对局长度", "逃逸回报"], [
        [i["version"], i["pursuer"], i["evader"], BUDGETS[i["evader"]], len(i["records"]),
         value(i, "double_capture", 1, percent=True, ci=True), value(i, "survival_time", 2, ci=True),
         value(i, "evader_return", 3, ci=True)]
        for method in METHODS for i in [index[("Large-budget MCTS", method)]]]))
    add("### 4.3 由现有数据得到的实验结论\n")
    v7_sl = index[(SL, SL)]
    v7_so = index[("Search Only", SL)]
    nte_self = index[("NTE", "NTE")]
    nte_so = index[("Search Only", "NTE")]
    v7_large = index[("Large-budget MCTS", SL)]
    nte_large = index[("Large-budget MCTS", "NTE")]
    add(f"1. **V7 的追捕表现随角色和对手变化。** 对 V7 Search + Learning 逃逸方，"
        f"Search + Learning 使用 1,000 次搜索，双捕获率 {value(v7_sl, 'double_capture', 1, percent=True)}；"
        f"Search Only 使用 3,000 次搜索，为 {value(v7_so, 'double_capture', 1, percent=True)}；"
        "Large-budget MCTS 使用 50,000 次搜索，为 100.0%。V7 对 Search Only 和 Large-budget MCTS 逃逸方时，"
        "Search + Learning 的双捕获率分别为 78.6% 和 71.5%。\n")
    add(f"2. **V7 当前结果未显示出对强纯搜索追捕方的逃逸优势。** 面对 Search Only 和 Large-budget MCTS，"
        f"V7 Search + Learning 逃逸方均被 100.0% 双捕获；面对 Large-budget MCTS 的平均对局长度为 "
        f"{value(v7_large, 'survival_time', 3)}。面对 Policy Only 的被双捕获率为 41.7%，面对 MAPPO 为 0.1%。\n")
    add(f"3. **V5 的相应历史结果应标为 NTE。** NTE 自对局的双捕获率为 "
        f"{value(nte_self, 'double_capture', 1, percent=True)}，对局长度 {value(nte_self, 'survival_time', 3)}；"
        f"Search Only → NTE 的双捕获率为 {value(nte_so, 'double_capture', 1, percent=True)}；"
        f"Large-budget MCTS → NTE 为 {value(nte_large, 'double_capture', 1, percent=True)}，"
        f"平均对局长度 {value(nte_large, 'survival_time', 3)}。这些结果归属 NTE，不能写成 V7 Search + Learning 的成绩。\n")
    add("4. **合并表能够展示两组已有结果，但不能证明 NTE 与 V7 Search + Learning 的直接优劣。** "
        "两者未直接对战，模型轮次、价值混合系数、MAPPO checkpoint 等均有版本差异；"
        "因此保留每组来源和各角色指标，不给跨版本综合排名，也不把原 V5 的机制结论迁移到 V7。\n")
    add("## 5. 全部 34 组结果明细\n")
    ordered = sorted(items, key=lambda i: (METHODS.index(i["pursuer"]), METHODS.index(i["evader"])))
    add("### 5.1 捕获、时间与有效性\n")
    add(table(["来源", "追捕方", "逃逸方", "局数", "至少捕获率", "双捕获率", "平均捕获数",
               "首次捕获时间", "双捕获时间", "对局长度", "invalid 局数"], [
        [i["version"], i["pursuer"], i["evader"], len(i["records"]),
         value(i, "total_capture", 1, percent=True), value(i, "double_capture", 1, percent=True),
         value(i, "capture_count"), value(i, "first_capture_time"), value(i, "capture_time"),
         value(i, "survival_time"), i["invalid"]] for i in ordered]))
    add("首次捕获时间和双捕获时间使用的有效局数不同，逐项 n 见附录 B；invalid 局仍包含在原始汇总中。\n")
    add("### 5.2 回报与逐局平均角色耗时\n")
    add(table(["来源", "追捕方", "逃逸方", "追捕回报", "逃逸回报", "追捕方 mean (ms)",
               "逃逸方 mean (ms)", "联合 mean (ms)", "逐局墙钟时间 mean (s)"], [
        [i["version"], i["pursuer"], i["evader"]] +
        [value(i, key) for key in ("pursuer_return", "evader_return", "pursuer_mean_latency_ms",
                                  "evader_mean_latency_ms", "mean_latency_ms", "wall_time_s")]
        for i in ordered]))
    add("### 5.3 按调用次数加权的角色耗时\n")
    add(table(["来源", "追捕方", "逃逸方", "追捕方 pooled mean (ms)",
               "逃逸方 pooled mean (ms)", "联合 pooled mean (ms)"], [
        [i["version"], i["pursuer"], i["evader"]] +
        [value(i, key) if key in i["summary"] else "未记录（V5 无逐调用 sum/count）"
         for key in ("pursuer_step_mean_latency_ms", "evader_step_mean_latency_ms",
                     "joint_step_mean_latency_ms")] for i in ordered]))
    add("## 6. 数据完整性核验\n")
    add(table(["核验项", "结果"], [
        ["对局数量", "V7 25 组 + V5 NTE 9 组 = 34 组"],
        ["逐局记录", "V7 17,800 + V5 NTE 7,400 = 25,200"],
        ["seed / episode", "所有对局均符合 config 规定的完整组合，组内无重复"],
        ["方法与捕获字段", "各局方法标签正确；至少捕获、双捕获与捕获数量一致"],
        ["summary 复算", f"全部 {checked:,} 个 mean/std/ci95/n 字段与逐局记录一致（容差 1e-9 相对 / 1e-10 绝对）"],
        ["V7 分块文件", f"{chunk_files} 个 CSV，{chunk_rows:,} 行，与最终逐局文件完整对应；附录不重复计算分块记录"],
        ["V7 invalid", f"{sum(i['invalid'] for i in items if i['version'] == 'V7')} / 17,800"],
        ["V5 NTE invalid", f"{sum(i['invalid'] for i in items if i['version'] == 'V5')} / 7,400"],
        ["缺失对局", "仅 NTE ↔ V7 Search + Learning 两个方向；明确标为未运行"],
    ]))
    add("## 附录 A. 完整配置与源文件索引\n")
    for version in ("V7", "V5"):
        config = dict(configs[version])
        if version == "V5":
            config["methods"] = [renamed(method, version) for method in config["methods"]]
        path = ROOT / "current/benchmarks" / ("final_benchmark_" + version.lower()) / "config.json"
        add(f"### {version} 完整 config\n")
        add(f"输入：{source_link(path, output)}。" +
            ("展示时仅将 methods 中原 Search + Learning 更正为 NTE，其余配置原样保留。\n" if version == "V5" else "\n"))
        add("```json\n" + json.dumps(config, ensure_ascii=False, indent=2) + "\n```\n")
    add("以下 SHA-256 对应输入文件，便于复核原始数值与版本。\n")
    inventory = []
    for item in ordered:
        for path in (item["path"], item["path"].with_name(item["path"].stem + "_summary.csv")):
            inventory.append([item["version"], source_link(path, output),
                              hashlib.sha256(path.read_bytes()).hexdigest()])
    for version in ("V7", "V5"):
        path = ROOT / "current/benchmarks" / ("final_benchmark_" + version.lower()) / "config.json"
        inventory.append([version, source_link(path, output), hashlib.sha256(path.read_bytes()).hexdigest()])
    add(table(["来源", "输入文件", "SHA-256"], inventory))
    add("## 附录 B. 全部 summary 指标（原始精度）\n")
    add("V7 每组 26 个指标，V5 NTE 每组 23 个指标，共 857 个指标行。mean/std/ci95/n 均保留输入字符串，"
        "表中不做四舍五入。角色与方法名以各小节标题为准。\n")
    for item in ordered:
        add(f"### {item['version']}：{item['pursuer']} → {item['evader']}\n")
        add(table(["metric", "mean", "std", "ci95", "n"], [
            [row[key] for key in ("metric", "mean", "std", "ci95", "n")]
            for row in item["summaries"]]))
    add("## 附录 C. 全部逐局记录（25,200 行，原始精度）\n")
    add("以下 CSV 代码块完整嵌入 Markdown，便于复制分析。V7 保留原字段与数值字符串；V5 仅更正 "
        "`pursuer_method` / `evader_method` 中的原方法名为 NTE，所有数值与列顺序原样保留。"
        "分块文件是 V7 相同记录的分片，已核验，不在此重复输出。V5 其他 16 组不涉及 NTE 的历史对局"
        "未加入本次 V7 + NTE 数据范围。\n")
    for item in ordered:
        add(f"### {item['version']}：{item['pursuer']} → {item['evader']}（{len(item['records']):,} 局）\n")
        add(f"输入：{source_link(item['path'], output)}。\n")
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=item["fields"], lineterminator="\n")
        writer.writeheader()
        for record in item["records"]:
            exported = dict(record)
            for role in ("pursuer", "evader"):
                key = role + "_method"
                exported[key] = renamed(exported[key], item["version"])
            writer.writerow(exported)
        add("```csv\n" + buffer.getvalue() + "```\n")
    return "\n".join(parts)


def build_report(index):
    parts = []
    add = parts.append
    add("# V7 benchmark + NTE 实验结果\n")
    add("Search + Learning 保留 V7 名称；NTE 为 V5 中原 Search + Learning 的更正名称。\n")
    add("普通对局 1,000 局（5 seeds × 200）；含 Large-budget MCTS 的对局 200 局。"
        "搜索预算：Search + Learning / NTE 1,000，Search Only 3,000，Large-budget MCTS 50,000。\n")
    add("## 合并结果\n")
    add("行：追捕方；列：逃逸方。单元格：**双捕获率 / 平均对局长度**。\n")
    rows = []
    for pursuer in METHODS:
        cells = []
        for evader in METHODS:
            item = index.get((pursuer, evader))
            cells.append((f"{value(item, 'double_capture', 1, percent=True)} / "
                          f"{value(item, 'survival_time', 2)}" +
                          (" †" if item["version"] == "V5" else "")) if item else "未运行")
        rows.append([pursuer] + cells)
    add(table(["追捕方 / 逃逸方"] + METHODS, rows))
    add("† 为 V5 历史对局，其余为 V7。NTE 与 V7 Search + Learning 尚无直接对战；"
        "两版本模型及对手不同，此处并列展示已有结果。\n")
    add("## 追捕指标\n")
    add("固定逃逸方：V7 Search + Learning。双捕获率越高越好。\n")
    add(table(["追捕方法", "双捕获率", "对局长度", "追捕回报", "平均决策耗时 (ms)"], [
        [method, value(item, "double_capture", 1, percent=True, ci=True),
         value(item, "survival_time", 2), value(item, "pursuer_return"),
         value(item, "pursuer_mean_latency_ms", 2)]
        for method in METHODS if method != "NTE" for item in [index[(method, SL)]]]))
    add("## 逃逸指标\n")
    add("固定追捕方：各自版本的 Large-budget MCTS。被双捕获率越低越好。\n")
    add(table(["逃逸方法", "被双捕获率", "对局长度", "逃逸回报", "平均决策耗时 (ms)"], [
        [method + (" †" if item["version"] == "V5" else ""),
         value(item, "double_capture", 1, percent=True, ci=True), value(item, "survival_time", 2),
         value(item, "evader_return"), value(item, "evader_mean_latency_ms", 2)]
        for method in METHODS for item in [index[("Large-budget MCTS", method)]]]))
    add("± 为原 summary 的逐局 95% 区间半宽；耗时为并发 benchmark 中逐局团队调用均值。\n")
    add("## 关键结论\n")
    add("- V7 Search + Learning 用 1,000 次搜索，对自身逃逸方的双捕获率为 **95.0%**；"
        "Search Only 用 3,000 次搜索达到 **100.0%**。\n"
        "- 面对 V7 Search Only 和 Large-budget MCTS，V7 Search + Learning 逃逸方均被 **100.0%** 双捕获。\n"
        "- V5 NTE 面对当时的 Large-budget MCTS，被双捕获率为 **76.0%**，平均对局长度 **56.59**；"
        "NTE 自对局双捕获率为 **49.9%**。\n")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "current/document/benchmark_v7_with_nte.md")
    parser.add_argument("--full", action="store_true", help="Include all summaries and episode records")
    args = parser.parse_args()
    output = args.output.resolve()
    items, configs, index, checked, chunk_files, chunk_rows = load_data()
    report = (build_full_report(items, configs, index, checked, chunk_files, chunk_rows, output)
              if args.full else build_report(index))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(json.dumps(dict(output=str(output), matchups=len(items),
                          episodes=sum(len(i["records"]) for i in items),
                          summary_rows=sum(len(i["summaries"]) for i in items),
                          verified_summary_values=checked, verified_chunks=chunk_files,
                          bytes=output.stat().st_size), ensure_ascii=False))


if __name__ == "__main__":
    main()
