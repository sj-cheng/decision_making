"""Bounded engineering audit; reads public development histories only.

No real games, privileged expert queries, labels or network updates. This is
not an effect evaluation and must not be merged into historical results.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import sys
from zoneinfo import ZoneInfo

import numpy as np

from solvers.local_pursuit import PublicMap, Observation, proposal as old_proposal
from solvers.reliable_local_pursuit import Controller, proposal, VERSION, SimulationReliabilityError
from verify_l45_occlusion_scenarios import movement


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT/"current/l45_occlusion_scenarios_20261025/bank"
DEFAULT = ROOT/"current/l45_rollout_reliability_20261025"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def observation(row):
    return Observation(row["step"], np.array(row["position"], float),
                       None if row["target"] is None else np.array(row["target"], float), row["remaining"])


def run(output):
    if output.exists() and (output/"results.json").exists():
        raise RuntimeError("Audit already completed; inspect its records or use a new --output")
    entries = json.loads((SOURCE/"manifest.json").read_text())["entries"]
    paths = [SOURCE/e["public"] for e in entries]
    if len(paths) != 24:
        raise RuntimeError("Expected 24 frozen public histories")
    code = [Path(__file__), ROOT/"code/solvers/reliable_local_pursuit.py",
            ROOT/"code/tests/test_reliable_local_pursuit.py", ROOT/"code/solvers/local_pursuit.py",
            ROOT/"code/verify_l45_occlusion_scenarios.py"]
    frozen = {str(p): sha(p) for p in code+paths+[SOURCE/"manifest.json"]}
    signature = dict(version=VERSION, type="engineering_only", public_histories=24,
        streams=2, root_simulations=16, sampler_seeds=list(range(256)),
        no_environment_inputs=True, no_real_games=True, no_teacher_queries=True,
        no_training=True, no_effect_claim=True, sources=frozen)
    if (output/"experiment.json").exists():
        if json.loads((output/"experiment.json").read_text()) != signature:
            raise RuntimeError("Audit sources or configuration changed; use a new output")
    save(output/"experiment.json", signature)
    save(output/"status.json", dict(stage="sampler_checks", terminal=False, pid=os.getpid()))
    # Paired draw checks exercise precisely the old unchecked-base path.
    sites = [("boundary", [], [9.95, 0.], [1., 0.]),
             ("wall", [[[-1., 1.], [-8., 8.]]], [-1.01, 0.], [1., 0.]),
             ("corner", [], [-9.99, 9.99], [-1., 1.])]
    draws = []
    for name, boxes, at, base in sites:
        public = PublicMap(boxes, 1.1)
        old_bad = new_bad = repairs = 0
        for seed in range(256):
            common = (np.array(at), np.array(base), 1., public.boxes)
            old, _ = old_proposal(*common, np.array([seed], np.int64), np.zeros(2, np.int64))
            audit = np.zeros(4, np.int64)
            new, _ = proposal(*common, np.array([seed], np.int64), np.zeros(2, np.int64), audit)
            old_bad += not movement(at, (np.array(at)+old).tolist(), 1., boxes)
            new_bad += not movement(at, (np.array(at)+new).tolist(), 1., boxes)
            repairs += int(audit[0])
        draws.append(dict(site=name, draws=256, old_illegal=old_bad, repaired_illegal=new_bad,
                          repaired_bases=repairs))
    save(output/"sampler_checks.json", draws)
    probes = []
    for entry, path in zip(entries, paths):
        packet = json.loads(path.read_text())
        public = PublicMap(packet["map"], packet["motion"]["pursuer_axis_limit"])
        for stream, seed in enumerate(packet["controller_seeds"]):
            own = Controller(public, "T", seed, simulations=16, deadline_ms=0.)
            row = dict(history=entry["id"], stream=stream, stage="history_initialization",
                       failure=None, passed=False)
            try:
                count = np.zeros(2, np.int64)
                for item in packet["history"][:-1]:
                    own.update(observation(item), count)
                before = own.snapshot()
                action = own.act(observation(packet["history"][-1]), fixed_simulations=16)
                after, stats = own.snapshot(), own.last_stats.copy()
                row.update(stage="root_search", action=action.tolist(), statistics=stats)
                # Exact fixed-budget recovery; latency is a new measurement.
                own.restore(before)
                again = own.act(observation(packet["history"][-1]), fixed_simulations=16)
                replay_stats = {k: v for k, v in own.last_stats.items() if k != "latency_ms"}
                science = {k: v for k, v in stats.items() if k != "latency_ms"}
                replay_ok = (np.array_equal(action, again) and science == replay_stats
                             and np.array_equal(after["rng"], own.rng)
                             and np.array_equal(after["filter_rng"], own.filter_rng)
                             and np.array_equal(after["particles"], own.particles))
                at = packet["history"][-1]["position"]
                lawful = movement(at, (np.array(at)+action).tolist(), public.speed, packet["map"])
                row.update(replay_passed=bool(replay_ok), independent_action_legal=bool(lawful),
                           passed=bool(replay_ok and lawful))
                if not row["passed"]:
                    raise RuntimeError("Independent legality or exact replay failed")
            except SimulationReliabilityError as exc:
                row.update(stage="estimator_or_search_reliability", failure=str(exc), statistics=own.last_stats)
            # Save public-only failure/success state for a reproducible audit.
            checkpoint = output/"snapshots"/f'{entry["id"]}_s{stream}.pkl'
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(pickle.dumps(own.snapshot(), protocol=5))
            row["snapshot"] = dict(path=str(checkpoint), sha256=sha(checkpoint))
            probes.append(row)
            save(output/"root_checks.json", probes)
            save(output/"status.json", dict(stage="root_checks", terminal=False, completed=len(probes),
                 total=48, failures=sum(not x["passed"] for x in probes), pid=os.getpid()))
            print(f'Public history roots: {len(probes)}/48; failed: {sum(not x["passed"] for x in probes)}', flush=True)
    unchanged = all(sha(Path(p)) == h for p, h in frozen.items())
    sampler_passed = sum(x["old_illegal"] for x in draws) > 0 and all(x["repaired_illegal"] == 0 for x in draws)
    passed = sampler_passed and unchanged and all(x["passed"] for x in probes)
    summary = dict(version=VERSION, engineering_checks_passed=passed, sampler_checks_passed=sampler_passed,
        source_hashes_unchanged=unchanged, public_root_requests=len(probes),
        public_roots_passed=sum(x["passed"] for x in probes), public_roots_failed=sum(not x["passed"] for x in probes),
        sampler=draws, new_real_games=0, new_teacher_queries=0, new_labels=0, network_updates=0,
        formal_comparison_started=False, deployment_changed=False,
        interpretation="Engineering roots only; not success-rate or estimation-coverage validation",
        time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"))
    save(output/"results.json", summary)
    save(output/"status.json", dict(stage="engineering_passed" if passed else "reliability_gate_failed",
        terminal=True, completed=len(probes), total=48, pid=None, training_started=False))
    problems = [x for x in probes if not x["passed"]]
    report = ["# 局部rollout可靠性修复：工程验收\n\n",
        f"版本：`{VERSION}`；工程检查通过：**{passed}**。\n\n",
        "只读取24组旧开发公共历史；没有读取隐藏环境初态，没有新真实对局、教师查询、标签或网络更新。\n\n",
        "|动作采样探针|种子数|旧非法动作|修复后非法动作|非法base回退修复|\n|---|---:|---:|---:|---:|\n"]
    for x in draws:
        report.append(f'|{x["site"]}|{x["draws"]}|{x["old_illegal"]}|{x["repaired_illegal"]}|{x["repaired_bases"]}|\n')
    report += [f'\n历史接管根检查：{summary["public_roots_passed"]}/48通过，{len(problems)}/48失败。每根16次模拟，成功根另做同预算精确恢复；这些根不是独立科学样本。\n\n',
        "失败被显式登记，未按正常零价值终止回传，不删除失败历史。通过只表明这批根搜索的合法性、恢复与执行可靠性，不能证明估计支持覆盖真值、概率校准、真实对手模型准确或捕获率提高。\n\n"]
    if problems:
        report.append("可靠性门槛尚未通过：先处理以下公共历史的共用估计/模拟问题，未启动G/T/T+Geo效果比较或网络训练。\n\n")
        for x in problems:
            report.append(f'- `{x["history"]}` / 流{x["stream"]}：{x["failure"]}\n')
    else:
        report.append("本批工程门槛通过。下一步仍需状态估计覆盖与对手模型验收，再冻结G/T/T+Geo独立新场景协议；未自动启动正式效果评测或教师训练。\n")
    report.append("\n详见`experiment.json`、`sampler_checks.json`、`root_checks.json`及公共状态`snapshots/`。历史源码和结果未覆盖。\n")
    (output/"REPORT.md").write_text("".join(report))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve()), ensure_ascii=False, indent=2))
