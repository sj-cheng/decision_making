"""Public-history recovery engineering checks, not an effect evaluation."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np

from audit_l45_rollout_reliability import sha, save, observation
from solvers.local_pursuit import PublicMap
from solvers.reliable_local_pursuit_v2 import Controller, VERSION, SimulationReliabilityError
from solvers.history_recovery import lattice_paths
from verify_l45_occlusion_scenarios import movement, visible


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT/"current/l45_occlusion_scenarios_20261025/bank"
DEFAULT = ROOT/"current/l45_estimator_recovery_20261025"


def run(output):
    if (output/"results.json").exists():
        raise RuntimeError("Completed audit cannot be overwritten; use a new output")
    manifest = json.loads((SOURCE/"manifest.json").read_text())
    entries = manifest["entries"]
    if len(entries) != 24:
        raise RuntimeError("Expected frozen 24-history development set")
    files = [Path(__file__), ROOT/"code/audit_l45_rollout_reliability.py",
        ROOT/"code/solvers/local_pursuit.py", ROOT/"code/solvers/reliable_local_pursuit.py",
        ROOT/"code/solvers/reliable_local_pursuit_v2.py", ROOT/"code/solvers/history_recovery.py",
        ROOT/"code/tests/test_reliable_local_pursuit.py", ROOT/"code/tests/test_reliable_local_pursuit_v2.py",
        ROOT/"code/tests/fixtures/particle_recovery_boundary_sliver.json",
        ROOT/"code/tests/fixtures/particle_recovery_corner_contact.json",
        ROOT/"code/verify_l45_occlusion_scenarios.py", SOURCE/"manifest.json"]
    files += [SOURCE/e["public"] for e in entries]
    frozen = {str(p): sha(p) for p in files}
    signature = dict(version=VERSION, stage="engineering_only", histories=24, streams=2,
        budgets=[16, 128], exact_resume=True, root_sources="frozen previously used public inputs only",
        no_hidden_environment_inputs=True, no_real_games=True, no_teacher_queries=True,
        no_labels=True, no_network_training=True, sources=frozen)
    if (output/"experiment.json").exists() and json.loads((output/"experiment.json").read_text()) != signature:
        raise RuntimeError("Source/configuration changed; create another audit output")
    save(output/"experiment.json", signature)
    fixture = json.loads((ROOT/"code/tests/fixtures/particle_recovery_boundary_sliver.json").read_text())
    last, observers, boxes = (np.array(fixture[k]) for k in ("last", "observers", "boxes"))
    # Warm the new lattice and verify the original failure's full ancestral
    # paths independently before any new root probes.
    witnesses = []
    for spacing in (.5, .25):
        particles, paths, n = lattice_paths(last, observers, boxes, spacing,
            np.array([17], np.int64), np.zeros(6, np.int64))
        passed = n == 64
        for path in paths[:n]:
            for t, at in enumerate(observers):
                passed = passed and movement(path[t].tolist(), path[t+1].tolist(), 1., boxes.tolist())
                passed = passed and not visible(at.tolist(), path[t+1].tolist(), boxes.tolist())
        witnesses.append(dict(spacing=spacing, sampled_endpoints=n,
            unique_endpoints=len(np.unique(particles[:n], axis=0)), independent_checks_passed=bool(passed)))
    save(output/"witness_checks.json", witnesses)
    probes = []
    for entry in entries:
        packet = json.loads((SOURCE/entry["public"]).read_text())
        p = PublicMap(packet["map"], packet["motion"]["pursuer_axis_limit"])
        for stream, seed in enumerate(packet["controller_seeds"]):
            for budget in signature["budgets"]:
                own = Controller(p, "T", seed, simulations=budget, deadline_ms=0.)
                row = dict(history=entry["id"], stream=stream, budget=budget, passed=False, failure=None)
                try:
                    for item in packet["history"][:-1]:
                        own.update(observation(item), np.zeros(6, np.int64))
                    before = own.snapshot()
                    a = own.act(observation(packet["history"][-1]), fixed_simulations=budget)
                    after, stats = own.snapshot(), own.last_stats.copy()
                    own.restore(before)
                    b = own.act(observation(packet["history"][-1]), fixed_simulations=budget)
                    deterministic = lambda s: {k: v for k, v in s.items() if k != "latency_ms"}
                    replay = (np.array_equal(a, b) and deterministic(stats) == deterministic(own.last_stats)
                        and np.array_equal(after["rng"], own.rng)
                        and np.array_equal(after["filter_rng"], own.filter_rng)
                        and np.array_equal(after["particles"], own.particles))
                    at = np.array(packet["history"][-1]["position"])
                    valid = movement(at.tolist(), (at+a).tolist(), p.speed, packet["map"])
                    row.update(passed=bool(replay and valid), replay_passed=bool(replay),
                        independent_action_legal=bool(valid), action=a.tolist(), statistics=stats)
                except SimulationReliabilityError as exc:
                    row.update(failure=str(exc), statistics=own.last_stats)
                probes.append(row)
                save(output/"root_checks.json", probes)
                save(output/"status.json", dict(stage="root_checks", terminal=False, completed=len(probes),
                    total=96, failures=sum(not x["passed"] for x in probes), pid=os.getpid()))
                if len(probes) % 8 == 0:
                    print(f'Root checks {len(probes)}/96; failures {sum(not x["passed"] for x in probes)}', flush=True)
    unchanged = all(sha(Path(p)) == h for p, h in frozen.items())
    failures = [x for x in probes if not x["passed"]]
    summary = dict(version=VERSION, engineering_passed=not failures and unchanged and all(x["independent_checks_passed"] for x in witnesses),
        root_requests=len(probes), passed=len(probes)-len(failures), failed=len(failures), source_hashes_unchanged=unchanged,
        witnesses=witnesses, by_budget={}, new_real_games=0, new_teacher_queries=0, new_labels=0,
        network_updates=0, formal_comparison_started=False, deployment_changed=False,
        estimator_coverage_validated=False, model_accuracy_validated=False,
        time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"))
    for budget in signature["budgets"]:
        rows = [x for x in probes if x["budget"] == budget]
        stats = [x["statistics"] for x in rows]
        summary["by_budget"][str(budget)] = dict(passed=sum(x["passed"] for x in rows), total=len(rows),
            simulations=sum(s.get("simulations", 0) for s in stats),
            terminations={k: sum(s.get("terminations", {}).get(k, 0) for s in stats)
                          for k in ("capture", "horizon", "own_invalid", "opponent_invalid", "belief_failure")},
            stochastic_recovery_calls=sum(s.get("stochastic_recovery_calls", 0) for s in stats),
            lattice_recovery_attempts=sum(s.get("lattice_recovery_attempts", 0) for s in stats),
            lattice_recovery_failures=sum(s.get("lattice_recovery_failures", 0) for s in stats),
            root_latency_mean_ms=float(np.mean([s.get("latency_ms", 0) for s in stats])),
            root_latency_p95_ms=float(np.percentile([s.get("latency_ms", 0) for s in stats], 95)))
    save(output/"results.json", summary)
    save(output/"status.json", dict(stage="engineering_passed" if summary["engineering_passed"] else "engineering_failed",
        terminal=True, pid=None, completed=96, total=96, training_started=False))
    report = ["# 粒子恢复可靠性：有合法祖先路径的恢复版本\n\n",
        f'版本`{VERSION}`，工程检查通过：**{summary["engineering_passed"]}**。\n\n',
        "旧版本与结果保留。只使用已查看开发公共历史；没有新真实对局、教师查询、标签或网络训练。\n\n",
        "## 原故障与修复\n\n",
        "已精确复现原第5次模拟、第22个转移后的失败，诊断副本与原数组/RNG一致。最后测量后仅两步空观测，合法支持被挤至场地边界的狭窄区域；旧随机恢复未采到支持。\n\n",
        "新版本保留原随机恢复；失败后才从最后真实/模拟可见测量和全部空观测枚举有限路径格。格点按最后测量锚定，并保留精确场地边界；每步都验证限幅、线段合法性与不可见。0.5米失败才用0.25米，不引入隐藏目标位置。\n\n",
        "原故障的两种分辨率均返回64个带祖先路径的样本，实际各只有2个不同末端，独立几何核验通过。64不是64个不同假设，更不是校准概率。固定世界中心格在这个狭窄区域仍会遗漏支持，相关工程草稿检查保留。\n\n",
        "## 全开发公共历史根检查\n\n",
        "|每根模拟预算|通过/总数|完成模拟|捕获结束|时限结束|非法动作结束|估计失败|格路径恢复次数|根检查均值/P95 ms|\n|---|---:|---:|---:|---:|---:|---:|---:|---|\n"]
    for budget, x in summary["by_budget"].items():
        t = x["terminations"]
        report.append(f'|{budget}|{x["passed"]}/{x["total"]}|{x["simulations"]}|{t["capture"]}|{t["horizon"]}|{t["own_invalid"]+t["opponent_invalid"]}|{t["belief_failure"]}|{x["lattice_recovery_attempts"]}|{x["root_latency_mean_ms"]:.2f}/{x["root_latency_p95_ms"]:.2f}|\n')
    report += ["\n每项均另做相同预算的精确恢复，动作/根统计/搜索与估计两个随机流及粒子相同；表中不重复计入恢复模拟。根检查是历史开发工程检查，不是独立捕获率实验，内部捕获计数不能当作真实胜率。\n\n",
        "计时来自固定模拟预算根检查，可能包含新Numba签名首次编译，不能当作50ms部署延迟证明。格路径恢复有额外开销，必须在后续真实预算比较中完整计费。\n\n",
        "## 判断边界与下一步\n\n",
        "该修复解决的是有限采样耗尽后的合法恢复与显式错误处理。粒子没有耗尽时仍沿用原估计，所以不自动修复正常运行但支持偏离真实目标的问题；有限格也不保证覆盖全部连续合法路径。任何无法恢复的历史仍显式失败，不回填真值，不按零终值继续评分。\n\n",
        "G/T共享同一恢复规则；当前没有T+Geo或神经训练。下一步在固定共同观测历史上检查估计覆盖/重发现条件，并冻结独立G/T/T+Geo等实际时间对照。工程通过不能替代这些阶段。\n\n",
        "详见`failure_reproduction.json`、`failure_public_inputs.npz`、`witness_checks.json`、`root_checks.json`、`results.json`及源码测试。\n"]
    if failures:
        report.append("\n失败项：\n\n")
        for row in failures:
            report.append(f'- `{row["history"]}`/流{row["stream"]}/预算{row["budget"]}：{row["failure"]}\n')
    (output/"REPORT.md").write_text("".join(report))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    print(json.dumps(run(parser.parse_args().output.resolve()), ensure_ascii=False, indent=2))
