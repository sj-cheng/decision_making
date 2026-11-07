"""Two concurrent full-state diagnostic runs in tmux:train."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "current/e45v_fullstate_hasac_20261103"


def read(path):
    try: return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError): return None


def write(path, obj):
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)); tmp.replace(path)


def main():
    audit = read(OUT / "smoke_audit.json")
    assert audit and audit["passed"]
    assert not (OUT / "launch_manifest.json").exists(), "Duplicate launch refused"
    records, jobs = [], []; started = time.time()
    for seed, gpu, cpus in ((1, 2, list(range(28, 56)) + list(range(84, 112))),
                            (2, 3, list(range(28, 56)) + list(range(84, 112)))):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1", PYTHONHASHSEED=str(seed),
                   PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
        name = f"seed_{seed}"
        cmd = ["taskset", "-c", ",".join(map(str, cpus)), str(OUT / "venv/bin/python"),
               str(ROOT / "code/run_e45v_fullstate.py"), "--seed", str(seed), "--name", name]
        logfile = (OUT / f"logs/{name}.log").open("w")
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logfile, stderr=subprocess.STDOUT)
        jobs.append((process, logfile)); records.append({"seed": seed, "name": name,
                "gpu": gpu, "pid": process.pid, "command": cmd, "exit_code": None, "status": None})
    write(OUT / "launch_manifest.json", {"started_unix": started,
          "started_beijing": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          "tmux_session": "train", "tmux_window": "V-fullstate", "runs": records,
          "planned_post_warmup_team_transitions": 2000000,
          "planned_team_transitions_with_warmup": 2020000, "no_automatic_extension": True})
    while True:
        for row, (process, _) in zip(records, jobs):
            row["exit_code"] = process.poll(); row["status"] = read(OUT / row["name"] / "status.json")
        done = all(row["exit_code"] is not None for row in records)
        write(OUT / "progress.json", {"state": "finished" if done else "running",
              "elapsed_seconds": time.time() - started, "runs": records})
        if done: break
        time.sleep(5)
    complete = all(r["exit_code"] == 0 and r["status"] and r["status"]["status"] == "complete" for r in records)
    tests = {str(r["seed"]): read(OUT / r["name"] / "sealed_test.json") for r in records}
    write(OUT / "completion_audit.json", {"complete": complete, "runs": records, "tests": tests,
          "wall_seconds": time.time() - started, "public_observation_training_started": False,
          "decision": "review_fullstate_curves_and_heldout_capture_before_next_stage" if complete else "engineering_failure",
          "budget_scope": "first 1M/seed diagnostic stage; no algorithm-invalidity claim"})
    lines = ["# E45-V 全状态诊断结果", "", f"完整结束：{complete}", "",
             "|种子|初始双捕|所选双捕|终点双捕|初始效用|所选效用|终点效用|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for seed, test in tests.items():
        if not test: continue
        a, b, c = (test[k]["score"] for k in ("initial", "validation_selected", "terminal"))
        lines.append(f"|{seed}|{a['double_capture_rate']:.1%}|{b['double_capture_rate']:.1%}|{c['double_capture_rate']:.1%}|{a['utility']:.4f}|{b['utility']:.4f}|{c['utility']:.4f}|")
    lines += ["", "全状态特权Actor，固定D移动对手；不进入公共局部观察主表。",
              "按case/public-history分组；world和对手种子是组内重复。没有自动启动公共观察阶段。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    for _, logfile in jobs: logfile.close()
    print("\n".join(lines), flush=True)
    return 0 if complete else 1


if __name__ == "__main__": sys.exit(main())
