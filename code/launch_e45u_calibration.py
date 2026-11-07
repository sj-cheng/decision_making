"""Run two native official recipes under tmux:train; summarize on completion."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "current/e45u_official_hasac_calibration_20261102"
PYTHON = OUT / "venv/bin/python"


def read(path):
    try: return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError): return None


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def summarize(records, started):
    complete = all(r["exit_code"] == 0 and r["status"] and r["status"]["status"] == "complete"
                   for r in records)
    decision = "engineering_or_training_failure" if not complete else "native_learning_inconclusive"
    tests = {}
    if complete:
        tests = {str(r["seed"]): read(OUT / r["name"] / "sealed_test.json") for r in records}
        selected = all(t["validation_selected_vs_initial"]["normal_approx_95ci"][0] > 0 for t in tests.values())
        terminal = all(t["terminal_vs_initial"]["normal_approx_95ci"][0] > 0 for t in tests.values())
        if selected and terminal: decision = "native_learning_positive_two_seeds"
        elif selected: decision = "learned_but_terminal_stability_insufficient"
    result = {"complete": complete, "decision": decision, "elapsed_seconds": time.time() - started,
              "runs": records, "tests": tests,
              "scope": "native continuous MPE calibration only; not pursuit baseline qualification",
              "automatic_next_training": False}
    write(OUT / "completion_audit.json", result)
    lines = ["# E45-U 官方原生任务校准结果", "", f"状态：{decision}", "",
             "|种子|初始测试回报|验证选模测试回报|终点测试回报|所选步数|",
             "|---|---:|---:|---:|---:|"]
    for seed, t in tests.items():
        lines.append(f"|{seed}|{t['initial']['mean']:.3f}|{t['validation_selected']['mean']:.3f}|{t['terminal']['mean']:.3f}|{t['selected_step']}|")
    lines += ["", "完整曲线、选模、配对测试差和实际单位见各种子目录。",
              "两种子仅用于工程阳性校准，不代表跨训练种子总体置信结论。未自动启动追逃任务。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    audit = read(OUT / "smoke_audit.json")
    assert audit and audit["passed"], "Cannot launch before exact smoke comparison passes"
    assert not (OUT / "launch_manifest.json").exists(), "Refuse duplicate formal launch"
    started = time.time()
    jobs = []
    records = []
    for seed, gpu, cpus in ((1, 1, list(range(0,28)) + list(range(56,84))),
                            (2, 2, list(range(28,56)) + list(range(84,112)))):
        name = f"seed_{seed}"
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", PYTHONHASHSEED=str(seed), PYTHONUNBUFFERED="1",
                   SDL_VIDEODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1",
                   PYTHONWARNINGS="ignore::DeprecationWarning,ignore::UserWarning")
        command = ["taskset", "-c", ",".join(map(str, cpus)), str(PYTHON),
                   str(ROOT / "code/run_e45u_official_calibration.py"), "--seed", str(seed), "--name", name]
        logfile = (OUT / "logs" / f"{name}.log").open("w")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=logfile, stderr=subprocess.STDOUT)
        jobs.append((process, logfile))
        records.append({"seed": seed, "gpu": gpu, "cpus": cpus, "name": name,
                        "pid": process.pid, "command": command, "exit_code": None, "status": None})
    write(OUT / "launch_manifest.json", {"started_unix": started,
          "started_asia_shanghai": datetime.datetime.fromtimestamp(started, datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
          "tmux_session": "train", "tmux_window": "U-calibration", "runs": records,
          "nominal_team_transitions_including_warmup": 20020000,
          "nominal_train_calls": 1000000})
    try:
        while True:
            for record, (process, logfile) in zip(records, jobs):
                record["exit_code"] = process.poll()
                record["status"] = read(OUT / record["name"] / "status.json")
            done = all(r["exit_code"] is not None for r in records)
            write(OUT / "progress.json", {"state": "finished" if done else "running",
                  "elapsed_seconds": time.time() - started, "runs": records})
            if done: break
            time.sleep(5)
        summarize(records, started)
    finally:
        for _, logfile in jobs: logfile.close()
    print((OUT / "REPORT.md").read_text(), flush=True)
    return 0 if all(r["exit_code"] == 0 for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
