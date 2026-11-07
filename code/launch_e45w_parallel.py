"""Gate and orchestrate the six authorized E45-W experiments in tmux:train.

Only counters, errors and training validation statuses are read while model
selection remains open. Planning and A test scores remain sealed until all
six learners have completed and their selected/terminal Actors are frozen.
"""
from __future__ import annotations
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "current/e45w_joint_ppo_and_planning_20261103"
PYTHON = OUT / "venv/bin/python"
TARGET_TEAMS = 976 * 8 * 256


def read(path):
    try: return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError): return None


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)); tmp.replace(path)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def beijing_now(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def integrity_check():
    path = OUT / "component_source_manifest.json"
    manifest = read(path)
    if not manifest or not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise RuntimeError("Nonempty component_source_manifest.json files mapping is required")
    expected = manifest.get("experiment_sha256")
    if not expected or sha(OUT / "experiment.json") != expected:
        raise RuntimeError("Frozen experiment.json hash changed")
    for name, digest in manifest["files"].items():
        file = Path(name) if Path(name).is_absolute() else ROOT / name
        if isinstance(digest, dict): digest = digest.get("sha256")
        if not digest or sha(file) != digest:
            raise RuntimeError("Frozen component source changed: " + str(file))
    return {"component_manifest_sha256": sha(path), "experiment_sha256": expected,
            "source_files_verified": len(manifest["files"])}


def gated():
    acceptance = read(OUT / "engineering_acceptance.json")
    if not acceptance or acceptance.get("passed") is not True:
        raise RuntimeError("Engineering acceptance has not passed; formal jobs will not launch")
    return integrity_check() | {"engineering_acceptance_sha256": sha(OUT / "engineering_acceptance.json")}


def train_cpus(start): return list(range(start, start + 4)) + list(range(start + 56, start + 60))


def definitions():
    manifest = str(OUT / "experiment.json")
    jobs = [
        {"name": "planning_wall_v2", "kind": "planning", "experiments": [1, 4],
         "cpus": list(range(16)), "status_path": str(OUT / "planning_wall_v2/status.json"),
         "audit_path": str(OUT / "planning_wall_v2/completion_audit.json"),
         "arguments": ["code/run_e45w_planning_evaluation.py", "--mode", "wall", "--manifest", manifest,
                       "--workers", "16", "--cpus", ",".join(map(str, range(16))),
                       "--output", str(OUT / "planning_wall_v2"), "--calibration", str(OUT / "planning_calibration_v2")]},
        {"name": "planning_sim_v2", "kind": "planning", "experiments": [3],
         "cpus": list(range(16, 28)), "status_path": str(OUT / "planning_sim_v2/status.json"),
         "audit_path": str(OUT / "planning_sim_v2/completion_audit.json"),
         "arguments": ["code/run_e45w_planning_evaluation.py", "--mode", "sim", "--manifest", manifest,
                       "--workers", "12", "--cpus", ",".join(map(str, range(16, 28))),
                       "--output", str(OUT / "planning_sim_v2")]},
        {"name": "frozen_reference", "kind": "reference", "experiments": [2],
         "cpus": [52, 53, 54, 55, 108, 109, 110, 111],
         "status_path": str(OUT / "frozen_reference/status.json"),
         "arguments": ["code/run_e45w_frozen_evaluation.py", "--mode", "reference", "--workers", "8",
                       "--cpus", "52,53,54,55,108,109,110,111", "--output", str(OUT / "frozen_reference")]},
    ]
    index = 0
    for seed, gpu in ((17, 1), (29, 2), (43, 3)):
        for arm in ("B", "C"):
            cpus = train_cpus(28 + index * 4); index += 1
            name = f"{arm}_{seed}"
            jobs.append({"name": name, "kind": "training", "experiments": [6], "arm": arm,
                "seed": seed, "gpu": gpu, "cpus": cpus,
                "status_path": str(OUT / "training" / name / "status.json"),
                "arguments": ["code/run_e45w_joint_training.py", "--arm", arm, "--seed", str(seed),
                              "--device", f"cuda:{gpu}", "--cpus", ",".join(map(str, cpus)), "--eval-workers", "2"]})
    return jobs


def final_definition():
    cpus = list(range(40))
    return {"name": "final_evaluation", "kind": "final", "experiments": [2, 6], "cpus": cpus,
            "status_path": str(OUT / "final_evaluation/status.json"),
            "arguments": ["code/run_e45w_frozen_evaluation.py", "--mode", "final", "--workers", "40",
                          "--cpus", ",".join(map(str, cpus)), "--output", str(OUT / "final_evaluation")]}


def start(job, handles):
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)  # --device keeps physical GPU numbering; other GPUs remain unused.
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
               PYTHONUNBUFFERED="1", MPLBACKEND="Agg",
               PYTHONPATH=str(ROOT / "code") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""))
    if "seed" in job: env["PYTHONHASHSEED"] = str(job["seed"])
    command = ["taskset", "-c", ",".join(map(str, job["cpus"])), str(PYTHON), *job["arguments"]]
    logfile = OUT / "logs" / (job["name"] + ".log"); logfile.parent.mkdir(parents=True, exist_ok=True)
    stream = logfile.open("w")
    try: process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    except BaseException:
        stream.close(); raise
    handles[job["name"]] = (process, stream)
    job.update(command=command, pid=process.pid, state="running", exit_code=None,
               started_beijing=beijing_now(), logfile=str(logfile), status=None)
    print(f"Launched {job['name']} pid={process.pid}", flush=True)


def safe_status(job):
    status = read(job["status_path"])
    if status is None: return None
    if job["kind"] != "planning": return status  # validation scores are permitted; reference status never exposes sealed scores.
    return {k: status[k] for k in ("stage", "completed", "planned", "workers", "invalid", "timestamp", "optimizer_updates") if k in status}


def poll(job, handles):
    if job["name"] not in handles: return
    process, _ = handles[job["name"]]
    job["status"] = safe_status(job)
    code = process.poll(); job["exit_code"] = code
    if code is None: return
    if job.get("finished_beijing"): return
    job["finished_beijing"] = beijing_now()
    valid = code == 0
    if job["kind"] == "planning":
        audit = read(job["audit_path"])
        valid = valid and bool(audit and audit.get("complete") and audit.get("source_sha256_unchanged") and audit.get("manifest_sha256_unchanged"))
    else:
        status = job.get("status") or {}
        valid = valid and status.get("status") == "complete"
        if job["kind"] == "training":
            valid = valid and status.get("training_budget_audit_passed") is True
            valid = valid and status.get("counts", {}).get("team_transitions") == TARGET_TEAMS
    job["state"] = "complete" if valid else "failed"
    if not valid:
        failure = {"job": job["name"], "exit_code": code, "status": job.get("status"),
                   "reason": "process failed or completion/budget/source audit not satisfied",
                   "logfile": job["logfile"], "timestamp_beijing": beijing_now()}
        failure_path = OUT / "coordinator_failures" / job["name"] / "failure.json"
        write(failure_path, failure); job["coordinator_failure"] = str(failure_path)
    print(f"Finished {job['name']}: {job['state']} exit={code}", flush=True)


def descendants(pid):
    """Current same-user descendant PIDs without requiring a new dependency."""
    parents = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit(): continue
        try:
            rest = (path / "stat").read_text().rsplit(")", 1)[1].split()
            parents[int(path.name)] = int(rest[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError): pass
    result, pending = {pid}, [pid]
    while pending:
        parent = pending.pop()
        children = [child for child, pp in parents.items() if pp == parent and child not in result]
        result.update(children); pending.extend(children)
    return sorted(result)


def expand_affinity(training):
    records = []
    for index, job in enumerate(training):
        if job["state"] != "running": continue
        cpus = set(range(index * 8, index * 8 + 8)); changed, errors = [], []
        for pid in descendants(job["pid"]):
            try: tids = [int(p.name) for p in (Path("/proc") / str(pid) / "task").iterdir()]
            except (FileNotFoundError, ProcessLookupError, PermissionError): continue
            for tid in tids:
                try: os.sched_setaffinity(tid, cpus); changed.append(tid)
                except (FileNotFoundError, ProcessLookupError): pass
                except (PermissionError, OSError) as exc: errors.append({"tid": tid, "error": repr(exc)})
        records.append({"job": job["name"], "cpus": sorted(cpus), "thread_ids_adjusted": changed, "errors": errors})
        job["expanded_cpus"] = sorted(cpus)
    write(OUT / "affinity_expansion.json", {"timestamp_beijing": beijing_now(), "records": records,
          "scope": "OS affinity only after both planning pools exited; rollout counts, policies, updates and budgets unchanged"})


def unlock(training):
    files = {}
    for job in training:
        directory = OUT / "training" / job["name"]
        for name in ("status.json", "selection.json", "terminal_checkpoint.json", "initial_actor.pt", "selected_actor.pt", "terminal_actor.pt"):
            path = directory / name
            if not path.exists(): raise RuntimeError("Missing frozen learning artifact: " + str(path))
            files[str(path)] = sha(path)
    write(OUT / "test_scores_unlock.json", {"unlocked": True, "timestamp_beijing": beijing_now(),
          "all_six_training_complete": True, "frozen_training_artifacts": files,
          "scope": "Only after all six learning runs and validation model choices are complete may planning/A test scores be examined"})


def experiment_states(jobs, final, acceptance):
    by_name = {j["name"]: j for j in jobs}
    learners = [j for j in jobs if j["kind"] == "training"]
    return {
        "1_wall_time_curve": by_name["planning_wall_v2"]["state"],
        "2_frozen_single_A": by_name["frozen_reference"]["state"],
        "3_equal_simulations": by_name["planning_sim_v2"]["state"],
        "4_cost_accounting": "attached_to_experiment_1:" + by_name["planning_wall_v2"]["state"],
        "5_probability_execution_acceptance": "passed" if acceptance else "not_passed",
        "6_joint_PPO": {j["name"]: j["state"] for j in learners},
        "unified_final_actor_evaluation": final["state"],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "coordinator.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / "launch_manifest.json").exists(): raise RuntimeError("Refuse duplicate formal launch")
    initial_integrity = gated()
    jobs = definitions(); final = final_definition() | {"state": "waiting_for_frozen_training_reference_and_planning_exit"}
    handles = {}; started = time.time(); expanded = False; scores_unlocked = False
    write(OUT / "launch_manifest.json", {"started_beijing": beijing_now(), "started_unix": started,
          "tmux_session": "train", "tmux_window": "W-coordinator", "gate": initial_integrity,
          "jobs": jobs, "final_dependency": final, "no_automatic_budget_extension": True,
          "planned_learning_team_transitions": TARGET_TEAMS * 6})
    try:
        for job in jobs:
            try: start(job, handles)
            except BaseException as exc:
                job.update(state="failed", exit_code=None, launch_error=repr(exc), finished_beijing=beijing_now())
                failure_path = OUT / "coordinator_failures" / job["name"] / "failure.json"
                write(failure_path, {"launch_error": repr(exc), "job": job["name"]})
                job["coordinator_failure"] = str(failure_path)
        while True:
            for job in jobs: poll(job, handles)
            poll(final, handles)
            training = [job for job in jobs if job["kind"] == "training"]
            planning = [job for job in jobs if job["kind"] == "planning"]
            reference = next(job for job in jobs if job["kind"] == "reference")
            planning_exited = all(job["state"] in ("complete", "failed") for job in planning)
            training_done = all(job["state"] == "complete" for job in training)
            initial_exited = all(job["state"] in ("complete", "failed") for job in jobs)
            if planning_exited and not expanded:
                expand_affinity(training); expanded = True
            if training_done and not scores_unlocked:
                integrity_check(); unlock(training); scores_unlocked = True
            if initial_exited and final["state"].startswith("waiting"):
                if training_done and reference["state"] == "complete":
                    integrity_check(); start(final, handles)
                else:
                    final.update(state="blocked", reason="Requires all six complete training runs and complete A reference")
                    write(OUT / "coordinator_failures/final_dependency/failure.json", final)
            state = "running"
            if initial_exited and final["state"] in ("complete", "failed", "blocked"): state = "finished"
            write(OUT / "progress.json", {"state": state, "elapsed_seconds": time.time() - started,
                  "coordinator_pid": os.getpid(), "timestamp_beijing": beijing_now(),
                  "test_scores_unlocked": scores_unlocked, "experiments": experiment_states(jobs, final, True),
                  "jobs": jobs, "final_evaluation": final})
            if state == "finished": break
            time.sleep(10)
        final_integrity = integrity_check()
        all_complete = all(job["state"] == "complete" for job in jobs) and final["state"] == "complete"
        write(OUT / "completion_audit.json", {"complete": all_complete, "jobs": jobs, "final_evaluation": final,
              "wall_seconds": time.time() - started, "initial_integrity": initial_integrity,
              "final_integrity": final_integrity, "test_scores_unlocked": scores_unlocked,
              "no_automatic_extension": True})
        lines = ["# E45-W parallel experiment overview", "", f"All experiments complete: {all_complete}", "",
                 "|Job|Status|Exit code|", "|---|---|---:|"]
        for job in jobs + [final]: lines.append(f"|{job['name']}|{job['state']}|{job.get('exit_code')}|")
        lines += ["", "Scientific reports remain separate; no cross-protocol ranking is synthesized here.",
                  "Planning: planning_wall_v2/REPORT.md and planning_sim_v2/REPORT.md.",
                  "Unified A/B/C evaluation: final_evaluation/REPORT.md.",
                  "Cost fields: policy_inference_ms is network-only; planning latency is complete act. Do not equate them."]
        (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
        return 0 if all_complete else 1
    except BaseException as exc:
        write(OUT / "coordinator_failure.json", {"error": repr(exc), "traceback": traceback.format_exc(),
              "jobs": jobs, "final_evaluation": final, "timestamp_beijing": beijing_now()})
        raise
    finally:
        for _, stream in handles.values(): stream.close()
        lock.close()


if __name__ == "__main__": sys.exit(main())
