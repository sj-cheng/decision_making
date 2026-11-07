"""Read-only E45-W coordinator display; never opens sealed scientific scores."""
import datetime
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "current/e45w_joint_ppo_and_planning_20261103"


def read(path):
    try: return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError): return None


def main():
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    print("E45-W | six authorized experiments | tmux:train/W-coordinator")
    print(now.isoformat(timespec="seconds"))
    progress = read(OUT / "progress.json")
    if progress is None:
        gate = read(OUT / "engineering_acceptance.json")
        print("Formal coordinator not yet publishing progress. Engineering gate:", bool(gate and gate.get("passed")))
        return
    print(f"Coordinator: {progress['state']} | wall {progress['elapsed_seconds']/3600:.2f} h")
    experiments = progress["experiments"]
    for name in ("1_wall_time_curve", "2_frozen_single_A", "3_equal_simulations", "4_cost_accounting", "5_probability_execution_acceptance"):
        print(f"{name}: {experiments[name]}")
    print("\nExperiment 6: independent B/C learners; fixed per-model budget 1,998,848 team transitions")
    for job in progress["jobs"]:
        if job["kind"] != "training": continue
        status = job.get("status") or {}; counts = status.get("counts", {})
        steps = counts.get("team_transitions", 0); target = status.get("target_team_transitions", 1998848)
        elapsed = status.get("elapsed_seconds", 0.)
        message = f" {job['name']}: {job['state']} | {steps:,}/{target:,} ({100*steps/max(1,target):.1f}%)"
        message += f" | Adam {counts.get('optimizer_steps',0):,} | GPU {job['gpu']}"
        if steps >= 2048 and elapsed > 0 and job["state"] == "running":
            rate = steps / elapsed; remaining = max(0, target - steps) / rate
            message += f" | observed {rate:.0f} team/s, linear ETA {remaining/3600:.1f} h"
        print(message)
        latest = status.get("latest_validation")
        if latest:
            row = latest["stochastic"]
            print(f"   public validation @ {latest['team_transitions']:,}: double {row['double_captures']}/{row['episodes']}, utility {row['utility']:.4f}")
        if job.get("launch_error") or status.get("error"):
            print("   ERROR:", job.get("launch_error") or status.get("error"))
    print("\nPlanning/reference counters (test efficacy remains sealed until all six learners freeze):")
    for job in progress["jobs"]:
        if job["kind"] == "training": continue
        status = job.get("status") or {}; counts = status.get("counts", {})
        if job["kind"] == "planning":
            line = f" {job['name']}: {job['state']} {status.get('completed',0)}/{status.get('planned','?')}"
            if not status:
                calibration = read(OUT / "planning_calibration_v2/attempts.json")
                if calibration: line += f" | timing calibration candidates tested={len(calibration.get('attempts',[]))}"
                else: line += " | timing calibration/worker initialization"
        else:
            line = f" {job['name']}: {job['state']} | new episodes={counts.get('new_episodes',0)}, cache hits={counts.get('cache_hits',0)}"
        print(line)
    print("Test scores unlocked:", progress["test_scores_unlocked"])
    print("Unified final evaluation:", progress["final_evaluation"]["state"])
    print("Cost note: network inference and complete controller act latency are different measures.")
    print("ETA is an early average-rate extrapolation, not a completion guarantee.")


if __name__ == "__main__": main()
