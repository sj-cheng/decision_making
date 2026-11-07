"""Offline estimator checks on fixed historical G observation histories.

Estimator helpers accept public packets/observations only. Simulator truth is
read by separate scoring code after all corresponding estimates are fixed.
No controller actions, real games, teacher labels or training are run here.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from audit_l45_rollout_reliability import sha, save, observation
from solvers.local_pursuit import Controller as HistoricalController, PublicMap
from solvers.reliable_local_pursuit_v2 import Controller, VERSION, SimulationReliabilityError
from solvers.history_recovery import lattice_paths
from verify_l45_occlusion_scenarios import movement, visible


ROOT = Path(__file__).resolve().parent.parent
BANK = ROOT/"current/l45_occlusion_scenarios_20261025/bank"
EPISODES = ROOT/"current/l45_occluded_comparison_20261025/episodes"
DEFAULT = ROOT/"current/l45_estimator_recovery_20261025"


def replay_estimates(packet, observations, seed, repaired):
    """Public data only; no task world identity or environment state."""
    p = PublicMap(packet["map"], packet["motion"]["pursuer_axis_limit"])
    own = (Controller if repaired else HistoricalController)(p, "G", seed, deadline_ms=0.)
    outputs = []
    for item in packet["history"][:-1]:
        own.update(observation(item), np.zeros(6, np.int64))
    for obs in observations:
        count = np.zeros(6, np.int64)
        started = time.perf_counter()
        try:
            own.update(obs, count)
            outputs.append(dict(particles=own.particles.copy(), failed=False,
                filter_rng=own.filter_rng.copy(), elapsed_ms=1000*(time.perf_counter()-started),
                lattice_attempts=int(count[3]) if repaired else 0))
        except (RuntimeError, ValueError) as exc:
            outputs.append(dict(failed=True, error=str(exc)))
            break
    return outputs


def public_root_support(packet, seed):
    last = None
    empty = []
    for item in packet["history"]:
        if item["target"] is not None:
            last = np.array(item["target"])
            empty = []
        else:
            empty.append(item["position"])
    # 42^2 is at least the number of anchored .5m grid points. Sampling at
    # least every feasible endpoint ensures the diagnostic includes all such
    # endpoints; it is not used as the controller particle count.
    endpoints, paths, n = lattice_paths(last, np.array(empty), np.array(packet["map"]),
        .5, np.array([seed], np.int64), np.zeros(6, np.int64), number=42*42)
    return endpoints[:n].copy(), paths[:n].copy()


def run(output):
    if (output/"coverage_results.json").exists():
        raise RuntimeError("Coverage audit already completed; use a new --output")
    files = sorted(EPISODES.glob("*_G.pt"))
    if len(files) != 96:
        raise RuntimeError("Expected all 96 frozen historical G requests")
    source_paths = [Path(__file__), ROOT/"code/solvers/reliable_local_pursuit_v2.py",
        ROOT/"code/solvers/history_recovery.py", ROOT/"code/solvers/local_pursuit.py"]+files
    source_paths += sorted((BANK/"controller_inputs").glob("*.json"))
    frozen = {str(p): sha(p) for p in source_paths}
    save(output/"coverage_experiment.json", dict(version=VERSION, records=96, history_source="historical G only",
        estimators=["historical_particles", "v2_recovery_particles"],
        diagnostic_support="all feasible anchored .5m lattice endpoints at initial handoff; not controller particles",
        no_new_games=True, no_actions=True, no_teacher_queries=True, no_training=True, sources=frozen))
    row_results, frames = [], []
    cached_support = {}
    for f in files:
        saved = torch.load(f, map_location="cpu", weights_only=False)
        task, trajectory = saved["task"], saved["trajectory"]
        packet = json.loads((BANK/task["entry"]["public"]).read_text())
        # Extract the public observation sequence before invoking either
        # estimator. Neither function sees trajectory states or world IDs.
        public_observations = [observation(dict(step=r["observation"].step,
            position=r["observation"].position.tolist(), target=None if r["observation"].target is None
            else r["observation"].target.tolist(), remaining=r["observation"].remaining)) for r in trajectory]
        seed = packet["controller_seeds"][task["stream"]]
        estimates = {"historical": replay_estimates(packet, public_observations, seed, False),
                     "repaired": replay_estimates(packet, public_observations, seed, True)}
        support_key = task["entry"]["id"]
        if support_key not in cached_support:
            cached_support[support_key] = public_root_support(packet, packet["controller_seeds"][0])
        endpoints, paths = cached_support[support_key]
        # Evaluator starts here: truth affects only diagnostic measurements.
        truth = np.asarray(trajectory[0]["state"]).reshape(-1)[:2]
        nearest = int(np.argmin(np.linalg.norm(endpoints-truth, axis=1))) if len(endpoints) else None
        support_error = float(np.linalg.norm(endpoints[nearest]-truth)) if nearest is not None else None
        witness_passed = nearest is not None
        if nearest is not None:
            public_history = [x for x in packet["history"] if x["step"] > 2]
            path = paths[nearest]
            for t, item in enumerate(public_history):
                witness_passed = witness_passed and movement(path[t].tolist(), path[t+1].tolist(), 1., packet["map"])
                witness_passed = witness_passed and not visible(item["position"], path[t+1].tolist(), packet["map"])
        record = dict(key=task["key"], history=task["entry"]["id"], frames=len(trajectory),
                      original_execution_failure=saved.get("failure") is not None,
                      handoff_lattice_nearest_error=support_error, witness_passed=bool(witness_passed), estimators={})
        for method, outputs in estimates.items():
            errors, hidden_errors, exact_matches, lattice_calls = [], [], 0, 0
            failure = next((x["error"] for x in outputs if x["failed"]), None)
            for i, estimate in enumerate(outputs):
                if estimate["failed"]:
                    break
                truth = np.asarray(trajectory[i]["state"]).reshape(-1)[:2]
                error = float(np.min(np.linalg.norm(estimate["particles"]-truth, axis=1)))
                hidden = public_observations[i].target is None
                if method == "historical":
                    exact_matches += int(np.array_equal(estimate["particles"], trajectory[i]["particles"])
                        and np.array_equal(estimate["filter_rng"], trajectory[i]["filter_rng"]))
                errors.append(error)
                if hidden:
                    hidden_errors.append(error)
                lattice_calls += estimate["lattice_attempts"]
                frames.append(dict(key=task["key"], method=method, frame=i, hidden=hidden,
                    nearest_error=error, covered_1m=bool(error <= 1.), latency_ms=estimate["elapsed_ms"]))
            record["estimators"][method] = dict(frames_replayed=len(errors), failure=failure,
                handoff_nearest_error=errors[0] if errors else None,
                handoff_covered_1m=bool(errors and errors[0] <= 1.), hidden_frames=len(hidden_errors),
                hidden_covered_1m=sum(e <= 1. for e in hidden_errors),
                historical_exact_matches=exact_matches if method == "historical" else None,
                lattice_recovery_attempts=lattice_calls)
        row_results.append(record)
        if len(row_results) % 16 == 0:
            print(f'Fixed historical estimator replay {len(row_results)}/96', flush=True)
    unchanged = all(sha(Path(p)) == h for p, h in frozen.items())
    summary = dict(version=VERSION, records=96, methods={}, sources_unchanged=unchanged,
        original_execution_failures=sum(x["original_execution_failure"] for x in row_results),
        lattice_handoff_covered_1m=sum(x["handoff_lattice_nearest_error"] is not None and x["handoff_lattice_nearest_error"] <= 1. for x in row_results),
        lattice_witness_checks_passed=all(x["witness_passed"] for x in row_results),
        no_new_games=True, no_new_actions=True, no_training=True,
        limitations="historical G-generated histories; all comparisons use the same histories; finite endpoint proximity is not continuous support or probability calibration")
    for method in ("historical", "repaired"):
        rs = [x["estimators"][method] for x in row_results]
        fs = [x for x in frames if x["method"] == method and x["hidden"]]
        summary["methods"][method] = dict(handoff_covered_1m=sum(x["handoff_covered_1m"] for x in rs),
            hidden_frames=len(fs), hidden_covered_1m=sum(x["covered_1m"] for x in fs),
            replay_failures=sum(x["failure"] is not None for x in rs),
            historical_exact_matches=sum(x["historical_exact_matches"] or 0 for x in rs),
            frames_replayed=sum(x["frames_replayed"] for x in rs),
            lattice_recovery_attempts=sum(x["lattice_recovery_attempts"] for x in rs))
    save(output/"coverage_rows.json", row_results)
    save(output/"coverage_frames.json", frames)
    save(output/"coverage_results.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    print(json.dumps(run(parser.parse_args().output.resolve()), ensure_ascii=False, indent=2))
