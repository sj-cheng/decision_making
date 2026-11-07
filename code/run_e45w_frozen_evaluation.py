"""Sealed A reference and unified post-training A/B/C Actor evaluation.

All inference uses the same Torch policy/environment helpers as training's
validation.  Per-episode caches are keyed by Actor tensors, exact specification,
case bytes and evaluation semantics, never just by a method's display name.
Reference test results remain sealed until all six learning runs complete.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from run_e45w_joint_training import (evaluation_specs, init_evaluation_worker,
    evaluate_frozen_actor, score, tensor_hash, validate_manifest)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "current/e45w_joint_ppo_and_planning_20261103"
MANIFEST = OUT / "experiment.json"
WORKER_STATES = {}


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def environment_fingerprint():
    import stable_baselines3
    files = [ROOT / "code" / name for name in (
        "learning/e45w_joint_env.py", "learning/e45w_joint_ppo.py",
        "run_e45w_joint_training.py", "learning/e45b_joint_network.py",
        "problems/example8.py", "corner_estimator_scenarios.py",
        "run_e45_team_pilot.py", "learning/e45g_ppo.py",
        "solvers/e45_team_adapter.py", "solvers/e45f_support_refinement.py")]
    return digest({"sources": {str(p): sha(p) for p in files},
                   "torch": torch.__version__, "numpy": np.__version__,
                   "sb3": stable_baselines3.__version__, "inference_device": "cpu"})


def init_worker(paths, queue):
    if queue is not None: os.sched_setaffinity(0, {queue.get()})
    init_evaluation_worker(paths)


def evaluate_job(job):
    model, spec, key, semantic_hash = job
    actor_hash = model["actor_tensor_sha256"]
    if actor_hash not in WORKER_STATES:
        if sha(model["actor_path"]) != model["actor_file_sha256"]:
            raise RuntimeError("Frozen Actor file changed before inference")
        state = torch.load(model["actor_path"], map_location="cpu", weights_only=False)
        if tensor_hash(state) != actor_hash: raise RuntimeError("Actor tensor hash differs")
        WORKER_STATES[actor_hash] = state
    row = evaluate_frozen_actor(WORKER_STATES[actor_hash], spec)
    return {"cache_key": key, "actor_tensor_sha256": actor_hash,
            "specification": spec, "spec_sha256": digest(spec),
            "case_sha256": sha(spec["path"]), "semantics_sha256": semantic_hash,
            "episode": row}


def descriptor(name, arm, seed, stage, path):
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    return {"name": name, "arm": arm, "seed": int(seed), "stage": stage,
            "actor_path": str(path), "actor_file_sha256": sha(path),
            "actor_tensor_sha256": tensor_hash(state)}


def reference_models(manifest, smoke):
    result = []
    for seed in manifest["seeds"][:1] if smoke else manifest["seeds"]:
        original = manifest["models"][str(seed)]
        if sha(original["pt"]) != original["pt_sha256"]:
            raise RuntimeError("Registered Single weights changed")
        result.append(descriptor(f"A_{seed}", "A", seed, "frozen", original["pt"]))
    return result


def completed_learning_models(manifest):
    result = []
    for arm in ("B", "C"):
        for seed in manifest["seeds"]:
            directory = OUT / "training" / f"{arm}_{seed}"
            status_path = directory / "status.json"
            status = read(status_path)
            if status.get("status") != "complete" or not status.get("training_budget_audit_passed"):
                raise RuntimeError(f"Final test locked: {arm}_{seed} has not completed its registered training")
            if status["counts"]["team_transitions"] != 976 * 8 * 256:
                raise RuntimeError("Formal training budget differs from registered W budget")
            selection = read(directory / "selection.json")
            terminal = read(directory / "terminal_checkpoint.json")
            for stage, checkpoint in (("selected", Path(selection["checkpoint"])),
                                       ("terminal", Path(terminal["model"]))):
                path = directory / ("selected_actor.pt" if stage == "selected" else "terminal_actor.pt")
                model = descriptor(f"{arm}_{seed}_{stage}", arm, seed, stage, path)
                archived = torch.load(checkpoint.parent / "actor.pt", map_location="cpu", weights_only=False)
                if tensor_hash(archived) != model["actor_tensor_sha256"]:
                    raise RuntimeError("Actor export is not the declared frozen checkpoint")
                model.update(training_status_sha256=sha(status_path),
                             selection_sha256=sha(directory / "selection.json"),
                             full_model_checkpoint=str(checkpoint), full_model_sha256=sha(checkpoint))
                result.append(model)
    return result


def build_specs(manifest, mode, smoke):
    entries = manifest["validation_entries"][:1] if smoke else manifest["validation_entries"]
    by_split = {"validation": evaluation_specs(entries, 1 if smoke else 2, False)
                                    + evaluation_specs(entries, 1, True)}
    # No test is touched by engineering smoke.  Formal reference results are
    # serialized but not printed to console or status until final unlock.
    if not smoke:
        by_split["test"] = evaluation_specs(manifest["test_entries"], 2, False)
        by_split["test"] += evaluation_specs(manifest["test_entries"], 1, True)
    if mode == "final": by_split.pop("validation")
    return by_split


def cache_identity(model, spec, semantic_hash):
    return digest({"actor": model["actor_tensor_sha256"], "spec": spec,
                   "case": sha(spec["path"]), "semantics": semantic_hash})


def load_cache(path, key, model, spec, semantic_hash):
    metadata = read(path.with_suffix(".sha256.json"))
    if sha(path) != metadata["sha256"]: raise RuntimeError("Episode cache bytes changed")
    data = read(path)
    expected = (key, model["actor_tensor_sha256"], digest(spec), sha(spec["path"]), semantic_hash)
    actual = (data["cache_key"], data["actor_tensor_sha256"], data["spec_sha256"],
              data["case_sha256"], data["semantics_sha256"])
    if expected != actual or data["specification"] != spec:
        raise RuntimeError("Episode cache identity mismatch")
    return data


def summaries(rows):
    result = {}
    for mode in (False, True):
        selected = [row for row in rows if row["deterministic"] == mode]
        if not selected: continue
        key = "deterministic_diagnostic" if mode else "stochastic_primary"
        result[key] = score(selected)
        result[key]["by_suite"] = {suite: score([row for row in selected if row["suite"] == suite])
                                    for suite in sorted({row["suite"] for row in selected})}
    return result


def paired_comparisons(models, evaluated):
    # Intervals resample public-history/map groups.  The three registered
    # training seeds remain inside each group; no seed-population CI claim.
    result = []
    by_name = {model["name"]: model for model in models}
    rng = np.random.default_rng(14530261103)
    for arm in ("B", "C"):
        for stage in ("selected", "terminal"):
            all_rows = []
            for name, model in by_name.items():
                if model["arm"] == arm and model["stage"] == stage:
                    rows = evaluated[name]["test"]
                    prior = evaluated[f"A_{model['seed']}"]["test"]
                    lookup = {(r["group"], r["world"], r["opponent"], r["repeat"], r["deterministic"]): r for r in prior}
                    for row in rows:
                        if row["deterministic"]: continue
                        key = (row["group"], row["world"], row["opponent"], row["repeat"], False)
                        baseline = lookup[key]
                        if row["initial_state_sha256"] != baseline["initial_state_sha256"]:
                            raise RuntimeError("Paired evaluation physical states differ")
                        all_rows.append({"group": row["group"], "suite": row["suite"], "opponent": row["opponent"],
                                         "utility_delta": row["utility"] - baseline["utility"],
                                         "double_delta": int(row["double_capture"]) - int(baseline["double_capture"])})
            for suite in sorted({row["suite"] for row in all_rows}):
                for opponent in ("fixed_D_kernel", "rootpuct"):
                    selected = [r for r in all_rows if r["suite"] == suite and r["opponent"] == opponent]
                    groups = sorted({r["group"] for r in selected})
                    delta = np.array([np.mean([r["utility_delta"] for r in selected if r["group"] == group]) for group in groups])
                    dc = np.array([np.mean([r["double_delta"] for r in selected if r["group"] == group]) for group in groups])
                    samples = rng.integers(len(groups), size=(10000, len(groups)))
                    result.append({"contrast": f"{arm}_{stage}-A", "suite": suite, "opponent": opponent,
                        "groups": len(groups), "training_seeds_held_fixed": 3,
                        "utility_difference": float(delta.mean()),
                        "group_bootstrap95_descriptive": np.quantile(delta[samples].mean(axis=1), [.025, .975]).tolist(),
                        "double_capture_rate_difference": float(dc.mean()),
                        "inference_scope": "descriptive uncorrected map-group CI; not seed-population uncertainty or simultaneous confirmatory intervals"})
    return result


def run(args):
    manifest = read(MANIFEST); validate_manifest(manifest)
    for key in ("train_entries", "validation_entries", "test_entries"):
        for entry in manifest[key]:
            if sha(entry["path"]) != entry["sha256"]:
                raise RuntimeError("Registered case bytes changed: " + entry["path"])
    if args.smoke and args.mode == "final": raise ValueError("Final test has no smoke unlock")
    directory = Path(args.output) if args.output else OUT / ("reference_smoke" if args.smoke else f"frozen_{args.mode}")
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / "cache" if args.smoke else OUT / "frozen_actor_episode_cache"
    cache.mkdir(parents=True, exist_ok=True)
    models = reference_models(manifest, args.smoke)
    if args.mode == "final": models += completed_learning_models(manifest)
    splits = build_specs(manifest, args.mode, args.smoke)
    semantic_hash = environment_fingerprint()
    write(directory / "model_manifest.json", {"mode": args.mode, "models": models,
        "semantics_sha256": semantic_hash, "experiment_sha256": sha(MANIFEST),
        "test_scores_sealed": args.mode != "final", "smoke": args.smoke})
    jobs, refs = {}, {}
    for model in models:
        for split, specs in splits.items():
            keys = []
            for spec in specs:
                key = cache_identity(model, spec, semantic_hash); keys.append(key)
                if key not in jobs: jobs[key] = (model, spec, key, semantic_hash)
            refs[(model["name"], split)] = keys
    data = {}; missing = []
    for key, job in jobs.items():
        path = cache / (key + ".json")
        if path.exists(): data[key] = load_cache(path, key, job[0], job[1], semantic_hash)
        else: missing.append(job)
    started = time.time(); new_steps = 0; counts = {"new_episodes": 0, "cache_reused_episodes": len(data),
            "unique_episode_tasks": len(jobs), "model_episode_cells": sum(len(v) for v in refs.values()),
            "new_physical_transitions": 0, "optimizer_updates": 0}
    def publish(status, **extra):
        write(directory / "status.json", {"status": status, "mode": args.mode, "pid": os.getpid(),
            "elapsed_seconds": time.time() - started, "counts": counts,
            "test_scores_sealed": args.mode != "final", "all_six_training_complete_required_for_final": True,
            "smoke": args.smoke, **extra})
    publish("running")
    try:
        cpus = [int(v) for v in args.cpus.split(",") if v]
        if cpus: os.sched_setaffinity(0, set(cpus))
        context = mp.get_context("spawn"); queue = context.Queue() if cpus else None
        workers = min(args.workers, len(cpus)) if cpus else args.workers
        if queue is not None:
            for cpu in cpus[:workers]: queue.put(cpu)
        paths = [entry["path"] for entry in manifest["validation_entries"]]
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=init_worker,
                                 initargs=(paths, queue)) as pool:
            futures = {pool.submit(evaluate_job, job): job for job in missing}
            for future in as_completed(futures):
                job = futures[future]; row = future.result(); key = job[2]
                path = cache / (key + ".json"); write(path, row)
                write(path.with_suffix(".sha256.json"), {"sha256": sha(path)})
                data[key] = row
                counts["new_episodes"] += 1; counts["new_physical_transitions"] += row["episode"]["decisions"]
                if counts["new_episodes"] % 4 == 0: publish("running")
        if queue is not None: queue.close()
        evaluated = {model["name"]: {} for model in models}
        for (name, split), keys in refs.items(): evaluated[name][split] = [data[key]["episode"] for key in keys]
        results = {model["name"]: {split: summaries(rows) for split, rows in evaluated[model["name"]].items()} for model in models}
        write(directory / ("sealed_results.json" if args.mode == "reference" else "results.json"),
              {"models": models, "summaries": results, "episode_cache_keys": {f"{name}/{split}": keys for (name, split), keys in refs.items()},
               "counts": counts, "test_scores_sealed": args.mode != "final"})
        if args.mode == "final":
            comparisons = paired_comparisons(models, evaluated)
            write(directory / "paired_group_statistics.json", {"comparisons": comparisons})
            lines = ["# E45-W unified final Actor evaluation", "", "Primary mode: stochastic. Deterministic scores are diagnostics.", "",
                     "|Model|Stochastic double capture|Raw utility|", "|---|---:|---:|"]
            for model in models:
                s = results[model["name"]]["test"]["stochastic_primary"]
                lines.append(f"|{model['name']}|{s['double_captures']}/{s['episodes']}|{s['utility']:.6f}|")
            lines += ["", "Map groups are independent units; worlds/action repeats are within-group repetitions.",
                      "Grouped intervals hold the three training seeds fixed; they do not quantify a seed population."]
            (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
        for model in models:
            if sha(model["actor_path"]) != model["actor_file_sha256"]:
                raise RuntimeError("Actor changed during frozen evaluation")
        validate_manifest(manifest)
        publish("complete", complete=True)
        print(f"{args.mode} complete: {len(models)} models, {counts['new_episodes']} new episodes, "
              f"{counts['new_physical_transitions']} physical steps, zero optimizer updates; "
              f"test scores {'sealed' if args.mode != 'final' else 'unlocked'}", flush=True)
    except BaseException as exc:
        write(directory / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        publish("failed", error=repr(exc)); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("reference", "final"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpus", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.workers < 1: parser.error("workers must be positive")
    run(args)
