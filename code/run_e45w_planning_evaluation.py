"""E45-W frozen planning evaluation: wall time, fixed simulations, and costs.

No model is trained. The original physical episode is reused with process-local
controller/support adapters. Wall-time concurrency is selected on validation
prefixes by throughput, never by capture outcomes.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as futures
import ctypes
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time
import traceback

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import run_e45b_role_free as base
import run_e45_team_pilot as pilot
from e45c_capture_audit import audit_simulated
from solvers.e45b_joint_nte import Controller
from solvers.e45f_baselines import GeometryController

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "current/e45w_joint_ppo_and_planning_20261103"
MODEL_ROOT = ROOT / "current/e45c_budget_scaling_20261028/learning/x20/round_12/models"
WORKER_OUT = None
WORKER_CFG = None
WALL_BUDGETS = (50, 100, 200)
SIMULATIONS = (16, 64, 128)
SEEDS = (17, 29, 43)
ARMS = (("SINGLE_PRIOR", 17), ("SINGLE_PRIOR", 29), ("SINGLE_PRIOR", 43),
        ("GEOM_SEARCH", 0), ("U", 0))
KNOWN_ERRORS = ("Empty weighted public-history support", "Empty team-conditioned finite-grid support",
                "Candidate capacity exceeded", "More important regions than the frozen sample budget")


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entries(config, split):
    explicit = config.get("test_entries" if split == "test" else "validation_entries")
    if explicit is not None:
        return explicit
    return [row for row in config.get("entries", []) if row.get("split") == split]


def manifest(path):
    config = read(path)
    for row in entries(config, "test") + entries(config, "validation"):
        digest = row.get("sha256")
        if digest and sha(row["path"]) != digest:
            raise RuntimeError("Frozen case changed: " + row["path"])
    for seed in SEEDS:
        if not (MODEL_ROOT / f"single_{seed}.npz").exists():
            raise FileNotFoundError(MODEL_ROOT / f"single_{seed}.npz")
    return config


def request(entry, world, opponent, arm, train_seed, mode, value, stage, prefix=False):
    # Neither method nor budget changes the environment/controller seed.
    seed = int(np.random.SeedSequence([14530261103, int(sha(entry["path"])[:8], 16),
                                      int(world == "BA")]).generate_state(1)[0])
    key = f"{entry['id']}_{world}_{opponent}_{arm}{train_seed}_{mode}{value}_{stage}"
    row = dict(key=key, case=entry["path"], group=entry["id"], family=entry["family"],
               pattern=entry.get("pattern", "unknown"), split=entry.get("split", "test"),
               suite=entry.get("suite", "ID"), world=world, opponent=opponent,
               method=f"{arm}{train_seed}" if arm == "SINGLE_PRIOR" else arm,
               arm=arm, train_seed=train_seed, seed=seed, stage=stage, round=0,
               model_round=12, mode=mode, control_value=int(value),
               budget_ms=float(value if mode == "wall" else 200),
               budget_enforced=(mode == "wall"),
               latency_reference_threshold_ms=float(value if mode == "wall" else 200),
               checkpoint=str(MODEL_ROOT / f"single_{train_seed}.npz") if arm == "SINGLE_PRIOR" else None)
    if mode == "sim":
        row["fixed"] = int(value)
    if prefix:
        row["limit"] = 2
    return row


def factory(case, task):
    arm = task["arm"]
    if arm == "GEOM_SEARCH":
        own = GeometryController(case["map"], case["speed"], task["seed"],
                                 use_policy=False, use_value=False, budget_ms=task["budget_ms"])
    elif arm == "U":
        own = Controller(case["map"], case["speed"], task["seed"],
                         use_policy=False, use_value=False, budget_ms=task["budget_ms"])
    elif arm == "SINGLE_PRIOR":
        own = Controller(case["map"], case["speed"], task["seed"], task["checkpoint"],
                         use_policy=True, use_value=False, budget_ms=task["budget_ms"])
    else:
        raise ValueError(arm)
    if not task.get("_timing_components", True):
        return own  # engineering-only numerical reference, no timer wrapping
    nested = collections.defaultdict(float)
    nested_calls = collections.defaultdict(int)
    def timed(name, function):
        def call(*args, **kwargs):
            started = time.perf_counter()
            try: return function(*args, **kwargs)
            finally:
                nested[name] += 1000 * (time.perf_counter() - started)
                nested_calls[name] += 1
        call.__name__ = function.__name__  # preserve author operation-estimate keys
        return call
    for name in ("observe", "features", "parameters", "propose", "trajectory"):
        setattr(own, name, timed(name, getattr(own, name)))
    original_act = own.act
    def act(obs, fixed_simulations=None):
        nested.clear(); nested_calls.clear()
        action = original_act(obs, fixed_simulations)
        own.last_stats["nested_method_wall_totals_ms"] = dict(nested)
        own.last_stats["nested_method_calls"] = dict(nested_calls)
        own.last_stats["callback_max_duration_estimates_ms"] = {str(k): float(v) for k, v in own.operation_ms.items()}
        own.last_stats["operation_measurements_scope"] = "actual nested method totals are not additive; callback estimates are maxima, not totals"
        return action
    own.act = act
    return own


def configure(output):
    global WORKER_OUT
    WORKER_OUT = Path(output)
    original_target_filter = pilot.TeamBelief
    from learning.e45g_ppo import install_belief, TeamBelief
    import solvers.e45_team_adapter as adapter
    import solvers.e45b_joint_nte as joint
    install_belief(WORKER_OUT / "belief_failures")
    assert adapter.TeamBelief is joint.TeamBelief is base.TeamBelief is TeamBelief
    assert pilot.TeamBelief is original_target_filter, "Real fixed-D opponent filter must remain unchanged"
    base.OUT = WORKER_OUT; base.SMOKE = False
    base.make_controller = factory; base.audit_simulated = audit_simulated


def worker_init(output, config, cpu_queue, parent):
    global WORKER_CFG
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() != parent:
        raise RuntimeError("Coordinator exited before worker initialization")
    os.sched_setaffinity(0, {cpu_queue.get()}); torch.set_num_threads(1)
    WORKER_CFG = config; configure(output)
    validation = entries(config, "validation")
    if not validation:
        raise RuntimeError("Validation prefixes required for JIT warmup")
    case = read(validation[0]["path"])
    for arm, seed in (("SINGLE_PRIOR", 17), ("GEOM_SEARCH", 0), ("U", 0)):
        task = request(validation[0], "AB", "fixed_D_kernel", arm, seed, "sim", 2, "warmup")
        own = factory(case, task)
        for obs in case["history"][:-1]: own.observe(obs)
        own.act(case["history"][-1], fixed_simulations=2)
    audit_simulated(np.zeros((0, 26)), np.zeros((0, 2, 2)), 1.1)


def artifact_id(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()


def cached(output, task):
    path = Path(output) / "cache" / (artifact_id(task) + ".json")
    if not path.exists(): return None
    row = read(path)
    if row.get("key") != task["key"]:
        raise RuntimeError("Cache task identity changed")
    if row.get("episode") and sha(row["episode"]) != row["episode_sha256"]:
        raise RuntimeError("Cached episode corrupted")
    return row


def episode(task):
    path = WORKER_OUT / "cache" / (artifact_id(task) + ".json")
    old = cached(WORKER_OUT, task)
    if old is not None: return old
    try:
        payload, _ = base.episode(task)
        row = dict(payload["summary"])
        trace = payload["trace"]
        row.update(valid_episode=True, simulations=sum(r["stats"].get("simulations", 0) for r in trace),
                   component_measurement_scope="actual nested method wall totals; not additive",
                   nested_method_wall_totals_ms={})
        operations = collections.defaultdict(float)
        for frame in trace:
            for name, duration in frame["stats"].get("nested_method_wall_totals_ms", {}).items():
                operations[name] += duration
        row["nested_method_wall_totals_ms"] = dict(operations)
        if task["mode"] == "sim":
            actual = [r["stats"].get("simulations", 0) for r in trace]
            if not all(n == task["fixed"] for n in actual):
                raise RuntimeError("Fixed simulation count not reached: " + repr(actual))
        dest = WORKER_OUT / "episodes" / (artifact_id(task) + ".json.gz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = gzip.compress(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(), mtime=0)
        temp = dest.with_suffix(".tmp"); temp.write_bytes(data); temp.replace(dest)
        row.update(episode=str(dest), episode_sha256=sha(dest))
        save(path, row); return row
    except BaseException as exc:
        detail = dict(task=task, error=repr(exc), traceback=traceback.format_exc())
        save(WORKER_OUT / "failures" / (artifact_id(task) + ".json"), detail)
        if isinstance(exc, RuntimeError) and any(s in str(exc) for s in KNOWN_ERRORS):
            row = task | dict(valid_episode=False, failure=detail, utility=None,
                              double_capture=None, completion_cost=None)
            save(path, row); return row
        raise


def run_tasks(tasks, output, config, workers, cpus, stage):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    results, pending = [], collections.deque()
    for task in tasks:
        old = cached(output, task)
        if old is None: pending.append(task)
        else: results.append(old)
    total = len(tasks)
    if not pending: return sorted(results, key=lambda r: r["key"])
    count = min(workers, len(cpus), len(pending)); ctx = mp.get_context("spawn"); queue = ctx.Queue()
    for cpu in cpus[:count]: queue.put(cpu)
    pool = futures.ProcessPoolExecutor(max_workers=count, mp_context=ctx, initializer=worker_init,
                                       initargs=(str(output), config, queue, os.getpid()))
    active = {}; failed = True
    try:
        while pending or active:
            while pending and len(active) < count * 2:
                task = pending.popleft(); active[pool.submit(episode, task)] = task
            done, _ = futures.wait(active, timeout=5, return_when=futures.FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)
                try: results.append(future.result())
                except BaseException:
                    save(output / "failed_task.json", task); raise
            save(output / "status.json", dict(stage=stage, completed=len(results), planned=total,
                 workers=count, invalid=sum(not r.get("valid_episode", True) for r in results),
                 timestamp=time.time(), optimizer_updates=0))
        failed = False
    finally:
        processes = list((pool._processes or {}).values()); pool.shutdown(wait=not failed, cancel_futures=failed)
        if failed:
            for process in processes:
                if process.is_alive(): process.terminate()
            for process in processes: process.join(timeout=5)
        queue.close()
    return sorted(results, key=lambda r: r["key"])


def calibration_tasks(config, stage):
    validation = entries(config, "validation")
    selected = []
    for family in sorted({row["family"] for row in validation}):
        selected.append(next(row for row in validation if row["family"] == family))
    return [request(row, world, opponent, arm, seed, "wall", 200, stage, prefix=True)
            for row in selected for world in ("AB", "BA")
            for opponent in ("fixed_D_kernel", "rootpuct")
            for arm, seed in (("SINGLE_PRIOR", 17), ("GEOM_SEARCH", 0), ("U", 0))]


def timing(rows):
    if any(not row.get("valid_episode", True) for row in rows):
        raise RuntimeError("Calibration has an invalid prefix")
    wall = sum(sum(row["latencies_ms"]) for row in rows)
    cpu = sum(sum(row["cpu_ms"]) for row in rows)
    latencies = [v for row in rows for v in row["latencies_ms"]]
    return dict(cpu_wall=cpu / max(wall, 1e-12),
                overrun_rate=float(np.mean(np.asarray(latencies) > 200)),
                simulations=sum(row["simulations"] for row in rows),
                by_arm={arm: sum(row["simulations"] for row in rows if row["arm"] == arm)
                        for arm in ("SINGLE_PRIOR", "GEOM_SEARCH", "U")})


def calibrate(config, output, maximum, cpus):
    output = Path(output); decision = output / "decision.json"
    input_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if decision.exists():
        result = read(decision)
        if result.get("cpu_pool") != cpus: raise RuntimeError("Calibration CPU pool changed")
        if result.get("source_sha256") != sha(__file__) or result.get("config_sha256") != input_sha:
            raise RuntimeError("Calibration source/config changed")
        return result
    ref_rows = run_tasks(calibration_tasks(config, "calibration_serial"), output / "serial", config, 1, cpus, "calibration_serial")
    ref = timing(ref_rows); attempts = []
    max_jobs = len(ref_rows)
    for count in sorted({min(maximum, len(cpus), max_jobs)} | {n for n in (24, 16, 8, 4, 1)
                        if n <= maximum and n <= len(cpus) and n <= max_jobs}, reverse=True):
        rows = run_tasks(calibration_tasks(config, f"calibration_parallel_{count}"), output / f"parallel_{count}",
                         config, count, cpus, f"calibration_parallel_{count}")
        metrics = timing(rows)
        ratio = metrics["simulations"] / max(1, ref["simulations"])
        minimum = min((metrics["by_arm"][a] / ref["by_arm"][a] for a in ref["by_arm"] if ref["by_arm"][a] > 0), default=1.)
        passed = (metrics["cpu_wall"] >= .85 and ratio >= .85 and minimum >= .65
                  and metrics["overrun_rate"] <= ref["overrun_rate"] + .002)
        attempts.append(dict(workers=count, metrics=metrics, simulation_ratio=ratio,
                             minimum_arm_simulation_ratio=minimum, passed=passed))
        save(output / "attempts.json", dict(reference=ref, attempts=attempts, outcomes_used=False))
        if passed or count == 1:
            result = dict(workers=count, cpu_pool=cpus, reference=ref, attempts=attempts,
                          source_sha256=sha(__file__), config_sha256=input_sha,
                          outcomes_used=False, policy="highest throughput-qualified concurrency; one-worker reference fallback")
            save(decision, result); return result
    raise RuntimeError("No calibration candidate")


def make_tasks(config, mode, smoke=False):
    cases = entries(config, "validation")[:1] if smoke else entries(config, "test")
    values = (2,) if smoke else (WALL_BUDGETS if mode == "wall" else SIMULATIONS)
    return [request(row, world, opponent, arm, seed, "sim" if smoke else mode, value,
                    "engineering_smoke" if smoke else f"planning_{mode}", prefix=smoke)
            for row in cases for world in ("AB", "BA")
            for opponent in ("fixed_D_kernel", "rootpuct") for arm, seed in ARMS for value in values]


def summarize(rows, output, mode, smoke=False):
    good = [row for row in rows if row.get("valid_episode", True)]
    cells = collections.defaultdict(list)
    for row in good: cells[row["suite"], row["opponent"], row["control_value"], row["arm"]].append(row)
    summaries = []
    for (suite, opponent, value, arm), values in sorted(cells.items()):
        lat = [v for row in values for v in row["latencies_ms"]]
        cpu = [v for row in values for v in row["cpu_ms"]]
        summaries.append(dict(suite=suite, opponent=opponent, control_value=value, arm=arm,
             episodes=len(values), double_capture=sum(row["double_capture"] for row in values),
             double_capture_rate=float(np.mean([row["double_capture"] for row in values])),
             utility=float(np.mean([row["utility"] for row in values])),
             completion_cost=float(np.mean([row["completion_cost"] for row in values])),
             median_ms=float(np.median(lat)), p95_ms=float(np.percentile(lat, 95)), max_ms=float(max(lat)),
             cpu_wall=sum(cpu) / max(1e-12, sum(lat)),
             above_reference_threshold_rate=float(np.mean(np.asarray(lat) > values[0]["latency_reference_threshold_ms"])),
             construction_mean_ms=float(np.mean([r["construction_ms"] for r in values])),
             history_total_mean_ms=float(np.mean([sum(r["history_ms"]) for r in values])),
             simulations=sum(row["simulations"] for row in values),
             simulated_transitions=sum(row["simulated_transitions"] for row in values),
             training_seeds={str(seed): dict(episodes=len(v := [r for r in values if r["train_seed"] == seed]),
                                          utility=float(np.mean([r["utility"] for r in v])),
                                          double_capture_rate=float(np.mean([r["double_capture"] for r in v])))
                             for seed in sorted({r["train_seed"] for r in values})}))
    contrasts = []
    if len(good) == len(rows) and not smoke:
        for suite in sorted({r["suite"] for r in rows}):
            subset = [r for r in rows if r["suite"] == suite]
            groups = sorted({r["group"] for r in subset})
            comparisons = [(opponent, value, second) for opponent in ("fixed_D_kernel", "rootpuct")
                           for value in sorted({r["control_value"] for r in subset}) for second in ("GEOM_SEARCH", "U")]
            means = {(group, opponent, value, arm): np.mean([r["utility"] for r in subset if
                     r["group"] == group and r["opponent"] == opponent and r["control_value"] == value and r["arm"] == arm])
                     for group in groups for opponent, value, _ in comparisons for arm in ("SINGLE_PRIOR", "GEOM_SEARCH", "U")}
            delta = np.asarray([[means[group, op, value, "SINGLE_PRIOR"] - means[group, op, value, second]
                                 for op, value, second in comparisons] for group in groups])
            estimate = delta.mean(0); rng = np.random.default_rng(14530261103)
            families = {group: next(r["family"] for r in subset if r["group"] == group) for group in groups}
            boot = np.zeros((10000, len(comparisons)))
            for family in sorted(set(families.values())):
                indices = np.array([i for i, group in enumerate(groups) if families[group] == family])
                draws = rng.integers(len(indices), size=(10000, len(indices)))
                boot += delta[indices[draws]].sum(1) / len(groups)
            radius = float(np.quantile(np.max(np.abs(boot - estimate), axis=1), .95))
            for j, (op, value, second) in enumerate(comparisons):
                contrasts.append(dict(suite=suite, groups=len(groups), opponent=op, control_value=value,
                                      first="SINGLE_PRIOR", second=second, utility_difference=float(estimate[j]),
                                      simultaneous95=[float(estimate[j] - radius), float(estimate[j] + radius)],
                                      scope="family-stratified public-history paired bootstrap; all 12 contrasts per suite"))
    result = dict(complete=len(good) == len(rows), engineering_smoke=smoke, mode=mode,
                  planned=len(rows), valid=len(good), invalid=len(rows) - len(good), optimizer_updates=0,
                  summaries=summaries, contrasts=contrasts,
                  fixed_simulation_scope="No decision wall-time limit enforced; 200ms is a descriptive latency threshold" if mode == "sim" else None)
    save(Path(output) / "results.json", rows); save(Path(output) / "summary.json", result)
    lines = [f"# E45-W planning {mode}", "", f"Complete: {result['complete']}; valid {len(good)}/{len(rows)}; no training.", "",
             "|Suite|Opponent|Budget/count|Arm|Double|Utility|P95 ms|",
             "|---|---|---:|---|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"|{row['suite']}|{row['opponent']}|{row['control_value']}|{row['arm']}|{row['double_capture']}/{row['episodes']}|{row['utility']:.4f}|{row['p95_ms']:.2f}|")
    (Path(output) / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("wall", "sim", "smoke", "calibrate"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_ROOT / "experiment.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--cpus", required=True, help="Distinct physical CPU IDs, comma separated")
    parser.add_argument("--calibration", type=Path, help="Wall calibration directory; default sibling planning_calibration")
    args = parser.parse_args(); cpus = [int(v) for v in args.cpus.split(",")]
    if len(cpus) != len(set(cpus)) or not cpus: parser.error("CPU pool must be nonempty and unique")
    if args.workers < 1: parser.error("workers must be positive")
    cfg = manifest(args.manifest)
    output = args.output or DEFAULT_ROOT / f"planning_{args.mode}"
    output.mkdir(parents=True, exist_ok=True)
    calibration = args.calibration or output.parent / "planning_calibration"
    if args.mode == "calibrate":
        decision = calibrate(cfg, output, args.workers, cpus)
        print(json.dumps(decision, indent=2), flush=True); return
    smoke = args.mode == "smoke"; mode = "sim" if smoke else args.mode
    workers = min(args.workers, len(cpus))
    if mode == "wall":
        decision = calibrate(cfg, calibration, workers, cpus)
        workers = decision["workers"]
    tasks = make_tasks(cfg, mode, smoke)
    if len({task["key"] for task in tasks}) != len(tasks):
        raise RuntimeError("Duplicate task identities in manifest")
    run_protocol = dict(manifest=str(args.manifest), manifest_sha256=sha(args.manifest), mode=mode,
                        engineering_smoke=smoke, tasks=len(tasks), workers=workers, cpus=cpus,
                        model_sha256={str(seed): sha(MODEL_ROOT / f"single_{seed}.npz") for seed in SEEDS},
                        source_sha256=sha(__file__), episode_impl_sha256=sha(base.__file__),
                        hypothesis="quality versus decision cost of frozen final Single, matched GEOM and uniform tree",
                        statistics="12 original-utility contrasts per suite, family-stratified public-history paired intervals",
                        component_timer_scope="actual nested method totals plus callback max estimates; not additive; end-to-end latency authoritative",
                        public_belief="learning.e45g_ppo.TeamBelief, identical normal and exhausted-support paths to W PPO environment",
                        optimizer_updates=0)
    path = output / "protocol.json"
    if path.exists() and read(path) != run_protocol: raise RuntimeError("Frozen run protocol changed")
    save(path, run_protocol); started = time.time()
    rows = run_tasks(tasks, output, cfg, workers, cpus, f"planning_{mode}")
    result = summarize(rows, output, mode, smoke)
    result["elapsed_seconds"] = time.time() - started
    if sha(__file__) != run_protocol["source_sha256"] or sha(args.manifest) != run_protocol["manifest_sha256"]:
        save(output / "source_changed_during_run.json", dict(failed=True, optimizer_updates=0))
        raise RuntimeError("Source or manifest changed during scientific run")
    save(output / "summary.json", result)
    save(output / "completion_audit.json", dict(complete=result["complete"], tasks=len(tasks),
         source_sha256_unchanged=sha(__file__) == run_protocol["source_sha256"],
         manifest_sha256_unchanged=sha(args.manifest) == run_protocol["manifest_sha256"],
         optimizer_updates=0, elapsed_seconds=result["elapsed_seconds"]))
    print(json.dumps({k: v for k, v in result.items() if k not in ("summaries", "contrasts")}, indent=2), flush=True)


if __name__ == "__main__": main()
