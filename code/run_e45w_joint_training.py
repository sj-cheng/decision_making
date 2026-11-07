"""E45-W B/C joint PPO training, using the installed SB3 updater unchanged.

Every saved model/Actor is evaluated on fixed public validation episodes.
Final test entry paths are deliberately never consumed by this runner.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import contextlib
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

from learning.e45w_joint_env import PublicJointPursuitEnv
from learning.e45w_joint_ppo import StructuredJointPolicy, provider_audit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "current/e45w_joint_ppo_and_planning_20261103"
MANIFEST = OUT / "experiment.json"
FORMAL_ROLLOUTS = 976
FORMAL_N_ENVS = 8
FORMAL_N_STEPS = 256


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)); tmp.replace(path)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        arr = value.detach().cpu().contiguous().numpy() if torch.is_tensor(value) else np.ascontiguousarray(value)
        digest.update(key.encode()); digest.update(str((arr.shape, arr.dtype.str)).encode()); digest.update(arr.tobytes())
    return digest.hexdigest()


def rng_state():
    # Capture only this learner's active device.  Reading every visible GPU's
    # generator would initialize unrelated CUDA contexts on shared hardware.
    cuda = None
    if torch.cuda.is_initialized():
        device = torch.cuda.current_device()
        cuda = {"device": device, "state": torch.cuda.get_rng_state(device)}
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": cuda}


@contextlib.contextmanager
def preserve_rng():
    states = rng_state()
    try: yield
    finally:
        random.setstate(states["python"]); np.random.set_state(states["numpy"])
        torch.set_rng_state(states["torch"])
        if states["cuda"] is not None:
            torch.cuda.set_rng_state(states["cuda"]["state"], states["cuda"]["device"])


def validate_manifest(manifest):
    # Root freezes all implementation and task sources before formal launch.
    for key in ("source_hashes", "implementation_and_task_sha256"):
        for path, digest in manifest.get(key, {}).items():
            if isinstance(digest, dict): digest = digest.get("sha256")
            if digest is not None and sha(path) != digest:
                raise RuntimeError("Frozen experiment source changed: " + path)
    for row in manifest["train_entries"] + manifest["validation_entries"]:
        if row.get("sha256") is not None and sha(row["path"]) != row["sha256"]:
            raise RuntimeError("Frozen train/validation case changed: " + row["path"])
    for models in manifest["models"].values():
        for suffix in ("pt", "npz"):
            if models.get(suffix + "_sha256") and sha(models[suffix]) != models[suffix + "_sha256"]:
                raise RuntimeError("Frozen Single initialization changed: " + models[suffix])


def evaluation_specs(entries, stochastic_repeats=2, deterministic=False):
    """Common physical/controller/action seeds across A/B/C and checkpoints.

    Deterministic diagnostics use one action seed; primary stochastic results
    use the registered repeat count.  Repeats change only Actor randomness.
    """
    result = []
    for entry in entries:
        path = entry["path"]
        digest = entry.get("sha256") or sha(path)
        group = entry["id"]
        for world in ("AB", "BA"):
            episode_seed = int(np.random.SeedSequence([14530261103, int(digest[:8], 16), int(world == "BA")]).generate_state(1)[0])
            for opponent in ("fixed_D_kernel", "rootpuct"):
                for repeat in range(1 if deterministic else stochastic_repeats):
                    token = f"W/actions/{group}/{world}/{opponent}/{repeat}"
                    action_seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little")
                    result.append({"path": path, "group": group, "family": entry["family"],
                        "suite": entry.get("suite", entry.get("split", "validation")), "world": world,
                        "opponent": opponent, "episode_seed": episode_seed, "controller_seed": episode_seed,
                        "action_seed": action_seed, "repeat": repeat, "deterministic": deterministic})
    return result


EVAL_ENV = EVAL_POLICY = None


def init_evaluation_worker(case_paths):
    global EVAL_ENV, EVAL_POLICY
    torch.set_num_threads(1)
    EVAL_ENV = PublicJointPursuitEnv({"case_paths": case_paths})
    EVAL_POLICY = StructuredJointPolicy(EVAL_ENV.observation_space, EVAL_ENV.action_space,
                                       lambda _: 0., initialization_seed=0)
    EVAL_POLICY.set_training_mode(False)


@torch.no_grad()
def evaluate_frozen_actor(actor_state, spec, deterministic=None):
    """Single fixed evaluation episode; call init_evaluation_worker first."""
    if EVAL_ENV is None or EVAL_POLICY is None:
        raise RuntimeError("Call init_evaluation_worker before evaluating Actors")
    EVAL_POLICY.load_actor_state(actor_state)
    torch.manual_seed(int(spec["action_seed"]))
    mode = bool(spec["deterministic"] if deterministic is None else deterministic)
    obs, info = EVAL_ENV.reset(options={key: spec[key] for key in
        ("world", "opponent", "episode_seed", "controller_seed")} | {"case_path": spec["path"]})
    initial_hash = info["initial_state_sha256"]
    latencies = []; construction_step = info["start_step"]
    while True:
        started = time.perf_counter()
        action = EVAL_POLICY.predict(obs, deterministic=mode)[0]
        latencies.append(1000 * (time.perf_counter() - started))
        obs, _, terminated, truncated, info = EVAL_ENV.step(action)
        if truncated: raise RuntimeError("Real pursuit horizon must be true terminal, not time-limit truncation")
        if terminated: break
    row = info["task_episode"] | {"group": spec["group"], "family": spec["family"],
        "suite": spec.get("suite", "validation"), "repeat": spec["repeat"],
        "action_seed": spec["action_seed"], "deterministic": mode,
        "initial_state_sha256": initial_hash, "policy_inference_ms": latencies,
        "privileged_actor": False, "start_step": construction_step}
    return row


def _evaluate_job(job): return evaluate_frozen_actor(*job)


def score(rows):
    if not rows: raise ValueError("Empty evaluation")
    groups = {}
    for row in rows: groups.setdefault(row["group"], []).append(row)
    result = {"episodes": len(rows), "groups": len(groups),
        "utility": float(np.mean([r["utility"] for r in rows])),
        "double_captures": sum(int(r["double_capture"]) for r in rows),
        "double_capture_rate": float(np.mean([r["double_capture"] for r in rows])),
        "first_capture_rate": float(np.mean([r["first_capture"] is not None for r in rows])),
        "completion_cost": float(np.mean([r["completion_cost"] for r in rows])),
        "group_utility": {key: float(np.mean([r["utility"] for r in values])) for key, values in groups.items()}}
    result["by_opponent"] = {opp: {"utility": float(np.mean([r["utility"] for r in rows if r["opponent"] == opp])),
        "double_captures": sum(int(r["double_capture"]) for r in rows if r["opponent"] == opp),
        "episodes": sum(r["opponent"] == opp for r in rows)} for opp in ("fixed_D_kernel", "rootpuct")}
    return result


class SeedStreamEnv(PublicJointPursuitEnv):
    """Map only the first SB3 reset onto the registered rank seed stream."""
    def __init__(self, args, initial_seed):
        super().__init__(args); self.initial_seed = int(initial_seed); self.first_reset = True
    def reset(self, *, seed=None, options=None):
        if self.first_reset:
            seed = self.initial_seed; self.first_reset = False
        return super().reset(seed=seed, options=options)


def make_training_env(env_args, rank, seed, cpus):
    def make():
        if cpus: os.sched_setaffinity(0, {cpus[rank % len(cpus)]})
        torch.set_num_threads(1)
        return SeedStreamEnv(env_args, seed * 100000 + rank * 1000)
    return make


class TrainingAudit(BaseCallback):
    def __init__(self, directory, manifest, args, eval_pool, validation_entries):
        super().__init__(verbose=0)
        self.directory, self.manifest, self.args, self.eval_pool = directory, manifest, args, eval_pool
        self.started_wall = time.time(); self.started_cpu = time.process_time()
        self.eval_specs = evaluation_specs(validation_entries, 1 if args.smoke else 2, deterministic=False)
        self.det_specs = evaluation_specs(validation_entries, 1, deterministic=True)
        self.counts = {"environment_vector_ticks": 0, "team_transitions": 0, "agent_transitions": 0,
            "rollouts_collected": 0, "optimizer_steps": 0, "completed_training_episodes": 0,
            "training_double_captures": 0, "training_first_captures": 0, "validation_environment_steps": 0,
            "validation_episodes": 0, "initial_evaluations": 0}
        self.best = None; self.last_evaluated = None; self.latest_evaluation = None
        self.group_families = {entry["id"]: entry["family"] for entry in manifest["train_entries"]}
        self.hook = None

    def _init_callback(self):
        if self.hook is None:
            def observed_step(optimizer, args, kwargs): self.counts["optimizer_steps"] += 1
            self.hook = self.model.policy.optimizer.register_step_post_hook(observed_step)

    def _on_training_start(self):
        self.initial_actor_hash = tensor_hash(self.model.policy.actor_network.state_dict())
        self.initial_critic_hash = tensor_hash(self.model.policy.critic_network.state_dict())
        write(self.directory / "initialization_audit.json", {
            "arm": self.args.arm, "seed": self.args.seed, "actor_sha256": self.initial_actor_hash,
            "critic_sha256": self.initial_critic_hash, "actor_pretrained": self.args.arm == "C",
            "critic_pretrained": False, "provider": provider_audit()})
        self.evaluate(0)

    def _on_rollout_start(self):
        # This is after the previous rollout's provider PPO.train.  Doing it
        # in on_rollout_end would incorrectly score before that update.
        step = self.model.num_timesteps
        if step and step % (100 * FORMAL_N_ENVS * FORMAL_N_STEPS) == 0 and step != self.last_evaluated:
            self.evaluate(step)

    def _on_step(self):
        self.counts["environment_vector_ticks"] += 1
        self.counts["team_transitions"] = self.model.num_timesteps
        self.counts["agent_transitions"] = self.model.num_timesteps * 2
        episodes = []
        for info in self.locals["infos"]:
            if info.get("TimeLimit.truncated", False):
                raise RuntimeError("Unexpected time-limit bootstrap in finite task")
            if "task_episode" in info:
                row = copy.deepcopy(info["task_episode"])
                row.update(collect_team_transitions=self.model.num_timesteps,
                           family=self.group_families.get(row["case"]))
                episodes.append(row)
                self.counts["completed_training_episodes"] += 1
                self.counts["training_double_captures"] += int(row["double_capture"])
                self.counts["training_first_captures"] += int(row["first_capture"] is not None)
        if episodes:
            with (self.directory / "training_episodes.jsonl").open("a") as f:
                for row in episodes: f.write(json.dumps(row, allow_nan=False) + "\n")
        if episodes or self.model.num_timesteps % 2048 == 0: self.publish("training")
        return True

    def _on_rollout_end(self):
        self.counts["rollouts_collected"] += 1

    def save_checkpoint(self, step, *, selected=False):
        dest = self.directory / "checkpoints" / f"{step:08d}"
        dest.mkdir(parents=True, exist_ok=True)
        temporary = dest / "model.tmp.zip"
        self.model.save(temporary); temporary.replace(dest / "model.zip")
        state = self.model.policy.actor_state_numpy()
        torch.save(state, dest / "actor.pt")
        torch.save({"rng": rng_state(), "counts": copy.deepcopy(self.counts),
            "SB3_reported_epoch_updates": self.model._n_updates,
            "actor_sha256": tensor_hash(state),
            "scope": "model.zip includes optimizer; env worker states and partial rollout are not saved; not exact-resume"}, dest / "rng_audit.pt")
        if selected:
            torch.save(state, self.directory / "selected_actor.pt")
            write(self.directory / "selection.json", self.best | {"checkpoint": str(dest / "model.zip")})
        return dest

    def evaluate(self, step):
        state = self.model.policy.actor_state_numpy()
        with preserve_rng():
            stochastic = list(self.eval_pool.map(_evaluate_job, [(state, spec) for spec in self.eval_specs]))
            deterministic = list(self.eval_pool.map(_evaluate_job, [(state, spec) for spec in self.det_specs]))
        primary, diagnostic = score(stochastic), score(deterministic)
        self.counts["validation_episodes"] += len(stochastic) + len(deterministic)
        self.counts["validation_environment_steps"] += sum(r["decisions"] for r in stochastic + deterministic)
        if step == 0: self.counts["initial_evaluations"] += 1
        record = {"team_transitions": step, "primary_mode": "stochastic", "stochastic": primary,
            "deterministic_diagnostic": diagnostic, "elapsed_seconds": time.time() - self.started_wall,
            "counts": copy.deepcopy(self.counts), "SB3_reported_epoch_updates": self.model._n_updates,
            "actor_sha256": tensor_hash(state)}
        with (self.directory / "validation.jsonl").open("a") as f: f.write(json.dumps(record) + "\n")
        write(self.directory / "validation_episodes" / f"{step:08d}.json",
              {"stochastic": stochastic, "deterministic": deterministic})
        selected = self.best is None or primary["utility"] > self.best["utility"]
        if selected:
            self.best = {"team_transitions": step, "utility": primary["utility"],
                         "double_capture_rate": primary["double_capture_rate"], "mode": "stochastic"}
        self.latest_evaluation = record; self.last_evaluated = step
        dest = self.save_checkpoint(step, selected=selected)
        if step == 0:
            torch.save(state, self.directory / "initial_actor.pt")
        print(f"VALIDATION {self.args.arm}{self.args.seed} teams={step} "
              f"stochastic double={primary['double_captures']}/{primary['episodes']} "
              f"utility={primary['utility']:.6f}; det={diagnostic['double_captures']}/{diagnostic['episodes']}", flush=True)
        self.publish("training")
        return dest

    def publish(self, status, **extra):
        last_metrics = {}
        if getattr(self.model, "_logger", None) is not None:
            for key, value in self.model.logger.name_to_value.items():
                if key.startswith("train/") and np.isscalar(value):
                    value = float(value)
                    if np.isfinite(value): last_metrics[key] = value
        write(self.directory / "status.json", {"status": status, "pid": os.getpid(), "arm": self.args.arm,
            "seed": self.args.seed, "device": str(self.model.device),
            "elapsed_seconds": time.time() - self.started_wall,
            "learner_cpu_seconds": time.process_time() - self.started_cpu,
            "counts": self.counts, "SB3_reported_epoch_updates": self.model._n_updates,
            "best_validation": self.best,
            "latest_validation": None if self.latest_evaluation is None else {
                "team_transitions": self.latest_evaluation["team_transitions"],
                "stochastic": self.latest_evaluation["stochastic"],
                "deterministic_diagnostic": self.latest_evaluation["deterministic_diagnostic"]},
            "last_provider_train_metrics": last_metrics,
            "target_team_transitions": self.model._total_timesteps,
            "test_episodes_evaluated": 0, **extra})


def train(args):
    manifest = read(MANIFEST); validate_manifest(manifest)
    run_name = args.run_name or f"{args.arm}_{args.seed}"
    if "/" in run_name or run_name in (".", ".."):
        raise ValueError("run-name must be a single directory label")
    directory = OUT / ("smoke_training" if args.smoke else "training") / run_name
    directory.mkdir(parents=True, exist_ok=False)
    cores = [int(v) for v in args.cpus.split(",") if v]
    if cores: os.sched_setaffinity(0, set(cores))
    torch.set_num_threads(1)
    if str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    train_entries = manifest["train_entries"]
    val_entries = manifest["validation_entries"][:1] if args.smoke else manifest["validation_entries"]
    env_args = {"case_paths": [r["path"] for r in train_entries],
                "opponents": ["fixed_D_kernel", "rootpuct"], "root_simulations": 3000}
    n_steps = 8 if args.smoke else FORMAL_N_STEPS
    rollouts = 1 if args.smoke else FORMAL_ROLLOUTS
    total = FORMAL_N_ENVS * n_steps * rollouts
    models = manifest["models"][str(args.seed)]
    pretrained = str(models["pt"]) if args.arm == "C" else None
    settings = {"algorithm": "Stable-Baselines3 PPO 2.7.1 with structured joint policy",
        "arm": args.arm, "seed": args.seed, "run_name": run_name, "pretrained_actor": pretrained,
        "pretrained_critic": False, "n_envs": FORMAL_N_ENVS, "n_steps": n_steps,
        "rollouts": rollouts, "team_transitions": total, "batch_size": 32 if args.smoke else 256,
        "n_epochs": 1 if args.smoke else 10, "learning_rate": 3e-4, "gamma": 1., "gae_lambda": .95,
        "clip_range": .2, "ent_coef": .001, "vf_coef": .5, "max_grad_norm": .5,
        "target_kl": .03, "manifest_sha256": sha(MANIFEST), "provider": provider_audit(),
        "validation_stochastic_episodes": len(val_entries) * 4 * (1 if args.smoke else 2),
        "validation_deterministic_episodes": len(val_entries) * 4,
        "test_evaluations": 0, "smoke": args.smoke, "cpus": cores}
    write(directory / "requested_config.json", settings)
    envs = evaluation = model = audit = None
    try:
        envs = SubprocVecEnv([make_training_env(env_args, rank, args.seed, cores)
                             for rank in range(FORMAL_N_ENVS)], start_method="spawn")
        evaluation = ProcessPoolExecutor(max_workers=args.eval_workers, mp_context=mp.get_context("spawn"),
                    initializer=init_evaluation_worker, initargs=([r["path"] for r in val_entries],))
        model = PPO(StructuredJointPolicy, envs, n_steps=n_steps, batch_size=settings["batch_size"],
            n_epochs=settings["n_epochs"], learning_rate=3e-4, gamma=1., gae_lambda=.95,
            clip_range=.2, ent_coef=.001, vf_coef=.5, max_grad_norm=.5, target_kl=.03,
            policy_kwargs={"pretrained_path": pretrained, "initialization_seed": args.seed},
            seed=args.seed, device=args.device, verbose=0)
        model.set_logger(configure(str(directory / "sb3_logs"), ["csv", "json"]))
        audit = TrainingAudit(directory, manifest, args, evaluation, val_entries)
        model.learn(total_timesteps=total, callback=audit, log_interval=1, reset_num_timesteps=True)
        if model.num_timesteps != total: raise RuntimeError("Provider completed an unexpected number of team transitions")
        assert audit.counts["rollouts_collected"] == rollouts
        assert audit.counts["environment_vector_ticks"] == rollouts * n_steps
        assert audit.counts["optimizer_steps"] > 0
        terminal = audit.evaluate(model.num_timesteps)
        torch.save(model.policy.actor_state_numpy(), directory / "terminal_actor.pt")
        write(directory / "terminal_checkpoint.json", {"model": str(terminal / "model.zip"),
              "team_transitions": model.num_timesteps})
        # Real provider serialization and action equivalence, no new collection.
        if args.smoke:
            sample = {k: v[:1].copy() for k, v in model._last_obs.items()}
            expected = model.policy.predict(sample, deterministic=True)[0]
            with preserve_rng():
                loaded = PPO.load(terminal / "model.zip", device=args.device)
                actual = loaded.policy.predict(sample, deterministic=True)[0]
            np.testing.assert_array_equal(actual, expected)
            del loaded
        validate_manifest(manifest)
        audit.publish("complete", training_budget_audit_passed=True,
                      real_save_load_equivalence_passed=True if args.smoke else None,
                      initial_actor_sha256=audit.initial_actor_hash,
                      initial_critic_sha256=audit.initial_critic_hash)
    except BaseException as exc:
        write(directory / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        if audit is not None and model is not None: audit.publish("failed", error=repr(exc))
        raise
    finally:
        if evaluation is not None: evaluation.shutdown(wait=True, cancel_futures=True)
        if envs is not None: envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("B", "C"), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--cpus", default="")
    parser.add_argument("--eval-workers", type=int, default=2)
    parser.add_argument("--run-name", help="Preserve failed engineering attempts under distinct names")
    parser.add_argument("--smoke", action="store_true")
    train(parser.parse_args())
