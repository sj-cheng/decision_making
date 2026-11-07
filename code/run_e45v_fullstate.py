"""Full official HASAC collection loop with explicit full-state pursuit adapters."""
from __future__ import annotations
import argparse
import contextlib
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "current/e45v_fullstate_hasac_20261103"
AUTHOR = ROOT / "current/e45u_official_hasac_calibration_20261102/official_source"
sys.path.insert(0, str(AUTHOR))
import numpy as np
import torch
from harl.runners.off_policy_ha_runner import OffPolicyHARunner
import harl.runners.off_policy_base_runner as author_base
import harl.utils.configs_tools as author_configs
from harl.envs.env_wrappers import ShareSubprocVecEnv
from learning.e45v_fullstate_env import FullStatePursuitEnv, OBS_DIM, CENT_DIM
from learning.e45v_retirement import (RetirementRunnerMixin, RetirementHASAC,
                                      RetirementReplayEP, train_source_audit)


def read(path): return json.loads(Path(path).read_text())
def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)); tmp.replace(path)


def verify():
    cfg = read(OUT / "experiment.json")
    for name, digest in cfg["implementation_and_task_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    for row in cfg["entries"]:
        assert hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"], row["path"]
    for name, row in read(AUTHOR.parent / "source_manifest.json")["files"].items():
        assert hashlib.sha256((AUTHOR / name).read_bytes()).hexdigest() == row["sha256"], name
    assert cfg["retirement_train_diff"] == train_source_audit()
    return cfg


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


@contextlib.contextmanager
def preserve_rng():
    state = rng_state()
    try: yield
    finally:
        random.setstate(state["python"]); np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"]); torch.cuda.set_rng_state_all(state["cuda"])


def actor_weights(runner):
    return [{k: v.detach().cpu().numpy().copy() for k, v in a.actor.state_dict().items()} for a in runner.actor]


def load_actors(runner, weights):
    for actor, state in zip(runner.actor, weights):
        actor.actor.load_state_dict({k: torch.as_tensor(v, device=runner.device) for k, v in state.items()})


def configuration(seed, directory, smoke):
    reg = verify()
    cfg = read(AUTHOR / "tuned_configs/pettingzoo_mpe/simple_spread_v2-continuous/hasac/config.json")
    cfg["main_args"].update(env="pursuit_fullstate", exp_name="e45v_fixed_D_privileged")
    a = cfg["algo_args"]
    a["seed"]["seed"] = seed; a["logger"]["log_dir"] = str(directory / "author_results")
    a["algo"]["gamma"] = 1.
    a["train"]["use_proper_time_limits"] = False
    a["train"]["num_env_steps"] = 1000 if smoke else 1000000
    a["train"]["eval_interval"] = 50 if smoke else 5000
    cfg["env_args"] = {"case_paths": [r["path"] for r in reg["entries"] if r["split"] == "train"],
                       "opponent": "fixed_D_kernel", "max_boxes": 8,
                       "reward": "privileged_geometric_potential", "include_transition_kernel": False,
                       "world_ids": ["AB", "BA"], "state_type": "EP"}
    return cfg, reg


EVAL_ENV = EVAL_ACTORS = None
def init_evaluator(env_args, model_args):
    global EVAL_ENV, EVAL_ACTORS
    torch.set_num_threads(1)
    EVAL_ENV = FullStatePursuitEnv(env_args)
    EVAL_ACTORS = [RetirementHASAC(model_args, EVAL_ENV.observation_space[i],
                                  EVAL_ENV.action_space[i], torch.device("cpu")) for i in range(2)]


@torch.no_grad()
def evaluate_episode(job):
    weights, spec = job
    for actor, state in zip(EVAL_ACTORS, weights):
        actor.actor.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})
    obs, _, _ = EVAL_ENV.reset(case_path=spec["path"], world=spec["world"], episode_seed=spec["episode_seed"])
    while True:
        actions = np.stack([actor.get_actions(obs[i:i+1], stochastic=False)[0].numpy()
                            for i, actor in enumerate(EVAL_ACTORS)])
        obs, _, _, done, info, _ = EVAL_ENV.step(actions)
        if all(done):
            return info[0]["episode"] | {"group": spec["group"], "family": spec["family"]}


class EvaluationPool:
    def __init__(self, env_args, model_args, workers):
        self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                        initializer=init_evaluator, initargs=(env_args, model_args))
    def evaluate(self, weights, specs):
        return list(self.pool.map(evaluate_episode, [(weights, spec) for spec in specs]))
    def close(self): self.pool.shutdown(wait=True, cancel_futures=True)


def evaluation_specs(reg, split, smoke=False):
    entries = [r for r in reg["entries"] if r["split"] == split]
    if smoke: entries = entries[:1]
    repeats = 1 if smoke else (5 if split == "validation" else 10)
    result = []
    for row in entries:
        for world_index, world in enumerate(("AB", "BA")):
            for rep in range(repeats):
                # Split namespaces are separated; unchanged across training seeds/checkpoints.
                digest = hashlib.sha256(f"V/{split}/{row['id']}/{world}/{rep}".encode()).digest()
                episode_seed = int.from_bytes(digest[:4], "little") % (2**31 - 2) + 1
                result.append({"path": row["path"], "group": row["id"], "family": row["family"],
                               "world": world, "episode_seed": episode_seed})
    return result


def score(rows):
    groups = {}
    for row in rows: groups.setdefault(row["group"], []).append(row)
    return {"episodes": len(rows), "groups": len(groups),
            "utility": float(np.mean([r["utility"] for r in rows])),
            "double_capture_rate": float(np.mean([r["double_capture"] for r in rows])),
            "double_captures": sum(int(r["double_capture"]) for r in rows),
            "first_capture_rate": float(np.mean([r["first_capture"] is not None for r in rows])),
            "completion_cost": float(np.mean([r["completion_cost"] for r in rows])),
            "group_utility": {k: float(np.mean([r["utility"] for r in v])) for k, v in groups.items()},
            "group_double_capture_rate": {k: float(np.mean([r["double_capture"] for r in v])) for k, v in groups.items()}}


@contextlib.contextmanager
def task_registration(cfg, smoke):
    """Scoped environment/actor registration; never edits upstream source files."""
    saved = {k: getattr(author_base, k) for k in ("make_train_env", "make_eval_env", "get_num_agents", "get_task_name", "OffPolicyBufferEP")}
    old_task = author_configs.get_task_name
    old_actor = author_base.ALGO_REGISTRY["hasac"]
    def task_name(env, args):
        return "fixed_D_fullstate" if env == "pursuit_fullstate" else old_task(env, args)
    def make_train(env, seed, n, args):
        assert env == "pursuit_fullstate"
        def factory(rank):
            def create():
                e = FullStatePursuitEnv(args); e.seed(seed + rank * 1000); return e
            return create
        return ShareSubprocVecEnv([factory(i) for i in range(n)])
    def make_eval(env, seed, n, args):
        return EvaluationPool(args, {**cfg["algo_args"]["model"], **cfg["algo_args"]["algo"]},
                              2 if smoke else n)
    author_base.make_train_env = make_train; author_base.make_eval_env = make_eval
    author_base.get_num_agents = lambda env, args, envs: 2
    author_base.get_task_name = task_name; author_configs.get_task_name = task_name
    author_base.ALGO_REGISTRY["hasac"] = RetirementHASAC
    author_base.OffPolicyBufferEP = RetirementReplayEP
    try: yield
    finally:
        for k, v in saved.items(): setattr(author_base, k, v)
        author_base.ALGO_REGISTRY["hasac"] = old_actor
        author_configs.get_task_name = old_task


class FullStateRunner(RetirementRunnerMixin, OffPolicyHARunner):
    def __init__(self, cfg, reg, directory, smoke=False):
        self.directory, self.reg, self.smoke = directory, reg, smoke
        self.started = time.time(); self.best = None; self.eval_history = []
        self.counts = {"vector_ticks": 0, "team_transitions": 0, "agent_transitions": 0,
                       "train_calls": 0, "sample_calls": 0, "sampled_team_rows": 0,
                       "optimizer_steps": {}, "training_episodes": 0, "training_double_captures": 0,
                       "training_first_captures": 0, "eval_episodes": 0}
        self.specs = evaluation_specs(reg, "validation", smoke)
        with task_registration(cfg, smoke):
            super().__init__(cfg["main_args"], cfg["algo_args"], cfg["env_args"])
        step = self.envs.step
        def observe_step(actions):
            result = step(actions)
            self.counts["vector_ticks"] += 1
            self.counts["team_transitions"] += len(result[0])
            self.counts["agent_transitions"] += 2 * len(result[0])
            episodes = [info[0]["episode"] for info in result[4] if "episode" in info[0]]
            for row in episodes:
                self.counts["training_episodes"] += 1
                self.counts["training_double_captures"] += int(row["double_capture"])
                self.counts["training_first_captures"] += int(row["first_capture"] is not None)
            if episodes:
                with (directory / "training_episodes.jsonl").open("a") as f:
                    for row in episodes: f.write(json.dumps(row) + "\n")
            return result
        self.envs.step = observe_step
        sample = self.buffer.sample
        def observe_sample():
            result = sample(); self.counts["sample_calls"] += 1
            self.counts["sampled_team_rows"] += len(result[0]); return result
        self.buffer.sample = observe_sample
        self.hooks = []
        opts = [(f"actor_{i}", a.actor_optimizer) for i, a in enumerate(self.actor)]
        opts += [(f"actor_alpha_{i}", o) for i, o in enumerate(self.alpha_optimizer)]
        opts += [("twin_critic_joint_adam", self.critic.critic_optimizer), ("critic_alpha", self.critic.alpha_optimizer)]
        for name, optimizer in opts:
            self.counts["optimizer_steps"][name] = 0
            def hook(opt, args, kwargs, key=name): self.counts["optimizer_steps"][key] += 1
            self.hooks.append(optimizer.register_step_post_hook(hook))
        torch.save(actor_weights(self), directory / "initial_actors.pt")
        self.eval(0)

    def train(self):
        super().train()
        self.counts["train_calls"] += 1
        if self.total_it % 1000 == 0:
            if not all(bool(torch.isfinite(p).all()) for a in self.actor for p in a.actor.parameters()):
                raise RuntimeError("Nonfinite Actor parameters")
            if not all(bool(torch.isfinite(p).all()) for m in (self.critic.critic, self.critic.critic2) for p in m.parameters()):
                raise RuntimeError("Nonfinite Critic parameters")
            self.publish("training")

    def warmup(self):
        result = super().warmup()
        self.counts["warmup_team_transitions"] = self.counts["team_transitions"]
        self.publish("training")
        return result

    def eval(self, step):
        with preserve_rng(): rows = self.eval_envs.evaluate(actor_weights(self), self.specs)
        summary = score(rows); self.counts["eval_episodes"] += len(rows)
        row = {"step": step, "elapsed_seconds": time.time() - self.started, "score": summary,
               "counts": copy.deepcopy(self.counts), "actor_alpha": [float(a) for a in self.alpha],
               "critic_alpha": float(self.critic.alpha)}
        self.eval_history.append(row)
        with (self.directory / "validation.jsonl").open("a") as f: f.write(json.dumps(row) + "\n")
        write(self.directory / "validation_episodes" / f"{step}.json", rows)
        if self.best is None or summary["utility"] > self.best["utility"]:
            self.best = {"step": step, "utility": summary["utility"],
                         "double_capture_rate": summary["double_capture_rate"]}
            torch.save(actor_weights(self), self.directory / "selected_actors.pt")
            write(self.directory / "selection.json", self.best)
        self.log_file.write(f"{step},{summary['utility']},{summary['double_capture_rate']}\n"); self.log_file.flush()
        self.writter.add_scalar("validation_raw_utility", summary["utility"], step)
        self.writter.add_scalar("validation_double_capture_rate", summary["double_capture_rate"], step)
        print(f"VALIDATION step={step} utility={summary['utility']:.4f} double={summary['double_captures']}/{summary['episodes']}", flush=True)
        self.publish("training")

    def save(self):
        super().save()
        directory = self.directory / "checkpoints" / str(self.counts["team_transitions"])
        directory.mkdir(parents=True, exist_ok=True)
        for p in Path(self.save_dir).glob("*.pt"): shutil.copy2(p, directory / p.name)
        torch.save({"actor_optimizers": [a.actor_optimizer.state_dict() for a in self.actor],
                    "critic_optimizer": self.critic.critic_optimizer.state_dict(),
                    "actor_log_alpha": self.log_alpha, "critic_log_alpha": self.critic.log_alpha,
                    "actor_alpha_optimizers": [a.state_dict() for a in self.alpha_optimizer],
                    "critic_alpha_optimizer": self.critic.alpha_optimizer.state_dict(),
                    "rng": rng_state(), "counts": self.counts,
                    "scope": "audit snapshot, excludes replay and env states; not exact-resume"}, directory / "optimizer_audit.pt")

    def publish(self, status, **extra):
        write(self.directory / "status.json", {"status": status, "pid": os.getpid(),
              "seed": self.algo_args["seed"]["seed"], "elapsed_seconds": time.time() - self.started,
              "counts": self.counts, "best_validation": self.best,
              "latest_validation": self.eval_history[-1] if self.eval_history else None,
              "retirement_insert": getattr(self, "retirement_insert_audit", {}),
              "retirement_resamples": self.buffer.retirement_resamples,
              "target_post_warmup_team_transitions": self.algo_args["train"]["num_env_steps"],
              "official_run_dir": str(self.run_dir), **extra})


def test_models(runner):
    states = actor_weights(runner); results = {}
    specs = evaluation_specs(runner.reg, "test")
    for name, weights in (("initial", torch.load(runner.directory / "initial_actors.pt", weights_only=False)),
                          ("validation_selected", torch.load(runner.directory / "selected_actors.pt", weights_only=False)),
                          ("terminal", states)):
        with preserve_rng(): rows = runner.eval_envs.evaluate(weights, specs)
        results[name] = {"score": score(rows), "episodes": rows}
        runner.counts["eval_episodes"] += len(rows)
    load_actors(runner, states)
    for name in ("validation_selected", "terminal"):
        selected = results[name]["score"]; initial = results["initial"]["score"]
        results[name + "_minus_initial"] = {"utility": selected["utility"] - initial["utility"],
                "double_capture_rate": selected["double_capture_rate"] - initial["double_capture_rate"],
                "paired_group_utility_differences": {k: selected["group_utility"][k] - v for k, v in initial["group_utility"].items()}}
    results["selection"] = runner.best
    write(runner.directory / "sealed_test.json", results)


def run(args):
    directory = OUT / args.name; directory.mkdir(parents=True, exist_ok=False)
    cfg, reg = configuration(args.seed, directory, args.smoke)
    write(directory / "requested_config.json", cfg)
    runner = None
    try:
        runner = FullStateRunner(cfg, reg, directory, args.smoke)
        assert runner.actor[0].actor.net is not None and runner.num_agents == 2
        runner.run()  # complete frozen author warmup/collect/burst schedule
        post = cfg["algo_args"]["train"]["num_env_steps"]
        assert runner.counts["team_transitions"] == post + 10000
        assert runner.counts["train_calls"] == post // 20
        assert runner.counts["sample_calls"] == post // 20
        assert runner.counts["sampled_team_rows"] == post // 20 * 1000
        assert all(n == runner.total_it for n in runner.counts["optimizer_steps"].values())
        assert len(runner.counts["optimizer_steps"]) == 6
        assert runner.retirement_insert_audit["team_rows"] == post + 10000
        n = runner.buffer.cur_size
        assert np.array_equal(runner.buffer.terms[:n], runner.buffer.dones[:n])
        assert bool(runner.buffer.dones[:n].any()), "Warmup must cover real episode terminals"
        for value in list(runner.log_alpha) + [runner.critic.log_alpha]:
            assert bool(torch.isfinite(value).all()), "Nonfinite temperature parameter"
        for agent in range(2):
            assert np.array_equal(runner.buffer.obs[agent][:n, -1:], runner.buffer.valid_transitions[agent][:n])
            mask = runner.buffer.valid_transitions[agent][:n, 0] == 0
            assert np.all(runner.buffer.actions[agent][:n][mask] == 0)
            terminals = runner.buffer.dones[:n, 0]
            # At true terminals the saved successor clock is 80 or its active
            # evader flags are zero, never the next episode's handover clock.
            successor = runner.buffer.next_obs[agent][:n][terminals]
            clock = successor[:, 8] * 80
            evader_flags = successor[:, runner.env_args.get("_unused", 9):11]
            assert np.all((np.abs(clock - 80) < 1e-4) | (evader_flags.sum(axis=1) == 0))
        if not args.smoke: test_models(runner)
        verify(); runner.publish("complete", unit_audit_passed=True, lifecycle_replay_audit_passed=True)
    except BaseException as exc:
        write(directory / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        if runner is not None: runner.publish("failed", error=repr(exc))
        raise
    finally:
        if runner is not None: runner.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name", required=True); parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
