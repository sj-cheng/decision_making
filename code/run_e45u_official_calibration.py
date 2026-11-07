"""Frozen HARL native-MPE calibration. Observer only; official run/train unchanged."""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "current/e45u_official_hasac_calibration_20261102"
SOURCE = OUT / "official_source"
sys.path.insert(0, str(SOURCE))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch
from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv
from harl.runners.off_policy_ha_runner import OffPolicyHARunner


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


@contextlib.contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state_all(state["cuda"])


def config(seed, directory, smoke):
    cfg = json.loads((SOURCE / "tuned_configs/pettingzoo_mpe/simple_spread_v2-continuous/hasac/config.json").read_text())
    cfg["algo_args"]["seed"]["seed"] = seed
    cfg["algo_args"]["logger"]["log_dir"] = str(directory / "official_results")
    cfg["main_args"]["exp_name"] = "e45u_native_calibration"
    if smoke:
        # Engineering smoke only: same full warmup/buffer/batch; 50 official updates.
        cfg["algo_args"]["train"]["num_env_steps"] = 1000
        cfg["algo_args"]["train"]["eval_interval"] = 50
    return cfg


def verify_source():
    manifest = json.loads((OUT / "source_manifest.json").read_text())
    for name, row in manifest["files"].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == row["sha256"], name
    return manifest["commit"]


def digest(runner):
    """Numerical fingerprint including parameters, optimizers, replay, and RNG."""
    h = hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            a = x.detach().cpu().contiguous().numpy()
            h.update(str((a.dtype, a.shape)).encode()); h.update(a.tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode()); add(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x: add(v)
        else:
            h.update(repr(x).encode())
    for a in runner.actor:
        add(a.actor.state_dict()); add(a.actor_optimizer.state_dict())
    for name in ("critic", "critic2", "target_critic", "target_critic2"):
        add(getattr(runner.critic, name).state_dict())
    add(runner.critic.critic_optimizer.state_dict())
    add(runner.critic.alpha_optimizer.state_dict()); add(runner.critic.log_alpha)
    add(runner.log_alpha)
    for opt in runner.alpha_optimizer: add(opt.state_dict())
    # Buffer's space objects have irrelevant process-specific identities.
    for k, v in sorted(vars(runner.buffer).items()):
        if isinstance(v, (np.ndarray, int, float, bool)) or (
            isinstance(v, list) and all(isinstance(z, np.ndarray) for z in v)
        ): add({k: v})
    add(runner.total_it); add(runner.agent_deaths); add(rng_state())
    return h.hexdigest()


def actor_weights(runner):
    return [{k: v.detach().cpu().clone() for k, v in a.actor.state_dict().items()}
            for a in runner.actor]


def load_actors(runner, states):
    for a, state in zip(runner.actor, states):
        a.actor.load_state_dict(state)


@torch.no_grad()
def fixed_evaluate(runner, seeds):
    """Independent native envs, fixed public starts, no RNG impact on training.

    The official adapter sums native agent rewards then repeats that team reward
    to all agents. Use one copy, never sum the repeated copies a second time.
    """
    rewards = []
    with preserve_rng():
        for start in range(0, len(seeds), 20):
            batch = seeds[start:start + 20]
            envs = [PettingZooMPEEnv(runner.env_args) for _ in batch]
            try:
                obs = []
                for env, seed in zip(envs, batch):
                    env.seed(int(seed) - 1)  # official reset increments seed once
                    obs.append(env.reset()[0])
                totals = np.zeros(len(batch), dtype=np.float64)
                for step in range(25):
                    actions = runner.get_actions(np.asarray(obs), None, add_random=False)
                    rows = [env.step(action) for env, action in zip(envs, actions)]
                    obs = [r[0] for r in rows]
                    for j, row in enumerate(rows):
                        assert bool(np.all(row[3])) == (step == 24)
                        totals[j] += float(row[2][0][0])
                assert np.isfinite(totals).all()
                rewards.extend(totals.tolist())
            finally:
                for env in envs: env.close()
    return {"mean": float(np.mean(rewards)), "std": float(np.std(rewards)),
            "episodes": len(rewards), "returns": rewards,
            "reward_unit": "25-step sum of native per-agent rewards, counted once per team"}


class ObservedRunner(OffPolicyHARunner):
    """Only delegates to frozen author methods; adds counters and extra evaluation."""
    def __init__(self, cfg, directory, smoke):
        self.directory = directory
        self.smoke = smoke
        self.started = time.time()
        self.counts = {"vector_ticks": 0, "team_transitions": 0,
                       "agent_transitions": 0, "train_calls": 0,
                       "sample_calls": 0, "sampled_team_rows": 0,
                       "optimizer_steps": {}, "official_eval_calls": 0,
                       "fixed_eval_episodes": 0}
        self.in_warmup = False
        self.best = None
        self.validation_seeds = list(range(2000001, 2000001 + (20 if smoke else 100)))
        super().__init__(cfg["main_args"], cfg["algo_args"], cfg["env_args"])
        base_step = self.envs.step
        def observed_step(actions):
            result = base_step(actions)
            n = len(result[0])
            self.counts["vector_ticks"] += 1
            self.counts["team_transitions"] += n
            self.counts["agent_transitions"] += n * self.num_agents
            return result
        self.envs.step = observed_step
        base_sample = self.buffer.sample
        def observed_sample():
            sample = base_sample()
            self.counts["sample_calls"] += 1
            self.counts["sampled_team_rows"] += len(sample[0])
            return sample
        self.buffer.sample = observed_sample
        optimizers = [(f"actor_{i}", a.actor_optimizer) for i, a in enumerate(self.actor)]
        optimizers += [(f"actor_alpha_{i}", opt) for i, opt in enumerate(self.alpha_optimizer)]
        optimizers += [("twin_critic_joint_adam", self.critic.critic_optimizer),
                       ("critic_alpha", self.critic.alpha_optimizer)]
        self.hooks = []
        for name, optimizer in optimizers:
            self.counts["optimizer_steps"][name] = 0
            def hook(opt, args, kwargs, key=name):
                self.counts["optimizer_steps"][key] += 1
            self.hooks.append(optimizer.register_step_post_hook(hook))
        torch.save(actor_weights(self), directory / "initial_actors.pt")
        self.validate_and_record(0)
        self.initial_digest = digest(self) if smoke else None

    def warmup(self):
        self.in_warmup = True
        result = super().warmup()
        self.in_warmup = False
        self.counts["warmup_vector_ticks"] = self.counts["vector_ticks"]
        self.counts["warmup_team_transitions"] = self.counts["team_transitions"]
        self.publish("training")
        return result

    def train(self):
        super().train()
        self.counts["train_calls"] += 1
        if self.counts["train_calls"] % 1000 == 0:
            self.publish("training")

    def eval(self, step):
        super().eval(step)
        self.counts["official_eval_calls"] += 1
        self.validate_and_record(step)

    def validate_and_record(self, step):
        result = fixed_evaluate(self, self.validation_seeds)
        self.counts["fixed_eval_episodes"] += result["episodes"]
        record = {"step": step, "elapsed_seconds": time.time() - self.started,
                  "counts": copy.deepcopy(self.counts), "validation": result,
                  "actor_alpha": [float(a) for a in self.alpha],
                  "critic_alpha": float(self.critic.alpha)}
        with (self.directory / "validation.jsonl").open("a") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")
        if self.best is None or result["mean"] > self.best["validation_mean"]:
            self.best = {"step": step, "validation_mean": result["mean"]}
            torch.save(actor_weights(self), self.directory / "selected_actors.pt")
            write_json(self.directory / "selection.json", self.best)
        self.publish("training")
        print(f"AUDIT step={step} val={result['mean']:.4f} train_calls={self.total_it}", flush=True)

    def save(self):
        super().save()
        step = self.counts["team_transitions"]
        dest = self.directory / "checkpoints" / str(step)
        dest.mkdir(parents=True, exist_ok=True)
        for f in Path(self.save_dir).glob("*.pt"):
            shutil.copy2(f, dest / f.name)
        torch.save({"actor_optimizers": [a.actor_optimizer.state_dict() for a in self.actor],
                    "critic_optimizer": self.critic.critic_optimizer.state_dict(),
                    "actor_alpha_optimizers": [a.state_dict() for a in self.alpha_optimizer],
                    "critic_alpha_optimizer": self.critic.alpha_optimizer.state_dict(),
                    "log_alpha": self.log_alpha, "critic_log_alpha": self.critic.log_alpha,
                    "rng": rng_state(), "counts": self.counts, "total_it": self.total_it,
                    "scope": "audit snapshot; replay and worker env states excluded; not exact-resume"},
                   dest / "optimizer_audit.pt")

    def publish(self, status, **extra):
        elapsed = time.time() - self.started
        warmup = self.counts.get("warmup_team_transitions", 0)
        post = self.counts["team_transitions"] - warmup
        write_json(self.directory / "status.json", {
            "status": status, "pid": os.getpid(), "seed": self.algo_args["seed"]["seed"],
            "elapsed_seconds": elapsed, "counts": self.counts, "best_validation": self.best,
            "post_warmup_train_calls_per_team_transition": self.counts["train_calls"] / post if post else None,
            "target_post_warmup_team_transitions": self.algo_args["train"]["num_env_steps"],
            "official_run_dir": str(self.run_dir), **extra})


def paired_difference(trained, initial):
    delta = np.asarray(trained["returns"]) - np.asarray(initial["returns"])
    mean = float(np.mean(delta))
    se = float(np.std(delta, ddof=1) / np.sqrt(len(delta)))
    return {"mean_improvement": mean, "normal_approx_95ci": [mean - 1.96 * se, mean + 1.96 * se],
            "episodes": len(delta), "inference_scope": "within-seed paired episode variation; not across-training-seed uncertainty"}


def final_test(runner):
    final = actor_weights(runner)
    tests = list(range(3000001, 3000201))
    result = {}
    for name, weights in (("initial", torch.load(runner.directory / "initial_actors.pt", weights_only=False)),
                          ("validation_selected", torch.load(runner.directory / "selected_actors.pt", weights_only=False)),
                          ("terminal", final)):
        load_actors(runner, weights)
        result[name] = fixed_evaluate(runner, tests)
    load_actors(runner, final)
    runner.counts["fixed_eval_episodes"] += 3 * len(tests)
    for name in ("validation_selected", "terminal"):
        result[name + "_vs_initial"] = paired_difference(result[name], result["initial"])
    result["selected_step"] = runner.best["step"]
    write_json(runner.directory / "sealed_test.json", result)
    return result


def run(args):
    verify_source()
    directory = OUT / args.name
    directory.mkdir(parents=True, exist_ok=False)
    cfg = config(args.seed, directory, args.smoke)
    write_json(directory / "requested_config.json", cfg)
    runner = None
    try:
        cls = OffPolicyHARunner if args.stock else ObservedRunner
        if args.stock:
            runner = cls(cfg["main_args"], cfg["algo_args"], cfg["env_args"])
            initial = digest(runner)
        else:
            runner = cls(cfg, directory, args.smoke)
            initial = runner.initial_digest
        runner.run()  # frozen official complete runner, including warmup and eval
        if args.smoke:
            write_json(directory / "smoke_fingerprint.json", {
                "initial": initial, "terminal": digest(runner), "total_it": runner.total_it,
                "replay_size": runner.buffer.cur_size})
        if not args.stock:
            post = cfg["algo_args"]["train"]["num_env_steps"]
            assert runner.counts["team_transitions"] == post + 10000
            assert runner.counts["train_calls"] == post // 20
            assert runner.counts["sampled_team_rows"] == post // 20 * 1000
            assert all(n == runner.total_it for n in runner.counts["optimizer_steps"].values())
            assert runner.counts["agent_transitions"] == runner.counts["team_transitions"] * 3
            if not args.smoke:
                result = final_test(runner)
                runner.publish("complete", unit_audit_passed=True,
                               selected_test_mean=result["validation_selected"]["mean"],
                               terminal_test_mean=result["terminal"]["mean"])
            else:
                runner.publish("complete", unit_audit_passed=True)
    except BaseException as exc:
        write_json(directory / "failure.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        if isinstance(runner, ObservedRunner): runner.publish("failed", error=repr(exc))
        raise
    finally:
        if runner is not None: runner.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stock", action="store_true")
    a = parser.parse_args()
    if a.stock and not a.smoke: parser.error("stock comparison is restricted to engineering smoke")
    run(a)
