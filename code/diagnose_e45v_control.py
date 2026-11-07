"""Frozen-policy matched training/validation diagnostics; zero optimizer updates."""
from __future__ import annotations
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[k] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch
from run_e45v_fullstate import FullStatePursuitEnv, RetirementHASAC, verify

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "current/e45v_fullstate_hasac_20261103"
OUT = ROOT / "current/e45v_control_diagnosis_20261103"
ENV = ACTORS = None
SINGLE = ROOT / "current/e45c_budget_scaling_20261028/learning/x20/round_12/models/single_17.npz"


def read(p): return json.loads(Path(p).read_text())
def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def init(cfg):
    global ENV, ACTORS
    torch.set_num_threads(1)
    ENV = FullStatePursuitEnv(cfg["env_args"])
    args = cfg["algo_args"]["model"] | cfg["algo_args"]["algo"]
    ACTORS = [RetirementHASAC(args, ENV.observation_space[i], ENV.action_space[i], torch.device("cpu")) for i in range(2)]
    # Install only the already-published public controller's support/capacity
    # compatibility. FixedKernelTargets retains its original TeamBelief alias.
    import run_e45_team_pilot as pilot
    original_target_filter = pilot.TeamBelief
    from solvers.e45f_support_refinement import install
    install(OUT / "controller_belief_evidence")
    from solvers.e45f_geometry_capacity import geometric_candidates_capacity
    import solvers.e45_team_adapter as adapter
    adapter.geometric_candidates = geometric_candidates_capacity
    assert pilot.TeamBelief is original_target_filter


@torch.no_grad()
def episode(job):
    method, seed, stage, spec, weights = job
    action_seed = int.from_bytes(hashlib.sha256(f"Vdiag/{seed}/{spec['group']}/{spec['world']}".encode()).digest()[:4], "little")
    torch.manual_seed(action_seed); rng = np.random.default_rng(action_seed)
    obs, _, _ = ENV.reset(case_path=spec["path"], world=spec["world"], episode_seed=spec["episode_seed"])
    initial_hash = hashlib.sha256(ENV.state.tobytes() + b"".join(r.tobytes() for r in ENV.other.rng)).hexdigest()
    if weights is not None:
        for actor, state in zip(ACTORS, weights):
            actor.actor.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})
    own = None
    if method in ("GEOM_SEARCH", "SINGLE_DIRECT_17"):
        from solvers.e45f_baselines import GeometryController
        from solvers.e45b_joint_nte import Controller
        if method == "GEOM_SEARCH":
            own = GeometryController(ENV.case["map"], ENV.speed, action_seed,
                                     use_policy=False, use_value=False, budget_ms=200.)
        else:
            own = Controller(ENV.case["map"], ENV.speed, action_seed, SINGLE,
                             use_policy=True, use_value=False, direct=True, budget_ms=200.)
        for public in ENV.case["history"][:-1]: own.observe(public)
    active_actions = 0; moving = []; intents = []; stds = []; blocked = []
    root_mean_actions, root_stds = [], []
    if weights is not None:
        for i, actor in enumerate(ACTORS):
            x = torch.as_tensor(obs[i:i+1]); net = actor.actor.net(x)
            mu = actor.actor.mu_layer(net); log_std = actor.actor.log_std_layer(net).clamp(-20, 2)
            root_mean_actions.extend(torch.tanh(mu).numpy().ravel().tolist())
            root_stds.extend(log_std.exp().numpy().ravel().tolist())
    simulations = 0; simulated_steps = 0; max_interface_error = 0.
    while True:
        active = ENV.active_p.copy()
        if own is not None:
            physical = own.act(ENV.public_obs())
            action = physical / ENV.speed
            simulations += own.last_stats.get("simulations", 0)
            simulated_steps += own.last_stats.get("simulated_transitions", 0)
        elif method == "stop": action = np.zeros((2, 2))
        elif method == "uniform": action = rng.uniform(-1, 1, (2, 2)) * active[:, None]
        else:
            action = np.stack([actor.get_actions(obs[i:i+1], stochastic=(method == "stochastic"))[0].numpy()
                               for i, actor in enumerate(ACTORS)])
            for i, actor in enumerate(ACTORS):
                if active[i]:
                    x = torch.as_tensor(obs[i:i+1]); h = actor.actor.net(x)
                    stds.extend(actor.actor.log_std_layer(h).clamp(-20, 2).exp().numpy().ravel().tolist())
        obs, _, _, done, info, _ = ENV.step(action)
        frame = info[0]; active_actions += int(active.sum())
        executed = np.asarray(frame["physical_action"])
        moving.extend((np.max(np.abs(executed[active]), axis=1) / ENV.speed).tolist())
        intents.extend(np.max(np.abs(np.asarray(frame["normalized_intent"])[active]), axis=1).tolist())
        blocked.append(frame["blocked_fraction"])
        if own is not None:
            error = float(np.max(np.abs(executed - physical)))
            max_interface_error = max(max_interface_error, error)
            if error > 2e-7: raise RuntimeError("Legal reference action changed by V execution interface")
        if all(done): break
    return {"method": method, "seed": seed, "stage": stage, "split": spec["split"],
            "group": spec["group"], "world": spec["world"], "episode_seed": spec["episode_seed"],
            "initial_state_and_rng_sha256": initial_hash, "episode": frame["episode"],
            "active_actuator_steps": active_actions, "physical_step_axis_mean_fraction_of_speed": float(np.mean(moving)),
            "physical_near_stop_fraction_below_005_speed": float(np.mean(np.array(moving) < .05)),
            "normalized_intent_axis_mean": float(np.mean(intents)),
            "mean_blocked_fraction": float(np.mean(blocked)),
            "latent_std_mean_visited": float(np.mean(stds)) if stds else None,
            "root_actor_mean_actions": root_mean_actions, "root_latent_stds": root_stds,
            "simulations": simulations, "simulated_transitions": simulated_steps,
            "reference_execution_max_error": max_interface_error}


def main():
    reg = verify(); OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "episodes.jsonl").exists(), "Refuse duplicate diagnostic sampling"
    cfg = read(SOURCE / "seed_1/requested_config.json")
    train = [r for r in reg["entries"] if r["split"] == "train"]
    chosen = [r for family in ("long_wall", "wall_corner", "multi_route")
              for r in [z for z in train if z["family"] == family][:2]]
    chosen += [r for r in reg["entries"] if r["split"] == "validation"]
    specs = []
    for row in chosen:
        for world in ("AB", "BA"):
            episode_seed = int.from_bytes(hashlib.sha256(f"V/{row['split']}/{row['id']}/{world}/0".encode()).digest()[:4], "little") % (2**31 - 2) + 1
            specs.append({"path": row["path"], "group": row["id"], "split": row["split"],
                          "world": world, "episode_seed": episode_seed})
    jobs = []
    for seed in (1, 2):
        directory = SOURCE / f"seed_{seed}"
        for stage in ("initial", "selected", "terminal"):
            if stage == "terminal":
                weights = [torch.load(directory / f"checkpoints/1010000/actor_agent{i}.pt", map_location="cpu", weights_only=False) for i in range(2)]
            else:
                weights = torch.load(directory / ("initial_actors.pt" if stage == "initial" else "selected_actors.pt"), weights_only=False)
            for method in ("deterministic", "stochastic"):
                for spec in specs: jobs.append((method, seed, stage, spec, weights))
    for method in ("stop", "uniform", "GEOM_SEARCH", "SINGLE_DIRECT_17"):
        for spec in specs: jobs.append((method, 0, "reference", spec, None))
    write(OUT / "protocol.json", {"scope": "post-hoc matched train/validation frozen-policy diagnosis; no final test reused",
          "groups": 12, "specs": specs, "episodes": len(jobs), "optimizer_updates": 0,
          "selection": "first two training groups per family plus all six validation groups; no outcome selection",
          "stochastic_action_seed": "same within model seed, case, world across checkpoint stages",
          "GEOM": "public observation only, 200ms joint search per decision; resource-unequal positive reference, not fairness/algorithm contrast",
          "SINGLE": "fixed previously trained single_17, original <=64 legal rejection execution; not an RL one-intent interface equivalence test",
          "protected_experiment_sha256": hashlib.sha256((SOURCE / "experiment.json").read_bytes()).hexdigest(),
          "single_model_sha256": hashlib.sha256(SINGLE.read_bytes()).hexdigest()})
    started = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=20, mp_context=mp.get_context("spawn"), initializer=init, initargs=(cfg,)) as pool:
        for row in pool.map(episode, jobs):
            rows.append(row)
            with (OUT / "episodes.jsonl").open("a") as f: f.write(json.dumps(row) + "\n")
            if len(rows) % 24 == 0: print("completed", len(rows), "/", len(jobs), flush=True)
    grouped = {}
    for row in rows:
        key = f"{row['method']}/{row['seed']}/{row['stage']}/{row['split']}"
        grouped.setdefault(key, []).append(row)
    summaries = {}
    for key, values in grouped.items():
        summaries[key] = {"episodes": len(values), "double": sum(v["episode"]["double_capture"] for v in values),
              "first": sum(v["episode"]["first_capture"] is not None for v in values),
              "utility": float(np.mean([v["episode"]["utility"] for v in values])),
              "physical_speed_fraction": float(np.mean([v["physical_step_axis_mean_fraction_of_speed"] for v in values])),
              "near_stop_fraction": float(np.mean([v["physical_near_stop_fraction_below_005_speed"] for v in values])),
              "blocked_fraction": float(np.mean([v["mean_blocked_fraction"] for v in values])),
              "latent_std": float(np.mean([v["latent_std_mean_visited"] for v in values])) if values[0]["latent_std_mean_visited"] is not None else None}
    signatures = {}
    for row in rows: signatures.setdefault((row["group"], row["world"]), set()).add(row["initial_state_and_rng_sha256"])
    assert all(len(v) == 1 for v in signatures.values()), "Controllers changed actual fixed-D initialization"
    result = {"complete": True, "episodes": len(rows), "physical_transitions": sum(v["episode"]["decisions"] for v in rows),
              "simulated_transitions": sum(v["simulated_transitions"] for v in rows), "optimizer_updates": 0,
              "matched_initial_state_and_opponent_rng_verified": True,
              "elapsed_seconds": time.time() - started, "summaries": summaries}
    write(OUT / "summary.json", result); verify()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
