"""Outcome-blind, grouped task registration for E45-V."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "current/e45v_fullstate_hasac_20261103"
UPSTREAM = ROOT / "current/e45u_official_hasac_calibration_20261102/official_source"


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x): return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False))


def bank():
    manifest = OUT / "bank/manifest.json"
    if manifest.exists():
        cfg = read(manifest)
        for row in cfg["entries"]: assert sha(row["path"]) == row["sha256"]
        return cfg
    from e45b_scenarios import make_case
    known = set()
    for folder in ("e45b_role_free_joint_nte_20261027", "e45c_budget_scaling_20261028", "e45j_repaired_benchmark_20261031"):
        for p in (ROOT / "current" / folder / "bank").rglob("*.json"):
            old = read(p)
            if isinstance(old, dict) and "map" in old: known.add(canonical(old["map"]))
    entries, attempts, seen_maps, seen_histories = [], [], set(), set()
    for split, count in (("train", 8), ("validation", 2), ("test", 2)):
        for family in ("long_wall", "wall_corner", "multi_route"):
            accepted = 0
            for index in range(500):
                if accepted == count: break
                attempt = {"split": split, "family": family, "index": index}
                try:
                    case = make_case(family, index, split, 14520261103)
                    mh, hh = canonical(case["map"]), canonical(case["history"])
                    if mh in known or mh in seen_maps or hh in seen_histories:
                        raise ValueError("Exact geometry/public-history overlap")
                except (AssertionError, ValueError) as exc:
                    attempts.append(attempt | {"accepted": False, "reason": repr(exc)})
                    continue
                path = OUT / "bank" / split / (case["id"] + ".json")
                write(path, case); seen_maps.add(mh); seen_histories.add(hh)
                entries.append({"id": case["id"], "path": str(path), "split": split,
                                "family": family, "pattern": case["pattern"], "source_index": index,
                                "map_sha256": mh, "public_history_sha256": hh, "sha256": sha(path)})
                attempts.append(attempt | {"accepted": True, "id": case["id"]})
                accepted += 1
            assert accepted == count, (split, family, accepted)
    cfg = {"entries": entries, "attempts": attempts, "seed_base": 14520261103,
           "selection": "first geometrically admissible cases, no policy results",
           "grouping": "case/public-history; world and dynamics seeds are within-group repetitions",
           "old_exact_map_hashes_checked": len(known), "train_groups": 24,
           "validation_groups": 6, "test_groups": 6}
    write(manifest, cfg)
    return cfg


def freeze():
    cfg = bank()
    from learning.e45v_fullstate_env import OBS_DIM, CENT_DIM, ENCODING
    from learning.e45v_retirement import train_source_audit
    names = ["prepare_e45v_fullstate.py", "run_e45v_fullstate.py", "launch_e45v_fullstate.py",
             "learning/e45v_fullstate_env.py", "learning/e45v_retirement.py",
             "tests/test_e45v_fullstate_env.py", "tests/test_e45v_retirement.py",
             "E45V_FULLSTATE_PROTOCOL_20261103.md"]
    task_sources = ["problems/example8.py", "corner_estimator_scenarios.py", "e45b_scenarios.py",
                    "run_e45_team_pilot.py", "run_frozen_d_closed_loop.py", "solvers/e45_team_adapter.py",
                    "solvers/e45b_joint_nte.py", "solvers/frozen_d_search_interface.py",
                    "learning/e45g_ppo.py", "e45g_training_acceleration.py"]
    files = [ROOT / "code" / n for n in names + task_sources]
    hashes = {str(p): sha(p) for p in files}
    source_manifest = read(UPSTREAM.parent / "source_manifest.json")
    for name, row in source_manifest["files"].items(): assert sha(UPSTREAM / name) == row["sha256"]
    experiment = {"experiment": "E45-V", "scope": "privileged full-state moving fixed-D pursuit diagnostic; not public benchmark",
                  "seeds": [1, 2], "entries": cfg["entries"], "actor_obs_dim": OBS_DIM, "critic_obs_dim": CENT_DIM,
                  "observation_encoding": ENCODING, "opponent": "fixed_D_kernel",
                  "post_warmup_team_transitions_per_seed": 1000000, "warmup_team_transitions": 10000,
                  "rollout_envs": 20, "train_interval_vector_ticks": 50, "train_calls_per_burst": 50,
                  "train_calls_per_seed": 50000, "train_calls_per_new_team_transition": .05,
                  "batch": 1000, "replay_capacity": 1000000, "n_step": 20, "gamma": 1.,
                  "auto_alpha": True, "initial_effective_alpha": 1., "actor_lr": .0005, "critic_lr": .0005,
                  "use_proper_time_limits": False, "use_valuenorm": False,
                  "reward": "raw double-minus-time utility + current-true-state unassigned geometric/progress potential difference (terminal0) - .05 blocked fraction",
                  "validation": "6 new public-history groups x2 worlds x5 fixed dynamics seeds, utility strictly greater, tie earlier including step0",
                  "sealed_test": "6 further groups x2 worlds x10 dynamics seeds; initial/validation-selected/terminal only after training",
                  "evaluation_every_post_warmup_team_transitions": 100000,
                  "retirement_train_diff": train_source_audit(), "implementation_and_task_sha256": hashes,
                  "official_source_manifest_sha256": sha(UPSTREAM.parent / "source_manifest.json"),
                  "official_commit": source_manifest["commit"], "bank_manifest_sha256": sha(OUT / "bank/manifest.json"),
                  "budget_scope": "first diagnostic stage; weak outcome does not establish algorithm invalidity; no automatic extension",
                  "runtime": "new Py311 venv borrowing dm_env_py311 Torch2.10 NumPy2.2; task C++ ABI requires Py311; U used Py310 Torch2.7",
                  "automatic_public_observation_training": False}
    path = OUT / "experiment.json"
    if path.exists(): assert read(path) == experiment, "Frozen experiment differs"
    else:
        write(path, experiment)
        for p in files:
            dest = OUT / "source_snapshot" / p.relative_to(ROOT / "code")
            dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(p.read_bytes())
    print(json.dumps({"frozen": True, "source_files": len(hashes), "groups": len(cfg["entries"]),
                      "actor_obs": OBS_DIM, "critic_obs": CENT_DIM}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze: freeze()
    else:
        cfg = bank(); print(json.dumps({"groups": len(cfg["entries"]), "attempts": len(cfg["attempts"])}))
