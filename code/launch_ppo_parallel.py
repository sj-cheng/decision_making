"""Launch the three E49 PPO-C seeds as isolated worker processes.

This launcher intentionally does not call ``adapt``: it only runs PPO-C and
leaves pilot/selection decisions to the normal adaptation protocol.  Workers
resume their existing seed directory transactionally through ``_ppo_train``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


SEEDS = (17, 29, 43)


def _cpu_set(text: str) -> set[int]:
    result: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = (int(x) for x in part.split("-", 1))
            if last < first:
                raise ValueError(f"invalid CPU range: {part}")
            result.update(range(first, last + 1))
        else:
            result.add(int(part))
    if not result:
        raise ValueError("empty CPU set")
    return result


def _worker(args: argparse.Namespace) -> int:
    # Set affinity before importing torch/SB3, and make the physical GPU
    # mapping explicit.  Every worker gets a fresh CUDA context.
    cpus = _cpu_set(args.cpus)
    os.sched_setaffinity(0, cpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, len(cpus))))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, len(cpus))))

    from e49_adaptation import _ppo_train
    from e49_controllers import install, method_specs

    out = Path(args.out)
    config = json.loads((out / "bank_manifest.json").read_text())
    frozen = out / "horizon_selection.json"
    horizon = int(json.loads(frozen.read_text())["horizon"])
    install(horizon, out)
    specs = [s for s in method_specs() if s["arm"] == "PPO_C" and s["train_seed"] == args.seed]
    if len(specs) != 1:
        raise RuntimeError(f"expected one PPO_C source for seed {args.seed}, got {len(specs)}")
    result = _ppo_train(
        out,
        args.seed,
        specs[0]["checkpoint"],
        [e for e in config["entries"] if e["split"] == "train"],
        horizon,
        args.target_steps,
        __import__("torch").device("cuda:0"),
    )
    print(json.dumps(dict(seed=args.seed, cpus=sorted(cpus), **result), sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, default=500_000)
    parser.add_argument(
        "--cpus",
        required=True,
        help="three semicolon-separated CPU sets, one per seeds 17,29,43; e.g. 0-3;4-7;8-11",
    )
    parser.add_argument("--worker", type=int, choices=SEEDS)
    args = parser.parse_args()
    sets = args.cpus.split(";")
    if args.worker is not None:
        return _worker(args)
    if len(sets) != 3:
        parser.error("--cpus must contain exactly three semicolon-separated sets")
    # Parent remains CPU-unpinned; each child is pinned before importing torch.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "3"
    procs = []
    for seed, cpus in zip(SEEDS, sets):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--out",
            str(args.out),
            "--target-steps",
            str(args.target_steps),
            "--cpus",
            cpus,
            "--worker",
            str(seed),
        ]
        procs.append(subprocess.Popen(cmd, env=env))
    statuses = [p.wait() for p in procs]
    return max(statuses, default=0)


if __name__ == "__main__":
    raise SystemExit(main())
