"""Frozen Single policy: one proposal + RL safe mapping, no learning."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import diagnose_e45v_control as d

OUT = d.ROOT / "current/e45v_single_interface_20261103"
COUNTS = None


def init(cfg):
    d.init(cfg)
    from solvers.e45b_joint_nte import Controller
    from learning.e45g_ppo import safe_action
    original = Controller.propose
    def one_proposal(self, node, direct=False):
        global COUNTS
        if not (direct and self.direct): return original(self, node, direct)
        self.allowed()
        normalized = self.runtime.sample(self.parameters(node), self.proposal_rng)
        normalized[~node.ap] = 0.
        physical, shortened = safe_action(node.positions, normalized, node.ap,
                                         self.public.speed, self.public.boxes)
        COUNTS["calls"] += 1; COUNTS["shortened_actuators"] += shortened
        # Same execution-only roundoff clipping as the V environment.
        return np.clip(physical, -self.public.speed, self.public.speed)
    Controller.propose = one_proposal


def episode(job):
    global COUNTS
    COUNTS = {"calls": 0, "shortened_actuators": 0}
    result = d.episode(job)
    result["interface"] = "one neural intent followed by V/HASAC target-free safe mapping"
    result["proposal_audit"] = dict(COUNTS)
    assert COUNTS["calls"] == result["episode"]["decisions"]
    assert result["simulations"] == 0
    return result


def main():
    d.verify(); OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "summary.json").exists()
    old = d.read(d.OUT / "protocol.json"); specs = old["specs"]
    cfg = d.read(d.SOURCE / "seed_1/requested_config.json")
    d.write(OUT / "protocol.json", {"scope": "post-hoc frozen Single_17 execution-interface diagnosis, training/validation only",
              "episodes": len(specs), "optimizer_updates": 0, "baseline": str(d.OUT / "episodes.jsonl"),
              "proposal_change": "<=64 accepted proposals -> one original neural draw then safe_action; all neural weights/public inputs unchanged",
              "no_RL_likelihood_or_finetuning_claim": True,
              "model_sha256": hashlib.sha256(d.SINGLE.read_bytes()).hexdigest()})
    jobs = [("SINGLE_DIRECT_17", 0, "reference", spec, None) for spec in specs]
    started = time.time()
    with ProcessPoolExecutor(max_workers=20, mp_context=mp.get_context("spawn"), initializer=init, initargs=(cfg,)) as pool:
        rows = list(pool.map(episode, jobs))
    prior = [json.loads(line) for line in (d.OUT / "episodes.jsonl").read_text().splitlines()]
    baseline = {(x["group"], x["world"]): x for x in prior if x["method"] == "SINGLE_DIRECT_17"}
    for row in rows:
        assert row["initial_state_and_rng_sha256"] == baseline[(row["group"], row["world"])]["initial_state_and_rng_sha256"]
    with (OUT / "episodes.jsonl").open("w") as file:
        for row in rows: file.write(json.dumps(row) + "\n")
    summary = {"complete": True, "episodes": len(rows), "physical_transitions": sum(x["episode"]["decisions"] for x in rows),
               "optimizer_updates": 0, "simulated_transitions": 0,
               "matched_initial_state_and_opponent_rng_verified": True,
               "original_interface_double": sum(x["episode"]["double_capture"] for x in baseline.values()),
               "one_intent_double": sum(x["episode"]["double_capture"] for x in rows),
               "one_intent_first": sum(x["episode"]["first_capture"] is not None for x in rows),
               "shortened_actuators": sum(x["proposal_audit"]["shortened_actuators"] for x in rows),
               "physical_speed_fraction": float(np.mean([x["physical_step_axis_mean_fraction_of_speed"] for x in rows])),
               "elapsed_seconds": time.time() - started,
               "split_double": {split: {"one_intent": sum(x["episode"]["double_capture"] for x in rows if x["split"] == split),
                                        "original": sum(x["episode"]["double_capture"] for x in baseline.values() if x["split"] == split)}
                                for split in ("train", "validation")}}
    d.write(OUT / "summary.json", summary); d.verify()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
