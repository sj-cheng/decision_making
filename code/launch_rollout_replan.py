"""Warm both immutable numerical paths under the same timing lock, then run."""
import sys,time
from pathlib import Path
import run_rollout_replan_experiment as experiment

original_preflight=experiment.preflight


def process_warm_preflight():
    started=time.perf_counter()
    experiment.warmup()
    entry=experiment.development_entries()[0]
    packet=experiment.inputs(dict(stage='development',entry=entry))[0]
    own,_=experiment.initialize(packet,'T+Geo-R',17)
    own.act(experiment.observation(packet['history'][-1]),fixed_simulations=2)
    experiment.save(experiment.OUT/'pipeline_warmup.json',dict(seconds=time.perf_counter()-started,
        scope='per-process JIT/cache loading, under serial lock, before formal games',
        source_sha256=experiment.sha(__file__),future_confirmation_observations_used=False))
    return original_preflight()


if __name__=='__main__':
    experiment.preflight=process_warm_preflight
    sys.argv=[sys.argv[0],'pipeline']
    experiment.main()
