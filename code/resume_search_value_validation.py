"""Resume immutable cached experiment after a client interruption."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import fcntl
import json
import time
from pathlib import Path
import run_search_value_validation as experiment


def main():
    output=experiment.OUT
    with (experiment.ROOT/'current/.l45_search_value_timing.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        experiment.freeze()
        cached=sorted((output/'development/cache').glob('*.json'))
        stamp=time.perf_counter()
        sample=experiment.tasks('development')[0]
        packet=json.loads(Path(sample['packet_path']).read_text())
        for method in experiment.METHODS:
            actor,_=experiment.initialize(packet,method,sample['seed'])
            actor.act(experiment.observation(packet['history'][-1]),fixed_simulations=1)
        record=dict(reason='client interrupted active turn; process ended',
            complete_cached_games=len(cached),cached_hashes={str(p):experiment.sha(p) for p in cached},
            uncommitted_game_restarted=True,parameters_changed=False,
            formal_confirmation_games_before_resume=len(list((output/'confirmation').glob('episodes/*.json'))),
            warmup_seconds=time.perf_counter()-stamp,excluded_from_timed_decisions=True,
            launcher_sha256=experiment.sha(__file__))
        target=output/'resume_record.json'
        if target.exists():
            raise RuntimeError('Do not overwrite an earlier resume record')
        experiment.save(target,record)
        experiment.pipeline()


if __name__=='__main__':main()
