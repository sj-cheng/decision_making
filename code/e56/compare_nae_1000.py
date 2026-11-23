"""Compare E52 frozen and branch evaluators at matched NAR settings."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import os
import tempfile
from pathlib import Path
import json
from run_1000_cruise import DEST, H80, LIMIT, cached_result, MODEL, config, outcome, run, save
from run_e55_parallel import FROZEN, BRANCH

def job(task, root=H80):
    case, mode, branch = task
    arm = mode + ('_branch' if branch else '_old_frozen')
    folder = root/'results'/arm/case['case']
    path = folder/'result.json'
    cached = cached_result(case,arm,path)
    if cached is not None:
        return cached
    args = dict(config(2), cycles=LIMIT, environment_horizon_floor=config(2)['cycles'], e52_mode=mode, e52_checkpoint=str(MODEL if branch else FROZEN),
                e52_advantage_checkpoint=str(BRANCH) if branch else None,
                e52_nae_strategy='budget_leaf_residual', evader_cruise=True)
    # For the both arms the NAR weights must match. Load the old value head
    # together with the current NAR through the existing checkpoint schema.
    if mode=='both' and not branch:
        args['e52_checkpoint']=str(DEST/'new_nar_old_value.pt')
    x=run(case['path'],folder/'trace.json',seed=case['seed'],**args)
    assert x['terminal_opponent_skips']==0
    row=dict(case=case['case'],mode=arm,seed=case['seed'],**outcome(x))
    save(path,row)
    return row

def supplement(task):
    case, mode, branch = task
    arm = mode + ('_branch' if branch else '_old_frozen')
    relative = Path('results')/arm/case['case']
    destination = H80/relative
    if destination.exists():
        return
    with tempfile.TemporaryDirectory(dir=H80) as stage:
        job(task, Path(stage))
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            (Path(stage)/relative).rename(destination)
        except OSError:
            if not destination.exists():
                raise


if __name__=='__main__':
    from solvers.e52_nar_nae import NARNAE
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=min(92,len(os.sched_getaffinity(0))))
    parser.add_argument('--supplement', action='store_true')
    args = parser.parse_args()
    if not args.supplement:
        fresh=NARNAE.load(MODEL)
        fresh.nae.load_state_dict(NARNAE.load(FROZEN).nae.state_dict())
        fresh.save(DEST/'new_nar_old_value.pt')
    cases=json.loads((H80/'paired_500.json').read_text())['cases']
    tasks=[(c,m,b) for c in cases for m in ('nae','both') for b in (False,True)]
    with ProcessPoolExecutor(args.workers) as pool:
        if args.supplement:
            for _ in pool.map(supplement, reversed(tasks)):
                pass
        else:
            save(H80/'nae_comparison.json',list(pool.map(job,tasks)))
