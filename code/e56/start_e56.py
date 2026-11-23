"""E56 paired difficulty calibration and closed-loop network screening."""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import traceback

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import repo_paths
from run_e55_parallel import OUT, config, run, save, outcome

DEST = OUT.parent / 'e56_network_screen'
MODEL = OUT / 'trained/mlp_52.pt'
SPEEDS = (1.5, 1.75, 2.)
MODES = ('none', 'nar', 'nae', 'both')


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def setup():
    src = json.loads((OUT / 'manifest.json').read_text())
    rng = random.Random(56)
    cases = []
    for family in ('e48', 'e49'):
        pool = [Path(p) for p in src['cases'] if Path(p).name.startswith(family)]
        for i, p in enumerate(rng.sample(sorted(pool), 8)):
            cases.append(dict(path=str(p), seed=(17,43)[i % 2], sha=digest(p)))
    used = {json.dumps(json.loads(Path(p).read_text())['map'],sort_keys=True)
            for paths in [src['cases'], *src['train'].values()] for p in paths}
    test = []
    bank = OUT.parent / 'e49_persistent_pursuit_benchmark_20261108/bank/test'
    pool = sorted(bank.glob('*.json'))
    rng.shuffle(pool)
    for p in pool:
        data = json.loads(p.read_text())
        layout = json.dumps(data['map'], sort_keys=True)
        if layout not in used and max(map(abs,data['worlds'][0]['initial'][:8])) <= 9:
            test.append(str(p))
            used.add(layout)
    assert len(test) >= 50
    sources = [Path(__file__), CODE/'e50/run_e50_distributed_dynamic_smoke.py',
               CODE/'e55/run_e55_parallel.py', CODE/'e49/e49_rules.py',
               *sorted((CODE/'solvers').glob('e5*.py'))]
    manifest = dict(cases=cases, holdout=test[:50], model=digest(MODEL),
                    sources={p.name:digest(p) for p in sources}, base=config(2))
    key = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
    folder = DEST / key
    save(folder/'manifest.json',manifest)
    return cases, folder


def evaluate(task):
    case, speed, mode, budget, folder = task
    p = Path(case['path'])
    dest = folder / f'{speed:.2f}_{mode}_{budget}' / f"{p.stem}_{case['seed']}"
    record = dest/'result.json'
    if record.exists():
        return json.loads(record.read_text())
    cfg = dict(config(2), e52_mode=mode, e52_budget=budget, shared_max_speed=speed,
               e52_checkpoint=str(MODEL) if mode!='none' else None,
               e52_nae_strategy='budget_leaf_residual')
    trace = run(p,dest/'trace.json',seed=case['seed'],**cfg)
    assert trace['terminal_opponent_skips']==0
    for event in trace['captures']:
        assert event['cycle'] <= event['time'] <= event['cycle']+1 <= 160
    searches=[s for row in trace['logs'] for s in row['search'].values()]
    assert all(s['nar_enabled']==(mode in ('nar','both')) and
               s['nae_enabled']==(mode in ('nae','both')) for s in searches)
    row=dict(case=p.stem,seed=case['seed'],speed=speed,mode=mode,budget=budget,
             **outcome(trace))
    save(record,row)
    return row


def batch(cases, arms, folder, workers):
    tasks=[(case,*arm,folder) for case in cases for arm in arms]
    rows=[]
    failures=[]
    with ProcessPoolExecutor(workers) as pool:
        futures={pool.submit(evaluate,t):t for t in tasks}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception:
                failures.append(str(futures[future])+traceback.format_exc())
                print(failures[-1],flush=True)
            print(f'complete {len(rows)}/{len(tasks)}, errors {len(failures)}',flush=True)
    if failures:
        raise RuntimeError(failures[0])
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=48)
    args=parser.parse_args()
    cases, folder=setup()
    print(folder,flush=True)
    arms=[(s,'none',256) for s in SPEEDS]+[(1.5,m,256) for m in MODES[1:]]
    rows=batch(cases,arms,folder,args.workers)
    save(folder/'screen.json',rows)
    rates={s:statistics.mean(r['captures']==2 for r in rows if r['speed']==s and r['mode']=='none') for s in SPEEDS}
    eligible=[s for s in SPEEDS if .4<=rates[s]<=.8]
    save(folder/'calibration.json',dict(rates=rates,eligible=eligible))
    if eligible:
        speed=min(eligible,key=lambda s:(abs(rates[s]-.6),s))
        more=[(speed,'none',b) for b in (512,1024)]+[(speed,m,256) for m in MODES[1:]]
        save(folder/'hard_screen.json',batch(cases,more,folder,args.workers))


if __name__=='__main__':
    main()
