"""Same-seed comparison of E56 pursuer ablations against the cruise evader."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from start_e56 import DEST, MODEL, config, outcome, run, save
from verify_cruise import BASE, OUT as CRUISE

OUT = DEST/'cruise_networks_150'


def evaluate(task):
    case, mode = task
    name = f"{Path(case['path']).stem}_{case['seed']}"
    folder = OUT/mode/name
    path = folder/'result.json'
    if path.exists():
        return json.loads(path.read_text())
    cfg = dict(config(2), e52_mode=mode, e52_checkpoint=str(MODEL),
               e52_nae_strategy='budget_leaf_residual', evader_cruise=True)
    data = run(case['path'], folder/'trace.json', seed=case['seed'], **cfg)
    assert data['terminal_opponent_skips']==0 and data['evader_cruise']
    assert all(0<=e['time']<=160 for e in data['captures'])
    for log in data['logs']:
        for record in log['search'].values():
            assert record['nar_enabled']==(mode in ('nar','both'))
            assert record['nae_enabled']==(mode in ('nae','both'))
    row = dict(case=name,mode=mode,**outcome(data))
    save(path,row)
    return row


if __name__=='__main__':
    cases=json.loads((BASE/'manifest.json').read_text())['cases']
    rows=[dict(case=r['case'],mode='none',**r['cruise']) for r in json.loads((CRUISE/'paired.json').read_text())]
    with ProcessPoolExecutor(48) as pool:
        jobs=[pool.submit(evaluate,(c,m)) for c in cases for m in ('nar','nae','both')]
        for job in as_completed(jobs):
            rows.append(job.result())
    assert len({(r['case'],r['mode']) for r in rows})==64
    save(OUT/'paired.json',rows)
