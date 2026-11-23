"""E56 paired 1,000-start cruise-evader evaluation."""
import argparse, hashlib, json, random, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"e55"))
from run_e55_parallel import OUT, config, run, save, outcome
MODEL = OUT / "trained/mlp_52.pt"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"e49"))
from e49_rules import _legal

DEST=OUT.parent/'e56_1000_cruise'
H80=DEST/'h80'
LIMIT=80
ARMS=('none','nar','nae','both')
COUNT=1000

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def make_manifest():
    src=json.loads((OUT/'manifest.json').read_text())
    templates=[Path(x) for x in src['cases']]
    rng=random.Random(56001)
    cases=[]
    for ti,t in enumerate(templates):
        data=json.loads(t.read_text()); boxes=np.asarray(data['map'],float).reshape(-1,2,2)
        for j in range(COUNT//len(templates)):
            for attempt in range(10000):
                points=[]
                while len(points)<4:
                    q=np.array([rng.uniform(-8.7,8.7),rng.uniform(-8.7,8.7)])
                    if _legal(q,boxes) and all(np.linalg.norm(q-p)>=2.0 for p in points): points.append(q)
                # first two are evaders, next two pursuers. Avoid trivial capture starts.
                if min(np.linalg.norm(points[i]-points[k]) for i in (0,1) for k in (2,3))<3.0: continue
                initial=np.asarray(data['worlds'][0]['initial'],float).copy()
                initial[:8]=np.asarray(points).reshape(-1)
                initial[8]=0.0
                initial[9:13]=1.0
                world=dict(data['worlds'][0],id=f'R{ti:02d}_{j:03d}',initial=initial.tolist())
                key=f'{ti:02d}_{j:03d}'
                p=DEST/'cases'/f'case_{key}.json'; save(p,dict(data,worlds=[world]))
                cases.append(dict(path=str(p),case=key,seed=560000+j,template=t.name))
                break
            else: raise RuntimeError(f'cannot sample {t}')
    assert len(cases)==COUNT
    save(DEST/'manifest.json',dict(cases=cases,arms=ARMS,count=COUNT,shared_initials=True,
      base=config(2),evader='cruise_hmcts',speed=1.5,seed=56001,
      template_hashes={p.name:digest(p) for p in templates}))
    return cases

def cached_result(case, mode, path):
    if path.exists():
        return json.loads(path.read_text())
    original = DEST/'results'/mode/case['case']/'result.json'
    if not original.exists():
        return None
    row = json.loads(original.read_text())
    assert row['case'] == case['case'] and row['seed'] == case['seed'] and row['mode'] == mode
    n = row['captures']
    collisions = row['collisions']
    if collisions and row['cycles'] > LIMIT:
        trace = json.loads(original.with_name('trace.json').read_text())
        collisions = sum(log['pursuer_collision'] for log in trace['logs'][:LIMIT])
    row.update(captures=int(n >= 1 and row['first'] <= LIMIT) + int(n == 2 and row['double'] <= LIMIT),
               first=min(LIMIT,row['first']), double=min(LIMIT,row['double']),
               cycles=min(LIMIT,row['cycles']), collisions=collisions, source='160s_prefix')
    save(path,row)
    return row

def evaluate(task):
    c,mode=task; p=Path(c['path']); dest=H80/'results'/mode/c['case']; rec=dest/'result.json'
    cached=cached_result(c,mode,rec)
    if cached is not None: return cached
    cfg=dict(config(2),cycles=LIMIT,environment_horizon_floor=config(2)['cycles'],e52_mode=mode,e52_checkpoint=str(MODEL),
      e52_nae_strategy='budget_leaf_residual',evader_cruise=True)
    x=run(p,dest/'trace.json',seed=c['seed'],**cfg)
    assert x['evader_cruise'] and x['terminal_opponent_skips']==0
    row=dict(case=c['case'],template=c['template'],seed=c['seed'],mode=mode,**outcome(x))
    save(rec,row); return row

def main():
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=64);a.add_argument('--start',action='store_true');a=a.parse_args()
    cases=make_manifest() if a.start or not (DEST/'manifest.json').exists() else json.loads((DEST/'manifest.json').read_text())['cases']
    tasks=[(c,m) for c in cases for m in ARMS]; rows=[]; failures=[]
    with ProcessPoolExecutor(a.workers) as pool:
        fs={pool.submit(evaluate,t):t for t in tasks}
        for f in as_completed(fs):
            try: rows.append(f.result())
            except Exception: failures.append((fs[f],traceback.format_exc()))
            print(f'complete {len(rows)}/{len(tasks)} errors {len(failures)}',flush=True)
    if failures: raise RuntimeError(failures[0][1])
    save(H80/'episodes.json',rows)

if __name__=='__main__': main()
