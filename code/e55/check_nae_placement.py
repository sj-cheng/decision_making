"""Paired full games and one-action interventions on E55 method 10."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from statistics import mean
from run_e55_parallel import OUT, Shadow, config, outcome, run, save

DEST = OUT / 'budget_leaf_m10'
METHOD = 10


def job(case):
    folder = DEST / Path(case).stem
    cfg = dict(config(METHOD), e52_nae_strategy='budget_leaf_residual')
    original = json.loads((OUT/'diagnosis/m10'/(Path(case).stem+'_17')/'reference.json').read_text())
    assert original['case']==case and original['e52_budget']==256 and original['e52_depth']==8
    hook = Shadow(METHOD, False)
    trace = run(case, folder/'reference.json', seed=17, decision_hook=hook, audit_nae=True, **cfg)
    base, new = outcome(original), outcome(trace)
    row = dict(case=Path(case).stem, original=base, budget_leaf=new, first_flip=hook.flip,
               decisions=len(hook.rows)*2)
    for k in ('queried','rank_changed','executed_plan_changed'):
        row[k]=sum(sum(r[k]) for r in hook.rows)
    save(folder/'decisions.json',hook.rows)
    if hook.flip is not None:
        control=Shadow(METHOD,True)
        y=run(case,folder/'one_action.json',seed=17,decision_hook=control,**cfg)
        assert control.flip==hook.flip
        for a,b in zip(y['logs'][:hook.flip],trace['logs'][:hook.flip]):
            assert a['pursuers']==b['pursuers'] and a['targets']==b['targets']
        row['seconds_saved_first_flip']=outcome(y)['double']-new['double']
    save(folder/'result.json',row)
    return row


if __name__=='__main__':
    cases=json.loads((OUT/'manifest.json').read_text())['diagnostic_cases']
    rows=[]
    with ProcessPoolExecutor(6) as pool:
        for f in as_completed([pool.submit(job,c) for c in cases]):
            row=f.result();rows.append(row);print(row,flush=True)
    save(DEST/'summary.json',dict(rows=rows,seconds_saved_mean=mean(
        r['original']['double']-r['budget_leaf']['double'] for r in rows)))
