"""Paired E56 effects, with resampling by physical map across both opponents."""
import json
from pathlib import Path
from statistics import mean
import numpy as np
from compare_cruise_networks import BASE, OUT, save


def summarize():
    old_rows=json.loads((BASE/'screen.json').read_text())
    old={(r['case']+'_'+str(r['seed']),r['mode']):r for r in old_rows if r['speed']==1.5}
    new={(r['case'],r['mode']):r for r in json.loads((OUT/'paired.json').read_text())}
    assert new.keys()==old.keys()
    grouped={}
    for c in json.loads((BASE/'manifest.json').read_text())['cases']:
        p=Path(c['path'])
        map_id=json.dumps(json.loads(p.read_text())['map'],sort_keys=True)
        grouped.setdefault(map_id,[]).append(p.stem+'_'+str(c['seed']))
    groups=list(grouped.values())
    cases=sum(groups,[])
    rng=np.random.default_rng(56)
    case_ids=[[cases.index(k) for k in g] for g in groups]
    resamples=[sum((case_ids[i] for i in sample),[]) for sample in rng.integers(len(groups),size=(5000,len(groups)))]
    result={'maps':len(groups),'arms':{},'effects':{}}
    for name,data in [('original',old),('cruise',new)]:
        result['arms'][name]={}
        for m in ['none','nar','nae','both']:
            rs=[data[k,m] for k in cases]
            result['arms'][name][m]=dict(n=len(rs),counts={h:sum(r['captures']==2 and r['double']<=h for r in rs) for h in (60,80,160)},
                                         capped_mean=mean(r['double'] for r in rs),collisions=sum(r['collisions'] for r in rs))
    for m1,m2 in [('none','both'),('nar','both'),('none','nar'),('none','nae'),('nae','both')]:
        effects={}
        contrasts=[]
        for name,data in [('original',old),('cruise',new)]:
            seconds=np.array([data[k,m1]['double']-data[k,m2]['double'] for k in cases])
            contrasts.append(seconds)
            effects[name]=dict(seconds_saved=float(seconds.mean()),faster=int(sum(seconds>0)),
              tied=int(sum(seconds==0)),slower=int(sum(seconds<0)),
              ci95=np.percentile([seconds[ids].mean() for ids in resamples],[2.5,97.5]).tolist())
        diff=contrasts[1]-contrasts[0]
        effects['change_in_gap']=dict(seconds=float(diff.mean()),
            ci95=np.percentile([diff[ids].mean() for ids in resamples],[2.5,97.5]).tolist())
        result['effects'][m1+'_'+m2]=effects
    save(OUT/'summary.json',result)
    return result

if __name__=='__main__':
    print(json.dumps(summarize(),indent=2))
