"""Summarize E58 paired closed-loop outcomes and measured planner latency."""
import json
import time
from pathlib import Path

import numpy as np
from network_search_positions import BUDGET, DEV, OUT
from run_1000_cruise import save


def summary(rows):
    latency=np.concatenate([r['team_ms'] for r in rows])
    return dict(double_capture=sum(r['captures']==2 for r in rows),
                mean_capped_time=float(np.mean([r['double'] for r in rows])),
                team_p95_ms=float(np.percentile(latency,95)),
                search_p95_ms=float(np.percentile(np.concatenate([r['search_ms'] for r in rows]),95)),
                collisions=sum(r['collisions'] for r in rows))


if __name__=='__main__':
    cases=[c['case'] for c in DEV]
    comparisons=[('first3','base64'),('rank3','first3'),('rank3','base64'),
                 ('set64','base64'),('value64','set64'),('value32','set32'),
                 ('value32','base128'),('set32','base128')]
    while True:
        groups={}
        for arm in BUDGET:
            groups[arm]={}
            for case in cases:
                path=OUT/'episodes'/arm/case/'result.json'
                if path.exists():
                    groups[arm][case]=json.loads(path.read_text())
        pairs=[]
        for arm,control in comparisons:
            common=groups[arm].keys() & groups[control].keys()
            if not common:continue
            x=[groups[arm][k] for k in sorted(common)]
            y=[groups[control][k] for k in sorted(common)]
            a,b=summary(x),summary(y)
            pairs.append(dict(arm=arm,control=control,paired=len(x),experiment=a,baseline=b,
                         wins=sum(i['captures']==2 and j['captures']!=2 for i,j in zip(x,y)),
                         losses=sum(j['captures']==2 and i['captures']!=2 for i,j in zip(x,y)),
                         team_p95_reduction=1-a['team_p95_ms']/b['team_p95_ms']))
        counts={arm:len(rows) for arm,rows in groups.items()}
        save(OUT/'comparison.json',dict(counts=counts,comparisons=pairs,development_cases=len(cases)))
        if min(counts.values())==len(cases):break
        time.sleep(60)
