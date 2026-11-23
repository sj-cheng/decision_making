"""Summarize completed shared starts, pairing methods within each opening."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from run_1000_cruise import ARMS, COUNT, DEST, H80, save


def summarize():
    manifest = json.loads((DEST/'manifest.json').read_text())
    rows = json.loads((H80/'episodes.json').read_text())
    cases = {c['case']:c for c in manifest['cases']}
    paired = defaultdict(dict)
    for row in rows:
        key, mode = row['case'], row['mode']
        assert cases[key]['seed'] == row['seed'] and mode not in paired[key]
        paired[key][mode] = row
    assert len(paired) == COUNT and all(set(group)==set(ARMS) for group in paired.values())
    summary = {}
    for m in ARMS:
        arm = [group[m] for group in paired.values()]
        summary[m] = dict(n=len(arm), double_by_seconds={
            h:sum(r['captures']==2 and r['double']<=h for r in arm) for h in (60,80)},
            capped_mean_seconds=mean(r['double'] for r in arm),
            collisions=sum(r['collisions'] for r in arm))
    contrasts = {}
    for baseline, variant in [('none','nar'),('none','nae'),('none','both'),('nar','both')]:
        diff = [g[baseline]['double']-g[variant]['double'] for g in paired.values()]
        contrasts[f'{baseline}_to_{variant}'] = dict(seconds_saved=mean(diff),
            faster=sum(v>0 for v in diff),equal=sum(v==0 for v in diff),slower=sum(v<0 for v in diff))
    save(H80/'summary.json',dict(arms=summary,paired_comparisons=contrasts))


if __name__=='__main__':
    summarize()
