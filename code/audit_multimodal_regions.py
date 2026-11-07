"""Known-map region retention audit; no hidden-world truth is loaded."""
from pathlib import Path
import json
import numpy as np
from audit_l45_rollout_reliability import observation,save
from solvers.multimodal_estimator import Estimator

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"current/l45_multimodal_estimator_20261025"


def regions(points,boxes,weights):
    labels=[]
    for q in points:
        label=[]
        for b in boxes:
            for axis in range(2):
                label.append(-1 if q[axis]<b[axis,0] else 1 if q[axis]>b[axis,1] else 0)
        labels.append(tuple(label))
    mass={}
    for label,w in zip(labels,weights):
        mass[label]=mass.get(label,0.)+float(w)
    return mass


def run():
    rows=[]
    for split,bank in [('development',ROOT/'current/l45_occlusion_scenarios_20261025/bank'),('confirmation',OUT/'bank')]:
        entries=json.loads((bank/'manifest.json').read_text())['entries']
        for entry in entries:
            packet=json.loads((bank/entry['public']).read_text())
            estimates={m:Estimator(packet['map'],m,packet['controller_seeds'][0]) for m in ('persistent','feasible')}
            for item in packet['history']:
                for e in estimates.values():e.update(observation(item))
            boxes=np.array(packet['map'])
            persistent,feasible=estimates['persistent'],estimates['feasible']
            full=regions(*[persistent.distribution()[0],boxes,persistent.distribution()[1]])
            reference=regions(*[feasible.distribution()[0],boxes,feasible.distribution()[1]])
            exported=regions(persistent.particles,boxes,persistent.particle_weights)
            important={k for k,v in full.items() if v>=.01}
            rows.append(dict(split=split,group=entry['id'],feasible_regions=len(reference),
                positive_weight_regions=len(full),exported_regions=len(exported),
                same_full_region_support=set(full)==set(reference),
                important_regions=len(important),exported_important_regions=len(important&set(exported)),
                full_mass_missing_from_exported_regions=sum(v for k,v in full.items() if k not in exported)))
    summary={}
    for split in ('development','confirmation'):
        rs=[r for r in rows if r['split']==split]
        summary[split]=dict(groups=len(rs),same_full_region_support=all(r['same_full_region_support'] for r in rs),
            min_regions=min(r['feasible_regions'] for r in rs),max_regions=max(r['feasible_regions'] for r in rs),
            important_regions_retained=sum(r['exported_important_regions'] for r in rs),
            important_regions_total=sum(r['important_regions'] for r in rs),
            maximum_missing_mass=max(r['full_mass_missing_from_exported_regions'] for r in rs))
    save(OUT/'region_audit.json',dict(definition='joint known-map rectangle side/end halfspace regions; posterior mass>=1% is important; seed0 per public history',no_hidden_inputs=True,rows=rows,summary=summary))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
