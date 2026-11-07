"""Posthoc cause diagnosis and reporting; never changes an estimator."""
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
from pathlib import Path
import json

import numpy as np

from audit_l45_rollout_reliability import observation,save
from estimator_audit_geometry import validate_paths
from solvers.estimator_particle_reference import ParticleEstimator
from solvers.multimodal_estimator import Estimator
from run_multimodal_estimator_validation import score,METHODS

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"current/l45_multimodal_estimator_20261025"
OLD=ROOT/"current/l45_occlusion_scenarios_20261025/bank"


def diagnose():
    entries=json.loads((OLD/"manifest.json").read_text())["entries"]
    records=[];motion=[];figures=[]
    for entry in entries:
        packet=json.loads((OLD/entry["public"]).read_text())
        environment=json.loads((OLD/entry["environment"]).read_text())
        hidden_start=max(i for i,x in enumerate(packet["history"]) if x["target"] is not None)
        for world in environment["worlds"]:
            states=world["history_states"]
            target=np.array([s["evader"] for s in states])
            velocities=np.diff(target,axis=0)[hidden_start:]
            direction=[]
            for a,b in zip(velocities,velocities[1:]):
                den=np.linalg.norm(a)*np.linalg.norm(b)
                if den>1e-9:direction.append(float(a@b/den))
            path=target[hidden_start:][None]
            observers=np.array([s["position"] for s in packet["history"][hidden_start+1:]])
            violations,steps=validate_paths(path,observers,np.array(packet["map"]))
            h=len(observers)
            motion.append(dict(group=entry["id"],world=world["id"],hidden_steps=h,
                true_path_violations=int(violations),transitions=int(steps),
                axis_speed_mean=float(np.max(np.abs(velocities),axis=1).mean()),
                direction_cosine_mean=float(np.mean(direction)),
                iid_axis_displacement_z=float(np.max(np.abs(target[-1]-target[hidden_start]))/np.sqrt(h/3.))))
        for stream,seed in enumerate(packet["controller_seeds"]):
            own={"particle":ParticleEstimator(packet["map"],seed),
                 "systematic":ParticleEstimator(packet["map"],seed,True),
                 "diffuse":Estimator(packet["map"],"diffuse",seed),
                 "persistent":Estimator(packet["map"],"persistent",seed),
                 "feasible":Estimator(packet["map"],"feasible",seed)}
            outputs=[]
            for item in packet["history"]:
                obs=observation(item)
                now={}
                for method,e in own.items():
                    samples,weights=e.update(obs)
                    now[method]=(samples,e.distribution(),e.last_stats.copy())
                outputs.append(now)
            # Truth is used here, after the complete public-only estimates.
            for world in environment["worlds"]:
                timeline=[]
                for i in range(hidden_start+1,len(packet["history"])):
                    truth=np.array(world["history_states"][i]["evader"])
                    row={"step":i,"methods":{}}
                    for method,(samples,(points,weights),stats) in outputs[i].items():
                        row["methods"][method]=score(points,weights,samples,truth)|stats
                    timeline.append(row)
                records.append(dict(group=entry["id"],stream=stream,world=world["id"],timeline=timeline))
            if stream==0:
                # Public ambiguity certificate: two distant endpoints from the
                # same feasible history, selected without target truth.
                e=own["feasible"];points,weights=e.distribution()
                a=points[0];b=points[np.argmax(np.linalg.norm(points-a,axis=1))]
                diameter=float(np.max(np.linalg.norm(points-b,axis=1)))
                for record in records[-2:]:
                    record['public_support_diameter_lower_bound']=diameter
                family=environment["family"]
                if family not in [x["family"] for x in figures]:
                    figures.append(dict(family=family,packet=packet,truth=environment["worlds"][0]["history_states"][-1]["evader"],
                        estimates={m:dict(particles=own[m].particles.copy(),distribution=own[m].distribution()) for m in METHODS}))
    summary=dict(prefix_groups=24,truth_paths=48,
        history_constraint_violations=sum(x["true_path_violations"] for x in motion),
        independent_true_transitions=sum(x["transitions"] for x in motion),
        directional_cosine_mean=float(np.mean([x["direction_cosine_mean"] for x in motion])),
        axis_speed_mean=float(np.mean([x["axis_speed_mean"] for x in motion])),
        iid_displacement_z_mean=float(np.mean([x["iid_axis_displacement_z"] for x in motion])),
        iid_displacement_z_range=[float(min(x["iid_axis_displacement_z"] for x in motion)),float(max(x["iid_axis_displacement_z"] for x in motion))],
        per_method={})
    for method in METHODS:
        last=[r["timeline"][-1]["methods"][method] for r in records]
        summary["per_method"][method]=dict(handoff_coverage=float(np.mean([x["sample_coverage"] for x in last])),
            handoff_mass1m=float(np.mean([x["mass_1m"] for x in last])),
            handoff_nll=float(np.mean([x["nll"] for x in last])),
            unique_ancestors_median=float(np.median([x["unique_ancestors"] for x in last])) if method in ("particle","systematic") else None,
            survivors_median=float(np.median([x["survivors"] for x in last])) if method in ("particle","systematic") else None,
            unique_particles_median=float(np.median([x["unique_particles"] for x in last])) if method in ("particle","systematic") else None)
    diam=[r['public_support_diameter_lower_bound'] for r in records if 'public_support_diameter_lower_bound' in r]
    summary['public_ambiguity_diameter_range']=[min(diam),max(diam)]
    save(OUT/"diagnosis_summary.json",summary)
    save(OUT/"diagnosis_prefix_rows.json",records)
    save(OUT/"diagnosis_true_motion_audit.json",motion)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(3,5,figsize=(17,10),sharex=True,sharey=True)
    for row,item in enumerate(figures):
        for col,method in enumerate(METHODS):
            ax=axes[row,col];estimate=item["estimates"][method]
            points,weights=estimate['distribution']
            ax.scatter(points[:,0],points[:,1],s=2+1500*weights,c=weights,cmap='Blues',alpha=.7)
            samples=estimate['particles'];ax.scatter(samples[:,0],samples[:,1],s=5,c='#117744',alpha=.6)
            for b in item['packet']['map']:ax.add_patch(Rectangle((b[0][0],b[1][0]),b[0][1]-b[0][0],b[1][1]-b[1][0],color='0.45'))
            ax.scatter(*item['truth'],marker='*',s=100,c='red',zorder=5)
            ax.set_title(f'{item["family"]}: {method}',fontsize=9);ax.set_aspect('equal');ax.set_xlim(-10,10);ax.set_ylim(-10,10)
    fig.suptitle('Fixed development handoff: weighted support (blue), 64 samples (green), truth for evaluation only (red)')
    fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True)
    fig.savefig(OUT/'figures/estimator_handoff_comparison.png',dpi=160)
    fig.savefig(OUT/'figures/estimator_handoff_comparison.pdf');plt.close(fig)
    return summary


if __name__=="__main__":
    print(json.dumps(diagnose(),ensure_ascii=False,indent=2))
