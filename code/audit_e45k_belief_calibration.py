"""Offline truth-vs-public-belief diagnostics; truth never enters a policy.

Replays all four K endpoint policies' original validation trajectories against
both opponents. Does not run controllers or alter the registered experiments.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
import run_e45k_main_task_learning as k

ROOT=k.ROOT
OUT=ROOT/'current/e45_marl_macro_audit_20261101'


def job(row):
    k.configure(OUT/'belief_replay')
    case=k.j.read(row['case']);payload=k.j.read(row['episode'])
    assert k.j.sha(row['episode'])==row['episode_sha256']
    belief=k.j.common.base.TeamBelief(np.asarray(case['map']),row['seed'])
    for obs in case['history'][:-1]:belief.update(obs)
    records=[]
    for step,t in enumerate(payload['trace']):
        obs=t['observation'];belief.update(obs)
        assert belief.signature()==t['belief_signature']
        truth=np.asarray(t['state'][:4]).reshape(2,2)
        for target,active in enumerate(obs['active_e']):
            if not active:continue
            est=belief.estimators[target];points=np.asarray(est.particles);weights=np.asarray(est.particle_weights)
            keep=weights>0
            if not keep.any():raise RuntimeError('No positive posterior weights')
            dist=np.linalg.norm(points[keep]-truth[target],axis=1)
            weight=weights[keep]/weights[keep].sum()
            visible=any(v is not None for v in obs['reports'][target])
            if visible:
                # A live sensor report is a point belief. The estimator may
                # deliberately defer its hidden-state graph initialization.
                full_dist=np.asarray([0.]);full_mass=np.asarray([1.])
            else:
                full_weights=np.asarray(est.weights).sum(axis=1)
                full_keep=full_weights>0
                full_points=np.asarray(est.graph[0])[full_keep]
                full_dist=np.linalg.norm(full_points-truth[target],axis=1)
                full_mass=full_weights[full_keep]/full_weights[full_keep].sum()
            records.append(dict(run=row['run'],key=row['key'],opponent=row['opponent'],target=target,
                                step=step,visible=visible,captures=row['captures'],
                                nearest=float(dist.min()),mean_distance=float(np.dot(dist,weight)),
                                mass_within1=float(weight[dist<=1].sum()),
                                full_support_nearest=float(full_dist.min()),
                                full_mass_within1=float(full_mass[full_dist<=1].sum()),
                                effective_particles=float(1/np.sum(weight**2))))
    return records


def aggregate(rows):
    if not rows:return dict(samples=0)
    d=np.asarray([r['nearest'] for r in rows])
    full=np.asarray([r['full_support_nearest'] for r in rows])
    return dict(samples=len(rows),nearest_median=float(np.median(d)),nearest_p90=float(np.quantile(d,.9)),
                nearest_over_half=float(np.mean(d>.5)),nearest_over_one=float(np.mean(d>1)),
                posterior_mean_distance=float(np.mean([r['mean_distance'] for r in rows])),
                posterior_mass_within_one=float(np.mean([r['mass_within1'] for r in rows])),
                full_support_nearest_median=float(np.median(full)),
                full_support_nearest_p90=float(np.quantile(full,.9)),
                full_support_over_one=float(np.mean(full>1)),
                full_posterior_mass_within_one=float(np.mean([r['full_mass_within1'] for r in rows])))


def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for f in sorted(k.DEFAULT_OUT.glob('runs/*')):
        history=k.j.read(f/'validation.json');endpoint=history[-1]
        for p in sorted((f/'validation/cache').glob('*.json')):
            row=k.j.read(p)
            if row['checkpoint_sha256']==endpoint['checkpoint_sha256']:rows.append(row|dict(run=f.name))
    assert len(rows)==192
    records=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        for i,result in enumerate(pool.map(job,rows)):
            records.extend(result)
            if (i+1)%24==0:print(f'Replayed {i+1}/{len(rows)} full original episodes',flush=True)
    groups={}
    for opp in ('rootpuct','fixed_D_kernel'):
        groups[opp]={}
        for vis in (False,True):
            rr=[r for r in records if r['opponent']==opp and r['visible']==vis]
            groups[opp]['visible' if vis else 'hidden']=aggregate(rr)
    data=dict(scope='post-hoc descriptive belief diagnostics on reused validation, not algorithm evaluation',
              original_episodes=len(rows),rows=len(records),groups=groups,
              source_sha256=k.j.sha(Path(__file__)),
              episodes={r['episode']:r['episode_sha256'] for r in rows})
    k.j.save(OUT/'belief_calibration.json',data)
    path=OUT/'belief_calibration_frames.json.gz'
    path.write_bytes(gzip.compress(json.dumps(records).encode(),mtime=0))
    print(json.dumps(groups,ensure_ascii=False,indent=2),flush=True)
    print('Saved',OUT,flush=True)


if __name__=='__main__':main()
