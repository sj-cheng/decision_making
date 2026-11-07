"""Exploratory depth-reach / equal-simulation check after the frozen X pilot.

Preserves the completed initial run. Adds a genuine frozen-policy rollout to
depth16 after a fresh node, and pairs the original five arms at exactly64 sims.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import copy
import time
import json
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import numpy as np
import torch

import run_e45x_event_diagnostic as base
from learning.e45x_event_diagnostic import (EventController,ARMS,seed,ETA,TIME_COST,
    simultaneous_capture,safe,target_step)
from learning.e45w_support_recovery import TeamBelief
from e45c_capture_audit import audit_simulated

OUT=base.ROOT/'current/e45x_first_capture_diagnostic_20261104'


class Rollout16(EventController):
    def trajectory(self):
        ids,modes=self.sample_world();node=self.root;path=[];rewards=[];trace=[];rolling=False
        self.active_trace=trace
        while node.depth<min(16,self.remaining):
            self.allowed()
            if rolling:
                # Rollout nodes are temporary public-belief states, not new
                # tree branches. Continue the same frozen neural proposal.
                node=copy.copy(node);node.children={};node.actions=[]
                action=self.propose(node,True);node.actions=[action];chosen=0
            else:
                chosen=self.select(node);action=node.actions[chosen]
                path.append((node,chosen,len(rewards)))
            before=np.zeros((2,2));targets=np.zeros((2,2))
            for j,g in enumerate(self.graphs):
                if not node.ae[j]:continue
                before[j]=g[0][ids[j]]
                ids[j],modes[j]=target_step(ids[j],modes[j],g[1],g[2],g[3],self.rng)
                targets[j]=g[0][ids[j]]
                if not safe(before[j],targets[j]-before[j],1.,self.public.boxes,np.zeros(2,np.int64)):
                    raise RuntimeError('Illegal rollout target action')
            positions=node.positions+action
            if not all(safe(node.positions[i],action[i],self.public.speed,self.motion_boxes,np.zeros(2,np.int64)) for i in range(2)):
                raise RuntimeError('Illegal rollout own action')
            ap,ae,visible=simultaneous_capture(positions,targets,node.ap,node.ae,self.public.boxes)
            observed=np.array([ids[j] if visible[j]>=0 else visible[j] for j in range(2)])
            trace.append(np.r_[node.positions.ravel(),positions.ravel(),before.ravel(),targets.ravel(),node.ap,node.ae,ap,ae,visible>=0])
            raw=float(not ae.any())-TIME_COST;previous=self.potential(node)
            if not ae.any() or node.depth+1>=self.remaining:
                rewards.append(raw-ETA*previous);break
            child,fresh=self.child(node,chosen,positions,ap,ae,observed)
            rewards.append(raw+ETA*(self.potential(child)-previous));node=child
            rolling=rolling or fresh
        self.allowed()
        for visited,chosen,offset in path:
            value=sum(rewards[offset:]);visited.n+=1;visited.visits[chosen]+=1;visited.totals[chosen]+=value
        self.traces.append(np.asarray(trace).reshape(-1,26));self.active_trace=None


def worker(job):
    directory=OUT/'followup';path=directory/'jobs'/f'{job["key"]}.json'
    if path.exists():return base.read(path)
    root=base.read(OUT/'public_roots'/f'{job["root_id"]}.json');case=base.read(root['case_path'])
    heads=base.LeafReadout(OUT/'leaf_readouts.npz');arms=list(ARMS)+(['rollout16'] if job['mode']=='wall' else [])
    np.random.default_rng(seed(job['key'],'order')).shuffle(arms);rows=[]
    for arm in arms:
        cls=Rollout16 if arm=='rollout16' else EventController
        own=cls(case['map'],case['speed'],seed(root['id'],job['repeat'],'planner'),actor=base.ACTOR,
            arm=arm if arm!='rollout16' else 'deeper16',heads=heads,candidates=np.asarray(root['candidates']),budget_ms=200.)
        own.belief=TeamBelief(case['map'],root['controller_seed'])
        for obs in root['history'][:-1]:own.observe(obs)
        torch.cuda.synchronize();start=time.perf_counter()
        action=own.act(root['history'][-1],fixed_simulations=64 if job['mode']=='fixed64' else None)
        torch.cuda.synchronize();latency=1000*(time.perf_counter()-start)
        distances=np.max(np.abs(np.asarray(root['candidates'])-action),axis=(1,2))
        chosen=int(np.argmin(distances))
        if distances[chosen]>1e-8:raise RuntimeError('Returned action not in fixed common pool')
        traces=np.concatenate(own.traces) if own.traces else np.zeros((0,26))
        errors=audit_simulated(traces,np.asarray(case['map']),case['speed'])
        if np.any(errors):raise RuntimeError('Independent followup trace audit failed')
        if job['mode']=='fixed64':assert own.last_stats['simulations']==64
        rows.append(dict(arm=arm,selected_candidate=chosen,wall_ms=latency,
            completed_simulations=own.last_stats['simulations'],
            max_simulation_depth=max([len(t) for t in own.traces],default=0),
            event_evaluations=own.last_stats['event_evaluations'],simulation_errors=errors.tolist()))
    result=job|dict(group=root['group'],opponent=root['opponent'],stratum=root['stratum'],arms=rows)
    base.write(path,result);return result


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise RuntimeError('Use physical GPU3 only')
    cfg=base.read(OUT/'experiment.json');base.verify(cfg)
    directory=OUT/'followup';directory.mkdir(exist_ok=True)
    roots=[r for r in base.read(OUT/'root_manifest.json')['roots'] if r['split']=='development']
    protocol=dict(source_sha256=base.sha(__file__),heads_sha256=base.sha(OUT/'leaf_readouts.npz'),
        root_ids=[r['id'] for r in roots],registered_before_followup=True,
        motivation='Initial event version did not improve; check actual depth and wall-budget confounding without tuning Actor/heads',
        modes=dict(wall='six arms incl genuine rollout16, 200ms, two repeats',fixed64='original five arms, exactly64 simulations, one repeat'),
        physical_gpu=3,analysis_scope='exploratory followup, same development roots; no new confirmation claim')
    base.write(directory/'protocol.json',protocol)
    jobs=[]
    for r in roots:
        for mode,repeats in [('wall',2),('fixed64',1)]:
            for repeat in range(repeats):jobs.append(dict(key=f'{r["id"]}_{mode}_{repeat}',root_id=r['id'],mode=mode,repeat=repeat))
    ctx=mp.get_context('spawn');queue=ctx.Queue()
    for cpu in range(48,52):queue.put(cpu)
    started=time.time();results=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=ctx,initializer=base.initializer,
                            initargs=(cfg,str(OUT),queue)) as pool:
        futures=[pool.submit(worker,j) for j in jobs]
        for f in as_completed(futures):
            results.append(f.result())
            base.write(directory/'status.json',dict(state='running',completed=len(results),planned=len(jobs),elapsed_seconds=time.time()-started))
    references={r['root_id']:r for r in base.read(OUT/'root_ranking_audit.json')['roots']}
    stats=[]
    for mode in ('wall','fixed64'):
        for opponent in ('fixed_D_kernel','rootpuct'):
            for arm in list(ARMS)+(['rollout16'] if mode=='wall' else []):
                rr=[(r,a) for r in results if r['mode']==mode and r['opponent']==opponent for a in r['arms'] if a['arm']==arm]
                stats.append(dict(mode=mode,opponent=opponent,arm=arm,decisions=len(rr),
                    mean_reference_utility=float(np.mean([references[r['root_id']]['reference_utilities'][a['selected_candidate']] for r,a in rr])),
                    max_observed_simulation_depth=max(a['max_simulation_depth'] for _,a in rr),
                    mean_max_depth=float(np.mean([a['max_simulation_depth'] for _,a in rr])),
                    decisions_reaching_depth8=sum(a['max_simulation_depth']>=8 for _,a in rr),
                    event_evaluations=sum(a['event_evaluations'] for _,a in rr),
                    mean_completed_simulations=float(np.mean([a['completed_simulations'] for _,a in rr])),
                    p95_ms=float(np.percentile([a['wall_ms'] for _,a in rr],95))))
    differences=[]
    for r in results:
        if r['mode']!='fixed64':continue
        byarm={a['arm']:a for a in r['arms']}
        event=byarm['event16'];specialized=byarm['specialized16']
        if event['event_evaluations']==0:
            differences.append(event['selected_candidate']!=specialized['selected_candidate'])
    assert not any(differences),'No-event paired fixed-simulation control must be identical'
    assert base.sha(__file__)==protocol['source_sha256']
    assert base.sha(OUT/'leaf_readouts.npz')==protocol['heads_sha256'];base.verify(cfg)
    base.write(directory/'results.json',dict(complete=True,statistics=stats,
        no_event_equal_simulation_identity_passed=True,followup_decision_jobs=len(results),
        elapsed_seconds=time.time()-started))
    base.write(directory/'status.json',dict(state='complete',completed=len(results),planned=len(jobs),elapsed_seconds=time.time()-started))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import traceback
        base.write(OUT/'followup/failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
