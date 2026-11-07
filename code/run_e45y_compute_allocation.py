"""State-level feasibility of public-decision-sensitive compute allocation.

Nested search prefixes use independent planning randomness and are evaluated
with X's unchanged fixed-policy reference worlds. This is trace-replay/batch
budget evidence, not a deployed full-episode scheduler or a novelty claim.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import argparse
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import fcntl
import numpy as np
import torch
from numba import njit

import run_e45x_event_diagnostic as x
from learning.e45x_event_diagnostic import (EventController,public_node,seed,ETA,TIME_COST,
    simultaneous_capture,target_step)
from learning.e45w_support_recovery import TeamBelief
from solvers.e45b_joint_nte import BudgetAbort
from solvers.frozen_d_search_interface import route
from e45c_capture_audit import audit_simulated

ROOT=x.ROOT
X=ROOT/'current/e45x_first_capture_diagnostic_20261104'
OUT=ROOT/'current/e45y_compute_allocation_20261104'
LOW=50.;HIGH=200.


def entropy_positions(samples,weights):
    """Merge duplicate spatial exports; a known point has zero entropy."""
    _,inverse=np.unique(np.asarray(samples),axis=0,return_inverse=True)
    mass=np.bincount(inverse,weights=np.asarray(weights))
    mass=mass[mass>0];mass=mass/mass.sum()
    return float(-np.sum(mass*np.log(mass)))


def decision_scores(matrix,bootstrap_seed):
    matrix=np.asarray(matrix,float);means=matrix.mean(0);winner=int(means.argmax())
    rng=np.random.default_rng(bootstrap_seed)
    indices=rng.integers(len(matrix),size=(128,len(matrix)))
    estimates=matrix[indices].mean(1)
    regret=np.maximum(0.,estimates.max(1)-estimates[:,winner])
    return dict(decision_score=float(regret.mean()),
        sampling_rank_disagreement=float(np.mean(estimates.argmax(1)!=winner)),
        world_rank_disagreement=float(1.-np.bincount(matrix.argmax(1),minlength=matrix.shape[1]).max()/len(matrix)),
        score_means=means.tolist())


@njit(cache=True)
def world_scores(positions,ap,ae,worlds,actions,speed,vertices,dist,boxes):
    result=np.zeros((len(worlds),len(actions)))
    for w in range(len(worlds)):
        for a in range(len(actions)):
            nxt=positions+actions[a]
            pp,ee,_=simultaneous_capture(nxt,worlds[w],ap,ae,boxes)
            score=(1. if not ee.any() else 0.)-TIME_COST
            distance=0.
            for j in range(2):
                if not ee[j]:continue
                best=80.
                for i in range(2):
                    if pp[i]:
                        steps,_=route(nxt[i],worlds[w,j],speed,vertices,dist,boxes,np.zeros(2,np.int64))
                        best=min(best,float(steps))
                distance+=best
            result[w,a]=score-ETA*distance
    return result


def probe(own,root,repeat):
    obs=root['history'][-1];start=time.perf_counter()
    entropy=0.
    for j,e in enumerate(own.belief.estimators):
        if obs['active_e'][j]:entropy+=entropy_positions(e.particles,e.particle_weights)
    entropy_ms=1000*(time.perf_counter()-start)
    rng=np.random.default_rng(seed('Y-probe',root['id'],repeat));worlds=np.zeros((16,2,2))
    for n in range(16):
        for j,g in enumerate(own.graphs):
            weights=own.root.joints[j]
            k=int(rng.choice(weights.size,p=weights.ravel()/weights.sum()));i,m=divmod(k,9)
            local=np.array([seed('Y-probe-motion',root['id'],repeat,n,j)],np.int64)
            i,m=target_step(i,m,g[1],g[2],g[3],local);worlds[n,j]=g[0][i]
    p=own.public
    matrix=world_scores(np.asarray(obs['positions']),np.asarray(obs['active_p']),np.asarray(obs['active_e']),
        worlds,np.asarray(root['candidates']),p.speed,p.vertices,p.dist,p.boxes)
    result=decision_scores(matrix,seed('Y-bootstrap',root['id'],repeat))
    result.update(entropy=entropy,entropy_ms=entropy_ms,signal_ms=1000*(time.perf_counter()-start))
    return result


def selected(visits,stop_index):
    return int(np.argmax(visits)) if np.sum(visits)>0 else int(stop_index)


def query(record,budget_ms,overhead_ms=0.):
    limit=max(0.,budget_ms-overhead_ms)
    eligible=[s for s in record['snapshots'] if s['at_ms']<=limit+1e-9]
    if not eligible:return record['stop_index']
    return eligible[-1]['selected_candidate']


def threshold_mask(scores,cutoff):return np.asarray(scores)>cutoff


def top_k_mask(scores,k):
    result=np.zeros(len(scores),bool)
    if k:
        order=np.lexsort((np.arange(len(scores)),-np.asarray(scores)))
        result[order[:k]]=True
    return result


def initializer(cfg,output,queue):
    x.initializer(cfg,str(output),queue)
    row=cfg['source_maps'][0];case=x.read(row['path'])
    root=dict(id='Y-engineering-warmup',controller_seed=seed('Y-warm'),history=case['history'])
    prepared=x.prepare_public(root,{k:case[k] for k in ('map','speed')},x.ACTOR)
    actions,_=x.root_candidates(root,prepared,x.ACTOR);root['candidates']=actions.tolist()
    probe(prepared.controller,root,0)
    own=EventController(case['map'],case['speed'],seed('Y-warm'),actor=x.ACTOR,arm='original8',candidates=actions)
    for obs in case['history'][:-1]:own.observe(obs)
    own.act(case['history'][-1],fixed_simulations=2)


def worker(job):
    path=Path(job['output'])/'jobs'/f'{job["key"]}.json'
    if path.exists():return x.read(path)
    root=x.read(X/'public_roots'/f'{job["root_id"]}.json');case=x.read(root['case_path'])
    own=EventController(case['map'],case['speed'],seed('Y-tree',root['id'],job['repeat']),
        actor=x.ACTOR,arm='original8',candidates=np.asarray(root['candidates']),budget_ms=HIGH)
    own.belief=TeamBelief(case['map'],root['controller_seed'])
    prefix_start=time.perf_counter()
    for obs in root['history'][:-1]:own.observe(obs)
    prefix_ms=1000*(time.perf_counter()-prefix_start)
    stop=next(i for i,a in enumerate(root['candidates']) if np.max(np.abs(a))<1e-12)
    snapshots=[dict(at_ms=0.,selected_candidate=stop,simulations=0,visits=[0]*6)]
    torch.cuda.synchronize();start=time.perf_counter();cpu_start=time.process_time()
    own.deadline=start+HIGH/1000.;own.operation_ms={};own.rejections=own.fallbacks=0
    own.nodes=1;own.traces=[];own.active_trace=None;own.remaining=root['history'][-1]['remaining']
    signal_ms=0.;signals=dict(decision_score=0.,entropy=0.,entropy_ms=0.,signal_ms=0.,
        sampling_rank_disagreement=0.,world_rank_disagreement=0.,probe_missing=True)
    completed=0
    try:
        own.observe(root['history'][-1]);own.root=public_node(own,root['history'][-1])
        own.features(own.root);own.potential(own.root);own.prepare_models(own.root)
        init_ms=1000*(time.perf_counter()-start)
        # Probe is an explicit extra cost for the decision gate. Baseline
        # search's trace clock excludes it; replay subtracts it from each
        # adaptive 50/200ms search prefix, preserving the charged budget.
        signals=probe(own,root,job['repeat']);signal_ms=signals['signal_ms']
        own.deadline+=signal_ms/1000.
        for _ in range(100000):
            own.trajectory();completed+=1
            at=1000*(time.perf_counter()-start)-signal_ms
            snapshots.append(dict(at_ms=at,selected_candidate=selected(own.root.visits,stop),
                simulations=completed,visits=list(own.root.visits),totals=list(own.root.totals)))
    except BudgetAbort:
        if own.active_trace:own.traces.append(np.asarray(own.active_trace).reshape(-1,26))
    torch.cuda.synchronize();wall_ms=1000*(time.perf_counter()-start)
    cpu_ms=1000*(time.process_time()-cpu_start)
    traces=np.concatenate(own.traces) if own.traces else np.zeros((0,26))
    errors=audit_simulated(traces,np.asarray(case['map']),case['speed'])
    if np.any(errors):raise RuntimeError('Independent Y simulation audit failed')
    row=job|dict(group=root['group'],split=root['split'],opponent=root['opponent'],stratum=root['stratum'],
        signals=signals,snapshots=snapshots,stop_index=stop,completed_simulations=completed,
        max_simulation_depth=max([len(t) for t in own.traces],default=0),
        prefix_restore_ms=prefix_ms,wall_including_probe_ms=wall_ms,
        baseline_trace_wall_ms=wall_ms-signal_ms,cpu_ms=cpu_ms,
        simulation_errors=errors.tolist(),physical_gpu=3)
    x.write(path,row);return row


def run_jobs(jobs,cfg,output,workers):
    ctx=mp.get_context('spawn');queue=ctx.Queue()
    for cpu in range(48,48+workers):queue.put(cpu)
    rows=[];started=time.time();pool=ProcessPoolExecutor(max_workers=workers,mp_context=ctx,
        initializer=initializer,initargs=(cfg,str(output),queue));failed=True
    try:
        futures={pool.submit(worker,j):j for j in jobs}
        for f in as_completed(futures):
            job=futures[f]
            try:rows.append(f.result())
            except BaseException as exc:
                import traceback
                x.write(output/'failures'/f'{job["key"]}.json',dict(job=job,error=repr(exc),traceback=traceback.format_exc()));raise
            x.write(output/'status.json',dict(state='running',stage=job.get('stage','traces'),
                completed=len(rows),planned=len(jobs),workers=workers,elapsed_seconds=time.time()-started,physical_gpu=3))
        failed=False
    finally:
        processes=list((pool._processes or {}).values());pool.shutdown(wait=not failed,cancel_futures=failed)
        if failed:
            for p in processes:
                if p.is_alive():p.terminate()
            for p in processes:p.join(timeout=2)
        queue.close()
    return rows


def prepare(output):
    output.mkdir(parents=True,exist_ok=True)
    cfg=x.read(X/'experiment.json');x.verify(cfg)
    roots=x.read(X/'root_manifest.json')['roots']
    inputs={str(X/'experiment.json'):x.sha(X/'experiment.json'),str(X/'root_manifest.json'):x.sha(X/'root_manifest.json')}
    reference={r['id']:{str(i):[] for i in range(6)} for r in roots}
    for f in sorted((X/'counterfactuals').glob('*.json')):
        data=x.read(f)['summary'];inputs[str(f)]=x.sha(f)
        reference[data['root_id']][str(data['candidate'])].append(dict(stream=data['stream'],utility=data['utility']))
    for r in roots:inputs[str(X/'public_roots'/f'{r["id"]}.json')]=r['sha256']
    for cells in reference.values():
        for rows in cells.values():rows.sort(key=lambda r:r['stream'])
    x.write(output/'references.json',reference);x.write(output/'input_manifest.json',inputs)
    paths=[Path(__file__),ROOT/'code/tests/test_e45y_compute_allocation.py',
        ROOT/'code/watch_e45y_compute_allocation.py',ROOT/'code/E45Y_COMPUTE_ALLOCATION_PROTOCOL_20261104.md']
    scientific=dict(name='E45-Y',source_hashes={str(p):x.sha(p) for p in paths}|cfg['source_hashes'],
        input_manifest_sha256=x.sha(output/'input_manifest.json'),references_sha256=x.sha(output/'references.json'),
        actor_frozen=True,new_policy_training=False,physical_gpu=3,search_repeats=4,
        low_ms=LOW,high_ms=HIGH,signal_worlds=16,bootstrap_draws=128,
        primary_signal='bootstrap expected ranking loss of a one-step geometry surrogate, not formal VOI or Q confidence',
        threshold='training-only per-opponent 2/3 quantile; strict >; no outcome labels',
        evaluation='development trace-replay budget matching; not full-episode live scheduling',
        controls=['cost-matched uniform','same-K random','same-K spatial entropy','same-stratum-K random',
                  'optimistic label allocator','cross-fitted reference allocator'],
        reference_worlds='reuse X 8 independent reference worlds per development candidate; no W sealed test use',
        stop_rule='one bounded feasibility audit; no threshold/hyperparameter sweep or automatic long training')
    x.write(output/'experiment.json',scientific)
    return cfg,roots,scientific


def verify(output,scientific):
    assert x.sha(output/'input_manifest.json')==scientific['input_manifest_sha256']
    for p,h in scientific['source_hashes'].items():
        if x.sha(p)!=h:raise RuntimeError('Frozen source changed: '+p)
    for p,h in x.read(output/'input_manifest.json').items():
        if x.sha(p)!=h:raise RuntimeError('Frozen input changed: '+p)
    assert x.sha(output/'references.json')==scientific['references_sha256']


def evaluate(rows,references):
    train=[r for r in rows if r['split']=='train'];development=[r for r in rows if r['split']=='development']
    thresholds={o:float(np.quantile([r['signals']['decision_score'] for r in train if r['opponent']==o],2/3))
                for o in ('fixed_D_kernel','rootpuct')}
    results=[];allocations=[]
    for repeat in range(4):
        for opponent in ('fixed_D_kernel','rootpuct'):
            batch=sorted([r for r in development if r['repeat']==repeat and r['opponent']==opponent],key=lambda r:r['root_id'])
            if not batch:continue
            n=len(batch);scores=[r['signals']['decision_score'] for r in batch]
            mask=threshold_mask(scores,thresholds[opponent]);k=int(mask.sum());matched=(LOW*(n-k)+HIGH*k)/n
            def outcome(record,budget,overhead=0.,fold=None):
                action=query(record,budget,overhead);values=references[record['root_id']][str(action)]
                y=[v['utility'] for v in values if fold is None or (v['stream']%2)==fold]
                return float(np.mean(y))
            low=np.asarray([outcome(r,LOW) for r in batch]);high=np.asarray([outcome(r,HIGH) for r in batch])
            uniform=np.asarray([outcome(r,matched) for r in batch])
            dec=np.asarray([outcome(r,HIGH if m else LOW,r['signals']['signal_ms']) for r,m in zip(batch,mask)])
            entropy_mask=top_k_mask([r['signals']['entropy'] for r in batch],k)
            ent=np.asarray([outcome(r,HIGH if m else LOW,r['signals']['entropy_ms']) for r,m in zip(batch,entropy_mask)])
            rng=np.random.default_rng(seed('Y-random',opponent,repeat));random=[];stratum_random=[]
            for _ in range(1000):
                random_mask=np.zeros(n,bool)
                if k:random_mask[rng.choice(n,k,replace=False)]=True
                random.append(np.where(random_mask,high,low))
                matched_mask=np.zeros(n,bool)
                for st in ('handover','public_imminent_first'):
                    ids=np.asarray([i for i,r in enumerate(batch) if r['stratum']==st],dtype=np.int64);kk=int(mask[ids].sum())
                    if kk:matched_mask[rng.choice(ids,kk,replace=False)]=True
                stratum_random.append(np.where(matched_mask,high,low))
            optimistic_mask=top_k_mask(high-low,k)
            oracle=np.where(optimistic_mask,high,low)
            cross=np.zeros(n)
            for fitfold in (0,1):
                lo=np.asarray([outcome(r,LOW,fold=fitfold) for r in batch]);hi=np.asarray([outcome(r,HIGH,fold=fitfold) for r in batch])
                chosen=top_k_mask(hi-lo,k)
                cross+=np.asarray([outcome(r,HIGH if m else LOW,fold=1-fitfold) for r,m in zip(batch,chosen)])/2
            policies=dict(uniform=uniform,decision=dec,entropy=ent,
                random=np.mean(random,axis=0),stratum_random=np.mean(stratum_random,axis=0),
                optimistic_label=oracle,crossfit_reference=cross)
            for i,r in enumerate(batch):
                allocations.append(dict(root_id=r['root_id'],group=r['group'],opponent=opponent,repeat=repeat,
                    stratum=r['stratum'],high_assigned=bool(mask[i]),score=scores[i],entropy=r['signals']['entropy'],
                    signal_ms=r['signals']['signal_ms'],matched_mean_budget_ms=matched,
                    low_reference_utility=float(low[i]),high_reference_utility=float(high[i]),
                    policies={p:float(v[i]) for p,v in policies.items()}))
            for policy,v in policies.items():
                results.append(dict(opponent=opponent,repeat=repeat,policy=policy,roots=n,high_assignments=k,
                    charged_mean_ms=matched,max_assigned_ms=HIGH,reference_utility=float(v.mean())))
    comparisons=[]
    for opponent in ('fixed_D_kernel','rootpuct'):
        subset=[r for r in allocations if r['opponent']==opponent]
        for comparator in ('uniform','random','stratum_random','entropy','optimistic_label','crossfit_reference'):
            groups={}
            for r in subset:groups.setdefault(r['group'],[]).append(r['policies']['decision']-r['policies'][comparator])
            deltas=np.asarray([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(seed('Y-CI',opponent,comparator))
            ci=np.percentile(rng.choice(deltas,(10000,len(deltas)),replace=True).mean(1),[2.5,97.5]).tolist()
            comparisons.append(dict(opponent=opponent,comparator=comparator,groups=len(groups),
                difference=float(deltas.mean()),descriptive_95_interval=ci))
    return dict(thresholds=thresholds,policy_statistics=results,allocations=allocations,comparisons=comparisons,
        scope='exploratory fixed-policy reference evaluation, six reused development maps, one Actor; uncorrected descriptive group intervals',
        label_allocator_scope='optimistic finite-reference diagnostic, not deployable or a theoretical upper bound',
        crossfit_scope='reference-based allocation on one half of worlds scored on the other half; not deployable',
        budget_scope='equal nominal state-batch charged budgets and <=200ms per decision; signal cost deducted from search prefixes; live latency not validated')


def main(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise RuntimeError('Use CUDA_VISIBLE_DEVICES=3')
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    lock=(output/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'experiment.json').exists():raise RuntimeError('Use a new directory for a separate scientific run')
    cfg,roots,scientific=prepare(output);verify(output,scientific);started=time.time()
    # Calibration examines compute only; no action/outcome utility is used.
    training=[r for r in roots if r['split']=='train'][:4]
    calibrations=[]
    for workers in (1,4):
        directory=output/f'calibration_{workers}';directory.mkdir()
        jobs=[dict(key=f'{r["id"]}_cal{workers}',root_id=r['id'],repeat=0,output=str(directory),stage='calibration') for r in training]
        rr=run_jobs(jobs,cfg,directory,workers)
        metrics=dict(workers=workers,simulations=sum(r['completed_simulations'] for r in rr),
            cpu_wall=sum(r['cpu_ms'] for r in rr)/max(1e-9,sum(r['wall_including_probe_ms'] for r in rr)))
        calibrations.append(metrics)
    reference,parallel=calibrations
    workers=4 if parallel['simulations']>=.8*reference['simulations'] and parallel['cpu_wall']>=.8 else 1
    x.write(output/'calibration.json',dict(chosen_workers=workers,measurements=calibrations,efficacy_used=False))
    chosen_roots=roots[:2] if args.smoke else roots
    repeats=1 if args.smoke else 4
    jobs=[dict(key=f'{r["id"]}_r{repeat}',root_id=r['id'],repeat=repeat,output=str(output),stage='traces')
          for r in chosen_roots for repeat in range(repeats)]
    rows=run_jobs(jobs,cfg,output,workers)
    if not args.smoke:
        result=evaluate(rows,x.read(output/'references.json'));x.write(output/'results.json',result)
        lines=['# E45-Y：决策难度与搜索预算可行性诊断','',
            '这是状态批次的搜索前缀重放，不是整局在线调度成功率。判断分歧的耗时已从该策略搜索预算中扣除。',
            '', '|对手|策略|参考效用|平均分配ms|','|---|---|---:|---:|']
        for opponent in ('fixed_D_kernel','rootpuct'):
            for policy in ('uniform','random','stratum_random','entropy','decision','optimistic_label','crossfit_reference'):
                rr=[r for r in result['policy_statistics'] if r['opponent']==opponent and r['policy']==policy]
                lines.append(f'|{opponent}|{policy}|{np.mean([r["reference_utility"] for r in rr]):.4f}|{np.mean([r["charged_mean_ms"] for r in rr]):.1f}|')
        lines+=['','标签分配器只用于诊断空间，不可部署。比较区间按地图分组，仅探索性描述，未证明新的方法贡献。']
        (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    verify(output,scientific)
    x.write(output/'completion_audit.json',dict(complete=True,jobs=len(rows),elapsed_seconds=time.time()-started,
        source_unchanged=True,actor_unchanged=True,physical_gpu=3,policy_optimizer_updates=0,W_test_read=False,smoke=args.smoke))
    x.write(output/'status.json',dict(state='complete',stage='complete',jobs=len(rows),physical_gpu=3))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    try:main(args)
    except BaseException as exc:
        import traceback
        x.write(args.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        x.write(args.output/'status.json',dict(state='failed',error=repr(exc)));raise
