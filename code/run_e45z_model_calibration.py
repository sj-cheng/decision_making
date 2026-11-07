"""One-step controlled audit of belief uncertainty versus transition mismatch.

No policy is trained or deployed. Hypothetical current worlds are sampled from
the same public prior. Position-oracle moments are diagnostic scoring only.
RootPUCT's full-state reference never supplies an Actor or leaf input.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import argparse
import time
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import fcntl
import numpy as np
import torch

import run_e45x_event_diagnostic as x
from learning.e45x_event_diagnostic import (prepare_public,initial_world,seed,boxes_of,
    simultaneous_capture,target_step)
from run_e45y_compute_allocation import world_scores
from solvers.root_audit_puct import RootPUCT
from solvers.e45_team_adapter import public_observation
from run_frozen_d_closed_loop import legal_action

ROOT=x.ROOT;X=ROOT/'current/e45x_first_capture_diagnostic_20261104'
OUT=ROOT/'current/e45z_model_calibration_20261104'
EVENT_NAMES=['capture_e0','capture_e1','retire_p0','retire_p1','see_live_e0','see_live_e1','double_capture','any_capture']


def conditional_moments(graph,joint):
    points,links,lengths,probabilities=graph[:4]
    mass=joint.sum(1);mass=mass/mass.sum();active=np.flatnonzero(mass>0)
    means=np.zeros((len(points),2));variance=np.zeros(len(points))
    for i in active:
        mode=joint[i]/joint[i].sum();mixed=.9*mode+.1/9.
        ids=links[i,:lengths[i]];p=probabilities[i,:lengths[i]]@mixed;p/=p.sum()
        means[i]=p@points[ids]
        variance[i]=float(np.sum(p*np.sum((points[ids]-means[i])**2,axis=1)))
    mean=mass@means;current_mean=mass@points
    within=float(mass@variance);between=float(np.sum(mass*np.sum((means-mean)**2,axis=1)))
    current_variance=float(np.sum(mass*np.sum((points-current_mean)**2,axis=1)))
    return dict(conditional_mean=means,predictive_mean=mean,within_variance=within,
        between_position_variance=between,total_variance=within+between,current_position_variance=current_variance)


def conditional_error(predicted_mean,reference):
    """Reference: [current-world, replicate, coordinate]. Preserve signed bias correction."""
    reference=np.asarray(reference);n=reference.shape[1]
    mean=reference.mean(1);variance=np.sum(np.var(reference,axis=1,ddof=1),axis=1)
    mse=np.mean(np.sum((reference-predicted_mean[:,None,:])**2,axis=-1))
    bias=np.mean(np.sum((mean-predicted_mean)**2,axis=-1)-variance/n)
    return dict(conditional_mse=float(mse),conditional_mean_bias_squared_unbiased=float(bias),
                reference_random_variance=float(variance.mean()))


def event_vectors(positions,targets,actions,ap,ae,boxes):
    result=[]
    for action in actions:
        pp,ee,seen=simultaneous_capture(positions+action,targets,ap,ae,boxes)
        result.append(np.r_[~ee,~pp,seen>=0,not ee.any(),ee.sum()<ae.sum()].astype(float))
    return np.asarray(result)


def energy_distance(a,b,device='cuda:0'):
    # Empirical V-statistic distance, not an IID significance test. Matched-D
    # controls have the same cluster/repetition design as the Root sample.
    a=torch.as_tensor(np.asarray(a),dtype=torch.float64,device=device)
    b=torch.as_tensor(np.asarray(b),dtype=torch.float64,device=device)
    def distance(x,y):return torch.cdist(x,y,p=2,compute_mode='donot_use_mm_for_euclid_dist').mean()
    with torch.no_grad():return float((2*distance(a,b)-distance(a,a)-distance(b,b)).item())


def enemy_reference(problem,state,other,noise):
    other.seeded_state(problem,seed(noise,'solver'),seed(noise,'problem'))
    action=np.asarray(other.search(problem,state.copy(),turn=0).best_action).ravel()[:4].reshape(2,2).copy()
    boxes=np.asarray(problem.obstacles,dtype=float).reshape(-1,2,2);shields=0
    for j in range(2):
        if not legal_action(state[2*j:2*j+2,0],action[j],1.,boxes):action[j]=0;shields+=1
    return action,shields


def check_physics(prepared,state,target_next,actions):
    p=prepared.problem;boxes=boxes_of(prepared.case)
    for own in actions:
        enemy=target_next-state[:4,0].reshape(2,2)
        for j in range(2):
            if not legal_action(state[2*j:2*j+2,0],enemy[j],1.,boxes):raise RuntimeError('Illegal target forecast')
        joint=np.r_[enemy.ravel(),own.ravel()][:,None];nxt=p.step(state,joint,1.)
        if not p.is_valid(nxt):raise RuntimeError('Invalid calibration successor')
        np.testing.assert_allclose(nxt[:8],state[:8]+joint,rtol=0,atol=1e-10)
        pp,ee,_=simultaneous_capture(nxt[4:8,0].reshape(2,2),nxt[:4,0].reshape(2,2),
            state[p.active_idxs[2:],0]>.5,state[p.active_idxs[:2],0]>.5,boxes)
        np.testing.assert_array_equal(pp,nxt[p.active_idxs[2:],0]>.5)
        np.testing.assert_array_equal(ee,nxt[p.active_idxs[:2],0]>.5)


def initialize(queue):
    os.sched_setaffinity(0,{queue.get()})
    # No CUDA context per CPU reference worker. GPU3 is used once by the
    # parent for distribution-distance scoring after references complete.
    torch.set_num_threads(1)


def worker(job):
    output=Path(job['output']);path=output/'jobs'/f'{job["root_id"]}.json'
    if path.exists():return x.read(path)
    root=x.read(X/'public_roots'/f'{job["root_id"]}.json');raw=x.read(root['case_path'])
    prepared=prepare_public(root,{k:raw[k] for k in ('map','speed')},None)
    obs=root['history'][-1];boxes=boxes_of(prepared.case);positions=np.asarray(obs['positions'])
    ap=np.asarray(obs['active_p']);ae=np.asarray(obs['active_e']);actions=np.asarray(root['candidates'])
    assert ap.all() and ae.all()
    moments=[conditional_moments(g,w) for g,w in zip(prepared.graphs,prepared.joints)]
    world_count=job['worlds'];replicates=job['replicates'];forecast_count=job['forecast_count']
    forecast=[];forecast_displacements=[]
    for sample in range(forecast_count):
        state,ids,modes=initial_world(root,prepared,sample,'Z-forecast');nxt=[]
        for j,g in enumerate(prepared.graphs):
            local=np.array([seed('Z-forecast-motion',root['id'],sample,j)],np.int64)
            i,m=target_step(ids[j],modes[j],g[1],g[2],g[3],local);nxt.append(g[0][i])
        nxt=np.asarray(nxt);forecast.append(nxt);forecast_displacements.append(nxt-state[:4,0].reshape(2,2))
    forecast=np.asarray(forecast);forecast_displacements=np.asarray(forecast_displacements)
    predicted_events=np.asarray([event_vectors(positions,t,actions,ap,ae,boxes) for t in forecast]).mean(0)
    predicted_scores=world_scores(positions,ap,ae,forecast,actions,raw['speed'],prepared.own.public.vertices,
        prepared.own.public.dist,boxes).mean(0)
    root_reference=RootPUCT([None]*4,None,3000,0.,0.)
    current=[];d_future=[];r_future=[];d_displacements=[];r_displacements=[];mu=[];shields=0
    physical_steps=0
    for world in range(world_count):
        state,ids,_=initial_world(root,prepared,world,'Z-reference');before=state[:4,0].reshape(2,2).copy()
        current.append(before);mu.append(np.asarray([moments[j]['conditional_mean'][ids[j]] for j in range(2)]))
        ds=[];rs=[]
        for replicate in range(replicates):
            target_d=[]
            for j,g in enumerate(prepared.graphs):
                mode_weights=prepared.joints[j][ids[j]];mode_weights=mode_weights/mode_weights.sum()
                generator=np.random.default_rng(seed('Z-D-mode',root['id'],world,replicate,j))
                mode=int(generator.choice(9,p=mode_weights))
                local=np.array([seed('Z-D-motion',root['id'],world,replicate,j)],np.int64)
                i,m=target_step(ids[j],mode,g[1],g[2],g[3],local);target_d.append(g[0][i])
            target_d=np.asarray(target_d)
            enemy,shield=enemy_reference(prepared.problem,state,root_reference,('Z-Root',root['id'],world,replicate))
            target_r=before+enemy;shields+=shield
            # Simultaneous actions: Root's action is computed from the same
            # current state once, before any of the six pursuer alternatives.
            check_physics(prepared,state,target_d,actions);check_physics(prepared,state,target_r,actions)
            physical_steps+=2*len(actions);ds.append(target_d);rs.append(target_r)
        ds=np.asarray(ds);rs=np.asarray(rs);d_future.append(ds);r_future.append(rs)
        d_displacements.append(ds-before);r_displacements.append(rs-before)
    current=np.asarray(current);mu=np.asarray(mu);d_future=np.asarray(d_future);r_future=np.asarray(r_future)
    d_events=np.asarray([[event_vectors(positions,t,actions,ap,ae,boxes) for t in group] for group in d_future])
    r_events=np.asarray([[event_vectors(positions,t,actions,ap,ae,boxes) for t in group] for group in r_future])
    # Squared probability discrepancies and Brier scores are different: the
    # latter includes outcome randomness and is not pure calibration error.
    event_metrics=dict(predicted_probabilities=predicted_events,matched_D_probabilities=d_events.mean((0,1)),
        root_probabilities=r_events.mean((0,1)),
        matched_D_probability_mse=float(np.mean((predicted_events-d_events.mean((0,1)))**2)),
        root_probability_mse=float(np.mean((predicted_events-r_events.mean((0,1)))**2)),
        matched_D_brier=float(np.mean((d_events-predicted_events)**2)),root_brier=float(np.mean((r_events-predicted_events)**2)))
    d_scores=world_scores(positions,ap,ae,d_future.reshape(-1,2,2),actions,raw['speed'],
        prepared.own.public.vertices,prepared.own.public.dist,boxes).reshape(world_count,replicates,6)
    r_scores=world_scores(positions,ap,ae,r_future.reshape(-1,2,2),actions,raw['speed'],
        prepared.own.public.vertices,prepared.own.public.dist,boxes).reshape(world_count,replicates,6)
    chosen=int(predicted_scores.argmax());cross={}
    for name,scores in [('matched_D',d_scores),('Root',r_scores)]:
        gains=[]
        for fold in (0,1):
            fit=scores[fold::2].mean((0,1));held=scores[1-fold::2].mean((0,1))
            gains.append(float(held[int(fit.argmax())]-held[chosen]))
        cross[name]=float(np.mean(gains))
    target_metrics=[]
    for j in range(2):
        d=conditional_error(mu[:,j],d_future[:,:,j]);r=conditional_error(mu[:,j],r_future[:,:,j])
        belief_mean=moments[j]['predictive_mean']
        d['belief_mean_mse']=float(np.mean(np.sum((d_future[:,:,j]-belief_mean)**2,axis=-1)))
        r['belief_mean_mse']=float(np.mean(np.sum((r_future[:,:,j]-belief_mean)**2,axis=-1)))
        target_metrics.append(dict(target=j,known=any(q is not None for q in obs['reports'][j]),
            D_model_current_position_variance=moments[j]['current_position_variance'],
            D_model_within_future_variance=moments[j]['within_variance'],
            D_model_between_position_future_variance=moments[j]['between_position_variance'],
            D_model_total_future_variance=moments[j]['total_variance'],matched_D=d,Root=r,
            conditional_excess_mean_bias_squared=r['conditional_mean_bias_squared_unbiased']-d['conditional_mean_bias_squared_unbiased']))
    row=dict(root_id=root['id'],group=root['group'],family=root['family'],split=root['split'],stratum=root['stratum'],
        source_opponent=root['opponent'],worlds=world_count,replicates=replicates,forecast_count=forecast_count,
        target_metrics=target_metrics,event_metrics=event_metrics,chosen_D_proxy_action=chosen,
        chosen_Root_proxy_action=int(r_scores.mean((0,1)).argmax()),proxy_crossfit_gains=cross,
        forecast_displacements=forecast_displacements,D_reference_displacements=np.asarray(d_displacements),
        Root_reference_displacements=np.asarray(r_displacements),root_shields=shields,
        Root_search_calls=world_count*replicates,physical_one_step_audits=physical_steps,
        policy_optimizer_updates=0,scope='one-step same-public-prior diagnostic, not deployed oracle prediction or task utility')
    x.write(path,row);return row


def summarize(output,rows):
    # Only this parent allocates CUDA memory, on physical GPU3 via visibility.
    for row in rows:
        row['energy']={}
        forecast=np.asarray(row['forecast_displacements']);d=np.asarray(row['D_reference_displacements']);r=np.asarray(row['Root_reference_displacements'])
        for j in range(2):
            baseline=energy_distance(forecast[:,j],d[:,:,j].reshape(-1,2))
            mismatch=energy_distance(forecast[:,j],r[:,:,j].reshape(-1,2))
            row['energy'][str(j)]=dict(D_control=baseline,Root=mismatch,excess=mismatch-baseline)
    development=[r for r in rows if r['split']=='development']
    metrics=[]
    for stratum in ('handover','public_imminent_first'):
        subset=[r for r in development if r['stratum']==stratum]
        groups={}
        for row in subset:
            groups.setdefault(row['group'],[]).append(dict(
                conditional_bias_excess=float(np.mean([m['conditional_excess_mean_bias_squared'] for m in row['target_metrics']])),
                event_probability_mse_excess=row['event_metrics']['root_probability_mse']-row['event_metrics']['matched_D_probability_mse'],
                displacement_energy_excess=float(np.mean([v['excess'] for v in row['energy'].values()])),
                Root_proxy_crossfit_gain=row['proxy_crossfit_gains']['Root'],
                matched_D_proxy_crossfit_gain=row['proxy_crossfit_gains']['matched_D']))
        for name in next(iter(groups.values()))[0]:
            values=np.asarray([np.mean([r[name] for r in g]) for g in groups.values()])
            generator=np.random.default_rng(seed('Z-bootstrap',stratum,name))
            interval=np.percentile(generator.choice(values,(10000,len(values)),replace=True).mean(1),[2.5,97.5]).tolist()
            metrics.append(dict(stratum=stratum,metric=name,groups=len(groups),mean=float(values.mean()),descriptive_95_interval=interval))
    for row in rows:
        for name in ('forecast_displacements','D_reference_displacements','Root_reference_displacements'):row.pop(name)
    result=dict(complete=True,root_summaries=rows,development_group_metrics=metrics,
        interpretation='Fixed public prior and one-step transitions. D-internal variance split is not a causal decomposition of real task failure.',
        action_scope='one-step geometry proxy, cross-fitted on current-world clusters; not optimal action or full-game utility',
        uncertainty='six reused development maps, one Actor candidate pool, exploratory uncorrected group intervals',
        next_gate='Only calibrated prediction plus decision relevance could justify a new forecaster experiment; no automatic model/policy training')
    x.write(output/'results.json',result)
    lines=['# E45-Z：对手预测校准诊断','',
        '固定同一公共后验，比较匹配D与RootPUCT3000；特权当前位置只用于离线误差诊断，不进入Actor。',
        '', '|状态类别|指标|地图等权均值|探索性95%区间|','|---|---|---:|---|']
    for r in metrics:lines.append(f'|{r["stratum"]}|{r["metric"]}|{r["mean"]:.5f}|[{r["descriptive_95_interval"][0]:.5f}, {r["descriptive_95_interval"][1]:.5f}]|')
    lines+=['','连续分布距离在位移空间计算，并扣除同样采样设计的D对照作为描述性比较。',
        '代理动作交叉评估收益不是原始任务效用或闭环双捕率。一步结果不能直接外推多步、真实先验覆盖或新颖性。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise RuntimeError('Use physical GPU3 only')
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    lock=(output/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'experiment.json').exists():raise RuntimeError('Use a fresh scientific output directory')
    cfg=x.read(X/'experiment.json');x.verify(cfg)
    roots=x.read(X/'root_manifest.json')['roots']
    paths=[Path(__file__),ROOT/'code/tests/test_e45z_model_calibration.py',ROOT/'code/watch_e45z_model_calibration.py',
           ROOT/'code/E45Z_MODEL_CALIBRATION_PROTOCOL_20261104.md']
    sources={str(p):x.sha(p) for p in paths}|cfg['source_hashes']
    ycfg=x.read(ROOT/'current/e45y_compute_allocation_20261104/experiment.json')
    sources.update(ycfg['source_hashes'])
    inputs={str(X/'public_roots'/f'{r["id"]}.json'):r['sha256'] for r in roots}
    settings=dict(name='E45-Z',source_hashes=sources,input_hashes=inputs,physical_gpu=3,CPU_workers=args.workers,
        current_worlds=2 if args.smoke else 32,replicates=2 if args.smoke else 4,forecast_worlds=8 if args.smoke else 128,
        shared_prior=True,position_oracle='offline scoring only, both target positions fixed; modes remain marginalized',
        scope='one-step controlled model calibration; no training or new deployed algorithm',smoke=args.smoke,
        event_names=EVENT_NAMES,W_test_read=False)
    x.write(output/'experiment.json',settings)
    for path in paths:
        snapshot=output/'source_snapshot'/path.relative_to(ROOT/'code')
        snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(path.read_bytes())
    def verify():
        for name,h in sources.items():assert x.sha(name)==h,name
        for name,h in inputs.items():assert x.sha(name)==h,name
        x.verify(cfg)
    verify();chosen=roots[:2] if args.smoke else roots;started=time.time()
    jobs=[dict(root_id=r['id'],worlds=settings['current_worlds'],replicates=settings['replicates'],
        forecast_count=settings['forecast_worlds'],output=str(output)) for r in chosen]
    cpus=list(range(48,56))+list(range(56,80));workers=min(args.workers,len(jobs),len(cpus))
    x.write(output/'status.json',dict(state='running',completed=0,planned=len(jobs),workers=workers,
        elapsed_seconds=0,physical_gpu=3,phase='CPU reference worlds'))
    context=mp.get_context('spawn');queue=context.Queue()
    for cpu in cpus[:workers]:queue.put(cpu)
    rows=[];pool=ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=initialize,initargs=(queue,));failed=True
    try:
        futures={pool.submit(worker,j):j for j in jobs}
        for f in as_completed(futures):
            job=futures[f]
            try:rows.append(f.result())
            except BaseException as exc:
                import traceback
                x.write(output/'failures'/f'{job["root_id"]}.json',dict(job=job,error=repr(exc),traceback=traceback.format_exc()));raise
            x.write(output/'status.json',dict(state='running',completed=len(rows),planned=len(jobs),workers=workers,
                elapsed_seconds=time.time()-started,physical_gpu=3,phase='CPU reference worlds'))
        failed=False
    finally:
        processes=list((pool._processes or {}).values());pool.shutdown(wait=not failed,cancel_futures=failed)
        if failed:
            for process in processes:
                if process.is_alive():process.terminate()
            for process in processes:process.join(timeout=2)
        queue.close()
    if not args.smoke:summarize(output,rows)
    else:
        # Exercise the GPU scoring path without consuming development data.
        row=rows[0]
        energy_distance(np.asarray(row['forecast_displacements'])[:,0],np.asarray(row['Root_reference_displacements'])[:,:,0].reshape(-1,2))
    verify()
    x.write(output/'completion_audit.json',dict(complete=True,roots=len(rows),elapsed_seconds=time.time()-started,
        Root_search_calls=sum(r['Root_search_calls'] for r in rows),physical_one_step_audits=sum(r['physical_one_step_audits'] for r in rows),
        actor_unchanged=True,source_unchanged=True,policy_optimizer_updates=0,physical_gpu=3,W_test_read=False,smoke=args.smoke))
    x.write(output/'status.json',dict(state='complete',phase='complete',roots=len(rows),physical_gpu=3))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--workers',type=int,default=32)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    try:main(args)
    except BaseException as exc:
        import traceback
        x.write(args.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        x.write(args.output/'status.json',dict(state='failed',error=repr(exc)));raise
