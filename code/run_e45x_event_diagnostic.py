"""Bounded three-stage event-value diagnostic, all CUDA workers on physical GPU3."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import fcntl
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import sys

import numpy as np
import torch
from learning.e45x_event_diagnostic import (FrozenActor, EventController, LeafReadout, ARMS,
    seed, prepare_public, root_candidates, counterfactual, imminent, advance, phi)
from learning.e45w_support_recovery import install
from learning.e45w_joint_env import PublicFeatureState
from solvers.root_audit_puct import RootPUCT
from run_e45_team_pilot import FixedKernelTargets
from solvers.e45_team_adapter import public_observation
from corner_estimator_scenarios import original_model
from e45c_capture_audit import audit_simulated
from run_e45w_joint_training import sha

ROOT=Path(__file__).resolve().parents[1]
W=ROOT/'current/e45w_joint_ppo_and_planning_20261103'
OUT=ROOT/'current/e45x_first_capture_diagnostic_20261104'
ACTOR=None;CFG=None;CPUS=None;CACHE={}


def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    def cast(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=cast));tmp.replace(p)


def register(output,smoke=False):
    if (output/'experiment.json').exists():return read(output/'experiment.json')
    source=read(W/'experiment.json');maps=[]
    for split,key in [('train','train_entries'),('development','validation_entries')]:
        for family in ('long_wall','wall_corner','multi_route'):
            candidates=sorted([r for r in source[key] if r['family']==family],key=lambda r:seed(14540261104,r['id']))
            for row in candidates[:1 if smoke else 2]:
                maps.append(row|dict(audit_split=split))
    if smoke:maps=maps[:1]
    files=[ROOT/'code'/n for n in ('learning/e45x_event_diagnostic.py','run_e45x_event_diagnostic.py',
        'watch_e45x_event_diagnostic.py','tests/test_e45x_event_diagnostic.py',
        'E45X_FIRST_CAPTURE_DIAGNOSTIC_PROTOCOL_20261104.md')]
    inherited=read(W/'component_source_manifest.json')['files']
    inherited.update(read(W/'repair_coordinator/source_manifest.json')['files'])
    files_hash={str(p):sha(p) for p in files}|inherited
    model=source['models']['17']
    cfg=dict(name='E45-X',smoke=smoke,source_maps=maps,actor_path=model['pt'],actor_sha256=model['pt_sha256'],
        actor_seed=17,actor_frozen=True,policy_optimizer_updates=0,physical_gpu=3,logical_device='cuda:0',
        source_hashes=files_hash,root_candidates=6,training_streams=1 if smoke else 4,
        development_reference_streams=1 if smoke else 8,planning_repeats=1 if smoke else 2,
        decision_budget_ms=200.,arms=list(ARMS),root_strata=['handover','public_imminent_first'],
        imminent_rule='first later public state with both teams intact and posterior mass >=.35 within 1+sqrt(2)*(speed+1) of some pursuer',
        source_policy='frozen Single17 one stochastic proposal + target-free safety',
        worlds='one AB/BA per map by deterministic hash; both opponents',
        opponent_seed_scope='per absolute physical step; identical exogenous random keys across candidate actions',
        reference_model='posterior-product worlds; root-anchored D kernel or original adaptive RootPUCT3000',
        inference='public packet only; opponent/simulator can see sampled world; no hidden world enters leaf or Actor',
        heads='ridge readouts on frozen Single embedding; generic/all phase and residual/1v1 from same training-only supervision pool',
        ridge_regularization=.01,head_hyperparameter_search=False,
        estimand='one root-action intervention then frozen Actor continuation; not optimal Q or full new controller closed-loop efficacy',
        root_search='same fixed six root candidates; original UCT below root; 8/16 depth ceilings and early fresh-node leaves',
        scoring='original raw double capture minus .0025 per physical step; potential removed from raw bootstrap',
        statistics='development maps are grouped units; worlds/candidates/strata/opponents are correlated; descriptive intervals, no threshold-based expansion',
        stopping='one bounded diagnostic; no automatic policy training or fresh test use',
        sealed_W_test_read=False)
    output.mkdir(parents=True,exist_ok=True);write(output/'experiment.json',cfg)
    snapshot=output/'source_snapshot'
    for p in files:
        q=snapshot/p.relative_to(ROOT/'code');q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
    return cfg


def verify(cfg):
    for p,h in cfg['source_hashes'].items():
        if sha(p)!=h:raise RuntimeError('Frozen source changed: '+p)
    if sha(cfg['actor_path'])!=cfg['actor_sha256']:raise RuntimeError('Frozen Actor changed')
    for row in cfg['source_maps']:
        if sha(row['path'])!=row['sha256']:raise RuntimeError('Map changed')


def initializer(config,output,queue):
    global ACTOR,CFG,OUT
    os.sched_setaffinity(0,{queue.get()});torch.set_num_threads(1)
    CFG=config;OUT=Path(output)
    install(OUT/'belief_failures')
    ACTOR=FrozenActor(config['actor_path'],config['logical_device'])
    # Warm only a registered training packet. Compile/import overhead is
    # recorded separately and never charged to a single randomized arm.
    row=config['source_maps'][0];case=read(row['path'])
    warm=dict(id='engineering_warmup',history=case['history'],controller_seed=seed('X-warmup'),
              case_path=row['path'])
    prepared=prepare_public(warm,case,ACTOR)
    ACTOR.infer(prepared.own.features(case['history'][-1]))
    if (OUT/'leaf_readouts.npz').exists():
        actions,_=root_candidates(warm,prepared,ACTOR)
        for arm in ARMS:
            own=EventController(case['map'],case['speed'],seed('X-warmup'),actor=ACTOR,
                arm=arm,heads=LeafReadout(OUT/'leaf_readouts.npz'),candidates=actions)
            for obs in case['history'][:-1]:own.observe(obs)
            own.act(case['history'][-1],fixed_simulations=1)
    torch.cuda.synchronize()


def get_prepared(root):
    if root['id'] not in CACHE:
        if len(CACHE)>=2:CACHE.pop(next(iter(CACHE)))
        raw=read(root['case_path'])
        public_case={k:raw[k] for k in ('map','speed')}
        CACHE[root['id']]=prepare_public(root,public_case,ACTOR)
    return CACHE[root['id']]


def packet(task,history,stratum):
    identity=digest(dict(group=task['map']['id'],opponent=task['opponent'],world=task['world'],history=history))
    return dict(id=identity,group=task['map']['id'],family=task['map']['family'],
        split=task['map']['audit_split'],case_path=task['map']['path'],case_sha256=task['map']['sha256'],
        opponent=task['opponent'],world=task['world'],controller_seed=task['episode_seed'],
        history=history,stratum=stratum,source_policy='Single17 frozen',private_state_saved=False)


def collect(task):
    existing=OUT/'collection'/f'{task["key"]}.json'
    if existing.exists():
        old=read(existing);ids=[r['id'] for r in old['roots']]
        if all((OUT/'public_roots'/f'{i}.json').exists() for i in ids):
            return dict(key=task['key'],root_ids=ids,physical_steps=old['physical_steps'],imminent_root_found=old['imminent_root_found'])
    case=read(task['map']['path']);p=original_model(case['map'],case['speed']);boxes=np.asarray(case['map'])
    own=PublicFeatureState(boxes,case['speed'],task['episode_seed'])
    for obs in case['history']:own.observe(obs)
    state=np.asarray(next(w['initial'] for w in case['worlds'] if w['id']==task['world']),float)[:,None]
    if task['opponent']=='fixed_D_kernel':
        other=FixedKernelTargets(case,task['episode_seed']);state[:4,0]=other.positions().ravel()
    else:other=RootPUCT([None]*4,None,3000,0.,0.)
    obs=public_observation(int(state[8,0]),state[4:8,0].reshape(2,2),state[:4,0].reshape(2,2),
        state[p.active_idxs[2:],0]>.5,state[p.active_idxs[:2],0]>.5,boxes)
    if obs!=case['history'][-1]:raise RuntimeError('Collection handover public mismatch')
    history=list(case['history']);roots=[packet(task,list(history),'handover')];near=False
    # Collection fixed-D keeps the original handover graph and latent state.
    prep=type('CollectionModel',(),{})();prep.problem=p;prep.case=case
    if task['opponent']=='fixed_D_kernel':prep.graphs=other.graphs;ids=other.ids;modes=other.modes;opponent=None
    else:prep.graphs=[None,None];ids=[0,0];modes=[0,0];opponent=other
    steps=0;first=None
    while True:
        if steps and not near and imminent(own,obs):
            roots.append(packet(task,list(history),'public_imminent_first'));near=True
        physical,_=ACTOR.action(own.features(obs),obs,case['speed'],boxes,seed(task['key'],'collection_actor',obs['step']))
        state,obs,reward,done,captures,_=advance(prep,state,physical,opponent,ids,modes,
            (task['key'],'collection_enemy',obs['step']))
        steps+=1
        if captures and first is None:first=steps
        if any(obs['active_p']):own.observe(obs)
        history.append(obs)
        if done:break
    write(OUT/'collection'/f'{task["key"]}.json',dict(task=task,roots=roots,
        physical_steps=steps,first_capture=first,imminent_root_found=near,complete=True))
    for root in roots:
        prep=get_prepared(root);actions,sources=root_candidates(root,prep,ACTOR,CFG['root_candidates'])
        root['candidates']=actions.tolist();root['candidate_sources']=sources
        write(OUT/'public_roots'/f'{root["id"]}.json',root)
    return dict(key=task['key'],root_ids=[r['id'] for r in roots],physical_steps=steps,imminent_root_found=near)


def rollout_job(job):
    path=OUT/'counterfactuals'/f'{job["key"]}.json'
    if path.exists():return read(path)['summary']
    root=read(OUT/'public_roots'/f'{job["root_id"]}.json');prepared=get_prepared(root)
    result=counterfactual(root,prepared,ACTOR,root['candidates'][job['candidate']],job['stream'],
        job['namespace'],examples=True)
    examples=result.pop('examples')
    summary=job|result|dict(group=root['group'],opponent=root['opponent'],stratum=root['stratum'],split=root['split'])
    write(path,dict(summary=summary,examples=examples))
    return summary


def planning_job(job):
    path=OUT/('planning_calibration' if job.get('calibration') else 'planning')/f'{job["key"]}.json'
    if path.exists():return read(path)
    root=read(OUT/'public_roots'/f'{job["root_id"]}.json');case=read(root['case_path'])
    heads=LeafReadout(OUT/'leaf_readouts.npz')
    output=[]
    # Block randomization avoids always giving one arm the same position in
    # warming/thermal/drift order. Every arm has its own matched RNG seeds.
    order=list(ARMS);np.random.default_rng(seed(root['id'],job['repeat'],'order')).shuffle(order)
    for arm in order:
        own=EventController(case['map'],case['speed'],seed(root['id'],job['repeat'],'planner'),
            actor=ACTOR,arm=arm,heads=heads,candidates=np.asarray(root['candidates']),
            budget_ms=CFG['decision_budget_ms'])
        # Match the actual posterior export RNG to the root packet; the tree
        # simulation stream is separately paired across all arm variants.
        from learning.e45w_support_recovery import TeamBelief
        own.belief=TeamBelief(case['map'],root['controller_seed'])
        setup=time.perf_counter()
        for obs in root['history'][:-1]:own.observe(obs)
        prefix_ms=1000*(time.perf_counter()-setup)
        torch.cuda.synchronize();start=time.perf_counter();cpu=time.process_time()
        action=own.act(root['history'][-1])
        torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-start);cpu_ms=1000*(time.process_time()-cpu)
        candidates=np.asarray(root['candidates']);dist=np.max(np.abs(candidates-action),axis=(1,2))
        chosen=int(np.argmin(dist)) if dist.min()<1e-8 else None
        traces=np.concatenate(own.traces) if own.traces else np.zeros((0,26))
        errors=audit_simulated(traces,np.asarray(case['map']),case['speed'])
        if np.any(errors):raise RuntimeError('Independent tree physics audit failed: '+str(errors))
        output.append(dict(arm=arm,selected_candidate=chosen,returned_stop_without_completed_simulation=own.last_stats['simulations']==0,
            wall_ms=elapsed,cpu_ms=cpu_ms,prefix_ms=prefix_ms,stats=own.last_stats,
            simulation_errors=errors.tolist(),simulated_transitions=len(traces)))
    row=job|dict(group=root['group'],opponent=root['opponent'],stratum=root['stratum'],arms=output)
    write(path,row);return row


def guarded(kind,job):
    try:return {'collection':collect,'rollout':rollout_job,'planning':planning_job}[kind](job)
    except BaseException as exc:
        write(OUT/'failures'/f'{kind}_{job["key"]}.json',dict(job=job,error=repr(exc),traceback=traceback.format_exc()))
        raise


def run_pool(kind,jobs,cfg,output,workers,cpus,status):
    if not jobs:return []
    ctx=mp.get_context('spawn');queue=ctx.Queue();workers=min(workers,len(cpus),len(jobs))
    for core in cpus[:workers]:queue.put(core)
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=initializer,
        initargs=(cfg,str(output),queue));pending=iter(jobs);active={};rows=[];failed=True
    started=time.time()
    try:
        while True:
            while len(active)<workers*2:
                job=next(pending,None)
                if job is None:break
                active[pool.submit(guarded,kind,job)]=job
            if not active:break
            done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for f in done:active.pop(f);rows.append(f.result())
            write(output/'status.json',status|dict(stage=kind,completed=len(rows),planned=len(jobs),workers=workers,
                elapsed_seconds=time.time()-started,physical_gpu=3,policy_optimizer_updates=0,timestamp=time.time()))
        failed=False
    finally:
        processes=list((pool._processes or {}).values());pool.shutdown(wait=not failed,cancel_futures=failed)
        if failed:
            for p in processes:
                if p.is_alive():p.terminate()
            for p in processes:p.join(timeout=3)
        queue.close()
    return rows


def fit_heads(output,rows,cfg):
    examples=[]
    for row in rows:
        if row['split']=='train':examples+=read(output/'counterfactuals'/f'{row["key"]}.json')['examples']
    if not examples:raise RuntimeError('No training-only leaf supervision')
    z=np.asarray([r['z'] for r in examples]);y=np.asarray([r['label'] for r in examples]);phase=np.asarray([r['phase'] for r in examples])
    mean=z.mean(0);scale=np.maximum(z.std(0),1e-3);x=np.c_[(z-mean)/scale,np.ones(len(z))]
    def ridge(mask):
        xx=x[mask];yy=y[mask];penalty=np.eye(x.shape[1])*cfg['ridge_regularization'];penalty[-1,-1]=0
        return np.linalg.solve(xx.T@xx+penalty,xx.T@yy)
    generic=ridge(np.ones(len(y),bool));available=bool((phase==1).sum()>=32)
    residual=ridge(phase==1) if available else generic.copy()
    np.savez_compressed(output/'leaf_readouts.npz',mean=mean,scale=scale,generic=generic,residual=residual,residual_available=available)
    write(output/'leaf_training_audit.json',dict(training_examples=len(examples),residual_examples=int((phase==1).sum()),
        residual_available=available,frozen_actor=True,actor_optimizer_updates=0,readout_fit='closed-form ridge, fixed lambda',
        development_or_test_labels_used=False,shared_supervision_pool=True,
        note='Generic uses all phases; specialized uses its 1v1 subset. No claim of equal phase-example counts.'))
    model=LeafReadout(output/'leaf_readouts.npz');heldout=[]
    for row in rows:
        if row['split']=='development':heldout+=read(output/'counterfactuals'/f'{row["key"]}.json')['examples']
    metrics=[]
    for phase_value in (0,1):
        rr=[r for r in heldout if r['phase']==phase_value]
        if not rr:continue
        y=np.asarray([r['label'] for r in rr])
        for name,specialized in [('generic',False),('specialized',phase_value==1)]:
            pred=np.asarray([model.raw(np.asarray(r['z']),r['remaining'],specialized) for r in rr])
            metrics.append(dict(phase=phase_value,readout=name,examples=len(rr),
                utility_mse=float(np.mean((pred-y)**2)),utility_mae=float(np.mean(np.abs(pred-y)))))
    write(output/'heldout_leaf_calibration.json',dict(metrics=metrics,
        used_for_fitting_or_hyperparameter_selection=False,scope='correlated trajectory snapshots; descriptive metrics only'))


def calibrate_timing(roots,cfg,output,common):
    path=output/'planning_calibration/decision.json'
    if path.exists():return read(path)
    selected=[r for r in roots if r['split']=='train'][:4]
    def run(label,workers):
        jobs=[dict(key=f'{r["id"]}_{label}',root_id=r['id'],repeat=0,calibration=True) for r in selected]
        rows=run_pool('planning',jobs,cfg,output,workers,[48,49,50,51],common|dict(substage='timing_calibration'))
        arms=[a for r in rows for a in r['arms']]
        return dict(simulations=sum(a['stats']['simulations'] for a in arms),
            cpu_wall=sum(a['cpu_ms'] for a in arms)/max(1e-9,sum(a['wall_ms'] for a in arms)),
            overrun_rate=float(np.mean([a['wall_ms']>cfg['decision_budget_ms'] for a in arms])))
    reference=run('serial',1);attempts=[];chosen=1
    for workers in (4,2):
        metrics=run(f'parallel{workers}',workers)
        ratio=metrics['simulations']/max(1,reference['simulations'])
        passed=metrics['cpu_wall']>=.8 and ratio>=.8 and metrics['overrun_rate']<=reference['overrun_rate']+.05
        attempts.append(dict(workers=workers,metrics=metrics,simulation_ratio=ratio,passed=passed))
        if passed:chosen=workers;break
    result=dict(workers=chosen,reference=reference,attempts=attempts,efficacy_used=False,
        fallback='serial if neither concurrency passes',gpu_shared_with_existing_W=True)
    write(path,result);return result


def summarize(output,roots,rows,planning):
    byroot={}
    for r in rows:byroot.setdefault(r['root_id'],{}).setdefault(r['candidate'],[]).append(r)
    diagnostic=[]
    for root in roots:
        cells=byroot[root['id']]
        q=np.asarray([np.mean([r['utility'] for r in cells[i]]) for i in range(len(root['candidates']))])
        success=np.asarray([np.mean([r['double_capture'] for r in cells[i]]) for i in range(len(q))])
        first=np.asarray([np.mean([r['first_capture'] if r['first_capture'] is not None else root['history'][-1]['remaining']+1 for r in cells[i]]) for i in range(len(q))])
        short=np.asarray([np.mean([r['short_scores']['8'] for r in cells[i]]) for i in range(len(q))])
        fastest=int(first.argmin());best=int(q.argmax());short_best=int(short.argmax())
        diagnostic.append(dict(root_id=root['id'],group=root['group'],split=root['split'],opponent=root['opponent'],stratum=root['stratum'],
            reference_utilities=q,reference_success=success,reference_first_penalized=first,
            fastest_first_candidate=fastest,reference_best_candidate=best,short_best_candidate=short_best,
            faster_first_but_lower_reference_utility=bool(first[fastest]+1<first[best] and q[fastest]+.02<q[best]),
            faster_first_but_lower_double_capture=bool(first[fastest]+1<first[best] and success[fastest]<success[best]),
            short_ranking_regret=float(q.max()-q[short_best]),
            streams_per_candidate=len(cells[0]),reference_scope='finite paired worlds, fixed Actor continuation; not optimal values'))
    write(output/'root_ranking_audit.json',dict(roots=diagnostic))
    rootmap={r['root_id']:r for r in diagnostic};selected=[]
    for record in planning:
        ref=rootmap[record['root_id']];q=np.asarray(ref['reference_utilities'])
        for arm in record['arms']:
            i=arm['selected_candidate']
            # A no-simulation stop is a separately identifiable action. The
            # common candidate pool contains exact stop, so score that stop.
            if i is None:
                rr=next(r for r in roots if r['id']==record['root_id'])
                i=next((j for j,a in enumerate(rr['candidates']) if np.max(np.abs(a))<1e-12),None)
            selected.append(dict(root_id=record['root_id'],group=record['group'],opponent=record['opponent'],stratum=record['stratum'],
                repeat=record['repeat'],arm=arm['arm'],reference_utility=float(q[i]) if i is not None else None,
                selection_regret=float(q.max()-q[i]) if i is not None else None,wall_ms=arm['wall_ms'],
                simulations=arm['stats']['simulations'],event_evaluations=arm['stats']['event_evaluations']))
    write(output/'planning_action_evaluation.json',dict(rows=selected,
        interpretation='held-out-world one-action diagnostic, not full closed-loop performance of each new planner'))
    stats=[]
    for opponent in ('fixed_D_kernel','rootpuct'):
        for arm in ARMS:
            rr=[r for r in selected if r['arm']==arm and r['opponent']==opponent and r['reference_utility'] is not None]
            if not rr:continue
            stats.append(dict(opponent=opponent,arm=arm,decisions=len(rr),groups=len(set(r['group'] for r in rr)),
                reference_utility=float(np.mean([r['reference_utility'] for r in rr])),
                selection_regret=float(np.mean([r['selection_regret'] for r in rr])),
                wall_p95_ms=float(np.percentile([r['wall_ms'] for r in rr],95)),
                event_evaluations=sum(r['event_evaluations'] for r in rr)))
    contrasts=[]
    for comparator in ('original8','deeper16','generic16','specialized16'):
        for opponent in ('fixed_D_kernel','rootpuct'):
            groups={}
            for r in selected:
                if r['opponent']==opponent and r['arm'] in ('event16',comparator) and r['reference_utility'] is not None:
                    groups.setdefault(r['group'],{}).setdefault(r['arm'],[]).append(r['reference_utility'])
            differences=[np.mean(v['event16'])-np.mean(v[comparator]) for v in groups.values() if 'event16' in v and comparator in v]
            if not differences:continue
            a=np.asarray(differences);rng=np.random.default_rng(seed(comparator,opponent,'bootstrap'))
            boot=rng.choice(a,size=(10000,len(a)),replace=True).mean(1)
            contrasts.append(dict(comparator=comparator,opponent=opponent,groups=len(a),difference=float(a.mean()),
                descriptive_95_interval=np.percentile(boot,[2.5,97.5]).tolist(),
                scope='exploratory map-group bootstrap, fixed Actor seed; not simultaneous or seed-population inference'))
    development=[r for r in diagnostic if r['split']=='development']
    summary=dict(complete=True,root_diagnostics=len(diagnostic),development_roots=len(development),
        development_rank_reversals=sum(r['faster_first_but_lower_reference_utility'] for r in development),
        development_double_capture_reversals=sum(r['faster_first_but_lower_double_capture'] for r in development),
        planning_statistics=stats,paired_comparisons=contrasts,
        conclusion='Exploratory diagnostic only. Check whether event-specific gains exceed ordinary depth/leaf effects; no automatic further training.',
        sealed_W_test_read=False,policy_optimizer_updates=0)
    write(output/'results.json',summary)
    lines=['# E45-X 首捕事件诊断','',f'开发状态数：{len(development)}；有限参考下首捕/团队效用排序倒挂：{summary["development_rank_reversals"]}。',
        '', '以下是一次根动作干预后固定 Single 续跑的参考效用，不是新规划器全程闭环双捕率。',
        '', '|对手|对照|参考效用|选择损失|决策P95 ms|事件评估数|','|---|---|---:|---:|---:|---:|']
    for r in stats:lines.append(f'|{r["opponent"]}|{r["arm"]}|{r["reference_utility"]:.4f}|{r["selection_regret"]:.4f}|{r["wall_p95_ms"]:.1f}|{r["event_evaluations"]}|')
    lines += ['', '地图为统计单位，候选、世界样本、根状态和重复不是独立地图。区间为探索性描述，不能证明最优排序或新颖性。',
        '原树一栏是固定相同根候选后的原评分/UCT诊断版本；deeper16 的16是最大上限，新叶仍会提前停止。',
        'common frozen Actor embedding / generic-all-phase vs residual-1v1 readouts share a training-only supervision pool.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main(args):
    global OUT,CFG
    OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise RuntimeError('Launch with CUDA_VISIBLE_DEVICES=3; all new CUDA work belongs to physical GPU3')
    CFG=register(OUT,args.smoke);verify(CFG);started=time.time()
    cpus=[int(x) for x in args.cpus.split(',')]
    common=dict(name='E45-X',state='running',pipeline_started=started)
    try:
        jobs=[]
        for row in CFG['source_maps']:
            world='AB' if seed(row['id'],'world')%2==0 else 'BA'
            for opponent in ('fixed_D_kernel','rootpuct'):
                key=f'{row["id"]}_{opponent}';jobs.append(dict(key=key,map=row,world=world,opponent=opponent,episode_seed=seed(14540261104,key)))
        collection=run_pool('collection',jobs,CFG,OUT,args.workers,cpus,common)
        roots=[read(OUT/'public_roots'/f'{i}.json') for r in collection for i in r['root_ids']]
        write(OUT/'root_manifest.json',dict(roots=[dict(id=r['id'],group=r['group'],split=r['split'],stratum=r['stratum'],opponent=r['opponent'],
            sha256=sha(OUT/'public_roots'/f'{r["id"]}.json')) for r in roots],collection=collection,
            no_imminent_root_is_explicit=True,selection_used_final_outcomes=False))
        jobs=[]
        for root in roots:
            n=CFG['training_streams'] if root['split']=='train' else CFG['development_reference_streams']
            namespace='training_labels' if root['split']=='train' else 'development_reference'
            for stream in range(n):
                for candidate in range(CFG['root_candidates']):
                    key=f'{root["id"]}_{namespace}_{stream}_{candidate}'
                    jobs.append(dict(key=key,root_id=root['id'],candidate=candidate,stream=stream,namespace=namespace))
        rows=run_pool('rollout',jobs,CFG,OUT,args.workers,cpus,common)
        fit_heads(OUT,rows,CFG)
        verify(CFG)
        planning=[]
        if args.smoke:
            jobs=[dict(key=f'{roots[0]["id"]}_engineering',root_id=roots[0]['id'],repeat=0)]
            planning=run_pool('planning',jobs,CFG,OUT,1,[48],common|dict(substage='engineering_only'))
        else:
            jobs=[dict(key=f'{r["id"]}_r{repeat}',root_id=r['id'],repeat=repeat)
                  for r in roots if r['split']=='development' for repeat in range(CFG['planning_repeats'])]
            # Timing workers occupy separate physical cores; no CPU siblings
            # shared with each other or the 48-core W training pool.
            timing=calibrate_timing(roots,CFG,OUT,common)
            timing_cpus=[48,49,50,51]
            planning=run_pool('planning',jobs,CFG,OUT,timing['workers'],timing_cpus,common|dict(substage='development_comparison'))
        summarize(OUT,roots,rows,planning);verify(CFG)
        write(OUT/'completion_audit.json',dict(complete=True,wall_seconds=time.time()-started,
            collection_episodes=len(collection),collection_physical_steps=sum(r['physical_steps'] for r in collection),
            counterfactual_episodes=len(rows),counterfactual_physical_steps=sum(r['decisions'] for r in rows),
            planning_jobs=len(planning),actor_unchanged=True,source_unchanged=True,physical_gpu=3,
            policy_optimizer_updates=0,sealed_W_test_read=False,smoke=args.smoke))
        write(OUT/'status.json',common|dict(state='complete',stage='complete',elapsed_seconds=time.time()-started,physical_gpu=3))
    except BaseException as exc:
        write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        write(OUT/'status.json',common|dict(state='failed',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=32)
    p.add_argument('--cpus',default=','.join(map(str,list(range(48,56))+list(range(56,80)))))
    p.add_argument('--smoke',action='store_true');main(p.parse_args())
