"""Fresh-development replication of the frozen E45-Z action selectors.

No policy/model training. An expensive Root transition reference is a
diagnostic selector; efficacy is one action then the frozen Single policy.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import argparse
import copy
import fcntl
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
import torch

import run_e45x_event_diagnostic as x
import run_e45z_model_calibration as z
from learning.e45x_event_diagnostic import (FrozenActor,prepare_public,boxes_of,
    root_candidates,counterfactual,advance,imminent,seed)
from learning.e45w_joint_env import PublicFeatureState
from learning.e45w_support_recovery import install
from corner_estimator_scenarios import original_model
from solvers.e45_team_adapter import public_observation
from solvers.root_audit_puct import RootPUCT

ROOT=x.ROOT;OUT=ROOT/'current/e45z_independent_confirmation_20261104'
CFG=None;ACTOR=None;CACHE={}
CPUS=list(range(48,56))+list(range(56,80))
NAMESPACE='Z-confirm-independent-long-reference-v1'


def geometry(row):return x.digest(x.read(row['path'])['map'])


def select_maps(entries,previous):
    used_ids={r['id'] for r in previous};used_geometry={geometry(r) for r in previous}
    result=[r|dict(audit_split='development') for r in entries if r['id'] not in used_ids and geometry(r) not in used_geometry]
    result.sort(key=lambda r:r['id'])
    if len({geometry(r) for r in result})!=len(result):raise RuntimeError('Duplicate new geometries')
    return result


def reference_jobs(selections,streams):
    jobs=[]
    for row in selections:
        for candidate in sorted({row['chosen_D_proxy_action'],row['chosen_Root_proxy_action']}):
            for stream in range(streams):
                jobs.append(dict(key=f'{row["root_id"]}_{candidate}_{stream}',root_id=row['root_id'],
                    candidate=candidate,stream=stream,namespace=NAMESPACE))
    return jobs


def initialize(cfg,output,queue):
    global CFG,OUT,ACTOR,CACHE
    os.sched_setaffinity(0,{queue.get()});torch.set_num_threads(1)
    CFG=cfg;OUT=Path(output);CACHE={};z.X=OUT
    install(OUT/'belief_failures');ACTOR=FrozenActor(cfg['actor_path'],'cuda:0')
    warm=x.read(cfg['warmup_map']['path'])
    packet=dict(id='ZC-training-warmup',history=warm['history'],controller_seed=seed('ZC-warmup'))
    public=prepare_public(packet,{k:warm[k] for k in ('map','speed')},ACTOR)
    ACTOR.infer(public.own.features(packet['history'][-1]));torch.cuda.synchronize()


def prepared(root):
    if root['id'] not in CACHE:
        if len(CACHE)>=2:CACHE.pop(next(iter(CACHE)))
        raw=x.read(root['case_path'])
        CACHE[root['id']]=prepare_public(root,{k:raw[k] for k in ('map','speed')},ACTOR)
        install(OUT/'belief_failures')
    return CACHE[root['id']]


def collect(task):
    raw=x.read(task['map']['path']);boxes=boxes_of(raw);p=original_model(boxes,raw['speed'])
    own=PublicFeatureState(boxes,raw['speed'],task['episode_seed'])
    for obs in raw['history']:own.observe(obs)
    state=np.asarray(next(w['initial'] for w in raw['worlds'] if w['id']==task['world']),float)[:,None]
    obs=public_observation(int(state[8,0]),state[4:8,0].reshape(2,2),state[:4,0].reshape(2,2),
        state[p.active_idxs[2:],0]>.5,state[p.active_idxs[:2],0]>.5,boxes)
    if obs!=raw['history'][-1]:raise RuntimeError('New collection handover mismatch')
    history=copy.deepcopy(raw['history']);roots=[];private={};steps=0;near=False;first=None;shields=0
    def save(stratum):
        root=x.packet(task,copy.deepcopy(history),stratum)
        root['source_policy']='Single17 frozen; Z-confirm new collection'
        roots.append(root)
        # Private evaluator-only sidecar is never passed to prepare_public,
        # Actor, candidate generation or either action selector.
        private[root['id']]=dict(target_positions=state[:4,0].reshape(2,2).copy(),
            step=int(state[8,0]),use='offline support diagnostic only')
    save('handover')
    prep=type('CollectionModel',(),{})();prep.problem=p;prep.case=raw;prep.graphs=[None,None]
    opponent=RootPUCT([None]*4,None,3000,0.,0.)
    while True:
        if steps and not near and imminent(own,obs):save('public_imminent_first');near=True
        action,_=ACTOR.action(own.features(obs),obs,raw['speed'],boxes,
            seed(task['key'],'collection_actor',obs['step']))
        state,obs,reward,done,captures,shield=advance(prep,state,action,opponent,[0,0],[0,0],
            (task['key'],'collection_enemy',obs['step']))
        steps+=1;shields+=shield
        if captures and first is None:first=steps
        if any(obs['active_p']):own.observe(obs)
        history.append(obs)
        if done:break
    for root in roots:
        public=prepared(root);actions,sources=root_candidates(root,public,ACTOR,6)
        root['candidates']=actions;root['candidate_sources']=sources
        x.write(OUT/'public_roots'/f'{root["id"]}.json',root)
        x.write(OUT/'private_support_audit'/f'{root["id"]}.json',private[root['id']])
    row=dict(task=task,root_ids=[r['id'] for r in roots],imminent_root_found=near,
        physical_steps=steps,first_capture=first,opponent_shields=shields,complete=True)
    x.write(OUT/'collection'/f'{task["key"]}.json',row);return row


def select(job):
    row=z.worker(job)
    # The immutable Z worker performs exactly the original 128-D versus
    # 32-current-world x 4-Root selector, including six-action physics checks.
    # No long-return labels are available at this point.
    root=x.read(OUT/'public_roots'/f'{job["root_id"]}.json');public=prepared(root)
    truth=x.read(OUT/'private_support_audit'/f'{root["id"]}.json')['target_positions']
    diagnostics=[]
    for j,(graph,joint) in enumerate(zip(public.graphs,public.joints)):
        mass=joint.sum(1);mass=mass/mass.sum();positive=mass>0
        distance=np.linalg.norm(graph[0]-np.asarray(truth[j]),axis=1)
        diagnostics.append(dict(target=j,currently_visible=any(q is not None for q in root['history'][-1]['reports'][j]),
            nearest_positive_support_distance=float(distance[positive].min()),
            posterior_mass_within_1=float(mass[distance<=1.].sum()),
            posterior_mass_within_2=float(mass[distance<=2.].sum()),
            posterior_mass_within_4=float(mass[distance<=4.].sum())))
    selection=dict(root_id=row['root_id'],group=row['group'],family=row['family'],stratum=row['stratum'],
        chosen_D_proxy_action=row['chosen_D_proxy_action'],chosen_Root_proxy_action=row['chosen_Root_proxy_action'],
        root_shields=row['root_shields'],Root_search_calls=row['Root_search_calls'],
        physical_one_step_audits=row['physical_one_step_audits'],support_diagnostic=diagnostics,
        selection_label_access=False)
    x.write(OUT/'selections'/f'{root["id"]}.json',selection);return selection


def rollout(job):
    root=x.read(OUT/'public_roots'/f'{job["root_id"]}.json')
    result=counterfactual(root,prepared(root),ACTOR,root['candidates'][job['candidate']],
        job['stream'],job['namespace'],examples=False)
    result.pop('examples')
    row=job|result|dict(group=root['group'],family=root['family'],stratum=root['stratum'])
    x.write(OUT/'counterfactuals'/f'{job["key"]}.json',row);return row


def dispatch(stage,job):
    try:return {'collection':collect,'selection':select,'rollout':rollout}[stage](job)
    except BaseException as exc:
        x.write(OUT/'failures'/f'{stage}_{job.get("key",job.get("root_id"))}.json',
            dict(job=job,error=repr(exc),traceback=traceback.format_exc()));raise


def pool(stage,jobs,cfg,output,workers):
    if not jobs:return []
    ctx=mp.get_context('spawn');queue=ctx.Queue();workers=min(workers,len(CPUS),len(jobs))
    for core in CPUS[:workers]:queue.put(core)
    started=time.time();rows=[];pending=iter(jobs);active={};failed=True
    executor=ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=initialize,
        initargs=(cfg,str(output),queue))
    try:
        while True:
            while len(active)<workers*2:
                job=next(pending,None)
                if job is None:break
                active[executor.submit(dispatch,stage,job)]=job
            if not active:break
            done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for future in done:active.pop(future);rows.append(future.result())
            x.write(output/'status.json',dict(state='running',phase=stage,completed=len(rows),planned=len(jobs),
                workers=workers,elapsed_seconds=time.time()-started,timestamp=time.time(),physical_gpu=3))
        failed=False
    finally:
        processes=list((executor._processes or {}).values());executor.shutdown(wait=not failed,cancel_futures=failed)
        if failed:
            for process in processes:
                if process.is_alive():process.terminate()
            for process in processes:process.join(timeout=2)
        queue.close()
    return rows


def evaluate(selections,rows,streams):
    lookup={(r['root_id'],r['candidate'],r['stream']):r for r in rows};pairs=[]
    for selection in selections:
        rid=selection['root_id'];a=selection['chosen_D_proxy_action'];b=selection['chosen_Root_proxy_action']
        for stream in range(streams):
            left=lookup[rid,a,stream];right=lookup[rid,b,stream]
            if left['initial_world_sha256']!=right['initial_world_sha256']:raise RuntimeError('Unpaired initial worlds')
            pairs.append(dict(root_id=rid,group=selection['group'],family=selection['family'],stratum=selection['stratum'],
                stream=stream,D_action=a,Root_action=b,D_utility=left['utility'],Root_utility=right['utility'],
                D_success=float(left['double_capture']),Root_success=float(right['double_capture']),
                utility_gain=right['utility']-left['utility'],
                success_gain=float(right['double_capture'])-float(left['double_capture'])))
    summaries=[]
    for stratum in ('handover','public_imminent_first'):
        subset=[r for r in pairs if r['stratum']==stratum];groups={}
        for row in subset:groups.setdefault(row['group'],[]).append(row)
        group_rows=[]
        for group,rr in sorted(groups.items()):
            group_rows.append(dict(group=group,family=rr[0]['family'],
                **{name:float(np.mean([r[name] for r in rr])) for name in
                    ('D_utility','Root_utility','D_success','Root_success','utility_gain','success_gain')}))
        metrics={}
        for name in ('D_utility','Root_utility','D_success','Root_success','utility_gain','success_gain'):
            values=np.asarray([r[name] for r in group_rows])
            if not len(values):metrics[name]=dict(mean=None,descriptive_95_interval=None);continue
            rng=np.random.default_rng(seed('ZC-map-bootstrap',stratum,name))
            interval=np.percentile(rng.choice(values,(10000,len(values)),replace=True).mean(1),[2.5,97.5])
            metrics[name]=dict(mean=float(values.mean()),descriptive_95_interval=interval.tolist())
        family={f:float(np.mean([r['utility_gain'] for r in group_rows if r['family']==f]))
            for f in sorted({r['family'] for r in group_rows})}
        summaries.append(dict(stratum=stratum,maps=len(groups),roots=len({r['root_id'] for r in subset}),
            metrics=metrics,map_results=group_rows,family_utility_gains=family,
            positive_maps=sum(r['utility_gain']>0 for r in group_rows)))
    primary=summaries[1];metric=primary['metrics']['utility_gain']
    gate=bool(primary['maps']>=4 and metric['mean'] is not None and metric['mean']>0
        and metric['descriptive_95_interval'][0]>0 and primary['positive_maps']>primary['maps']/2
        and sum(v>0 for v in primary['family_utility_gains'].values())>=2)
    return dict(complete=True,paired_reference_rows=pairs,summaries=summaries,
        advance_to_one_cheap_predictor_pilot=gate,
        interpretation='Fresh-to-X/Y/Z development geometries; posterior-world one-action intervention, not deployed planner efficacy or sealed test.',
        gate_scope='Resource allocation rule, not proof of sufficiency, novelty or general failure.')


def register(output,smoke):
    previous=x.read(x.OUT/'experiment.json');x.verify(previous)
    w=x.read(x.W/'experiment.json');old_z=x.read(z.OUT/'experiment.json')
    training=next(r for r in previous['source_maps'] if r['audit_split']=='train')
    maps=[training|dict(audit_split='development')] if smoke else select_maps(w['validation_entries'],previous['source_maps'])
    if not smoke:
        assert len(maps)==6
        assert all(sum(r['family']==family for r in maps)==2 for family in ('long_wall','wall_corner','multi_route'))
    paths=[Path(__file__),ROOT/'code/tests/test_e45z_confirmation.py',ROOT/'code/watch_e45z_confirmation.py',
        ROOT/'code/E45Z_CONFIRMATION_PROTOCOL_20261104.md']
    cfg=dict(name='E45-Z-confirm',smoke=smoke,source_maps=maps,warmup_map=training,
        actor_path=previous['actor_path'],actor_sha256=previous['actor_sha256'],
        source_hashes=old_z['source_hashes']|{str(p):x.sha(p) for p in paths},
        physical_gpu=3,logical_device='cuda:0',policy_optimizer_updates=0,opponent='rootpuct',
        root_candidates=6,forecast_count=8 if smoke else 128,current_worlds=2 if smoke else 32,
        replicates=2 if smoke else 4,reference_streams=1 if smoke else 16,
        selection_rule='immutable E45-Z worker; no label access or threshold tuning',
        primary='imminent Root-reference minus D-reference raw utility; eligible maps equally weighted',
        missing='retain all episodes and missing imminent states; do not replace or count AB/BA as independent maps',
        inference_scope='no model training; diagnostic Root simulator is expensive; no same-time superiority claim',
        truth_sidecar='offline support geometry only; no private state enters selectors or Actor',
        W_test_read=False,namespace=NAMESPACE)
    x.write(output/'experiment.json',cfg)
    for p in paths:
        snapshot=output/'source_snapshot'/p.relative_to(ROOT/'code');snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes(p.read_bytes())
    return cfg


def verify(cfg):
    for path,h in cfg['source_hashes'].items():
        if x.sha(path)!=h:raise RuntimeError('Frozen source changed: '+path)
    if x.sha(cfg['actor_path'])!=cfg['actor_sha256']:raise RuntimeError('Actor changed')
    for row in cfg['source_maps']+[cfg['warmup_map']]:
        if x.sha(row['path'])!=row['sha256']:raise RuntimeError('Map changed')


def main(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise RuntimeError('Use physical GPU3 only')
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    lock=(output/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'experiment.json').exists():raise RuntimeError('Use a fresh scientific output directory')
    cfg=register(output,args.smoke);verify(cfg);started=time.time()
    tasks=[]
    for row in cfg['source_maps']:
        for world in (['AB'] if args.smoke else ['AB','BA']):
            key=x.digest(dict(namespace='ZC-source',map=row['id'],world=world))
            tasks.append(dict(key=key,map=row,world=world,opponent='rootpuct',episode_seed=seed(key,'collection')))
    collected=pool('collection',tasks,cfg,output,args.workers)
    roots=[x.read(output/'public_roots'/f'{rid}.json') for row in collected for rid in row['root_ids']]
    manifest=[dict(id=r['id'],group=r['group'],family=r['family'],stratum=r['stratum'],
        sha256=x.sha(output/'public_roots'/f'{r["id"]}.json')) for r in roots]
    x.write(output/'root_manifest.json',dict(roots=manifest,collection=collected))
    def verify_roots():
        for r in manifest:assert x.sha(output/'public_roots'/f'{r["id"]}.json')==r['sha256'],r['id']
    selection_jobs=[dict(root_id=r['id'],worlds=cfg['current_worlds'],replicates=cfg['replicates'],
        forecast_count=cfg['forecast_count'],output=str(output/'one_step')) for r in roots]
    selections=pool('selection',selection_jobs,cfg,output,args.workers)
    # Freeze every selector decision before generating any long-return labels.
    x.write(output/'selection_manifest.json',dict(rows=selections,
        frozen_before_long_returns=True,selection_hashes={r['root_id']:x.sha(output/'selections'/f'{r["root_id"]}.json') for r in selections}))
    jobs=reference_jobs(selections,cfg['reference_streams']);rows=pool('rollout',jobs,cfg,output,args.workers)
    result=evaluate(selections,rows,cfg['reference_streams']);result['collection']=dict(episodes=len(collected),
        imminent_found=sum(r['imminent_root_found'] for r in collected),missing=[r['task'] for r in collected if not r['imminent_root_found']])
    x.write(output/'results.json',result);verify(cfg);verify_roots()
    frozen=x.read(output/'selection_manifest.json')['selection_hashes']
    for rid,h in frozen.items():assert x.sha(output/'selections'/f'{rid}.json')==h,rid
    x.write(output/'completion_audit.json',dict(complete=True,smoke=args.smoke,elapsed_seconds=time.time()-started,
        maps=len(cfg['source_maps']),collection_episodes=len(collected),public_roots=len(roots),
        unique_reference_episodes=len(rows),reference_physical_steps=sum(r['decisions'] for r in rows),
        collection_physical_steps=sum(r['physical_steps'] for r in collected),
        selector_Root_calls=sum(r['Root_search_calls'] for r in selections),
        selector_physical_steps=sum(r['physical_one_step_audits'] for r in selections),
        opponent_shields=sum(r['opponent_shields'] for r in collected+rows)+sum(r['root_shields'] for r in selections),
        source_unchanged=True,actor_unchanged=True,public_roots_unchanged=True,selections_unchanged=True,
        policy_optimizer_updates=0,physical_gpu=3,W_test_read=False))
    x.write(output/'status.json',dict(state='complete',phase='complete',timestamp=time.time(),
        physical_gpu=3,advance_to_predictor_pilot=result['advance_to_one_cheap_predictor_pilot']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=32);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    try:main(args)
    except BaseException as exc:
        x.write(args.output/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        x.write(args.output/'status.json',dict(state='failed',error=repr(exc)));raise
