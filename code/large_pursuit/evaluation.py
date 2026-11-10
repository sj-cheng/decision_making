from __future__ import annotations
import gzip
import json
import time
from pathlib import Path
import numpy as np
from .core import Config,World,PublicMap,from_initial,move,State,Belief
from .methods import Controller,evader_action,rootpuct_evader,MotionKernel
from .training import load_model,read_case,write_json
from .provenance import bind_job,sha
from .motion import planning_model


def evaluate(job):
    folder=Path(job['folder']);folder.mkdir(parents=True,exist_ok=True)
    identity=bind_job(folder,job)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    case,public=read_case(job['case']);world=World(public.cfg,case['boxes']);state=from_initial(case,job['initial'])
    for path_key,hash_key in (('checkpoint','checkpoint_sha256'),('opponent_checkpoint','opponent_checkpoint_sha256')):
        if job.get(hash_key) and sha(job[path_key])!=job[hash_key]:raise RuntimeError('Frozen evaluation checkpoint changed')
    model=load_model(job['checkpoint']) if job.get('checkpoint') else None
    opponent_model=load_model(job['opponent_checkpoint']) if job.get('opponent_checkpoint') else None
    controller=Controller(public,job['method'],job['seed'],model,job.get('simulations',64),job.get('budget_ms'))
    rng=np.random.default_rng(job['environment_seed']);motion=MotionKernel(public,rng);trajectory=[];survival=[int(state.alive.sum())]
    capture_times=np.full(public.cfg.n,public.cfg.horizon,float);latencies=[];opponent_latencies=[];simulations=[];timeouts=[];zeros=[];prediction_errors=[]
    while not world.done(state):
        obs=world.observe(state);pa=controller.act(obs)
        t=time.perf_counter()
        ea=rootpuct_evader(public,state,rng,job.get('rootpuct_simulations',3000)) if job['opponent']=='RootPUCT' else (evader_action(public,state,rng,opponent_model) if opponent_model is not None else motion.actions(state))
        opponent_latencies.append(1000*(time.perf_counter()-t))
        nxt=world.transition(state,pa,ea)
        caught=state.alive & ~nxt.alive;capture_times[caught]=nxt.t
        stats=controller.last_stats
        trajectory.append(dict(t=state.t,p=state.p.tolist(),e=state.e.tolist(),alive=state.alive.tolist(),
                               pa=pa.tolist(),ea=ea.tolist(),stats=stats))
        latencies.append(stats['elapsed_ms']);simulations.append(stats['simulations']);timeouts.append(stats['timeout']);zeros.append(stats['zero_complete_simulations'])
        state=nxt;survival.append(int(state.alive.sum()))
    # Curves carry the last survival count to the full horizon, including full elimination.
    survival+= [survival[-1]]*(public.cfg.horizon+1-len(survival))
    temporary=folder/'trajectory.json.gz.tmp'
    with gzip.open(temporary,'wt',encoding='utf-8',compresslevel=3) as f:
        json.dump(dict(job=job,steps=trajectory,final=dict(p=state.p.tolist(),e=state.e.tolist(),alive=state.alive.tolist(),t=state.t)),f)
    temporary.replace(folder/'trajectory.json.gz')
    diagnostics=prediction_diagnostics(case,trajectory,state,job['seed'])
    result=dict(identity=identity['digest'],n=case['n'],map_id=case['id'],suite=case['suite'],family=case['family'],initial=job['initial'],
        method=job['method'],seed=job['seed'],opponent=job['opponent'],budget=job['budget'],
        capture_fraction=float(1-state.alive.mean()),elimination=float(not state.alive.any()),
        completion_cost=state.t if not state.alive.any() else public.cfg.horizon,
        mean_target_capture_cost=float(capture_times.mean()),capture_times=capture_times.tolist(),survival=survival,
        controller_setup_ms=controller.setup_ms,
        latency_mean_ms=float(np.mean(latencies)),latency_p95_ms=float(np.percentile(latencies,95)),
        latency_max_ms=float(np.max(latencies)),timeouts=int(sum(timeouts)),zero_complete=int(sum(zeros)),
        decisions=len(latencies),decision_latencies_ms=latencies,simulations_by_decision=simulations,
        training_seed=job['seed'] if job.get('checkpoint') else None,controller_seed=job['seed'],simulations_total=int(sum(simulations)),simulations_mean=float(np.mean(simulations)),
        opponent_latency_mean_ms=float(np.mean(opponent_latencies)),
        prediction_diagnostics=diagnostics,
        belief_recoveries=controller.belief.recoveries,trajectory=str(folder/'trajectory.json.gz'),trajectory_sha256=sha(folder/'trajectory.json.gz'))
    write_json(folder/'result.json',result);return result


def prediction_diagnostics(case,trajectory,final,seed):
    """Offline only: independent map caches/RNG never enter timed controller calls."""
    cfg=Config.scale(case['n']);public=PublicMap(cfg,case['boxes']);world=World(cfg,public.boxes)
    belief=Belief(public,seed);rng=np.random.default_rng(912371+seed);errors={}
    for i,row in enumerate(trajectory):
        state=State(np.array(row['p']),np.array(row['e']),np.array(row['alive'],bool),row['t'])
        nxt=final if i+1==len(trajectory) else State(np.array(trajectory[i+1]['p']),np.array(trajectory[i+1]['e']),np.array(trajectory[i+1]['alive'],bool),trajectory[i+1]['t'])
        obs=world.observe(state);belief.update(obs);posterior=[];conditional=[]
        # Integrate mode switching and the exact discrete conditional kernel by
        # independent Monte Carlo draws. Truth is used ONLY for offline motion diagnosis.
        for _ in range(4):
            predicted,_=planning_model(public).advance(belief.samples,belief.modes,belief.anchors[:,None],rng)
            actual=np.repeat(state.e[:,None,:],belief.count,axis=1)
            local,_=planning_model(public).advance(actual,belief.modes,belief.anchors[:,None],rng)
            posterior.append(predicted.mean(1));conditional.append(local.mean(1))
        predictions={'posterior_position':np.mean(posterior,axis=0),'conditional_motion':np.mean(conditional,axis=0)}
        for name,prediction in predictions.items():
            for visible,label in ((True,'visible'),(False,'hidden')):
                selected=state.alive & (obs['seen']==visible)
                errors.setdefault(name+'_'+label,[]).extend(np.sum((prediction[selected]-nxt.e[selected])**2,axis=1).tolist())
    return dict(scope='offline; never fed to policy or timed action caches',monte_carlo_draws_per_particle=4,
                **{k:dict(rmse=float(np.sqrt(np.mean(v))) if v else None,targets=len(v)) for k,v in errors.items()})


def summarize(out,rows):
    """Paired maps; shared resampling across seed/opponent/budget, global max interval."""
    from collections import defaultdict
    if not rows:raise ValueError('No episodes to summarize')
    out=Path(out);groups=defaultdict(list)
    for row in rows:groups[(row['n'],row['suite'],row['seed'],row['opponent'],row['budget'])].append(row)
    from .protocol import PRIMARY
    primary=PRIMARY
    descriptive=[];contrasts=[];by_geometry=defaultdict(list)
    for key,group in groups.items():
        methods=defaultdict(list)
        for row in group:methods[row['method']].append(row)
        for method,values in methods.items():
            latencies=[v for r in values for v in r['decision_latencies_ms']]
            descriptive.append(dict(group=key,method=method,episodes=len(values),
                **{field:float(np.mean([r[field] for r in values])) for field in ('capture_fraction','elimination','completion_cost','mean_target_capture_cost','simulations_mean')},
                latency_mean_ms=float(np.mean(latencies)),latency_p95_ms=float(np.percentile(latencies,95)),latency_max_ms=max(latencies),
                timeouts=sum(r['timeouts'] for r in values),zero_complete=sum(r['zero_complete'] for r in values),
                survival=np.mean([r['survival'] for r in values],axis=0).tolist()))
        lookup={}
        for r in group:
            entry=(r['map_id'],r['initial'],r['method'])
            if entry in lookup:raise ValueError('Duplicate paired episode')
            lookup[entry]=r['capture_fraction']
        maps=sorted({r['map_id'] for r in group})
        for left,right in primary:
            if left not in methods or right not in methods:raise ValueError('Missing registered primary comparison')
            values=[]
            for map_id in maps:
                try:values.append(np.mean([lookup[(map_id,i,left)]-lookup[(map_id,i,right)] for i in (0,1)]))
                except KeyError:raise ValueError('Incomplete paired map')
            record=dict(group=key,left=left,right=right,maps=len(maps),difference=float(np.mean(values)))
            contrasts.append(record);by_geometry[key[:2]].append((len(contrasts)-1,maps,np.array(values)))
    rng=np.random.default_rng(20261108);bootstraps=4000;maximum=np.zeros(bootstraps)
    for key,items in sorted(by_geometry.items()):
        maps=items[0][1]
        if any(item[1]!=maps for item in items):raise ValueError('Inconsistent map clusters across budgets/opponents/seeds')
        matrix=np.stack([item[2] for item in items],axis=1);center=matrix.mean(0)
        for start in range(0,bootstraps,64):
            count=min(64,bootstraps-start);indices=rng.integers(len(maps),size=(count,len(maps)))
            deviation=np.abs(matrix[indices].mean(1)-center)
            maximum[start:start+count]=np.maximum(maximum[start:start+count],deviation.max(1))
    radius=float(np.quantile(maximum,.95)) if contrasts else 0.
    for record in contrasts:
        record['simultaneous_ci95']=[record['difference']-radius,record['difference']+radius]
        record['family']='all preregistered primary contrasts, both scales, ID/OOD, all seeds/opponents/budgets'
    write_json(out/'summary.json',dict(descriptive=descriptive,primary_comparisons=contrasts,
        inference='two initials clustered by map; same map resample across seed/opponent/budget; global maximum centered bootstrap deviation; no opponent averaging'))
