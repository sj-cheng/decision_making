"""Isolated E45-I code/trajectory audit. Does not alter the frozen experiment."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import collections
import concurrent.futures as futures
import json
import multiprocessing as mp
from pathlib import Path
import shutil
from types import SimpleNamespace
import numpy as np
import torch
import run_e45i_baselines as run
from learning.e45i_official_mappo import (PublicState, policy, actor_inputs, apf_action,
    settings, finish_episodes)
from learning.e45b_joint_network import Network
from solvers.frozen_d_search_interface import PublicMap,move
from estimator_audit_geometry import point_legal,blocked
from run_frozen_d_closed_loop import save,sha
from run_e44_parallel import choose_cpus

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'current/e45i_official_baselines_20261031'
OUT=ROOT/'current/e45i_code_audit_20261031'
LAST=None


def prepare():
    run.OUT=SOURCE;run.verify();OUT.mkdir(exist_ok=True)
    save(OUT/'scope.json',dict(source_experiment=str(SOURCE),source_experiment_sha256=sha(SOURCE/'experiment.json'),
        code_sha256=sha(__file__),scope='code audit, public-trajectory replay and validation-only diagnostics',
        original_scientific_results_unchanged=True))


def fake_controller(boxes,target):
    estimator=SimpleNamespace(particles=np.repeat(np.asarray(target)[None],64,axis=0),particle_weights=np.full(64,1/64))
    return SimpleNamespace(public=PublicMap(boxes,1.1),belief=SimpleNamespace(known=[np.asarray(target),None],
        estimators=[estimator,None],last_stats={}),last_stats={})


def wall_counterexample():
    boxes=np.array([[[-1.,1.],[-5.,5.]]]);goal=np.array([4.,0.])
    gains=run.read(SOURCE/'apf_selected.json')['coefficients'];results=[]
    for mode in ('fixed','random_positive_gains','public_route_control'):
        own=fake_controller(boxes,goal);positions=np.array([[-3.,.2],[-8.,-8.]])
        trace=[positions[0].copy()];rng=np.random.default_rng(17)
        for step in range(100):
            obs=dict(positions=positions.tolist(),active_p=[True,False],active_e=[True,False])
            if mode=='public_route_control':
                action=move(positions[0],goal,1.1,own.public.vertices,own.public.dist,boxes,np.zeros(2,np.int64))
            else:
                raw=rng.uniform(-8,8,(2,4)) if mode=='random_positive_gains' else None
                action=apf_action(own,obs,raw,gains)[0]
            positions[0]+=action;trace.append(positions[0].copy())
            if np.linalg.norm(positions[0]-goal)<1.:break
        results.append(dict(mode=mode,captured=bool(np.linalg.norm(positions[0]-goal)<1.),
            steps=len(trace)-1,maximum_absolute_y=float(np.max(np.abs(np.array(trace)[:,1]))),trace=np.array(trace).tolist()))
    save(OUT/'wall_counterexample.json',dict(task='fully observed stationary single target, one active pursuer, one long wall; diagnostic only',
        boxes=boxes.tolist(),target=goal.tolist(),results=results,
        invariant='Inside wall vertical span, APF attraction and tangential y components both point toward goal.y; positive gains cannot choose a route away from it. With |dy|<=1.1 the |y|<=1.1 strip is invariant.'))


def replay_apf():
    run.configure(SOURCE);groups=collections.defaultdict(lambda:collections.Counter());rows=[]
    for path in sorted((SOURCE/'static_test/episodes').glob('*.json')):
        payload=run.read(path);summary=payload['summary']
        if summary['arm']!='APF':continue
        case=run.read(summary['case']);own=PublicState(case['map'],case['speed'],summary['seed'])
        for obs in case['history'][:-1]:own.observe(obs)
        counts=collections.Counter()
        for frame in payload['trace']:
            obs=frame['observation'];own.observe(obs)
            assert own.belief.signature()==frame['belief_signature']
            positions=np.asarray(obs['positions'])
            for i,j in frame['stats']['assignment'].items():
                i,j=int(i),int(j)
                if own.belief.known[j] is not None:continue
                e=own.belief.estimators[j];goal=np.average(e.particles,axis=0,weights=e.particle_weights)
                counts['hidden_assigned_decisions']+=1
                counts['mean_inside_obstacle']+=int(not point_legal(goal,own.public.boxes))
                counts['straight_goal_path_blocked']+=int(blocked(positions[i],goal,own.public.boxes))
            counts['active_actions']+=sum(obs['active_p'])
            counts['shortened_actions']+=frame['stats']['safety_shortened']
        group=(summary['suite'],summary['opponent']);groups[group].update(counts)
        rows.append(dict(key=summary['key'],**counts))
    save(OUT/'apf_public_replay.json',dict(episodes=len(rows),independent_replay_exact=True,
        grouped=[dict(suite=s,opponent=o,**v) for (s,o),v in sorted(groups.items())],per_episode=rows))


def permutation_probe():
    run.configure(SOURCE);manifest=run.read(SOURCE/'bank/manifest.json');entries=manifest['validation'][::2]
    models=[]
    for seed in (17,29,43):
        folder=SOURCE/'learning'/f'RMAPPO_{seed}';last=run.read(folder/'resume.json')['checkpoint']
        model,_=policy('RMAPPO');model.actor.load_state_dict(torch.load(last,map_location='cpu',weights_only=False)['actor']);model.actor.eval()
        models.append((seed,model.actor))
    single=np.load(run.CROOT/'learning/x20/round_12/models/single_17.npz')
    net=Network(str(single['kind']),int(single['width']));net.load_state_dict({k:torch.tensor(single[k],dtype=torch.float32) for k in net.state_dict()});net.eval()
    rows=[];rng=np.random.default_rng(123)
    for entry in entries:
        case=run.read(entry['path']);own=PublicState(case['map'],case['speed'],17)
        for obs in case['history']:own.observe(obs)
        original=own.features(obs);variants=[]
        for index in range(33):
            x={k:v.copy() for k,v in original.items()}
            if index:
                for j in range(2):
                    order=rng.permutation(64);x['particles'][j]=x['particles'][j,order];x['weights'][j]=x['weights'][j,order]
            variants.append(x)
        with torch.no_grad():
            params,_=net({k:torch.tensor(np.stack([x[k] for x in variants]),dtype=torch.float32) for k in original})
            direct=torch.tanh(params[1][:,0]).numpy()
            single_delta=float(np.max(np.abs(direct-direct[:1])))
            for seed,actor in models:
                obs_batch=np.concatenate([actor_inputs(x) for x in variants]);hidden=np.zeros((66,1,64),np.float32)
                raw,_,_=actor(obs_batch,hidden,np.ones((66,1),np.float32),deterministic=True)
                actions=np.tanh(raw.numpy()).reshape(33,2,2)
                rows.append(dict(case=entry['id'],seed=seed,baseline_action=actions[0].tolist(),
                    max_permutation_action_change=float(np.max(np.abs(actions-actions[:1]))),
                    single_direct_max_change=single_delta))
    save(OUT/'permutation_probe.json',dict(scope='same weighted particle set, only row order changed, zero GRU state to isolate current-input sensitivity',rows=rows))


def diagnostics_factory(case,task):
    global LAST
    own=run.factory(case,task)
    if task.get('audit_mode'):
        own.deterministic=task['audit_mode']=='mean'
        own.distributions=[]
        def hook(module,args,dist):
            own.distributions.append((dist.mean.detach().numpy().copy(),dist.stddev.detach().numpy().copy()))
        own.audit_hook=own.actor.act.action_out.register_forward_hook(hook)
        original=own.act
        def act(obs,fixed_simulations=None):
            action=original(obs,fixed_simulations);mean,std=own.distributions[-1];active=np.asarray(obs['active_p'],bool)
            own.last_stats['audit_mean_abs']=float(np.abs(mean[active]).mean())
            own.last_stats['audit_std']=float(std[active].mean())
            own.last_stats['audit_normalized_action_max']=float(np.max(np.abs(action[active]),axis=1).mean()/case['speed'])
            return action
        own.act=act;LAST=own
        torch.manual_seed(task['seed']+120261031)
    return own


def init_worker(output,queue,parent):
    run.worker_init(output,queue,parent)
    run.base.make_controller=diagnostics_factory


def diagnostic_job(task):
    global LAST
    LAST=None
    try:
        row=run.evaluate_job(task);payload=run.read(row['episode'])
        active=[frame for frame in payload['trace']]
        return dict(kind=row['arm'],seed=row['train_seed'],version=task['audit_version'],mode=task['audit_mode'],
            opponent=row['opponent'],case=row['group'],capture=row['double_capture'],utility=row['utility'],
            decisions=row['decisions'],mean_abs=float(np.mean([f['stats']['audit_mean_abs'] for f in active])),
            std=float(np.mean([f['stats']['audit_std'] for f in active])),
            normalized_action_max=float(np.mean([f['stats']['audit_normalized_action_max'] for f in active])),
            public_replay_exact=row['public_replay_exact'])
    finally:
        if LAST is not None:LAST.audit_hook.remove()


def deploy():
    workspace=OUT/'deployment';workspace.mkdir(exist_ok=True);(workspace/'bank').mkdir(exist_ok=True)
    for src,dest in [(SOURCE/'bank/manifest.json',workspace/'bank/manifest.json'),(SOURCE/'apf_selected.json',workspace/'apf_selected.json')]:
        shutil.copy2(src,dest)
    run.configure(workspace)
    entries=run.read(SOURCE/'bank/manifest.json')['validation'][::2]
    tasks=[]
    for kind in run.RL:
        for seed in run.SEEDS:
            folder=SOURCE/'learning'/f'{kind}_{seed}'
            paths={'initial':folder/'checkpoints/steps_0.pt','final':Path(run.read(folder/'resume.json')['checkpoint'])}
            for version,path in paths.items():
                for mode in ('mean','sample'):
                    for entry in entries:
                        for opponent in run.base.OPPONENTS:
                            task=run.request(entry,'AB',opponent,kind,seed,f'audit_{version}_{mode}',path)
                            tasks.append(task|dict(audit_mode=mode,audit_version=version))
    cpus=choose_cpus(16);ctx=mp.get_context('spawn');queue=ctx.Queue()
    for cpu in cpus:queue.put(cpu)
    with futures.ProcessPoolExecutor(max_workers=16,mp_context=ctx,initializer=init_worker,
            initargs=(str(workspace),queue,os.getpid())) as pool:
        rows=[]
        for row in pool.map(diagnostic_job,tasks):
            rows.append(row)
            if len(rows)%24==0:print('diagnostic episodes',len(rows),'/',len(tasks),flush=True)
    queue.close();groups=collections.defaultdict(list)
    for row in rows:groups[row['kind'],row['version'],row['mode'],row['opponent']].append(row)
    grouped=[]
    for (kind,version,mode,opp),rr in sorted(groups.items()):
        grouped.append(dict(kind=kind,version=version,mode=mode,opponent=opp,episodes=len(rr),
            capture=sum(x['capture'] for x in rr),rate=float(np.mean([x['capture'] for x in rr])),
            mean_abs=float(np.mean([x['mean_abs'] for x in rr])),std=float(np.mean([x['std'] for x in rr])),
            action_fraction=float(np.mean([x['normalized_action_max'] for x in rr]))))
    save(OUT/'deployment_diagnostic.json',dict(scope='validation-only, six histories, AB world; new diagnostics not official benchmark',
        all_public_replay_exact=all(r['public_replay_exact'] for r in rows),episodes=len(rows),groups=grouped,rows=rows))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['static','deploy']);args=parser.parse_args()
    torch.set_num_threads(1);prepare()
    if args.command=='static':
        defaults=run.read(SOURCE/'experiment.json')
        from onpolicy.config import get_config
        original=get_config().parse_args([]);current=settings()
        save(OUT/'parameter_differences.json',{k:dict(upstream_default=getattr(original,k),used=getattr(current,k)) for k in vars(current) if hasattr(original,k) and getattr(original,k)!=getattr(current,k)})
        wall_counterexample();replay_apf();permutation_probe()
    else:deploy()
    print('complete',args.command,flush=True)


if __name__=='__main__':main()
