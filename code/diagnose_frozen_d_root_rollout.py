"""Common-history 2x2 rollout diagnostic, not a replacement controller."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import json,fcntl,time
from pathlib import Path
import numpy as np
from numba import njit
from run_frozen_d_closed_loop import ROOT,OUT as SOURCE,OLD,save,sha,legal_action
from solvers.frozen_d_search_interface import (Controller,Observation,belief_view,
    root_sample,target_step,sensor_visible,geometric_candidates,COMPASS,route,safe)

OUT=ROOT/'current/l45_search_diagnosis_repeat_20261026'


def obs(item):
    return Observation(item['step'],np.array(item['position'],float),
        None if item['target'] is None else np.array(item['target'],float),item['remaining'])


@njit(cache=True)
def target_paths(points,links,lengths,probabilities,ids,weights,modes,draws,depth,seed):
    rng=np.array([seed],np.int64);paths=np.zeros((draws,depth+1,2))
    for k in range(draws):
        _,index,mode=root_sample(ids,weights,modes,rng)
        paths[k,0]=points[index]
        for t in range(depth):
            index,mode=target_step(index,mode,links,lengths,probabilities,rng)
            paths[k,t+1]=points[index]
    return paths


@njit(cache=True)
def evaluate(own,targets,boxes,horizon):
    result=np.zeros((len(targets),3))
    for k in range(len(targets)):
        seen_once=False
        for t in range(horizon):
            seen=sensor_visible(own[t+1],targets[k,t+1],boxes)
            capture=np.linalg.norm(own[t+1]-targets[k,t+1])<1.
            result[k,2]+=(.95**t)*(int(seen and not seen_once)+int(capture))
            seen_once=seen_once or seen
            result[k,0]=int(seen_once)
            if capture:
                result[k,1]=1.
                break
    return result


@njit(cache=True)
def expected_travel(at,samples,weights,speed,vertices,dist,boxes):
    result=0.;count=np.zeros(2,np.int64)
    for k in range(64):
        steps,_=route(at,samples[k],speed,vertices,dist,boxes,count)
        result+=weights[k]*steps
    return result


def own_path(controller,at,first,base,feedback,depth):
    p=controller.public;e=controller.estimator
    path=[at.copy()];rejections=0;stops=0
    for t in range(depth):
        if t==0:action=first.copy()
        elif feedback:
            _,action=geometric_candidates(path[-1],e.particles,e.particle_weights,e.regions[e.ids],
                p.speed,p.vertices,p.dist,p.boxes,controller.motion_boxes)
        else:action=base.copy()
        if not safe(path[-1],action,p.speed,controller.motion_boxes,np.zeros(2,np.int64)):
            rejections+=1;action=np.zeros(2)
        if np.max(np.abs(action))<1e-10:stops+=1
        assert legal_action(path[-1],action,p.speed,p.boxes)
        path.append(path[-1]+action)
    return np.array(path),rejections,stops


def requests():
    entries=json.loads((SOURCE/'bank/manifest.json').read_text())['entries'][::2]
    prior=json.loads((SOURCE/'confirmation/results.json').read_text())
    selected=[]
    for entry in entries:
        row=next(r for r in prior if r['history_id']==entry['id'] and r['world']=='A' and r['method']=='G')
        episode=json.loads(Path(row['episode']).read_text())
        hidden=[i for i,r in enumerate(episode['trace']) if r['observation']['target'] is None]
        for label,index in (('handoff',hidden[0]),('hidden_midpoint',hidden[len(hidden)//2])):
            selected.append(dict(entry=entry,episode=row['episode'],episode_sha256=row['episode_sha256'],
                root_index=index,label=label,seed=episode['task']['seed']))
    return selected


def main():
    protocol=json.loads((OUT/'experiment.json').read_text())
    assert all(sha(p)==h for p,h in protocol['sources'].items())
    if (OUT/'root_diagnostics.json').exists():raise RuntimeError('Completed diagnostics cannot be overwritten')
    chosen=requests();rows=[];started=time.perf_counter()
    save(OUT/'root_diagnostic_requests.json',chosen)
    for n,task in enumerate(chosen):
        packet=json.loads((SOURCE/'bank'/task['entry']['public']).read_text())
        assert sha(task['episode'])==task['episode_sha256']
        episode=json.loads(Path(task['episode']).read_text());index=task['root_index']
        history=packet['history'][:-1]+[r['observation'] for r in episode['trace'][:index+1]]
        models={};actions={};signatures=[]
        for method in ('G','T','T+Geo'):
            c=Controller(packet['map'],packet['motion']['pursuer_axis_limit'],method,task['seed'],budget_ms=200.)
            for item in history[:-1]:c.estimator.update(obs(item))
            actions[method]=c.act(obs(history[-1]),fixed_simulations=1000)
            models[method]=c
            e=c.estimator
            signatures.append(e.particles.tobytes()+e.particle_weights.tobytes()+e.rng.tobytes())
        assert signatures[0]==signatures[1]==signatures[2]
        c=models['G'];e=c.estimator;p=c.public;at=np.array(history[-1]['position'])
        geo,_=geometric_candidates(at,e.particles,e.particle_weights,e.regions[e.ids],p.speed,p.vertices,p.dist,p.boxes,c.motion_boxes)
        in_compass=bool(np.any(np.max(np.abs(COMPASS*p.speed-actions['G']),axis=1)<1e-10))
        in_geo=bool(np.any(np.max(np.abs(geo-actions['G']),axis=1)<1e-10));assert in_geo
        horizon=min(24,history[-1]['remaining'])
        points,links,lengths,probabilities,*_=e.graph
        targets=target_paths(points,links,lengths,probabilities,*belief_view(e),1024,horizon,(82020261026+n)%4294967296)
        baseline=expected_travel(at,e.particles,e.particle_weights,p.speed,p.vertices,p.dist,p.boxes)
        cases=[];raw={}
        for continuation,feedback in (('repeated_root_G_displacement',False),('replan_G_using_fixed_root_belief',True)):
            for method,action in actions.items():
                path,rejections,stops=own_path(c,at,action,actions['G'],feedback,horizon)
                for depth in (8,24):
                    h=min(depth,horizon)
                    values=evaluate(path,targets,p.boxes,h)
                    travel=expected_travel(path[h],e.particles,e.particle_weights,p.speed,p.vertices,p.dist,p.boxes)
                    cases.append(dict(continuation=continuation,horizon=h,method=method,
                        action=action.tolist(),model_rediscovery=float(values[:,0].mean()),
                        model_capture=float(values[:,1].mean()),model_return=float(values[:,2].mean()),
                        static_geometric_progress=float(baseline-travel),
                        illegal_displacement_fallbacks=rejections,stop_steps=stops,own_path=path.tolist()))
                    raw[(continuation,h,method)]=values[:,2]
        differences=[]
        for continuation in ('repeated_root_G_displacement','replan_G_using_fixed_root_belief'):
            for h in (8,24):
                for a,b in (('T','G'),('T+Geo','G'),('T+Geo','T')):
                    delta=raw[(continuation,min(h,horizon),a)]-raw[(continuation,min(h,horizon),b)]
                    differences.append(dict(continuation=continuation,horizon=min(h,horizon),comparison=a+'-'+b,
                        mean=float(delta.mean()),paired_mc_se=float(delta.std(ddof=1)/np.sqrt(len(delta)))))
        rows.append(dict(group=task['entry']['id'],family=task['entry']['family'],root_label=task['label'],
            step=history[-1]['step'],G_in_T_candidates=in_compass,G_in_TGeo_candidates=in_geo,
            root_actions={k:v.tolist() for k,v in actions.items()},common_public_estimator_identical=True,
            cases=cases,paired_model_differences=differences))
        save(OUT/'root_diagnostics'/(f'{n:02d}.json'),rows[-1])
        print(json.dumps(dict(stage='root_diagnosis',completed=n+1,total=24)),flush=True)
    save(OUT/'root_diagnostics.json',dict(roots=24,public_groups=12,draws_per_root=1024,rows=rows,
        elapsed_seconds=time.perf_counter()-started,
        scope='same known roots, hypothetical model returns; no new real closed-loop controller or capture-benefit claim',
        no_hidden_truth_used_for_candidates=True,estimator_unchanged=True))


if __name__=='__main__':
    with (ROOT/'current/.l45_local_pursuit_serial.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
