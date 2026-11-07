"""Outcome-blind hidden handoff histories on the unchanged E48 geometry."""
import copy
import numpy as np
from corner_estimator_scenarios import original_model
from solvers.e45_team_adapter import public_observation
from solvers.frozen_d_search_interface import PublicMap,move,route,sensor_visible,safe
from solvers.local_pursuit import clear
from run_e48_dense_obstacles import geometry

BASE=14920261107


def construct(index,world):
    boxes=geometry(index);speed=1.3;model=original_model(boxes,speed);public=PublicMap(boxes,speed)
    rng=np.random.default_rng(np.random.SeedSequence([BASE,index,int(world=='BA')]))
    count=np.zeros(2,np.int64);side=1 if world=='AB' else -1
    for attempt in range(3000):
        ee=np.c_[side*rng.uniform(2.,6.,2),[-rng.uniform(2.,5.7),rng.uniform(2.,5.7)]]
        angles=rng.uniform(-np.pi,np.pi,2);radius=rng.uniform(2.8,4.3,2)
        pp=ee+radius[:,None]*np.c_[np.cos(angles),np.sin(angles)]
        state=np.zeros((model.state_dim,1));state[:8,0]=np.r_[ee.ravel(),pp.ravel()];state[model.active_idxs,0]=1
        if not model.is_valid(state) or model.get_capture_pairs(state):continue
        if np.linalg.norm(pp[0]-pp[1])<2.5:continue
        if np.linalg.norm(ee[:,None]-pp[None],axis=2).min()<2.4:continue
        if not all(sensor_visible(pp[j],ee[j],boxes) for j in range(2)):continue
        goals=np.array([[-side*8.,-rng.uniform(2.5,6.)],[-side*8.,rng.uniform(2.5,6.)]])
        if not all(safe(g,np.zeros(2),speed,boxes,count) for g in goals):continue
        model._update_observation_memory(state,state,0.)
        states=[state[:,0].tolist()];history=[];obstacle_loss=False;valid=True
        for step in range(13):
            pos=state[4:8,0].reshape(2,2);target=state[:4,0].reshape(2,2)
            obs=public_observation(step,pos,target,np.ones(2,bool),np.ones(2,bool),boxes);history.append(obs)
            for i in range(2):
                for j in range(2):
                    if np.linalg.norm(pos[i]-target[j])<=6 and not clear(pos[i],target[j],boxes,count):obstacle_loss=True
            if step==12:break
            actions=np.array([move(pos[i],goals[i],speed,public.vertices,public.dist,boxes,count) for i in range(2)])
            nxt=model.step(state,np.r_[np.zeros(4),actions.ravel()][:,None],1.)
            if not model.is_valid(nxt) or model.get_capture_pairs(nxt):valid=False;break
            states.append(nxt[:,0].tolist());state=nxt
        if not valid or not obstacle_loss:continue
        if any(q is not None for obs in history[-4:] for reports in obs['reports'] for q in reports):continue
        pp=state[4:8,0].reshape(2,2);ee=state[:4,0].reshape(2,2)
        distances=np.linalg.norm(ee[:,None]-pp[None],axis=2)
        if distances.min()<8.:continue
        routed=np.array([[route(a,b,speed,public.vertices,public.dist,boxes,count)[0] for a in pp] for b in ee])
        if np.any(routed>=1000000) or routed.min()<7 or routed.max()>50:continue
        if np.max(np.abs(state[:8,0]))>8.6:continue
        identifier=f'e49_hard_dense_{index:02d}_{world}'
        selected=dict(id=world,initial=state[:,0].tolist(),scripted_states=states)
        return dict(id=identifier,family='dense15_hidden_handoff',map=boxes.tolist(),speed=speed,
            history=history,worlds=[selected],suite='hard_exploratory',split='diagnostic',
            scope='Legal visible anchor then scripted observer withdrawal; targets stationary in history, adaptive after handoff',
            construction=dict(attempts=attempt+1,handoff_step=12,hidden_steps_at_least=4,
                nearest_distance=float(distances.min()),min_route_steps=int(routed.min()),
                obstacle_occlusion_observed=True,outcome_filter=False))
    raise RuntimeError(f'Cannot construct registered hard handoff {index}/{world}')


def audit(case):
    boxes=np.array(case['map']);model=original_model(boxes,case['speed']);states=case['worlds'][0]['scripted_states']
    for step,item in enumerate(states):
        state=np.array(item)[:,None];assert model.is_valid(state) and not model.get_capture_pairs(state)
        if step:
            prev=np.array(states[step-1])[:,None];action=state[:8]-prev[:8]
            for robot in range(4):assert safe(prev[2*robot:2*robot+2,0],action[2*robot:2*robot+2,0],1. if robot<2 else case['speed'],boxes,np.zeros(2,np.int64))
            np.testing.assert_allclose(model.step(prev,action,1.),state,rtol=0,atol=1e-10)
        obs=public_observation(step,state[4:8,0].reshape(2,2),state[:4,0].reshape(2,2),np.ones(2,bool),np.ones(2,bool),boxes)
        assert obs==case['history'][step]
    assert all(any(q is not None for q in reports) for reports in case['history'][0]['reports'])
    assert all(q is None for obs in case['history'][-4:] for reports in obs['reports'] for q in reports)
    return True
