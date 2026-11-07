"""Exact algebraic reuse of fixed visibility-graph tails in G scoring.

No new policy, approximate position key, candidate, motion or value model.
The target points here are public root samples/map vertices, never hidden truth.
"""
import time
import numpy as np
from numba import njit
from solvers.frozen_d_search_interface import COMPASS,sensor_visible
from l45_corner_shared_20261025.local_pursuit import clear,edge,safe
import solvers.rollout_replan as slow


@njit(cache=True)
def prepare_geometry(samples,weights,labels,vertices,dist,boxes,speed):
    targets=np.empty((64+len(vertices),2));targets[:64]=samples;targets[64:]=vertices
    goals=np.empty(64,np.int64);number=0
    for r in range(int(labels.max())+1):
        mass=0.;center=np.zeros(2)
        for i in range(64):
            if labels[i]==r:
                center+=weights[i]*samples[i];mass+=weights[i]
        if mass<=0:continue
        center/=mass;winner=-1;distance=1e100
        for i in range(64):
            if labels[i]==r and np.sum((samples[i]-center)**2)<distance:
                winner,distance=i,np.sum((samples[i]-center)**2)
        goals[number]=winner;number+=1
    for j in range(len(vertices)):goals[number]=64+j;number+=1
    tails=np.full((len(targets),len(vertices)),1000000,np.int64)
    count=np.zeros(2,np.int64)
    for t in range(len(targets)):
        for j in range(len(vertices)):
            if clear(vertices[j],targets[t],boxes,count):
                cost=edge(vertices[j],targets[t],speed)
                for i in range(len(vertices)):tails[t,i]=min(tails[t,i],dist[i,j]+cost)
    return targets,goals[:number].copy(),tails


@njit(cache=True)
def heads(at,vertices,boxes,speed,count):
    result=np.full(len(vertices),1000000,np.int64)
    for j in range(len(vertices)):
        if clear(at,vertices[j],boxes,count):result[j]=edge(at,vertices[j],speed)
    return result


@njit(cache=True)
def geometric_candidates_exact(at,samples,weights,labels,speed,vertices,dist,boxes,motion_boxes,targets,goals,tails):
    count=np.zeros(2,np.int64);head=heads(at,vertices,boxes,speed,count)
    actions=np.zeros((64,2));actions[:9]=COMPASS*speed;n=9
    for g in goals:
        target=targets[g];best=1000000;first=-1
        if clear(at,target,boxes,count):best=edge(at,target,speed)
        for j in range(len(vertices)):
            cost=head[j]+tails[g,j]
            if cost<best:best,first=cost,j
        waypoint=vertices[first] if first>=0 else target
        delta=waypoint-at
        action=delta/max(1.,np.max(np.abs(delta))/speed)
        if not safe(at,action,speed,boxes,count):action=np.zeros(2)
        if n>=64:raise RuntimeError('Candidate capacity exceeded')
        actions[n]=action;n+=1
    unique=np.zeros_like(actions);used=0
    for j in range(n):
        if not safe(at,actions[j],speed,motion_boxes,count):continue
        duplicate=False
        for k in range(used):
            if np.max(np.abs(actions[j]-unique[k]))<1e-10:duplicate=True
        if not duplicate:unique[used]=actions[j];used+=1
    best=0;best_score=-1e100
    for j in range(used):
        nxt=at+unique[j];score=0.;next_head=heads(nxt,vertices,boxes,speed,count)
        for i in range(64):
            steps=1000000
            if clear(nxt,samples[i],boxes,count):steps=edge(nxt,samples[i],speed)
            for k in range(len(vertices)):
                candidate=next_head[k]+tails[i,k]
                if candidate<steps:steps=candidate
            score+=weights[i]*(-steps+int(sensor_visible(nxt,samples[i],boxes)))
        if score>best_score:best,best_score=j,score
    return unique[:used].copy(),unique[best].copy()


_make_context=slow.make_context


def make_context(controller,started,fixed):
    context=_make_context(controller,started,fixed)
    stamp=time.perf_counter()
    if context['labels'] is not None:
        context['geometry_prepared']=prepare_geometry(context['samples'],context['weights'],context['labels'],
            context['vertices'],context['dist'],context['boxes'],context['speed'])
    context['geometry_prepare_ms']=1000*(time.perf_counter()-stamp)
    return context


def rollout_action(at):
    ctx=slow._active;key=(float(at[0]),float(at[1]))
    if ctx['cache_enabled'] and key in ctx['cache']:
        ctx['hits']+=1
        return ctx['cache'][key].copy(),slow.step_allowed()
    if time.perf_counter()+(max(5.,2*ctx['max_query_ms'])+2.)/1000.>=ctx['deadline']:
        return np.zeros(2),False
    stamp=time.perf_counter()
    _,action=geometric_candidates_exact(at,ctx['samples'],ctx['weights'],ctx['labels'],ctx['speed'],
        ctx['vertices'],ctx['dist'],ctx['boxes'],ctx['motion_boxes'],*ctx['geometry_prepared'])
    elapsed=1000*(time.perf_counter()-stamp);ctx['misses']+=1;ctx['query_ms']+=elapsed
    ctx['max_query_ms']=max(ctx['max_query_ms'],elapsed)
    if ctx['cache_enabled']:ctx['cache'][key]=action.copy()
    return action,slow.step_allowed()


def install():
    """Process-local execution binding; original source files remain immutable."""
    slow.make_context=make_context;slow.rollout_action=rollout_action
