"""Audited adapter candidates, separate from every frozen E45-I artifact.

Canonical particle order requires fresh policy training. RouteAPF is a NEW
route-assisted controller, not a replacement for the reported plain APF.
"""
import numpy as np
from learning.e45i_official_mappo import (actor_inputs, PublicState, DEFAULT_APF, repulsion)
from learning.e45g_ppo import safe_action
from solvers.frozen_d_search_interface import route
from solvers.e45_team_adapter import optimal_assignment


def canonical_features(features):
    result={k:np.asarray(v).copy() for k,v in features.items()}
    for j in range(2):
        points=result['particles'][j];weights=result['weights'][j]
        order=np.lexsort((weights,points[:,1],points[:,0]))
        result['particles'][j]=points[order];result['weights'][j]=weights[order]
    return result


def canonical_actor_inputs(features):
    return actor_inputs(canonical_features(features))


def progress_waypoint(public,position,target):
    # route() admits the current vertex as a zero-cost first hop. Remove only
    # those zero-length candidates, retaining all pairwise shortest distances.
    keep=np.max(np.abs(public.vertices-position),axis=1)>1e-7
    vertices=public.vertices[keep];dist=public.dist[np.ix_(keep,keep)]
    distance,index=route(position,target,public.speed,vertices,dist,public.boxes,np.zeros(2,np.int64))
    if distance>=1_000_000:raise RuntimeError('No public route to target representative')
    return vertices[index].copy() if index>=0 else np.asarray(target).copy()


def progress_move(public,position,target):
    waypoint=progress_waypoint(public,position,target);delta=waypoint-position
    return delta/max(1.,np.max(np.abs(delta))/public.speed)


def representative(estimator,known):
    if known is not None:return np.asarray(known).copy()
    center=np.average(estimator.particles,axis=0,weights=estimator.particle_weights)
    # A weighted Euclidean medoid is an actual legal posterior support point;
    # the arithmetic mean can lie in the interior of a wall.
    scores=np.sum((estimator.particles-center)**2,axis=1)
    valid=np.asarray(estimator.particle_weights)>0
    scores=np.where(valid,scores,np.inf)
    order=np.lexsort((estimator.particles[:,1],estimator.particles[:,0],scores))
    return estimator.particles[int(order[0])].copy()


def route_apf_action(own,obs,raw=None,coefficients=None):
    positions=np.asarray(obs['positions'],float);public=own.public
    goals=np.zeros((2,2));cost=np.full((2,2),1e9)
    for j in np.flatnonzero(obs['active_e']):
        estimator=own.belief.estimators[j]
        goals[j]=representative(estimator,own.belief.known[j])
        for i in np.flatnonzero(obs['active_p']):
            cost[i,j]=np.average(np.linalg.norm(estimator.particles-positions[i],axis=1),weights=estimator.particle_weights)
    assignment=optimal_assignment(cost,obs['active_p'],obs['active_e'])
    gains=np.tile(DEFAULT_APF if coefficients is None else coefficients,(2,1))
    if raw is not None:gains*=np.exp(np.log(4.)*np.tanh(raw))
    normalized=np.zeros((2,2));fallbacks=0
    for i,j in assignment.items():
        waypoint=progress_waypoint(public,positions[i],goals[j]);delta=waypoint-positions[i]
        route_action=delta/max(1.,np.max(np.abs(delta))/public.speed)
        pull=delta/max(np.linalg.norm(delta),1e-12)
        away,tangent=repulsion(positions[i],public.boxes,waypoint)
        separation=np.zeros(2)
        for other in np.flatnonzero(obs['active_p']):
            if other==i:continue
            displacement=positions[i]-positions[other];distance=max(np.linalg.norm(displacement),.1)
            if distance<2:separation+=displacement/distance*min(4.,1/distance-.5)
        force=gains[i,0]*pull+gains[i,1]*away+gains[i,2]*separation+gains[i,3]*tangent
        amplitude=min(1.,np.max(np.abs(delta))/public.speed)
        action=public.speed*amplitude*force/max(1e-12,np.max(np.abs(force)))
        # An explicit geometric progress safeguard, disclosed as part of the
        # NEW hybrid. This is never presented as learned policy improvement.
        progress=np.max(np.abs(delta))-np.max(np.abs(delta-action))
        if progress<=1e-5 or np.dot(action,route_action)<=0:
            action=route_action;fallbacks+=1
        normalized[i]=action/public.speed
    action,changed=safe_action(positions,normalized,obs['active_p'],public.speed,public.boxes)
    own.last_stats=dict(simulations=0,assignment_used=True,route_progress_fallbacks=fallbacks,
        safety_shortened=changed,controller='RouteAPF candidate, not frozen APF',
        assignment={str(i):int(j) for i,j in assignment.items()})
    return action


class RouteAPFController(PublicState):
    def __init__(self,boxes,speed,seed,coefficients=None):
        super().__init__(boxes,speed,seed);self.coefficients=coefficients

    def act(self,obs,fixed_simulations=None):
        self.observe(obs)
        return route_apf_action(self,obs,coefficients=self.coefficients)
