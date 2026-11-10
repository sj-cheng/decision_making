"""Full-state team evader MCTS, variable team size, progressive widening."""
import numpy as np
from numba import njit
from .core import move,captures,legal,clear
from .semantics import STEP_COST

@njit(cache=True)
def random_uniform(rng):
    rng[0]=(1664525*rng[0]+1013904223)%4294967296
    return (rng[0]+.5)/4294967296.

@njit(cache=True)
def joint_sample(positions,active,boxes,size,rng,rollout):
    """Original uninformed action sampler; rollout adds only lawful rejection."""
    action=np.zeros_like(positions)
    for i in range(len(positions)):
        if not active[i]:continue
        for attempt in range(20 if rollout else 1):
            candidate=np.array([2*random_uniform(rng)-1,2*random_uniform(rng)-1])
            if not rollout or (legal(positions[i]+candidate,boxes,size) and clear(positions[i],positions[i]+candidate,boxes)):
                action[i]=candidate;break
    return action

@njit(cache=True)
def search(p0,e0,alive0,boxes,size,capture,horizon,simulations,depth,rng):
    """Alternating team max/min PUCT with simultaneous physical transitions.

    Even plies select an evader commitment; odd plies select the pursuer reply.
    Both teams expand full joint actions, C_pw=2, alpha_pw=.5, C_exp=1,
    alpha_exp=.25. One completed simulation includes its entire default rollout.
    """
    n=len(e0);capacity=simulations+1
    first=np.full(capacity,-1,np.int64);sibling=np.full(capacity,-1,np.int64)
    visits=np.zeros(capacity,np.int64);totals=np.zeros(capacity);children=np.zeros(capacity,np.int64)
    actions=np.zeros((capacity,n,2));nodes=1;path=np.empty(depth+1,np.int64)
    limit_depth=min(depth,2*horizon);expansions=np.zeros(2,np.int64)
    for simulation in range(simulations):
        p=p0.copy();e=e0.copy();alive=alive0.copy();node=0;length=1;path[0]=0
        pending=np.zeros_like(e);value=0.
        for ply in range(limit_depth):
            maximum=max(1,int(np.ceil(2.*visits[node]**.5)))
            if children[node]<maximum and nodes<capacity:
                child=nodes;nodes+=1;expansions[ply%2]+=1;sibling[child]=first[node];first[node]=child;children[node]+=1
                proposal=joint_sample(e if ply%2==0 else p,alive if ply%2==0 else np.ones(n,np.bool_),boxes,size,rng,False)
                actions[child]=proposal;expanded=True
            else:
                child=first[node];winner=child;best=-1e100
                sign=1. if ply%2==0 else -1.
                while child>=0:
                    score=sign*totals[child]/max(1,visits[child])+np.sqrt(max(1,visits[node])**.25/max(1,visits[child]))
                    if score>best:best=score;winner=child
                    child=sibling[child]
                child=winner;expanded=False
            if ply%2==0:pending=actions[child]
            else:
                before=alive.sum()
                p=move(p,actions[child],np.ones(n,np.bool_),boxes,size);e=move(e,pending,alive,boxes,size)
                alive=alive & ~captures(p,e,alive,boxes,capture)
                value-=float(before-alive.sum())/n-STEP_COST
            node=child;path[length]=node;length+=1
            if not alive.any():break
            if expanded:
                for rest in range(ply+1,min(ply+1+depth,2*horizon)):
                    if rest%2==0:pending=joint_sample(e,alive,boxes,size,rng,True)
                    else:
                        before=alive.sum();pa=joint_sample(p,np.ones(n,np.bool_),boxes,size,rng,True)
                        p=move(p,pa,np.ones(n,np.bool_),boxes,size);e=move(e,pending,alive,boxes,size)
                        alive=alive & ~captures(p,e,alive,boxes,capture)
                        value-=float(before-alive.sum())/n-STEP_COST
                    if not alive.any():break
                break
        for k in range(length):visits[path[k]]+=1;totals[path[k]]+=value
    winner=first[0];child=winner
    while child>=0:
        if totals[child]/max(1,visits[child])>totals[winner]/max(1,visits[winner]):winner=child
        child=sibling[child]
    return actions[winner],nodes,expansions


def rootpuct_evader(public,state,rng,simulations=3000,depth=100,return_stats=False):
    if simulations<1 or depth<2:raise ValueError('RootPUCT requires positive simulations and at least one synchronous team step')
    seed=np.array([int(rng.integers(1,2**32))],np.int64)
    action,nodes,expansions=search(state.p,state.e,state.alive,public.boxes,public.cfg.size,public.cfg.capture,
                    public.cfg.horizon-state.t,simulations,depth,seed)
    if return_stats:return action,dict(simulations=simulations,depth_plies=depth,nodes=int(nodes),evader_expansions=int(expansions[0]),pursuer_expansions=int(expansions[1]),C_pw=2.,alpha_pw=.5,C_exp=1.,alpha_exp=.25)
    return action
