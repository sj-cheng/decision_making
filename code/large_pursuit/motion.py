"""Fixed D reference and its corresponding continuous physical-motion approximation.

The reference retains anchored .5m lawful edges and normalized Gaussian weights.
Filter/tree use the same Gaussian law conditioned on continuous legal actions,
not a mixture: same nine intents, persistence .9 and sigma .35. Both share the
map routing graph. This representation change supports the plan's continuous
learned evaders; it does not add an opponent or a formal method.
"""
from functools import lru_cache
import numpy as np
from numba import njit
from .core import clear,legal

MODES=np.array([[0,0],[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]],float)
SIGMA=.35
PERSISTENCE=.9

@njit(cache=True)
def feasible_options(at,xs,ys,boxes,size):
    out=np.empty((len(xs)*len(ys)+1,2));count=0
    for x in xs:
        for y in ys:
            q=np.array([x,y])
            if np.max(np.abs(q-at))<=1.+1e-9 and legal(q,boxes,size) and clear(at,q,boxes):
                out[count]=q;count+=1
    # A continuous diagnostic position may be off lattice. Staying remains lawful.
    found=False
    for k in range(count):
        if np.max(np.abs(out[k]-at))<1e-9:found=True
    if not found and legal(at,boxes,size):out[count]=at;count+=1
    return out[:count]

@njit(cache=True)
def lawful_batch(positions,actions,boxes,size):
    keep=np.empty(len(positions),np.bool_)
    for i in range(len(positions)):
        q=positions[i]+actions[i]
        keep[i]=np.max(np.abs(actions[i]))<=1. and legal(q,boxes,size) and clear(positions[i],q,boxes)
    return keep

class MotionModel:
    def __init__(self,public,continuous=False):
        self.public=public;self.continuous=bool(continuous)
        self.options=lru_cache(maxsize=8192)(self._options)
        self.distribution=lru_cache(maxsize=16384)(self._distribution)
        self._goal_cache=lru_cache(maxsize=2048)(self._make_goals)

    def goals(self,anchor):return self._goal_cache(tuple(anchor))

    def _make_goals(self,anchor):
        anchor=np.asarray(anchor)
        h=self.public.cfg.size/2
        goals=np.clip(np.asarray(anchor)[None]+self.public.cfg.size*MODES,-h+.05,h-.05)
        goals[0]=anchor
        # Edge goals are guaranteed free by generator's 2m exterior corridor.
        if any(not legal(g,self.public.boxes,self.public.cfg.size) for g in goals):
            raise ValueError('An intent goal is not geometrically lawful')
        return goals

    def _options(self,at_key,anchor_key):
        at=np.array(at_key);anchor=np.array(anchor_key);h=self.public.cfg.size/2;axes=[]
        for k in range(2):
            lo=int(np.ceil((at[k]-1.-anchor[k])/.5-1e-9));hi=int(np.floor((at[k]+1.-anchor[k])/.5+1e-9))
            axis=anchor[k]+.5*np.arange(lo,hi+1)
            axis=np.r_[axis,[-h,h]]
            axis=np.unique(axis[(axis>=max(-h,at[k]-1)-1e-9)&(axis<=min(h,at[k]+1)+1e-9)])
            axes.append(axis)
        return feasible_options(at,axes[0],axes[1],self.public.boxes,self.public.cfg.size)

    def _distribution(self,at_key,anchor_key,mode):
        at=np.array(at_key);anchor=np.array(anchor_key)
        support=self.options(at_key,anchor_key)
        if not len(support):raise RuntimeError('Motion kernel has no legal edge')
        mean=np.zeros(2) if mode==0 else self.public.action(at,self.goals(anchor)[mode])
        probabilities=np.exp(-.5*np.sum((support-at-mean)**2,axis=1)/SIGMA**2)
        probabilities/=probabilities.sum()
        return support,probabilities

    def advance(self,positions,modes,anchors,rng,active=None):
        shape=positions.shape;flat=np.asarray(positions).reshape(-1,2)
        old_modes=np.asarray(modes).reshape(-1)
        active=np.ones(len(flat),bool) if active is None else np.broadcast_to(active,shape[:-1]).reshape(-1)
        anchors=np.broadcast_to(np.asarray(anchors),shape).reshape(-1,2)
        change=rng.random(len(flat))>=PERSISTENCE
        draws=rng.integers(9,size=len(flat));new_modes=np.where(change,draws,old_modes)
        # Fixed consumption for the discrete reference enables paired seeds even
        # when different pursuer methods capture different target subsets.
        uniforms=rng.random(len(flat))
        if self.continuous and len(flat)>4096:
            # Same rejection law, batched only for large public-history recovery.
            # Repeated anchors/modes need one route query rather than N queries.
            result=flat.copy();indices=np.flatnonzero(active)
            keys=np.c_[flat[indices],anchors[indices],new_modes[indices]]
            unique,inverse=np.unique(keys,axis=0,return_inverse=True)
            means=np.array([np.zeros(2) if int(key[4])==0 else
                self.public.action(key[:2],self.goals(key[2:4])[int(key[4])]) for key in unique])
            means=means[inverse];pending=np.arange(len(indices))
            for attempt in range(10000):
                if not len(pending):break
                action=means[pending]+rng.normal(0,SIGMA,(len(pending),2))
                selected=indices[pending]
                keep=lawful_batch(flat[selected],action,self.public.boxes,self.public.cfg.size)
                result[selected[keep]]=flat[selected[keep]]+action[keep]
                pending=pending[~keep]
            if len(pending):raise RuntimeError('Continuous D rejection sampler has no resolved legal support')
            return result.reshape(shape),new_modes.reshape(np.shape(modes))
        result=np.empty_like(flat)
        for i,(q,anchor,mode,u) in enumerate(zip(flat,anchors,new_modes,uniforms)):
            if not active[i]:result[i]=q;continue
            if self.continuous:
                mean=np.zeros(2) if mode==0 else self.public.action(q,self.goals(anchor)[mode])
                for attempt in range(10000):
                    action=mean+rng.normal(0,SIGMA,2)
                    if np.max(np.abs(action))>1.:continue
                    candidate=q+action
                    if legal(candidate,self.public.boxes,self.public.cfg.size) and clear(q,candidate,self.public.boxes):break
                else:raise RuntimeError('Continuous D rejection sampler has no resolved legal support')
                result[i]=candidate
            else:
                support,probabilities=self.distribution(tuple(q),tuple(anchor),int(mode))
                index=min(int(np.searchsorted(np.cumsum(probabilities),u)),len(support)-1)
                result[i]=support[index]
        return result.reshape(shape),new_modes.reshape(np.shape(modes))


def shared_model(public):
    if not hasattr(public,'_motion_model'):public._motion_model=MotionModel(public)
    return public._motion_model


def planning_model(public):
    if not hasattr(public,'_planning_motion_model'):public._planning_motion_model=MotionModel(public,continuous=True)
    return public._planning_motion_model

class MotionKernel:
    def __init__(self,public,rng,anchors=None,modes=None):
        from .isolation import isolated_routes
        self.public=isolated_routes(public);self.rng=rng;self.anchors=None if anchors is None else np.array(anchors,copy=True)
        self.modes=rng.integers(9,size=public.cfg.n) if modes is None else np.array(modes,copy=True)

    def actions(self,state):
        if self.anchors is None:self.anchors=state.e.copy()
        nxt,self.modes=shared_model(self.public).advance(state.e,self.modes,self.anchors,self.rng,state.alive)
        actions=nxt-state.e;actions[~state.alive]=0.
        return actions
