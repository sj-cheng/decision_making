"""E45-J: canonical public beliefs, clearance inputs and RouteAPF adapter.

The physical legal-action mapping stays identical to I. Collision-intent
feedback and progressive curriculum are TRAINING changes, explicitly logged.
"""
import numpy as np
import torch
from numba import njit
from learning.e45i_official_mappo import (PublicState,settings,EpisodeBuffer,HIDDEN,
    R_MAPPOPolicy,R_MAPPO,gae)
from learning.e45j_adapter_repairs import canonical_features,route_apf_action
from learning.e45g_ppo import safe_action
from learning.e45b_joint_network import KEYS
from solvers.frozen_d_search_interface import safe
from solvers.e45b_joint_nte import coverage_potential

OBS_DIM=844
CENT_DIM=858
DIRECTIONS=np.array([[np.cos(t),np.sin(t)] for t in np.linspace(0,2*np.pi,16,endpoint=False)])
DIRECTIONS/=np.max(np.abs(DIRECTIONS),axis=1,keepdims=True)


def training_rewards(raw,phi,blocked):
    raw=np.asarray(raw);phi=np.asarray(phi);blocked=np.asarray(blocked)
    return (raw+np.r_[phi[1:],0.]-phi-.05*blocked).astype(np.float32)


@njit(cache=True)
def clearance(positions,active,speed,boxes):
    result=np.zeros((2,16))
    for i in range(2):
        if not active[i]:continue
        for k in range(16):
            displacement=DIRECTIONS[k]*speed;count=np.zeros(2,np.int64)
            if safe(positions[i],displacement,speed,boxes,count):
                result[i,k]=1.;continue
            low,high=0.,1.
            for _ in range(14):
                middle=(low+high)/2
                if safe(positions[i],displacement*middle,speed,boxes,count):low=middle
                else:high=middle
            result[i,k]=max(0.,low-1e-5)
    return result


def inputs(features,public,obs):
    x=canonical_features(features)
    flat=np.concatenate([np.asarray(x[k],np.float32).ravel() for k in KEYS])
    moments=[];offsets=[]
    for j in range(2):
        w=x['weights'][j];points=x['particles'][j,:,:2]
        if w.sum()>0:
            mean=np.average(points,axis=0,weights=w)
            variance=np.average((points-mean)**2,axis=0,weights=w)
            covariance=np.average((points[:,0]-mean[0])*(points[:,1]-mean[1]),weights=w)
            moments.extend(np.r_[mean,np.sqrt(np.maximum(variance,0)),covariance])
            offsets.append(mean[None,:]-x['agents'][:,:2])
        else:
            moments.extend(np.zeros(5));offsets.append(np.zeros((2,2)))
    rays=clearance(np.asarray(obs['positions'],float),np.asarray(obs['active_p'],bool),public.speed,public.boxes)
    shared=np.r_[flat,moments]
    result=np.stack([np.r_[shared,offsets[0][i],offsets[1][i],rays[i],np.eye(2)[i]] for i in range(2)]).astype(np.float32)
    critic_public=np.r_[shared,rays.ravel()].astype(np.float32)
    assert result.shape==(2,OBS_DIM) and critic_public.shape==(CENT_DIM-4,)
    return result,critic_public


def policy(kind,device='cpu'):
    from gym.spaces import Box
    args=settings();dimensions=4 if kind=='MAPPO_APF' else 2
    model=R_MAPPOPolicy(args,Box(-np.inf,np.inf,(OBS_DIM,),dtype=np.float32),
        Box(-np.inf,np.inf,(CENT_DIM,),dtype=np.float32),Box(-np.inf,np.inf,(dimensions,),dtype=np.float32),torch.device(device))
    if kind=='MAPPO_APF':
        with torch.no_grad():model.actor.act.action_out.logstd._bias.fill_(np.log(.35))
    return model,R_MAPPO(args,model,torch.device(device))


class Controller(PublicState):
    def __init__(self,boxes,speed,seed,actor,kind,deterministic=True,coefficients=None):
        super().__init__(boxes,speed,seed)
        self.actor=actor;self.actor.eval();self.kind=kind;self.deterministic=deterministic
        self.coefficients=coefficients;self.hidden=np.zeros((2,1,HIDDEN),np.float32);self.samples=[]

    @torch.no_grad()
    def observe(self,obs):
        PublicState.observe(self,obs);x,_=inputs(self.features(obs),self.public,obs)
        _,_,h=self.actor(x,self.hidden,np.ones((2,1),np.float32),deterministic=True)
        self.hidden=h.numpy()

    @torch.no_grad()
    def act(self,obs,fixed_simulations=None):
        PublicState.observe(self,obs);features=self.features(obs);x,critic_public=inputs(features,self.public,obs)
        before=self.hidden.copy();raw,logp,h=self.actor(x,before,np.ones((2,1),np.float32),deterministic=self.deterministic)
        self.hidden=h.numpy();raw=raw.numpy()
        if self.kind=='MAPPO_APF':
            action=route_apf_action(self,obs,raw,self.coefficients)
        else:
            action,changed=safe_action(np.asarray(obs['positions']),np.tanh(raw),obs['active_p'],self.public.speed,self.public.boxes)
            self.last_stats=dict(simulations=0,joint_action_search=False,assignment_used=False,roles_used=False,safety_shortened=changed)
        self.last_stats['team_belief']=self.belief.last_stats
        self.last_stats['input_adapter']='canonical particles + public moments + ego target offsets +16 public clearance rays'
        self.samples.append(dict(obs=x,critic_public=critic_public,actor_hidden=before,raw=raw,logp=logp.numpy(),
            active=np.asarray(obs['active_p'],np.float32)[:,None]))
        if not self.deterministic:
            distance_phi=coverage_potential(np.asarray(obs['positions']),np.asarray(obs['active_p']),np.asarray(obs['active_e']),
                features['particles'][:,:,:2]*20+features['public'][:2]*10,features['weights'],
                self.public.speed,self.public.vertices,self.public.dist,self.public.boxes)
            self.samples[-1]['phi']=float(.05*distance_phi+.25*(2-sum(obs['active_e'])))
            self.samples[-1]['blocked_fraction']=self.last_stats['safety_shortened']/max(1,sum(obs['active_p']))
        return action


@torch.no_grad()
def finish_episodes(episodes,model):
    count=len(episodes);hidden=torch.zeros(count*2,1,HIDDEN,device=model.device)
    for e in episodes:
        n=len(e['obs']);common=np.c_[e['critic_public'],e['privileged']]
        e['central']=np.repeat(common[:,None,:],2,axis=1).astype(np.float32)
        e['values']=np.zeros((n,2,1),np.float32);e['critic_hidden']=np.zeros((n,2,1,HIDDEN),np.float32)
    for step in range(max(len(e['obs']) for e in episodes)):
        x=np.zeros((count,2,CENT_DIM),np.float32);masks=np.zeros((count,2,1),np.float32)
        before=hidden.cpu().numpy().reshape(count,2,1,HIDDEN)
        for i,e in enumerate(episodes):
            if step<len(e['obs']):x[i]=e['central'][step];masks[i]=1;e['critic_hidden'][step]=before[i]
        value,hidden=model.critic(x.reshape(-1,CENT_DIM),hidden,masks.reshape(-1,1));value=value.cpu().numpy().reshape(count,2,1)
        for i,e in enumerate(episodes):
            if step<len(e['obs']):e['values'][step]=value[i]
    for e in episodes:
        _,returns=gae(e['reward'][:,None],e['values'][...,0],gamma=1.,lam=.95)
        e['returns']=returns[...,None]
    return EpisodeBuffer(episodes)
