"""Public-belief HASAC controller, full-episode replay, bounded EP storage."""
import numpy as np
import torch
from learning.e45i_official_mappo import PublicState
from learning.e45j_repaired_mappo import inputs,training_rewards
from learning.e45g_ppo import safe_action
from solvers.e45b_joint_nte import coverage_potential

OBS_DIM=845
CENT_DIM=858


class Controller(PublicState):
    def __init__(self,boxes,speed,seed,actors,stochastic=True,uniform=False,policy_seed=0):
        super().__init__(boxes,speed,seed)
        self.actors=actors;self.stochastic=stochastic;self.uniform=uniform
        self.rng=np.random.default_rng(policy_seed);self.samples=[]
        torch.manual_seed(policy_seed)

    @torch.no_grad()
    def act(self,obs,fixed_simulations=None):
        PublicState.observe(self,obs);features=self.features(obs)
        public,critic=inputs(features,self.public,obs)
        active=np.asarray(obs['active_p'],np.float32)[:,None]
        public=np.c_[public,active].astype(np.float32)
        if self.uniform:action=self.rng.uniform(-1,1,(2,2)).astype(np.float32)*active
        else:
            action=np.stack([a.get_actions(public[i:i+1],stochastic=self.stochastic)[0].cpu().numpy()
                             for i,a in enumerate(self.actors)])
        physical,shortened=safe_action(np.asarray(obs['positions']),action,obs['active_p'],self.public.speed,self.public.boxes)
        potential=coverage_potential(np.asarray(obs['positions']),np.asarray(obs['active_p']),np.asarray(obs['active_e']),
            features['particles'][:,:,:2]*20+features['public'][:2]*10,features['weights'],
            self.public.speed,self.public.vertices,self.public.dist,self.public.boxes)
        self.samples.append(dict(obs=public,central_public=critic,action=action,active=active,
                                 phi=float(.05*potential+.25*(2-sum(obs['active_e']))),
                                 blocked=float(shortened/max(1,sum(obs['active_p'])))))
        self.last_stats=dict(simulations=0,joint_action_search=False,assignment_used=False,roles_used=False,
                             safety_shortened=shortened,team_belief=self.belief.last_stats,
                             input_adapter='J canonical public inputs plus ego lifecycle flag; author HASAC feedforward',
                             normalized_intent=action.tolist())
        return physical


def episode_arrays(samples,trace):
    n=len(trace)
    if n!=len(samples) or not n:raise ValueError('Missing full episode samples')
    obs=np.stack([s['obs'] for s in samples]).astype(np.float32)
    state=np.stack([np.r_[s['central_public'],np.asarray(t['state'][:4])/10]
                    for s,t in zip(samples,trace)]).astype(np.float32)
    action=np.stack([s['action'] for s in samples]).astype(np.float32)
    active=np.stack([s['active'] for s in samples]).astype(np.float32)
    # Every collected episode ends at double capture or the finite horizon.
    # Retirement does not set team done. Terminal successor is an absorbing
    # public zero state, never bootstrapped by the author critic.
    done=np.zeros((n,1),np.float32);done[-1]=1
    reward=training_rewards([t['reward'] for t in trace],[s['phi'] for s in samples],[s['blocked'] for s in samples])[:,None]
    result=dict(obs=obs,central=state,action=action,active=active,reward=reward,done=done,
                next_obs=np.concatenate([obs[1:],np.zeros_like(obs[:1])]),
                next_central=np.concatenate([state[1:],np.zeros_like(state[:1])]))
    if not np.array_equal(obs[:,:,-1:],active):raise ValueError('Public active flags inconsistent')
    if np.any(np.abs(action[active[...,0]==0])>1e-7):raise ValueError('Retired desired actions nonzero')
    return result


class Replay:
    def __init__(self,capacity=50000,seed=17):
        self.capacity=capacity;self.size=0;self.position=0;self.rng=np.random.default_rng(seed)
        self.arrays={};self.resamples=0

    def add(self,episode):
        n=len(episode['obs'])
        if n>self.capacity:raise ValueError('Episode exceeds replay capacity')
        if not self.arrays:
            self.arrays={k:np.empty((self.capacity,*v.shape[1:]),np.float32) for k,v in episode.items()}
        idx=(np.arange(n)+self.position)%self.capacity
        for key,value in episode.items():self.arrays[key][idx]=value
        self.position=(self.position+n)%self.capacity;self.size=min(self.capacity,self.size+n)

    def sample(self,n):
        if self.size<1:raise ValueError('Empty replay')
        for attempt in range(100):
            ids=self.rng.integers(self.size,size=n)
            active=self.arrays['active'][ids]
            if np.all(active.sum(axis=(0,2))>0):break
            self.resamples+=1
        else:raise RuntimeError('Replay cannot supply a valid actor minibatch; no silent divide by zero')
        d={key:v[ids] for key,v in self.arrays.items()}
        return (d['central'],d['obs'].transpose(1,0,2),d['action'].transpose(1,0,2),None,
                d['reward'],d['done'],d['active'].transpose(1,0,2),d['done'].copy(),
                d['next_central'],d['next_obs'].transpose(1,0,2),None,np.ones((n,1),np.float32))

    def save(self,path):
        np.savez_compressed(path,**{k:v[:self.size] for k,v in self.arrays.items()},
                            size=self.size,position=self.position,capacity=self.capacity)

    def load(self,path,rng_state):
        with np.load(path) as d:
            assert int(d['capacity'])==self.capacity
            self.size=int(d['size']);self.position=int(d['position'])
            for k in ('obs','central','action','active','reward','done','next_obs','next_central'):
                self.arrays[k]=np.empty((self.capacity,*d[k].shape[1:]),np.float32)
                self.arrays[k][:self.size]=d[k]
        self.rng.bit_generator.state=rng_state
