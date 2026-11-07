"""Thin construction/update bridge to unchanged HARL HASAC author code.

The task environment adapter lives in e45m_task_adapter. Caller must supply
EP replay transitions, public actor observations and training-only critic
states. No new actor, critic, loss, or update rule is implemented here.
"""
from pathlib import Path
import sys
import numpy as np
import torch
import yaml

UPSTREAM=Path(__file__).resolve().parents[2]/'third_party/harl_source'
sys.path.insert(0,str(UPSTREAM))
from harl.algorithms.actors.hasac import HASAC
from harl.algorithms.critics.soft_twin_continuous_q_critic import SoftTwinContinuousQCritic
from harl.runners.off_policy_ha_runner import OffPolicyHARunner


class BatchSource:
    def __init__(self,batch):self.batch=batch
    def sample(self):return self.batch


class RetirementHASAC(HASAC):
    """Retired actuators have a deterministic zero action and zero entropy.

    The last public input is the ego active flag. The author Gaussian and all
    trainable parameters are unchanged; this implements task lifecycle rules.
    """
    def get_actions(self,obs,available_actions=None,stochastic=True):
        result=super().get_actions(obs,available_actions,stochastic)
        active=torch.as_tensor(obs,device=self.device,dtype=torch.float32)[...,-1:]
        return result*active

    def get_actions_with_logprobs(self,obs,available_actions=None,stochastic=True):
        action,logp=super().get_actions_with_logprobs(obs,available_actions,stochastic)
        active=torch.as_tensor(obs,device=self.device,dtype=torch.float32)[...,-1:]
        return action*active,logp*active


class AuthorHASAC:
    def __init__(self,obs_dim=844,critic_dim=858,device='cpu',seed=17,retirement_mask=False):
        from gym.spaces import Box
        self.device=torch.device(device);torch.manual_seed(seed);np.random.seed(seed)
        config=yaml.safe_load((UPSTREAM/'harl/configs/algos_cfgs/hasac.yaml').read_text())
        # These choices preserve our finite-horizon objective and avoid
        # introducing n-step buffer or normalizer semantics before validation.
        config['train']['use_proper_time_limits']=False
        config['train']['use_valuenorm']=False
        config['algo']['gamma']=1.
        config['algo']['n_step']=1
        args={**config['model'],**config['algo'],**config['train']}
        observation=Box(-np.inf,np.inf,(obs_dim,),dtype=np.float32)
        central=Box(-np.inf,np.inf,(critic_dim,),dtype=np.float32)
        actions=[Box(-1.,1.,(2,),dtype=np.float32) for _ in range(2)]
        cls=RetirementHASAC if retirement_mask else HASAC
        self.actor=[cls(args,observation,a,self.device) for a in actions]
        self.critic=SoftTwinContinuousQCritic(args,central,actions,2,'EP',self.device)
        # Author runner init creates its own unrelated simulator. Construct
        # its algorithm state and call the unmodified author train method.
        runner=OffPolicyHARunner.__new__(OffPolicyHARunner)
        runner.args=dict(algo='hasac');runner.algo_args=config
        runner.actor=self.actor;runner.critic=self.critic;runner.device=self.device
        runner.num_agents=2;runner.state_type='EP';runner.total_it=0
        runner.policy_freq=1;runner.fixed_order=False;runner.value_normalizer=None
        runner.alpha=[args['alpha'],args['alpha']]
        self.runner=runner;self.config=config;self.obs_dim=obs_dim;self.critic_dim=critic_dim
        self.retirement_mask=retirement_mask

    def update(self,batch):
        if len(batch)!=12:raise ValueError('Expected author EP replay sample tuple of length 12')
        central,obs,actions,_,reward,done,valid,term,next_central,next_obs,_,gamma=batch
        n=len(central)
        expected=[(central,(n,self.critic_dim)),(obs,(2,n,self.obs_dim)),(actions,(2,n,2)),
                  (reward,(n,1)),(done,(n,1)),(valid,(2,n,1)),(term,(n,1)),
                  (next_central,(n,self.critic_dim)),(next_obs,(2,n,self.obs_dim)),(gamma,(n,1))]
        for value,shape in expected:
            if not isinstance(value,np.ndarray) or value.shape!=shape or not np.isfinite(value).all():
                raise ValueError('Invalid replay array, expected '+str(shape))
        if np.any(np.sum(valid,axis=(1,2))<=0):
            raise ValueError('Author masked loss needs at least one active sample per actor; resample batch')
        if np.max(np.abs(actions))>1.000001:raise ValueError('Replay expects tanh action, before legal environment mapping')
        if not np.isin(done,[0,1]).all() or not np.isin(term,[0,1]).all():raise ValueError('Invalid terminal flags')
        if self.retirement_mask:
            if not np.array_equal(obs[:,:,-1:],valid):raise ValueError('Retirement flags disagree with public observation')
            if np.any(np.abs(actions[valid[...,0]==0])>1e-7):raise ValueError('Retired replay actions must be zero')
        self.runner.buffer=BatchSource(batch)
        self.runner.train()
        modules=[a.actor for a in self.actor]+[self.critic.critic,self.critic.critic2]
        if not all(torch.isfinite(p).all() for m in modules for p in m.parameters()):
            raise RuntimeError('Nonfinite HASAC update')

