"""Actor-only HASAC deployment, behavior matched to the frozen M/N adapter.

No Q network, privileged state, reward, potential, or replay construction is
called online. Actor action RNG is per-controller, independent of interleaving.
"""
import hashlib
from pathlib import Path
import numpy as np
import torch
import yaml
from learning.e45m_author_hasac import RetirementHASAC,UPSTREAM
from learning.e45i_official_mappo import PublicState
from learning.e45j_repaired_mappo import inputs
from learning.e45g_ppo import safe_action

OBS_DIM=845


def load_actors(checkpoint,expected_sha256):
    from gym.spaces import Box
    path=Path(checkpoint)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=expected_sha256:raise ValueError('Actor checkpoint SHA mismatch')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if not isinstance(saved.get('actors'),list) or len(saved['actors'])!=2:
        raise ValueError('Expected a two-agent M/N HASAC checkpoint')
    config=yaml.safe_load((UPSTREAM/'harl/configs/algos_cfgs/hasac.yaml').read_text())
    args={**config['model'],**config['algo'],**config['train']}
    obs_space=Box(-np.inf,np.inf,(OBS_DIM,),dtype=np.float32)
    act_space=Box(-1.,1.,(2,),dtype=np.float32)
    actors=[]
    # Loading must not consume a caller's environment or policy RNG stream.
    with torch.random.fork_rng(devices=[]):
        for weights in saved['actors']:
            if weights['net.mlp.0.weight'].shape[1]!=OBS_DIM:
                raise ValueError('HASAC public-input architecture mismatch')
            actor=RetirementHASAC(args,obs_space,act_space,torch.device('cpu'))
            actor.actor.load_state_dict(weights);actor.actor.eval();actor.turn_off_grad();actors.append(actor)
    metadata=dict(checkpoint=str(path),checkpoint_sha256=digest,steps=saved['steps'],
                  algorithm='HARL HASAC',obs_dim=OBS_DIM,critic_loaded=False,
                  experiment_sha256=saved['experiment_sha256'])
    return actors,metadata


class Controller(PublicState):
    def __init__(self,boxes,speed,belief_seed,actors,stochastic=False,policy_seed=0):
        super().__init__(boxes,speed,belief_seed)
        self.actors=actors;self.stochastic=stochastic
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(policy_seed);self.action_rng=torch.get_rng_state()

    @torch.inference_mode()
    def act(self,obs,fixed_simulations=None):
        PublicState.observe(self,obs)
        public,_=inputs(self.features(obs),self.public,obs)
        active=np.asarray(obs['active_p'],np.float32)[:,None]
        public=np.c_[public,active].astype(np.float32)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.action_rng)
            action=np.stack([a.get_actions(public[i:i+1],stochastic=self.stochastic)[0].cpu().numpy()
                             for i,a in enumerate(self.actors)])
            self.action_rng=torch.get_rng_state()
        result,shortened=safe_action(np.asarray(obs['positions']),action,obs['active_p'],self.public.speed,self.public.boxes)
        self.last_stats=dict(simulations=0,joint_action_search=False,assignment_used=False,roles_used=False,
                             normalized_intent=action.tolist(),safety_shortened=shortened,
                             team_belief=self.belief.last_stats,
                             input_adapter='J public845, author HASAC actors only; no reward/critic online')
        return result
