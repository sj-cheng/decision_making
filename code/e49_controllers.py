"""E49 process-local adaptations; historical source and checkpoints are read-only.

Role speeds are explicit: pursuer PublicMaps always use 1, target maps 1.5.
Only target-only estimator namespaces get target-map factories.
"""
from __future__ import annotations
import importlib
import inspect
import json
import textwrap
import time
from pathlib import Path
import types
import numpy as np
import torch
from solvers.frozen_d_search_interface import PublicMap, safe, sensor_visible, target_step
from learning.e45b_joint_network import public_features as _old_features, Runtime, Network, KEYS
from learning.l5_followup import SharedPoolController
from solvers.e45b_joint_nte import Controller, Node, TIME_COST, ETA
from solvers.e45f_baselines import GeometryController

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17,29,43)
HORIZON = 80
OUT = None
_INSTALLED = False


def method_specs():
    result=[]
    learned={'A3','A5','SINGLE_PRIOR','MAPPO_RouteAPF','PPO_C'}
    for arm in ('IG','AG','RouteAPF','MAPPO_RouteAPF','GEOM_SEARCH','SINGLE_PRIOR','A1','A2','A3','A4','A5','PPO_C'):
        for seed in (SEEDS if arm in learned else (17,)):
            spec=dict(name=f'{arm}_{seed}' if arm in learned else arm,arm=arm,train_seed=seed)
            if arm in ('A3','A5'):
                spec['checkpoint']=str(ROOT/f'current/e46_geometry_teacher_20261105/training/L5_{seed}_base/selected.npz')
            elif arm=='SINGLE_PRIOR':
                spec['checkpoint']=str(ROOT/f'current/e45c_budget_scaling_20261028/learning/x20/round_12/models/single_{seed}.npz')
            elif arm=='PPO_C':
                spec['checkpoint']=str(ROOT/f'current/e45w_joint_ppo_and_planning_20261103/training/C_{seed}/selected_actor.pt')
            elif arm=='MAPPO_RouteAPF':
                selected=ROOT/f'current/e45j_repaired_benchmark_20261031/learning/MAPPO_APF_{seed}/selected.json'
                spec['checkpoint']=json.loads(selected.read_text())['checkpoint']
            result.append(spec)
    return result


def public_features(boxes,positions,active_p,active_e,samples,weights,known,records,speed,remaining):
    """Same masked obstacle sum and widths; 24 tokens, true H clock."""
    boxes=np.asarray(boxes,float).reshape(-1,2,2)
    if len(boxes)>24: raise ValueError('E49 obstacle capacity is 24')
    # Reuse feature arithmetic on a capacity-safe empty map then fill every box.
    x=_old_features(np.empty((0,2,2)),positions,active_p,active_e,samples,weights,known,records,speed,remaining)
    center=positions[np.asarray(active_p,bool)].mean(0) if np.any(active_p) else positions.mean(0)
    x['maps']=np.zeros((24,4));x['map_mask']=np.zeros(24)
    x['maps'][:len(boxes),:2]=(boxes.mean(2)-center)/20
    x['maps'][:len(boxes),2:]=(boxes[:,:,1]-boxes[:,:,0])/20
    x['map_mask'][:len(boxes)]=1
    step=records[-1]['step'] if records else HORIZON-remaining
    for i,r in enumerate(records[-16:]):
        x['history'][i,0]=(step-r['step'])/HORIZON
        x['history'][i,13]=r['remaining']/HORIZON
    x['public'][2]=remaining/HORIZON
    return x


def optimal_assignment(cost,active_p,active_e,independent=False):
    from solvers.e45_team_adapter import optimal_assignment as legacy
    pp=np.flatnonzero(active_p);ee=np.flatnonzero(active_e)
    if len(ee)==1:return {int(i):int(ee[0]) for i in pp}
    return legacy(cost,active_p,active_e,independent)


def _clone(fn, replacements, extra=None):
    """Adapt a frozen function in memory, retaining its original globals."""
    fn=inspect.unwrap(fn)
    source=textwrap.dedent(inspect.getsource(fn))
    for before,after in replacements:source=source.replace(before,after)
    namespace=dict(fn.__globals__);namespace.update(extra or {})
    exec(compile(source,'<e49:'+fn.__qualname__+'>','exec'),namespace)
    return namespace[fn.__name__]


def install(horizon=80,out=None):
    """Install E49-only process hooks. Call before constructing controllers."""
    global HORIZON,OUT,_INSTALLED
    horizon=int(horizon)
    if horizon<=0:raise ValueError('horizon must be positive')
    HORIZON=horizon
    if out is not None:OUT=Path(out)
    torch.set_num_threads(1)
    from e49_rules import warmup_rules
    warmup_rules()
    import solvers.e45b_joint_nte as joint
    import solvers.e45_team_adapter as team
    import solvers.e45f_support_refinement as refine
    import learning.e45g_ppo as g
    import learning.e45i_official_mappo as official
    import learning.e45j_adapter_repairs as repairs
    import learning.e45j_repaired_mappo as mappo
    import learning.e46_geometry_teacher as teacher
    import e45c_belief_diagnostics as diagnostic
    from learning.e45w_support_recovery import TeamBelief as RecoveringBelief
    import learning.e45w_support_recovery as recovery
    diagnostic.EVIDENCE=recovery.EVIDENCE=OUT/'belief_failures' if OUT else None
    from speed_difficulty_adapter import speed_lattice,speed_map_probabilities,speed_probabilities
    frozen=importlib.import_module('l45_corner_shared_20261025.corner_estimator')
    # Clock translations are confined to clock checks, not spatial constants.
    if not _INSTALLED:
        frozen.Estimator.update=_clone(frozen.Estimator.update,[('80-obs.step','E49_H()-obs.step')],{'E49_H':lambda:HORIZON})
        diagnostic.RecordedTeamBelief.update_team=_clone(diagnostic.RecordedTeamBelief.update_team,[('80-obs[\'step\']','E49_H()-obs[\'step\']')],{'E49_H':lambda:HORIZON})
    def make_graph(last,boxes):
        p,l,n,s,r,q=speed_lattice(last,boxes,.75,1.5)
        return p,l,n,np.zeros((len(p),l.shape[1],9)),s,r,q
    def probabilities(p,l,n,last,boxes,v,d):return speed_map_probabilities(p,l,n,last,boxes,v,d,1.5)
    def target_map(boxes,_legacy_limit):return PublicMap(boxes,1.5)
    frozen.make_graph=joint.make_graph=make_graph
    frozen.map_probabilities=joint.map_probabilities=refine.map_probabilities=probabilities
    frozen.PublicMap=refine.PublicMap=target_map
    frozen.Estimator.update.__globals__.update(make_graph=make_graph,map_probabilities=probabilities,PublicMap=target_map)
    refine.lattice=lambda last,boxes,spacing:speed_lattice(last,boxes,spacing*1.5,1.5)[:5]
    refine.motion_probabilities=lambda p,l,n,m:speed_probabilities(p,l,n,m,1.5)
    class Belief(RecoveringBelief):
        def update(self,obs):
            if self.public_history and obs['step']!=self.public_history[-1]['step']+1:raise ValueError('Nonconsecutive E49 history')
            if obs['remaining']!=HORIZON-obs['step']:raise ValueError('Bad E49 clock')
            return super().update(obs)
    team.TeamBelief=joint.TeamBelief=g.TeamBelief=official.TeamBelief=Belief
    from solvers.e45f_geometry_capacity import geometric_candidates_capacity
    team.geometric_candidates=geometric_candidates_capacity
    joint.public_features=teacher.public_features=g.public_features=official.public_features=public_features
    official.optimal_assignment=repairs.optimal_assignment=optimal_assignment
    mappo.inputs=mappo_inputs
    _INSTALLED=True
    return dict(revision='E49-persistent-full24-v1',horizon=HORIZON,pursuer_speed=1.,evader_speed=1.5,
                capture_dt=.01,obstacle_capacity=24,pursuer_factory=make_controller)


def _trajectory(self):
    from e49_rules import joint_transition
    ids,modes=self.sample_world();node=self.root;path=[];rewards=[];trace=[];self.active_trace=trace
    while node.depth<min(8,self.remaining):
        self.allowed();chosen=self.select(node);action=node.actions[chosen]
        before=np.zeros((2,2));target=np.zeros((2,2))
        for j,g in enumerate(self.graphs):
            if not node.ae[j]:continue
            before[j]=g[0][ids[j]]
            ids[j],modes[j]=target_step(ids[j],modes[j],g[1],g[2],g[3],self.rng)
            target[j]=g[0][ids[j]]
            if not safe(before[j],target[j]-before[j],1.5,self.public.boxes,np.zeros(2,np.int64)):raise RuntimeError('Illegal E49 target transition')
        evader_action=target-before
        positions,target,ap,ae,events=joint_transition(node.positions,before,action,evader_action,node.ap,node.ae,self.public.boxes)
        self.e49_simulated_evidence.append(dict(positions=node.positions.copy(),targets=before.copy(),pursuer_actions=action.copy(),evader_actions=evader_action.copy(),active_p=node.ap.copy(),active_e=node.ae.copy(),next_positions=positions.copy(),next_targets=target.copy(),next_active_p=ap.copy(),next_active_e=ae.copy(),events=events))
        visible=np.full(2,-2,np.int64)
        for j in range(2):
            if ae[j]:visible[j]=0 if any(sensor_visible(positions[i],target[j],self.public.boxes) for i in range(2) if ap[i]) else -1
        observed=np.array([ids[j] if visible[j]>=0 else visible[j] for j in range(2)])
        trace.append(np.r_[node.positions.ravel(),positions.ravel(),before.ravel(),target.ravel(),node.ap,node.ae,ap,ae,visible>=0])
        path.append((node,chosen));raw=float(not ae.any())-TIME_COST
        previous=self.potential(node)
        if not ae.any() or node.depth+1>=self.remaining:rewards.append(raw-ETA*previous);value=0.;break
        child,fresh=self.child(node,chosen,positions,ap,ae,observed)
        # The inherited child creates the same beliefs; only its public clock changes.
        child.records[-1]['step']=self.records[-1]['step']+node.depth+1
        rewards.append(raw+ETA*(self.potential(child)-previous));node=child
        if fresh or node.depth>=min(8,self.remaining):
            self.parameters(node);value=0.;break
    self.allowed()
    for (visited,chosen),reward in zip(reversed(path),reversed(rewards)):
        value=reward+value;visited.n+=1;visited.visits[chosen]+=1;visited.totals[chosen]+=value
    self.traces.append(np.asarray(trace).reshape(-1,26));self.active_trace=None


def mappo_inputs(features,public,obs):
    from learning.e45j_adapter_repairs import canonical_features
    from learning.e45j_repaired_mappo import clearance
    x=canonical_features(features)
    flat=np.concatenate([np.asarray(x[k],np.float32).ravel() for k in KEYS]);moments=[];offsets=[]
    for j in range(2):
        w=x['weights'][j];points=x['particles'][j,:,:2]
        if w.sum()>0:
            mean=np.average(points,axis=0,weights=w);variance=np.average((points-mean)**2,axis=0,weights=w)
            covariance=np.average((points[:,0]-mean[0])*(points[:,1]-mean[1]),weights=w)
            moments.extend(np.r_[mean,np.sqrt(np.maximum(variance,0)),covariance]);offsets.append(mean[None,:]-x['agents'][:,:2])
        else:moments.extend(np.zeros(5));offsets.append(np.zeros((2,2)))
    rays=clearance(np.asarray(obs['positions'],float),np.asarray(obs['active_p'],bool),public.speed,public.boxes)
    shared=np.r_[flat,moments]
    return np.stack([np.r_[shared,offsets[0][i],offsets[1][i],rays[i],np.eye(2)[i]] for i in range(2)]).astype(np.float32),np.r_[shared,rays.ravel()].astype(np.float32)


def semantic_columns(old_dim):
    # old maps32 -> maps96; old mask8 -> mask24; everything after stays ordered.
    return np.r_[np.arange(32),np.arange(32,40)+64,np.arange(40,old_dim)+80]


def migrate_flat_state(state,old_dim):
    columns=semantic_columns(old_dim);new_dim=old_dim+80;result={}
    for key,value in state.items():
        value=value.detach().cpu().clone()
        if key.endswith('feature_norm.weight') and value.shape==(old_dim,):
            new=torch.ones(new_dim,dtype=value.dtype);new[columns]=value;value=new
        elif key.endswith('feature_norm.bias') and value.shape==(old_dim,):
            new=torch.zeros(new_dim,dtype=value.dtype);new[columns]=value;value=new
        elif key.endswith('mlp.fc1.0.weight') and value.shape[1]==old_dim:
            new=torch.zeros((value.shape[0],new_dim),dtype=value.dtype);new[:,columns]=value;value=new
        result[key]=value
    return result


def load_mappo(path):
    from learning.e45i_official_mappo import settings,R_MAPPOPolicy
    from gym.spaces import Box
    payload=torch.load(path,map_location='cpu',weights_only=False)
    model=R_MAPPOPolicy(settings(),Box(-np.inf,np.inf,(924,),dtype=np.float32),Box(-np.inf,np.inf,(938,),dtype=np.float32),Box(-np.inf,np.inf,(4,),dtype=np.float32),torch.device('cpu'))
    model.actor.load_state_dict(migrate_flat_state(payload['actor'],844),strict=True)
    model.critic.load_state_dict(migrate_flat_state(payload['critic'],858),strict=True)
    return model,payload


class PPOController:
    def __init__(self,boxes,seed,path):
        from learning.e45i_official_mappo import PublicState
        self.state=PublicState(boxes,1.,seed);self.public=self.state.public;self.belief=self.state.belief
        raw=torch.load(path,map_location='cpu',weights_only=False)
        self.network=Network('single',width=raw['trunk.0.weight'].shape[0]);self.network.load_state_dict({k:torch.as_tensor(v) for k,v in raw.items()},strict=True);self.network.eval()
        self.rng=np.random.default_rng(seed+94572030);self.last_stats={}
    def observe(self,obs):self.state.observe(obs)
    @torch.no_grad()
    def act(self,obs,fixed_simulations=None):
        from learning.e45g_ppo import safe_action
        self.observe(obs);features=self.state.features(obs)
        batch={k:torch.as_tensor(v[None],dtype=torch.float32) for k,v in features.items()}
        (_,mu,logs),_=self.network(batch)
        latent=mu[0,0].numpy()+np.exp(logs[0,0].numpy())*self.rng.standard_normal(4)
        action,short=safe_action(np.asarray(obs['positions']),np.tanh(latent).reshape(2,2),obs['active_p'],1.,self.public.boxes)
        self.last_stats=dict(simulations=0,safety_shortened=short,deployment='stochastic',team_belief=self.belief.last_stats)
        return action


def make_controller(case,task):
    if not _INSTALLED:install(task.get('horizon',case.get('horizon',80)),task.get('out'))
    arm=task['arm'];seed=int(task.get('seed',17));budget=float(task.get('budget_ms',200));boxes=np.asarray(case['map'],float).reshape(-1,2,2)
    checkpoint=task.get('checkpoint')
    if checkpoint is None:checkpoint=next((s.get('checkpoint') for s in method_specs() if s['arm']==arm and s['train_seed']==int(task.get('train_seed',17))),None)
    if arm in ('A1','A2','A3','A4','A5'):
        own=SharedPoolController(boxes,1.,seed,ordering=dict(A1='uniform',A2='geometric',A3='learned',A4='geometric',A5='learned')[arm],rank_path=checkpoint,use_policy=False,use_value=False,direct=arm in ('A4','A5'),budget_ms=budget,injected_root=task.get('injected_root'))
    elif arm=='GEOM_SEARCH':own=GeometryController(boxes,1.,seed,use_policy=False,use_value=False,budget_ms=budget)
    elif arm=='SINGLE_PRIOR':own=Controller(boxes,1.,seed,checkpoint=checkpoint,use_policy=True,use_value=False,budget_ms=budget)
    elif arm in ('IG','AG'):
        from solvers.e45_team_adapter import TeamController
        own=TeamController(boxes,1.,seed,arm,budget_ms=budget)
        # Module-local geometry assignment reference is resolved in act().
        import solvers.e45_team_adapter as team
        if not hasattr(team,'_e49_original_assignment'):team._e49_original_assignment=team.optimal_assignment
        # Avoid recursive alias: optimal_assignment delegates only 2v2 to saved original.
        own.act=types.MethodType(_team_act,own)
    elif arm=='RouteAPF':
        from learning.e45j_adapter_repairs import RouteAPFController
        own=RouteAPFController(boxes,1.,seed)
    elif arm=='MAPPO_RouteAPF':
        from learning.e45j_repaired_mappo import Controller as MC
        model,_=load_mappo(checkpoint);own=MC(boxes,1.,seed,model.actor,'MAPPO_APF',deterministic=True)
    elif arm=='PPO_C':return PPOController(boxes,seed,checkpoint)
    else:raise ValueError('Unknown E49 method '+str(arm))
    if isinstance(own,Controller):
        own.target_map=PublicMap(boxes,1.5);own.trajectory=types.MethodType(_trajectory,own)
        original_act=own.act
        def act(obs,fixed_simulations=None):
            own.e49_simulated_evidence=[]
            return original_act(obs,fixed_simulations)
        own.act=act
        if task.get('search_seed') is not None:
            s=int(task['search_seed']);own.rng[:]=(s+1)%4294967296;own.world_rng=np.random.default_rng(s+3);own.proposal_rng=np.random.default_rng(s+2)
    return own


def _team_act(self,obs,fixed_simulations=None):
    started=time.perf_counter();self.observe(obs);cost,options=self.geometry(obs)
    assignment=optimal_assignment(cost,obs['active_p'],obs['active_e'],self.method=='IG');action=np.zeros((2,2))
    for i,j in assignment.items():action[i]=options[i,j]
    self.last_stats=dict(total_ms=1000*(time.perf_counter()-started),simulations=0,assignment={str(i):int(j) for i,j in assignment.items()},team_belief=self.belief.last_stats)
    return action
