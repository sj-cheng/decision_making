"""Public-root counterfactuals and event-continuation planning diagnostics.

All policies/leaf inputs are public packets. Sampled world states occur only
inside the physical/generative simulator. The fixed-D reference is explicitly
root-anchored; RootPUCT references use the original adaptive 3000-sim opponent.
"""
import copy
import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from learning.e45b_joint_network import Network, KEYS
from learning.e45w_joint_env import PublicFeatureState
from learning.e45g_ppo import safe_action
from learning.e45w_support_recovery import install
from solvers.e45b_joint_nte import (Controller, Node, BudgetAbort, ETA, TIME_COST,
    coverage_potential, simultaneous_capture)
from solvers.e45_team_adapter import public_observation
from solvers.frozen_d_search_interface import safe, target_step
from solvers.root_audit_puct import RootPUCT
from corner_estimator_scenarios import original_model
from run_frozen_d_closed_loop import legal_action


ARMS = ('original8', 'deeper16', 'generic16', 'specialized16', 'event16')


def boxes_of(case):
    return np.asarray(case['map'],dtype=float).reshape(-1,2,2)


def seed(*parts):
    return int.from_bytes(hashlib.sha256('/'.join(map(str, parts)).encode()).digest()[:4], 'little')


class FrozenActor:
    def __init__(self, path, device='cuda:0'):
        self.device = torch.device(device)
        self.model = Network('single').to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.embedding = None
        self.model.trunk.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.embedding = output.detach()

    @torch.no_grad()
    def infer(self, features):
        x = {k: torch.as_tensor(features[k], device=self.device, dtype=torch.float32)[None] for k in KEYS}
        (logits, mean, logs), value = self.model(x)
        probs = logits.softmax(-1)[0].cpu().numpy()
        params = (probs, mean[0].cpu().numpy(), logs[0].cpu().numpy())
        return params, float(value.item()), self.embedding[0].cpu().numpy().astype(float)

    def __call__(self, features):
        params, value, _ = self.infer(features)
        return params, value

    @staticmethod
    def sample(parameters, rng):
        p, mean, logs = parameters
        k = rng.choice(len(p), p=p)
        return np.tanh(mean[k] + np.exp(logs[k])*rng.standard_normal(4)).reshape(2, 2)

    def action(self, features, obs, speed, boxes, noise_seed, deterministic=False):
        params, _, z = self.infer(features)
        normalized = np.tanh(params[1][0]).reshape(2, 2) if deterministic else self.sample(params, np.random.default_rng(noise_seed))
        action, _ = safe_action(np.asarray(obs['positions']), normalized,
                                np.asarray(obs['active_p']), speed, boxes)
        return action, z


def public_node(controller, obs):
    samples = np.zeros((2, 64, 2)); weights = np.zeros((2, 64))
    for j, e in enumerate(controller.belief.estimators):
        if obs['active_e'][j]:
            samples[j] = e.particles; weights[j] = e.particle_weights
    return Node(obs['positions'], obs['active_p'], obs['active_e'], [None, None],
                list(controller.belief.known), 0, list(controller.records), samples, weights)


def prepare_public(root, case, actor):
    install()
    controller = Controller(boxes_of(case), case['speed'], root['controller_seed'], use_value=False)
    controller.runtime = actor
    for obs in root['history']: controller.observe(obs)
    controller.remaining = root['history'][-1]['remaining']
    controller.root = public_node(controller, root['history'][-1])
    controller.prepare_models(controller.root)
    own = PublicFeatureState.__new__(PublicFeatureState)
    own.public = controller.public; own.belief = controller.belief
    own.records = copy.deepcopy(controller.records)
    return SimpleNamespace(own=own, controller=controller,
        graphs=controller.graphs, joints=controller.root.joints,
        case=case, problem=original_model(boxes_of(case), case['speed']))


def clone_public(own):
    # Geometry/kernel arrays are immutable in the estimator. Preserve separate
    # posterior/RNG/parent state while avoiding copying its full motion graph.
    memo = {}
    for e in own.belief.estimators:
        if e.graph is not None:
            for a in e.graph:
                if isinstance(a, np.ndarray): memo[id(a)] = a
    result = copy.copy(own)
    result.belief = copy.deepcopy(own.belief, memo)
    result.records = copy.deepcopy(own.records)
    return result


def initial_world(root, prepared, stream, namespace):
    obs = root['history'][-1]; p = prepared.problem
    state = np.zeros((p.state_dim, 1))
    state[4:8, 0] = np.asarray(obs['positions']).ravel()
    state[8, 0] = obs['step']; state[p.active_idxs, 0] = np.r_[obs['active_e'], obs['active_p']]
    ids, modes = [], []
    for j, (g, weights) in enumerate(zip(prepared.graphs, prepared.joints)):
        rng = np.random.default_rng(seed(root['id'], namespace, stream, 'world', j))
        k = int(rng.choice(weights.size, p=weights.ravel()/weights.sum()))
        i, m = divmod(k, 9); ids.append(i); modes.append(m)
        state[2*j:2*j+2, 0] = g[0][i]
    check = public_observation(obs['step'], state[4:8, 0].reshape(2, 2), state[:4, 0].reshape(2, 2),
                               np.asarray(obs['active_p']), np.asarray(obs['active_e']), boxes_of(prepared.case))
    if check != obs: raise RuntimeError('Sampled public-root world contradicts the root observation')
    if not p.is_valid(state): raise RuntimeError('Invalid posterior world')
    return state, ids, modes


def new_opponent(prepared, opponent):
    return RootPUCT([None]*4, None, 3000, 0., 0.) if opponent == 'rootpuct' else None


def advance(prepared, state, physical, opponent, ids, modes, noise_key):
    p=prepared.problem; boxes=boxes_of(prepared.case); speed=prepared.case['speed']
    ap=state[p.active_idxs[2:],0]>.5; ae=state[p.active_idxs[:2],0]>.5
    for i in range(2):
        if not legal_action(state[4+2*i:6+2*i,0], physical[i], speed, boxes): raise RuntimeError('Illegal own action')
        if not ap[i] and np.any(physical[i] != 0): raise RuntimeError('Retired own action')
    if opponent is None:
        enemy=np.zeros((2,2))
        for j,g in enumerate(prepared.graphs):
            if not ae[j]: continue
            previous=g[0][ids[j]].copy()
            rng=np.array([seed(noise_key,j)],np.int64)
            ids[j],modes[j]=target_step(ids[j],modes[j],g[1],g[2],g[3],rng)
            enemy[j]=g[0][ids[j]]-previous
    else:
        opponent.seeded_state(p,seed(noise_key,'solver'),seed(noise_key,'problem'))
        enemy=np.asarray(opponent.search(p,state.copy(),turn=0).best_action).ravel()[:4].reshape(2,2).copy()
    shields=0
    for j in range(2):
        if not ae[j]:enemy[j]=0
        elif not legal_action(state[2*j:2*j+2,0],enemy[j],1.,boxes):
            if opponent is None:raise RuntimeError('Illegal fixed-D transition')
            enemy[j]=0;shields+=1
    joint=np.r_[enemy.ravel(),physical.ravel()][:,None]
    nxt=p.step(state,joint,1.)
    if not p.is_valid(nxt):raise RuntimeError('Invalid physical successor')
    np.testing.assert_allclose(nxt[:8],state[:8]+joint,atol=1e-10,rtol=0)
    nap,nae,_=simultaneous_capture(nxt[4:8,0].reshape(2,2),nxt[:4,0].reshape(2,2),ap,ae,boxes)
    np.testing.assert_array_equal(nap,nxt[p.active_idxs[2:],0]>.5)
    np.testing.assert_array_equal(nae,nxt[p.active_idxs[:2],0]>.5)
    done=bool(p.is_terminal(nxt)); reward=float(not nae.any())-TIME_COST
    obs=public_observation(int(nxt[8,0]),nxt[4:8,0].reshape(2,2),nxt[:4,0].reshape(2,2),nap,nae,boxes)
    return nxt,obs,reward,done,int(ae.sum()-nae.sum()),shields


def phi(own, obs):
    samples=np.zeros((2,64,2)); weights=np.zeros((2,64))
    for j,e in enumerate(own.belief.estimators):
        if obs['active_e'][j]:samples[j]=e.particles;weights[j]=e.particle_weights
    m=own.public
    return float(coverage_potential(np.asarray(obs['positions']),np.asarray(obs['active_p']),
        np.asarray(obs['active_e']),samples,weights,m.speed,m.vertices,m.dist,m.boxes))


def imminent(own, obs):
    if sum(obs['active_p'])!=2 or sum(obs['active_e'])!=2:return False
    positions=np.asarray(obs['positions']);reach=1+np.sqrt(2)*(own.public.speed+1.)
    mass=0.
    for e in own.belief.estimators:
        near=np.linalg.norm(e.particles[:,None,:]-positions[None,:,:],axis=-1).min(1)<=reach
        mass=max(mass,float(e.particle_weights[near].sum()))
    return mass>=.35


def root_candidates(root, prepared, actor, number=6):
    obs=root['history'][-1];features=prepared.own.features(obs);params,_,_=actor.infer(features)
    rng=np.random.default_rng(seed(root['id'],'candidates')); speed=prepared.case['speed'];boxes=boxes_of(prepared.case)
    normalized=[np.tanh(params[1][0]).reshape(2,2)]
    normalized += [actor.sample(params,rng) for _ in range(3)]
    normalized += [np.zeros((2,2)),rng.uniform(-1,1,(2,2))]
    result=[]; sources=[]
    for n,raw in enumerate(normalized):
        action,_=safe_action(np.asarray(obs['positions']),raw,np.asarray(obs['active_p']),speed,boxes)
        if not any(np.max(np.abs(action-old))<1e-9 for old in result):
            result.append(action);sources.append(('mean','sample','sample','sample','stop','uniform')[n])
    for _ in range(128):
        if len(result)>=number:break
        action,_=safe_action(np.asarray(obs['positions']),rng.uniform(-1,1,(2,2)),np.asarray(obs['active_p']),speed,boxes)
        if not any(np.max(np.abs(action-old))<1e-9 for old in result):result.append(action);sources.append('uniform_fill')
    if len(result)!=number:raise RuntimeError('Could not produce the registered distinct root candidates')
    return np.asarray(result),sources


def counterfactual(root, prepared, actor, candidate, stream, namespace, examples=False):
    own=clone_public(prepared.own);state,ids,modes=initial_world(root,prepared,stream,namespace)
    opponent=new_opponent(prepared,root['opponent']);obs=root['history'][-1]
    rewards=[];first=None;phi0=phi(own,obs);cuts={};labels=[];shields=0;lost_after_first=0
    for k in range(obs['remaining']):
        features=own.features(obs)
        physical,z=actor.action(features,obs,prepared.case['speed'],boxes_of(prepared.case),
                                seed(root['id'],namespace,stream,'actor',obs['step']))
        if k==0:physical=np.asarray(candidate).copy()
        if examples and (k%8==0 or (first is not None and k==first)):
            labels.append(dict(z=z.copy(),offset=k,phase=int(sum(obs['active_e'])==1),remaining=obs['remaining']))
        state,obs,reward,done,captures,shield=advance(prepared,state,physical,opponent,ids,modes,
            (root['id'],namespace,stream,'enemy',obs['step']))
        rewards.append(reward);shields+=shield
        if captures and first is None:first=k+1
        if any(obs['active_p']):own.observe(obs)
        if first is not None and sum(obs['active_e'])==1:
            target=next(j for j,a in enumerate(obs['active_e']) if a)
            lost_after_first+=int(all(q is None for q in obs['reports'][target]))
        if k+1 in (8,16):cuts[str(k+1)]=sum(rewards)+ETA*((0. if done else phi(own,obs))-phi0)
        if done:break
    total=float(sum(rewards));success=not any(obs['active_e'])
    for horizon in (8,16):
        if str(horizon) not in cuts:cuts[str(horizon)]=total-ETA*phi0
    for label in labels:label['label']=float(sum(rewards[label['offset']:]))
    return dict(utility=total,double_capture=success,decisions=len(rewards),first_capture=first,
        remaining_after_first=(len(rewards)-first if first is not None else None),
        loss_after_first_steps=lost_after_first,short_scores=cuts,examples=labels,
        opponent_shields=shields,initial_world_sha256=hashlib.sha256(initial_world(root,prepared,stream,namespace)[0].tobytes()).hexdigest())


class LeafReadout:
    def __init__(self,path):
        with np.load(path) as f:self.arrays={k:f[k].copy() for k in f.files}

    def raw(self,z,remaining,residual=False):
        a=self.arrays;x=np.r_[(z-a['mean'])/a['scale'],1.]
        key='residual' if residual and bool(a['residual_available']) else 'generic'
        value=float(x@a[key])
        return float(np.clip(value,-TIME_COST*remaining,1.-TIME_COST))


class EventController(Controller):
    def __init__(self,*args,actor,arm,heads=None,candidates=None,**kwargs):
        super().__init__(*args,use_value=False,**kwargs)
        self.runtime=actor;self.arm=arm;self.heads=heads;self.candidates=candidates
        self.horizon=8 if arm=='original8' else 16
        self.event_evaluations=0;self.generic_evaluations=0

    def parameters(self,node):
        if node.parameters is None:
            params,value,z=self.runtime.infer(self.features(node))
            node.parameters=params;node.value=value;node.embedding=z
        return node.parameters

    def select(self,node):
        if node is not self.root:return super().select(node)
        if not node.actions:
            node.actions=[a.copy() for a in self.candidates]
            node.visits=[0]*len(node.actions);node.totals=[0.]*len(node.actions)
        for i,n in enumerate(node.visits):
            if n==0:return i
        visits=np.asarray(node.visits)
        return int(np.argmax(np.asarray(node.totals)/visits+.1*np.sqrt(np.log(node.n+1)/visits)))

    def leaf(self,node,event=False):
        if self.arm in ('original8','deeper16'):return 0.
        self.parameters(node);z=node.embedding
        residual=self.arm in ('specialized16','event16') and node.ae.sum()==1
        raw=self.heads.raw(z,self.remaining-node.depth,residual)
        if event:self.event_evaluations+=1
        else:self.generic_evaluations+=1
        # Undo the leaf potential in the bootstrap; raw task return stays the
        # target, and terminal success is never added a second time.
        return raw-ETA*self.potential(node)

    def trajectory(self):
        ids,modes=self.sample_world();node=self.root;path=[];rewards=[];trace=[];self.active_trace=trace
        while node.depth<min(self.horizon,self.remaining):
            self.allowed();chosen=self.select(node);action=node.actions[chosen]
            before=np.zeros((2,2));targets=np.zeros((2,2))
            for j,g in enumerate(self.graphs):
                if not node.ae[j]:continue
                before[j]=g[0][ids[j]]
                ids[j],modes[j]=target_step(ids[j],modes[j],g[1],g[2],g[3],self.rng)
                targets[j]=g[0][ids[j]]
                if not safe(before[j],targets[j]-before[j],1.,self.public.boxes,np.zeros(2,np.int64)):
                    raise RuntimeError('Illegal simulated enemy action')
            positions=node.positions+action
            if not all(safe(node.positions[i],action[i],self.public.speed,self.motion_boxes,np.zeros(2,np.int64)) for i in range(2)):
                raise RuntimeError('Illegal simulated own action')
            ap,ae,visible=simultaneous_capture(positions,targets,node.ap,node.ae,self.public.boxes)
            observed=np.asarray([ids[j] if visible[j]>=0 else visible[j] for j in range(2)])
            trace.append(np.r_[node.positions.ravel(),positions.ravel(),before.ravel(),targets.ravel(),node.ap,node.ae,ap,ae,visible>=0])
            path.append((node,chosen));raw=float(not ae.any())-TIME_COST
            previous=self.potential(node)
            if not ae.any() or node.depth+1>=self.remaining:
                rewards.append(raw-ETA*previous);value=0.;break
            child,fresh=self.child(node,chosen,positions,ap,ae,observed)
            rewards.append(raw+ETA*(self.potential(child)-previous))
            event=bool(node.ae.sum()==2 and ae.sum()==1)
            node=child
            if fresh or node.depth>=min(self.horizon,self.remaining) or (self.arm=='event16' and event):
                self.parameters(node);value=self.leaf(node,event=self.arm=='event16' and event);break
        self.allowed()
        for (visited,chosen),reward in zip(reversed(path),reversed(rewards)):
            value=reward+value;visited.n+=1;visited.visits[chosen]+=1;visited.totals[chosen]+=value
        self.traces.append(np.asarray(trace).reshape(-1,26));self.active_trace=None

    def act(self,obs,fixed_simulations=None):
        # The original implementation hard-codes horizon8 only in its stats
        # and trajectory. Our trajectory override carries the actual horizon.
        action=super().act(obs,fixed_simulations)
        self.last_stats.update(planning_depth=self.horizon,event_evaluations=self.event_evaluations,
                               generic_evaluations=self.generic_evaluations)
        return action
