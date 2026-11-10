from __future__ import annotations
import time
import hashlib
import numpy as np
import torch
from torch import nn
from .geometry import bottleneck_assignment
from .core import World, State, Belief, move, visibility
from .isolation import isolated_routes

BRANCHES=('prior','joint','fusion','rank')
METHODS=('IG','AG','RANDOM_MCTS','GEOM','NET','PRIOR','JOINT','JOINT_NV','FUSION',
         'POOL_UNIFORM','POOL_GEOM','POOL_LEARNED','GEOM_DIRECT','RANK_DIRECT','NET_PRIOR','NET_FUSION')

class Network(nn.Module):
    """Masked set encoders and cross attention; team action is one joint distribution."""
    def __init__(self,width=64):
        super().__init__()
        self.width=width
        self.pe=nn.Sequential(nn.Linear(2,width),nn.Tanh(),nn.Linear(width,width))
        self.te=nn.Sequential(nn.Linear(5,width),nn.Tanh(),nn.Linear(width,width))
        self.particle_encoder=nn.Sequential(nn.Linear(11,width),nn.Tanh(),nn.Linear(width,width))
        self.be=nn.Sequential(nn.Linear(4,width),nn.Tanh(),nn.Linear(width,width))
        self.attn=nn.MultiheadAttention(width,4,batch_first=True)
        self.actor=nn.Sequential(nn.Linear(width*5,width),nn.Tanh(),nn.Linear(width,2))
        self.logstd=nn.Parameter(torch.full((2,),-.7))
        self.critic=nn.Sequential(nn.Linear(width*3,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.critic[-1].weight);nn.init.zeros_(self.critic[-1].bias)
        self.clock=nn.Sequential(nn.Linear(1,width),nn.Tanh())
        self.action_encoder=nn.Sequential(nn.Linear(4,width),nn.Tanh(),nn.Linear(width,width))
        self.ranker=nn.Sequential(nn.Linear(width*4,width),nn.Tanh(),nn.Linear(width,1))

    def context(self,x):
        active=x.get('actor_alive',torch.ones_like(x['alive']))
        p=self.pe(x['p'])*active[:,:,None]
        particles=self.particle_encoder(x['particles'])*x['particle_mask'][:,:,:,None]
        particle_context=particles.sum(2)/x['particle_mask'].sum(2).clamp(min=1)[:,:,None]
        target=self.te(x['target'])+particle_context;boxes=self.be(x['boxes'])
        alive=x['alive'].bool()
        # Dummy target prevents all-masked attention at terminal states.
        safe=alive.clone();empty=~safe.any(1);safe[empty,0]=True
        target=target*alive[:,:,None]
        attended=self.attn(p,target,target,key_padding_mask=~safe,need_weights=False)[0]
        target_mean=(target*alive[:,:,None]).sum(1)/alive.sum(1).clamp(min=1)[:,None]
        time=self.clock(x['time'][:,None])
        pooled=torch.cat((p.sum(1)/active.sum(1).clamp(min=1)[:,None],target_mean,(boxes*x['box_mask'][:,:,None]).sum(1)/x['box_mask'].sum(1).clamp(min=1)[:,None]),dim=-1)
        pooled=pooled+time.repeat(1,3)
        actors=torch.cat((p,attended,pooled[:,None,:].expand(-1,p.shape[1],-1)),dim=-1)
        return actors,pooled

    def forward(self,x):
        actors,pooled=self.context(x)
        return self.actor(actors),self.logstd.expand(*actors.shape[:2],2),self.critic(pooled).squeeze(-1)

    def scores(self,x,actions):
        _,pooled=self.context(x)
        p=x['p'][:,None].expand(-1,actions.shape[1],-1,-1)
        encoded=self.action_encoder(torch.cat((p,actions),dim=-1)).mean(2)
        return self.ranker(torch.cat((pooled[:,None].expand(-1,actions.shape[1],-1),encoded),-1)).squeeze(-1)


def features(public,obs,estimates,age=None,belief=None):
    c=public.cfg
    if age is None: age=np.zeros(c.n)
    targets=np.column_stack((estimates/(c.size/2),obs['seen'],np.minimum(age, c.horizon)/c.horizon,obs['alive']))
    samples=estimates[:,None,:] if belief is None else belief.samples
    modes=np.zeros((*samples.shape[:2],9)) if belief is None else np.eye(9)[belief.modes]
    particles=np.concatenate((samples/(c.size/2),modes),axis=-1).astype(np.float32)
    particle_mask=np.broadcast_to(obs['alive'][:,None],samples.shape[:2]).astype(np.float32)
    return dict(p=np.asarray(obs['p']/(c.size/2),np.float32),target=targets.astype(np.float32),particles=particles,particle_mask=particle_mask,
                boxes=(public.boxes.reshape(-1,4)/(c.size/2)).astype(np.float32) if len(public.boxes) else np.zeros((1,4),np.float32),
                box_mask=np.ones(max(1,len(public.boxes)),np.float32) if len(public.boxes) else np.zeros(1,np.float32),alive=obs['alive'].astype(np.float32),actor_alive=np.asarray(obs.get('actor_alive',np.ones(c.n)),np.float32),time=np.float32(obs.get('t',0)/c.horizon))


def tensors(x,device='cpu'):
    return {k:torch.as_tensor(v,device=device)[None] for k,v in x.items()}


def stack_features(rows,device):
    maximum=max(len(x['boxes']) for x in rows)
    maximum_particles=max(x['particles'].shape[1] for x in rows)
    prepared=[]
    for row in rows:
        x=dict(row)
        x.setdefault('actor_alive',np.ones(len(x['p']),np.float32))
        x.setdefault('time',np.float32(0.))
        x['boxes']=np.pad(x['boxes'],((0,maximum-len(x['boxes'])),(0,0)))
        x['box_mask']=np.pad(x['box_mask'],(0,maximum-len(x['box_mask'])))
        x['particles']=np.pad(x['particles'],((0,0),(0,maximum_particles-x['particles'].shape[1]),(0,0)))
        x['particle_mask']=np.pad(x['particle_mask'],((0,0),(0,maximum_particles-x['particle_mask'].shape[1])))
        prepared.append(x)
    return {k:torch.as_tensor(np.stack([x[k] for x in prepared]),device=device) for k in prepared[0]}


def normal_logprob(raw,mean,logstd,active=None):
    result=-.5*((raw-mean)/logstd.exp())**2-logstd-.5*np.log(2*np.pi)
    if active is not None:result=result*active[:,:,None]
    return result.sum((-1,-2))


def geometric(public,p,estimates,alive,assignment=True,variant=0,belief=None):
    active=np.flatnonzero(alive)
    if not len(active):return np.zeros_like(p)
    goals=estimates[active]
    if belief is None:cost=public.distance_matrix(p,goals)
    else:
        samples=belief.samples[active]
        cost=public.distance_matrix(p,samples.reshape(-1,2)).reshape(len(p),len(active),belief.count).mean(2)
    selected=np.argmin(cost,axis=1)
    if assignment:
        rows,cols=bottleneck_assignment(cost);selected[rows]=cols
    if variant:selected=(selected+variant)%len(active)
    return np.array([target_action(public,q,belief.samples[active[j]] if belief is not None else goals[j][None]) for q,j in zip(p,selected)])


def target_action(public,at,samples):
    """Existing per-target G objective: expected route steps plus visibility gain."""
    from .core import visibility
    public._ensure_route_graph()
    compass=np.array([[0,0],[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]],float)
    actions=np.vstack((compass,public.actions_to(at,np.vstack((samples,public._route_vertices)))))
    actions=np.array([a for a in actions if legal_point_action(public,at,a)])
    _,indices=np.unique(np.round(actions,10),axis=0,return_index=True);actions=actions[np.sort(indices)]
    costs=public.distance_matrix(at[None]+actions,samples)
    scores=-costs.mean(1)
    for k,a in enumerate(actions):scores[k]+=visibility((at+a)[None],samples,public.boxes,public.cfg.sensing).mean()
    return actions[int(scores.argmax())]


def geometry_choices(public,p,estimates,alive,belief=None):
    """Scale the existing GEOM waypoint support; no assignment or AG fallback."""
    public._ensure_route_graph();goals=[]
    for j in np.flatnonzero(alive):
        if belief is None:goals.append(estimates[j])
        else:
            indices=np.unique(np.minimum(belief.count-1,((np.arange(8)+.5)*belief.count/8).astype(int)))
            goals.extend(belief.samples[j,indices])
    goals=np.concatenate((np.asarray(goals).reshape(-1,2),public._route_vertices),axis=0)
    result=[]
    for q in p:
        actions=np.vstack((np.zeros((1,2)),public.actions_to(q,goals)))
        _,indices=np.unique(np.round(actions,10),axis=0,return_index=True)
        result.append(actions[np.sort(indices)])
    return result


def candidate_pool(public,p,estimates,alive,rng,model=None,x=None,kind='pool',count=32,belief=None):
    if count!=32:raise ValueError('Registered bounded candidate batch is 32')
    geometry_count=12 if kind=='fusion' else (0 if kind in ('random','prior','joint') else 24)
    options=geometry_choices(public,p,estimates,alive,belief) if kind not in ('random','prior','joint') else None
    def random_action():
        for _ in range(64):
            action=rng.uniform(-1,1,p.shape)
            if rng.random()<.5:action/=np.maximum(np.abs(action).max(1,keepdims=True),1e-12)
            if rng.random()<.02:action[:]=0
            if all(legal_point_action(public,q,a) for q,a in zip(p,action)):return action
        return np.zeros_like(p)
    def geometric_action():return np.array([row[rng.integers(len(row))] for row in options])
    if kind=='geom':
        # Existing GEOM: independently draw .75 route/corner, .25 global.
        actions=[geometric_action() if rng.random()<.75 else random_action() for _ in range(count)]
    else:
        random_count=32 if kind=='random' else 8
        actions=[random_action() for _ in range(random_count)]
        network_count=12 if kind=='fusion' else (24 if kind in ('prior','joint') else 0)
        if network_count:
            if model is None:raise ValueError('Missing learned proposal checkpoint')
            with torch.no_grad():mean,logstd,_=model(tensors(x,next(model.parameters()).device))
            mean=mean[0].detach().cpu().numpy();std=logstd[0].detach().exp().cpu().numpy()
            for _ in range(network_count):
                for attempt in range(64):
                    action=np.tanh(mean+rng.normal(size=p.shape)*std)
                    if all(legal_point_action(public,q,a) for q,a in zip(p,action)):break
                else:action=np.zeros_like(p)
                actions.append(action)
        actions.extend(geometric_action() for _ in range(geometry_count))
    return np.asarray(actions)


def legal_point_action(public,q,action):
    from .core import legal,clear
    return np.max(np.abs(action))<=public.cfg.speed+1e-9 and legal(q+action,public.boxes,public.cfg.size) and clear(q,q+action,public.boxes)


def geometry_scores(public,p,estimates,alive,actions,belief=None):
    if not alive.any():return np.zeros(len(actions))
    samples=estimates[alive,None,:] if belief is None else belief.samples[alive]
    costs=public.coverage(p[None]+actions,samples.reshape(-1,2)).reshape(len(actions),int(alive.sum()),-1)
    return -costs.mean((1,2))/public.cfg.size


from .motion import MotionKernel,shared_model,planning_model


def kernel_action(public,state,rng):
    return MotionKernel(public,rng).actions(state)


def simulated_step(public,belief,state,latent_modes,pa,rng):
    """Same motion prediction and conditioning; only a simulated latent witness is retained."""
    predicted,modes=belief.predict()
    targets,next_modes=planning_model(public).advance(state.e,latent_modes,belief.anchors,rng,state.alive)
    targets[~state.alive]=state.e[~state.alive]
    nxt=World(public.cfg,public.boxes).transition(state,pa,targets-state.e)
    # Conditional SMC: retain one simulated posterior trajectory. Real update()
    # never accepts hidden-state arguments and uses this same predict/condition.
    predicted[:,0]=nxt.e;modes[:,0]=next_modes
    observation=World(public.cfg,public.boxes).observe(nxt)
    belief.condition(observation,predicted,modes)
    next_modes[observation['seen']]=belief.modes[observation['seen'],0]
    return nxt,next_modes


def evader_features(public,state):
    # Reverse role; evaders intentionally receive complete opposing state.
    obs=dict(p=state.e,seen=np.ones(len(state.p),bool),alive=np.ones(len(state.p),bool),actor_alive=state.alive,t=state.t)
    return features(public,obs,state.p)


def evader_action(public,state,rng,model=None):
    if model is None:return kernel_action(public,state,rng)
    with torch.no_grad():mean,_,_=model(tensors(evader_features(public,state),next(model.parameters()).device))
    a=np.tanh(mean[0].cpu().numpy())
    return move(state.e,a,state.alive,public.boxes,public.cfg.size)-state.e


from .rootpuct import rootpuct_evader


DIRECT_METHODS=('IG','AG','NET','NET_PRIOR','NET_FUSION','GEOM_DIRECT','RANK_DIRECT')
BRANCH_METHOD={'prior':'PRIOR','joint':'JOINT','fusion':'FUSION','rank':'POOL_LEARNED'}
METHOD_BRANCH={'NET':'joint','NET_PRIOR':'prior','NET_FUSION':'fusion','PRIOR':'prior','JOINT':'joint','JOINT_NV':'joint','FUSION':'fusion','POOL_LEARNED':'rank','RANK_DIRECT':'rank'}


def public_key(obs,parent=b''):
    h=hashlib.sha256(parent)
    for name in ('p','e','alive','seen'):
        value=np.asarray(obs[name]);h.update(np.nan_to_num(value).tobytes())
    h.update(str(obs['t']).encode());return h.digest()

class ActionNode:
    def __init__(self,action,prior):
        self.action=action;self.prior=float(prior);self.visits=0;self.total=0.;self.observations={}

class Node:
    def __init__(self,key):
        self.key=key;self.children=[];self.visits=0

class Controller:
    def __init__(self,public,method,seed=0,model=None,simulations=64,budget_ms=None,depth=8):
        if method not in METHODS:raise ValueError(method)
        if method in METHOD_BRANCH and model is None:raise ValueError('Missing model for '+method)
        started=time.perf_counter()
        self.public=isolated_routes(public);self.method=method;self.seed=int(seed);self.rng=np.random.default_rng(seed)
        self.belief=Belief(self.public,seed);self.model=model;self.simulations=int(simulations);self.budget_ms=budget_ms
        self.depth=depth;self.world=World(public.cfg,public.boxes);self.last_stats={}
        self.last_seen=np.zeros(public.cfg.n);self.history_key=b'';self.last_history_t=-1;self.traces=[]
        self.setup_ms=1000*(time.perf_counter()-started)

    def observe(self,obs):
        self.belief.update(obs)
        if self.last_history_t!=obs['t']:
            self.history_key=public_key(obs,self.history_key);self.last_history_t=obs['t']
        self.last_seen=self.belief.last_seen.copy()

    def _device(self):return next(self.model.parameters()).device

    def _options(self,node,belief,obs,injected=None):
        estimated=belief.estimates();x=features(self.public,obs,estimated,belief.ages(),belief);m=self.method
        kind={'RANDOM_MCTS':'random','GEOM':'geom','PRIOR':'prior','JOINT':'joint','JOINT_NV':'joint','FUSION':'fusion'}.get(m,'pool')
        seed=int.from_bytes(hashlib.sha256(node.key+str(self.seed).encode()).digest()[:8],'little')
        rng=np.random.default_rng(seed)
        pool=candidate_pool(self.public,obs['p'],estimated,obs['alive'],rng,self.model,x,kind,belief=belief) if injected is None else np.array(injected,copy=True)
        if pool.shape!=(32,self.public.cfg.n,2) or not np.isfinite(pool).all() or np.max(np.abs(pool))>1+1e-8:
            raise ValueError('Malformed injected common action pool')
        if not all(legal_point_action(self.public,q,a) for joint in pool for q,a in zip(obs['p'],joint)):
            raise ValueError('Illegal edge in shared candidate array')
        scores=np.zeros(len(pool))
        if m in ('POOL_GEOM','GEOM_DIRECT'):
            scores=geometry_scores(self.public,obs['p'],estimated,obs['alive'],pool,belief)/.05
        if m in ('POOL_LEARNED','RANK_DIRECT'):
            with torch.no_grad():scores=self.model.scores(tensors(x,self._device()),torch.as_tensor(pool,dtype=torch.float32,device=self._device())[None])[0].cpu().numpy()
        if m in ('POOL_UNIFORM','POOL_GEOM','POOL_LEARNED'):
            order=rng.permutation(len(pool)) if m=='POOL_UNIFORM' else np.argsort(-scores,kind='stable')
        elif m in ('PRIOR','JOINT','JOINT_NV','FUSION'):order=rng.permutation(len(pool))
        else:order=np.arange(len(pool))
        node.children=[ActionNode(pool[i],1./len(pool)) for i in order]
        return x,pool,scores

    def act(self,obs,candidates=None):
        from .semantics import team_reward,backed_return,GAMMA,STEP_COST,ETA,pursuit_potential,shaped_reward
        started=time.perf_counter();c=self.public.cfg;m=self.method
        deadline=np.inf if self.budget_ms is None else started+self.budget_ms/1000
        self.observe(obs)
        estimates=self.belief.estimates();x=features(self.public,obs,estimates,self.belief.ages(),self.belief)
        completed=0;first=None;self.traces=[];root=Node(self.history_key)
        if m in ('IG','AG'):
            action=geometric(self.public,obs['p'],estimates,obs['alive'],assignment=m=='AG',belief=self.belief)
        elif m in ('NET','NET_PRIOR','NET_FUSION'):
            with torch.no_grad():mean,_,_=self.model(tensors(x,self._device()))
            action=np.tanh(mean[0].cpu().numpy())
        else:
            x,pool,scores=self._options(root,self.belief,obs,candidates)
            if m in ('GEOM_DIRECT','RANK_DIRECT'):action=pool[int(scores.argmax())]
            else:
                budget=self.simulations if self.budget_ms is None else 1000000
                root_phi=pursuit_potential(self.public,State(obs['p'],estimates,obs['alive'],obs['t']),self.belief)
                for simulation in range(budget):
                    if time.perf_counter()>=deadline:break
                    belief=self.belief.fork(int(self.rng.integers(2**31)))
                    target,modes=belief.sample_joint(self.rng)
                    state=State(obs['p'].copy(),target,obs['alive'].copy(),obs['t'])
                    node=root;path=[];full=True;trace=[]
                    phi=root_phi
                    for depth in range(self.depth):
                        if time.perf_counter()>=deadline:full=False;break
                        if not node.children:self._options(node,belief,self.world.observe(state))
                        cap=min(len(node.children),max(2,int(np.ceil(2*(node.visits+1)**.35))))
                        eligible=node.children[:cap]
                        selected=next((edge for edge in eligible if edge.visits==0),None)
                        if selected is None:
                            selected=max(eligible,key=lambda edge:edge.total/edge.visits+.1*np.sqrt(np.log(node.visits+1)/edge.visits))
                        previous=state
                        state,modes=simulated_step(self.public,belief,state,modes,selected.action,self.rng)
                        following_phi=pursuit_potential(self.public,state,belief)
                        reward=shaped_reward(team_reward(c,previous,state),phi,following_phi);phi=following_phi
                        path.append((node,selected,reward))
                        if len(self.traces)<2:trace.append((previous.copy(),selected.action.copy(),state.copy()))
                        if self.world.done(state):break
                        key=public_key(self.world.observe(state),node.key)
                        fresh=key not in selected.observations
                        if fresh:selected.observations[key]=Node(key)
                        node=selected.observations[key]
                        if fresh:break
                    if not full:break
                    value=0.
                    if m=='JOINT' and not self.world.done(state):
                        leaf_obs=self.world.observe(state)
                        with torch.no_grad():_,_,leaf=self.model(tensors(features(self.public,leaf_obs,belief.estimates(),belief.ages(),belief),self._device()))
                        value=float(leaf.item())
                        if not np.isfinite(value):raise RuntimeError('Nonfinite learned continuation value')
                    # Each edge stores the local continuation return in exactly
                    # the same units as PPO. No heuristic/double-counted value.
                    for parent,edge,reward in reversed(path):
                        value=reward+GAMMA*value;edge.visits+=1;edge.total+=value;parent.visits+=1
                    completed+=1
                    if trace:self.traces.append(trace)
                    if first is None:first=1000*(time.perf_counter()-started)
                winner=max(root.children,key=lambda z:(z.visits,z.total/max(1,z.visits)))
                action=winner.action if completed else root.children[0].action
            self.last_candidates=pool.copy()
        action=move(obs['p'],action,np.ones(c.n,np.bool_),self.public.boxes,c.size)-obs['p']
        elapsed=1000*(time.perf_counter()-started)
        self.last_stats=dict(elapsed_ms=elapsed,simulations=completed,first_simulation_ms=first,
            zero_complete_simulations=completed==0 and m not in DIRECT_METHODS,
            timeout=self.budget_ms is not None and elapsed>self.budget_ms,belief_recoveries=self.belief.recoveries,
            scope='public update, features, candidates, complete simulations, leaf and legal action return')
        self.last_features=x;self.root=root
        return action
