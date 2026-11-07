"""Role-free public-belief JOINT MCTS. No assignment, pursuit or role oracle.

Node decisions receive public positions, observations and conditional beliefs.
Sampled hidden indices exist only inside the generative transition. The fixed
D dynamics model is an approximation to the adaptive real opponent, not a
claim of solving the adversarial game exactly.
"""
import time
import numpy as np
from numba import njit
from solvers.e45_team_adapter import TeamBelief,team_hidden_mask
from solvers.frozen_d_search_interface import PublicMap,safe,sensor_visible,root_sample,target_step,route
from search_value_model import root_joint,predict_joint
from l45_corner_shared_20261025.multimodal_estimator import make_graph
from l45_corner_shared_20261025.corner_estimator import map_probabilities,geometry_labels,export_regions
from learning.e45b_joint_network import public_features,Runtime

ETA=.05
TIME_COST=.0025


@njit(cache=True)
def simultaneous_capture(positions,targets,ap,ae,boxes):
    pp,ee=ap.copy(),ae.copy()
    for _ in range(2):
        distance=1.;pi,ej=-1,-1
        for i in range(2):
            if not pp[i]:continue
            for j in range(2):
                if not ee[j]:continue
                d=np.sqrt(np.sum((positions[i]-targets[j])**2))
                if d<distance:distance,pi,ej=d,i,j
        if pi<0:break
        pp[pi]=False;ee[ej]=False
    observations=np.full(2,-2,np.int64)
    for j in range(2):
        if ee[j]:
            observations[j]=-1
            for i in range(2):
                if pp[i] and sensor_visible(positions[i],targets[j],boxes):observations[j]=0
    return pp,ee,observations


@njit(cache=True)
def condition_team(prediction,positions,active,points,boxes,observed):
    result=np.zeros_like(prediction)
    for k in range(len(points)):
        if prediction[k].sum()<=0:continue
        seen=False
        for i in range(2):
            if active[i] and sensor_visible(positions[i],points[k],boxes):seen=True
        if (observed>=0 and seen and k==observed) or (observed<0 and not seen):result[k]=prediction[k]
    mass=result.sum()
    if mass<=0 or not np.isfinite(mass):raise RuntimeError('Impossible joint-tree observation')
    return result/mass


@njit(cache=True)
def coverage_potential(positions,active_p,active_e,samples,weights,speed,vertices,dist,boxes):
    # A SCALAR team potential, not an assignment or an action generator.
    # Terminal potential is zero; full shaped returns differ by a root constant.
    value=0.;count=np.zeros(2,np.int64)
    for j in range(2):
        if not active_e[j]:continue
        for k in range(64):
            if weights[j,k]<=0:continue
            best=80.
            for i in range(2):
                if active_p[i]:best=min(best,float(route(positions[i],samples[j,k],speed,vertices,dist,boxes,count)[0]))
            value-=weights[j,k]*best
    return value


class BudgetAbort(Exception):pass


class Node:
    def __init__(self,positions,ap,ae,joints,known,depth,records,samples=None,weights=None):
        self.positions=np.asarray(positions).copy();self.ap=np.asarray(ap,bool).copy();self.ae=np.asarray(ae,bool).copy()
        self.joints=joints;self.known=known;self.depth=depth;self.records=records
        self.samples=samples;self.weights=weights
        self.actions=[];self.visits=[];self.totals=[];self.children={};self.n=0
        self.features=None;self.parameters=None;self.value=None;self.phi=None;self.predictions=[None,None]


class Controller:
    def __init__(self,boxes,speed,seed,checkpoint=None,use_policy=True,use_value=True,direct=False,budget_ms=200.):
        self.public=PublicMap(boxes,speed);self.motion_boxes=self.public.boxes.copy()
        self.motion_boxes[:,:,0]-=1e-6;self.motion_boxes[:,:,1]+=1e-6
        self.target_map=PublicMap(boxes,1.);self.belief=TeamBelief(boxes,seed)
        self.runtime=Runtime(checkpoint) if checkpoint else None
        self.use_policy=use_policy;self.use_value=use_value;self.direct=direct;self.budget_ms=budget_ms
        self.rng=np.array([(seed+94522027)%4294967296],np.int64)
        self.export_rng=np.array([(seed+94522028)%4294967296],np.int64)
        self.proposal_rng=np.random.default_rng(seed+94522029)
        self.world_rng=np.random.default_rng(seed+94522030)
        self.records=[];self.graphs=[None,None];self.regions=[None,None];self.graph_cache=[None,None]
        self.last_stats={};self.root=None;self.traces=[];self.deadline=float('inf');self.operation_ms={}

    def observe(self,obs):
        self.belief.update(obs);self.records.append(obs)

    def allowed(self):
        if time.perf_counter()>=self.deadline-.003:raise BudgetAbort()

    def operation(self,callback,*args):
        key=callback.__name__;estimate=self.operation_ms.get(key,1.)
        if time.perf_counter()+(2*estimate+3)/1000.>=self.deadline:raise BudgetAbort()
        started=time.perf_counter();result=callback(*args)
        self.operation_ms[key]=max(estimate,1000*(time.perf_counter()-started));self.allowed();return result

    def features(self,node):
        if node.features is None:
            if node.samples is None:
                node.samples=np.zeros((2,64,2));node.weights=np.zeros((2,64))
                for j in range(2):
                    if not node.ae[j]:continue
                    ids,w,_=self.operation(export_regions,node.joints[j],self.regions[j],self.export_rng,64)
                    node.samples[j]=self.graphs[j][0][ids];node.weights[j]=w
            node.features=public_features(self.public.boxes,node.positions,node.ap,node.ae,node.samples,node.weights,
                node.known,node.records,self.public.speed,self.remaining-node.depth)
        return node.features

    def potential(self,node):
        if node.phi is None:
            self.features(node)
            p=self.public
            node.phi=self.operation(coverage_potential,node.positions,node.ap,node.ae,node.samples,node.weights,
                p.speed,p.vertices,p.dist,p.boxes)
        return node.phi

    def parameters(self,node):
        if node.parameters is None:
            if self.runtime is None:node.parameters=False;node.value=0.
            else:node.parameters,node.value=self.runtime(self.features(node))
        return node.parameters

    def propose(self,node,direct=False):
        neural=self.runtime is not None and self.use_policy and (direct or self.proposal_rng.random()<.75)
        params=self.parameters(node) if neural else None
        for _ in range(64):
            self.allowed()
            if neural:a=self.runtime.sample(params,self.proposal_rng)*self.public.speed
            else:
                a=self.proposal_rng.uniform(-1,1,(2,2))
                if self.proposal_rng.random()<.5:
                    a/=np.maximum(np.max(np.abs(a),axis=1,keepdims=True),1e-12)
                if self.proposal_rng.random()<.02:a[:]=0
                a*=self.public.speed
            a[~node.ap]=0.
            if all(safe(node.positions[i],a[i],self.public.speed,self.motion_boxes,np.zeros(2,np.int64)) for i in range(2)):
                return a
            self.rejections+=1
        self.fallbacks+=1;return np.zeros((2,2))

    def select(self,node):
        cap=min(16,max(2,int(np.ceil(2*(node.n+1)**.35))))
        if len(node.actions)<cap:
            action=self.propose(node)
            if not any(np.max(np.abs(action-a))<1e-10 for a in node.actions):
                node.actions.append(action);node.visits.append(0);node.totals.append(0.);return len(node.actions)-1
        for i,n in enumerate(node.visits):
            if n==0:return i
        v=np.asarray(node.visits)
        return int(np.argmax(np.asarray(node.totals)/v+.1*np.sqrt(np.log(node.n+1)/v)))

    def prepare_models(self,node):
        for j,e in enumerate(self.belief.estimators):
            if not node.ae[j]:continue
            if e.graph is not None:
                graph=e.graph;regions=e.regions;prior=root_joint(e)
            else:
                key=tuple(np.asarray(self.belief.known[j]).tolist())
                if self.graph_cache[j] is not None and self.graph_cache[j][0]==key:
                    graph,regions=self.graph_cache[j][1:]
                else:
                    graph=list(self.operation(make_graph,np.asarray(key),self.public.boxes))
                    graph[3],_,_=self.operation(map_probabilities,graph[0],graph[1],graph[2],np.asarray(key),
                        self.public.boxes,self.target_map.vertices,self.target_map.dist)
                    graph=tuple(graph);regions=geometry_labels(graph[0],self.public.boxes)
                    self.graph_cache[j]=(key,graph,regions)
                prior=np.zeros((len(graph[0]),9));prior[graph[5]]=1/9.
            self.graphs[j],self.regions[j]=graph,regions;node.joints[j]=prior

    def sample_world(self):
        ids=np.zeros(2,np.int64);modes=np.zeros(2,np.int64)
        for j in range(2):
            if not self.root.ae[j]:continue
            flat=self.root.joints[j].reshape(-1);cdf=np.cumsum(flat)
            value=self.world_rng.random()*cdf[-1];k=min(len(flat)-1,int(np.searchsorted(cdf,value,side='right')))
            ids[j],modes[j]=divmod(k,9)
        return ids,modes

    def child(self,node,chosen,positions,ap,ae,observed):
        key=(chosen,tuple(observed),tuple(ap),tuple(ae))
        if key in node.children:return node.children[key],False
        joints=[None,None];known=[None,None];reports=[]
        for j in range(2):
            if ae[j]:
                if node.predictions[j] is None:
                    g=self.graphs[j];node.predictions[j]=self.operation(predict_joint,node.joints[j],g[1],g[2],g[3])
                joints[j]=self.operation(condition_team,node.predictions[j],positions,ap,self.graphs[j][0],self.public.boxes,observed[j])
                if observed[j]>=0:known[j]=self.graphs[j][0][observed[j]].copy()
            reports.append([known[j].tolist() if known[j] is not None and ap[i]
                and sensor_visible(positions[i],known[j],self.public.boxes) else None for i in range(2)])
        obs=dict(step=80-self.remaining+node.depth+1,remaining=self.remaining-node.depth-1,
            positions=positions.tolist(),active_p=ap.tolist(),active_e=ae.tolist(),reports=reports)
        result=Node(positions,ap,ae,joints,known,node.depth+1,node.records+[obs])
        node.children[key]=result;self.nodes+=1
        return result,True

    def trajectory(self):
        ids,modes=self.sample_world();node=self.root;path=[];rewards=[];trace=[];self.active_trace=trace
        while node.depth<min(8,self.remaining):
            self.allowed();chosen=self.select(node);action=node.actions[chosen]
            before_targets=np.zeros((2,2));targets=np.zeros((2,2))
            for j,g in enumerate(self.graphs):
                if not node.ae[j]:continue
                before_targets[j]=g[0][ids[j]]
                ids[j],modes[j]=target_step(ids[j],modes[j],g[1],g[2],g[3],self.rng)
                targets[j]=g[0][ids[j]]
                if not safe(before_targets[j],targets[j]-before_targets[j],1.,self.public.boxes,np.zeros(2,np.int64)):
                    raise RuntimeError('Illegal simulated target transition')
            positions=node.positions+action
            if not all(safe(node.positions[i],action[i],self.public.speed,self.motion_boxes,np.zeros(2,np.int64)) for i in range(2)):
                raise RuntimeError('Illegal simulated joint action')
            ap,ae,visible=simultaneous_capture(positions,targets,node.ap,node.ae,self.public.boxes)
            observed=np.array([ids[j] if visible[j]>=0 else visible[j] for j in range(2)])
            trace.append(np.r_[node.positions.ravel(),positions.ravel(),before_targets.ravel(),targets.ravel(),node.ap,node.ae,ap,ae,visible>=0])
            path.append((node,chosen));raw=float(not ae.any())-TIME_COST
            terminal=not ae.any() or node.depth+1>=self.remaining
            previous_phi=self.potential(node)
            if terminal:
                rewards.append(raw-ETA*previous_phi);value=0.;break
            child,fresh=self.child(node,chosen,positions,ap,ae,observed)
            rewards.append(raw+ETA*(self.potential(child)-previous_phi))
            node=child
            if fresh or node.depth>=min(8,self.remaining):
                self.parameters(node);value=float(node.value) if self.runtime is not None and self.use_value else 0.;break
        self.allowed()
        for (visited,chosen),reward in zip(reversed(path),reversed(rewards)):
            value=reward+value;visited.n+=1;visited.visits[chosen]+=1;visited.totals[chosen]+=value
        self.traces.append(np.asarray(trace).reshape(-1,26))
        self.active_trace=None

    def act(self,obs,fixed_simulations=None):
        started=time.perf_counter();self.deadline=float('inf') if fixed_simulations is not None else started+self.budget_ms/1000.
        self.operation_ms={};self.rejections=self.fallbacks=0;self.nodes=1;self.traces=[];self.active_trace=None;self.remaining=obs['remaining']
        self.observe(obs)
        samples=np.zeros((2,64,2));weights=np.zeros((2,64))
        for j,e in enumerate(self.belief.estimators):
            if obs['active_e'][j]:samples[j]=e.particles;weights[j]=e.particle_weights
        self.root=Node(np.array(obs['positions']),obs['active_p'],obs['active_e'],[None,None],
            list(self.belief.known),0,list(self.records),samples,weights)
        self.features(self.root);action=np.zeros((2,2));completed=0
        try:
            if self.direct:action=self.propose(self.root,True)
            else:
                self.potential(self.root);self.prepare_models(self.root)
                for _ in range(100000 if fixed_simulations is None else fixed_simulations):
                    self.trajectory();completed+=1
                if completed:action=self.root.actions[int(np.argmax(self.root.visits))].copy()
        except BudgetAbort:
            if self.active_trace:self.traces.append(np.asarray(self.active_trace).reshape(-1,26))
            if completed:action=self.root.actions[int(np.argmax(self.root.visits))].copy()
            else:
                self.fallbacks+=1
                # No hidden-state or geometry-controller fallback. A stop is a
                # legal returned decision when no complete simulation fits.
        if not all(safe(self.root.positions[i],action[i],self.public.speed,self.public.boxes,np.zeros(2,np.int64)) for i in range(2)):
            raise RuntimeError('Illegal final joint action')
        self.last_stats=dict(total_ms=1000*(time.perf_counter()-started),simulations=completed,
            root_actions=[a.reshape(-1).tolist() for a in self.root.actions],root_visits=list(self.root.visits),
            root_totals=list(self.root.totals),nodes=self.nodes,rejections=self.rejections,fallbacks=self.fallbacks,
            joint_action_search=not self.direct,assignment_used=False,roles_used=False,
            root_potential=self.root.phi,planning_depth=min(8,self.remaining),
            simulated_transitions=sum(len(x) for x in self.traces))
        return action
