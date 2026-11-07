"""Observation-indexed neural PUCT on public grid support.

The privileged expert is a separate offline entry. Real Controller.act accepts
only Observation. Rollout/opponent/reward kernels match frozen local_pursuit.
"""
import copy
import math
import time
import numpy as np
from numba import njit
from solvers.local_pursuit import (uniform, safe, visible, move, modal, proposal,
    predict, recover, Observation)
from solvers.branch_support import Support, SupportFailure, cell_id


@njit(cache=True)
def world_step(at,target,action,speed,boxes,rng,count,absolute_step):
    away=target-at
    away/=max(1.,np.max(np.abs(away)))
    other,_=proposal(target,away,1.,boxes,rng,count)
    count[1]+=1
    nxt,end=at+action,target+other
    if not safe(at,action,speed,boxes,count): return nxt,end,-1.,True,False
    if not safe(target,other,1.,boxes,count): return nxt,end,0.,True,False
    d0=np.linalg.norm(target-at);d1=np.linalg.norm(end-nxt)
    if d1<1.: return nxt,end,.5*(1.-absolute_step/80.),True,False
    reward=.03*min(1.,max(-1.,(d0-d1)/4.))
    return nxt,end,reward,False,visible(nxt,end,boxes,count)


@njit(cache=True)
def rollout(at,target,particles,last,observers,speed,remaining,absolute_step,vertices,dist,boxes,rng,count):
    # The old local T rollout, including its auxiliary particle cutoff. This is
    # shared by every new search arm; it is not the real controller estimator.
    total=0.; pool=particles.copy();old=last.copy()
    hist=np.empty((len(observers)+remaining+1,2)); hist[:len(observers)]=observers;hn=len(observers)
    for d in range(min(100,remaining)):
        base=move(at,modal(pool),speed,vertices,dist,boxes,count)
        action,_=proposal(at,base,speed,boxes,rng,count)
        nxt,end,reward,done,seen=world_step(at,target,action,speed,boxes,rng,count,absolute_step+d+1)
        total+=reward
        if done: break
        at,target=nxt,end
        if seen:
            pool[:]=target;old=target.copy();hn=0
        else:
            hist[hn]=at;hn+=1
            if not predict(pool,at,boxes,rng,count):
                fresh,n=recover(old,hist[:hn],boxes,rng,count)
                if not n: break
                pool[:]=fresh
    return total


@njit(cache=True)
def normal(rng):
    return math.sqrt(-2.*math.log(uniform(rng)))*math.cos(2.*math.pi*uniform(rng))


def candidate_bank(public,at,features,means,logvars,weights,rng,geometry=False,uniform_prior=False):
    count=np.zeros(2,np.int64);raw=[];mass=[];sources=[];invalid=0
    valid=np.flatnonzero(features[:,11]>.5)
    if not len(valid): raise SupportFailure('No public approach branch')
    allocation=np.zeros(4,np.int64)
    for k in range(8): allocation[valid[k%len(valid)]]+=1
    for k in valid:
        for _ in range(allocation[k]):
            if geometry:
                action=means[k].copy()+np.array([normal(rng),normal(rng)])*.1*public.speed
            else:
                action=(means[k]+np.exp(logvars[k]*.5)*np.array([normal(rng),normal(rng)]))*public.speed
            action=np.clip(action,-public.speed,public.speed)
            if not safe(at,action,public.speed,public.boxes,count): action=np.zeros(2);invalid+=1
            raw.append(action);mass.append(.5*weights[k]/allocation[k]);sources.append(int(k))
    for _ in range(8):
        action=np.array([2*uniform(rng)-1,2*uniform(rng)-1])*public.speed
        if not safe(at,action,public.speed,public.boxes,count): action=np.zeros(2);invalid+=1
        raw.append(action);mass.append(.5/8);sources.append(4)
    actions=[];priors=[];labels=[]
    for a,w,s in zip(raw,mass,sources):
        match=next((j for j,b in enumerate(actions) if np.array_equal(a,b)),None)
        if match is None: actions.append(a);priors.append(1/16 if uniform_prior else w);labels.append([s])
        else: priors[match]+=1/16 if uniform_prior else w;labels[match].append(s)
    # T and H-U require uniformity over the actual deduplicated candidates.
    # Uniform raw-slot mass would give merged zero actions a larger prior.
    priors=np.ones(len(actions)) if uniform_prior else np.asarray(priors)
    priors/=priors.sum()
    return np.asarray(actions),priors,labels,invalid,int(count[0]),allocation


class Node:
    def __init__(self,support,records,privileged=False):
        self.support=support;self.records=records;self.actions=None;self.privileged=privileged
        self.visits=0;self.action_visits=None;self.values=None;self.children={}
        self.index=0


class Search:
    def __init__(self,controller,obs,privileged_target=None):
        self.c=controller; self.public=controller.public;self.count=np.zeros(2,np.int64)
        self.network_calls=0;self.invalid=0;self.nodes=0;self.support_queries=0
        self.root=Node(controller.support.clone(),copy.deepcopy(controller.records),privileged_target is not None)
        if privileged_target is not None:
            self.root.support.exact=np.asarray(privileged_target,float).copy()
            self.root.support.mask[:]=False;self.root.support.mask[cell_id(privileged_target)]=True
        self.particles,queries=self.root.support.sample(controller.rng)
        self.count[0]+=queries
        self.root_observation=obs;self.completed=0
        self.root_branches=None
        self.node_list=[self.root]
        self.latest_delta=None

    def build(self,node):
        c=self.c;p=self.public
        features,assignment,summary,queries=node.support.branches();self.count[0]+=queries
        if node is self.root: self.root_branches=(features,assignment,summary)
        geometry=c.method in ('T','E')
        if geometry:
            mu=np.zeros((4,2));lv=np.full((4,2),-8.);weights=np.zeros(4)
            valid=np.flatnonzero(features[:,11]>.5)
            for k in valid:
                ids=np.flatnonzero(assignment==k)
                q=node.support.exact if np.isfinite(node.support.exact[0]) else p.points[ids[len(ids)//2]]
                mu[k]=move(node.support.at,q,p.speed,p.vertices,p.dist,p.boxes,self.count)
                weights[k]=1/len(valid)
        else:
            weights,mu,lv=c.policy(node.records,features)
            self.network_calls+=1
            if c.method=='M':
                # M has four free mixture components, regardless of the current
                # number of geometric branches. Region information stays input.
                features=features.copy();features[:,11]=1.
        node.actions,node.priors,node.sources,invalid,queries,node.allocation=candidate_bank(
            p,node.support.at,features,mu,lv,weights,c.rng,geometry,c.method in ('T','E','H-U'))
        self.invalid+=invalid;self.count[0]+=queries
        node.action_visits=np.zeros(len(node.actions),np.int64);node.values=np.zeros(len(node.actions));self.nodes+=1

    def one(self):
        c=self.c;p=self.public;rng=c.rng;node=self.root
        at=node.support.at.copy();target=self.particles[min(63,int(uniform(rng)*64))].copy()
        particles=self.particles.copy()
        last=c.last.copy();observers=np.array(c.hidden_observers,float).reshape(-1,2)
        path=[];rewards=[];remaining=self.root_observation.remaining
        built=[];created=None
        for depth in range(min(100,remaining)):
            if node.actions is None:
                self.build(node)
                built.append(dict(index=node.index,actions=node.actions,priors=node.priors,sources=node.sources,allocation=node.allocation))
            q=np.divide(node.values,node.action_visits,out=np.zeros_like(node.values),where=node.action_visits>0)
            scores=q+1.5*node.priors*math.sqrt(node.visits)/(1+node.action_visits)
            chosen=int(np.argmax(scores));path.append((node,chosen))
            nxt,end,reward,done,seen=world_step(at,target,node.actions[chosen],p.speed,p.boxes,rng,self.count,self.root_observation.step+depth+1)
            rewards.append(float(reward))
            if done or depth+1>=min(100,remaining): break
            # Simulated sensor output is quantized; hidden coordinates never
            # enter a local node's observation history.
            perceived=seen or self.root.privileged
            key=cell_id(end) if perceived else -1
            if perceived:
                measured=p.points[key].copy()
                particles[:]=measured;last=measured.copy();observers=np.empty((0,2))
            else:
                measured=None;observers=np.vstack((observers,nxt))
                if not predict(particles,nxt,p.boxes,rng,self.count):
                    fresh,n=recover(last,observers,p.boxes,rng,self.count)
                    if not n: break
                    particles=fresh
            child_key=(chosen,int(key))
            child=node.children.get(child_key)
            if child is None:
                belief=node.support.clone()
                simobs=Observation(self.root_observation.step+depth+1,nxt,measured,80-self.root_observation.step-depth-1)
                belief.update(simobs)
                # A quantized positive observation denotes the whole cell.
                if perceived: belief.exact=np.full(2,np.nan)
                records=node.records+[dict(step=simobs.step,position=nxt.tolist(),target=None if measured is None else measured.tolist(),remaining=simobs.remaining)]
                child=Node(belief,records,self.root.privileged);child.index=len(self.node_list);self.node_list.append(child);node.children[child_key]=child
                created=dict(parent=node.index,key=child_key,mask=np.packbits(belief.mask),exact=belief.exact,at=belief.at,step=belief.step,observation=records[-1])
                tail=rollout(nxt,end,particles,last,observers,p.speed,remaining-depth-1,self.root_observation.step+depth+1,p.vertices,p.dist,p.boxes,rng,self.count)
                rewards.append(float(tail));break
            node=child;at,target=nxt,end
        total=float(sum(rewards))
        updated=[]
        for j,(parent,chosen) in enumerate(path):
            parent.visits+=1;parent.action_visits[chosen]+=1;parent.values[chosen]+=total
            updated.append((parent.index,chosen,parent.visits,int(parent.action_visits[chosen]),float(parent.values[chosen])))
            total-=rewards[j]
        self.completed+=1
        self.latest_delta=dict(completed=self.completed,built=built,created=created,updated=updated,rng=c.rng.copy(),
            count=self.count.copy(),network_calls=self.network_calls,invalid=self.invalid,nodes=self.nodes)

    def apply_delta(self,delta):
        if delta['completed']!=self.completed+1:raise RuntimeError('Teacher simulation journal gap')
        for b in delta['built']:
            node=self.node_list[b['index']]
            node.actions=b['actions'];node.priors=b['priors'];node.sources=b['sources'];node.allocation=b['allocation']
            node.action_visits=np.zeros(len(node.actions),np.int64);node.values=np.zeros(len(node.actions))
        item=delta['created']
        if item is not None:
            parent=self.node_list[item['parent']];support=parent.support.clone()
            support.mask=np.unpackbits(item['mask'])[:6400].astype(bool)
            for key in ('exact','at','step'):setattr(support,key,item[key])
            child=Node(support,parent.records+[item['observation']],self.root.privileged)
            child.index=len(self.node_list);self.node_list.append(child);parent.children[item['key']]=child
        for index,chosen,visits,n,value in delta['updated']:
            node=self.node_list[index];node.visits=visits;node.action_visits[chosen]=n;node.values[chosen]=value
        self.completed=delta['completed'];self.c.rng=delta['rng'].copy();self.count=delta['count'].copy()
        self.network_calls=delta['network_calls'];self.invalid=delta['invalid'];self.nodes=delta['nodes']

    def run(self,budget,deadline=None,progress=None,start=0):
        for _ in range(start,budget):
            if deadline is not None and time.perf_counter()>=deadline: break
            self.one()
            if progress is not None: progress(self)
        if self.root.actions is None: self.build(self.root)
        root=self.root
        means=np.divide(root.values,root.action_visits,out=np.zeros_like(root.values),where=root.action_visits>0)
        best=max(range(len(root.actions)),key=lambda i:(int(root.action_visits[i]),float(means[i]),-i))
        rows=[dict(action=a.tolist(),prior=float(root.priors[i]),visits=int(root.action_visits[i]),value=float(means[i]),sources=root.sources[i]) for i,a in enumerate(root.actions)]
        return root.actions[best].copy(),rows


class Controller:
    def __init__(self,public,method,seed,policy=None,simulations=1000,deadline_ms=50.):
        if method not in ('G','T','M','H','H-U','H-D'): raise ValueError(method)
        self.public=public;self.method=method;self.rng=np.array([int(seed)%4294967296],np.int64)
        self.filter_rng=np.array([0],np.int64)  # Deterministic grid; no filter draws.
        self.support=Support(public);self.records=[];self.hidden_observers=[];self.last=None
        self.policy=policy;self.simulations=simulations;self.deadline_ms=deadline_ms
        self.last_stats={};self.particles=np.zeros((64,2));self.recoveries=0

    def update(self,observation,count=None):
        self.support.update(observation)
        self.records.append(dict(step=observation.step,position=observation.position.tolist(),target=None if observation.target is None else observation.target.tolist(),remaining=observation.remaining))
        if observation.target is not None: self.last=observation.target.copy();self.hidden_observers=[]
        else: self.hidden_observers.append(observation.position.copy())

    def snapshot(self):
        return copy.deepcopy({k:v for k,v in self.__dict__.items() if k not in ('public','policy','support')}) | dict(support=dict(mask=self.support.mask.copy(),exact=self.support.exact.copy(),at=self.support.at.copy(),step=self.support.step))

    def restore(self,saved):
        saved=copy.deepcopy(saved);support=saved.pop('support');self.__dict__.update(saved)
        self.support=Support(self.public)
        for k,v in support.items():setattr(self.support,k,v)

    def act(self,obs,fixed_simulations=None):
        started=time.perf_counter();self.update(obs);updated=time.perf_counter()
        count=np.zeros(2,np.int64);search=None;network_calls=0
        if self.method=='G':
            target=self.support.representative()
            action=move(obs.position,target,self.public.speed,self.public.vertices,self.public.dist,self.public.boxes,count)
            root=[];completed=0;self.particles[:]=target
        else:
            search=Search(self,obs)
            if self.method=='H-D':
                features,_,_,queries=self.support.branches();count[0]+=queries
                weights,means,_=self.policy(self.records,features);network_calls+=1
                top=int(np.argmax(weights));action=np.clip(means[top],-1,1)*self.public.speed
                if not safe(obs.position,action,self.public.speed,self.public.boxes,count):
                    search.build(search.root);action=np.zeros(2)
                    ordered=sorted(range(len(search.root.actions)),key=lambda i:(top not in search.root.sources[i],i))
                    for i in ordered:
                        candidate=search.root.actions[i]
                        if safe(obs.position,candidate,self.public.speed,self.public.boxes,count):action=candidate.copy();break
                root=[];completed=0
            else:
                cap=self.simulations if fixed_simulations is None else fixed_simulations
                deadline=started+self.deadline_ms/1000 if fixed_simulations is None and self.deadline_ms>0 else None
                action,root=search.run(cap,deadline);completed=search.completed
            self.particles=search.particles.copy();count+=search.count;network_calls+=search.network_calls
        latency=(time.perf_counter()-started)*1000
        if not safe(obs.position,action,self.public.speed,self.public.boxes,count):raise RuntimeError('Illegal branch controller action')
        self.last_stats=dict(simulations=completed,latency_ms=latency,support_update_ms=(updated-started)*1000,
            geometry_queries=int(count[0]),transitions=int(count[1]),network_calls=network_calls,
            support_cells=int(self.support.mask.sum()),source=self.method,root=root,recoveries=0,
            invalid_candidates=0 if search is None else search.invalid,overrun=bool(self.deadline_ms>0 and latency>self.deadline_ms))
        return action


def expert_search(public,history,target,seed,budget=10000,progress=None,resume=None):
    """Privileged offline service. Never called by a deployment controller."""
    own=Controller(public,'T',seed,deadline_ms=0.)
    for r in history: own.update(Observation(r['step'],np.asarray(r['position'],float),None if r['target'] is None else np.asarray(r['target'],float),r['remaining']))
    obs=Observation(history[-1]['step'],own.support.at.copy(),None if history[-1]['target'] is None else np.array(history[-1]['target']),history[-1]['remaining'])
    search=Search(own,obs,np.asarray(target,float));own.method='E'
    if resume is not None:
        search.__dict__.update(resume['search']);search.c=own;search.public=public
        own.rng=resume['rng'].copy()
    action,root=search.run(budget,progress=progress,start=search.completed)
    return action,root,search
