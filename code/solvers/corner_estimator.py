"""Isolated 2x2 intervention: map-following motion means / geometric export.

The reachable graph, observations, nine latent intents, sigma and persistence
are frozen from the previous estimator. No hidden state or route identity is
accepted. Fixed intents point from the last public measurement to arena-edge
waypoints; the map supplies their legal first step, including wall turns.
"""
import time
import numpy as np
from numba import njit

from solvers.multimodal_estimator import Estimator as Base, make_graph, propagate, export_modes, MODES, SIGMA
from solvers.local_pursuit import PublicMap, clear, edge, safe, legal, visible, uniform

VERSION="public_wall_turn_regions_v1"
IMPORTANT_MASS=.01


@njit(cache=True)
def map_probabilities(points,links,lengths,last,boxes,vertices,dist):
    count=np.zeros(2,np.int64)
    goals=np.empty((9,2));goals[0]=last
    tails=np.full((9,len(vertices)),1000000,np.int64)
    for m in range(1,9):
        goals[m]=np.clip(last+20.*MODES[m],-9.95,9.95)
        if not legal(goals[m],boxes,count):
            raise ValueError("Public fixed-intent goal is illegal")
        for j in range(len(vertices)):
            if clear(vertices[j],goals[m],boxes,count):
                tail=edge(vertices[j],goals[m],1.)
                for k in range(len(vertices)):
                    tails[m,k]=min(tails[m,k],dist[k,j]+tail)
    means=np.zeros((len(points),9,2))
    probabilities=np.zeros((len(points),49,9))
    for i in range(len(points)):
        if lengths[i]==0:continue
        head=np.full(len(vertices),1000000,np.int64)
        for j in range(len(vertices)):
            if clear(points[i],vertices[j],boxes,count):head[j]=edge(points[i],vertices[j],1.)
        for m in range(1,9):
            best,first=1000000,-1
            if clear(points[i],goals[m],boxes,count):best=edge(points[i],goals[m],1.)
            for j in range(len(vertices)):
                candidate=head[j]+tails[m,j]
                if candidate<best:best,first=candidate,j
            if best>=1000000:continue
            waypoint=vertices[first] if first>=0 else goals[m]
            delta=waypoint-points[i]
            action=delta/max(1.,np.max(np.abs(delta)))
            if safe(points[i],action,1.,boxes,count):means[i,m]=action
        for k in range(lengths[i]):
            delta=points[links[i,k]]-points[i]
            for m in range(9):
                probabilities[i,k,m]=np.exp(-.5*np.sum((delta-means[i,m])**2)/(SIGMA**2))
        for m in range(9):
            z=probabilities[i,:lengths[i],m].sum()
            if z>0:probabilities[i,:lengths[i],m]/=z
    return probabilities,means,int(count[0])


def geometry_labels(points,boxes):
    labels=np.empty((len(points),2*len(boxes)),np.int8)
    for j,b in enumerate(boxes):
        for axis in range(2):
            labels[:,2*j+axis]=np.where(points[:,axis]<b[axis,0],-1,np.where(points[:,axis]>b[axis,1],1,0))
    if not len(boxes):return np.zeros(len(points),np.int64)
    _,inverse=np.unique(labels,axis=0,return_inverse=True)
    return inverse.astype(np.int64)


@njit(cache=True)
def export_regions(weights,regions,rng,number=64):
    marginal=weights.sum(axis=1)
    count=int(regions.max())+1
    mass=np.zeros(count)
    for i in range(len(regions)):mass[regions[i]]+=marginal[i]
    # Every >=1% region has its own stratum. Smaller regions share one tail
    # stratum, whose total probability is preserved rather than discarded.
    group=np.full(count,-1,np.int64);n=0
    for r in range(count):
        if mass[r]>=IMPORTANT_MASS:group[r]=n;n+=1
    tail=n
    for r in range(count):
        if group[r]<0:group[r]=tail
    total=np.zeros(n+1)
    for r in range(count):total[group[r]]+=mass[r]
    allocation=np.zeros(n+1,np.int64)
    for g in range(n+1):
        if total[g]>0:allocation[g]=1
    if allocation.sum()>number:raise ValueError("More important regions than the frozen sample budget")
    for _ in range(number-allocation.sum()):allocation[np.argmax(number*total-allocation)]+=1
    ids=np.empty(number,np.int64);sample_weights=np.empty(number);labels=np.empty(number,np.int64)
    cursor=0
    for g in range(n+1):
        if allocation[g]==0:continue
        local=np.zeros(len(marginal))
        for i in range(len(local)):
            if group[regions[i]]==g:local[i]=marginal[i]
        cumulative=np.cumsum(local);offset=uniform(rng)
        for k in range(allocation[g]):
            value=(k+offset)*total[g]/allocation[g]
            i=min(len(local)-1,np.searchsorted(cumulative,value,side="right"))
            ids[cursor]=i;sample_weights[cursor]=total[g]/allocation[g];labels[cursor]=regions[i];cursor+=1
    return ids,sample_weights,labels


class Estimator(Base):
    def __init__(self,boxes,method="D",seed=0,particles=64):
        if method not in ("A","B","C","D","U0","U1"):
            raise ValueError(method)
        super().__init__(boxes,"feasible" if method.startswith("U") else "persistent",seed,particles)
        self.arm=method
        self.wall_turn=method in ("C","D")
        self.region_export=method in ("B","D","U1")
        self.regions=None
        self.geometry_means=None

    def update(self,obs):
        started=time.perf_counter()
        if self.step>=0 and obs.step!=self.step+1:raise ValueError("Nonconsecutive observation")
        if obs.remaining!=80-obs.step or not np.all(np.isfinite(obs.position)):raise ValueError("Bad public clock/position")
        count=np.zeros(2,np.int64)
        if not legal(obs.position,self.boxes,count):raise ValueError("Illegal observer")
        build_ms=0.;geometry=0
        if obs.target is not None:
            target=np.asarray(obs.target,float)
            if not legal(target,self.boxes,count) or not visible(obs.position,target,self.boxes,count):raise ValueError("Illegal measurement")
            self.last=target.copy();self.graph=None;self.parents=[];self.observers=[]
            self.particles=np.tile(target,(64,1));self.particle_weights=np.ones(64)/64
            self.weights=None;self.reachable=None;self.regions=None;self.geometry_means=None
        else:
            if self.last is None:raise ValueError("Positive initialization required")
            if self.graph is None:
                stamp=time.perf_counter()
                graph=list(make_graph(self.last,self.boxes))
                points,links,lengths,probabilities,sensor,root,queries=graph
                geometry+=queries
                if self.wall_turn:
                    public=PublicMap(self.boxes,1.)
                    probabilities,self.geometry_means,queries=map_probabilities(points,links,lengths,self.last,self.boxes,public.vertices,public.dist)
                    graph[3]=probabilities;geometry+=queries
                self.graph=tuple(graph)
                self.regions=geometry_labels(points,self.boxes)
                self.weights=np.zeros((len(points),9))
                self.weights[root]=np.array([1.,0.,0.,0.,0.,0.,0.,0.,0.]) if self.method=="feasible" else np.ones(9)/9
                self.reachable=np.zeros(len(points),bool);self.reachable[root]=True
                build_ms=1000*(time.perf_counter()-stamp)
            points,links,lengths,probabilities,sensor,root,queries=self.graph
            self.weights,self.reachable,parent,queries=propagate(self.weights,self.reachable,obs.position,
                points,links,lengths,probabilities,sensor,2 if self.method=="feasible" else 1)
            geometry+=queries;self.parents.append(parent);self.observers.append(obs.position.copy())
            exporter=export_regions if self.region_export else export_modes
            if self.region_export:self.ids,self.particle_weights,self.labels=exporter(self.weights,self.regions,self.rng,64)
            else:self.ids,self.particle_weights,self.labels=exporter(self.weights,self.rng,64)
            self.particles=points[self.ids].copy()
        self.step=obs.step
        self.last_stats=dict(update_ms=1000*(time.perf_counter()-started),build_ms=build_ms,
            geometry_queries=geometry,support_points=int(self.reachable.sum()) if self.graph else 1,
            mode_mass=self.weights.sum(axis=0).tolist() if self.graph else [1.]+[0.]*8)
        return self.particles.copy(),self.particle_weights.copy()
