from __future__ import annotations
from dataclasses import dataclass, asdict
from functools import lru_cache
import hashlib
import json
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

@dataclass(frozen=True)
class Config:
    n: int
    size: float
    horizon: int
    speed: float = 1.
    sensing: float = 6.
    capture: float = 1.
    announce: bool = True

    @classmethod
    def scale(cls, n):
        if n not in (5, 16): raise ValueError(n)
        return cls(n, 40. if n == 5 else 80., 160 if n == 5 else 320)

    def __post_init__(self):
        if self.n < 1 or self.size <= 0 or self.horizon < 1: raise ValueError('Invalid task')

@njit(cache=True)
def clear(a, b, boxes):
    for box in boxes:
        lo, hi = 0., 1.
        hit = True
        for k in range(2):
            delta = b[k]-a[k]
            if abs(delta) < 1e-12:
                if a[k] < box[k,0] or a[k] > box[k,1]: hit = False
            else:
                x, y = (box[k,0]-a[k])/delta, (box[k,1]-a[k])/delta
                lo, hi = max(lo, min(x,y)), min(hi, max(x,y))
        if hit and lo <= hi+1e-12: return False
    return True

@njit(cache=True)
def legal(q, boxes, size):
    if np.any(q < -size/2) or np.any(q > size/2): return False
    for b in boxes:
        if b[0,0] <= q[0] <= b[0,1] and b[1,0] <= q[1] <= b[1,1]: return False
    return True

@njit(cache=True)
def move(points, actions, active, boxes, size):
    """Continuous swept wall contact; point robots do not collide with each other."""
    result = points.copy()
    for i in range(len(points)):
        if not active[i]: continue
        a = points[i]; delta = actions[i]; fraction = 1.
        for k in range(2):
            if delta[k] > 0: fraction = min(fraction, (size/2-a[k])/delta[k])
            elif delta[k] < 0: fraction = min(fraction, (-size/2-a[k])/delta[k])
        for box in boxes:
            lo, hi = 0., 1.; hit = True
            for k in range(2):
                if abs(delta[k]) < 1e-12:
                    if a[k] < box[k,0] or a[k] > box[k,1]: hit = False
                else:
                    x,y = (box[k,0]-a[k])/delta[k], (box[k,1]-a[k])/delta[k]
                    lo,hi = max(lo,min(x,y)),min(hi,max(x,y))
            if hit and lo <= hi+1e-12: fraction = min(fraction,max(0.,lo-1e-7))
        result[i] = a+max(0.,fraction)*delta
    return result

@njit(cache=True)
def captures(p, e, alive, boxes, radius):
    caught = np.zeros(len(e),np.bool_)
    for j in range(len(e)):
        if alive[j]:
            for i in range(len(p)):
                if np.sum((p[i]-e[j])**2) < radius**2 and clear(p[i],e[j],boxes):
                    caught[j] = True; break
    return caught

@njit(cache=True)
def visibility(p, e, boxes, radius):
    seen = np.zeros(len(e),np.bool_)
    for j in range(len(e)):
        for i in range(len(p)):
            if np.sum((p[i]-e[j])**2) <= radius**2 and clear(p[i],e[j],boxes):
                seen[j] = True; break
    return seen

@dataclass
class State:
    p: np.ndarray
    e: np.ndarray
    alive: np.ndarray
    t: int = 0

    def copy(self): return State(self.p.copy(),self.e.copy(),self.alive.copy(),self.t)

class World:
    def __init__(self, cfg, boxes):
        self.cfg = cfg
        self.boxes = np.asarray(boxes,float).reshape(-1,2,2)

    def transition(self, state, pa, ea):
        c = self.cfg
        for action in (pa, ea):
            if np.shape(action) != (c.n,2) or not np.isfinite(action).all() or np.max(np.abs(action)) > c.speed+1e-8:
                raise ValueError('Nonfinite, malformed or over-speed team action')
        p = move(state.p,pa,np.ones(c.n,np.bool_),self.boxes,c.size)
        e = move(state.e,ea,state.alive,self.boxes,c.size)
        alive = state.alive & ~captures(p,e,state.alive,self.boxes,c.capture)
        return State(p,e,alive,state.t+1)

    def done(self, s): return s.t >= self.cfg.horizon or not s.alive.any()

    def observe(self, s):
        seen = visibility(s.p,s.e,self.boxes,self.cfg.sensing) & s.alive
        if s.t == 0 and self.cfg.announce: seen = s.alive.copy()
        # No hidden target coordinates cross this interface.
        return dict(p=s.p.copy(),e=np.where(seen[:,None],s.e,np.nan),
                    seen=seen,alive=s.alive.copy(),t=s.t)

class PublicMap:
    """One shared exact-edge grid per geometry; bounded LRU route cache per goal."""
    def __init__(self, cfg, boxes):
        self.cfg,self.boxes = cfg,np.asarray(boxes,float).reshape(-1,2,2)
        axis = np.arange(-cfg.size/2+.5,cfg.size/2,1.)
        self.width = len(axis)
        self.points = np.stack(np.meshgrid(axis,axis,indexing='ij'),axis=-1).reshape(-1,2)
        self.free = np.array([legal(q,self.boxes,cfg.size) for q in self.points])
        rows,cols,data = [],[],[]
        for i in np.flatnonzero(self.free):
            x,y = divmod(i,self.width)
            for dx,dy in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
                xx,yy = x+dx,y+dy
                if 0 <= xx < self.width and 0 <= yy < self.width:
                    j = xx*self.width+yy
                    if self.free[j] and clear(self.points[i],self.points[j],self.boxes):
                        rows.append(i);cols.append(j);data.append(1.)
        self.graph = csr_matrix((data,(rows,cols)),shape=(len(self.points),)*2)
        _,self.components = connected_components(self.graph,directed=False)
        self.free_indices = np.flatnonzero(self.free)
        self._distance = lru_cache(maxsize=128)(self._compute_distance)

    def nearest(self, q):
        index = np.clip(np.rint(q+self.cfg.size/2-.5).astype(int),0,self.width-1)
        root = index[0]*self.width+index[1]
        if self.free[root] and clear(q,self.points[root],self.boxes): return int(root)
        order = np.argsort(np.sum((self.points[self.free_indices]-q)**2,axis=1))
        for z in order:
            root = self.free_indices[z]
            if clear(q,self.points[root],self.boxes): return int(root)
        if not legal(q,self.boxes,self.cfg.size):raise ValueError('Illegal position in public routing')
        # A narrow continuous passage may contain no visible 1m grid center.
        # Grid distance remains a shared approximation; executed routing below
        # switches to the exact public corner graph for these attachments.
        return int(self.free_indices[order[0]])

    def _compute_distance(self, root): return dijkstra(self.graph,directed=False,indices=root)

    def field(self, goal): return self._distance(self.nearest(goal))

    def action(self,at,goal):
        return self.kernel_mean(at,goal)

    def corner_action(self,at,goal):
        """Exact swept-visible waypoints for continuous pockets below grid resolution."""
        if not legal(at,self.boxes,self.cfg.size) or not legal(goal,self.boxes,self.cfg.size):
            raise ValueError('Illegal endpoint in public corner routing')
        if clear(at,goal,self.boxes):
            delta=goal-at;return delta/max(1.,np.max(np.abs(delta)))
        if not hasattr(self,'_corner_vertices'):
            vertices=[]
            for box in self.boxes:
                for x in (box[0,0]-1e-5,box[0,1]+1e-5):
                    for y in (box[1,0]-1e-5,box[1,1]+1e-5):
                        q=np.array([x,y])
                        if legal(q,self.boxes,self.cfg.size):vertices.append(q)
            self._corner_vertices=np.unique(np.asarray(vertices).reshape(-1,2),axis=0)
            weights=corner_edges(self._corner_vertices,self.boxes)
            self._corner_dist=dijkstra(csr_matrix(weights),directed=False)
        vertices=self._corner_vertices
        head=np.array([np.max(np.abs(v-at)) if np.max(np.abs(v-at))>1e-7 and clear(at,v,self.boxes) else np.inf for v in vertices])
        tail=np.array([np.max(np.abs(v-goal)) if clear(v,goal,self.boxes) else np.inf for v in vertices])
        costs=head+np.min(self._corner_dist+tail[None,:],axis=1)
        if not np.isfinite(costs).any():raise ValueError('Disconnected continuous public route')
        waypoint=vertices[int(costs.argmin())]
        delta=waypoint-at
        return delta/max(1.,np.max(np.abs(delta)))

    def _ensure_route_graph(self):
        if hasattr(self,'_route_vertices'):return
        h=self.cfg.size/2-.05;vertices=[]
        for box in self.boxes:
            for i in range(2):
                for j in range(2):vertices.append([box[0,i]+(.05 if i else -.05),box[1,j]+(.05 if j else -.05)])
        vertices.extend([[-h,-h],[-h,h],[h,-h],[h,h],[0.,-h],[0.,h],[-h,0.],[h,0.]])
        self._route_vertices=np.array([q for q in np.asarray(vertices) if legal(q,self.boxes,self.cfg.size)])
        weights=corner_edges(self._route_vertices,self.boxes)
        weights=np.where(weights>0,np.ceil(weights-1e-9),0.)
        self._route_dist=dijkstra(csr_matrix(weights),directed=False)
        self._route_heads=lru_cache(maxsize=8192)(lambda key:route_heads(np.asarray(key),self._route_vertices,self.boxes))
        self._vertex_ids={tuple(v):i for i,v in enumerate(self._route_vertices)}
        self._route_tail=lru_cache(maxsize=2048)(lambda key:self._route_dist[self._vertex_ids[key]] if key in self._vertex_ids else np.min(self._route_dist+self._route_heads(key)[None,:],axis=1))

    def kernel_mean(self,at,goal):
        self._ensure_route_graph()
        direct=clear(at,goal,self.boxes)
        best=np.ceil(np.max(np.abs(goal-at))-1e-9) if direct else np.inf
        costs=self._route_heads(tuple(at))+self._route_tail(tuple(goal))
        winner=int(costs.argmin())
        if costs[winner]<best:waypoint=self._route_vertices[winner]
        elif direct:waypoint=goal
        elif np.isfinite(costs[winner]):waypoint=self._route_vertices[winner]
        else:return self.corner_action(at,goal)
        delta=waypoint-at;action=delta/max(1.,np.max(np.abs(delta)))
        if not legal(at+action,self.boxes,self.cfg.size) or not clear(at,at+action,self.boxes):return np.zeros(2)
        return action

    def actions_to(self,at,goals):
        self._ensure_route_graph();goals=np.asarray(goals)
        tails=np.array([self._route_tail(tuple(g)) for g in goals])
        costs=tails+self._route_heads(tuple(at))[None,:];indices=costs.argmin(1)
        actions=[]
        for j,goal in enumerate(goals):
            direct=clear(at,goal,self.boxes)
            if direct and np.ceil(np.max(np.abs(goal-at))-1e-9)<=costs[j,indices[j]]:waypoint=goal
            elif np.isfinite(costs[j,indices[j]]):waypoint=self._route_vertices[indices[j]]
            else:
                actions.append(self.corner_action(at,goal));continue
            delta=waypoint-at;actions.append(delta/max(1.,np.max(np.abs(delta))))
        return np.asarray(actions)

    def distance_matrix(self,positions,goals):
        self._ensure_route_graph();positions=np.asarray(positions);goals=np.asarray(goals)
        heads=np.array([self._route_heads(tuple(q)) for q in positions]);cost=np.empty((len(positions),len(goals)))
        for j,g in enumerate(goals):cost[:,j]=np.min(heads+self._route_tail(tuple(g))[None,:],axis=1)
        cost=improve_direct_costs(positions,goals,cost,self.boxes)
        for i,j in zip(*np.nonzero(~np.isfinite(cost))):cost[i,j]=self.corner_route(positions[i],goals[j])[1]
        return cost

    def coverage(self,future,goals):
        self._ensure_route_graph();count,agents,_=future.shape
        heads=np.array([self._route_heads(tuple(q)) for q in future.reshape(-1,2)]).reshape(count,agents,-1).min(1)
        cost=np.empty((count,len(goals)))
        for j,g in enumerate(goals):cost[:,j]=np.min(heads+self._route_tail(tuple(g))[None,:],axis=1)
        cost=improve_direct_coverage(future,goals,cost,self.boxes)
        for i,j in zip(*np.nonzero(~np.isfinite(cost))):cost[i,j]=min(self.corner_route(q,goals[j])[1] for q in future[i])
        return cost

    def corner_route(self,at,goal):
        if clear(at,goal,self.boxes):return goal.copy(),float(np.max(np.abs(goal-at)))
        # corner_action constructs and caches the shared visibility graph.
        self.corner_action(at,goal)
        v=self._corner_vertices
        head=np.array([np.max(np.abs(q-at)) if clear(at,q,self.boxes) else np.inf for q in v])
        tail=np.array([np.max(np.abs(q-goal)) if clear(q,goal,self.boxes) else np.inf for q in v])
        costs=head+np.min(self._corner_dist+tail[None,:],axis=1)
        winner=int(costs.argmin())
        if not np.isfinite(costs[winner]):raise ValueError('Disconnected certified continuous route')
        return v[winner].copy(),float(costs[winner])

    def sample_free(self, rng, count):
        result=[]
        while len(result)<count:
            q=rng.uniform(-self.cfg.size/2+.1,self.cfg.size/2-.1,2)
            if legal(q,self.boxes,self.cfg.size): result.append(q)
        return np.array(result)

class Belief:
    """Public-history particle posterior, shared verbatim by real and tree updates."""
    def __init__(self,public,seed=0,particles=48):
        self.public=public;self.rng=np.random.default_rng(seed);self.count=particles
        self.samples=None;self.modes=None;self.anchors=None;self.t=-1
        self.history=[];self.last=None;self.recoveries=0;self.last_seen=np.zeros(public.cfg.n,int)

    def fork(self,seed):
        clone=Belief(self.public,seed,self.count)
        for name in ('samples','modes','anchors','last_seen'):
            value=getattr(self,name);setattr(clone,name,None if value is None else value.copy())
        clone.t=self.t;clone.history=list(self.history);clone.last=self.last;clone.recoveries=self.recoveries
        return clone

    def predict(self):
        if self.samples is None:raise ValueError('Uninitialized public posterior')
        from .motion import planning_model
        return planning_model(self.public).advance(self.samples,self.modes,self.anchors[:,None,:],self.rng,self.last['alive'][:,None])

    def condition(self,obs,predicted,predicted_modes,record=True):
        if obs['t']!=self.t+1:raise ValueError('Nonconsecutive public observation')
        if record:self.history.append({k:v.copy() if isinstance(v,np.ndarray) else v for k,v in obs.items()})
        for j in range(self.public.cfg.n):
            if not obs['alive'][j]:continue
            if obs['seen'][j]:
                # Existing D reset at an exact positive report: public re-anchor
                # with uniform latent intent prior, without sensor-noise additions.
                self.modes[j]=self.rng.integers(9,size=self.count)
                self.samples[j]=obs['e'][j];self.anchors[j]=obs['e'][j];self.last_seen[j]=obs['t']
            else:
                keep=~visibility(obs['p'],predicted[j],self.public.boxes,self.public.cfg.sensing)
                if not keep.any():
                    samples,modes=self._replay(j)
                    indices=self.rng.integers(len(samples),size=self.count)
                    self.samples[j]=samples[indices];self.modes[j]=modes[indices];self.recoveries+=1
                else:
                    indices=self.rng.choice(np.flatnonzero(keep),size=self.count)
                    self.samples[j]=predicted[j,indices];self.modes[j]=predicted_modes[j,indices]
        self.t=obs['t'];self.last={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in obs.items()}

    def update(self,obs):
        if obs['t']==self.t:return
        if self.samples is None:
            if obs['t']!=0 or not np.all(obs['seen'][obs['alive']]):raise ValueError('Initial public announcement required')
            self.samples=np.repeat(np.nan_to_num(obs['e'])[:,None,:],self.count,axis=1)
            self.anchors=np.nan_to_num(obs['e']).copy();self.modes=self.rng.integers(9,size=(self.public.cfg.n,self.count))
            self.t=0;self.last_seen[:]=0;self.last=obs
            self.history=[{k:v.copy() if isinstance(v,np.ndarray) else v for k,v in obs.items()}]
        else:self.condition(obs,*self.predict())

    def _replay(self,j):
        from .motion import planning_model
        anchor_index=max(k for k,row in enumerate(self.history) if row['seen'][j])
        anchor=self.history[anchor_index]['e'][j]
        count=512
        # One pass through 512/2048/8192/32768. Repeating 32768 forever does not
        # prove the posterior is empty, and on a long 16v16 occlusion it never
        # returns. A public speed-1 grid is the finite continuation.
        while count<=8192:
            samples=np.repeat(anchor[None],count,axis=0);modes=self.rng.integers(9,size=count)
            for row in self.history[anchor_index+1:]:
                samples,modes=planning_model(self.public).advance(samples,modes,anchor,self.rng)
                keep=~visibility(row['p'],samples,self.public.boxes,self.public.cfg.sensing)
                if not keep.any():break
                indices=self.rng.choice(np.flatnonzero(keep),size=count);samples=samples[indices];modes=modes[indices]
            else:return samples,modes
            count*=4
        return self._grid_replay(j,anchor_index,anchor)

    def _grid_replay(self,j,anchor_index,anchor):
        frames=np.stack([np.asarray(row['p'],float) for row in self.history[anchor_index+1:]])
        for spacing in (1.,.5):
            found,point=grid_recover(np.asarray(anchor,float),frames,self.public.boxes,float(self.public.cfg.size),float(self.public.cfg.sensing),spacing)
            if not found:continue
            if visibility(frames[-1],point.reshape(1,2),self.public.boxes,self.public.cfg.sensing).any():continue
            print(f'public-history grid support target={j} spacing={spacing} hidden_steps={frames.shape[0]}',flush=True)
            return np.repeat(point.reshape(1,2),512,axis=0),self.rng.integers(9,size=512)
        for spacing in (.1,.05,.02):
            found,point=fine_recover(np.asarray(anchor,float),frames,self.public.boxes,float(self.public.cfg.size),float(self.public.cfg.sensing),spacing)
            if not found:continue
            if visibility(frames[-1],point.reshape(1,2),self.public.boxes,self.public.cfg.sensing).any():continue
            print(f'public-history fine support target={j} spacing={spacing} hidden_steps={frames.shape[0]}',flush=True)
            return np.repeat(point.reshape(1,2),512,axis=0),self.rng.integers(9,size=512)
        point=_boundary_exit(np.asarray(anchor,float),frames,self.public.boxes,float(self.public.cfg.size),float(self.public.cfg.sensing))
        if point is not None:
            print(f'public-history boundary support target={j} hidden_steps={frames.shape[0]}',flush=True)
            return np.repeat(point.reshape(1,2),512,axis=0),self.rng.integers(9,size=512)
        point=_least_visible(np.asarray(anchor,float),frames[-1],self.public.boxes,float(self.public.cfg.size),float(self.public.cfg.sensing))
        if point is None:raise RuntimeError(f'No legal public point within speed 1 for target {j}')
        print(f'public-history support incomplete target={j} hidden_steps={frames.shape[0]}',flush=True)
        return np.repeat(point.reshape(1,2),512,axis=0),self.rng.integers(9,size=512)

    def sample_joint(self,rng):
        indices=rng.integers(self.count,size=self.public.cfg.n);rows=np.arange(self.public.cfg.n)
        return self.samples[rows,indices].copy(),self.modes[rows,indices].copy()

    def ages(self):return self.t-self.last_seen

    def estimates(self):
        out=[]
        for sample in self.samples:
            cells=np.floor(sample).astype(int)
            _,inverse,counts=np.unique(cells,axis=0,return_inverse=True,return_counts=True)
            members=sample[inverse==counts.argmax()]
            out.append(members[np.argmin(np.sum((members-members.mean(0))**2,axis=1))])
        return np.array(out)

FAMILIES=('long_wall','wall_corner','multi_route','split_gate','staggered_gate','u_courtyard','post_field')

def generate_case(n, family, seed, obstacle_count):
    """Outcome-blind tiled versions of the seven established rectangle layout families."""
    c=Config.scale(n);rng=np.random.default_rng(seed);boxes=[]
    cells=int(np.ceil(np.sqrt(obstacle_count)));pitch=(c.size-4)/cells
    order=rng.permutation(cells*cells)
    for index in order:
        if len(boxes)>=obstacle_count: break
        center=np.array([index//cells,index%cells])*pitch-(c.size-4)/2+pitch/2
        center+=rng.uniform(-.1,.1,2)*pitch
        thick=rng.uniform(.35,.55)*min(pitch,4)
        length=rng.uniform(.55,.8)*pitch
        x,y=center; h=length/2; w=thick/2
        if family=='post_field': local=[[[x-w,x+w],[y-w,y+w]]]
        elif family=='long_wall': local=[[[x-w,x+w],[y-h,y+h]]]
        elif family=='wall_corner': local=[[[x-w,x+w],[y-h,y+h]],[[x-w,x+h],[y+h-thick,y+h]]]
        elif family=='multi_route': local=[[[x-h,x-h+thick],[y-h,y+h]],[[x+h-thick,x+h],[y-h,y+h]]]
        elif family=='split_gate': local=[[[x-w,x+w],[y-h,y-.6]],[[x-w,x+w],[y+.6,y+h]]]
        elif family=='staggered_gate': local=[[[x-h,x-h+thick],[y-h,y+.2]],[[x+h-thick,x+h],[y-.2,y+h]]]
        elif family=='u_courtyard': local=[[[x-h,x-h+thick],[y-h,y+h]],[[x-h,x+h],[y-h,y-h+thick]],[[x-h,x+h],[y+h-thick,y+h]]]
        else: raise ValueError(family)
        for b in local:
            a=np.array(b,float)
            if np.all(a[:,1]>a[:,0]) and len(boxes)<obstacle_count: boxes.append(b)
    if len(boxes)!=obstacle_count: raise ValueError('Obstacle count')
    boxes=np.array(boxes)
    public=PublicMap(c,boxes)
    from .geometry import continuous_certificate
    certificate=continuous_certificate(c,boxes)
    # Exact rectangle union area via vertical sweep (corners may overlap).
    xs=np.unique(boxes[:,0,:]);area=0.
    for a,b in zip(xs[:-1],xs[1:]):
        intervals=sorted(boxes[(boxes[:,0,0]<b)&(boxes[:,0,1]>a),1,:].tolist())
        total=0.;end=-np.inf
        for lo,hi in intervals: total+=max(0.,hi-max(end,lo));end=max(end,hi)
        area+=(b-a)*total
    if not .02 <= area/c.size**2 <= .4: raise ValueError('Occupancy outside [2%,40%]')
    initials=[]
    for _ in range(2):
        p=public.sample_free(rng,n)
        for attempt in range(1000):
            e=public.sample_free(rng,n)
            if np.min(np.linalg.norm(p[:,None]-e[None,:],axis=-1))>=3.:break
        else: raise ValueError('Spawn rejection limit')
        initials.append(dict(p=p.tolist(),e=e.tolist(),alive=[True]*n,t=0))
    continuous_certificate(c,boxes,[q for row in initials for team in ('p','e') for q in row[team]])
    return dict(n=n,family=family,seed=int(seed),boxes=boxes.tolist(),initials=initials,
                geometry=dict(obstacles=len(boxes),occupancy=area/c.size**2,**certificate,route_grid_spacing=1.),
                acceptance='geometry/connectivity/spawn only; no controller used')

def from_initial(case, index):
    row=case['initials'][index]
    return State(np.array(row['p']),np.array(row['e']),np.array(row['alive'],bool),row['t'])

@njit(cache=True)
def corner_edges(vertices,boxes):
    weights=np.zeros((len(vertices),len(vertices)))
    for i in range(len(vertices)):
        for j in range(i):
            if clear(vertices[i],vertices[j],boxes):
                weights[i,j]=np.max(np.abs(vertices[i]-vertices[j]));weights[j,i]=weights[i,j]
    return weights


@njit(cache=True)
def grid_recover(anchor, pursuers, boxes, size, sensing, spacing):
    """One public position reachable from the anchor under speed 1 and every negative report.

    The lattice is anchored at the last positive report, not at map-cell centers:
    a tiny corner just outside the sensing circle is often off the map lattice.
    This does not sample the Gaussian motion law. Hidden coordinates are not an input.
    """
    i_min=int(np.ceil((-size/2-anchor[0])/spacing-1e-9));i_max=int(np.floor((size/2-anchor[0])/spacing+1e-9))
    j_min=int(np.ceil((-size/2-anchor[1])/spacing-1e-9));j_max=int(np.floor((size/2-anchor[1])/spacing+1e-9))
    nx=i_max-i_min+1;ny=j_max-j_min+1
    if nx<1 or ny<1:return False,np.zeros(2)
    radius=int(np.floor(1./spacing+1e-9));cap=nx*ny
    current=np.empty(cap,np.int32);following=np.empty(cap,np.int32);occupied=np.zeros(cap,np.uint8)
    count=0
    for i in range(i_min,i_max+1):
        for j in range(j_min,j_max+1):
            if max(abs(i*spacing),abs(j*spacing))>1.+1e-8:continue
            q=np.array((anchor[0]+i*spacing,anchor[1]+j*spacing))
            if not legal(q,boxes,size) or not clear(anchor,q,boxes):continue
            if _seen_by(pursuers[0],q,boxes,sensing):continue
            index=(i-i_min)*ny+(j-j_min)
            current[count]=index;count+=1;occupied[index]=1
    if count==0:return False,np.zeros(2)
    for frame in range(1,pursuers.shape[0]):
        occupied[:]=0;next_count=0
        for cursor in range(count):
            index=current[cursor];oi=index//ny;oj=index-oi*ny;i=oi+i_min;j=oj+j_min
            base=np.array((anchor[0]+i*spacing,anchor[1]+j*spacing))
            for di in range(-radius,radius+1):
                for dj in range(-radius,radius+1):
                    ii=i+di;jj=j+dj
                    if ii<i_min or jj<j_min or ii>i_max or jj>j_max:continue
                    if max(abs(di*spacing),abs(dj*spacing))>1.+1e-8:continue
                    nxt=(ii-i_min)*ny+(jj-j_min)
                    if occupied[nxt]:continue
                    q=np.array((anchor[0]+ii*spacing,anchor[1]+jj*spacing))
                    if not legal(q,boxes,size) or not clear(base,q,boxes):continue
                    if _seen_by(pursuers[frame],q,boxes,sensing):continue
                    following[next_count]=nxt;next_count+=1;occupied[nxt]=1
        if next_count==0:return False,np.zeros(2)
        current,following=following,current;count=next_count
    chosen=current[0];oi=chosen//ny;oj=chosen-oi*ny
    return True,np.array((anchor[0]+(oi+i_min)*spacing,anchor[1]+(oj+j_min)*spacing))


@njit(cache=True)
def fine_recover(anchor,pursuers,boxes,size,sensing,spacing):
    """Beam search on an anchor-centered spacing inside each speed-1 square.

    Coarse map lattices miss a thin crescent just outside the sensing circle.
    The beam keeps only public, legal, invisible points. Hidden state is absent.
    """
    limit=16
    beam=np.empty((limit,2));beam[0]=anchor;count=1
    radius=int(np.floor(1./spacing+1e-9))
    for frame in range(pursuers.shape[0]):
        nxt=np.empty((limit,2));ncount=0
        for b in range(count):
            base=beam[b]
            for ix in range(-radius,radius+1):
                for iy in range(-radius,radius+1):
                    if max(abs(ix*spacing),abs(iy*spacing))>1.+1e-8:continue
                    q=np.array((base[0]+ix*spacing,base[1]+iy*spacing))
                    if not legal(q,boxes,size) or not clear(base,q,boxes):continue
                    if _seen_by(pursuers[frame],q,boxes,sensing):continue
                    nxt[ncount]=q;ncount+=1
                    if ncount==limit:break
                if ncount==limit:break
            if ncount==limit:break
        if ncount==0:return False,np.zeros(2)
        beam=nxt;count=ncount
    return True,beam[0].copy()


def _least_visible(anchor,frame,boxes,size,sensing):
    """Legal speed-1 point seen by as few pursuers as possible, then farthest from them."""
    best=None;best_key=None
    spacing=.1;radius=int(np.floor(1./spacing+1e-9))
    for ix in range(-radius,radius+1):
        for iy in range(-radius,radius+1):
            step=np.array((ix*spacing,iy*spacing))
            if max(abs(step[0]),abs(step[1]))>1.+1e-8:continue
            q=anchor+step
            if not legal(q,boxes,size) or not clear(anchor,q,boxes):continue
            seen=visibility(frame,q.reshape(1,2),boxes,sensing)
            key=(-int(np.count_nonzero(seen)),float(np.linalg.norm(frame-q,axis=1).min()))
            if best_key is None or key>best_key:best,best_key=q,key
    return best


def _boundary_exit(anchor,frames,boxes,size,sensing):
    """Step just outside a sensing disk, or onto the near side of a blocking corner.

    Used when a lattice misses a crescent thinner than its spacing. Every candidate
    is still checked for the speed box, legality, clearance and all negative reports.
    """
    def accept(q):
        if not np.isfinite(q).all() or np.max(np.abs(q-anchor))>1.+1e-8:return False
        if not legal(q,boxes,size) or not clear(anchor,q,boxes):return False
        return not any(visibility(frame,q.reshape(1,2),boxes,sensing).any() for frame in frames)
    if accept(anchor):return anchor.copy()
    for frame_index,frame in enumerate(frames):
        for pursuer in frame:
            delta=anchor-pursuer;dist=float(np.linalg.norm(delta))
            if dist<1e-8:dirs=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.],[1.,1.],[1.,-1.],[-1.,1.],[-1.,-1.]])
            else:
                unit=delta/dist;perp=np.array([-unit[1],unit[0]])
                dirs=[unit+angle*perp for angle in (0.,-.5,-.25,.25,.5)]
            for direction in dirs:
                direction=np.asarray(direction,float);norm=float(np.linalg.norm(direction))
                if norm<1e-8:continue
                unit=direction/norm
                reach=1./max(abs(unit[0]),abs(unit[1]))
                need=sensing+.02-dist
                if need>reach+1e-6:continue
                q=anchor+unit*min(reach,max(need,0.))
                if accept(q):return q
        if frame_index:continue
        for box in boxes:
            for corner in (box[:,0],box[:,1],np.array([box[0,0],box[1,1]]),np.array([box[0,1],box[1,0]])):
                delta=corner-anchor
                if np.max(np.abs(delta))>1.+1e-8 or np.linalg.norm(delta)<1e-8:continue
                q=anchor+delta*min(1.,.98*1./np.max(np.abs(delta)))
                if accept(q):return q
    return None


@njit(cache=True)
def _seen_by(pursuers,q,boxes,sensing):
    for i in range(pursuers.shape[0]):
        if np.sum((pursuers[i]-q)**2)<=sensing*sensing and clear(pursuers[i],q,boxes):return True
    return False


@njit(cache=True)
def route_heads(q,vertices,boxes):
    result=np.full(len(vertices),np.inf)
    for i in range(len(vertices)):
        if clear(q,vertices[i],boxes):result[i]=np.ceil(np.max(np.abs(q-vertices[i]))-1e-9)
    return result

@njit(cache=True)
def improve_direct_costs(positions,goals,cost,boxes):
    for i in range(len(positions)):
        for j in range(len(goals)):
            direct=np.ceil(np.max(np.abs(goals[j]-positions[i]))-1e-9)
            if direct<=cost[i,j] and clear(positions[i],goals[j],boxes):cost[i,j]=direct
    return cost

@njit(cache=True)
def improve_direct_coverage(future,goals,cost,boxes):
    for a in range(len(future)):
        for j in range(len(goals)):
            for q in future[a]:
                direct=np.ceil(np.max(np.abs(goals[j]-q))-1e-9)
                if direct<=cost[a,j] and clear(q,goals[j],boxes):cost[a,j]=direct
    return cost
