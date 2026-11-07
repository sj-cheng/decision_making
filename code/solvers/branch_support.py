"""Public-history reachable support and deterministic approach branches.

Cells are outer sets, not calibrated posterior probabilities. Obstacle rejection
uses sufficient certificates; ambiguous cells/transitions are retained. No API
in this module accepts a hidden environment state.
"""
import math
import numpy as np
from numba import njit
from solvers.local_pursuit import PublicMap, clear, legal, visible, edge, uniform

RESOLUTION = .25
N = 80
EPS = 4e-6


class SupportFailure(RuntimeError):
    pass


@njit(cache=True)
def bounds(i):
    x, y = i // N, i % N
    return np.array([-10.+x*.25, -10.+y*.25]), np.array([-10.+(x+1)*.25, -10.+(y+1)*.25])


@njit(cache=True)
def cell_id(q):
    return min(79, max(0, int((q[0]+10.)*4))) * 80 + min(79, max(0, int((q[1]+10.)*4)))


@njit(cache=True)
def separated(lo, hi, qlo, qhi, boxes):
    # A rectangle spans the whole endpoint bounding interval in the other
    # coordinate: every possible straight transition intersects that rectangle.
    for b in boxes:
        for a in range(2):
            k = 1-a
            across = (hi[a] < b[a, 0] and qlo[a] > b[a, 1]) or (qhi[a] < b[a, 0] and lo[a] > b[a, 1])
            covered = b[k, 0] <= min(lo[k], qlo[k]) and b[k, 1] >= max(hi[k], qhi[k])
            if across and covered:
                return True
    return False


@njit(cache=True)
def free_geometry(boxes):
    area = np.zeros(6400)
    representative = np.zeros((6400, 2))
    counter = np.zeros(2, np.int64)
    # Partition at obstacle boundaries: exact free area, also for thin slivers.
    for i in range(6400):
        lo, hi = bounds(i)
        xs = np.empty(2+2*len(boxes)); ys = np.empty_like(xs)
        xs[0], xs[1], ys[0], ys[1] = lo[0], hi[0], lo[1], hi[1]
        for j, b in enumerate(boxes):
            xs[2+2*j:4+2*j] = np.minimum(hi[0], np.maximum(lo[0], b[0]))
            ys[2+2*j:4+2*j] = np.minimum(hi[1], np.maximum(lo[1], b[1]))
        xs.sort(); ys.sort(); largest = 0.
        for x in range(len(xs)-1):
            for y in range(len(ys)-1):
                size = (xs[x+1]-xs[x])*(ys[y+1]-ys[y])
                q = np.array([(xs[x+1]+xs[x])*.5, (ys[y+1]+ys[y])*.5])
                if size > 0 and legal(q, boxes, counter):
                    area[i] += size
                    if size > largest:
                        largest = size; representative[i] = q
    return area, representative


@njit(cache=True)
def transitions(area, boxes):
    links = np.full((6400, 121), -1, np.int32)
    lengths = np.zeros(6400, np.int32)
    for i in range(6400):
        if area[i] == 0: continue
        lo, hi = bounds(i)
        for dx in range(-5, 6):
            x = i//80 + dx
            if x < 0 or x >= 80: continue
            for dy in range(-5, 6):
                y = i%80 + dy
                if y < 0 or y >= 80: continue
                j = x*80+y
                if area[j] == 0: continue
                qlo, qhi = bounds(j)
                if np.max(np.maximum(lo-qhi, qlo-hi)) > 1.+EPS: continue
                if separated(lo, hi, qlo, qhi, boxes): continue
                links[i, lengths[i]] = j; lengths[i] += 1
    return links, lengths


@njit(cache=True)
def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


@njit(cache=True)
def cell_certainly_visible(at, lo, hi, boxes):
    points = np.array([[lo[0],lo[1]], [lo[0],hi[1]], [hi[0],lo[1]], [hi[0],hi[1]], [at[0],at[1]]])
    for j in range(4):
        if np.sum((points[j]-at)**2) > 36.-EPS: return False
    # Sort five points and construct their convex hull. If the viewing hull
    # intersects any obstacle, visibility is uncertain and the cell is kept.
    for i in range(5):
        for j in range(i+1,5):
            if points[j,0] < points[i,0] or (points[j,0] == points[i,0] and points[j,1] < points[i,1]):
                tmp=points[i].copy(); points[i]=points[j]; points[j]=tmp
    hull=np.zeros((10,2)); k=0
    for i in range(5):
        while k>=2 and cross(hull[k-2],hull[k-1],points[i]) <= 0: k-=1
        hull[k]=points[i]; k+=1
    lower=k+1
    for i in range(3,-1,-1):
        while k>=lower and cross(hull[k-2],hull[k-1],points[i]) <= 0: k-=1
        hull[k]=points[i]; k+=1
    k-=1
    for b in boxes:
        disjoint=False
        if np.max(hull[:k,0]) < b[0,0] or np.min(hull[:k,0]) > b[0,1] or np.max(hull[:k,1]) < b[1,0] or np.min(hull[:k,1]) > b[1,1]:
            continue
        for j in range(k):
            a=hull[j]; z=hull[(j+1)%k]
            outside=True
            for x in range(2):
                for y in range(2):
                    if cross(a,z,np.array([b[0,x],b[1,y]])) >= -EPS: outside=False
            if outside: disjoint=True; break
        if not disjoint: return False
    return True


@njit(cache=True)
def advance(mask, exact, at, area, boxes, links, lengths):
    nxt=np.zeros(6400,np.bool_)
    if np.isfinite(exact[0]):
        for i in range(6400):
            if area[i]==0: continue
            lo,hi=bounds(i)
            if np.max(np.maximum(lo-exact, exact-hi)) <= 1.+EPS and not separated(exact,exact,lo,hi,boxes): nxt[i]=True
    else:
        for i in range(6400):
            if mask[i]:
                for j in range(lengths[i]): nxt[links[i,j]]=True
    for i in range(6400):
        if nxt[i]:
            lo,hi=bounds(i)
            if cell_certainly_visible(at,lo,hi,boxes): nxt[i]=False
    return nxt


@njit(cache=True)
def tail_table(points, area, vertices, dist, speed, boxes):
    table=np.full((6400,len(vertices)),1000000,np.int64)
    count=np.zeros(2,np.int64)
    for i in range(6400):
        if area[i]==0: continue
        for j in range(len(vertices)):
            if clear(vertices[j],points[i],boxes,count):
                cost=edge(vertices[j],points[i],speed)
                for k in range(len(vertices)): table[i,k]=min(table[i,k],dist[k,j]+cost)
    return table


@njit(cache=True)
def approach(at, mask, exact, points, vertices, dist, tails, speed, boxes):
    costs=np.full(6400,1000000,np.int64); first=np.full(6400,-1,np.int64)
    counts=np.zeros(2,np.int64)
    head=np.full(len(vertices),1000000,np.int64)
    for j in range(len(vertices)):
        if clear(at,vertices[j],boxes,counts): head[j]=edge(at,vertices[j],speed)
    for i in range(6400):
        if not mask[i]: continue
        q=exact if np.isfinite(exact[0]) else points[i]
        if clear(at,q,boxes,counts): costs[i]=edge(at,q,speed)
        for j in range(len(vertices)):
            c=head[j]+tails[i,j]
            if c<costs[i]: costs[i]=c; first[i]=j
    return costs,first,counts[0]


@njit(cache=True)
def draw(mask, exact, at, points, area, boxes, rng, number):
    out=np.empty((number,2)); count=np.zeros(2,np.int64)
    if np.isfinite(exact[0]):
        out[:]=exact; return out,0
    cumulative=np.cumsum(area*mask); total=cumulative[-1]
    for k in range(number):
        found=False
        for _ in range(10000):
            i=np.searchsorted(cumulative,uniform(rng)*total,side='right')
            if i>=6400: i=6399
            lo,hi=bounds(i)
            q=lo+np.array([uniform(rng),uniform(rng)])*.25
            if legal(q,boxes,count) and not visible(at,q,boxes,count):
                out[k]=q; found=True; break
        if not found: return out,-1
    return out,count[0]


class GridMap(PublicMap):
    def __init__(self, boxes, speed):
        super().__init__(boxes,speed)
        self.area,self.points=free_geometry(self.boxes)
        self.links,self.lengths=transitions(self.area,self.boxes)
        self.tails=tail_table(self.points,self.area,self.vertices,self.dist,self.speed,self.boxes)


@njit(cache=True)
def partition(at,mask,exact,points,area,vertices,costs,first):
    size=len(vertices)+12
    labels=np.full(6400,-1,np.int64);mass=np.zeros(size);vectors=np.zeros((size,2));keys=np.arange(size)
    for i in range(6400):
        if not mask[i]:continue
        q=exact if np.isfinite(exact[0]) else points[i]
        d=(vertices[first[i]] if first[i]>=0 else q)-at
        a=math.atan2(d[1],d[0])
        key=first[i] if first[i]>=0 else len(vertices)+int(math.floor((a+math.pi)/(math.pi/6)))%12
        labels[i]=key;mass[key]+=area[i];vectors[key]+=d/max(np.linalg.norm(d),1e-12)*area[i]
    while True:
        alive=np.flatnonzero(mass>0)
        if len(alive)<=1:break
        best=100.;left=-1;right=-1
        for ii in range(len(alive)):
            i=alive[ii];a=math.atan2(vectors[i,1],vectors[i,0])
            for jj in range(ii+1,len(alive)):
                j=alive[jj];b=math.atan2(vectors[j,1],vectors[j,0])
                delta=abs(math.atan2(math.sin(a-b),math.cos(a-b)))
                if delta<best:
                    best=delta;left=i;right=j
        if len(alive)<=4 and best>=math.pi/6:break
        mass[left]+=mass[right];mass[right]=0;vectors[left]+=vectors[right]
        for i in range(6400):
            if labels[i]==right:labels[i]=left
    active=np.flatnonzero(mass>0);features=np.zeros((4,12));assignment=np.full(6400,-1,np.int64)
    info=np.zeros((4,5));total=mass.sum()
    for k in range(len(active)):
        label=active[k];lo=np.full(2,1e9);hi=np.full(2,-1e9);center=np.zeros(2);mn=1000000;mx=0;nc=0
        for i in range(6400):
            if labels[i]!=label:continue
            assignment[i]=k;nc+=1
            celllo,cellhi=bounds(i);lo=np.minimum(lo,celllo);hi=np.maximum(hi,cellhi)
            center+=points[i]*area[i];mn=min(mn,costs[i]);mx=max(mx,costs[i])
        center/=mass[label];angle=math.atan2(vectors[label,1],vectors[label,0])
        features[k,:2]=np.array([math.cos(angle),math.sin(angle)])
        features[k,2:4]=(lo-at)/20.;features[k,4:6]=(hi-at)/20.;features[k,6:8]=(center-at)/20.
        features[k,8:]=np.array([mass[label]/total,mn/80.,mx/80.,1.])
        info[k]=np.array([label,nc,mass[label],mn,mx])
    return features,assignment,info,len(active)


@njit(cache=True)
def mode_point(mask,points,area):
    masses=np.zeros(400)
    for i in range(6400):
        if mask[i]:masses[min(19,max(0,int(math.floor(points[i,0])+10)))*20+min(19,max(0,int(math.floor(points[i,1])+10)))]+=area[i]
    winner=np.argmax(masses);center=np.array([winner//20-9.5,winner%20-9.5]);distance=1e9;index=-1
    for i in range(6400):
        if mask[i] and int(math.floor(points[i,0])+10)*20+int(math.floor(points[i,1])+10)==winner:
            d=np.sum((points[i]-center)**2)
            if d<distance:distance=d;index=i
    return points[index].copy()


class Support:
    def __init__(self, public):
        self.public=public
        self.mask=np.zeros(6400,bool)
        self.exact=np.full(2,np.nan)
        self.at=np.zeros(2)
        self.step=-1

    def clone(self):
        other=Support(self.public)
        other.mask=self.mask.copy(); other.exact=self.exact.copy()
        other.at=self.at.copy(); other.step=self.step
        return other

    def update(self, observation):
        if observation.step != self.step+1 and self.step>=0:
            raise ValueError('Support observations must be consecutive')
        if observation.target is not None:
            self.exact=np.asarray(observation.target,float).copy()
            self.mask=np.zeros(6400,bool); self.mask[cell_id(self.exact)]=True
        else:
            if self.step<0: raise ValueError('A legal positive initialization is required')
            p=self.public
            self.mask=advance(self.mask,self.exact,observation.position,p.area,p.boxes,p.links,p.lengths)
            self.exact=np.full(2,np.nan)
        self.at=np.asarray(observation.position,float).copy(); self.step=int(observation.step)
        if not self.mask.any(): raise SupportFailure('Empty lawful grid support')

    def sample(self,rng,n=64):
        p=self.public
        result,queries=draw(self.mask,self.exact,self.at,p.points,p.area,p.boxes,rng,n)
        if queries<0: raise SupportFailure('No lawful point in retained grid support')
        return result,int(queries)

    def covers(self,q):
        return bool(self.mask[cell_id(np.asarray(q))])

    def branches(self):
        p=self.public
        costs,first,queries=approach(self.at,self.mask,self.exact,p.points,p.vertices,p.dist,p.tails,p.speed,p.boxes)
        features,assignment,info,n=partition(self.at,self.mask,self.exact,p.points,p.area,p.vertices,costs,first)
        summary=[dict(key=int(info[k,0]),cells=int(info[k,1]),area=float(info[k,2]),angle=math.atan2(features[k,1],features[k,0]),min_steps=int(info[k,3]),max_steps=int(info[k,4])) for k in range(n)]
        return features,assignment,summary,int(queries)

    def representative(self):
        if np.isfinite(self.exact[0]): return self.exact.copy()
        p=self.public
        return mode_point(self.mask,p.points,p.area)
