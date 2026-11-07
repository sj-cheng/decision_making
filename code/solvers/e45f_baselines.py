"""Matched-tree geometric and VOO-style proposals, not full VOMCPOW replicas."""
import numpy as np
from solvers.e45b_joint_nte import Controller
from solvers.frozen_d_search_interface import move,safe


def best_voronoi_mask(proposals,actions,best,active):
    """Normalized Euclidean metric on the active action coordinates."""
    p=np.asarray(proposals).reshape(len(proposals),4)
    a=np.asarray(actions).reshape(len(actions),4)
    mask=np.repeat(np.asarray(active,bool),2)
    distances=((p[:,None,mask]-a[None,:,mask])**2).sum(-1)
    return distances[:,best]<=distances.min(1)+1e-12


class VoronoiController(Controller):
    """epsilon=.1 global, otherwise uniform rejection in best-Q Voronoi cell.

    Finite rejection cap64 then legal global fallback. Uses original UCT,
    observation branches, D and geometry potential; no learned parameters.
    """
    def propose(self,node,direct=False):
        visited=np.flatnonzero(np.asarray(node.visits)>0)
        local=len(visited)>0 and self.proposal_rng.random()>=.1
        best=int(visited[np.argmax(np.asarray(node.totals)[visited]/np.asarray(node.visits)[visited])]) if local else None
        for phase in range(2 if local else 1):
            for _ in range(64):
                self.allowed();a=self.proposal_rng.uniform(-1,1,(2,2))*self.public.speed;a[~node.ap]=0.
                if local and phase==0 and not best_voronoi_mask(a[None]/self.public.speed,
                        np.asarray(node.actions)/self.public.speed,best,node.ap)[0]:continue
                if all(safe(node.positions[i],a[i],self.public.speed,self.motion_boxes,np.zeros(2,np.int64)) for i in range(2)):
                    return a
                self.rejections+=1
        self.fallbacks+=1;return np.zeros((2,2))


class GeometryController(Controller):
    """75% route-to-belief/visible-corner options, 25% original uniform.

    Each car independently draws a public waypoint; no target assignment or
    AG action fallback. The same JOINT tree evaluates all paired actions.
    """
    def propose(self,node,direct=False):
        if self.proposal_rng.random()>=.75:return super().propose(node,direct)
        if not hasattr(node,'geometric_options'):
            self.features(node);p=self.public;options=[]
            for i in range(2):
                choices=[np.zeros(2)]
                if node.ap[i]:
                    for j in range(2):
                        if not node.ae[j]:continue
                        mass=np.asarray(node.weights[j]);cdf=np.cumsum(mass)
                        order=np.unique(np.minimum(63,np.searchsorted(cdf,(np.arange(8)+.5)/8)))
                        for k in order:
                            self.allowed()
                            if mass[k]>0:choices.append(move(node.positions[i],node.samples[j,k],p.speed,p.vertices,p.dist,p.boxes,np.zeros(2,np.int64)))
                    for v in p.vertices:
                        self.allowed();choices.append(move(node.positions[i],v,p.speed,p.vertices,p.dist,p.boxes,np.zeros(2,np.int64)))
                unique=[]
                for a in choices:
                    if not any(np.max(abs(a-b))<1e-10 for b in unique):unique.append(a)
                options.append(np.asarray(unique))
            node.geometric_options=options
        a=np.array([c[self.proposal_rng.integers(len(c))] for c in node.geometric_options]);a[~node.ap]=0.
        self.allowed()
        if all(safe(node.positions[i],a[i],self.public.speed,self.motion_boxes,np.zeros(2,np.int64)) for i in range(2)):
            return a
        self.rejections+=1;return super().propose(node,direct)
