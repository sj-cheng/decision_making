"""Full-state geometry escape reference: finite lookahead, not a trained expert."""
from types import SimpleNamespace
import time
import numpy as np
from numba import njit
from solvers.frozen_d_search_interface import PublicMap,move,route,safe,sensor_visible

ACTIONS=np.array([[x,y] for scale in (1.,.5) for x in (-scale,0.,scale) for y in (-scale,0.,scale)
                  if x!=0 or y!=0]+[[0.,0.]])


@njit(cache=True)
def escape_one(at,pursuers,active,boxes,vertices,pdist,edist,pursuer_speed,depth=4,width=8):
    # State = evader xy, two pursuer xy; each pursuer conservatively chases
    # this target. Both actions are then applied simultaneously.
    states=np.zeros((width,6));states[0,:2]=at;states[0,2:]=pursuers.reshape(4)
    first=np.zeros((width,2));scores=np.zeros(width);alive=np.ones(width,np.bool_);n=1
    count=np.zeros(2,np.int64)
    for level in range(depth):
        ns=np.zeros((width*len(ACTIONS),6));nf=np.zeros((width*len(ACTIONS),2))
        value=np.full(width*len(ACTIONS),-1e12);live=np.zeros(width*len(ACTIONS),np.bool_);size=0
        for k in range(n):
            if not alive[k]:
                ns[size]=states[k];nf[size]=first[k];value[size]=scores[k]-100.;size+=1;continue
            q=states[k,:2];pp=states[k,2:].reshape(2,2);nextp=pp.copy()
            for i in range(2):
                if active[i]:nextp[i]+=move(pp[i],q,pursuer_speed,vertices,pdist,boxes,count)
            for action in ACTIONS:
                if not safe(q,action,1.,boxes,count):continue
                nq=q+action;minimum=100.;geodesic=100.;hidden=True
                for i in range(2):
                    if active[i]:
                        minimum=min(minimum,np.sqrt(np.sum((nq-nextp[i])**2)))
                        geodesic=min(geodesic,float(route(nextp[i],nq,1.,vertices,edist,boxes,count)[0]))
                        if sensor_visible(nextp[i],nq,boxes):hidden=False
                survives=minimum>=1.
                margin=min(minimum,10.)+.25*min(geodesic,20.)+(.4 if hidden else 0.)
                boundary=max(0.,min(10.-abs(nq[0]),10.-abs(nq[1])))
                score=scores[k]+(.9**level)*(margin+.05*min(boundary,2.))
                if not survives:score-=1000.+100.*(depth-level)
                ns[size,:2]=nq;ns[size,2:]=nextp.reshape(4)
                nf[size]=action if level==0 else first[k];value[size]=score;live[size]=survives;size+=1
        order=np.argsort(-value[:size]);n=min(width,size)
        for k in range(n):
            idx=order[k];states[k]=ns[idx];first[k]=nf[idx];scores[k]=value[idx];alive[k]=live[idx]
    return first[0],scores[0],alive[0]


class GeometricEvader:
    """RootPUCT-compatible environment-side interface; no pursuer policy access."""
    def __init__(self,*args,**kwargs):self.public=None;self.records=[]
    def seeded_state(self,problem,*args):return None
    def search(self,problem,state,turn=0):
        if self.public is None:self.public=PublicMap(problem.obstacles,problem.current_pursuer_speed_lim)
        start=time.perf_counter();p=self.public;out=np.zeros((2,2));ap=state[problem.active_idxs[2:],0]>.5
        for j in range(2):
            if state[problem.active_idxs[j],0]>.5:
                out[j],score,predicted_safe=escape_one(state[2*j:2*j+2,0],state[4:8,0].reshape(2,2),ap,
                    p.boxes,p.vertices,p.dist,p.edist,p.speed)
                assert safe(state[2*j:2*j+2,0],out[j],1.,p.boxes,np.zeros(2,np.int64))
        self.records.append(dict(wall_ms=1000*(time.perf_counter()-start),depth=4,beam_width=8,
            full_state=True,predicted_pursuer='map-aware direct chase of this target; not actual L5',
            trained=False))
        return SimpleNamespace(best_action=np.r_[out.ravel(),np.zeros(4)],success=True)


def acceptance():
    p=PublicMap([],1.3);count=np.zeros(2,np.int64)
    cases=[(np.array([0.,0.]),np.array([[-2.,0.],[-8.,-8.]])),
           (np.array([9.5,0.]),np.array([[7.5,0.],[-8.,-8.]]))]
    rows=[]
    for q,pp in cases:
        action,score,alive=escape_one(q,pp,np.array([True,False]),p.boxes,p.vertices,p.dist,p.edist,p.speed)
        assert safe(q,action,1.,p.boxes,count)
        nextp=pp[0]+move(pp[0],q,1.3,p.vertices,p.dist,p.boxes,count)
        assert np.linalg.norm(q+action-nextp)>=1.,'Avoidable immediate capture was not avoided'
        assert np.linalg.norm(action)>0,'Must move when stationary action would be captured'
        rows.append(dict(evader=q.tolist(),pursuer=pp[0].tolist(),action=action.tolist(),immediate_capture_avoided=True))
    return rows
