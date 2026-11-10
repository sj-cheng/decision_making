"""Polynomial bottleneck matching and exact rectangle-cell connectivity certificates."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label


def bottleneck_assignment(cost):
    cost=np.asarray(cost,float)
    if cost.ndim!=2 or not cost.size:return np.array([],int),np.array([],int)
    levels=np.unique(cost[np.isfinite(cost)])
    if not len(levels):raise ValueError('No finite assignment')
    def solve(threshold):
        try:
            rows,cols=linear_sum_assignment(np.where(cost<=threshold,cost,np.inf))
            return rows,cols
        except ValueError:return None
    lo,hi=0,len(levels)-1
    while lo<hi:
        mid=(lo+hi)//2
        if solve(levels[mid]) is None:lo=mid+1
        else:hi=mid
    result=solve(levels[lo])
    if result is None:raise ValueError('Incomplete rectangular bottleneck assignment')
    return result


def continuous_certificate(cfg,boxes,points=()):
    """Exact free-space components for closed axis-aligned rectangles, not a coarse grid."""
    boxes=np.asarray(boxes,float).reshape(-1,2,2);h=cfg.size/2
    if (not np.isfinite(boxes).all() or np.any(boxes[:,:,1]<=boxes[:,:,0])
        or np.any(boxes<=-h) or np.any(boxes>=h)):
        raise ValueError('Invalid rectangles or boundary contact')
    xs=np.unique(np.r_[-h,boxes[:,0,:].ravel(),h]);ys=np.unique(np.r_[-h,boxes[:,1,:].ravel(),h])
    free=np.ones((len(xs)-1,len(ys)-1),bool)
    for b in boxes:
        x0,x1=np.searchsorted(xs,b[0]);y0,y1=np.searchsorted(ys,b[1])
        free[x0:x1,y0:y1]=False
    components,count=label(free,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    if count!=1:raise ValueError(f'Continuous free space has {count} components')
    for q in np.asarray(points,float).reshape(-1,2):
        if np.any(q<=-h) or np.any(q>=h):raise ValueError('Spawn outside open arena interior')
        if any(np.all(q>=b[:,0]) and np.all(q<=b[:,1]) for b in boxes):raise ValueError('Spawn in closed obstacle')
        i=min(len(xs)-2,max(0,int(np.searchsorted(xs,q[0],side='right')-1)))
        j=min(len(ys)-2,max(0,int(np.searchsorted(ys,q[1],side='right')-1)))
        if not components[i,j]:raise ValueError('Spawn outside certified free component')
    return dict(components=int(count),decomposition_cells=int(free.size),method='exact rectangle boundary decomposition, open cells and positive-width adjacency')
