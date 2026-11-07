"""Independent orientation/edge geometry for estimator certificates."""
import numpy as np
from numba import njit

EPS = 2e-7


@njit(cache=True)
def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


@njit(cache=True)
def on(a, b, q):
    return (abs(cross(a,b,q)) <= EPS and np.all(q >= np.minimum(a,b)-EPS)
            and np.all(q <= np.maximum(a,b)+EPS))


@njit(cache=True)
def intersect(a,b,c,d):
    x,y,z,w = cross(a,b,c),cross(a,b,d),cross(c,d,a),cross(c,d,b)
    proper = ((x>EPS and y<-EPS) or (x<-EPS and y>EPS)) and ((z>EPS and w<-EPS) or (z<-EPS and w>EPS))
    return proper or on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b)


@njit(cache=True)
def point_legal(q, boxes):
    if not np.all(np.isfinite(q)) or np.any(q < -10.-EPS) or np.any(q > 10.+EPS):
        return False
    for b in boxes:
        if b[0,0]-EPS <= q[0] <= b[0,1]+EPS and b[1,0]-EPS <= q[1] <= b[1,1]+EPS:
            return False
    return True


@njit(cache=True)
def blocked(a,b,boxes):
    for box in boxes:
        corners = np.array([[box[0,0],box[1,0]],[box[0,1],box[1,0]],
                            [box[0,1],box[1,1]],[box[0,0],box[1,1]]])
        for i in range(4):
            if intersect(a,b,corners[i],corners[(i+1)%4]):
                return True
    return False


@njit(cache=True)
def validate_paths(paths, observers, boxes):
    errors, transitions = 0, 0
    for path in paths:
        for t in range(len(observers)):
            a,b = path[t],path[t+1]
            lawful = point_legal(a,boxes) and point_legal(b,boxes) and np.max(np.abs(b-a)) <= 1.+EPS and not blocked(a,b,boxes)
            seen = np.sum((b-observers[t])**2) <= (6.+EPS)**2 and not blocked(observers[t],b,boxes)
            errors += int(not lawful or seen)
            transitions += 1
    return errors, transitions
