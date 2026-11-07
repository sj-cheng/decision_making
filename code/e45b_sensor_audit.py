"""Independent closed-box sensor reference; exact arithmetic near tangencies.

The old orientation audit uses a 2e-7 *area* tolerance, which is unsuitable
for visibility near obstacle vertices. This module keeps the frozen sensor's
range/parallel/segment-parameter conventions, without calling its predicate.
"""
from fractions import Fraction
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=65536)
def exact_visible(a, q, boxes):
    if sum((x-y)**2 for x,y in zip(a,q)) > 36+1e-12:
        return False
    f=Fraction.from_float; eps=f(1e-12)
    for box in boxes:
        lower,upper=Fraction(0),Fraction(1)
        possible=True
        for axis in range(2):
            start=f(float(a[axis])); delta=f(float(q[axis]))-start
            lo,hi=map(lambda x:f(float(x)),box[axis])
            if abs(delta)<eps:
                if not lo<=start<=hi:possible=False;break
            else:
                bounds=sorted(((lo-start)/delta,(hi-start)/delta))
                lower=max(lower,bounds[0]);upper=min(upper,bounds[1])
        if possible and lower<=upper+eps:return False
    return True


def visible_batch(starts, ends, boxes):
    """Vectorized reference, certifying numerically ambiguous clips exactly."""
    starts=np.asarray(starts,dtype=float).reshape(-1,2)
    ends=np.asarray(ends,dtype=float).reshape(-1,2)
    boxes=np.asarray(boxes,dtype=float).reshape(-1,2,2)
    delta=ends-starts
    visible=np.sum(delta*delta,axis=1)<=36+1e-12
    ambiguous=np.zeros(len(starts),bool)
    for box in boxes:
        lower=np.zeros(len(starts));upper=np.ones(len(starts));possible=np.ones(len(starts),bool)
        for axis in range(2):
            parallel=np.abs(delta[:,axis])<1e-12
            possible &= ~parallel | ((starts[:,axis]>=box[axis,0]) & (starts[:,axis]<=box[axis,1]))
            divisor=np.where(parallel,1.,delta[:,axis])
            left=(box[axis,0]-starts[:,axis])/divisor
            right=(box[axis,1]-starts[:,axis])/divisor
            lower=np.maximum(lower,np.where(parallel,-np.inf,np.minimum(left,right)))
            upper=np.minimum(upper,np.where(parallel,np.inf,np.maximum(left,right)))
        ambiguous |= possible & (np.abs(lower-upper)<=1e-10)
        visible &= ~(possible & (lower<=upper+1e-12))
    frozen=tuple(tuple(tuple(float(x) for x in axis) for axis in box) for box in boxes)
    for index in np.flatnonzero(ambiguous):
        visible[index]=exact_visible(tuple(starts[index]),tuple(ends[index]),frozen)
    return visible


def audit_visibility(trace,boxes):
    if not len(trace):return 0
    positions=trace[:,4:8].reshape(-1,2,2)
    targets=trace[:,12:16].reshape(-1,2,2)
    starts=np.broadcast_to(positions[:,:,None,:],(len(trace),2,2,2)).reshape(-1,2)
    ends=np.broadcast_to(targets[:,None,:,:],(len(trace),2,2,2)).reshape(-1,2)
    detections=visible_batch(starts,ends,boxes).reshape(-1,2,2)
    detections &= trace[:,20:22,None].astype(bool)
    detections &= trace[:,None,22:24].astype(bool)
    return int(np.count_nonzero(np.any(detections,axis=1)!=trace[:,24:26].astype(bool)))
