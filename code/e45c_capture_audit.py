"""Independent capture pairing using the frozen environment's norm ordering.

Squared-distance ordering is not equivalent to rounded Euclidean ordering:
distinct squares can have equal rounded square roots. The environment sorts
Euclidean distances stably in pursuer/target order, with a strict radius <1.
"""
import numpy as np
from e45b_scenarios import audit_simulated as previous_audit


def environment_pairing(positions,targets,active_p,active_e):
    """Separate sorted-candidate reference, matching Example8.get_capture_pairs."""
    candidates=[]
    for i in range(2):
        if not active_p[i]:continue
        for j in range(2):
            if not active_e[j]:continue
            distance=float(np.linalg.norm((positions[i]-targets[j])[:,None]))
            if distance<1.:candidates.append((distance,i,j))
    candidates.sort(key=lambda item:item[0])
    ap=np.asarray(active_p,bool).copy();ae=np.asarray(active_e,bool).copy()
    for _,i,j in candidates:
        if ap[i] and ae[j]:ap[i]=False;ae[j]=False
    return ap,ae


def capture_errors(trace):
    if not len(trace):return 0
    positions=trace[:,4:8].reshape(-1,2,2)
    targets=trace[:,12:16].reshape(-1,2,2)
    ap=trace[:,16:18].astype(bool).copy();ae=trace[:,18:20].astype(bool).copy()
    delta=positions[:,:,None,:]-targets[:,None,:,:]
    distances=np.sqrt(np.sum(delta*delta,axis=-1))
    valid=ap[:,:,None]&ae[:,None,:]
    close=valid&(distances<1.)
    candidates=np.where(close,distances,np.inf).reshape(-1,4)
    ordered=np.sort(candidates,axis=1)
    # Exact environment-style norm calls adjudicate all rounding-sensitive cases.
    eps=16*np.finfo(float).eps
    ambiguous=np.any(valid&(abs(distances-1.)<=eps),axis=(1,2))
    for k in range(3):
        finite=np.isfinite(ordered[:,k])&np.isfinite(ordered[:,k+1])
        difference=np.zeros(len(trace));np.subtract(ordered[:,k+1],ordered[:,k],out=difference,where=finite)
        ambiguous |= finite&(difference<=eps)
    for _ in range(2):
        current=np.where(ap[:,:,None]&ae[:,None,:]&close,distances,np.inf).reshape(-1,4)
        winner=np.argmin(current,axis=1);rows=np.flatnonzero(np.min(current,axis=1)<1.)
        ap[rows,winner[rows]//2]=False;ae[rows,winner[rows]%2]=False
    mismatch=np.any(ap!=trace[:,20:22].astype(bool),axis=1)|np.any(ae!=trace[:,22:24].astype(bool),axis=1)
    # Also use the scalar reference for any fast-path disagreement, never waive it.
    for n in np.flatnonzero(ambiguous|mismatch):
        pp,ee=environment_pairing(positions[n],targets[n],trace[n,16:18],trace[n,18:20])
        mismatch[n]=np.any(pp!=trace[n,20:22].astype(bool)) or np.any(ee!=trace[n,22:24].astype(bool))
    return int(mismatch.sum())


def audit_simulated(trace,boxes,speed):
    errors=previous_audit(trace,boxes,speed)
    errors[2]=capture_errors(trace)
    return errors
