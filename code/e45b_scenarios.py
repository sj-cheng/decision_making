"""Outcome-blind 2v2 histories and an independent simulation trace audit."""
import numpy as np
from numba import njit
from corner_estimator_scenarios import construct,original_model
from prepare_l45_occlusion_scenarios import transform
from run_e45_team_pilot import mirror
from solvers.e45_team_adapter import public_observation
from run_frozen_d_closed_loop import legal_action
from estimator_audit_geometry import point_legal,blocked


def make_case(family,index,split,seed_base):
    fi=('long_wall','wall_corner','multi_route').index(family);si=('train','validation','test').index(split)
    seed=int(np.random.SeedSequence([seed_base,si,fi,index]).generate_state(1)[0]);rng=np.random.default_rng(seed)
    H=4 if index%2==0 else 6;height=float(rng.uniform(5.8,6.8) if H==4 else rng.uniform(7.8,8.45))
    speed=1.1 if index%4<2 else 1.3
    old,packet=construct(family,H,speed,seed,height,index%2,pause=index%2)
    pattern=(index+fi)%3;pattern_name=('opposed_entries','shared_entry','staggered_team')[pattern]
    frame=old['frame'];shift_y=transform([0,.35],frame)-transform([0,0],frame)
    shift_x=transform([-.6,0],frame)-transform([0,0],frame)
    shift_right=transform([.3,0],frame)-transform([0,0],frame)
    p=original_model(packet['map'],speed);worlds=[];shared=None
    for name,order in (('AB',(0,1)),('BA',(1,0))):
        a,b=[old['worlds'][i]['history_states'] for i in order];states=[];history=[]
        for x,y in zip(a,b):
            s=np.asarray(x['full_state']).reshape(-1,1).copy()
            if pattern==1:
                s[2:4,0]=np.asarray(y['evader'])+shift_right;s[6:8,0]=np.asarray(y['pursuer'])+shift_right
            else:
                s[2:4,0]=mirror(y['evader'],frame)
                s[6:8,0]=mirror(y['pursuer'],frame)+(-shift_y if pattern==0 else shift_x)
            s[p.active_idxs,0]=1
            assert p.is_valid(s) and not p.get_capture_pairs(s)
            if states:
                previous=np.asarray(states[-1]).reshape(-1,1);joint=np.zeros((8,1))
                for robot in range(4):
                    delta=s[2*robot:2*robot+2,0]-previous[2*robot:2*robot+2,0]
                    assert legal_action(previous[2*robot:2*robot+2,0],delta,1 if robot<2 else speed,np.asarray(packet['map'])), (family,index,split,pattern_name,robot,previous[2*robot:2*robot+2,0].tolist(),s[2*robot:2*robot+2,0].tolist())
                    joint[2*robot:2*robot+2,0]=delta
                nxt=p.step(previous,joint,1.)
                np.testing.assert_allclose(nxt[:8],s[:8],atol=1e-10,rtol=0);assert np.all(nxt[p.active_idxs]>.5);s=nxt
            states.append(s[:,0].tolist())
            history.append(public_observation(int(s[8,0]),s[4:8,0].reshape(2,2),s[:4,0].reshape(2,2),np.ones(2,bool),np.ones(2,bool),np.asarray(packet['map'])))
        if shared is None:shared=history
        else:assert history==shared
        assert all(all(q is None for q in reports) for reports in history[-1]['reports'])
        worlds.append(dict(id=name,initial=states[-1],scripted_states=states))
    return dict(id=f'rolefree_{split}_{pattern_name}_{old["id"]}',family=family,pattern=pattern_name,
        map=packet['map'],speed=speed,history=shared,worlds=worlds,split=split,
        acceptance='geometry, reachability, causal sensor and identical history only; no policy outcome used')


@njit(cache=True)
def audit_simulated_legacy(trace,boxes,speed):
    errors=np.zeros(4,np.int64)
    for row in trace:
        for i in range(2):
            a=row[2*i:2*i+2];b=row[4+2*i:6+2*i]
            if not point_legal(a,boxes) or not point_legal(b,boxes) or np.max(np.abs(b-a))>speed+2e-7 or blocked(a,b,boxes):errors[0]+=1
            if not row[16+i] and np.max(np.abs(b-a))>1e-10:errors[0]+=1
        for j in range(2):
            if row[18+j]:
                a=row[8+2*j:10+2*j];b=row[12+2*j:14+2*j]
                if not point_legal(a,boxes) or not point_legal(b,boxes) or np.max(np.abs(b-a))>1+2e-7 or blocked(a,b,boxes):errors[1]+=1
        ap=row[16:18].copy();ae=row[18:20].copy()
        for _ in range(2):
            best=1.;bi,bj=-1,-1
            for i in range(2):
                for j in range(2):
                    d=np.sum((row[4+2*i:6+2*i]-row[12+2*j:14+2*j])**2)
                    if ap[i] and ae[j] and d<best:best,bi,bj=d,i,j
            if bi<0:break
            ap[bi]=0;ae[bj]=0
        if np.any(ap!=row[20:22]) or np.any(ae!=row[22:24]):errors[2]+=1
        for j in range(2):
            seen=False
            for i in range(2):
                if ap[i] and ae[j]:
                    a=row[4+2*i:6+2*i];q=row[12+2*j:14+2*j]
                    if np.sum((a-q)**2)<=36+1e-12 and not blocked(a,q,boxes):seen=True
            if seen!=bool(row[24+j]):errors[3]+=1
    return errors


def audit_simulated(trace,boxes,speed):
    from e45b_sensor_audit import audit_visibility
    errors=audit_simulated_legacy(trace,boxes,speed)
    errors[3]=audit_visibility(trace,boxes)
    return errors
