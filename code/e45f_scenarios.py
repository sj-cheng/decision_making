"""Outcome-blind E45-F layout shifts, preserving causal two-world histories."""
import copy
import hashlib
import json
import numpy as np
from e45b_scenarios import make_case
from corner_estimator_scenarios import original_model
from prepare_l45_occlusion_scenarios import transform, transformed_boxes
from solvers.e45_team_adapter import public_observation
from solvers.frozen_d_search_interface import PublicMap, route
from run_frozen_d_closed_loop import legal_action

ID_FAMILIES=('long_wall','wall_corner','multi_route')
OOD_FAMILIES=('split_gate','staggered_gate','u_courtyard','post_field')


def geometry_key(boxes):
    """Remove translation and quarter-turn duplicates, 1e-4 coordinate precision."""
    b=np.asarray(boxes,float);center=b[0].mean(1);signatures=[]
    for q in range(4):
        converted=[]
        for box in b:
            points=np.array([[x,y] for x in box[0] for y in box[1]])-center
            for _ in range(q):points=points[:,[1,0]]*np.array([-1.,1.])
            converted.append(np.round(np.stack((points.min(0),points.max(0)),axis=1),4).tolist())
        signatures.append(json.dumps(sorted(converted),separators=(',',':')))
    return hashlib.sha256(min(signatures).encode()).hexdigest()


def canonical_frame(case):
    box=np.asarray(case['map'][0]);center=box.mean(1)
    point=np.asarray(case['history'][0]['positions'][0])-center
    height=float(np.max(box[:,1]-box[:,0])/2)
    errors=[]
    for q in range(4):
        at=point.copy()
        for _ in range(q):at=np.array([at[1],-at[0]])
        errors.append(np.linalg.norm(at-[-3.,height+.8]))
    turn=int(np.argmin(errors))
    if errors[turn]>1e-7:raise ValueError('Cannot recover source public frame')
    return dict(quarter_turns=turn,translation=center.tolist()),height


def audit_case(case):
    boxes=np.asarray(case['map'],float)
    if not 1<=len(boxes)<=8:raise ValueError('Outside frozen map encoder capacity')
    if not np.isfinite(boxes).all() or np.any(boxes[:,:,0]>=boxes[:,:,1]):raise ValueError('Bad rectangle')
    if np.max(np.abs(boxes))>=9.8:raise ValueError('Obstacle touches arena-edge intent goals')
    p=original_model(boxes,case['speed']);shared=None
    for world in case['worlds']:
        history=[];states=world['scripted_states'];replayed=[]
        for i,state in enumerate(states):
            s=np.array(state).reshape(-1,1)
            if not p.is_valid(s) or p.get_capture_pairs(s):raise ValueError('Illegal scripted state')
            if i:
                previous=np.array(replayed[-1]).reshape(-1,1);action=s[:8]-previous[:8]
                for robot in range(4):
                    at=previous[2*robot:2*robot+2,0];a=action[2*robot:2*robot+2,0]
                    if not legal_action(at,a,1. if robot<2 else case['speed'],boxes):raise ValueError('Script crosses new wall')
                nxt=p.step(previous,action,1.)
                if not np.allclose(nxt[:p.base_state_dim],s[:p.base_state_dim],atol=1e-10,rtol=0):raise ValueError('Script dynamics mismatch')
                # New walls change the legacy simulator's sensor memory too.
                # Recompute it by ORIGINAL dynamics instead of retaining old-map memory.
                s=nxt
            else:
                s[p.base_state_dim:]=0.;p._update_observation_memory(s,s,0.)
            replayed.append(s[:,0].tolist())
            history.append(public_observation(int(s[8,0]),s[4:8,0].reshape(2,2),s[:4,0].reshape(2,2),
                           np.ones(2,bool),np.ones(2,bool),boxes))
        if shared is None:shared=history
        elif shared!=history:raise ValueError('Worlds have distinguishable public histories')
        world['scripted_states']=replayed;world['initial']=replayed[-1]
    if any(q is not None for reports in shared[-1]['reports'] for q in reports):raise ValueError('Handoff not hidden')
    if any(not any(q is not None for q in shared[0]['reports'][j]) for j in range(2)):
        raise ValueError('Each target needs an initial positive measurement')
    # Construction-only reachability certificate, not a controller success test.
    public=PublicMap(boxes,case['speed'])
    for world in case['worlds']:
        at=np.array(world['initial'])[:8].reshape(4,2)
        for a in at[2:]:
            for b in at[:2]:
                if route(a,b,case['speed'],public.vertices,public.dist,boxes,np.zeros(2,np.int64))[0]>=1000000:
                    raise ValueError('Disconnected handoff')
    case['history']=shared
    case['geometry_summary']=dict(obstacles=len(boxes),area=float(np.prod(boxes[:,:,1]-boxes[:,:,0],axis=1).sum()),
        last_seen_steps=[max(r['step'] for r in shared if any(q is not None for q in r['reports'][j])) for j in range(2)],
        handoff_step=shared[-1]['step'])
    return case


def shifted_case(layout,index,split,seed_base):
    source='wall_corner' if layout=='u_courtyard' else 'long_wall'
    case=copy.deepcopy(make_case(source,index,split,seed_base))
    frame,h=canonical_frame(case)
    seed=int(np.random.SeedSequence([seed_base,OOD_FAMILIES.index(layout),index,9457]).generate_state(1)[0])
    rng=np.random.default_rng(seed);dense=bool(index%2)
    half_gap=float(rng.uniform(.65,.95));g=float(rng.uniform(.36,.44)*h)
    thickness=float(rng.uniform(.65,1.05))
    split_wall=[[[-thickness,thickness],[-h,-g-half_gap]],
                [[-thickness,thickness],[-g+half_gap,g-half_gap]],
                [[-thickness,thickness],[g+half_gap,h]]]
    posts=[[[4.8,5.8],[-1.8,-.6]],[[-6.4,-5.2],[1.,2.2]],[[5.5,6.8],[3.2,4.4]]]
    if layout=='split_gate':
        base=split_wall+(posts if dense else [])
    elif layout=='staggered_gate':
        offset=float(rng.uniform(.2,.7))
        base=split_wall+[[[4.1,4.8],[-h,-.8+offset]],[[4.1,4.8],[1.1+offset,h]]]
        if dense:base.append(posts[1])
    elif layout=='u_courtyard':
        base=[[[-1.,1.],[-h,h]],[[1.,3.],[h-1.4,h]],[[1.,3.],[-h,-h+1.4]]]
        if dense:base+=posts
    elif layout=='post_field':
        # Three separated wall sections plus blocks, genuinely absent in training.
        base=split_wall+posts[:3 if dense else 1]
    else:raise ValueError(layout)
    # Vary every additional block size/location without touching the source path rails.
    for b in base[3:]:
        if np.min(np.abs(np.asarray(b)[0]))>4.9:
            jitter=rng.uniform(-.15,.15,2)
            for axis in range(2):b[axis]=[x+float(jitter[axis]) for x in b[axis]]
    case.update(id=f'e45f_{split}_{layout}_{case["id"]}',family=layout,map=transformed_boxes(base,frame),
                source_family=source,density='dense' if dense else 'sparse',
                shift_parameters=dict(gap_width=2*half_gap,core_half_thickness=thickness),
                suite='generalization',split=split,
                acceptance='Public geometry, exact history legality/indistinguishability and connectivity only; no policy or estimator outcome selection')
    return audit_case(case)


def new_id_case(family,index,split,seed_base):
    case=make_case(family,index,split,seed_base)
    case.update(id='e45f_'+case['id'],suite='benchmark',density='training_range',source_family=family)
    return audit_case(case)
