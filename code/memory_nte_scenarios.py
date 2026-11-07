"""New geometric instances; scripted histories use the original environment.
Constructor adapted from prepare_l45_occlusion_scenarios.construct. Only wall
height becomes a public geometry parameter. No controller is used to select cases.
"""
import math
import numpy as np
from prepare_l45_occlusion_scenarios import (original_model, transform,
    transformed_boxes, position_state, advance, toward)
from solvers.local_pursuit import PublicMap

def construct(family,H,speed,seed,height,variant=0):
    rng=np.random.default_rng(seed)
    frame=dict(quarter_turns=int((variant+2*int(rng.integers(2)))%4),translation=rng.uniform(-.10,.10,2).tolist())
    h=float(height)
    base=[[[-1.,1.],[-h,h]]]
    if family=='wall_corner':base.append([[1.,3.],[h-1.4,h]])
    if family=='multi_route':base.append([[3.6,4.6],[-h+2,h-2]])
    boxes=transformed_boxes(base,frame)
    identifier=f'{family}_H{H}_h{h:.6f}_v{speed:.1f}_{seed}'
    worlds=[];public_history=None
    # Three genuinely visible states precede the common empty-history block.
    starts=[([-3.,h+.8],[2.2,h+.8]),([-2.9,h+.7],[2.3,h+.7]),([-2.8,h+.6],[2.4,h+.6])]
    for branch in ('A','B'):
        p=original_model(boxes,speed);sensor=PublicMap(boxes,speed)
        state=np.zeros((p.state_dim,1))
        state[p.state_idxs[0],0]=transform(starts[0][1],frame)
        state[p.state_idxs[2],0]=transform(starts[0][0],frame)
        state[p.state_idxs[1],0]=[-9.8,-9.8];state[p.state_idxs[3],0]=[9.8,9.8]
        state[p.active_idxs,0]=[1,0,1,0]
        history_states=[position_state(p,state)];initial=state.copy()
        for pp,ee in starts[1:]:
            state,s=advance(p,state,transform(pp,frame),transform(ee,frame));history_states.append(s)
        hidden_count=math.ceil(h+.6)
        for j in range(1,hidden_count+1):
            y=max(0.,h+.6-j)
            ep=[2.4,y]
            if family=='wall_corner':
                ep=[3.4 if j<3 else 2.4,max(0.,h+.7-j)]
            if j==hidden_count:ep=([2.2,.2] if branch=='A' else [2.6,-.2])
            state,s=advance(p,state,transform([-2.8,y],frame),transform(ep,frame));history_states.append(s)
        observations=[]
        for s in history_states:
            obs=sensor.observe(s['step'],np.array(s['pursuer']),np.array(s['evader']))
            observations.append(dict(step=obs.step,position=obs.position.tolist(),target=None if obs.target is None else obs.target.tolist(),remaining=obs.remaining))
        if public_history is not None and observations!=public_history:
            raise ValueError('Prescribed worlds expose different public histories')
        public_history=observations
        handoff=state.copy();routes={}
        for route_name,sign in (('upper',1),('lower',-1)):
            state=handoff.copy();path=[position_state(p,state)]
            observer_goal=transform([-1.05,sign*(h+.4)],frame)
            # Routes from each possible hidden state to both exits are retained.
            inv_x=2.2 if branch=='A' else 2.6
            escape_waypoints=[transform([inv_x,sign*(h+.4)],frame)]
            if family=='wall_corner' and route_name=='upper':
                escape_waypoints=[transform([3.6,sign*(h+.4)],frame)]
            if family=='multi_route':
                escape_waypoints=[transform([3.4,sign*(h-1.8)],frame),transform([4.8,sign*(h-1.8)],frame),transform([4.8,sign*(h+.4)],frame)]
            waypoint=0
            while waypoint<len(escape_waypoints) or np.linalg.norm(state[p.state_idxs[2],0]-observer_goal)>1e-8:
                goal=escape_waypoints[waypoint] if waypoint<len(escape_waypoints) else escape_waypoints[-1]
                state,s=advance(p,state,toward(state[p.state_idxs[2],0],observer_goal,speed),toward(state[p.state_idxs[0],0],goal,1.));path.append(s)
                if waypoint<len(escape_waypoints) and np.linalg.norm(state[p.state_idxs[0],0]-goal)<1e-8:waypoint+=1
                if len(path)>40:raise ValueError('Witness route exceeds fixed construction horizon')
            for _ in range(2):
                state,s=advance(p,state,observer_goal,escape_waypoints[-1]);path.append(s)
            routes[route_name]=path
        worlds.append(dict(id=branch,history_states=history_states,handoff_full_state=handoff[:,0].tolist(),continuations=routes))
    public=dict(protocol='local_range6_occluded_scripted_history_v1',map=boxes,
        motion=dict(pursuer_axis_limit=speed,evader_axis_limit=1.,dt=1.,collision='closed_rectangles_swept_segment',capture_radius=1.,horizon=80),
        sensor=dict(radius=6.,field_of_view_degrees=360,position_noise=0.,velocity_measurement=False,obstacle_occlusion=True),
        history=public_history,controller_seeds=[int(rng.integers(1,2**31-1)) for _ in range(2)])
    env=dict(id=identifier,family=family,seed=seed,frame=frame,barrier_index=0,barrier_half_height=h,required_blind_steps=H,
        handoff_step=len(public_history)-1,worlds=worlds,scope='prescribed_environment_histories_and_geometric_witnesses_only; no_algorithm_evaluation')
    return env,public

