"""Independent geometry-only acceptance. No environment/controller imports.

Rectangle intersection uses orientation/edge tests, independent of the
production sensor's slab algorithm. Reachability is an outer-set proof,
not a sampled-controller test or a finite grid approximation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

EPS = 2e-7


def cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def on_segment(a, b, q):
    return abs(cross(a,b,q)) <= EPS and all(min(a[i],b[i])-EPS <= q[i] <= max(a[i],b[i])+EPS for i in (0,1))


def intersects(a,b,c,d):
    v = cross(a,b,c),cross(a,b,d),cross(c,d,a),cross(c,d,b)
    if ((v[0] > EPS and v[1] < -EPS) or (v[0] < -EPS and v[1] > EPS)) and ((v[2] > EPS and v[3] < -EPS) or (v[2] < -EPS and v[3] > EPS)):
        return True
    return on_segment(a,b,c) or on_segment(a,b,d) or on_segment(c,d,a) or on_segment(c,d,b)


def inside(q,box):
    return all(box[i][0]-EPS <= q[i] <= box[i][1]+EPS for i in (0,1))


def blocked(a,b,boxes):
    for box in boxes:
        if inside(a,box) or inside(b,box):
            return True
        x,y=box
        corners=((x[0],y[0]),(x[1],y[0]),(x[1],y[1]),(x[0],y[1]))
        if any(intersects(a,b,corners[i],corners[(i+1)%4]) for i in range(4)):
            return True
    return False


def visible(p,e,boxes,radius=6.):
    return math.dist(p,e) <= radius+EPS and not blocked(p,e,boxes)


def legal(q,boxes):
    return all(-10-EPS <= x <= 10+EPS for x in q) and not any(inside(q,b) for b in boxes)


def movement(a,b,limit,boxes):
    return legal(a,boxes) and legal(b,boxes) and max(abs(b[i]-a[i]) for i in (0,1)) <= limit+EPS and not blocked(a,b,boxes)


def untransform(q,frame):
    x,y=q[0]-frame['translation'][0],q[1]-frame['translation'][1]
    for _ in range((-frame['quarter_turns'])%4):
        x,y=-y,x
    return [x,y]


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


def saved_state_consistent(s,previous=None):
    values=s.get('full_state',[])
    if len(values)!=53 or not all(math.isfinite(x) for x in values):return False
    if math.dist(values[:2],s['evader'])>EPS or math.dist(values[4:6],s['pursuer'])>EPS or abs(values[8]-s['step'])>EPS or values[9]!=1 or values[11]!=1:return False
    if previous is not None:
        action=s.get('previous_joint_action')
        if action is None or len(action)!=8:return False
        if any(abs(s['evader'][i]-previous['evader'][i]-action[i])>EPS or abs(s['pursuer'][i]-previous['pursuer'][i]-action[i+4])>EPS for i in (0,1)):return False
        if s['step']!=previous['step']+1:return False
    return True


def geometry_tests():
    box=[[[-1,1],[-1,1]]]
    cases=[('cross',blocked([-2,0],[2,0],box),True),('clear',blocked([-2,2],[2,2],box),False),
           ('edge_touch',blocked([-2,1],[2,1],box),True),('corner_touch',blocked([-2,0],[0,2],box),True),
           ('parallel',blocked([-2,-2],[-2,2],box),False),('within',blocked([0,0],[.5,0],box),True),
           ('range_in',visible([0,0],[6,0],[]),True),('range_out',visible([0,0],[6.0001,0],[]),False),
           ('motion_limit',movement([-2,2],[0,2],1,box),False),('swept_cross',movement([-2,0],[2,0],4,box),False)]
    return dict(passed=all(actual==expected for _,actual,expected in cases),cases=[dict(name=n,actual=a,expected=e) for n,a,e in cases])


def verify_case(environment,public):
    reasons=[]
    boxes=public['map'];vp=public['motion']['pursuer_axis_limit'];ve=public['motion']['evader_axis_limit']
    frame=environment['frame'];h=environment['barrier_half_height'];H=environment['required_blind_steps']
    paths=environment['worlds'];history=public['history'];visibility_rows=[];reacquisition=[];reachability=[];decision_checks=[]
    if len(paths)!=2:
        reasons.append('two_world_witnesses_required')
    if not history or not any(obs['target'] is not None for obs in history):
        reasons.append('missing_real_positive_observation_history')
    canonical_histories=[]
    initial_visibility=[]
    barrier=environment['barrier_index']
    b=boxes[barrier]
    base_corners=[untransform([x,y],frame) for x in b[0] for y in b[1]]
    if min(q[0] for q in base_corners)>-1+EPS or max(q[0] for q in base_corners)<1-EPS or min(q[1] for q in base_corners)>-h+EPS or max(q[1] for q in base_corners)<h-EPS:
        reasons.append('declared_separator_not_present_in_map')
    for world in paths:
        states=world['history_states'];observations=[]
        if len(states)!=len(history):
            reasons.append('history_length_mismatch');continue
        for i,s in enumerate(states):
            if not saved_state_consistent(s,states[i-1] if i else None):reasons.append('original_environment_state_or_action_record_mismatch')
            p,e=s['pursuer'],s['evader'];seen=visible(p,e,boxes)
            observations.append(dict(step=i,position=p,target=e if seen else None,remaining=80-i))
            if not legal(p,boxes) or not legal(e,boxes):reasons.append('illegal_history_position')
            if math.dist(p,e)<1-EPS:reasons.append('capture_during_prescribed_history')
            if i and (not movement(states[i-1]['pursuer'],p,vp,boxes) or not movement(states[i-1]['evader'],e,ve,boxes)):
                reasons.append('history_action_or_swept_collision_violation')
            if history[i]['step']!=i or history[i]['remaining']!=80-i or math.dist(p,history[i]['position'])>EPS or (history[i]['target'] is None)!= (not seen) or (seen and math.dist(e,history[i]['target'])>EPS):
                reasons.append('independent_sensor_history_mismatch')
            visibility_rows.append(dict(world=world['id'],step=i,visible=seen,distance=math.dist(p,e),blocked=blocked(p,e,boxes)))
        canonical_histories.append(canonical(observations))
        flags=[o['target'] is not None for o in observations]
        first=next((i for i,x in enumerate(flags) if not x),None)
        tail=0
        for x in flags[::-1]:
            if x:break
            tail+=1
        if first is None or tail<4:reasons.append('less_than_four_observed_hidden_states')
        if flags and not flags[0]:reasons.append('history_starts_hidden')
        initial_visibility.append(dict(world=world['id'],first_invisible_step=first,hidden_history_steps=tail,last_positive_step=max((i for i,x in enumerate(flags) if x),default=None)))
        if environment['family']=='wall_corner':
            aligned=[untransform(s['evader'],frame) for s in states]
            negotiated=any(q[0]>3 and h-1.4<q[1]<h for q in aligned) and any(q[0]<3 and q[1]<h-1.4 for q in aligned)
            if not negotiated:reasons.append('history_does_not_negotiate_added_corner')
        # All dynamically reachable positions remain separated by the solid
        # rectangle: neither robot can reach either wall end before H.
        p0=untransform(states[-1]['pursuer'],frame);e0=untransform(states[-1]['evader'],frame)
        for k in range(H+1):
            p_y=[p0[1]-vp*k,p0[1]+vp*k];e_y=[e0[1]-ve*k,e0[1]+ve*k]
            margin=h-max(abs(p_y[0]),abs(p_y[1]),abs(e_y[0]),abs(e_y[1]))
            if p0[0]>=-1-EPS or e0[0]<=1+EPS or margin<=EPS:
                reasons.append('cannot_certify_required_controller_independent_blind_window')
            reachability.append(dict(world=world['id'],after_handoff_step=k,frame='separator_aligned',pursuer_outer_box=[[p0[0]-vp*k,min(-1.,p0[0]+vp*k)],p_y],evader_outer_box=[[max(1.,e0[0]-ve*k),e0[0]+ve*k],e_y],blocking_margin=margin,proof='axis_displacement_bound + no_legal_crossing_without_reaching_wall_end + convex_line_height'))
        for route_name,route in world['continuations'].items():
            if not route or math.dist(route[0]['pursuer'],states[-1]['pursuer'])>EPS or math.dist(route[0]['evader'],states[-1]['evader'])>EPS:
                reasons.append('route_not_connected_to_history');continue
            visible_at=[]
            for k,s in enumerate(route):
                if not saved_state_consistent(s,route[k-1] if k else None):reasons.append('continuation_environment_record_mismatch')
                if k and (not movement(route[k-1]['pursuer'],s['pursuer'],vp,boxes) or not movement(route[k-1]['evader'],s['evader'],ve,boxes)):
                    reasons.append('continuation_action_or_collision_violation')
                if math.dist(s['pursuer'],s['evader'])<1-EPS:reasons.append('capture_during_visibility_witness')
                seen=visible(s['pursuer'],s['evader'],boxes)
                if seen:visible_at.append(k)
                if k<=H and seen:reasons.append('route_reveals_before_guaranteed_window')
                visibility_rows.append(dict(world=world['id'],route=route_name,step=len(history)-1+k,after_handoff=k,visible=seen,distance=math.dist(s['pursuer'],s['evader']),blocked=blocked(s['pursuer'],s['evader'],boxes)))
            if not visible_at:reasons.append('no_feasible_reacquisition_witness')
            if environment['family']=='multi_route':
                start=untransform(route[0]['evader'],frame);end=untransform(route[-1]['evader'],frame)
                if not(start[0]<3.6 and end[0]>4.6):reasons.append('route_does_not_negotiate_second_barrier')
            reacquisition.append(dict(world=world['id'],route=route_name,first_visible_after_handoff=min(visible_at) if visible_at else None,pursuer_move_steps=sum(math.dist(a['pursuer'],b['pursuer'])>EPS for a,b in zip(route,route[1:])),evader_move_steps=sum(math.dist(a['evader'],b['evader'])>EPS for a,b in zip(route,route[1:])),end_distance=math.dist(route[-1]['pursuer'],route[-1]['evader']),end_relative=untransform(route[-1]['evader'],frame)))
        # Two continuations must lead to opposite wall ends and require
        # materially different initial pursuit headings, with comparable cost.
        if set(world['continuations'])!={'upper','lower'}:
            reasons.append('two_distinct_exit_routes_required')
        else:
            upper,lower=[world['continuations'][q] for q in ('upper','lower')]
            end_a,end_b=[untransform(r[-1]['evader'],frame) for r in (upper,lower)]
            if not(end_a[1]>h and end_b[1]<-h):reasons.append('routes_do_not_reach_opposite_exits')
            v=[[r[1]['pursuer'][i]-r[0]['pursuer'][i] for i in (0,1)] for r in (upper,lower)]
            den=math.hypot(*v[0])*math.hypot(*v[1]);angle=math.degrees(math.acos(max(-1,min(1,sum(a*b for a,b in zip(*v))/den)))) if den else 0
            if angle<60:reasons.append('initial_pursuer_decisions_not_distinct')
            wrong_upper=visible(upper[-1]['pursuer'],lower[-1]['evader'],boxes)
            wrong_lower=visible(lower[-1]['pursuer'],upper[-1]['evader'],boxes)
            if wrong_upper or wrong_lower:reasons.append('opposite_observer_locations_do_not_distinguish_routes')
            decision_checks.append(dict(world=world['id'],first_action_angle_degrees=angle,upper_action=v[0],lower_action=v[1],upper_observer_sees_lower_exit=wrong_upper,lower_observer_sees_upper_exit=wrong_lower))
            a,b=[x for x in reacquisition if x['world']==world['id']][-2:]
            if abs(a['evader_move_steps']-b['evader_move_steps'])>2 or abs(a['pursuer_move_steps']-b['pursuer_move_steps'])>2:
                reasons.append('exit_route_costs_not_comparable')
    if len(canonical_histories)==2 and canonical_histories[0]!=canonical_histories[1]:
        reasons.append('hidden_worlds_do_not_share_identical_observation_history')
    if environment['handoff_step']!=len(history)-1 or environment['handoff_step']+max((len(r) for w in paths for r in w['continuations'].values()),default=0)>=80:
        reasons.append('history_or_routes_exceed_original_time_horizon')
    forbidden={'evader_velocity','hidden_state','initial_state','history_states','worlds','branch','true_position','particles'}
    def keys(v):
        if isinstance(v,dict):
            return set(v)|set().union(*(keys(x) for x in v.values()))
        if isinstance(v,list):return set().union(*(keys(x) for x in v)) if v else set()
        return set()
    if keys(public)&forbidden:reasons.append('private_truth_in_controller_packet')
    return dict(id=environment['id'],family=environment['family'],required_blind_steps=H,passed=not reasons,reasons=sorted(set(reasons)),history_timing=initial_visibility,controller_history_equal=len(canonical_histories)==2 and canonical_histories[0]==canonical_histories[1],public_packet_sha256=hashlib.sha256(canonical(public)).hexdigest(),reachable_outer_sets=reachability,reacquisition_witnesses=reacquisition,decision_checks=decision_checks,visibility_timeline=visibility_rows,proof_scope='all_legal_controls_of_both_robots_for_H_steps_from_handoff; no_controller_sampling')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();manifest=json.loads(args.manifest.read_text());records=[]
    for entry in manifest['entries']:
        env=json.loads((args.manifest.parent/entry['environment']).read_text());public=json.loads((args.manifest.parent/entry['public']).read_text())
        records.append(verify_case(env,public))
    result=dict(passed=geometry_tests()['passed'] and all(r['passed'] for r in records),geometry_tests=geometry_tests(),cases=records)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],cases=len(records),accepted=sum(r['passed'] for r in records))))


if __name__=='__main__':main()
