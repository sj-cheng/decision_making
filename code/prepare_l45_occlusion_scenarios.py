"""Construct and certify scripted hard scenes; never evaluate L/G/T/V.

Only original environment propagation and original local sensing are used.
The separate verifier imports neither and proves the blind interval from
continuous outer reachable sets rather than sampled controller behavior.
"""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
from problems.problem import get_problem
from solvers.local_pursuit import PublicMap

ROOT=Path(__file__).resolve().parent.parent
DEFAULT=ROOT/'current/l45_occlusion_scenarios_20261025'
FAMILIES=('long_wall','wall_corner','multi_route')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def transform(q,frame):
    x,y=q
    for _ in range(frame['quarter_turns']):x,y=-y,x
    return np.array([x,y])+frame['translation']


def transformed_boxes(boxes,frame):
    result=[]
    for b in boxes:
        pts=np.array([transform([x,y],frame) for x in b[0] for y in b[1]])
        result.append(np.stack((pts.min(0),pts.max(0)),axis=1).tolist())
    return result


def original_model(boxes,speed):
    p=get_problem('example8');p.obstacles=[np.array(b,float) for b in boxes]
    p.use_minco_dynamics=False;p.swept_collision=True
    p.current_evader_speed_lim=1.;p.current_pursuer_speed_lim=speed
    for robot in range(4):
        cap=1. if robot<2 else speed
        p.action_lims[p.action_idxs[robot]]=[[-cap,cap],[-cap,cap]]
        p.control_lims[p.action_idxs[robot]]=p.action_lims[p.action_idxs[robot]]/p.dt
    p.clear_closed_loop_cache()
    assert p.tf==80 and p.dt==1 and p.desired_distance==1
    return p


def position_state(p,state,action=None):
    return dict(pursuer=state[p.state_idxs[2],0].tolist(),evader=state[p.state_idxs[0],0].tolist(),
        full_state=state[:,0].tolist(),step=int(state[p.time_idx,0]),
        previous_joint_action=None if action is None else action[:,0].tolist())


def advance(p,state,target_p,target_e):
    action=np.zeros((p.action_dim,1))
    action[p.action_idxs[2],0]=target_p-state[p.state_idxs[2],0]
    action[p.action_idxs[0],0]=target_e-state[p.state_idxs[0],0]
    if np.max(np.abs(action[p.action_idxs[2],0]))>p.current_pursuer_speed_lim+1e-8 or np.max(np.abs(action[:2,0]))>1+1e-8:
        raise ValueError('Script exceeds fixed original action bounds')
    nxt=p.step(state,action,p.dt)
    if not p.is_valid(nxt) or not p.is_active(nxt,0) or not p.is_active(nxt,2):
        raise ValueError('Script collided, captured, or invalidated original environment')
    return nxt,position_state(p,nxt,action)


def toward(at,goal,speed):
    delta=np.array(goal)-at
    return at+delta/max(1.,float(np.max(np.abs(delta)))/speed)


def construct(family,H,speed,seed,variant=0,short_wall=False):
    rng=np.random.default_rng(seed)
    frame=dict(quarter_turns=int((variant+2*int(rng.integers(2)))%4),translation=rng.uniform(-.10,.10,2).tolist())
    h=1. if short_wall else (6. if H==4 else 8.)
    base=[[[-1.,1.],[-h,h]]]
    if family=='wall_corner':base.append([[1.,3.],[h-1.4,h]])
    if family=='multi_route':base.append([[3.6,4.6],[-h+2,h-2]])
    boxes=transformed_boxes(base,frame)
    identifier=f'{family}_H{H}_v{speed:.1f}_{seed}'
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


def store_case(folder,env,public):
    ident=env['id'];private=Path('environment')/(ident+'.json');packet=Path('controller_inputs')/(ident+'.json')
    put(folder/private,env);put(folder/packet,public)
    return dict(id=ident,environment=str(private),public=str(packet))


def verify(folder,entries):
    put(folder/'manifest.json',dict(entries=entries))
    subprocess.run([sys.executable,str(ROOT/'code/verify_l45_occlusion_scenarios.py'),str(folder/'manifest.json'),'--output',str(folder/'acceptance.json')],check=True)
    return json.loads((folder/'acceptance.json').read_text())


def render(folder,entries,acceptance):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle
    folder.joinpath('figures').mkdir(exist_ok=True)
    with PdfPages(folder/'figures/scenarios.pdf') as pdf:
        for entry,cert in zip(entries,acceptance['cases']):
            env=json.loads((folder/entry['environment']).read_text());pub=json.loads((folder/entry['public']).read_text())
            fig,axes=plt.subplots(1,3,figsize=(14,4.5))
            for ax in axes[:2]:
                for b in pub['map']:ax.add_patch(Rectangle((b[0][0],b[1][0]),b[0][1]-b[0][0],b[1][1]-b[1][0],facecolor='.65'))
                ax.set(xlim=(-10,10),ylim=(-10,10),aspect='equal')
            ax=axes[0]
            for wi,world in enumerate(env['worlds']):
                path=world['history_states'];p=np.array([s['pursuer'] for s in path]);e=np.array([s['evader'] for s in path])
                if wi==0:ax.plot(*p.T,c='blue',label='scripted pursuer');ax.scatter(*p[-1],c='blue',marker='x')
                ax.plot(*e.T,c=('red' if wi==0 else 'orange'),ls=('-' if wi==0 else '--'),label='hidden world '+world['id'])
            ax.set_title('Real visible history -> hidden handoff');ax.legend(fontsize=7)
            ax=axes[1]
            world=env['worlds'][0]
            for name,color in (('upper','purple'),('lower','green')):
                path=world['continuations'][name];p=np.array([s['pursuer'] for s in path]);e=np.array([s['evader'] for s in path])
                ax.plot(*p.T,c=color,label=name+' observer');ax.plot(*e.T,c=color,ls='--',label=name+' escape')
            ax.set_title(f'Opposite exits; guaranteed blind H={env["required_blind_steps"]}');ax.legend(fontsize=7)
            ax=axes[2]
            for world in ('A','B'):
                z=[r for r in cert['visibility_timeline'] if r['world']==world and 'route' not in r]
                ax.step([r['step'] for r in z],[int(r['visible']) for r in z],where='post',label='world '+world)
            for name in ('upper','lower'):
                z=[r for r in cert['visibility_timeline'] if r['world']=='A' and r.get('route')==name]
                ax.step([r['step'] for r in z],[int(r['visible']) for r in z],where='post',ls='--',label='witness '+name)
            ax.axvline(env['handoff_step'],c='k',ls=':');ax.set(ylim=(-.1,1.1),yticks=(0,1),xlabel='Original environment step',ylabel='Visible');ax.legend(fontsize=7)
            fig.suptitle(env['id']+' | '+('ACCEPTED' if cert['passed'] else 'REJECTED'),fontsize=10);fig.tight_layout();pdf.savefig(fig)
            if len(entries)<=6:fig.savefig(folder/'figures'/(env['id']+'.png'),dpi=150)
            plt.close(fig)


def all_stages(output):
    if (output/'frozen.json').exists():
        lock=json.loads((output/'frozen.json').read_text())
        if not all(sha(output/name)==value for name,value in lock['files'].items()):raise ValueError('Frozen scene artifact changed')
        print('Existing frozen bank verified; no regeneration');return
    output.mkdir(parents=True,exist_ok=True)
    before={str(path):sha(path) for path in (ROOT/'code/solvers/local_pursuit.py',ROOT/'code/run_l45_local_pursuit.py',ROOT/'code/problems/example8.py',ROOT/'current/l45_local_pursuit_20261024/results.json')}
    pilot=output/'pilot';entries=[]
    for fi,family in enumerate(FAMILIES):
        for H in (4,6):
            seed=int(np.random.SeedSequence([61020261025,0,fi,H]).generate_state(1)[0]);env,public=construct(family,H,1.3,seed)
            entries.append(store_case(pilot,env,public))
    acceptance=verify(pilot,entries);render(pilot,entries,acceptance)
    if not acceptance['passed']:raise RuntimeError('Pilot geometry acceptance failed; bank generation stopped')
    # A real short-wall proposal and two malformed-input negative controls.
    rejected=output/'rejected';negative=[]
    env,public=construct('long_wall',4,1.3,6102501,short_wall=True)
    env['id']='short_wall_design_proposal';negative.append(store_case(rejected,env,public))
    sample_env=json.loads((pilot/entries[0]['environment']).read_text());sample_pub=json.loads((pilot/entries[0]['public']).read_text())
    env=copy.deepcopy(sample_env);public=copy.deepcopy(sample_pub);env['id']='negative_control_missing_positive_history'
    public['history']=[dict(o,target=None) for o in public['history']];negative.append(store_case(rejected,env,public))
    env=copy.deepcopy(sample_env);public=copy.deepcopy(sample_pub);env['id']='negative_control_same_exit_twice'
    for world in env['worlds']:world['continuations']['lower']=copy.deepcopy(world['continuations']['upper'])
    negative.append(store_case(rejected,env,public));negative_result=verify(rejected,negative)
    if any(r['passed'] for r in negative_result['cases']):raise RuntimeError('A negative acceptance case was not rejected')
    bank=output/'bank';entries=[]
    for fi,family in enumerate(FAMILIES):
        for H in (4,6):
            for si,speed in enumerate((1.1,1.3)):
                for variant in (0,1):
                    seed=int(np.random.SeedSequence([61020261025,1,fi,H,si,variant]).generate_state(1)[0]);env,public=construct(family,H,speed,seed,variant)
                    entries.append(store_case(bank,env,public))
    acceptance=verify(bank,entries);render(bank,entries,acceptance)
    if not acceptance['passed']:raise RuntimeError('Frozen bank has rejected geometry; stop without evaluation')
    ids=[]
    for entry in entries:
        env=json.loads((bank/entry['environment']).read_text())
        for world in env['worlds']:
            ids.append(hashlib.sha256(json.dumps(dict(map=json.loads((bank/entry['public']).read_text())['map'],p=world['history_states'][-1]['pursuer'],e=world['history_states'][-1]['evader'])).encode()).hexdigest())
    if len(set(ids))!=len(ids):raise RuntimeError('Duplicate bank physical handoff states')
    if any(sha(path)!=value for path,value in before.items()):raise RuntimeError('An original algorithm or old result changed')
    put(output/'provenance.json',dict(original_files_unchanged=before,preparation_sources={str(Path(__file__)):sha(__file__),str(ROOT/'code/verify_l45_occlusion_scenarios.py'):sha(ROOT/'code/verify_l45_occlusion_scenarios.py')},
        generation_seed_namespace=61020261025,pilot_history_groups=6,bank_history_groups=24,bank_hidden_worlds=48,all_public_histories_shared_within_world_pairs=True,
        training=False,algorithm_evaluation=False,controllers_called=False,environment_only_scripted_propagation=True))
    put(output/'status.json',dict(stage='awaiting_user_confirmation',geometry_acceptance_passed=True,bank_history_groups=24,hidden_worlds=48,formal_evaluation_started=False))
    files={str(path.relative_to(output)):sha(path) for path in output.rglob('*.json') if path.name!='frozen.json'}
    put(output/'frozen.json',dict(schema=1,scope='scenario_construction_only',files=files,formal_evaluation_authorized=False))
    print(json.dumps(dict(pilot=6,bank=24,hidden_worlds=48,rejected=len(negative_result['cases']),status='awaiting_user_confirmation')))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args();all_stages(args.output.resolve())


if __name__=='__main__':main()
