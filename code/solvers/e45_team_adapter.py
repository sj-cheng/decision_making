"""E45-A public 2v2 adapter and strong assignment/transfer baselines.

Frozen D kernels are reused per target. Team negative observations are an
explicit new wrapper intervention, not claimed to be the unchanged 1v1 D.
Factorized NTE is a transfer baseline, NOT a joint-action search algorithm.
"""
import hashlib
import itertools
import time
import numpy as np
from numba import njit
from solvers.frozen_d_search_interface import loader,PublicMap,Observation,sensor_visible,geometric_candidates,move,route
from solvers.e44_branch_nte import Controller as NTE
from l45_corner_shared_20261025.corner_estimator import export_regions
from solvers.e50_lidar_map import lidar_target_visibility
from solvers.e50_camera import wrap_angle, segment_clear


@njit(cache=True)
def team_hidden_mask(points,observers,active,boxes):
    allowed=np.ones(len(points),np.bool_)
    for i in range(len(points)):
        for j in range(2):
            if active[j] and sensor_visible(observers[j],points[i],boxes):
                allowed[i]=False
    return allowed


def team_hidden_mask_lidar(points, observers, active, boxes, num_rays=256,
                           target_radius=.25):
    """Negative target observation mask using discrete LiDAR returns."""
    points = np.asarray(points, dtype=float)
    observers = np.asarray(observers, dtype=float)
    active = np.asarray(active, dtype=bool)
    if not len(points):
        return np.ones(0, dtype=bool)
    seen = lidar_target_visibility(observers[active], points, boxes,
                                   num_rays=int(num_rays),
                                   target_radius=float(target_radius))
    allowed = np.ones(len(points), dtype=bool)
    if np.any(active):
        allowed[np.any(seen, axis=0)] = False
    return allowed


def team_hidden_mask_camera(points, observers, active, boxes, camera=None):
    """Negative camera observation only removes particles in the current FOV."""
    points = np.asarray(points, float).reshape(-1, 2)
    observers = np.asarray(observers, float).reshape(-1, 2)
    active = np.asarray(active, bool)
    if camera is None:
        return np.ones(len(points), dtype=bool)
    camera = dict(camera or {})
    yaw = np.asarray(camera.get('gimbal_yaw', np.zeros(len(observers))), float)
    heading = np.asarray(camera.get('headings', np.zeros(len(observers))), float)
    fov = float(camera.get('fov_rad', np.deg2rad(60.)))
    allowed = np.ones(len(points), dtype=bool)
    if fov <= 0.:
        return allowed
    for q in range(len(points)):
        for i in np.flatnonzero(active):
            delta = points[q] - observers[i]
            distance = float(np.linalg.norm(delta))
            bearing = wrap_angle(np.arctan2(delta[1], delta[0]) - heading[i])
            if distance > float(camera.get('max_range', 20.)):
                continue
            if abs(wrap_angle(bearing - yaw[i])) <= fov / 2. and segment_clear(observers[i], points[q], boxes):
                allowed[q] = False
                break
    return allowed


def public_observation(step,positions,targets,active_p,active_e,boxes,visible=None,
                       reports_override=None, visibility_model=None, camera_state=None):
    """Environment-side sensor; only this output is passed into a controller.

    ``visible`` is an optional sensor-produced boolean matrix indexed by
    pursuer and target.  It lets a deployment adapter replace the legacy
    map-line-of-sight test with a LiDAR detector without changing the public
    report format consumed by the controllers.
    """
    if visible is not None:
        visible = np.asarray(visible, dtype=bool)
        if visible.shape != (len(positions), len(targets)):
            raise ValueError('visible must have shape (pursuers, targets)')
    if reports_override is not None:
        reports = reports_override
    else:
        reports=[]
        for j in range(2):
            reports.append([targets[j].tolist() if active_e[j] and active_p[i]
                and (bool(visible[i, j]) if visible is not None
                     else sensor_visible(positions[i],targets[j],boxes)) else None for i in range(2)])
    result = dict(step=int(step),remaining=80-int(step),positions=positions.tolist(),
                  active_p=active_p.astype(bool).tolist(),
                  active_e=active_e.astype(bool).tolist(),reports=reports)
    if visibility_model is not None:
        result['visibility_model'] = str(visibility_model)
    elif visible is not None:
        result['visibility_model'] = 'lidar'
    if camera_state is not None:
        result['camera'] = camera_state
    return result


class TeamBelief:
    def __init__(self,boxes,seed):
        self.boxes=np.asarray(boxes,float)
        self.estimators=[loader.load_estimator(boxes,seed+1009*j) for j in range(2)]
        self.known=[None,None];self.step=-1;self.last_stats={}

    def update(self,obs):
        if self.step>=0 and obs['step']!=self.step+1:raise ValueError('Nonconsecutive team history')
        if obs['remaining']!=80-obs['step']:raise ValueError('Bad team clock')
        positions=np.asarray(obs['positions']);ap=np.asarray(obs['active_p']);ae=obs['active_e']
        live=np.flatnonzero(ap)
        if not len(live):raise ValueError('No active observer')
        masses=[]
        for j,e in enumerate(self.estimators):
            if not ae[j]:self.known[j]=None;continue
            positive=[i for i in live if obs['reports'][j][i] is not None]
            if positive:
                at=int(positive[0])
                measurements = [np.asarray(obs['reports'][j][i], float) for i in positive]
                if obs.get('visibility_model') == 'camera_fov':
                    # Independent camera range/bearing estimates need not be
                    # bit-identical.  Fuse the public estimates while keeping
                    # the target slot identity stable for the belief update.
                    target = np.mean(np.asarray(measurements), axis=0)
                else:
                    target = measurements[0]
                    for measurement in measurements:
                        if not np.array_equal(target, measurement):
                            raise ValueError('Conflicting identified measurements')
                for i in positive:
                    if obs.get('visibility_model') not in ('lidar', 'camera_fov') and not sensor_visible(
                            positions[i], measurements[positive.index(i)], self.boxes):
                        raise ValueError('Invalid team measurement')
                self.known[j]=target.copy()
            else:
                at=j if ap[j] else int(live[0]);target=None;self.known[j]=None
            # Exactly ONE transition per physical step, then multiply all
            # simultaneous negative sensor likelihoods without moving time.
            e.update(Observation(obs['step'],positions[at],target,obs['remaining']))
            if target is None:
                if obs.get('visibility_model') == 'lidar':
                    allowed=team_hidden_mask_lidar(e.graph[0],positions,ap,self.boxes)
                elif obs.get('visibility_model') == 'camera_fov':
                    allowed=team_hidden_mask_camera(e.graph[0],positions,ap,self.boxes,obs.get('camera'))
                else:
                    allowed=team_hidden_mask(e.graph[0],positions,ap,self.boxes)
                e.weights[~allowed]=0.;e.reachable &= allowed
                mass=float(e.weights.sum());masses.append(mass)
                if mass<=0 or not np.isfinite(mass):raise RuntimeError('Empty team-conditioned finite-grid support')
                e.weights/=mass
                e.ids,e.particle_weights,e.labels=export_regions(e.weights,e.regions,e.rng,64)
                e.particles=e.graph[0][e.ids].copy()
                if obs.get('visibility_model') == 'lidar':
                    valid_hidden = team_hidden_mask_lidar(e.particles,positions,ap,self.boxes)
                elif obs.get('visibility_model') == 'camera_fov':
                    valid_hidden = team_hidden_mask_camera(e.particles,positions,ap,self.boxes,obs.get('camera'))
                else:
                    valid_hidden = team_hidden_mask(e.particles,positions,ap,self.boxes)
                if obs.get('visibility_model') not in ('lidar', 'camera_fov') and not valid_hidden.all():
                    raise RuntimeError('Export violates team observation')
        self.step=obs['step'];self.last_stats=dict(min_conditioning_mass=min(masses,default=1.))

    def signature(self):
        return hashlib.sha256(b''.join(e.particles.tobytes()+e.particle_weights.tobytes()+e.rng.tobytes()
            for e in self.estimators if e.particles is not None)).hexdigest()


class CurrentEstimate:
    """Read the already team-updated estimator without a second time update."""
    def __init__(self,source):self.source=source
    def __getattr__(self,name):return getattr(self.source,name)
    def update(self,obs):
        if obs.step!=self.source.step:raise ValueError('NTE using stale team belief')
        return self.source.particles.copy(),self.source.particle_weights.copy()


class AnchoredTransferNTE(NTE):
    def propose(self,node,direct=False):
        # Preserve an executable geometric option; the NN adds alternatives.
        if not node.actions and not direct:return self.rollout_action(node)
        return super().propose(node,direct)


def optimal_assignment(cost,active_p,active_e,independent=False):
    pp=list(np.flatnonzero(active_p));ee=list(np.flatnonzero(active_e));answer={}
    if not pp or not ee:return answer
    if independent:
        return {i:min(ee,key=lambda j:(cost[i,j],j)) for i in pp}
    if len(pp)!=len(ee):raise ValueError('E45-A preserves paired retirement; counts must match')
    permutation=min(itertools.permutations(ee),key=lambda order:
        (max(cost[i,j] for i,j in zip(pp,order)),sum(cost[i,j] for i,j in zip(pp,order)),order))
    return dict(zip(pp,permutation))


class TeamController:
    def __init__(self,boxes,speed,seed,method,checkpoint=None,budget_ms=200.):
        self.public=PublicMap(boxes,speed);self.belief=TeamBelief(boxes,seed)
        self.motion_boxes=self.public.boxes.copy();self.motion_boxes[:,:,0]-=1e-6;self.motion_boxes[:,:,1]+=1e-6
        self.method=method;self.budget_ms=budget_ms;self.last_stats={};self.models={}
        if method.startswith('AFN'):
            if checkpoint is None:raise ValueError('Transfer checkpoint required')
            for i in range(2):
                for j in range(2):
                    c=AnchoredTransferNTE(boxes,speed,seed+7919*i+1009*j,'neural','focused',checkpoint)
                    c.estimator=CurrentEstimate(self.belief.estimators[j]);self.models[i,j]=c
        self.records={pair:[] for pair in itertools.product(range(2),repeat=2)}

    def observe(self,obs):
        self.belief.update(obs)
        for i,j in self.records:
            q=self.belief.known[j]
            self.records[i,j].append(dict(step=obs['step'],remaining=obs['remaining'],
                position=obs['positions'][i],target=None if q is None else q.tolist()))
        for pair,c in self.models.items():c.public_records=list(self.records[pair])

    def geometry(self,obs):
        p=self.public;positions=np.asarray(obs['positions']);cost=np.full((2,2),1e9);actions=np.zeros((2,2,2))
        for i in np.flatnonzero(obs['active_p']):
            for j in np.flatnonzero(obs['active_e']):
                e=self.belief.estimators[j];q=self.belief.known[j]
                if q is not None:
                    a=move(positions[i],q,p.speed,p.vertices,p.dist,p.boxes,np.zeros(2,np.int64))
                    d=route(positions[i],q,p.speed,p.vertices,p.dist,p.boxes,np.zeros(2,np.int64))[0]
                else:
                    _,a=geometric_candidates(positions[i],e.particles,e.particle_weights,e.regions[e.ids],
                        p.speed,p.vertices,p.dist,p.boxes,self.motion_boxes)
                    d=sum(w*route(positions[i],x,p.speed,p.vertices,p.dist,p.boxes,np.zeros(2,np.int64))[0]
                        for w,x in zip(e.particle_weights,e.particles))
                actions[i,j]=a;cost[i,j]=d
        return cost,actions

    def act(self,obs,fixed_simulations=None):
        started=time.perf_counter();self.observe(obs)
        cost,options=self.geometry(obs)
        assignment=optimal_assignment(cost,obs['active_p'],obs['active_e'],independent=self.method=='IG')
        action=np.zeros((2,2));searches=[];pairs=list(assignment.items())
        for index,(i,j) in enumerate(pairs):
            action[i]=options[i,j]
            if self.models:
                c=self.models[i,j];left=self.budget_ms-1000*(time.perf_counter()-started)-2.
                if fixed_simulations is not None or left>5.:
                    c.budget_ms=max(0.,left/(len(pairs)-index))
                    result=c.act(Observation(obs['step'],np.asarray(obs['positions'][i]),self.belief.known[j],obs['remaining']),fixed_simulations)
                    if self.belief.known[j] is not None or c.last_stats['simulations']>0:action[i]=result
                    searches.append(dict(pursuer=int(i),target=int(j),**c.last_stats))
        self.last_stats=dict(total_ms=1000*(time.perf_counter()-started),assignment={str(i):int(j) for i,j in assignment.items()},
            pair_costs=cost.tolist(),searches=searches,team_belief=self.belief.last_stats,
            factorized_search=bool(self.models),joint_action_search=False)
        return action
