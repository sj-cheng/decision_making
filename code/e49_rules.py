"""Isolated E49 physics and private original RootPUCT binding.

Robot order remains E0,E1,P0,P1. Public joint_transition uses local team
indexes and displacement actions. No collision avoidance is implicit: the
historical swept collision moves an unsafe action just inside its first box.
"""
from functools import lru_cache
from pathlib import Path
import importlib.util
import fcntl
import hashlib
import json
import subprocess
import sys
import types
import numpy as np
from problems.example8 import Example8
from local_observation import segment_interval, first_collision_point

ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / 'current/e49_persistent_pursuit_benchmark_20261108/cpp'


def _legal(point, boxes):
    return bool(np.all(point >= -10.) and np.all(point <= 10.) and
                not any(np.all(point >= b[:, 0]) and np.all(point <= b[:, 1]) for b in boxes))


def capture_pairs(positions, targets, active_pursuers, active_evaders, boxes):
    """Distance priority; ties retain historical pursuer then evader order."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 2, 2)
    candidates = []
    for p in range(2):
        if not active_pursuers[p] or not _legal(positions[p], boxes):
            continue
        for e in range(2):
            if not active_evaders[e] or not _legal(targets[e], boxes):
                continue
            distance = float(np.linalg.norm(positions[p] - targets[e]))
            if distance < 1. and not any(segment_interval(positions[p], targets[e], b) is not None for b in boxes):
                candidates.append((distance, p, e))
    candidates.sort(key=lambda item: item[0])
    used_p, used_e, pairs = set(), set(), []
    for _, p, e in candidates:
        if p not in used_p and e not in used_e:
            used_p.add(p); used_e.add(e); pairs.append((p, e))
    return pairs


def reference_joint_transition(positions, targets, pursuer_actions, evader_actions,
                     active_pursuers, active_evaders, boxes, capture_dt=.01, dt=1.):
    """Return P,E,activeP,activeE,events. Events use relative substep time.

    Capture checks include t=0 and t=dt. A captured target freezes at its
    sampled position; pursuers execute the full action even after double catch.
    Each sample has a fresh one-to-one matching.
    """
    if not (np.isfinite(capture_dt) and capture_dt > 0 and np.isfinite(dt) and dt > 0):
        raise ValueError('positive finite capture_dt and dt required')
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 2, 2)
    p0 = np.asarray(positions, dtype=float).reshape(2, 2)
    e0 = np.asarray(targets, dtype=float).reshape(2, 2)
    pa = np.asarray(pursuer_actions, dtype=float).reshape(2, 2)
    ea = np.asarray(evader_actions, dtype=float).reshape(2, 2)
    if not all(np.isfinite(a).all() for a in (p0,e0,pa,ea)):
        raise ValueError('nonfinite state or action')
    ap = np.asarray(active_pursuers, bool).copy()
    ae = np.asarray(active_evaders, bool).copy()
    pv = np.clip(pa / dt, -1., 1.) * ap[:, None]
    ev = np.clip(ea / dt, -1.5, 1.5) * ae[:, None]
    p, e = p0.copy(), e0.copy()
    events = []
    samples = np.r_[np.arange(0., dt, capture_dt), float(dt)]
    starts = np.concatenate((p0,e0))
    velocities = np.concatenate((pv,ev))
    delta = samples[:,None,None] * velocities[None,:,:]
    paths = starts[None,:,:] + delta
    first = np.full(paths.shape[:2],2.)
    for box in boxes:
        moving = np.abs(delta)>1e-12
        denom = np.where(moving,delta,1.)
        a = (box[:,0]-starts)[None,:,:] / denom
        b = (box[:,1]-starts)[None,:,:] / denom
        lo = np.maximum(0.,np.where(moving,np.minimum(a,b),0.).max(axis=2))
        hi = np.minimum(1.,np.where(moving,np.maximum(a,b),1.).min(axis=2))
        possible = (moving | ((starts>=box[:,0]) & (starts<=box[:,1]))[None,:,:]).all(axis=2)
        hit = possible & (lo<=hi) & (lo<first)
        point = np.clip(starts[None,:,:]+delta*np.minimum(hi,lo+1e-5)[:,:,None],box[:,0],box[:,1])
        paths = np.where(hit[:,:,None],point,paths)
        first = np.where(hit,lo,first)
    legal = ((paths>=-10.) & (paths<=10.)).all(axis=2)
    for box in boxes:
        legal &= ~((paths>=box[:,0]) & (paths<=box[:,1])).all(axis=2)
    distances = np.linalg.norm(paths[:,:2,None,:]-paths[:,None,2:,:],axis=3)
    possible = (distances<1.) & legal[:,:2,None] & legal[:,None,2:]
    possible &= ap[None,:,None] & ae[None,None,:]
    for sample in np.flatnonzero(possible.any(axis=(1,2))):
        p = paths[sample,:2]
        current_e = paths[sample,2:]
        for pi, ei in capture_pairs(p, current_e, ap, ae, boxes):
            ae[ei] = False; e[ei] = current_e[ei]
            events.append(dict(time=float(samples[sample]), pursuer=int(pi), evader=int(ei)))
        if not ae.any(): break
    p = paths[-1,:2].copy()
    e[ae] = paths[-1,2:][ae]
    return p, e, ap, ae, events


class E49Environment(Example8):
    def __init__(self, boxes, horizon=80, capture_dt=.01):
        super().__init__()
        if int(horizon) != horizon or not 40 <= horizon <= 160:
            raise ValueError('E49 horizon must be an integer in [40,160]')
        if not np.isfinite(capture_dt) or capture_dt <= 0 or capture_dt > 1:
            raise ValueError('capture_dt must be in (0,1]')
        self.tf = int(horizon); self.horizon = self.tf
        self.capture_dt = float(capture_dt)
        self.times = np.arange(self.tf + 1)
        self.state_lims[self.time_idx] = [0, self.tf]
        self.obstacles = [np.asarray(b, float).copy() for b in boxes]
        self.use_minco_dynamics = False; self.swept_collision = True
        self.evader_speed_lim_range = (1.5, 1.5)
        self.pursuer_speed_lim_range = (1., 1.)
        self.randomize_speed_limits()
        self.last_capture_events = []
        self.clear_closed_loop_cache()

    def randomize_speed_limits(self):
        self.current_evader_speed_lim = 1.5
        self.current_pursuer_speed_lim = 1.
        for robot in range(4):
            cap = 1.5 if robot in self.evaders else 1.
            self.action_lims[self.action_idxs[robot]] = [[-cap,cap],[-cap,cap]]
            self.control_lims[self.action_idxs[robot]] = self.action_lims[self.action_idxs[robot]] / self.dt
            self.state_lims[self.vel_idxs[robot]] = [[-cap,cap],[-cap,cap]]
        self.update_init_lims()

    def get_capture_pairs(self, state):
        p = np.array([state[self.state_idxs[r],0] for r in self.pursuers])
        e = np.array([state[self.state_idxs[r],0] for r in self.evaders])
        ap = [self.is_active(state,r) for r in self.pursuers]
        ae = [self.is_active(state,r) for r in self.evaders]
        return [(pi+2,ei) for pi,ei in capture_pairs(p,e,ap,ae,self.obstacles)]

    def step(self, s, a, dt=1.):
        s = np.asarray(s, float).reshape(self.state_dim,1)
        a = np.asarray(a, float).reshape(self.action_dim,1)
        if float(s[self.time_idx,0]) >= self.tf:
            self.last_capture_events = []
            return s.copy()
        dt = min(float(dt), self.tf - float(s[self.time_idx,0]))
        p,e,ap,ae,events = joint_transition(
            [s[self.state_idxs[r],0] for r in self.pursuers],
            [s[self.state_idxs[r],0] for r in self.evaders],
            [a[self.action_idxs[r],0] for r in self.pursuers],
            [a[self.action_idxs[r],0] for r in self.evaders],
            [self.is_active(s,r) for r in self.pursuers],
            [self.is_active(s,r) for r in self.evaders],self.obstacles,self.capture_dt,dt)
        nxt = s.copy()
        nxt[:8,0] = np.concatenate((e.ravel(),p.ravel()))
        nxt[self.active_idxs,0] = np.concatenate((ae,ap))
        nxt[13:29,0] = 0.
        nxt[self.time_idx,0] += dt
        self.last_capture_events = [dict(event,time=event['time']+float(s[self.time_idx,0])) for event in events]
        self._update_observation_memory(s,nxt,dt)
        return nxt


def create_environment(boxes, horizon=80, capture_dt=.01):
    return E49Environment(boxes, horizon, capture_dt)


@lru_cache(maxsize=1)
def private_cpp():
    """Build/load only the E49 snapshot; never overwrite historical binaries."""
    source = CPP / 'source'; build = CPP / 'build'
    CPP.mkdir(parents=True,exist_ok=True)
    with (CPP/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        hashes = {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(source.rglob('*')) if p.is_file()}
        receipt_path = CPP/'build_manifest.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        binary = next(build.glob('bindings_e49*.so'), None) if build.exists() else None
        if binary is None or receipt.get('source_hashes') != hashes:
            cmake_dir = subprocess.check_output([sys.executable,'-m','pybind11','--cmakedir'], text=True).strip()
            subprocess.run(['cmake','-S',str(source),'-B',str(build),'-Dpybind11_DIR='+cmake_dir,
                            '-DPYTHON_EXECUTABLE='+sys.executable],check=True)
            subprocess.run(['cmake','--build',str(build),'--parallel','2'],check=True)
            binary = next(build.glob('bindings_e49*.so'))
            receipt_path.write_text(json.dumps(dict(source_hashes=hashes,
                binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                python=sys.executable,base_snapshot='code/cpp/build_root_audit/source',
                model='original RootPUCT C_PUCT_V1; depth100; unmodified original reward weights'),indent=2)+'\n')
        if hashlib.sha256(binary.read_bytes()).hexdigest() != json.loads(receipt_path.read_text())['binary_sha256']:
            raise RuntimeError('E49 private binary hash mismatch')
    spec = importlib.util.spec_from_file_location('bindings_e49',binary)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def make_opponent(env, seed, simulations=3000):
    from solvers.root_audit_puct import RootPUCT
    from solvers.c_puct import C_PUCT
    cpp = private_cpp()
    def bind(method):
        namespace = dict(method.__globals__)
        for key in ('Problem_Settings','Problem_Wrapper','Policy_Network_Wrapper','Value_Network_Wrapper'):
            namespace[key] = getattr(cpp,key)
        return types.FunctionType(method.__code__,namespace,method.__name__,method.__defaults__,method.__closure__)
    class E49RootPUCT(RootPUCT):
        create_cpp_policy_oracle = bind(C_PUCT.create_cpp_policy_oracle)
        create_cpp_value_oracle = bind(C_PUCT.create_cpp_value_oracle)
        _base_wrapper = bind(C_PUCT._problem_wrapper_for)
        def _problem_wrapper_for(self, problem):
            wrapper = self._base_wrapper(problem)
            wrapper.set_capture_dt(problem.capture_dt)
            return wrapper
        def search(self, problem, root_state, turn=0):
            result = cpp.cpp_search(self._problem_wrapper_for(problem),self.solver_wrapper,root_state,turn)
            if not result.success or not np.isfinite(result.best_action).all():
                raise RuntimeError('E49 private RootPUCT search failed')
            self.calls += 1; self.last_result = result
            return result
    # RootPUCT.__init__ globals point to its historical module: initialize directly.
    opponent = E49RootPUCT.__new__(E49RootPUCT)
    opponent.calls=0; opponent.last_result=None
    opponent.policy_oracle=opponent.create_cpp_policy_oracle([None]*4)
    opponent.value_oracle=opponent.create_cpp_value_oracle(None)
    opponent.vis_on=False; opponent.solver_name='C_PUCT_V1'
    ss=opponent.solver_settings=cpp.Solver_Settings()
    ss.number_simulations=int(simulations);ss.search_depth=100
    ss.C_pw=2.;ss.alpha_pw=.5;ss.C_exp=1.;ss.alpha_exp=.25
    ss.beta_policy=0.;ss.beta_value=0.;ss.export_tree=False;ss.export_root=False
    opponent.solver_wrapper=cpp.Solver_Wrapper(opponent.solver_name,ss,opponent.policy_oracle,opponent.value_oracle)
    opponent.seeded_state(env,int(seed)+17,int(seed)+29)
    return opponent

# Compiled scalar kernel mirrors the reference above; first-call compilation is
# construction/warmup cost and must occur before any decision timing interval.
from numba import njit


@njit(cache=True)
def _interval_numba(ax,ay,bx,by,box):
    lo=0.;hi=1.
    for k in range(2):
        start=ax if k==0 else ay
        delta=(bx-ax) if k==0 else (by-ay)
        if abs(delta)<1e-12:
            if start<box[k,0] or start>box[k,1]:return False,lo,hi
        else:
            a=(box[k,0]-start)/delta;b=(box[k,1]-start)/delta
            lo=max(lo,min(a,b));hi=min(hi,max(a,b))
    return lo<=hi,lo,hi


@njit(cache=True)
def _position_numba(start,velocity,t,boxes):
    end=start+velocity*t
    result=end.copy();first=2.
    for box in boxes:
        hit,lo,hi=_interval_numba(start[0],start[1],end[0],end[1],box)
        if hit and lo<first:
            first=lo
            point=start+(end-start)*min(hi,lo+1e-5)
            for k in range(2):result[k]=max(box[k,0],min(box[k,1],point[k]))
    return result


@njit(cache=True)
def _legal_numba(x,y,boxes):
    if x < -10. or y < -10. or x > 10. or y > 10.:return False
    for box in boxes:
        if box[0,0]<=x<=box[0,1] and box[1,0]<=y<=box[1,1]:return False
    return True


@njit(cache=True)
def _joint_numba(p0,e0,pa,ea,ap,ae,boxes,capture_dt,dt):
    p=p0.copy();e=e0.copy();ap=ap.copy();ae=ae.copy()
    pv=np.minimum(1.,np.maximum(-1.,pa/dt))*ap.reshape((2,1))
    ev=np.minimum(1.5,np.maximum(-1.5,ea/dt))*ae.reshape((2,1))
    possible=False
    for pi in range(2):
        if not ap[pi]:continue
        for ei in range(2):
            if not ae[ei]:continue
            d=p0[pi]-e0[ei];v=pv[pi]-ev[ei]
            vv=v[0]*v[0]+v[1]*v[1]
            t=max(0.,min(dt,-(d[0]*v[0]+d[1]*v[1])/vv)) if vv>0 else 0.
            if (d[0]+t*v[0])**2+(d[1]+t*v[1])**2<1.000000000001:possible=True
    events=np.zeros((2,3));count=0
    n=int(np.ceil(dt/capture_dt)) if possible else 0
    for sample in range(n+1):
        t=min(dt,sample*capture_dt) if possible else dt
        for r in range(2):
            if ap[r]:p[r]=_position_numba(p0[r],pv[r],t,boxes)
            if ae[r]:e[r]=_position_numba(e0[r],ev[r],t,boxes)
        # At most four candidates; insertion sort keeps p/e tie order.
        candidates=np.zeros((4,3));nc=0
        for pi in range(2):
            if not ap[pi] or not _legal_numba(p[pi,0],p[pi,1],boxes):continue
            for ei in range(2):
                if not ae[ei] or not _legal_numba(e[ei,0],e[ei,1],boxes):continue
                dx=p[pi,0]-e[ei,0];dy=p[pi,1]-e[ei,1]
                distance=np.sqrt(dx*dx+dy*dy)
                if distance>=1.:continue
                clear=True
                for box in boxes:
                    hit,_,_=_interval_numba(p[pi,0],p[pi,1],e[ei,0],e[ei,1],box)
                    if hit:clear=False;break
                if not clear:continue
                i=nc
                while i>0 and candidates[i-1,0]>distance:
                    candidates[i]=candidates[i-1];i-=1
                candidates[i,0]=distance;candidates[i,1]=pi;candidates[i,2]=ei;nc+=1
        usedp=np.zeros(2,np.bool_);usede=np.zeros(2,np.bool_)
        for i in range(nc):
            pi=int(candidates[i,1]);ei=int(candidates[i,2])
            if not usedp[pi] and not usede[ei]:
                usedp[pi]=True;usede[ei]=True;ae[ei]=False
                events[count,0]=t;events[count,1]=pi;events[count,2]=ei;count+=1
        if not ae[0] and not ae[1]:
            for r in range(2):
                if ap[r]:p[r]=_position_numba(p0[r],pv[r],dt,boxes)
            break
    return p,e,ap,ae,events[:count]


def joint_transition(positions, targets, pursuer_actions, evader_actions,
                     active_pursuers, active_evaders, boxes, capture_dt=.01, dt=1.):
    """Accelerated reference_joint_transition; identical public return contract."""
    if not (np.isfinite(capture_dt) and 0<capture_dt<=1 and np.isfinite(dt) and dt>0):
        raise ValueError('positive finite capture_dt and dt required')
    arrays=[np.ascontiguousarray(a,dtype=np.float64).reshape(2,2) for a in
            (positions,targets,pursuer_actions,evader_actions)]
    if not all(np.isfinite(a).all() for a in arrays):raise ValueError('nonfinite state or action')
    p,e,ap,ae,events=_joint_numba(*arrays,np.ascontiguousarray(active_pursuers,dtype=np.bool_),
            np.ascontiguousarray(active_evaders,dtype=np.bool_),np.ascontiguousarray(boxes,dtype=np.float64).reshape(-1,2,2),float(capture_dt),float(dt))
    return p,e,ap,ae,[dict(time=float(row[0]),pursuer=int(row[1]),evader=int(row[2])) for row in events]


def warmup_rules():
    """Compile the shared physics kernel outside measured decision latency."""
    joint_transition(np.zeros((2,2)),np.full((2,2),5.),np.zeros((2,2)),
                     np.zeros((2,2)),np.ones(2,bool),np.ones(2,bool),[])
