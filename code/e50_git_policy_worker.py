"""Isolated historical d47d188 pursuer process; JSON stdin/stdout protocol."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'current/pursuit_evasion_figures_20261107/source/git_d47d188/code'
sys.path.insert(0,str(ARCHIVE))
import numpy as np
import torch
torch.set_num_threads(1)

with contextlib.redirect_stdout(sys.stderr):
    from problems.example8 import Example8
    from learning.oracles import get_oracles
    from solvers.solver import get_solver
    case=json.loads(Path(sys.argv[1]).read_text());p=Example8()
    p.obstacles=[np.asarray(b,float) for b in case['map']];p.use_minco_dynamics=False
    p.current_evader_speed_lim=1.;p.current_pursuer_speed_lim=case['speed']
    for i in range(4):
        cap=1. if i<2 else case['speed']
        p.action_lims[p.action_idxs[i]]=[[-cap,cap],[-cap,cap]]
        p.control_lims[p.action_idxs[i]]=p.action_lims[p.action_idxs[i]]/p.dt
    p.clear_closed_loop_cache()
    weights=ROOT/'current/memory_observation_v3/models'
    paths=[str(weights/f'model_policy_l45_i{i}.pt') for i in range(4)]
    value=str(weights/'model_value_l45.pt')
    policy,v=get_oracles(p,policy_oracle_name='gaussian',policy_oracle_paths=paths,
                        value_oracle_name='deterministic',value_oracle_path=value)
    own=get_solver('C_PUCT_V1',policy_oracle=policy,value_oracle=v,search_depth=100,
            number_simulations=1000,C_pw=2.,alpha_pw=.5,C_exp=1.,alpha_exp=.25,
            beta_policy=.9,beta_value=0.,vis_on=False)
print(json.dumps(dict(ready=True,source_commit='d47d188',weights={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in paths+[value]},
    privileged_root=True,simulations=1000,depth=100,beta_policy=.9,beta_value=0.,
    note='Original controller; available V3 iteration45, not proven historical commit-time default weights')),flush=True)
for line in sys.stdin:
    req=json.loads(line)
    if req.get('quit'):break
    s=np.asarray(req['state'],float)[:,None];start=time.perf_counter()
    action=own.policy(p,s,team=[2,3])[4:8,0].reshape(2,2)
    print(json.dumps(dict(action=action.tolist(),act_ms=1000*(time.perf_counter()-start))),flush=True)
