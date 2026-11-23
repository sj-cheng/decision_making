"""Paired E56 cruise-waypoint ablation and straight-line execution check."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
from start_e56 import DEST, config, outcome, run, save
import numpy as np
from solvers.e53_evader_hmcts import EvaderHMCTS
from solvers.e50_evader_minco import EvaderMinco
from solvers.e50_distributed_geometry import reachable_waypoint

BASE = DEST/'d070c02b0be6ad33'
OUT = DEST/'cruise_ab_150'
DIR = np.array([1.,0.])


def straight(initial, cruise):
    planner = EvaderHMCTS(cruise=cruise)
    executor = EvaderMinco([],1.5,100,max_acceleration=1.5,continuous_speed=True)
    p = np.array([-5.,0.])
    v = initial*DIR
    a = np.zeros(2)
    executor.tail[0] = executor._braking_tail(p,v,a) if initial else None
    if initial == 0:
        executor.tail.clear()
    values = []
    for cycle in range(5):
        track = planner.cruise_track(p,v,a,DIR) if cruise else None
        end = track['positions'][-1] if cruise else reachable_waypoint(p,p+2*DIR,v,2.,1.5,1.5)
        assert track is None or track['feasible']
        prefix,mode = executor.prefix(0,p,end-p,waypoint=True,cruise_track=track)
        if cruise:
            assert mode=='cruise' and np.allclose(track['positions'][:51],prefix['positions'][::2])
        v,a = prefix['velocities'],prefix['accelerations']
        values.append((np.linalg.norm(v,axis=1).max(),np.linalg.norm(a,axis=1).max(),
                       np.linalg.norm(np.diff(a,axis=0)/.01,axis=1).max()))
        p,v,a = prefix['positions'][-1],v[-1],a[-1]
    return dict(zip(('peak_speed','peak_accel','peak_jerk_fd'),np.max(values,axis=0)))


def metrics(x):
    row=outcome(x)
    recs=[rec for l in x['logs'] for rec in l['evader_minco'].values()]
    vs=[np.linalg.norm(r['velocities'],axis=1) for r in recs]
    row.update(active_speed=sum(((v[1:]+v[:-1])*.5).sum() for v in vs)/sum(len(v)-1 for v in vs),
               peak_speed=max(v.max() for v in vs),
               cruise_executed=sum(r['mode']=='cruise' for r in recs),
               cruise_chosen=sum(l['evader_search']['options'].count('cruise') for l in x['logs']),
               active_decisions=len(recs),
               peak_accel=max(np.linalg.norm(r['accelerations'],axis=1).max() for r in recs),
               peak_jerk_fd=max((np.linalg.norm(np.diff(r['accelerations'],axis=0)/.01,axis=1).max()
                                 for r in recs if len(r['accelerations'])>1),default=0.))
    return row


def game(case):
    name=f"{Path(case['path']).stem}_{case['seed']}"
    folder=OUT/name
    trace=run(case['path'],folder/'trace.json',seed=case['seed'],evader_cruise=True,**config(2))
    assert trace['terminal_opponent_skips']==0
    before=json.loads((BASE/'1.50_none_256'/name/'trace.json').read_text())
    result=dict(case=name,original=metrics(before),cruise=metrics(trace))
    save(folder/'result.json',result)
    return result


if __name__=='__main__':
    save(OUT/'straight.json',{str(v):{str(c):straight(v,c) for c in (False,True)} for v in (0.,.6)})
    cases=json.loads((BASE/'manifest.json').read_text())['cases']
    with ProcessPoolExecutor(len(cases)) as pool:
        save(OUT/'paired.json',list(pool.map(game,cases)))
