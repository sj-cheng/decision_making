"""Event-time evaluation of frozen controllers; no wall-clock sleeping.

One-second command/sensor epochs and the original step/capture model are kept.
At intermediate act-completion/deadline events the same step integrates the
corresponding fractional displacement. Opponent updates remain at integer epochs.
"""
import math
import time

import numpy as np

from corner_estimator_scenarios import original_model
from solvers.e45_team_adapter import public_observation
from solvers.root_audit_puct import RootPUCT
from run_frozen_d_closed_loop import legal_action
from learning.l5_followup import make_controller


def evaluate(task, case, mode, limit=None, fixed=None):
    boxes=np.asarray(case['map']); p=original_model(case['map'],case['speed'])
    state=np.asarray(next(w['initial'] for w in case['worlds'] if w['id']==task['world'])).reshape(-1,1)
    own=make_controller(case,task)
    for obs in case['history'][:-1]:own.observe(obs)
    enemy=RootPUCT([None]*4,None,3000,0.,0.)
    enemy.seeded_state(p,task['seed']+17,task['seed']+29)
    start=float(state[8,0]); end=80.; clock=start
    command=np.zeros((2,2)); enemy_command=np.zeros((2,2)); enemy_epoch=None
    decisions=[]; events=[]; pending=[]; shields=0

    def observation():
        return public_observation(int(round(state[8,0])),state[4:8,0].reshape(2,2),state[:4,0].reshape(2,2),
                                  state[p.active_idxs[2:],0]>.5,state[p.active_idxs[:2],0]>.5,boxes)

    def advance(target):
        nonlocal state,clock,enemy_command,enemy_epoch,shields,command
        while clock<target-1e-9 and not p.is_terminal(state):
            epoch=int(math.floor(clock+1e-9))
            if enemy_epoch!=epoch:
                enemy_command=np.asarray(enemy.search(p,state.copy(),turn=0).best_action).reshape(-1)[:4].reshape(2,2).copy()
                enemy_epoch=epoch
            to=min(target,float(epoch+1));dt=to-clock
            if dt<=1e-9:raise RuntimeError('Nonadvancing logical event clock')
            velocity=np.vstack((enemy_command.copy(),command.copy()))
            for robot in range(4):
                cap=1. if robot<2 else case['speed']
                if not p.is_active(state,robot):velocity[robot]=0.
                elif not legal_action(state[2*robot:2*robot+2,0],velocity[robot]*dt,cap*dt,boxes):
                    velocity[robot]=0.;shields+=1
                    if robot>=2:command[robot-2]=0.
                    else:enemy_command[robot]=0.
            state=p.step(state,(velocity*dt).reshape(-1,1),dt)
            clock=float(state[8,0])
            if not p.is_valid(state):raise RuntimeError('Illegal event-time state')
            if abs(clock-round(clock))<1e-8:pending.append(observation())
            events.append(dict(time=clock,dt=dt,positions=state[:8,0].tolist(),active=state[p.active_idxs,0].tolist()))

    pending.append(observation())
    while clock<end-1e-9 and not p.is_terminal(state) and (limit is None or len(decisions)<limit):
        for obs in pending[:-1]:own.observe(obs)
        obs=pending[-1];pending=[]
        origin=clock;wall=time.perf_counter();cpu=time.process_time()
        action=own.act(obs,fixed_simulations=fixed)
        latency=time.perf_counter()-wall;cpu_ms=1000*(time.process_time()-cpu)
        delay=0. if mode=='zero' else latency
        deadline=float(task.get('budget_ms',200))/1000.
        late=delay>deadline
        first=min(origin+delay,origin+deadline if late else origin+delay,end)
        advance(first)
        if not p.is_terminal(state) and clock<end:
            if late:
                command[:]=0.
                advance(min(origin+delay,end))
            else:
                command=np.asarray(action).copy()
                for i in range(2):
                    if not p.is_active(state,i+2):command[i]=0.
                    elif not legal_action(state[4+2*i:6+2*i,0],command[i],case['speed'],boxes):
                        command[i]=0.;shields+=1
        decisions.append(dict(observation=obs,requested_at=origin,completed_at=origin+delay,
                              physical_latency_ms=delay*1000,measured_latency_ms=latency*1000,
                              cpu_ms=cpu_ms,deadline_missed=late,discarded=late,
                              action=action.tolist(),stats=own.last_stats))
        if not p.is_terminal(state) and clock<end:
            advance(min(float(math.floor(clock+1e-9)+1),end))
    complete=not (state[p.active_idxs[:2],0]>.5).any();elapsed=clock-start
    return dict(summary=dict(double_capture=bool(complete),utility=float(complete)-.0025*elapsed,
                completion_cost=elapsed if complete else 80-start,elapsed_physical_seconds=elapsed,
                decisions=len(decisions),deadline_misses=sum(r['deadline_missed'] for r in decisions),
                shields=shields,mode=mode,control_period_s=1.,sensor_period_s=1.,
                timeout_behavior='independent plant stops at 0.2s; late result discarded; solver CPU is not magically cancelled',
                opponent_timing='ideal logical-time updates each 1s; opponent computation not charged to pursuer',
                scope='event-time simulation, not a hard real-time implementation guarantee'),
                decisions=decisions,events=events,final_state=state[:,0].tolist())
