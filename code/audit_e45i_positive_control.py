"""Bounded learning control for the SAME author policy and episode buffer.

Fully observed, stationary one-target proxy with the project's displacement,
legal-action and capture rules. Not the hidden-observation main benchmark.
Weights are never injected into a baseline experiment.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from learning.e45i_official_mappo import policy,actor_inputs,finish_episodes,HIDDEN,OBS_DIM
from learning.e45b_joint_network import public_features
from learning.e45g_ppo import safe_action
from solvers.e45b_joint_nte import simultaneous_capture
from run_frozen_d_closed_loop import save,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'current/e45i_code_audit_20261031/positive_control'
BOXES=np.zeros((0,2,2));SPEED=1.1;HORIZON=20


def starts(rng,n):
    angle=rng.uniform(-np.pi,np.pi,n);radius=rng.uniform(2.5,3.5,n)
    return np.stack([np.cos(angle)*radius,np.sin(angle)*radius],axis=1)


@torch.no_grad()
def collect(model,initial,deterministic=False):
    count=len(initial);positions=np.tile(np.array([[0.,0.],[-8.,-8.]]),(count,1,1))
    positions[:,0]=initial;targets=np.array([[0.,0.],[-8.,-8.]])
    hidden=np.zeros((count*2,1,HIDDEN),np.float32);done=np.zeros(count,bool)
    histories=[[] for _ in range(count)];episodes=[{key:[] for key in
        ('obs','actor_hidden','raw','logp','active','privileged','reward')} for _ in range(count)]
    captures=np.zeros(count,bool)
    for step in range(HORIZON):
        x=np.zeros((count,2,OBS_DIM),np.float32);mask=np.zeros((count,2,1),np.float32)
        for e in np.flatnonzero(~done):
            obs=dict(step=step,remaining=80-step,positions=positions[e].tolist(),active_p=[True,False],
                active_e=[True,False],reports=[[[0.,0.],None],[None,None]])
            histories[e].append(obs)
            samples=np.zeros((2,64,2));weights=np.zeros((2,64));weights[0]=1/64
            features=public_features(BOXES,positions[e],[True,False],[True,False],samples,weights,
                [targets[0],None],histories[e],SPEED,80-step)
            x[e]=actor_inputs(features);mask[e]=1
        before=hidden.copy()
        raw,logp,after=model.actor(x.reshape(-1,OBS_DIM),hidden,mask.reshape(-1,1),deterministic=deterministic)
        raw=raw.cpu().numpy().reshape(count,2,2);logp=logp.cpu().numpy().reshape(count,2,1)
        hidden=after.cpu().numpy()
        for e in np.flatnonzero(~done):
            phi=-np.max(np.abs(positions[e,0]-targets[0]))/SPEED
            action,_=safe_action(positions[e],np.tanh(raw[e]),[True,False],SPEED,BOXES)
            positions[e]+=action
            ap,ae,_=simultaneous_capture(positions[e],targets,np.array([True,False]),np.array([True,False]),BOXES)
            captured=not ae[0];terminal=captured or step==HORIZON-1
            phi_next=0. if terminal else -np.max(np.abs(positions[e,0]-targets[0]))/SPEED
            values=dict(obs=x[e],actor_hidden=before.reshape(count,2,1,HIDDEN)[e],
                raw=raw[e],logp=logp[e],active=np.array([[1.],[0.]],np.float32),
                privileged=targets.ravel().astype(np.float32)/10,
                reward=float(captured)-.0025+.05*(phi_next-phi))
            for key,value in values.items():episodes[e][key].append(value)
            done[e]=terminal;captures[e]=captured
        if done.all():break
    for ep in episodes:
        for key,value in ep.items():ep[key]=np.asarray(value,np.float32)
    return episodes,captures


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--updates',type=int,default=80)
    parser.add_argument('--device',default='cuda:2');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.cuda.set_device(torch.device(args.device))
    torch.manual_seed(17);rng=np.random.default_rng(17);validation=starts(np.random.default_rng(20261031),32)
    model,trainer=policy('RMAPPO',args.device);began=time.time();steps=0;records=[]
    trainer.prep_rollout();_,initial=collect(model,validation,True)
    save(OUT/'protocol.json',dict(scope='learning positive control, NOT main task; fully observed stationary 1v1',
        source_sha256=sha(__file__),max_updates=args.updates,max_team_steps=args.updates*32*HORIZON,
        train_episodes_per_update=32,horizon=HORIZON,held_out_initial_states=32,
        stop='two successive deterministic validation rates >=90%, evaluated every10updates',
        algorithm='same E45-I settings, author R_MAPPO, same EpisodeBuffer and legal/capture functions',
        initial_validation_capture=float(initial.mean()),weights_reused_in_main_experiment=False))
    successes=0
    for update in range(1,args.updates+1):
        trainer.prep_rollout();episodes,training=collect(model,starts(rng,32))
        steps+=sum(len(e['obs']) for e in episodes);buffer=finish_episodes(episodes,model)
        trainer.prep_training();info=trainer.train(buffer)
        record=dict(update=update,steps=steps,training_capture=float(training.mean()),
            value_loss=float(info['value_loss']),policy_loss=float(info['policy_loss']))
        if update%10==0 or update==args.updates:
            trainer.prep_rollout();_,validation_result=collect(model,validation,True)
            record['validation_capture']=float(validation_result.mean())
            successes=successes+1 if validation_result.mean()>=.9 else 0
        records.append(record);save(OUT/'progress.json',dict(records=records))
        if 'validation_capture' in record:print(record,flush=True)
        if successes>=2:break
    trainer.prep_rollout();_,final=collect(model,validation,True)
    save(OUT/'decision.json',dict(status='complete',updates=update,steps=steps,
        initial_capture=float(initial.mean()),final_capture=float(final.mean()),
        positive_control_passed=bool(final.mean()>=.9),wall_seconds=time.time()-began,
        scope='proxy learning control, not main-benchmark performance'))
    torch.save(dict(actor=model.actor.state_dict(),critic=model.critic.state_dict()),OUT/'proxy_only.pt')


if __name__=='__main__':main()
