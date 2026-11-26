"""Distributed LiDAR waypoint policy. Offline imitation followed by on-policy PPO."""
import argparse
import fcntl
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from advanced_network_runs import TRAIN, VALID, DEV, StudentEvader, old, play, read, outcome, save
from network_search_positions import OUT as E58, OPPONENT, play_with_opponent
from solvers.e50_lidar_map import raycast_scan
from solvers.e50_distributed_runtime import LocalPlan

CAPACITY=os.environ.get('E58_WAYPOINT_CAPACITY','base')
TRANSFORMERS={'transformer':(128,5,4), 'transformer_xl':(512,6,8), 'transformer_xxl':(768,8,12)}
if CAPACITY not in ('base','xl','xxl',*TRANSFORMERS):
    raise ValueError('unknown waypoint policy capacity')
OUT=E58/('direct_waypoint' if CAPACITY=='base' else f'direct_waypoint_{CAPACITY}')
STABLE_XXL = os.environ.get('E58_STABLE_XXL') == '1'
if STABLE_XXL:
    if CAPACITY != 'transformer_xxl':
        raise ValueError('stable XXL requires transformer_xxl capacity')
    OUT = OUT.with_name(OUT.name + '_stable')
STEPS=32
SCALE=2.25


def features(scan,pose,peer,belief,cycle,captured,agent):
    targets=np.zeros((2,6))
    for track in belief:
        i=int(track['target_id'])
        if i in (0,1):
            targets[i]=[*((np.asarray(track['position'])-pose)/10),*track['velocity'],
                        min(1.,(cycle-track['timestamp'])/80),1]
    return np.r_[scan.ranges/scan.max_range,pose/10,peer,targets.ravel(),cycle/80,captured,agent].astype('float32')


def extract(case):
    dest=OUT/'demonstrations'/(case['case']+'.npz')
    if dest.exists():return
    trace=read(old.PREVIOUS/'teacher'/case['case']/'trace.json')
    world=read(Path(case['path']));pose=np.array(world['worlds'][0]['initial'][4:8]).reshape(2,2)
    xs=[];ys=[];captured=0
    for l in trace['logs']:
        for i in range(2):
            k=str(i);root=l['search'][k]['root']
            # Capture notifications prune the logged belief after this decision.
            if not root or len(l['captures'])!=captured:continue
            peer=np.array(root[0]['features'][9:14]);peer[:2]/=10
            scan=raycast_scan(pose[i],world['map'],num_rays=256,max_range=10.)
            xs.append(features(scan,pose[i],peer,l['beliefs'][k],l['cycle'],captured,i))
            ys.append(l['plans'][k]['action'])
        pose=np.asarray(l['pursuers']);captured=len(l['captures'])
    tmp=dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,x=np.array(xs),y=np.array(ys,dtype='float32'));tmp.replace(dest)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.logstd=nn.Parameter(torch.full((2,),-1.6))
        if CAPACITY in TRANSFORMERS:
            width,depth,heads=TRANSFORMERS[CAPACITY]
            self.sector=nn.Linear(16,width)
            self.context=nn.Linear(22,width)
            self.positions=nn.Parameter(torch.randn(1,17,width)*.02)
            self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(
                width,heads,2*width,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
                for _ in range(depth)])
            self.norm=nn.LayerNorm(width)
            self.actor=nn.Linear(width,2);self.critic=nn.Linear(width,1)
            return
        c1,c2,width,latent,depth={'base':(32,64,384,256,1),'xl':(96,192,1536,512,3),
            'xxl':(128,256,2560,1024,4)}[CAPACITY]
        self.lidar=nn.Sequential(nn.Conv1d(1,c1,7,2,3,padding_mode='circular'),nn.SiLU(),
            nn.Conv1d(c1,c2,5,2,2,padding_mode='circular'),nn.SiLU(),nn.AdaptiveAvgPool1d(16))
        layers=[nn.Linear(16*c2+22,width),nn.LayerNorm(width),nn.SiLU()]
        for _ in range(depth):
            layers.extend([nn.Linear(width,width),nn.LayerNorm(width),nn.SiLU()])
        layers.extend([nn.Linear(width,latent),nn.SiLU()])
        self.trunk=nn.Sequential(*layers)
        self.actor=nn.Linear(latent,2);self.critic=nn.Linear(latent,1)

    def forward(self,x):
        if hasattr(self,'blocks'):
            sectors=self.sector(x[:,:256].reshape(-1,16,16))
            tokens=torch.cat((self.context(x[:,256:])[:,None],sectors),1)+self.positions
            for block in self.blocks:
                tokens=block(tokens)
            h=self.norm(tokens[:,0])
        else:
            z=self.lidar(x[:,:256,None].transpose(1,2)).flatten(1)
            h=self.trunk(torch.cat((z,x[:,256:]),1))
        return Normal(self.actor(h),self.logstd.clamp(-3,.2).exp()),self.critic(h).flatten()


def checkpoint(model,path):
    tmp=path.with_suffix('.tmp');torch.save(model.state_dict(),tmp);tmp.replace(path)


def weights(path):
    model=ActorCritic();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    return model.eval()


def demonstrations(split,device):
    xs=[];ys=[]
    for case in split:
        with np.load(OUT/'demonstrations'/(case['case']+'.npz')) as z:
            xs.append(z['x']);ys.append(z['y'])
    return tuple(torch.from_numpy(np.concatenate(v)).float().to(device) for v in (xs,ys))


def imitate():
    (OUT/'demonstrations').mkdir(exist_ok=True)
    missing=[c for c in TRAIN+VALID if not (OUT/'demonstrations'/(c['case']+'.npz')).exists()]
    if missing:
        with ProcessPoolExecutor(len(os.sched_getaffinity(0)),mp_context=get_context('spawn')) as pool:
            list(pool.map(extract,missing))
    x,y=demonstrations(TRAIN,'cuda');v,t=demonstrations(VALID,'cuda')
    model=ActorCritic().cuda()
    if STABLE_XXL:
        nn.init.normal_(model.actor.weight, std=.001)
        nn.init.zeros_(model.actor.bias)
    opt=torch.optim.AdamW(model.parameters(),lr=.00003 if STABLE_XXL else .0003)
    best=float('inf')
    for epoch in range(100):
        train_total = torch.zeros((), device=x.device)
        for ix in torch.randperm(len(x)).split(128):
            dist,_=model(x[ix]);loss=F.smooth_l1_loss(SCALE*dist.mean.tanh(),y[ix])
            opt.zero_grad();loss.backward();opt.step()
            train_total += loss.detach() * len(ix)
        with torch.no_grad():
            predicted = torch.cat([SCALE * model(v[ix])[0].mean.tanh()
                                   for ix in torch.arange(len(v)).split(256)])
            val = F.mse_loss(predicted, t).item()
            saturation = (predicted.abs() > 2.24).float().mean().item()
        if val<best:best=val;checkpoint(model,OUT/'bc.pt')
        metrics = dict(epoch=epoch+1, best=best, validation_mse=val, examples=len(x),
                       training_loss=train_total.item()/len(x), saturation=saturation,
                       output_std=predicted.std(dim=0).tolist())
        save(OUT/'bc_progress.json',metrics)
        with (OUT/'bc_history.jsonl').open('a') as history:
            history.write(json.dumps(metrics) + '\n')
    save(OUT/'bc_ready.json',True)


class Direct:
    def __init__(self,model,sample):
        self.model=model;self.sample=sample;self.rows=[]
    @torch.inference_mode()
    def act(self,obs,scan,captured):
        peer=np.zeros(5)
        if obs.peer_messages:
            p=obs.peer_messages[0];peer=np.r_[(p.pose-obs.pose)/10,p.velocity,1.]
        x=features(scan,obs.pose,peer,obs.belief.payload(),obs.cycle,captured,obs.agent_id)
        dist,val=self.model(torch.from_numpy(x)[None])
        latent=dist.sample() if self.sample else dist.mean
        delta=(SCALE*latent.tanh())[0].numpy()
        if self.sample:self.rows.append((x,latent[0].numpy(),dist.log_prob(latent).sum().item(),val.item()))
        return LocalPlan(obs.agent_id,obs.pose+delta,delta,val.item(),0,-1,True,2.,True)


def episode(task):
    mode,iteration,index,case=task
    seed=int(case['seed'])+iteration*113+index if mode=='ppo' else case['seed']
    torch.manual_seed(seed);torch.set_num_threads(1)
    policy=Direct(weights(OUT/('bc.pt' if mode=='bc' else 'ppo.pt')),mode=='ppo')
    kw=dict(pursuer_structure='local',e52_mode='none',direct_policy=policy)
    with TemporaryDirectory(dir=OUT) as tmp:
        trace=play_with_opponent(dict(case,seed=seed),Path(tmp),**kw)
    summary=dict(opponent=OPPONENT,evader_simulations=32,**outcome(trace),team_ms=[l['coordination']['team_decision_ms'] for l in trace['logs']],
                 search_ms=[r['elapsed_ms'] for l in trace['logs'] for r in l['search'].values()])
    if mode!='ppo':
        save(OUT/'eval'/mode/case['case']/'result.json',summary);return
    data=policy.rows;rewards=[];last=0
    for l in trace['logs']:
        n=len(l['captures']);rewards.append(n-last+float(n==2)-.005);last=n
    vals=np.array([r[3] for r in data]).reshape(-1,2);adv=np.zeros_like(vals);tail=np.zeros(2)
    for t in reversed(range(len(vals))):
        future=vals[t+1] if t+1<len(vals) else 0
        tail=rewards[t]+.99*future-vals[t]+.99*.95*tail;adv[t]=tail
    return (np.stack([r[0] for r in data]),np.stack([r[1] for r in data]),
            np.array([r[2] for r in data]),adv.ravel(),(adv+vals).ravel())


def finetune():
    model=weights(OUT/'bc.pt').cuda()
    dx,dy=demonstrations(TRAIN,'cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=.00005)
    checkpoint(model,OUT/'ppo.pt');rng=np.random.default_rng(58)
    with ProcessPoolExecutor(len(os.sched_getaffinity(0)),mp_context=get_context('spawn')) as pool:
        for it in range(STEPS):
            batch=list(pool.map(episode,[('ppo',it,i,TRAIN[j]) for i,j in enumerate(rng.choice(len(TRAIN),12,replace=False))]))
            x,u,lp,adv,ret=[torch.tensor(np.concatenate(z),device='cuda',dtype=torch.float32) for z in zip(*batch)]
            adv=(adv-adv.mean())/(adv.std()+1e-6)
            for _ in range(4):
                for ix in torch.randperm(len(x)).split(128):
                    dist,v=model(x[ix]);ratio=(dist.log_prob(u[ix]).sum(1)-lp[ix]).exp()
                    pg=-torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix]).mean()
                    j=torch.randint(len(dx),(len(ix),));bc=F.smooth_l1_loss(SCALE*model(dx[j])[0].mean.tanh(),dy[j])
                    loss=pg+.5*F.mse_loss(v,ret[ix])+.1*bc-.005*dist.entropy().sum(1).mean()
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step()
            checkpoint(model,OUT/'ppo.pt');save(OUT/'ppo_progress.json',dict(update=it+1,episodes=(it+1)*12))
    save(OUT/'ppo_ready.json',True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['train','bc','final','imitate','ppo']);phase=parser.parse_args().phase
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.manual_seed(58)
    if phase=='imitate':
        imitate()
    elif phase=='ppo':
        while not (OUT/'bc_ready.json').exists():time.sleep(10)
        if STABLE_XXL:
            with (OUT/'ppo.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not (OUT/'ppo_ready.json').exists():
                    finetune()
        else:
            finetune()
    elif phase=='train':
        if not (OUT/'bc_ready.json').exists():imitate()
        finetune()
    else:
        ready='bc' if phase=='bc' else 'ppo'
        while not (OUT/(ready+'_ready.json')).exists():time.sleep(10)
        with ProcessPoolExecutor(len(os.sched_getaffinity(0)),mp_context=get_context('spawn')) as pool:
            list(pool.map(episode,[(phase,0,i,c) for i,c in enumerate(DEV)
                                   if not (OUT/'eval'/phase/c['case']/'result.json').exists()]))
