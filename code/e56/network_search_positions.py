"""E58. Matched trials of candidate shortlisting, set policy and first-visit edge value."""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from advanced_network_runs import BASE, DEV, TRAIN, VALID, SCALE, StudentEvader, load, read
from network_geometry_experiment import encode, play, Ranker, OUT as GEOM
from run_1000_cruise import save, outcome
from attention_diffusion_experiment import Attention, teacher_tensors

OPPONENT='learned_hmcts32'
OUT=SCALE.parent/'e58_search_positions'/OPPONENT


def play_with_opponent(case, folder, **settings):
    trace=play(case,folder,evader_factory=lambda **kwargs:StudentEvader(True,**kwargs),
               e53_simulations=32,**settings)
    if trace['e53_simulations']!=32:
        raise RuntimeError('unexpected evader search budget')
    return trace

BUDGET={'base128':128,'base64':64,'first3':64,'rank3':64,'set64':64,'value64':64,'value32':32,'set32':32}


class SetPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(34,96),nn.SiLU(),nn.Linear(96,96),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(192,96),nn.SiLU(),nn.Linear(96,2))
        self.use_edge_value=False

    def forward(self,x,mask):
        h=self.embed(x)
        context=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        return self.head(torch.cat((h,context[:,None].expand_as(h)),dim=-1))

    @torch.inference_mode()
    def rank_actions(self,obs,endpoints,options,durations):
        x=torch.from_numpy(np.stack([encode(obs,p,i) for p,i in zip(endpoints,options)]))[None]
        y=self(x,torch.ones(x.shape[:2],dtype=torch.bool))[0].numpy()
        self.edge_values=y[:,1]
        return y[:,0]


def train():
    torch.manual_seed(52)
    data=[tuple(x.cuda() for x in teacher_tensors(split)) for split in (TRAIN,VALID)]
    model=SetPolicy().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0006)
    best=float('inf')
    for epoch in range(50):
        for training,(x,v,q,m) in zip((True,False),data):
            model.train(training);total=0.
            for ix in (torch.randperm(len(x)) if training else torch.arange(len(x))).split(64):
                with torch.set_grad_enabled(training):
                    y=model(x[ix],m[ix]);weight=v[ix].sqrt()
                    policy=-(F.normalize(v[ix],p=1,dim=1)*y[:,:,0].masked_fill(~m[ix],-1e9).log_softmax(1)).sum(1).mean()
                    value=(F.smooth_l1_loss(y[:,:,1],q[ix],reduction='none')*weight).sum()/weight.sum()
                    loss=policy+.25*value
                if training:
                    opt.zero_grad();loss.backward();opt.step()
                total+=len(ix)*loss.item()
            if not training:
                metric=total/len(x)
                if metric<best:
                    best=metric;torch.save(model.state_dict(),OUT/'best.tmp');(OUT/'best.tmp').replace(OUT/'best.pt')
        save(OUT/'train.json',dict(epoch=epoch+1,best=best,validation=metric))
    save(OUT/'ready.json',True)


class Setup:
    measure_team_latency=True

    def __init__(self,arm):
        self.arm=arm
        if arm=='rank3':self.ranker=load(Ranker('geometry'),GEOM/'geometry.pt')

    def begin(self,controller,obs,captured):
        if self.arm in ('first3','rank3'):controller.candidate_shortlist=self.shortlist

    @torch.inference_mode()
    def shortlist(self,obs,pools):
        result=[]
        for option,points in pools:
            if option!=3 and len(points)>3:
                if self.arm=='rank3':
                    x=np.stack([encode(obs,p,option) for p in points])
                    order=np.argsort(-self.ranker(torch.from_numpy(x)).numpy(),kind='stable')
                    points=[points[i] for i in order[:3]]
                else:points=points[:3]
            result.append((option,points))
        return result


def run(task):
    arm,c=task
    folder=OUT/'episodes'/arm/c['case'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return
    if arm.startswith(('set','value')):
        model=load(SetPolicy(),OUT/'best.pt');model.use_edge_value=arm.startswith('value')
    else:model=load(Attention(128,4),BASE)
    with TemporaryDirectory(dir=folder) as tmp:
        trace=play_with_opponent(c,Path(tmp),nar_override=model,e52_mode='nar',e52_budget=BUDGET[arm],
                   search_experiment=Setup(arm))
    save(folder/'result.json',dict(opponent=OPPONENT,evader_simulations=32,case=c['case'],seed=c['seed'],arm=arm,**outcome(trace),
        team_ms=[l['coordination']['team_decision_ms'] for l in trace['logs']],
        search_ms=[s['elapsed_ms'] for l in trace['logs'] for s in l['search'].values()],
        evaluations=[s['evaluations'] for l in trace['logs'] for s in l['search'].values()]))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('arm',choices=['train',*BUDGET]);arm=parser.parse_args().arm
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    if arm=='train':train()
    else:
        if arm.startswith(('set','value')):
            while not (OUT/'ready.json').exists():time.sleep(5)
        with ProcessPoolExecutor(len(os.sched_getaffinity(0))) as pool:
            list(pool.map(run,[(arm,c) for c in DEV]))
