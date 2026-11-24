"""E56 held-out map comparisons with token attention and conditional waypoint diffusion."""
import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from network_geometry_experiment import OUT as PREVIOUS, CFG, encode, manifest, play
from compare_nae_1000 import MODEL, H80, outcome, save
from run_e55_parallel import Shadow

OUT = H80/'attention_diffusion'
BUDGETS = (256, 128)


class Attention(nn.Module):
    """One action token attends to belief, teammate, and measured obstacle tokens."""
    def __init__(self):
        super().__init__()
        self.action = nn.Linear(5,64)
        self.target = nn.Linear(4,64)
        self.peer = nn.Linear(5,64)
        self.obstacle = nn.Linear(5,64)
        layer = nn.TransformerEncoderLayer(64,4,128,dropout=0.,batch_first=True)
        self.encoder = nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.head = nn.Linear(64,1)

    def forward(self,x):
        boxes=x[:,14:].reshape(-1,4,5)
        tokens=torch.cat([self.action(x[:,:5])[:,None],self.target(x[:,5:9])[:,None],
                          self.peer(x[:,9:14])[:,None],self.obstacle(boxes)],1)
        mask=torch.cat([torch.zeros_like(x[:,:1]),1-x[:,8:9],1-x[:,13:14],1-boxes[:,:,4]],1).bool()
        return self.head(self.encoder(tokens,src_key_padding_mask=mask)[:,0]).flatten()

    @torch.inference_mode()
    def rank_actions(self,obs,endpoints,options,durations):
        x=np.stack([encode(obs,e,o) for e,o in zip(endpoints,options)])
        return self(torch.from_numpy(x)).numpy()


def teacher_tensors(cases):
    rows=[]
    for c in cases:
        folder=PREVIOUS/'teacher'/c['case']
        groups=json.loads((folder/'groups.json').read_text())
        logs=json.loads((folder/'trace.json').read_text())['logs']
        roots=[log['search'][i]['root'] for log in logs for i in ('0','1')]
        roots=[r for r in roots if len(r)>=2 and sum(a['visits'] for a in r)>0]
        assert len(roots)==len(groups)
        for group,root in zip(groups,roots):
            visits=[a['visits'] for a in root]
            assert visits==group['visits'] and np.allclose(np.array(group['x'])[:,:14],[a['features'] for a in root])
            rows.append((group['x'],visits,[a['value'] for a in root]))
    width=max(len(x) for x,_,_ in rows)
    x=torch.zeros(len(rows),width,34);v=torch.zeros(len(rows),width);q=v.clone();m=v.bool()
    for i,(features,visits,values) in enumerate(rows):
        n=len(visits);x[i,:n]=torch.tensor(features);v[i,:n]=torch.tensor(visits)
        q[i,:n]=torch.tensor(values);m[i,:n]=True
    return x,v,q,m


def loss_att(model,batch,kind):
    x,v,q,m=batch
    pred=torch.zeros_like(q).masked_scatter(m,model(x[m]))
    if kind=='nar':
        return -(v/v.sum(1,keepdim=True)*pred.masked_fill(~m,-1e9).log_softmax(1)).sum(1).mean()
    # Search mean edge return is defined only for visited siblings.
    weight=v.sqrt()
    return ((pred-q).square()*weight).sum()/weight.sum()


def condition(obs):
    # Independent of the teacher's chosen action, usable before proposing a waypoint.
    boxes=np.asarray(obs.local_map.obstacle_boxes()).reshape(-1,2,2)-obs.pose[None,:,None]
    gap=np.maximum(np.maximum(boxes[:,:,0],-boxes[:,:,1]),0)
    boxes=boxes[np.argsort(np.linalg.norm(gap,axis=1),kind='stable')[:4]]
    rect=np.zeros((4,5));rect[:len(boxes),:4]=boxes.reshape(-1,4);rect[:len(boxes),4]=1
    return np.r_[encode(obs,obs.pose,0)[:14],obs.velocity,np.sin(obs.heading),np.cos(obs.heading),rect.ravel()].astype('float32')


def collect(case):
    folder=OUT/'diffusion_teacher'/case['case']
    if (folder/'actions.json').exists():return
    rows=[]
    def record(cycle,obs,controllers,plans):
        for i in (0,1):
            actions=[a for a in controllers[i].last_search['root'] if a['visits'] and a['option']!='recover']
            if actions:
                rows.append(dict(state=condition(obs[i]).tolist(),delta=[(a['endpoint']-obs[i].pose).tolist() for a in actions],
                                 visits=[a['visits'] for a in actions]))
    play(case,folder,decision_hook=record,e52_budget=2048,cycles=40)
    save(folder/'actions.json',rows)


def diffusion_tensors(cases):
    states=[];actions=[];weights=[]
    for c in cases:
        for r in json.loads((OUT/'diffusion_teacher'/c['case']/'actions.json').read_text()):
            states.extend([r['state']]*len(r['delta']));actions.extend(r['delta'])
            weights.extend(np.asarray(r['visits'])/sum(r['visits']))
    return tuple(torch.tensor(np.asarray(x),dtype=torch.float32) for x in (states,actions,weights))


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoise=nn.Sequential(nn.Linear(43,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,2))
        t=torch.linspace(0,1,17)
        cumulative=((t+.008)/1.008*math.pi/2).cos().square()
        self.register_buffer('alpha',(cumulative[1:]/cumulative[:-1]).clamp(.001,.999).cumprod(0))
        self.rng=torch.Generator().manual_seed(52)

    def forward(self,state,noisy,t):
        angle=t[:,None]/15*math.pi
        return self.denoise(torch.cat([state,noisy,angle,angle.sin(),angle.cos()],1))

    @torch.inference_mode()
    def propose(self,obs):
        state=torch.from_numpy(condition(obs))[None].expand(2,-1)
        action=torch.randn(2,2,generator=self.rng)
        for t in (15,12,9,6,3,0):
            a=self.alpha[t]
            eps=self(state,action,torch.full((2,),t))
            clean=((action-(1-a).sqrt()*eps)/a.sqrt()).clamp(-1,1)
            prev=self.alpha[max(0,t-3)]
            action=prev.sqrt()*clean+(1-prev).sqrt()*eps
        return obs.pose+2*clean.numpy()


def loss_diff(model,batch,generator):
    state,actions,w=batch
    t=torch.randint(16,(len(w),),device=w.device,generator=generator)
    noise=torch.randn(actions.shape,device=w.device,generator=generator)
    a=model.alpha[t,None]
    noisy=a.sqrt()*actions/2+(1-a).sqrt()*noise
    return ((model(state,noisy,t)-noise).square().mean(1)*w).sum()/w.sum()


def train(kind,splits):
    device=torch.device('cuda:0')
    torch.manual_seed(52)
    diff=kind=='diffusion'
    model=(Diffusion() if diff else Attention()).to(device)
    load=diffusion_tensors if diff else teacher_tensors
    data={k:tuple(t.to(device) for t in load(splits[k])) for k in ('train','valid')}
    folder=OUT/kind
    folder.mkdir(parents=True,exist_ok=True)
    opt=torch.optim.AdamW(model.parameters(),lr=.0005)
    best=float('inf');history=[]
    for epoch in range(40):
        for phase in ('train','valid'):
            training=phase=='train'
            model.train(training)
            tensors=data[phase];n=len(tensors[0])
            idx=torch.randperm(n) if training else torch.arange(n)
            generator=torch.Generator(device=device).manual_seed(52+epoch if training else 999)
            total=0.
            for ids in idx.split(32 if not diff else 128):
                batch=[t[ids] for t in tensors]
                with torch.set_grad_enabled(training):
                    loss=loss_diff(model,batch,generator) if diff else loss_att(model,batch,kind)
                if training:
                    opt.zero_grad();loss.backward();opt.step()
                total+=loss.item()*len(ids)
            if phase=='valid':
                history.append(total/n)
                if history[-1]<best:
                    best=history[-1]
                    torch.save(model.state_dict(),folder/'model.tmp')
                    (folder/'model.tmp').replace(folder/'model.pt')
        save(folder/'training.json',dict(validation=history,best=best,epochs=epoch+1))
    print(kind,'training complete',flush=True)


def evaluate(task, folder=None):
    kind,budget,c=task
    if folder is None:
        folder=OUT/kind/f'b{budget}'/c['case']
    p=folder/'result.json'
    if p.exists():return json.loads(p.read_text())
    kw=dict(e52_budget=budget)
    hook=None
    if kind in ('nar','nae','diffusion'):
        model=Diffusion() if kind=='diffusion' else Attention()
        model.load_state_dict(torch.load(OUT/kind/'model.pt',weights_only=True,map_location='cpu'));model.eval()
        if kind=='nar':kw.update(nar_override=model,e52_mode='nar')
        elif kind=='nae':
            hook=Shadow(2,False)
            kw.update(nae_override=model,e52_mode='both',e52_checkpoint=str(MODEL),decision_hook=hook,audit_nae=True)
        else:
            model.rng.manual_seed(c['seed'])
            kw['waypoint_generator']=model
    elif kind=='nar_control':
        kw.update(e52_mode='nar',e52_checkpoint=str(MODEL))
    start=perf_counter()
    x=play(c,folder,**kw)
    row=dict(kind=kind,case=c['case'],budget=budget,wall_seconds=perf_counter()-start,
             search_ms=[s['elapsed_ms'] for l in x['logs'] for s in l['search'].values()],**outcome(x))
    if hook:
        row['decisions']=2*len(hook.rows)
        for key in ('queried','root_action_changed','executed_plan_changed'):
            row[key]=sum(sum(r[key]) for r in hook.rows)
    save(p,row);return row


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=('nar','nae','diffusion'))
    args=parser.parse_args();kind=args.kind;torch.set_num_threads(1)
    splits=manifest();folder=OUT/kind;save(folder/'manifest.json',splits)
    # Spawn evaluation workers after CUDA training to avoid forking CUDA state.
    from multiprocessing import get_context
    def pool():
        return ProcessPoolExecutor(len(os.sched_getaffinity(0)),mp_context=get_context('spawn'))
    if kind=='diffusion':
        with pool() as workers:
            list(workers.map(collect,splits['train']+splits['valid']))
    train(kind,splits)
    variants={'nar':['nar'],'nae':['nae','nar_control'],'diffusion':['diffusion','none']}[kind]
    tasks=[(k,b,c) for b in BUDGETS for c in splits['test'] for k in variants]
    with pool() as workers:
        save(folder/'outcomes.json',list(workers.map(evaluate,tasks)))
