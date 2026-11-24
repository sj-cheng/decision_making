"""Paired E56 geometry-input ablation and leaf critic interventions."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from compare_nae_1000 import H80, MODEL, BRANCH, DEST, config, run, save, outcome
from run_e55_parallel import Shadow
from solvers.e52_nar_nae import observation_features
from solvers.e52_hmcts import HierarchicalMCTS

OUT = H80/'geometry_network_ablation'
KINDS = ('base','geometry','graph')
CFG = dict(config(2), cycles=80, environment_horizon_floor=160, evader_cruise=True,
           e52_nae_strategy='budget_leaf_residual')


def encode(obs, endpoint, option):
    base = observation_features(obs,endpoint,level=option)
    boxes = np.asarray(obs.local_map.obstacle_boxes()).reshape(-1,2,2)-obs.pose[None,:,None]
    # Four nearest observed boxes to the proposed motion segment. Coordinates
    # and half sizes also specify clearance and inter-obstacle corridor widths.
    delta = np.asarray(endpoint)-obs.pose
    points = np.linspace(np.zeros(2),delta,5)
    gap = np.maximum(np.maximum(boxes[None,:,:,0]-points[:,None],points[:,None]-boxes[None,:,:,1]),0)
    distance = np.linalg.norm(gap,axis=2).min(axis=0)
    closest = boxes[np.argsort(distance,kind='stable')[:4]]
    nodes = np.zeros((4,5),np.float32)
    nodes[:len(closest),:2] = closest.mean(axis=2)
    nodes[:len(closest),2:4] = (closest[:,:,1]-closest[:,:,0])/2
    nodes[:len(closest),4] = 1
    return np.concatenate([base,nodes.ravel()])


class Ranker(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind=='graph':
            self.node = nn.Linear(5,48)
            self.message = nn.Linear(96,48)
        width = dict(base=14,geometry=34,graph=62)[kind]
        self.head = nn.Sequential(nn.Linear(width,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))

    def forward(self,x):
        base=x[...,:14]
        if self.kind=='graph':
            nodes=x[...,14:].reshape(*x.shape[:-1],4,5)
            mask=nodes[...,4]
            # Edges are derived only from the same boxes supplied to the MLP.
            gap=(nodes[..., :,None,:2]-nodes[...,None,:,:2]).abs()-nodes[..., :,None,2:4]-nodes[...,None,:,2:4]
            adj=torch.exp(-gap.clamp_min(0).norm(dim=-1))*mask.unsqueeze(-2)*mask.unsqueeze(-1)
            h=self.node(nodes).relu()
            neighbors=(adj@h)/adj.sum(-1,keepdim=True).clamp_min(1)
            h=self.message(torch.cat([h,neighbors],dim=-1)).relu()
            x=torch.cat([base,(h*mask.unsqueeze(-1)).sum(-2)/mask.sum(-1,keepdim=True).clamp_min(1)],dim=-1)
        elif self.kind=='base':
            x=base
        return self.head(x).squeeze(-1)

    @torch.inference_mode()
    def score(self,obs,endpoint,option):
        return self(torch.from_numpy(encode(obs,endpoint,option))).item(),0.


def manifest():
    cases=json.loads((H80/'paired_500.json').read_text())['cases']
    groups=defaultdict(list)
    for c in cases:
        boxes=json.loads(Path(c['path']).read_text())['map']
        key=hashlib.sha256(np.asarray(boxes,dtype='float64').tobytes()).hexdigest()
        groups[key].append(c)
    keys=sorted(groups)
    return {k:[c for key in selected for c in groups[key] if int(c['case'][-3:])<limit]
            for k,selected,limit in [('train',keys[:-4],3),('valid',keys[-4:-2],3),('test',keys[-2:],5)]}


def play(case,folder,**overrides):
    return run(case['path'],folder/'trace.json',seed=case['seed'],**dict(CFG,**overrides))


def teach(case):
    folder=OUT/'teacher'/case['case']
    if (folder/'groups.json').exists(): return
    rows=[]
    def record(cycle,obs,controllers,plans):
        for i in (0,1):
            roots=controllers[i].last_search['root']
            visits=[r['visits'] for r in roots]
            if len(roots)<2 or sum(visits)==0: continue
            x=[encode(obs[i],r['endpoint'],HierarchicalMCTS.OPTIONS.index(r['option'])).tolist() for r in roots]
            rows.append(dict(x=x,visits=visits))
    play(case,folder,decision_hook=record,e52_budget=2048,cycles=40)
    save(folder/'groups.json',rows)


def data(cases):
    rows=[r for c in cases for r in json.loads((OUT/'teacher'/c['case']/'groups.json').read_text())]
    width=max(len(r['visits']) for r in rows)
    x=torch.zeros(len(rows),width,34);y=torch.zeros(len(rows),width);mask=torch.zeros_like(y,dtype=torch.bool)
    for i,r in enumerate(rows):
        n=len(r['x']);x[i,:n]=torch.tensor(r['x']);v=torch.tensor(r['visits']);y[i,:n]=v/v.sum();mask[i,:n]=True
    return x,y,mask


def loss(model,data,idx):
    x,y,mask=data
    z=model(x[idx]).masked_fill(~mask[idx],-1e9)
    return -(y[idx]*z.log_softmax(-1)).sum(-1).mean()


def train(kind,splits,device):
    trainset=tuple(t.to(device) for t in data(splits['train']))
    valid=tuple(t.to(device) for t in data(splits['valid']))
    torch.manual_seed(52);model=Ranker(kind).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001)
    rng=torch.Generator().manual_seed(52); best=float('inf')
    for epoch in range(40):
        for idx in torch.randperm(len(trainset[0]),generator=rng).split(64):
            opt.zero_grad();l=loss(model,trainset,idx);l.backward();opt.step()
        with torch.no_grad(): v=loss(model,valid,slice(None)).item()
        if v<best:
            best=v;torch.save(model.state_dict(),OUT/(kind+'.pt'))
    return dict(validation_cross_entropy=best,device=str(device),parameters=sum(p.numel() for p in model.parameters()))


def test(task):
    case,kind=task
    folder=OUT/'evaluation'/kind/case['case']
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    model=Ranker(kind);model.load_state_dict(torch.load(OUT/(kind+'.pt'),weights_only=True,map_location='cpu'));model.eval()
    row=dict(case=case['case'],kind=kind,**outcome(play(case,folder,e52_mode='nar',nar_override=model)))
    save(folder/'result.json',row);return row


def diagnose(task):
    case,kind=task
    folder=OUT/'diagnosis'/kind/case['case']
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    cfg=dict(e52_mode='both',e52_checkpoint=str(MODEL if kind=='branch' else DEST/'new_nar_old_value.pt'),
             e52_advantage_checkpoint=str(BRANCH) if kind=='branch' else None)
    hook=Shadow(2,False)
    ref=play(case,folder/'reference',decision_hook=hook,audit_nae=True,**cfg)
    row=dict(case=case['case'],kind=kind,reference=outcome(ref),first_flip=hook.flip,decisions=len(hook.rows)*2)
    for k in ('queried','root_action_changed','executed_plan_changed','rank_changed'):
        row[k]=sum(sum(r[k]) for r in hook.rows)
    for k in ('peer_groups','peer_reorders','leaf_winner_changes'):
        row[k]=sum(s[k] for l in ref['logs'] for s in l['search'].values())
    row['decisions_with_sibling_reorder']=sum(s['peer_reorders']>0 for l in ref['logs'] for s in l['search'].values())
    if hook.flip is not None:
        ctrl=Shadow(2,True)
        replay=play(case,folder/'intervention',decision_hook=ctrl,**cfg)
        assert ctrl.flip==hook.flip
        for a,b in zip(ref['logs'][:hook.flip],replay['logs'][:hook.flip]):
            assert a['targets']==b['targets'] and a['pursuers']==b['pursuers']
        row['without_first_changed_action']=outcome(replay)
    save(folder/'decisions.json',hook.rows);save(folder/'result.json',row);return row


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['nar','nae']);parser.add_argument('--workers',type=int,required=True)
    parser.add_argument('--training-device',default='cpu')
    args=parser.parse_args();torch.set_num_threads(1)
    splits=manifest();save(OUT/(args.phase+'_manifest.json'),splits)
    with ProcessPoolExecutor(args.workers) as pool:
        if args.phase=='nae':
            cases=json.loads((H80/'paired_500.json').read_text())['cases']
            tasks=[(c,k) for c in cases if c['case'].endswith('_000') for k in ('frozen','branch')]
            save(OUT/'nae_outcomes.json',list(pool.map(diagnose,tasks)))
        else:
            list(pool.map(teach,splits['train']+splits['valid']))
            save(OUT/'validation.json',{k:train(k,splits,args.training_device) for k in KINDS})
            save(OUT/'nar_outcomes.json',list(pool.map(test,[(c,k) for c in splits['test'] for k in KINDS])))
