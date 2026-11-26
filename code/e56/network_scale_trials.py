"""Paired data/capacity, search budget, leaf return and proposal experiments."""
import argparse
import copy
import fcntl
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch
from torch.nn import functional as F
import attention_diffusion_experiment as prior
from network_geometry_experiment import teach, manifest, encode, play
from run_1000_cruise import H80, save, outcome, _legal
from solvers.e52_hmcts import Node

OUT=H80/'network_scale_trials'
SEEDS=(52,73,97)
CAPACITIES={'small':(64,2), 'large':(128,4), 'xl':(192,6), 'xxl':(256,8)}
LARGER=('xl','xxl')


def model_folder(size, capacity, seed):
    parent='larger_models' if capacity in LARGER else 'models'
    return OUT/parent/f'{size}_{capacity}_{seed}'


def make_manifest():
    old=manifest()
    templates={}
    for c in old['train']:
        x=json.loads(Path(c['path']).read_text())
        templates.setdefault(str(x['map']),x)
    templates=list(templates.values())
    rng=np.random.default_rng(57241)
    all_maps={str(json.loads(Path(c['path']).read_text())['map']) for cs in old.values() for c in cs}
    def generate(name,sizes):
        cases=[]
        for index,count in enumerate(sizes):
            template=templates[index%len(templates)]
            boxes=np.array(template['map'],float)
            # Perturb individual interior rectangles. Preserve arena bounds
            # and keep positive-width passages between initially separated boxes.
            for j,box in enumerate(boxes):
                if np.abs(box).max()>9:continue
                for _ in range(100):
                    moved=box+rng.uniform(-1,1,(2,1))
                    others=np.delete(boxes,j,axis=0)
                    separation=np.maximum(moved[:,0]-others[:,:,1],others[:,:,0]-moved[:,1]).max(axis=1)
                    if np.abs(moved).max()<9 and np.all(separation>.25):
                        boxes[j]=moved;break
            key=str(boxes.tolist())
            assert key not in all_maps
            all_maps.add(key)
            for n in range(count):
                pts=[]
                while len(pts)<4:
                    q=rng.uniform(-8.7,8.7,2)
                    if _legal(q,boxes) and all(np.linalg.norm(q-v)>=(3 if len(pts)>=2 and i<2 else 2) for i,v in enumerate(pts)):pts.append(q)
                state=list(template['worlds'][0]['initial'])
                state[:8]=np.array(pts).flatten().tolist()
                state[8:13]=[0,1,1,1,1]
                case=f'{name}_{index:02}_{n:03}';p=OUT/'cases'/f'{case}.json'
                save(p,dict(map=boxes.tolist(),worlds=[dict(initial=state)],horizon=160))
                cases.append(dict(case=case,path=str(p),seed=int(rng.integers(2**30))))
        return cases
    save(OUT/'manifest.json',dict(small=old['train'],valid=old['valid'],extra=generate('extra',[11]*11),
         dev=generate('dev',[8]*6),formal=generate('formal',[25]*20)))


def tensor_data(cases):
    return prior.teacher_tensors(cases)


def fit(size,capacity,seed,m,datasets=None):
    folder=model_folder(size,capacity,seed)
    if (folder/'done.json').exists():return
    torch.manual_seed(seed)
    width,layers=CAPACITIES[capacity]
    model=prior.Attention(width,layers).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.0005)
    samples=m['small']+(m['extra'] if size=='expanded' else [])
    tr,val=datasets if datasets is not None else (tensor_data(samples),tensor_data(m['valid']))
    best=float('inf');hist=[]
    for epoch in range(40):
        for training,tensors in ((True,tr),(False,val)):
            model.train(training);n=len(tensors[0]);total=0
            ids=torch.randperm(n) if training else torch.arange(n)
            for ix in ids.split(32):
                batch=[x[ix].cuda() for x in tensors]
                with torch.set_grad_enabled(training):loss=prior.loss_att(model,batch,'nar')
                if training:
                    opt.zero_grad();loss.backward();opt.step()
                total+=len(ix)*loss.item()
            if not training:
                score=total/n;hist.append(score)
                if score<best:
                    best=score;folder.mkdir(exist_ok=True,parents=True)
                    torch.save(model.state_dict(),folder/'best.tmp');(folder/'best.tmp').replace(folder/'best.pt')
        save(folder/'progress.json',dict(epoch=epoch+1,validation=hist,best=best))
    save(folder/'done.json',dict(seed=seed,width=width,layers=layers,best=best,groups=len(tr[0])))


class LeafPairs(prior.Attention):
    def __init__(self):
        super().__init__()
        self.rows=[];self.sampled=set()

    def record_leaf(self,controller,parent,scores,bonuses):
        peers=parent.children;raw=np.array([p.reward for p in peers])
        geo=int(raw.argmax());model=int(np.argmax(raw+controller.gamma*bonuses))
        # One disagreement and one agreement per agent per four root cycles.
        key=(parent.obs.agent_id,(parent.obs.cycle-parent.depth)//4,geo!=model)
        if key in self.sampled:return
        self.sampled.add(key)
        other=model if model!=geo else int(np.argsort(-raw,kind='stable')[1])
        choices=[peers[i] for i in (geo,other)]
        planner=copy.copy(controller);planner.nae=None;planner.nar=None
        returns=[]
        for child in choices:
            node=Node(child.obs,depth=child.depth);value=child.reward
            for t in range(1,9):
                planner.evaluations=0;planner._expand(node,None)
                if not node.children:break
                node=max(node.children,key=lambda c:c.reward)
                value+=planner.gamma**t*node.reward
            returns.append(value)
        self.rows.append(dict(features=[encode(parent.obs,c.endpoint,c.option).tolist() for c in choices],
            returns=returns, scores=[float(scores[i]) for i in (geo,other)],
            rewards=[float(raw[i]) for i in (geo,other)],disagree=geo!=model,depth=parent.depth))


class ProposalAudit:
    def __init__(self,kind,seed):
        self.net=prior.Diffusion()
        self.net.load_state_dict(torch.load(prior.OUT/'diffusion/model.pt',map_location='cpu',weights_only=True))
        self.net.eval();self.net.rng.manual_seed(seed)
        self.rng=np.random.default_rng(seed);self.kind=kind;self.pending={};self.accepted={};self.records=[]

    def propose(self,obs):
        if self.kind=='diffusion':out=self.net.propose(obs)
        else:
            from solvers.e52_hmcts import single_agent_endpoints,reachable_waypoint
            p=single_agent_endpoints(obs.pose,obs.belief.particles,obs.belief.weights,obs.local_map,
                max_distance=2.,max_candidates=8,heading=obs.heading,cycle=obs.cycle,
                previous_action=obs.previous_prefix,agent_id=obs.agent_id)
            anchors=[reachable_waypoint(obs.pose,x,obs.velocity,2.,1.125,1.5) for x in p] or [obs.pose]
            out=[anchors[self.rng.integers(len(anchors))]+self.rng.normal(0,.35,2) for _ in range(2)]
        self.pending[obs.agent_id]=np.asarray(out)
        self.accepted[obs.agent_id]=[]
        return out

    def record_candidate(self, agent, index):
        self.accepted[agent].append(index)

    def inspect(self,cycle,obs,controllers,plans):
        for i in range(2):
            ids=self.accepted.pop(i,[])
            roots=controllers[i].last_search['root']
            row=dict(cycle=cycle,agent=i,generated=len(self.pending.pop(i,[])),
                     feasible=len(ids),visited=sum(roots[j]['visits']>0 for j in ids),
                     selected=plans[i].candidate_id in ids)
            self.records.append(row)


def job(task, wait=True):
    method,b,c=task
    if method=='teacher':
        return run_case(task)
    folder=OUT/'episodes'/method/f'b{b}'/c['case']
    folder.mkdir(exist_ok=True,parents=True)
    with (folder/'job.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX if wait else fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        return run_case(task)


def run_case(task):
    method,b,c=task
    if method=='teacher':
        teach(c);return
    folder=OUT/'episodes'/method/f'b{b}'/c['case'];p=folder/'result.json'
    if p.exists():return json.loads(p.read_text())
    cfg=dict(e52_budget=b)
    if method=='leaf':
        net=LeafPairs();net.load_state_dict(torch.load(prior.OUT/'nae/model.pt',map_location='cpu',weights_only=True));net.eval()
        cfg.update(nae_override=net,e52_mode='both',e52_checkpoint=str(prior.MODEL))
    elif method in ('root_nae','leaf_nae'):
        net=prior.Attention()
        weights=OUT/'leaf_candidate.pt' if method=='leaf_nae' else prior.OUT/'nae/model.pt'
        net.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));net.eval()
        cfg.update(nae_override=net,e52_mode='both',e52_checkpoint=str(prior.MODEL))
    elif method in ('jitter','diffusion'):
        audit=ProposalAudit(method,c['seed']);cfg.update(waypoint_generator=audit,decision_hook=audit.inspect)
    elif method!='none':
        if method=='original':
            net=prior.Attention();path=prior.OUT/'nar/model.pt'
        else:
            size,cap,seed=method.split('_')
            net=prior.Attention(*CAPACITIES[cap])
            path=model_folder(size,cap,seed)/'best.pt'
        net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));net.eval()
        cfg.update(nar_override=net,e52_mode='nar')
    folder.mkdir(exist_ok=True,parents=True)
    # Persist the requested statistics; keep large raw traces temporary.
    with TemporaryDirectory(dir=folder) as tmp:
        x=play(c,Path(tmp),**cfg)
        row=dict(case=c['case'],method=method,budget=b,**outcome(x),
            search_ms=[r['elapsed_ms'] for l in x['logs'] for r in l['search'].values()])
        if method=='leaf':save(folder/'pairs.json',net.rows)
        if method in ('jitter','diffusion'):
            for z in audit.records:
                track=x['logs'][z['cycle']]['minco'].get(str(z['agent']),{})
                z['executed']=bool(z['selected'] and not track.get('continued_braking_tail') and
                    len(track.get('executed_positions',[]))>1 and np.linalg.norm(np.subtract(track['executed_positions'][-1],track['executed_positions'][0]))>1e-6)
            save(folder/'proposal_audit.json',audit.records)
        save(p,row);return row


def steal(task):
    return job(task,wait=False)


def batch(tasks, worker=job):
    with ProcessPoolExecutor(len(os.sched_getaffinity(0)),mp_context=get_context('spawn')) as pool:
        return list(pool.map(worker,tasks,chunksize=1))


def main(phase):
    torch.set_num_threads(1)
    if phase=='init':make_manifest();return
    m=json.loads((OUT/'manifest.json').read_text())
    if phase=='larger_models':
        from network_geometry_experiment import data
        def labels(cases):
            # data returns normalized teacher visits. loss_att normalizes them
            # again and only uses the Q tensor to allocate its output.
            x,visits,mask=data(cases)
            return x,visits,torch.zeros_like(visits),mask
        datasets=labels(m['small']+m['extra']),labels(m['valid'])
        for seed in SEEDS:
            for capacity in LARGER:
                print(f'Training {capacity} seed {seed}',flush=True)
                fit('expanded',capacity,seed,m,datasets)
    if phase=='larger_eval':
        for seed in SEEDS:
            for capacity in ('large',*LARGER):
                folder=model_folder('expanded',capacity,seed)
                while not (folder/'done.json').exists():time.sleep(15)
                rows=batch([(folder.name,64,c) for c in m['dev']])
                save(OUT/'larger_eval'/f'{folder.name}.json',rows)
    if phase=='teacher':
        batch([('teacher',2048,c) for c in m['extra']]);save(OUT/'extra_ready.json',True)
    if phase=='models':
        for size in ('small','expanded'):
            if size=='expanded':
                while not (OUT/'extra_ready.json').exists():time.sleep(30)
            for seed in SEEDS:
                for capacity in ('large','small'):
                    fit(size,capacity,seed,m)
    if phase=='budget_help':
        tasks=[(k,b,c) for c in reversed(m['dev']) for b in (256,128,64) for k in ('none','original')]
        batch(tasks,steal)
    if phase=='budget':
        rs=batch([(k,b,c) for c in m['dev'] for b in (64,128,256) for k in ('original','none')])
        save(OUT/'budget.json',rs)
    if phase=='leaf':
        batch([('leaf',128,c) for c in m['small']+m['valid']])
        leaf_fit(m)
    if phase=='proposal':
        save(OUT/'proposal.json',batch([(k,256,c) for c in m['dev'] for k in ('diffusion','jitter')]))
    if phase=='assist':
        # The leaf lane starts this phase after training. Evaluate both label
        # choices with the same NAR and the same held-out starts.
        if os.sched_getaffinity(0)==set(range(37,42))|set(range(93,98)):
            leaf_tasks=[(k,128,c) for c in m['dev'] for k in ('root_nae','leaf_nae')]
            save(OUT/'leaf_online.json',batch(leaf_tasks))
        while not (OUT/'formal.json').exists():
            tasks=[]
            for path in sorted((OUT/'models').glob('*/done.json')):
                tasks.extend((path.parent.name,128,c) for c in reversed(m['dev']))
            selected_file=OUT/'selection.json'
            final=[('none',128),('none',256)]
            if selected_file.exists():
                final.insert(0,(json.loads(selected_file.read_text())['selected'],128))
            tasks.extend((k,b,c) for c in reversed(m['formal']) for k,b in final)
            missing=[(k,b,c) for k,b,c in tasks if not (OUT/'episodes'/k/f'b{b}'/c['case']/'result.json').exists()]
            if missing:
                count=len(os.sched_getaffinity(0))
                offset=(min(os.sched_getaffinity(0))*count)%len(missing)
                queue=missing[offset:]+missing[:offset]
                batch(queue[:4*count],steal)
            else:time.sleep(30)
    if phase=='selection':
        methods=[f'{size}_{cap}_{seed}' for size in ('small','expanded') for cap in ('small','large') for seed in SEEDS]
        rows=[]
        for method in methods:
            while not (OUT/'models'/method/'done.json').exists():time.sleep(10)
            rows+=batch([(method,128,c) for c in m['dev']])
        def score(group):
            a=[r for r in rows if r['method'].rsplit('_',1)[0]==group]
            return (-sum(r['captures']==2 for r in a),np.mean([r['double'] for r in a]),group)
        winner=min({k.rsplit('_',1)[0] for k in methods},key=score)
        # Predeclared seed, never select a lucky seed on held-out performance.
        selected=winner+'_52'
        save(OUT/'selection.json',dict(selected=selected,dev=rows))
        save(OUT/'formal.json',batch([(k,b,c) for c in m['formal'] for k,b in [(selected,128),('none',128),('none',256)]]))


def leaf_fit(m):
    def load(cases):
        rows=[r for c in cases for r in json.loads((OUT/'episodes/leaf/b128'/c['case']/'pairs.json').read_text())
              if abs(r['returns'][0]-r['returns'][1])>1e-6]
        x=torch.tensor([r['features'] for r in rows],dtype=torch.float32).cuda()
        y=torch.tensor([int(r['returns'][0]>r['returns'][1]) for r in rows],dtype=torch.float32).cuda()
        return x,y
    torch.manual_seed(52);model=prior.Attention().cuda()
    model.load_state_dict(torch.load(prior.OUT/'nae/model.pt',weights_only=True))
    opt=torch.optim.AdamW(model.parameters(),lr=.0002)
    x,y=load(m['small']);v,t=load(m['valid']);scores=[]
    for epoch in range(21):
        with torch.no_grad():
            pred=model(v.flatten(0,1)).reshape(-1,2)
            scores.append(float(((pred[:,0]>pred[:,1])==t.bool()).float().mean()))
        if scores[-1]>=max(scores):torch.save(model.state_dict(),OUT/'leaf_candidate.pt')
        if epoch==20:break
        for idx in torch.randperm(len(x)).split(64):
            s=model(x[idx].flatten(0,1)).reshape(-1,2)
            loss=F.binary_cross_entropy_with_logits(s[:,0]-s[:,1],y[idx])
            opt.zero_grad();loss.backward();opt.step()
    save(OUT/'leaf_validation.json',dict(validation_accuracy=scores,train_pairs=len(x),valid_pairs=len(v)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('init','teacher','models','budget','budget_help','leaf','proposal','selection','assist','larger_models','larger_eval'))
    main(parser.parse_args().phase)
