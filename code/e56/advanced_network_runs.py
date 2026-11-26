"""Train and run causal history, distillation, long-horizon value, stop and evader arms."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from network_scale_trials import OUT as SCALE, model_folder, CAPACITIES
import attention_diffusion_experiment as old
from network_geometry_experiment import encode, play
from run_1000_cruise import H80, outcome, save
from solvers.e53_evader_hmcts import EvaderHMCTS

OUT=SCALE/'advanced'
M=json.loads((SCALE/'manifest.json').read_text())
TRAIN=M['small']+M['extra']
VALID=M['valid']
DEV=M['dev']
BASE=model_folder('expanded','large',52)/'best.pt'
HIST=3


def read(path):
    return json.loads(path.read_text())


def load(model,path):
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    return model.eval()


def mlp(width,out):
    return nn.Sequential(nn.Linear(width,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,out))


class Temporal(old.Attention):
    def __init__(self):
        super().__init__(128,4)
        self.residual=mlp(34+14*HIST,1)
        self.context=np.zeros(14*HIST,np.float32)

    def forward(self,x):
        return super().forward(x[:,:34])+self.residual(x).flatten()

    @torch.inference_mode()
    def rank_actions(self,obs,endpoints,options,durations):
        x=np.stack([np.r_[encode(obs,e,o),self.context] for e,o in zip(endpoints,options)])
        return self(torch.from_numpy(x.astype('float32'))).numpy()


def groups(cases,history):
    rows=[]
    for c in cases:
        folder=old.PREVIOUS/'teacher'/c['case']
        data=read(folder/'groups.json')
        if not history:
            rows.extend((r['x'],r['visits']) for r in data)
            continue
        memories=np.zeros((2,HIST,14),np.float32);j=0
        for l in read(folder/'trace.json')['logs']:
            for agent in range(2):
                root=l['search'][str(agent)]['root']
                if len(root)>=2 and sum(v['visits'] for v in root)>0:
                    r=data[j];j+=1
                    assert r['visits']==[v['visits'] for v in root]
                    xx=np.array(r['x'])
                    rows.append((np.c_[xx,np.tile(memories[agent].flatten(),(len(xx),1))],r['visits']))
                memories[agent]=np.roll(memories[agent],-1,axis=0)
                memories[agent,-1]=root[0]['features'] if root else 0
        assert j==len(data)
    return rows


def pad(rows):
    n=len(rows);k=max(len(y) for x,y in rows)
    x=torch.zeros(n,k,len(rows[0][0][0])); y=torch.zeros(n,k); mask=y.bool()
    for i,(xx,yy) in enumerate(rows):
        j=len(yy);x[i,:j]=torch.as_tensor(np.asarray(xx));y[i,:j]=torch.as_tensor(yy);mask[i,:j]=True
    return x.cuda(),y.cuda(),mask.cuda()


def store_model(name,net):
    tmp=OUT/(name+'.tmp');torch.save(net.state_dict(),tmp);tmp.replace(OUT/(name+'.pt'))


def rank_train(name):
    temporal=name=='temporal'
    teacher=None
    if not temporal:
        path=model_folder('expanded','xl',52)
        while not (path/'done.json').exists():time.sleep(10)
        teacher=load(old.Attention(*CAPACITIES['xl']),path/'best.pt').cuda()
    train,valid=(pad(groups(split,temporal)) for split in (TRAIN,VALID))
    model=Temporal() if temporal else old.Attention(64,2)
    if temporal:model.load_state_dict(torch.load(BASE,weights_only=True,map_location='cpu'),strict=False)
    model.cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0002)
    best=float('inf')
    for epoch in range(30):
        for training,(x,y,mask) in ((True,train),(False,valid)):
            model.train(training);losses=[]
            ids=torch.randperm(len(x)) if training else torch.arange(len(x))
            with torch.set_grad_enabled(training):
                for idx in ids.split(48):
                    m=mask[idx];xx=x[idx][m];pred=y[idx].new_zeros(m.shape).masked_scatter(m,model(xx)).masked_fill(~m,-1e9)
                    target=F.normalize(y[idx],p=1,dim=1)
                    loss=-(target*pred.log_softmax(1)).sum(1).mean()
                    if training and teacher is not None:
                        with torch.no_grad():
                            logits=y[idx].new_zeros(m.shape).masked_scatter(m,teacher(xx)).masked_fill(~m,-1e9)
                        soft=-((logits/2).softmax(1)*(pred/2).log_softmax(1)).sum(1).mean()
                        loss=.5*loss+2*soft
                    if training:opt.zero_grad();loss.backward();opt.step()
                    losses.append((len(idx),float(loss.detach())))
            if not training:
                metric=sum(n*l for n,l in losses)/len(x)
                if metric<best:best=metric;store_model(name,model)
        save(OUT/(name+'_train.json'),dict(epoch=epoch+1,valid_loss=metric,best=best))
    save(OUT/(name+'_ready.json'),True)


def efeatures(e,p,v,a,active,boxes):
    # Each student's joint input has a fixed agent order and local clearances
    # derived from its permitted global map. Inactive vehicles are masked.
    rect=np.asarray(boxes).reshape(-1,2,2);local=[]
    for q in e:
        rel=rect-q[None,:,None];gap=np.maximum(np.maximum(rel[:,:,0],-rel[:,:,1]),0)
        near=rel[np.argsort(np.linalg.norm(gap,axis=1))[:4]].ravel()
        local.extend(np.pad(near,(0,16-len(near))))
    return np.concatenate([np.ravel(z) for z in (e,p,v,a,active,local)]).astype('float32')


def evader_rows(cases):
    xx=[];yy=[];mm=[]
    for c in cases:
        world=read(Path(c['path']));pos=np.array(world['worlds'][0]['initial'][:8]).reshape(4,2)
        vel=np.zeros((2,2));acc=vel.copy()
        for l in read(old.PREVIOUS/'teacher'/c['case']/'trace.json')['logs']:
            tracks=l['evader_minco'];active=[str(i) in tracks for i in range(2)]
            xx.append(efeatures(pos[:2],pos[2:],vel,acc,active,world['map']))
            target=np.zeros((2,2));weights=np.zeros((2,2))
            for key,t in tracks.items():
                i=int(key);pts=np.asarray(t['positions'])
                captured=any(ev['target']==i and ev['cycle']==l['cycle'] for ev in l['captures'])
                if len(pts)>1 and not captured:
                    target[i]=pts[-1]-pts[0];weights[i]=1
                vel[i]=t['velocities'][-1];acc[i]=t['accelerations'][-1]
            yy.append(target.ravel());mm.append(weights.ravel())
            pos=np.array(l['targets']+l['pursuers'])
    return [torch.tensor(np.array(z),dtype=torch.float32).cuda() for z in (xx,yy,mm)]


def evader_train():
    train,valid=map(evader_rows,(TRAIN,VALID))
    model=mlp(50,4).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0005);best=float('inf')
    for epoch in range(60):
        x,y,m=train
        for idx in torch.randperm(len(x)).split(128):
            loss=((model(x[idx])-y[idx]).square()*m[idx]).sum()/m[idx].sum().clamp_min(1)
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            x,y,m=valid;score=float(((model(x)-y).square()*m).sum()/m.sum())
        if score<best:best=score;store_model('evader',model)
        save(OUT/'evader_train.json',dict(epoch=epoch+1,valid_displacement_mse=score,best=best))
    if not np.isfinite(best) or not (OUT/'evader.pt').exists():
        raise RuntimeError('no finite evader validation checkpoint')
    save(OUT/'evader_ready.json',True)


class StudentEvader(EvaderHMCTS):
    def __init__(self,hybrid,**kwargs):
        super().__init__(**kwargs);self.hybrid=hybrid;self.net=load(mlp(50,4),OUT/'evader.pt')

    @torch.inference_mode()
    def search(self,problem,state,turn=0):
        start=time.perf_counter();e=state[:4].reshape(2,2);p=state[4:8].reshape(2,2)
        active=np.array([problem.is_active(state,i) for i in problem.evaders])
        x=efeatures(e,p,self.previous_velocity,self.previous_acceleration,active,self.public.boxes)
        self.desired=self.net(torch.from_numpy(x)).numpy().reshape(2,2)
        if self.hybrid:
            result=super().search(problem,state,turn)
            result.metadata['search_ms']=(time.perf_counter()-start)*1000
            return result
        # Teacher labels are one-second executed displacements. Map them to
        # two-second requests before the standard constrained MINCO executor.
        action=2*self.desired;action[~active]=0
        self.selected_cruise=(None,None);self.last_search={'simulations':0,'search_ms':(time.perf_counter()-start)*1000}
        return SimpleNamespace(best_action=np.r_[action.ravel(),np.zeros(4)],metadata=self.last_search)

    def _candidates(self,node):
        pairs=super()._candidates(node)
        if node.depth==0:
            # Progressive widening pops from the end. Compare like-for-like
            # one-second MINCO prefixes, not final waypoint displacements.
            def error(row):
                end=np.array([t['positions'][min(50,len(t['positions'])-1)] if t is not None else node.evaders[i]
                              for i,t in enumerate(row[2:4])])
                return float(np.linalg.norm((end-node.evaders-self.desired)[node.active_e]))
            pairs.sort(key=error,reverse=True)
        return pairs


def state_input(features,captured):
    f=np.asarray(features)
    return np.r_[f[4:14],captured].astype('float32')


class CaptureValue(nn.Module):
    def __init__(self):
        super().__init__();self.state=mlp(11,1);self.effect=mlp(25,1)
        nn.init.zeros_(self.effect[-1].weight);nn.init.zeros_(self.effect[-1].bias)

    def forward(self,x):
        return (self.state(x[:,:11])+self.effect(x)).flatten()


def value_rows(cases):
    rows=[];labels=[]
    for c in cases:
        # Only use training-map cases with a completed 80 s outcome.
        for path in sorted((H80/'results').glob('*/'+c['case']+'/trace.json')):
            try:trace=read(path)
            except (ValueError,OSError):continue
            if trace['requested_cycles']!=80 or (len(trace['logs'])<80 and len(trace['captures'])!=2):continue
            cap=0
            for l in trace['logs']:
                for search in l['search'].values():
                    if search['root']:
                        rows.append(state_input(search['root'][0]['features'],cap))
                        labels.append(len(trace['captures'])==2)
                cap=len(l['captures'])
    return torch.tensor(np.array(rows),dtype=torch.float32),torch.tensor(labels,dtype=torch.float32)


def train_state():
    train,valid=map(value_rows,(M['small'],VALID))
    if not all(len(y) and len(y.unique())==2 for _,y in (train,valid)):
        raise ValueError('state pretraining requires both complete outcome classes in each split')
    model=CaptureValue();opt=torch.optim.AdamW(model.state.parameters(),lr=.0003);best=float('inf')
    for epoch in range(35):
        x,y=train
        for idx in torch.randperm(len(x)).split(128):
            loss=F.binary_cross_entropy_with_logits(model.state(x[idx]).flatten(),y[idx])
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            x,y=valid;metric=float(F.binary_cross_entropy_with_logits(model.state(x).flatten(),y))
        if metric<best:best=metric;store_model('state',model)
    save(OUT/'state_ready.json',dict(train=len(train[1]),valid=len(valid[1]),best=best))


def value_input(root,idx,count):
    return np.r_[state_input(root[0]['features'],count),root[idx]['features']].astype('float32').tolist()


class SearchDriver:
    def __init__(self,arm,net=None,stop=None):
        self.arm=arm;self.net=net;self.stop=stop;self.history=np.zeros((2,HIST,14),np.float32)
        self.logs=[];self.at=None;self.intervention=None;self.points={};self.stop_records=[]

    def begin(self,controller,obs,captured):
        self.agent=obs.agent_id;self.cycle=obs.cycle;self.count=captured
        if self.arm=='temporal':
            self.net.context=self.history[self.agent].flatten()
        controller.search_experiment=self

    @torch.inference_mode()
    def root_ready(self,root):
        if self.arm!='value' or not root.children:return
        from solvers.e52_nar_nae import observation_features
        feats=[dict(features=observation_features(root.obs,c.endpoint,level=c.option)) for c in root.children]
        x=torch.tensor([value_input(feats,i,self.count) for i in range(len(feats))])
        scores=self.net(x).sigmoid().numpy()
        for c,v in zip(root.children,scores):c.value_hint=float(v-np.median(scores))

    @torch.inference_mode()
    def checkpoint(self,root,n):
        if self.arm not in ('collect','threshold','learned_stop'):return False
        ids=sorted(range(len(root.children)),key=lambda i:(root.children[i].visits,
                   root.children[i].total/max(1,root.children[i].visits)),reverse=True)
        a=root.children[ids[0]];b=root.children[ids[min(1,len(ids)-1)]]
        f=[n/128,a.visits/n,(a.visits-b.visits)/n,a.total/max(1,a.visits)-b.total/max(1,b.visits),
           a.prior,b.prior,self.cycle/80,self.count,len(root.obs.belief.particles)/8,len(ids)/32]
        self.points.setdefault((self.cycle,self.agent),[]).append(dict(n=n,x=f,idx=ids[0]))
        if self.arm=='threshold':return f[2]>=.4
        if self.arm=='learned_stop':return float(self.stop(torch.tensor(f,dtype=torch.float32)).sigmoid())<.5
        return self.intervention=='cut' and self.at==(self.cycle,self.agent,n)

    def after(self,cycle,observations,controllers,plans):
        for i in range(2):
            root=controllers[i].last_search['root']
            self.history[i]=np.roll(self.history[i],-1,axis=0)
            self.history[i,-1]=root[0]['features'] if root else 0
            if self.arm!='collect' or not root:continue
            idx=plans[i].candidate_id
            if not 0<=idx<len(root):continue
            if self.intervention=='alt' and self.at[:2]==(cycle,i):
                others=[j for j,c in enumerate(root) if c['visits'] and
                        not np.allclose(c['endpoint'],root[idx]['endpoint'])]
                if others:
                    j=max(others,key=lambda j:root[j]['visits']);r=root[j];end=np.array(r['endpoint'])
                    plans[i]=replace(plans[i],candidate_id=j,endpoint=end,action=end-observations[i].pose,
                                     duration=r['duration'],pass_through=r['pass_through'],visits=r['visits'],value=r['value'])
                    idx=j
            checkpoints=self.points.get((cycle,i),[])
            self.logs.append(dict(cycle=cycle,agent=i,cap=self.count,idx=idx,point=plans[i].endpoint.tolist(),
                                  state=value_input(root,idx,self.count),checkpoints=checkpoints))


def simulate(case,folder,arm='collect',intervention=None,at=None):
    cfg=dict(e52_budget=128,e52_mode='nar')
    nar=load(old.Attention(128,4),BASE);net=None;stop=None
    if arm=='temporal':net=nar=load(Temporal(),OUT/'temporal.pt')
    if arm=='distill':nar=load(old.Attention(64,2),OUT/'distill.pt')
    if arm=='value':net=load(CaptureValue(),OUT/'value.pt')
    if arm=='learned_stop':stop=load(mlp(10,1),OUT/'stop.pt')
    if arm in ('distill','temporal','evader_direct','evader_hybrid','evader_teacher'):cfg['e52_budget']=64
    if arm.startswith('evader_') and arm!='evader_teacher':
        hybrid=arm=='evader_hybrid'
        cfg.update(evader_factory=lambda **kw:StudentEvader(hybrid,**kw),e53_simulations=32 if hybrid else 1)
    d=SearchDriver(arm,net,stop);d.intervention=intervention;d.at=at
    cfg.update(nar_override=nar,search_experiment=d,decision_hook=d.after)
    start=time.monotonic()
    trace=play(case,folder,**cfg)
    return dict(**outcome(trace),elapsed=time.monotonic()-start,
                search_ms=[s['elapsed_ms'] for l in trace['logs'] for s in l['search'].values()],
                evader_ms=[l['evader_search'].get('search_ms',0) for l in trace['logs']],
                simulations=[s['simulations'] for l in trace['logs'] for s in l['search'].values()]),d


def collect(case):
    folder=OUT/'pairs'/case['case'];folder.mkdir(parents=True,exist_ok=True)
    dest=folder/'result.json'
    if dest.exists():return
    # Persist each full trajectory outcome and the small decision records,
    # release the bulky sensor trace before replaying.
    def run(stage,at=None):
        path=folder/(stage+'.json')
        if path.exists():return read(path)
        with TemporaryDirectory(dir=folder) as tmp:
            stats,d=simulate(case,Path(tmp),intervention=stage,at=at)
        record=dict(stats=stats,rows=d.logs);save(path,record);return record
    ref=run('reference');rows=ref['rows'];rng=np.random.default_rng(case['seed'])
    candidates=[(r,a) for r in rows for a in r['checkpoints'] if a['idx']!=r['idx']]
    if not candidates:candidates=[(r,a) for r in rows for a in r['checkpoints']]
    if not candidates:save(dest,dict(samples=[]));return
    preferred=[pair for pair in candidates if pair[0]['cap']==(case['seed']%2)]
    choices=preferred or candidates
    row,cp=choices[int(rng.integers(len(choices)))];at=(row['cycle'],row['agent'],cp['n'])
    samples=[]
    for kind in ('cut','alt'):
        alt=run(kind,at);rr=next(r for r in alt['rows'] if (r['cycle'],r['agent'])==at[:2])
        samples.append(dict(kind=kind,at=at,x=cp['x'],a=row['state'],b=rr['state'],
                            changed=not np.allclose(row['point'],rr['point']),
                            reference=ref['stats'],alternative=alt['stats']))
    save(dest,dict(samples=samples))


def get_pairs(cases):
    return [r for c in cases for r in read(OUT/'pairs'/c['case']/'result.json')['samples']]


def utility(s):
    return (s['captures']==2)+.25*s['captures']/2+.15*(80-s['double'])/80


def fit_paired(kind):
    tr,va=map(get_pairs,(TRAIN,VALID))
    initial=OUT/'historical_value.pt' if (OUT/'historical_value.pt').exists() else OUT/'state.pt'
    model=load(CaptureValue(),initial) if kind=='value' else mlp(10,1)
    def build(rows):
        if kind=='stop':
            rows=[r for r in rows if r['kind']=='cut']
            return torch.tensor([r['x'] for r in rows]),torch.tensor([float(utility(r['reference'])>utility(r['alternative'])+.01) for r in rows])
        rows=[r for r in rows if r['changed'] and utility(r['reference'])!=utility(r['alternative'])]
        return torch.tensor([[r['a'],r['b']] for r in rows]),torch.tensor([float(utility(r['reference'])>utility(r['alternative'])) for r in rows])
    def logits(x):
        return model(x).flatten() if kind=='stop' else model(x[:,0])-model(x[:,1])
    train,valid=build(tr),build(va)
    if min(len(train[1]),len(valid[1]))==0:raise ValueError('no informative paired labels for '+kind)
    opt=torch.optim.AdamW(model.parameters(),lr=.0005);best=float('inf')
    for epoch in range(100):
        x,y=train
        for idx in torch.randperm(len(x)).split(32):
            loss=F.binary_cross_entropy_with_logits(logits(x[idx]),y[idx])
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            x,y=valid;score=float(F.binary_cross_entropy_with_logits(logits(x),y))
        if score<best:best=score;store_model(kind,model)
    save(OUT/(kind+'_ready.json'),dict(train=len(train[1]),valid=len(valid[1]),best=best))


def evaluate(task):
    arm,c=task;folder=OUT/'eval'/arm/c['case'];p=folder/'result.json'
    if arm=='evader_teacher':
        from reuse_teacher_results import share, timing_cases
        share(c,timing_cases(DEV))
    if p.exists():return
    folder.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=folder) as tmp:
        row,_=simulate(c,Path(tmp),arm)
    save(p,row)


def parallel(fn,items):
    with ProcessPoolExecutor(len(os.sched_getaffinity(0))) as pool:
        for _ in pool.map(fn,items):pass


def main(phase):
    OUT.mkdir(parents=True,exist_ok=True);torch.manual_seed(52);torch.set_num_threads(1)
    if phase=='evader':evader_train();return
    if phase=='state':train_state();return
    if phase=='collect':
        # Start both train and validation map episodes in the first wave.
        cases=sorted(TRAIN+VALID,key=lambda c:(c['case'][-3:],c['case']))
        parallel(collect,cases);save(OUT/'pairs_ready.json',True);return
    if phase=='paired':
        while not all((OUT/(k+'_ready.json')).exists() for k in ('state','pairs')):time.sleep(20)
        for k in ('stop','value'):fit_paired(k)
        from paired_backfill import claim
        parallel(claim,[(arm,c) for c in DEV for arm in ('value','learned_stop')]);return
    else:
        arms={'eval-distill':['distill'],'eval-temporal':['temporal'],
              'eval-evader':['evader_direct','evader_hybrid','evader_teacher'],
              'threshold':['threshold']}[phase]
    parallel(evaluate,[(arm,c) for c in DEV for arm in arms])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase');main(parser.parse_args().phase)
