"""Role-free joint continuous policy and public-belief team value.

No target assignment or role label enters any network. Inactive action
coordinates are marginalized, not fitted as point densities at zero.
"""
import numpy as np
import torch
from torch import nn

KINDS=('single','factorized','joint')
KEYS=('maps','map_mask','agents','particles','weights','history','history_mask','public')
K=8


def public_features(boxes,positions,active_p,active_e,samples,weights,known,records,speed,remaining):
    center=positions[np.asarray(active_p,bool)].mean(0) if np.any(active_p) else positions.mean(0)
    maps=np.zeros((8,4));mask=np.zeros(8);boxes=np.asarray(boxes).reshape(-1,2,2)
    if len(boxes)>8:raise ValueError('Map capacity exceeded')
    maps[:len(boxes),:2]=(boxes.mean(2)-center)/20.;maps[:len(boxes),2:]=(boxes[:,:,1]-boxes[:,:,0])/20.;mask[:len(boxes)]=1
    agents=np.zeros((2,6));agents[:,:2]=(positions-center)/20.;agents[:,4]=active_p;agents[:,5]=speed/2.
    if len(records)>1:agents[:,2:4]=(positions-np.asarray(records[-2]['positions']))/2.
    particles=np.zeros((2,64,3));outweights=np.zeros((2,64))
    for j in range(2):
        if active_e[j]:
            if abs(np.sum(weights[j])-1)>1e-10:raise ValueError('Invalid target marginal')
            particles[j,:,:2]=(samples[j]-center)/20.;particles[j,:,2]=weights[j];outweights[j]=weights[j]
    history=np.zeros((16,14));hm=np.zeros(16);step=80-remaining
    for i,r in enumerate(records[-16:]):
        history[i,0]=(step-r['step'])/80.;history[i,1:5]=(np.asarray(r['positions'])-center).reshape(-1)/20.
        for j in range(2):
            reports=[x for x in r['reports'][j] if x is not None]
            if reports:history[i,5+2*j:7+2*j]=(np.asarray(reports[0])-center)/20.;history[i,9+j]=1.
        history[i,11:13]=r['active_p'];history[i,13]=r['remaining']/80.;hm[i]=1
    pub=np.r_[center/10.,remaining/80.,speed/2.,active_e,[q is not None for q in known]].astype(float)
    return dict(maps=maps,map_mask=mask,agents=agents,particles=particles,weights=outweights,
        history=history,history_mask=hm,public=pub)


class Network(nn.Module):
    def __init__(self,kind,width=None):
        super().__init__()
        if kind not in KINDS:raise ValueError(kind)
        self.kind=kind
        outputs={'single':9,'factorized':2*K*5,'joint':K*9}[kind]
        if width is None:
            width=min(range(64,193),key=lambda h:abs(sum(n*32+32+32*32+32 for n in (4,6,3,14))+(200*h+h+h*h+h)+(h+1)*(outputs+1)-65000))
        self.width=width
        for name,n in (('map_encoder',4),('agent_encoder',6),('particle_encoder',3),('history_encoder',14)):
            setattr(self,name,nn.Sequential(nn.Linear(n,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU()))
        self.trunk=nn.Sequential(nn.Linear(200,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU())
        self.policy=nn.Linear(width,outputs);self.value=nn.Linear(width,1)
        nn.init.zeros_(self.value.weight);nn.init.zeros_(self.value.bias)

    def forward(self,x):
        m=(self.map_encoder(x['maps'])*x['map_mask'][...,None]).sum(1)
        a=self.agent_encoder(x['agents']).flatten(1)
        p=(self.particle_encoder(x['particles'])*x['weights'][...,None]).sum(2).flatten(1)
        h=(self.history_encoder(x['history'])*x['history_mask'][...,None]).sum(1)/x['history_mask'].sum(1,keepdim=True).clamp_min(1)
        z=self.trunk(torch.cat((m,a,p,h,x['public']),-1));y=self.policy(z)
        if self.kind=='factorized':y=y.reshape(-1,2,K,5);parameters=(y[...,0],y[...,1:3].clamp(-6,6),y[...,3:5].clamp(-3,1))
        else:
            y=y.reshape(-1,1 if self.kind=='single' else K,9);parameters=(y[...,0],y[...,1:5].clamp(-6,6),y[...,5:9].clamp(-3,1))
        return parameters,self.value(z)[:,0]

    def export(self,path):
        np.savez_compressed(path,kind=np.array(self.kind),width=np.array(self.width),
            **{k:v.detach().cpu().numpy().astype(np.float64) for k,v in self.state_dict().items()})


def log_density(parameters,actions,active,kind):
    logits,mu,logs=parameters
    actions=actions.clamp(-1+1e-4,1-1e-4)
    if kind=='factorized':
        a=actions.reshape(actions.shape[0],actions.shape[1],2,2);z=torch.atanh(a)
        comp=(-.5*((z[:,:,:,None,:]-mu[:,None,:,:,:])*torch.exp(-logs[:,None,:,:,:]))**2-logs[:,None,:,:,:]-.5*np.log(2*np.pi)).sum(-1)
        comp+=torch.log_softmax(logits,-1)[:,None,:,:]
        per_agent=torch.logsumexp(comp,-1)-torch.log1p(-a*a).sum(-1)
        return (per_agent*active[:,None,:]).sum(-1)
    mask=active.repeat_interleave(2,-1);z=torch.atanh(actions)
    comp=((- .5*((z[:,:,None,:]-mu[:,None,:,:])*torch.exp(-logs[:,None,:,:]))**2-logs[:,None,:,:]-.5*np.log(2*np.pi))*mask[:,None,None,:]).sum(-1)
    comp+=torch.log_softmax(logits,-1)[:,None,:]
    return torch.logsumexp(comp,-1)-(torch.log1p(-actions*actions)*mask[:,None,:]).sum(-1)


def losses(model,batch):
    parameters,value=model(batch)
    nll=-(log_density(parameters,batch['label_actions'],batch['agents'][:,:,4],model.kind)*batch['label_weights']).sum(-1).mean()
    vl=torch.nn.functional.smooth_l1_loss(value,batch['label_value'])
    return nll+.5*vl,nll,vl


def augment(batch,swap_p,swap_e):
    x={k:v.clone() for k,v in batch.items()}
    if swap_p:
        x['agents']=x['agents'][:,[1,0]]
        x['history'][:,:,1:5]=x['history'][:,:,[3,4,1,2]]
        x['history'][:,:,11:13]=x['history'][:,:,[12,11]]
        a=x['label_actions'].reshape(len(x['agents']),-1,2,2)[:,:,[1,0],:]
        x['label_actions']=a.reshape(len(x['agents']),-1,4)
    if swap_e:
        x['particles']=x['particles'][:,[1,0]];x['weights']=x['weights'][:,[1,0]]
        x['public'][:,4:8]=x['public'][:,[5,4,7,6]]
        x['history'][:,:,5:9]=x['history'][:,:,[7,8,5,6]]
        x['history'][:,:,9:11]=x['history'][:,:,[10,9]]
    return x


class Runtime:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as d:
            self.kind,self.width=str(d['kind']),int(d['width'])
            self.arrays={k:d[k].copy() for k in d.files if k not in ('kind','width')}
    def block(self,x,name):
        for i in (0,2):x=np.maximum(0.,x@self.arrays[f'{name}.{i}.weight'].T+self.arrays[f'{name}.{i}.bias'])
        return x
    def __call__(self,x):
        m=(self.block(x['maps'],'map_encoder')*x['map_mask'][:,None]).sum(0)
        a=self.block(x['agents'],'agent_encoder').reshape(-1)
        p=(self.block(x['particles'],'particle_encoder')*x['weights'][...,None]).sum(1).reshape(-1)
        h=(self.block(x['history'],'history_encoder')*x['history_mask'][:,None]).sum(0)/max(1,x['history_mask'].sum())
        z=self.block(np.r_[m,a,p,h,x['public']],'trunk')
        y=z@self.arrays['policy.weight'].T+self.arrays['policy.bias']
        value=float(z@self.arrays['value.weight'][0]+self.arrays['value.bias'][0])
        if self.kind=='factorized':y=y.reshape(2,K,5);logits,mu,logs=y[...,0],y[...,1:3],y[...,3:5]
        else:y=y.reshape(-1,9);logits,mu,logs=y[...,0],y[...,1:5],y[...,5:9]
        probs=np.exp(logits-np.max(logits,axis=-1,keepdims=True));probs/=probs.sum(-1,keepdims=True)
        return (probs,np.clip(mu,-6,6),np.clip(logs,-3,1)),value
    def sample(self,parameters,rng):
        p,mu,logs=parameters
        if self.kind=='factorized':
            return np.array([np.tanh(mu[i,(k:=rng.choice(K,p=p[i]))]+np.exp(logs[i,k])*rng.standard_normal(2)) for i in range(2)])
        k=rng.choice(len(p),p=p)
        return np.tanh(mu[k]+np.exp(logs[k])*rng.standard_normal(4)).reshape(2,2)
