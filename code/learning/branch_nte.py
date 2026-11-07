"""Parameter-matched free-mixture / public-branch Gaussian policies."""
import numpy as np
import torch
from torch import nn
from learning.memory_nte import features,observation_array


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.map_encoder=nn.Sequential(nn.Linear(4,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.history_encoder=nn.Sequential(nn.Linear(7,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.branch_encoder=nn.Sequential(nn.Linear(12,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.trunk=nn.Sequential(nn.Linear(103,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(80,64),nn.ReLU(),nn.Linear(64,5))

    def forward(self,obstacles,map_mask,history,history_mask,public,branches,mode='H'):
        active=branches[:,:,11]
        a=(self.map_encoder(obstacles)*map_mask[:,:,None]).sum(1)
        b=(self.history_encoder(history)*history_mask[:,:,None]).sum(1)
        d=(self.branch_encoder(branches)*active[:,:,None]).sum(1)
        context=self.trunk(torch.cat((a,b,d,public),-1))
        if mode=='M':
            local=(branches*active[:,:,None]).sum(1)/active.sum(1).clamp_min(1)[:,None]
            local=local[:,None,:].expand(-1,4,-1)
            mask=torch.ones_like(active)
        else:local=branches;mask=active
        identity=torch.eye(4,device=context.device)[None].expand(len(context),-1,-1)
        raw=self.head(torch.cat((context[:,None,:].expand(-1,4,-1),local,identity),-1))
        logits=raw[:,:,0].masked_fill(mask<.5,-1e9)
        return logits,raw[:,:,1:3].tanh(),raw[:,:,3:5].clamp(-8,2)


def encoded(public,history,branches):
    records=observation_array(history)
    obstacles,hist,mask,state=features(public.boxes,records,len(records),public.speed,True)
    return dict(obstacles=obstacles,map_mask=np.ones(len(obstacles)),history=hist,
                history_mask=mask,public=state,branches=np.asarray(branches))


def loss(logits,mu,logvar,actions,visits,branch,mode):
    logp=logits.log_softmax(-1)
    logdensity=-.5*(((actions[:,:,None,:]-mu[:,None,:,:]).square()*(-logvar[:,None,:,:]).exp())+logvar[:,None,:,:]+np.log(2*np.pi)).sum(-1)
    weights=visits/visits.sum(-1,keepdim=True).clamp_min(1)
    mixture=-(weights*torch.logsumexp(logdensity+logp[:,None,:],-1)).sum(-1)
    if mode=='M': objective=mixture.mean()
    else:
        chosen=logdensity.gather(2,branch[:,None,None].expand(-1,actions.shape[1],1)).squeeze(-1)
        objective=-(weights*chosen).sum(-1).mean()+nn.functional.cross_entropy(logits,branch)
    return objective,mixture.mean()


class Runtime:
    """Float64 NumPy inference; weights are identical to CPU training weights."""
    def __init__(self,model,public,mode):
        self.public=public;self.mode=mode
        self.weights={}
        for name in ('map_encoder','history_encoder','branch_encoder','trunk','head'):
            self.weights[name]=[(x.weight.detach().cpu().numpy().astype(float),x.bias.detach().cpu().numpy().astype(float)) for x in getattr(model,name) if isinstance(x,nn.Linear)]
        # Static map encoding belongs to explicitly measured map preprocessing.
        obstacles=np.zeros((len(public.boxes),4))
        # Relative map centers depend on the observer; no per-map shortcut here.

    def block(self,name,x):
        layers=self.weights[name]
        for i,(w,b) in enumerate(layers):
            x=x@w.T+b
            if i<len(layers)-1:x=np.maximum(0,x)
        return x

    def __call__(self,history,branches):
        v=encoded(self.public,history,branches);active=branches[:,11]
        a=np.maximum(0,self.block('map_encoder',v['obstacles'])).sum(0)
        b=(np.maximum(0,self.block('history_encoder',v['history']))*v['history_mask'][:,None]).sum(0)
        d=(np.maximum(0,self.block('branch_encoder',branches))*active[:,None]).sum(0)
        context=np.maximum(0,self.block('trunk',np.r_[a,b,d,v['public']]))
        if self.mode=='M':local=np.broadcast_to((branches*active[:,None]).sum(0)/max(1,active.sum()),(4,12));mask=np.ones(4)
        else:local=branches;mask=active
        raw=self.block('head',np.concatenate((np.broadcast_to(context,(4,64)),local,np.eye(4)),axis=1))
        logits=np.where(mask>.5,raw[:,0],-1e9);w=np.exp(logits-logits.max());w/=w.sum()
        return w,np.tanh(raw[:,1:3]),np.clip(raw[:,3:5],-8,2)
