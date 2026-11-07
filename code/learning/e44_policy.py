"""Three parameter-matched tanh-Gaussian policies on the SAME 2-D action space.

NLL is the pre-legality proposal density, not the rejection-normalized executed
density. Endpoint labels use a common 1e-4 resolution for all three networks.
All networks see identical route, map, particle and public history features.
"""
import numpy as np
import torch
from torch import nn
from solvers.e44_routes import MAX_ROUTES, ROUTE_DIM

KINDS = ('single', 'gmm', 'route')
LABEL_EPSILON = 1e-4


class Network(nn.Module):
    def __init__(self, kind, width=None):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(kind)
        self.kind = kind
        if width is None:
            # Select architecture size deterministically, before seeing data.
            choices = [(abs(self.parameter_formula(kind, h)-25000), h) for h in range(32, 129)]
            width = min(choices)[1]
        self.width = int(width)
        def encoder(n):
            return nn.Sequential(nn.Linear(n,32), nn.ReLU(), nn.Linear(32,32), nn.ReLU())
        self.map_encoder = encoder(4)
        self.particle_encoder = encoder(3)
        self.history_encoder = encoder(7)
        self.route_encoder = encoder(ROUTE_DIM)
        self.trunk = nn.Sequential(nn.Linear(137,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU())
        self.head = (nn.Sequential(nn.Linear(width+32,16),nn.ReLU(),nn.Linear(16,5))
                     if kind == 'route' else nn.Linear(width,5 if kind == 'single' else MAX_ROUTES*5))

    @staticmethod
    def parameter_formula(kind, h):
        enc = sum(n*32+32+32*32+32 for n in (4,3,7,ROUTE_DIM))
        trunk = 137*h+h+h*h+h
        head = ((h+32)*16+16+16*5+5 if kind == 'route' else (h+1)*(5 if kind == 'single' else MAX_ROUTES*5))
        return enc+trunk+head

    def forward(self, x):
        m = (self.map_encoder(x['maps'])*x['map_mask'][...,None]).sum(1)
        p = (self.particle_encoder(x['particles'])*x['weights'][...,None]).sum(1)
        h = (self.history_encoder(x['history'])*x['history_mask'][...,None]).sum(1)
        re = self.route_encoder(x['routes'])
        rm = x['route_mask']
        r = (re*rm[...,None]).sum(1)/rm.sum(1,keepdim=True).clamp_min(1.)
        z = self.trunk(torch.cat((m,p,h,r,x['public']),-1))
        if self.kind == 'route':
            y = self.head(torch.cat((z[:,None,:].expand(-1,MAX_ROUTES,-1),re),-1))
            logits = y[...,0].masked_fill(rm == 0,-1e9)
            location = y[...,1:3]+torch.atanh(x['anchors'].clamp(-.95,.95))
        else:
            y = self.head(z).reshape(-1,1 if self.kind == 'single' else MAX_ROUTES,5)
            logits, location = y[...,0], y[...,1:3]
        return logits, location.clamp(-6.,6.), y[...,3:5].clamp(-3.,1.)

    def export(self, path):
        d = dict(kind=np.array(self.kind),width=np.array(self.width))
        d.update({k:v.detach().cpu().numpy().astype(np.float64) for k,v in self.state_dict().items()})
        np.savez_compressed(path,**d)


def action_log_density(parameters, actions):
    """B x A log densities under one shared endpoint-resolution convention."""
    logits, location, logstd = parameters
    a = actions.clamp(-1+LABEL_EPSILON,1-LABEL_EPSILON)
    z = torch.atanh(a)
    component = -.5*(((z[:,:,None,:]-location[:,None,:,:])*torch.exp(-logstd[:,None,:,:]))**2).sum(-1)
    component -= logstd.sum(-1)[:,None,:]+np.log(2*np.pi)
    component += torch.log_softmax(logits,-1)[:,None,:]
    return torch.logsumexp(component,-1)-torch.log1p(-a*a).sum(-1)


def weighted_nll(model, batch):
    logp = action_log_density(model(batch), batch['label_actions'])
    return -(logp*batch['label_weights']).sum(1).mean()


class Runtime:
    def __init__(self, path):
        with np.load(path,allow_pickle=False) as d:
            self.kind, self.width = str(d['kind']), int(d['width'])
            self.arrays = {k:np.ascontiguousarray(d[k]) for k in d.files if k not in ('kind','width')}

    def block(self,x,name,layers):
        for i in range(layers):
            j = 2*i
            x = np.maximum(0.,x@self.arrays[f'{name}.{j}.weight'].T+self.arrays[f'{name}.{j}.bias'])
        return x

    def __call__(self,x):
        m = (self.block(x['maps'],'map_encoder',2)*x['map_mask'][:,None]).sum(0)
        p = (self.block(x['particles'],'particle_encoder',2)*x['weights'][:,None]).sum(0)
        h = (self.block(x['history'],'history_encoder',2)*x['history_mask'][:,None]).sum(0)
        re = self.block(x['routes'],'route_encoder',2)
        r = (re*x['route_mask'][:,None]).sum(0)/max(1.,x['route_mask'].sum())
        z = self.block(np.concatenate((m,p,h,r,x['public'])),'trunk',2)
        if self.kind == 'route':
            q = np.maximum(0.,np.concatenate((np.tile(z,(MAX_ROUTES,1)),re),1)@self.arrays['head.0.weight'].T+self.arrays['head.0.bias'])
            y = q@self.arrays['head.2.weight'].T+self.arrays['head.2.bias']
            logits = np.where(x['route_mask']>0,y[:,0],-1e9)
            location = y[:,1:3]+np.arctanh(np.clip(x['anchors'],-.95,.95))
        else:
            y = (z@self.arrays['head.weight'].T+self.arrays['head.bias']).reshape(-1,5)
            logits,location = y[:,0],y[:,1:3]
        probabilities = np.exp(logits-logits.max()); probabilities /= probabilities.sum()
        return probabilities,np.clip(location,-6.,6.),np.clip(y[:,3:5],-3.,1.)

    def sample(self,x,rng):
        probabilities,location,logstd = self(x)
        k = int(rng.choice(len(probabilities),p=probabilities))
        return np.tanh(location[k]+np.exp(logstd[k])*rng.standard_normal(2))
