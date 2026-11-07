"""Matched current-observation / history DeepSets Gaussian proposals.

Only public observations enter features. The particle estimator is deliberately
absent from this interface. Normalized actions share the same physical scaling.
"""
import numpy as np
import torch
from torch import nn
from numba import njit


def observation_array(records):
    out=np.zeros((len(records),6),np.float64)
    for i,r in enumerate(records):
        out[i,:3]=[r['step'],*r['position']]
        if r['target'] is not None:out[i,3:]=[1.,*r['target']]
    return out


@njit(cache=True)
def features(boxes,records,n,speed,memory):
    current=records[n-1];at=current[1:3];step=current[0]
    obstacles=np.zeros((len(boxes),4))
    for i in range(len(boxes)):
        obstacles[i,:2]=((boxes[i,:,0]+boxes[i,:,1])*.5-at)/20.
        obstacles[i,2:]=(boxes[i,:,1]-boxes[i,:,0])/20.
    history=np.zeros((17,7));mask=np.zeros(17)
    start=max(0,n-16) if memory else n-1
    last=-1
    for i in range(start):
        if records[i,3]>0:last=i
    k=0
    for i in range(start,n):
        r=records[i]
        history[k,0]=(step-r[0])/80.
        history[k,1:3]=(r[1:3]-at)/20.
        history[k,3]=r[3]
        if r[3]>0:history[k,4:6]=(r[4:6]-at)/20.
        history[k,6]=1.;mask[k]=1.;k+=1
    if memory and last>=0:
        r=records[last]
        history[k,0]=(step-r[0])/80.;history[k,1:3]=(r[1:3]-at)/20.
        history[k,3]=1.;history[k,4:6]=(r[4:6]-at)/20.;history[k,6]=1.;mask[k]=1.
    public=np.zeros(7)
    public[:2]=at/10.;public[2]=speed/2.;public[3]=(80-step)/80.;public[4]=current[3]
    if current[3]>0:public[5:7]=(current[4:6]-at)/20.
    return obstacles,history,mask,public


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.map_encoder=nn.Sequential(nn.Linear(4,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.history_encoder=nn.Sequential(nn.Linear(7,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(71,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,4))

    def forward(self,obstacles,map_mask,history,history_mask,public):
        a=(self.map_encoder(obstacles)*map_mask.unsqueeze(-1)).sum(1)
        b=(self.history_encoder(history)*history_mask.unsqueeze(-1)).sum(1)
        raw=self.head(torch.cat((a,b,public),dim=-1))
        return raw[:,:2].tanh(),raw[:,2:].clamp(-8,2)

    def export(self):
        arrays=[]
        for block in (self.map_encoder,self.history_encoder,self.head):
            for layer in block:
                if isinstance(layer,nn.Linear):
                    arrays.extend([np.ascontiguousarray(layer.weight.detach().cpu().numpy(),dtype=np.float64),
                        np.ascontiguousarray(layer.bias.detach().cpu().numpy(),dtype=np.float64).reshape(1,-1)])
        return tuple(arrays)


@njit(cache=True)
def affine(x,w,b):return w@x+b[0]


@njit(cache=True)
def inference(boxes,records,n,speed,memory,weights):
    obstacles,history,mask,public=features(boxes,records,n,speed,memory)
    a=np.zeros(32);b=np.zeros(32)
    for x in obstacles:
        y=np.maximum(0.,affine(x,weights[0],weights[1]))
        a+=np.maximum(0.,affine(y,weights[2],weights[3]))
    for i in range(17):
        if mask[i]>0:
            y=np.maximum(0.,affine(history[i],weights[4],weights[5]))
            b+=np.maximum(0.,affine(y,weights[6],weights[7]))
    x=np.concatenate((a,b,public))
    x=np.maximum(0.,affine(x,weights[8],weights[9]))
    x=np.maximum(0.,affine(x,weights[10],weights[11]))
    x=affine(x,weights[12],weights[13])
    return np.tanh(x[:2]),np.minimum(2.,np.maximum(-8.,x[2:]))


def weighted_nll(mu,logvar,actions,visits):
    # Gaussian loss on normalized pre-clipping model, evaluated at executed
    # bounded teacher actions, matching the registered Gaussian supervision.
    weights=visits/visits.sum(-1,keepdim=True).clamp_min(1)
    nll=.5*((actions-mu[:,None,:]).square()*(-logvar[:,None,:]).exp()+logvar[:,None,:]+np.log(2*np.pi)).sum(-1)
    return (nll*weights).sum(-1).mean()
