"""Train the E55 policy/value MLPs from high-budget visits and matched returns."""
import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
import repo_paths
repo_paths.install()
import argparse
from copy import deepcopy
import json
import numpy as np
import torch
from run_e55_parallel import OUT, FROZEN, save
from solvers.e52_nar_nae import NARNAE


def tensors(groups, device):
    # Each policy label appears once. Independent branch returns supervise NAE.
    rows=[r for g in groups for r in g['records']]
    values=rows+[p['alternative'] for g in groups for p in g['pairs']]
    pairs=[p for g in groups for p in g['pairs']]
    width=max(len(r['features']) for r in rows)
    features=np.zeros((len(rows),width,14),np.float32)
    visits=np.zeros((len(rows),width),np.float32)
    valid=np.zeros((len(rows),width),bool)
    for i,r in enumerate(rows):
        n=len(r['features']);features[i,:n]=r['features'];visits[i,:n]=r['policy'];valid[i,:n]=True
    def chosen(r):return r['features'][r['value_index']]
    def target(r):return r['returns'][r['value_index']]
    def tensor(x):return torch.as_tensor(np.asarray(x),device=device,dtype=torch.float32)
    return (tensor(features),tensor(visits),torch.as_tensor(valid,device=device),
            tensor([chosen(r) for r in values]),tensor([target(r) for r in values]),
            tensor([[chosen(p['reference']),chosen(p['alternative'])] for p in pairs]).reshape(-1,2,14),
            tensor([target(p['alternative'])-target(p['reference']) for p in pairs]))


def losses(net,b,ip,iv,ig):
    x,y,mask,v,t,p,delta=b
    logits=net.nar(x[ip]).squeeze(-1).masked_fill(~mask[ip],-1e9)
    policy=-(y[ip]*logits.log_softmax(1)).sum(1).mean()
    value=torch.nn.functional.smooth_l1_loss(net.nae(v[iv]).flatten(),t[iv])
    pred=net.nae(p[ig]).squeeze(-1)
    pair=torch.nn.functional.smooth_l1_loss(pred[:,1]-pred[:,0],delta[ig]) if pred.numel() else value*0.
    return torch.stack([policy,value,pair])


def main(seed):
    m=json.loads((OUT/'manifest.json').read_text())
    from pathlib import Path
    device=torch.device('cuda:0')
    batches={}
    for split,paths in m['train'].items():
        groups=[json.loads((OUT/'teacher'/split/Path(p).stem/'labels.json').read_text()) for p in paths]
        batches[split]=tensors(groups,device)
    def indices(b):return [torch.arange(len(b[k]),device=device) for k in (0,3,5)]
    torch.manual_seed(seed)
    net=NARNAE()
    def gpu(model):
        model.nar.to(device);model.nae.to(device)
    gpu(net)
    optimizer=torch.optim.AdamW(list(net.nar.parameters())+list(net.nae.parameters()),lr=.001)
    train=batches['train']
    if len(train[0])<=1534 or not len(train[5]):raise ValueError('insufficient new states or matched labels')
    best=float('inf')
    for epoch in range(200):
        for ip in torch.randperm(len(train[0]),device=device).split(64):
            iv=torch.randint(len(train[3]),ip.shape,device=device)
            ig=torch.randint(len(train[5]),ip.shape,device=device)
            loss=losses(net,train,ip,iv,ig).sum()
            optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():
            b=batches['validation'];score=float(losses(net,b,*indices(b)).sum())
        if score<best:
            best=score;best_epoch=epoch+1
            weights=[deepcopy(model.state_dict()) for model in (net.nar,net.nae)]
        if epoch+1-best_epoch>=25:break
    net.nar.load_state_dict(weights[0]);net.nae.load_state_dict(weights[1])
    baseline=NARNAE.load(FROZEN);gpu(baseline)
    metadata=dict(seed=seed,epoch=best_epoch,val_loss=best,
                  samples={k:[len(v[i]) for i in (0,3,5)] for k,v in batches.items()},
                  loss_names=['visit_cross_entropy','return_huber','paired_delta_huber'])
    with torch.no_grad():
        for name,model in [('new',net),('frozen',baseline)]:
            metadata[name]={k:losses(model,b,*indices(b)).tolist() for k,b in batches.items()}
    folder=OUT/'trained';folder.mkdir(exist_ok=True)
    net.nar.cpu();net.nae.cpu()
    net.save(folder/f'mlp_{seed}.pt',metadata)
    save(folder/f'mlp_{seed}.json',metadata)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('seed',type=int)
    main(parser.parse_args().seed)
