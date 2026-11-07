"""Latent-space BC on public teacher features; preserve PPO action semantics."""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from learning.e45g_ppo import Actor
from learning.e45b_joint_network import KEYS, augment
from run_frozen_d_closed_loop import sha, save


def executed_targets(arrays, payload, speed):
    trace = payload['trace']
    n = len(arrays['public'])
    ids = np.unique(np.linspace(0, len(trace)-1, min(32, len(trace))).astype(int))
    if len(ids) != n:
        raise RuntimeError('Teacher state subsampling does not match original 32-state protocol')
    remaining = np.array([trace[i]['observation']['remaining'] for i in ids])
    np.testing.assert_allclose(arrays['public'][:, 2]*80, remaining, atol=1e-5, rtol=0)
    targets = np.array([trace[i]['action'] for i in ids], np.float32)/speed
    if not np.isfinite(targets).all() or np.max(abs(targets)) > 1.000001:
        raise RuntimeError('Invalid normalized executed teacher action')
    return targets.reshape(n, 1, 4)


def load_teacher(rows, device):
    pieces = []
    for row in rows:
        for key, digest in [('data', 'data_sha256'), ('episode', 'episode_sha256')]:
            if sha(row[key]) != row[digest]:
                raise RuntimeError('Frozen teacher artifact changed: '+row[key])
        with np.load(row['data'], allow_pickle=False) as packet:
            arrays = {key: packet[key].astype(np.float32) for key in KEYS}
        payload = json.loads(Path(row['episode']).read_text())
        arrays['label_actions'] = executed_targets(arrays, payload, row['speed'])
        arrays['label_weights'] = np.ones((len(arrays['public']), 1), np.float32)
        pieces.append(arrays)
    if not pieces:
        raise RuntimeError('Empty teacher split')
    return {key: torch.as_tensor(np.concatenate([p[key] for p in pieces]), device=device)
            for key in pieces[0]}


def bc_loss(actor, batch):
    mean, _ = actor({key: batch[key] for key in KEYS})
    target = torch.atanh(batch['label_actions'][:, 0].reshape(-1, 2, 2).clamp(-1+1e-4,1-1e-4))
    mask = batch['agents'][:, :, 4]
    return ((mean-target).square().sum(-1)*mask).sum()/(2*mask.sum().clamp_min(1))


def fit_bc(folder, inventory, seed, device, epochs, emit):
    folder = Path(folder)
    decision = folder/'bc_decision.json'
    selected = folder/'bc_selected.pt'
    if decision.exists():
        record = json.loads(decision.read_text())
        if sha(selected) != record['checkpoint_sha256']:
            raise RuntimeError('Frozen BC checkpoint changed')
        return record
    torch.manual_seed(seed)
    actor = Actor().to(device)
    with torch.no_grad(): actor.log_std.fill_(math.log(.15))
    optimizer = torch.optim.Adam([p for name,p in actor.named_parameters() if name!='log_std'], lr=3e-4, weight_decay=1e-5)
    train = load_teacher(inventory['train'], device)
    validation = load_teacher(inventory['validation'], device)
    rng = np.random.default_rng(seed+94592031)
    logs=[];best=float('inf');start=1;chosen=None
    resume=folder/'bc_resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location=device,weights_only=False)
        actor.load_state_dict(saved['actor']);optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['rng'];logs=saved['logs'];best=saved['best'];chosen=saved['chosen'];start=saved['epoch']+1
    began=time.time()
    for epoch in range(start,epochs+1):
        losses=[];actor.train()
        for indices in np.array_split(rng.permutation(len(train['public'])),max(1,int(np.ceil(len(train['public'])/512)))):
            ids=torch.as_tensor(indices,device=device);batch={key:value[ids] for key,value in train.items()}
            batch=augment(batch,bool(rng.integers(2)),bool(rng.integers(2)))
            loss=bc_loss(actor,batch)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite BC loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),10.);optimizer.step()
            losses.append(float(loss.detach()))
        actor.eval()
        with torch.no_grad():score=float(bc_loss(actor,validation))
        row=dict(epoch=epoch,train_mse=float(np.mean(losses)),validation_mse=score)
        logs.append(row)
        if score<best:
            best=score;chosen=epoch
            torch.save(dict(actor={key:value.detach().cpu() for key,value in actor.state_dict().items()},
                kind='BC_ONLY',seed=seed,teacher='executed public joint-search actions',epochs=epoch,
                exploration_std=.15),selected)
        saved=dict(actor=actor.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,
                   logs=logs,best=best,chosen=chosen,epoch=epoch)
        temp=resume.with_suffix('.tmp.pt');torch.save(saved,temp);temp.replace(resume)
        emit(folder,'bc_pretraining',**row,epochs=epochs,train_states=len(train['public']),validation_states=len(validation['public']))
    record=dict(status='complete',seed=seed,epochs=epochs,selected_epoch=chosen,validation_mse=best,
        train_states=len(train['public']),validation_states=len(validation['public']),
        checkpoint=str(selected),checkpoint_sha256=sha(selected),logs=logs,seconds=time.time()-began,
        no_test_data=True,no_online_teacher=True)
    save(decision,record)
    return record
