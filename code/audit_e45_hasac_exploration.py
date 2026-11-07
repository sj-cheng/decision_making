"""Matched-public-state actor distribution audit, no training or new tests."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import gzip
import json
from pathlib import Path
import numpy as np
import torch
import run_e45m_hasac_task as m
from learning.e45_hasac_deployment import load_actors


def summary(values):
    if not len(values):return None
    return dict(samples=len(values),median=float(np.median(values)),p10=float(np.quantile(values,.1)),
                p90=float(np.quantile(values,.9)),mean=float(np.mean(values)))


def actor_stats(actors,obs,visible):
    active=obs[:,:,-1]>0;means=[];stds=[]
    with torch.inference_mode():
        for i,a in enumerate(actors):
            net=a.actor.net(torch.as_tensor(obs[:,i],dtype=torch.float32))
            means.append(a.actor.mu_layer(net).numpy());stds.append(a.actor.log_std_layer(net).clamp(-20,2).exp().numpy())
    mu=np.stack(means,1);std=np.stack(stds,1)
    rng=np.random.default_rng(924017)
    z=rng.normal(size=(32,*mu.shape));raw=mu[None]+std[None]*z;action=np.tanh(raw)
    logp=(-.5*np.log(2*np.pi)-np.log(std[None])-.5*z*z).sum(-1)
    logp-=(2*(np.log(2)-raw-np.logaddexp(0,-2*raw))).sum(-1)
    entropy=-logp.mean(0);det=np.linalg.norm(np.tanh(mu),axis=-1)
    mc=np.linalg.norm(action.mean(0),axis=-1);sat=np.mean(np.abs(np.tanh(mu))>.95,axis=-1)
    out={}
    for name,mask in [('all',active),('currently_visible',active&visible[:,None]),('currently_hidden',active&~visible[:,None])]:
        out[name]=dict(raw_std=summary(std[mask].reshape(-1)),raw_abs_mean=summary(np.abs(mu[mask]).reshape(-1)),
                       tanh_entropy=summary(entropy[mask]),deterministic_intent_norm=summary(det[mask]),
                       monte_carlo_mean_intent_norm=summary(mc[mask]),saturated_mean_coordinate_fraction=summary(sat[mask]))
    return out


def main():
    torch.set_num_threads(1);results={}
    for folder in sorted(m.OUT.glob('runs/*')):
        h=m.j.read(folder/'evaluation.json');available=[r for r in h if r['steps']>=20000 and
              (folder/'checkpoints'/f'steps_{r["steps"]}.pt').exists()]
        if not available:continue
        point=available[0];rows=[m.j.read(p) for p in sorted((folder/'updates').glob('*.json'))]
        rows=[r for r in rows if r['steps']<=point['steps']]
        # Stratify diagnostics for exposure and lifecycle coverage. This is
        # not a representative sample used to estimate task performance.
        candidates=[e for r in rows for e in r['episodes'] if e['captures']>0]
        candidates+=rows[0]['episodes'][:3]+rows[-1]['episodes'][:3]
        unique={e['episode']:e for e in candidates};selected=list(unique.values())[:30]
        xs=[];seen=[]
        for e in selected:
            assert m.j.sha(e['replay_episode'])==e['replay_sha256']
            with np.load(e['replay_episode']) as d:xs.append(d['obs'].copy())
            payload=json.loads(gzip.decompress(Path(e['episode']).read_bytes()))
            seen.extend(any(q is not None for reports in t['observation']['reports'] for q in reports) for t in payload['trace'])
        obs=np.concatenate(xs);visible=np.asarray(seen,bool);assert len(obs)==len(visible)
        out=dict(scope='matched training public-state probe, selected for exposure/lifecycle coverage; not performance estimation',
                 selected_episodes=len(selected),frames=len(obs),visible_frames=int(visible.sum()),checkpoints={})
        for steps in (0,point['steps']):
            p=folder/'checkpoints'/f'steps_{steps}.pt';actors,metadata=load_actors(p,m.j.sha(p))
            out['checkpoints'][str(steps)]=dict(metadata=metadata,statistics=actor_stats(actors,obs,visible))
        results[folder.name]=out
    assert len(results)==2
    target=m.ROOT/'current/e45_marl_macro_audit_20261101/HASAC_exploration_20k.json'
    m.j.save(target,dict(results=results,source_sha256=m.j.sha(Path(__file__)),training_unchanged=True))
    for name,r in results.items():
        print(name,'probe frames',r['frames'],'visible',r['visible_frames'],flush=True)
        for step,d in r['checkpoints'].items():
            s=d['statistics']['all']
            print(step,{k:v for k,v in s.items() if k in ('raw_std','tanh_entropy','deterministic_intent_norm')},flush=True)


if __name__=='__main__':main()
