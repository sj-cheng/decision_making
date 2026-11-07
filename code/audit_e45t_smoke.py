"""One-factor fork, historical continuation and resumed-update audit."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import numpy as np
import torch
import run_e45t_critic_lr_stability as t


def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:equal(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b,(a,b)


def snapshot(model):
    critic=model.critic
    return copy.deepcopy(dict(actors=[a.actor.state_dict() for a in model.actor],
        actor_optimizers=[a.actor_optimizer.state_dict() for a in model.actor],
        **{k:getattr(critic,k).state_dict() for k in ('critic','critic2','target_critic','target_critic2','critic_optimizer')},
        total_it=model.runner.total_it))


def next_update(root,cfg,folder,seed,device):
    model=t.m.AuthorHASAC(obs_dim=845,device=device,seed=seed,retirement_mask=True)
    replay=t.m.Replay(cfg['replay_capacity'],seed)
    t.m.restore(folder,model,replay,t.j.sha(root/'experiment.json'))
    model.update(replay.sample(cfg['update_batch']))
    result=snapshot(model);del model,replay
    return result


def core_trajectory(path):
    with gzip.open(path,'rt') as handle:payload=json.load(handle)
    assert payload['summary']['public_replay_exact']
    rows=[]
    for frame in payload['trace']:
        stats=frame['stats']
        assert not stats['joint_action_search'] and not stats['assignment_used'] and not stats['roles_used']
        assert 'teacher_query' not in stats
        rows.append({key:frame[key] for key in ('observation','action','opponent_action','next_state','reward')})
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=t.SMOKE)
    parser.add_argument('--device',default='cuda:1');args=parser.parse_args()
    torch.set_num_threads(1);root=args.output;cfg=t.verify(root)
    assert cfg['smoke'] and t.j.read(root/'decision.json')['status']=='complete'
    keys=('actors','actor_optimizers','critic','critic2','target_critic','target_critic2',
        'gradient_updates','iteration','torch_rng','cuda_rng','numpy_rng','replay_rng','replay_resamples')
    reports=[]
    for seed in cfg['seeds']:
        source=cfg['sources'][str(seed)]
        parent=torch.load(source['path'],map_location='cpu',weights_only=False)
        historical=t.j.read(t.n.OUT/'runs'/f'HASAC_Curriculum_{seed}'/'updates'/f'{source["iteration"]+1:05d}.json')
        historical_traces=[core_trajectory(e['episode']) for e in historical['episodes']]
        paired=[]
        for arm in cfg['arms']:
            folder=root/'runs'/f'{arm}_{seed}'
            initial=torch.load(folder/'checkpoints/steps_0.pt',map_location='cpu',weights_only=False)
            for key in keys:equal(parent[key],initial[key])
            optimizer=copy.deepcopy(initial['critic_optimizer'])
            for group,old in zip(optimizer['param_groups'],parent['critic_optimizer']['param_groups']):
                assert group['lr']==cfg['critic_lr_by_arm'][arm]
                group['lr']=old['lr']
            equal(optimizer,parent['critic_optimizer'])
            assert t.j.sha(initial['replay_path'])==source['replay_sha256']
            h=t.j.read(folder/'evaluation.json')
            for mode in ('deterministic','stochastic'):equal(h[0][mode],parent['history'][-1][mode])
            decision=t.j.read(folder/'decision.json');assert decision['new_gradient_updates']==2
            records=[t.j.read(p) for p in sorted((folder/'updates').glob('*.json'))]
            assert len(records)==1 and len(records[0]['episodes'])==12
            traces=[core_trajectory(e['episode']) for e in records[0]['episodes']]
            equal(traces,historical_traces);paired.append(traces)
            terminal=torch.load(t.j.read(folder/'latest.json')['path'],map_location='cpu',weights_only=False)
            assert terminal['gradient_updates']==source['gradient_updates']+2
            assert any(not torch.equal(value,terminal['critic'][key]) for key,value in initial['critic'].items())
            for actor in range(2):
                assert any(not torch.equal(value,terminal['actors'][actor][key]) for key,value in initial['actors'][actor].items())
            for path in (folder/'evaluation/episodes').glob('*.json.gz'):core_trajectory(path)
            a=next_update(root,cfg,folder,seed,args.device);b=next_update(root,cfg,folder,seed,args.device);equal(a,b)
            reports.append(dict(arm=arm,seed=seed,source_steps=source['steps'],new_steps=decision['steps'],
                new_updates=2,only_critic_lr_changed=True,initial_evaluation_matches_N=True,
                first_batch_matches_original_N=True,resumed_next_update_exact=True))
            del a,b,initial,terminal,optimizer
        equal(*paired);del parent,paired,historical_traces
    t.verify(root)
    result=dict(passed=True,runner_sha256=t.j.sha(Path(t.__file__).resolve()),runs=reports,
        full_state_preserved_except_explicit_critic_lr=True,pure_actor_no_teacher=True,capability_claim=False)
    t.j.save(root/'smoke_audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
