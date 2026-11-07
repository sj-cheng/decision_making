"""Verify full-state O forks, genuine GPU updates, no-teacher traces and resume."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
import run_e45r_hasac_continuation as r
from learning.e45m_author_hasac import AuthorHASAC
from learning.e45p_actor_supervision import public_teacher,install_anchor


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
        **{k:getattr(critic,k).state_dict() for k in
            ('critic','critic2','target_critic','target_critic2','critic_optimizer')},
        gradient_updates=model.runner.total_it))


def next_update(folder,cfg,seed,device,weight):
    model=AuthorHASAC(obs_dim=845,device=device,seed=seed,retirement_mask=True)
    replay=r.new_replay(cfg,seed)
    r.o.restore(folder,model,replay,r.j.sha(folder.parent.parent/'experiment.json'))
    teacher=public_teacher(r.checked_data(cfg['demonstration_file']),device)
    handles,stats=install_anchor(model,teacher,weight)
    model.update(replay.sample(cfg['update_batch']))
    out=snapshot(model)
    assert stats['calls']==(2 if weight else 0)
    for handle in handles:handle.remove()
    del model,replay
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=r.SMOKE)
    parser.add_argument('--device',default='cuda:2');args=parser.parse_args()
    torch.set_num_threads(1)
    cfg=r.verify(args.output)
    assert cfg['smoke'] and r.j.read(args.output/'decision.json')['status']=='complete'
    data=r.checked_data(cfg['demonstration_file'])
    # The runtime data gate must reject a contradictory label, not just log it.
    changed={k:v.copy() for k,v in data.items()}
    for key in ('obs','active'):changed[key][1]=changed[key][0]
    changed['action'][1]=-changed['action'][0]
    with tempfile.TemporaryDirectory(dir=args.output) as temp:
        file=Path(temp)/'conflicting.npz';np.savez_compressed(file,**changed)
        try:r.checked_data(file)
        except RuntimeError:pass
        else:raise AssertionError('Conflicting teacher labels were accepted')
    reports=[]
    state_keys=('actors','actor_optimizers','critic','critic2','target_critic','target_critic2',
        'critic_optimizer','gradient_updates','iteration','torch_rng','cuda_rng','numpy_rng','replay_rng','replay_resamples')
    for seed in cfg['seeds']:
        source=cfg['sources'][str(seed)]
        parent=torch.load(source['pointer']['path'],map_location='cpu',weights_only=False)
        first_training=[];initials=[]
        for arm in cfg['arms']:
            folder=args.output/'runs'/f'{arm}_{seed}'
            initial=torch.load(folder/'checkpoints/steps_0.pt',map_location='cpu',weights_only=False)
            initials.append(initial)
            for key in state_keys:equal(parent[key],initial[key])
            with np.load(parent['replay_path']) as a,np.load(initial['replay_path']) as b:
                assert set(a.files)==set(b.files)
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
            decision=r.j.read(folder/'decision.json')
            assert decision['new_gradient_updates']==2
            assert decision['gradient_updates']==source['gradient_updates']+2
            assert decision['anchor']['calls']==(4 if arm=='replay_bc' else 0)
            pointer=r.j.read(folder/'latest.json')
            terminal=torch.load(pointer['path'],map_location='cpu',weights_only=False)
            for actor in range(2):
                assert any(not torch.equal(v,terminal['actors'][actor][key]) for key,v in initial['actors'][actor].items())
            assert any(not torch.equal(v,terminal['critic'][key]) for key,v in initial['critic'].items())
            rows=[e for p in sorted((folder/'updates').glob('*.json')) for e in r.j.read(p)['episodes']]
            assert len(rows)==2
            pair=[]
            for row in rows:
                assert r.j.sha(row['episode'])==row['episode_sha256']
                payload=json.loads(gzip.decompress(Path(row['episode']).read_bytes()))
                assert payload['summary']['public_replay_exact']
                pair.append([{k:t[k] for k in ('observation','action','opponent_action','next_state','reward')} for t in payload['trace']])
            first_training.append(pair)
            # Check ALL smoke traces, including stochastic evaluations.
            for path in folder.glob('evaluation/episodes/*.json.gz'):
                payload=json.loads(gzip.decompress(path.read_bytes()))
                assert payload['summary']['reason'] in ('timeout','double_capture')
                assert payload['summary']['public_replay_exact']
                for frame in payload['trace']:
                    stats=frame['stats']
                    assert not stats['joint_action_search'] and not stats['assignment_used'] and not stats['roles_used']
                    assert 'teacher_query' not in stats
            weight=cfg['anchor_weight'] if arm=='replay_bc' else 0.
            a=next_update(folder,cfg,seed,args.device,weight)
            b=next_update(folder,cfg,seed,args.device,weight)
            equal(a,b)
            reports.append(dict(arm=arm,seed=seed,online_steps=decision['steps'],
                full_O_state_and_replay_exact=True,new_updates=2,anchor_calls=decision['anchor']['calls'],
                resumed_next_update_exact=True,pure_actor_evaluation=True))
            del a,b,terminal
        equal(first_training[0],first_training[1])
        # At the inherited state, installing lambda=0 equals no hook at all.
        no_hook=[]
        for install in (False,True):
            model=AuthorHASAC(obs_dim=845,device=args.device,seed=seed,retirement_mask=True)
            replay=r.new_replay(cfg,seed)
            r.o.restore(Path(source['folder']),model,replay,cfg['parent_O_experiment_sha256'])
            if install:install_anchor(model,public_teacher(data,args.device),0.)
            model.update(replay.sample(cfg['update_batch']));no_hook.append(snapshot(model))
            del model,replay
        equal(*no_hook)
        del parent,initials,no_hook
    r.verify(args.output)
    result=dict(passed=True,runner_sha256=r.j.sha(Path(r.__file__).resolve()),
        full_actor_critic_targets_optimizers_replay_rng_iteration_preserved=True,
        zero_weight_exact_author_update=True,conflicting_label_gate_rejects=True,
        first_training_batch_pair_exact=True,runs=reports,capability_claim=False)
    r.j.save(args.output/'smoke_audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
