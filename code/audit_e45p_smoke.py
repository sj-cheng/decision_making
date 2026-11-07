"""Inspect completed E45-P real-environment smoke and resumed GPU updates."""
import gzip
import json
from pathlib import Path
import numpy as np
import torch
import run_e45p_hasac_bc_anchor as p
from learning.e45m_author_hasac import AuthorHASAC
from learning.e45o_demonstration_replay import DemonstrationReplay
from learning.e45p_actor_supervision import public_teacher,install_anchor


def main():
    torch.set_num_threads(1)
    root=p.ROOT/'current/e45p_hasac_bc_anchor_smoke_20261101';cfg=p.verify(root)
    assert cfg['smoke'] and p.j.read(root/'decision.json')['status']=='complete'
    data=p.load_data(cfg);teacher=public_teacher(data,'cuda:1');reports=[]
    for seed in cfg['seeds']:
        folder=root/'bc'/f'HASAC_{seed}';pointer=p.j.read(folder/'latest.json')
        before=torch.load(folder/'checkpoints/steps_0.pt',map_location='cpu',weights_only=False)
        bc=torch.load(pointer['path'],map_location='cpu',weights_only=False)
        assert bc['steps']==2 and bc['gradient_updates']==0
        for key in before['critic']:torch.testing.assert_close(before['critic'][key],bc['critic'][key],rtol=0,atol=0)
        for i in range(2):
            assert any(not torch.equal(v,bc['actors'][i][key]) for key,v in before['actors'][i].items())
            for key in ('log_std_layer.weight','log_std_layer.bias'):
                torch.testing.assert_close(before['actors'][i][key],bc['actors'][i][key],rtol=0,atol=0)
        initials=[]
        for arm in cfg['arms']:
            folder=root/'runs'/f'{arm}_{seed}'
            initial=torch.load(folder/'checkpoints/steps_0.pt',map_location='cpu',weights_only=False)
            initials.append(initial)
            for i in range(2):
                for key in bc['actors'][i]:torch.testing.assert_close(bc['actors'][i][key],initial['actors'][i][key],rtol=0,atol=0)
                assert not initial['actor_optimizers'][i]['state']
            decision=p.j.read(folder/'decision.json');assert decision['gradient_updates']==2
            assert decision['anchor']['calls']==(4 if arm=='replay_bc' else 0)
            rows=[e for path in sorted((folder/'updates').glob('*.json')) for e in p.j.read(path)['episodes']]
            assert len(rows)==2
            for row in rows:
                assert p.j.sha(row['episode'])==row['episode_sha256']
                payload=json.loads(gzip.decompress(Path(row['episode']).read_bytes()))
                assert all(not t['stats']['joint_action_search'] and not t['stats']['assignment_used'] for t in payload['trace'])
            resumed=[]
            for _ in range(2):
                model=AuthorHASAC(obs_dim=845,device='cuda:1',seed=seed,retirement_mask=True)
                replay=DemonstrationReplay(data,cfg['replay_capacity'],seed,cfg['demonstration_fraction'])
                p.o.restore(folder,model,replay,p.j.sha(root/'experiment.json'))
                handles,stats=install_anchor(model,teacher,cfg['anchor_weight'] if arm=='replay_bc' else 0)
                model.update(replay.sample(cfg['update_batch']))
                resumed.append({name:[v.detach().cpu().clone() for v in module.parameters()] for name,module in
                    [('actor0',model.actor[0].actor),('actor1',model.actor[1].actor),('critic',model.critic.critic),
                     ('critic2',model.critic.critic2),('target',model.critic.target_critic),('target2',model.critic.target_critic2)]})
                for handle in handles:handle.remove()
                del model
            for name in resumed[0]:
                for a,b in zip(resumed[0][name],resumed[1][name]):torch.testing.assert_close(a,b,rtol=0,atol=0)
            reports.append(dict(arm=arm,seed=seed,online_steps=decision['steps'],real_updates=2,
                pure_actor_collection=True,resumed_next_update_exact=True,anchor_calls=decision['anchor']['calls']))
        for key in initials[0]['critic']:torch.testing.assert_close(initials[0]['critic'][key],initials[1]['critic'][key],rtol=0,atol=0)
        assert initials[0]['history']==initials[1]['history']
    result=dict(passed=True,runs=reports,bc_only_does_not_update_critic_or_std=True,
        paired_bc_initialization_exact=True,gate_bypassed_for_smoke_only=True,capability_claim=False)
    p.j.save(root/'smoke_audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
