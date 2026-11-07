"""Independent E45-O smoke inspection, including real resumed author updates."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import torch
import run_e45o_hasac_demonstration as o
from learning.e45m_task_adapter import Replay
from learning.e45o_demonstration_replay import DemonstrationReplay
from learning.e45m_author_hasac import AuthorHASAC
from learning.e45g_ppo import safe_action


def audit(root):
    torch.set_num_threads(1)
    cfg=o.verify(root)
    assert cfg['smoke'] and o.j.read(root/'decision.json')['status']=='complete'
    manifest=o.j.read(root/'demonstrations/manifest.json')
    assert len(manifest['episodes'])==2 and manifest['first_capture_continuations']>0
    with np.load(root/'demonstrations/data.npz') as z:
        demos={key:z[key].copy() for key in z.files}
    for row in manifest['episodes']:
        assert o.j.sha(row['source'])==row['source_sha256']
        assert o.j.sha(row['archive'])==row['archive_sha256']
        with np.load(row['archive']) as z:
            np.testing.assert_array_equal(z['next_obs'][:-1],z['obs'][1:])
            assert z['done'].sum()==1 and z['done'][-1,0]==1
    reports=[]
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            folder=root/'runs'/f'{arm}_{seed}'
            initial=torch.load(folder/'checkpoints/steps_0.pt',map_location='cpu',weights_only=False)
            pointer=o.j.read(folder/'latest.json')
            state=torch.load(pointer['path'],map_location='cpu',weights_only=False)
            assert state['gradient_updates']==2
            assert all(any(not torch.equal(v,state['actors'][i][key]) for key,v in a.items())
                       for i,a in enumerate(initial['actors']))
            assert any(not torch.equal(v,state['critic'][key]) for key,v in initial['critic'].items())
            rows=[e for p in sorted((folder/'updates').glob('*.json')) for e in o.j.read(p)['episodes']]
            assert len(rows)==2
            for row in rows:
                assert o.j.sha(row['episode'])==row['episode_sha256']
                assert o.j.sha(row['replay_episode'])==row['replay_sha256']
                payload=json.loads(gzip.decompress(Path(row['episode']).read_bytes()))
                case=o.j.read(row['case'])
                with np.load(row['replay_episode']) as z:
                    for i,t in enumerate(payload['trace']):
                        assert not t['stats']['joint_action_search']
                        assert not t['stats']['assignment_used'] and not t['stats']['roles_used']
                        physical,_=safe_action(np.asarray(t['observation']['positions']),z['action'][i],
                            t['observation']['active_p'],case['speed'],np.asarray(case['map']))
                        np.testing.assert_allclose(physical,t['action'],rtol=0,atol=1e-10)
            def new_replay():
                return (DemonstrationReplay(demos,cfg['replay_capacity'],seed,cfg['demonstration_fraction'])
                        if arm=='guided' else Replay(cfg['replay_capacity'],seed))
            # Load identical complete checkpoints twice and run the next actual
            # author update. This verifies RNG, optimizer, replay and target
            # networks jointly, beyond just comparing sampled indices.
            resumed=[]
            for _ in range(2):
                model=AuthorHASAC(obs_dim=845,device='cuda:1',seed=seed,retirement_mask=True)
                replay=new_replay()
                o.restore(folder,model,replay,o.j.sha(root/'experiment.json'))
                model.update(replay.sample(cfg['update_batch']))
                resumed.append({name:[p.detach().cpu().clone() for p in module.parameters()]
                    for name,module in [('actor0',model.actor[0].actor),('actor1',model.actor[1].actor),
                        ('critic',model.critic.critic),('critic2',model.critic.critic2),
                        ('target',model.critic.target_critic),('target2',model.critic.target_critic2)]})
                del model
            for name in resumed[0]:
                for a,b in zip(resumed[0][name],resumed[1][name]):
                    torch.testing.assert_close(a,b,rtol=0,atol=0)
            reports.append(dict(arm=arm,seed=seed,steps=state['steps'],updates=2,
                pure_actor_collection=True,actors_and_critic_changed=True,resumed_update_exact=True))
    # Paired runs start with exactly the same networks and initial evaluations.
    for seed in cfg['seeds']:
        a=torch.load(root/f'runs/baseline_{seed}/checkpoints/steps_0.pt',map_location='cpu',weights_only=False)
        b=torch.load(root/f'runs/guided_{seed}/checkpoints/steps_0.pt',map_location='cpu',weights_only=False)
        for x,y in zip(a['actors'],b['actors']):
            for key in x:torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
        assert a['history']==b['history']
    result=dict(passed=True,runs=reports,teacher_transitions=manifest['transitions'],
        first_capture_continuations=manifest['first_capture_continuations'],paired_initialization_exact=True,
        teacher_remap_max_error=manifest['maximum_physical_remap_error'],capability_claim=False)
    o.j.save(root/'smoke_audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,
        default=o.ROOT/'current/e45o_hasac_demonstration_smoke_20261101')
    audit(parser.parse_args().output)
