"""Independent physical/replay/lifecycle audit; no teacher data used in training."""
import gzip
import json
from pathlib import Path
import numpy as np
import torch
import run_e45m_hasac_task as m
from learning.e45m_task_adapter import Controller,Replay,episode_arrays
from learning.e45g_ppo import safe_action


def main():
    torch.set_num_threads(1)
    root=m.ROOT/'current/e45m_hasac_task_smoke_20261101'
    assert m.j.read(root/'decision.json')['status']=='complete'
    cfg=m.verify(root);reports=[]
    for folder in sorted((root/'runs').iterdir()):
        pointer=m.j.read(folder/'latest.json');state=torch.load(pointer['path'],map_location='cpu',weights_only=False)
        assert m.j.sha(pointer['path'])==pointer['sha256']
        initial=torch.load(folder/'checkpoints/steps_0.pt',map_location='cpu',weights_only=False)
        assert state['gradient_updates']==2
        assert all(any(not torch.equal(v,state['actors'][i][k]) for k,v in a.items()) for i,a in enumerate(initial['actors']))
        assert any(not torch.equal(v,state['critic'][k]) for k,v in initial['critic'].items())
        updates=[m.j.read(p) for p in sorted((folder/'updates').glob('*.json'))]
        episodes=[e for row in updates for e in row['episodes']]
        assert len(episodes)==2 and state['steps']==sum(e['decisions'] for e in episodes)
        array_list=[]
        for row in episodes:
            assert m.j.sha(row['episode'])==row['episode_sha256']
            assert m.j.sha(row['replay_episode'])==row['replay_sha256']
            payload=json.loads(gzip.decompress(Path(row['episode']).read_bytes()))
            case=m.j.read(row['case']);trace=payload['trace']
            with np.load(row['replay_episode']) as z:
                arrays={k:z[k].copy() for k in z.files}
            assert len(trace)==len(arrays['obs'])
            for i,t in enumerate(trace):
                expected,_=safe_action(np.asarray(t['observation']['positions']),arrays['action'][i],
                                       t['observation']['active_p'],case['speed'],np.asarray(case['map']))
                np.testing.assert_allclose(expected,t['action'],rtol=0,atol=1e-10)
                np.testing.assert_array_equal(arrays['active'][i,:,0],t['observation']['active_p'])
                np.testing.assert_allclose(arrays['central'][i,-4:],np.asarray(t['state'][:4])/10,atol=1e-7)
            assert arrays['done'].sum()==1 and arrays['done'][-1,0]==1
            np.testing.assert_array_equal(arrays['next_obs'][:-1],arrays['obs'][1:])
            np.testing.assert_array_equal(arrays['next_obs'][-1],np.zeros((2,845)))
            array_list.append(arrays)
        assert m.j.sha(state['replay_path'])==state['replay_sha256']
        replay=Replay(cfg['replay_capacity']);replay.load(state['replay_path'],state['replay_rng'])
        for key in array_list[0]:np.testing.assert_array_equal(replay.arrays[key][:replay.size],np.concatenate([d[key] for d in array_list]))
        other=Replay(cfg['replay_capacity']);other.load(state['replay_path'],state['replay_rng'])
        for a,b in zip(replay.sample(32),other.sample(32)):
            if a is not None:np.testing.assert_array_equal(a,b)
        assert not list((folder/'failures').glob('*.json'))
        reports.append(dict(run=folder.name,steps=state['steps'],training_episodes=2,gradient_updates=2,
                            action_mapping_exact=True,replay_checkpoint_exact=True,actors_and_critic_changed=True))
    # The random smoke may contain no captures. Replay existing successful L
    # positive-control PUBLIC observations solely to test actual retirement.
    # No labels or transitions from this audit enter HASAC replay/training.
    m.configure(root/'lifecycle_audit_only');single_capture_transitions=0;checked=0
    control=m.l.OUT/'positive_control/evaluation/cache'
    for p in sorted(control.glob('*.json')):
        row=m.j.read(p);payload=m.j.read(control.parent/'episodes'/p.name);case=m.j.read(row['case'])
        own=Controller(case['map'],case['speed'],row['seed'],actors=None,uniform=True)
        for obs in case['history'][:-1]:own.observe(obs)
        for t in payload['trace']:
            own.act(t['observation']);assert own.belief.signature()==t['belief_signature']
            # A legal physical action is also a valid desired intent under
            # the same projection. Used only for this isolated replay audit.
            own.samples[-1]['action']=np.asarray(t['action'],np.float32)/case['speed']
            own.samples[-1]['blocked']=0.
        d=episode_arrays(own.samples,payload['trace'])
        for i,t in enumerate(payload['trace'][:-1]):
            if t['new_captures']==1:
                assert d['done'][i,0]==0
                assert int(d['active'][i+1].sum())==1
                assert np.max(np.abs(d['action'][i+1][d['active'][i+1,:,0]==0]))==0
                single_capture_transitions+=1
        checked+=1
    assert checked==6 and single_capture_transitions>0
    result=dict(passed=True,runs=reports,real_control_episodes_replayed=checked,
                first_capture_not_team_done_checked=single_capture_transitions,
                control_trajectories_used_for_training=False)
    m.j.save(root/'smoke_audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
