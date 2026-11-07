"""Q smoke: actual learner-state labels, no oracle actuation, resumed RL."""
from pathlib import Path
import gzip
import json
import numpy as np
import torch
import run_e45q_corrective_hasac as q
from learning.e45m_author_hasac import AuthorHASAC
from learning.e45o_demonstration_replay import DemonstrationReplay
from learning.e45p_actor_supervision import public_teacher,install_anchor


def main():
    torch.set_num_threads(1)
    root=q.ROOT/'current/e45q_corrective_hasac_smoke_20261101';cfg=q.verify(root)
    assert cfg['smoke'] and q.j.read(root/'decision.json')['status']=='complete'
    reports=[];all_different=0
    for seed in cfg['seeds']:
        folder=root/'imitation'/f'seed_{seed}'
        base=q.j.read(folder/'base/latest.json');initial=torch.load(base['path'],map_location='cpu',weights_only=False)
        collected=q.j.read(folder/'round_1_collection.json');assert len(collected['episodes'])==2
        for row in collected['episodes']:
            assert q.j.sha(row['labels_path'])==row['labels_sha256'] and not row['teacher_actuates']
            payload=json.loads(gzip.decompress(Path(row['episode']).read_bytes()))
            records=[t['stats']['teacher_query'] for t in payload['trace']]
            assert all(r['public_belief_exact'] and r['simulation_audit_passed'] and not r['teacher_actuates'] for r in records)
            valid=[i for i,r in enumerate(records) if r['valid']]
            with np.load(row['labels_path']) as labels,np.load(row['replay_episode']) as replay:
                np.testing.assert_array_equal(labels['obs'],replay['obs'][valid])
                np.testing.assert_array_equal(labels['active'],replay['active'][valid])
                for k,i in enumerate(valid):
                    np.testing.assert_allclose(labels['action'][k],records[i]['normalized_teacher_action'],atol=1e-7,rtol=0)
                    all_different+=not np.allclose(labels['action'][k],replay['action'][i],atol=1e-4)
        for route in q.ROUTES:
            final_pointer=q.j.read(folder/route/'latest.json')
            final=torch.load(final_pointer['path'],map_location='cpu',weights_only=False)
            assert final['steps']==4 and final['gradient_updates']==0
            for key in ('critic','critic2','target_critic','target_critic2'):
                for name,value in initial[key].items():torch.testing.assert_close(value,final[key][name],rtol=0,atol=0)
            for i in (0,1):
                for key in ('log_std_layer.weight','log_std_layer.bias'):
                    torch.testing.assert_close(initial['actors'][i][key],final['actors'][i][key],rtol=0,atol=0)
        selected=q.j.read(root/'bc'/f'HASAC_{seed}'/'decision.json')
        with np.load(selected['supervision_path']) as z:labels={k:z[k].copy() for k in z.files}
        assert len(labels['obs'])>cfg['demonstration_steps']
        data=q.p.load_data(cfg)
        initial_rl=[]
        for arm in cfg['arms']:
            f=root/'runs'/f'{arm}_{seed}';decision=q.j.read(f/'decision.json')
            assert decision['gradient_updates']==2
            assert decision['anchor']['calls']==(4 if arm=='replay_bc' else 0)
            initial_rl.append(torch.load(f/'checkpoints/steps_0.pt',map_location='cpu',weights_only=False))
            for optimizer in initial_rl[-1]['actor_optimizers']:
                assert not optimizer['state'] and optimizer['param_groups'][0]['lr']==1e-4
            resumed=[]
            for _ in range(2):
                model=AuthorHASAC(obs_dim=845,device='cuda:2',seed=seed,retirement_mask=True)
                replay=DemonstrationReplay(data,cfg['replay_capacity'],seed,cfg['demonstration_fraction'])
                q.o.restore(f,model,replay,q.j.sha(root/'experiment.json'))
                handles,_=install_anchor(model,public_teacher(labels,'cuda:2'),cfg['anchor_weight'] if arm=='replay_bc' else 0.)
                model.update(replay.sample(cfg['update_batch']))
                resumed.append({name:[p.detach().cpu().clone() for p in module.parameters()] for name,module in
                    [('actor0',model.actor[0].actor),('actor1',model.actor[1].actor),('critic',model.critic.critic),('target',model.critic.target_critic)]})
                for handle in handles:handle.remove()
                del model
            for name in resumed[0]:
                for a,b in zip(resumed[0][name],resumed[1][name]):torch.testing.assert_close(a,b,rtol=0,atol=0)
            reports.append(dict(seed=seed,arm=arm,resumed_update_exact=True,real_updates=2))
        for i in (0,1):
            for name,value in initial_rl[0]['actors'][i].items():torch.testing.assert_close(value,initial_rl[1]['actors'][i][name],rtol=0,atol=0)
    assert all_different>0
    result=dict(passed=True,runs=reports,teacher_labels_different_from_executed_actions_checked=all_different,
        learner_public_features_exact=True,teacher_never_actuates=True,oracle_simulations_audited=True,
        augmented_labels_used_only_for_supervision=True,smoke_gate_bypass_only=True,capability_claim=False)
    q.j.save(root/'smoke_audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
