"""Read-only completion and on-policy soft-Q audit for completed E45-T."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import torch
import run_e45t_critic_lr_stability as t
import diagnose_e45s_n29_collapse as s


def equal_core(a,b):
    if len(a)!=len(b):return False
    return all(x['observation']==y['observation'] and x['reward']==y['reward'] and
        np.array_equal(x['action'],y['action']) and np.array_equal(x['next_state'],y['next_state']) and
        np.array_equal(x['opponent_action'],y['opponent_action']) for x,y in zip(a,b))


def calibrate(folder,step,model):
    rows=[]
    for path in sorted((folder/'evaluation/episodes').glob('*.json.gz')):
        if f'_T_eval_{step}_' not in path.name or '_stochastic_' not in path.name:continue
        payload=s.read(path);row=payload['summary'];case=s.read(row['case'])
        assert row['public_replay_exact']
        own=s.Controller(case['map'],case['speed'],row['seed'],None,uniform=True)
        for obs in case['history'][:-1]:own.observe(obs)
        with s.public_geometry_cache():
            for frame in payload['trace']:
                own.act(frame['observation']);assert own.belief.signature()==frame['belief_signature']
                own.samples[-1]['action']=np.asarray(frame['stats']['normalized_intent'],np.float32)
                own.samples[-1]['blocked']=frame['stats']['safety_shortened']/max(1,sum(frame['observation']['active_p']))
        data=s.episode_arrays(own.samples,payload['trace'])
        lp=s.expected_logp(model,data['obs']);value=float(data['reward'][-1,0])
        for i in range(len(lp)-2,-1,-1):value=float(data['reward'][i,0])+value-model.critic.alpha*lp[i+1]
        q1,q2=s.q_values(model,data['central'][:1],data['action'][:1]);q=float(min(q1[0],q2[0]))
        rows.append(dict(case=row['case'],world=row['world'],double=bool(row['double_capture']),
            initial_Q=q,initial_soft_MC=value,initial_error=q-value,raw_utility=row['utility']))
    assert len(rows)==12,(folder,step,len(rows))
    return dict(steps=step,episodes=rows,initial_Q=s.stats([x['initial_Q'] for x in rows]),
        initial_soft_MC=s.stats([x['initial_soft_MC'] for x in rows]),
        initial_error=s.stats([x['initial_error'] for x in rows]),double=sum(x['double'] for x in rows))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=t.OUT);args=parser.parse_args()
    root=args.output;cfg=t.verify(root);torch.set_num_threads(1)
    assert t.j.read(root/'decision.json')['status']=='complete' and not cfg['smoke']
    reports=[];all_calibration={}
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            folder=root/'runs'/f'{arm}_{seed}';decision=t.j.read(folder/'decision.json')
            assert decision['status']=='complete' and decision['new_gradient_updates']==decision['steps']
            assert cfg['additional_online_steps']<=decision['steps']<cfg['additional_online_steps']+960
            records=[t.j.read(p) for p in sorted((folder/'updates').glob('*.json'))]
            rows=[e for r in records for e in r['episodes']]
            assert sum(r['decisions'] for r in rows)==decision['steps']
            assert not list((folder/'failures').glob('*.json'))
            comparison=[]
            if arm=='original':
                for record in records:
                    old=t.j.read(t.n.OUT/'runs'/f'HASAC_Curriculum_{seed}'/'updates'/f'{record["iteration"]:05d}.json')
                    for newrow,oldrow in zip(record['episodes'],old['episodes']):
                        new=s.read(newrow['episode'])['trace'];past=s.read(oldrow['episode'])['trace']
                        comparison.append(equal_core(new,past))
            calibration=[]
            for row in decision['history']:
                step=row['steps'];path=folder/'checkpoints'/f'steps_{step}.pt'
                model,state=s.load_model(path)
                assert state['steps']==step
                assert state['gradient_updates']==cfg['sources'][str(seed)]['gradient_updates']+step
                assert state['history'][-1]==row
                for group in state['critic_optimizer']['param_groups']:assert group['lr']==cfg['critic_lr_by_arm'][arm]
                for optimizer in state['actor_optimizers']:
                    for group in optimizer['param_groups']:assert group['lr']==5e-4
                point=calibrate(folder,step,model);calibration.append(point)
                del model,state
                t.j.save(root/'calibration_progress.json',dict(run=f'{arm}_{seed}',steps=step))
            pointer=t.j.read(folder/'latest.json');assert t.j.sha(pointer['path'])==pointer['sha256']
            state=torch.load(pointer['path'],map_location='cpu',weights_only=False)
            assert t.j.sha(state['replay_path'])==state['replay_sha256']
            all_calibration[f'{arm}_{seed}']=calibration
            reports.append(dict(arm=arm,seed=seed,new_steps=decision['steps'],new_updates=decision['new_gradient_updates'],
                training_episodes=len(rows),evaluation_points=len(decision['history']),
                original_N_trajectories_checked=len(comparison),original_N_trajectories_exact=sum(comparison),
                final_deterministic=decision['history'][-1]['deterministic'],final_stochastic=decision['history'][-1]['stochastic'],
                initial_Q_error=calibration[0]['initial_error'],final_Q_error=calibration[-1]['initial_error'],
                checkpoint_and_replay_hash_verified=True))
            t.j.save(root/'on_policy_calibration.json',all_calibration)
    t.verify(root)
    result=dict(passed=True,optimizer_updates_during_audit=0,new_environment_episodes_during_audit=0,
        total_new_online_steps=sum(r['new_steps'] for r in reports),total_new_RL_updates=sum(r['new_updates'] for r in reports),
        source_hashes_verified=True,runs=reports,audit_source_sha256=t.j.sha(Path(__file__).resolve()),
        helper_source_sha256=t.j.sha(Path(s.__file__).resolve()),
        limitation='Soft MC uses 32 frozen-policy entropy samples on stored stochastic evaluation paths; reused training cases, not independent generalization.')
    t.j.save(root/'completion_audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
