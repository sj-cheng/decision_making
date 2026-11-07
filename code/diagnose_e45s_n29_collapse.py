"""E45-S read-only audit; no optimizer update or new environment rollout."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import run_e45n_hasac_curriculum as n
from learning.e45m_author_hasac import AuthorHASAC
from learning.e45m_task_adapter import Controller,episode_arrays
from learning.e45g_ppo import safe_action
from e45g_training_acceleration import public_geometry_cache

ROOT=n.ROOT
OUT=ROOT/'current/e45s_n29_collapse_20261102'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as f:return json.load(f)
    return json.loads(path.read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
def stats(values):
    v=np.asarray(values).reshape(-1)
    if not len(v):return dict(n=0)
    return dict(n=len(v),mean=float(v.mean()),std=float(v.std()),minimum=float(v.min()),
        median=float(np.median(v)),p95=float(np.quantile(v,.95)),maximum=float(v.max()))


class Audit:
    def __init__(self,out):self.out=out;self.inputs={}
    def protect(self,path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        if expected is not None:assert digest==expected,str(path)
        self.inputs[str(path)]=digest
        return path
    def packet(self,path,expected=None):return read(self.protect(path,expected))
    def status(self,stage,**kw):
        row=dict(stage=stage,time=time.strftime('%FT%T%z'),**kw);save(self.out/'status.json',row)
        (self.out/'LIVE.txt').write_text('E45-S N29退化只读审计\n'+json.dumps(row,ensure_ascii=False,indent=2)+'\n优化更新0；新环境对局0。\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)


def load_model(path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    model=AuthorHASAC(obs_dim=845,seed=29,retirement_mask=True)
    for actor,weights in zip(model.actor,state['actors']):actor.actor.load_state_dict(weights);actor.actor.eval()
    for key in ('critic','critic2','target_critic','target_critic2'):
        getattr(model.critic,key).load_state_dict(state[key]);getattr(model.critic,key).eval()
    return model,state


def training_data(audit,cfg):
    root=n.OUT/'runs/HASAC_Curriculum_29';pool=[];trajectory=[]
    cumulative_success_transitions=0;cumulative_transitions=0
    for path in sorted((root/'updates').glob('*.json')):
        record=audit.packet(path);assert record['phase_index']==0
        actions=[];physical=[];short=[];visible=[];rewards=[];active_count=0;batch_success_steps=0
        for row in record['episodes']:
            archive=audit.protect(row['replay_episode'],row['replay_sha256'])
            payload=audit.packet(row['episode'],row['episode_sha256'])
            audit.protect(row['case']);frames=payload['trace']
            assert payload['summary']['public_replay_exact']
            with np.load(archive,allow_pickle=False) as z:
                data={key:z[key].copy() for key in z.files}
            assert len(data['obs'])==len(frames)==row['decisions']
            np.testing.assert_array_equal(data['action'],np.asarray([t['stats']['normalized_intent'] for t in frames],np.float32))
            np.testing.assert_array_equal(data['active'][...,0],np.asarray([t['observation']['active_p'] for t in frames],np.float32))
            cumulative_transitions+=len(frames)
            if row['double_capture']:
                cumulative_success_transitions+=len(frames);batch_success_steps+=len(frames)
            mask=data['active'][...,0]>0
            actions.extend(data['action'][mask]);physical.extend((np.asarray([t['action'] for t in frames])/read(row['case'])['speed'])[mask])
            short.extend(t['stats']['safety_shortened'] for t in frames);active_count+=int(mask.sum())
            visible.extend(any(q is not None for reports in t['observation']['reports'] for q in reports) for t in frames)
            rewards.extend(data['reward'][:,0])
            # Reproducible candidate sampling, not filtered by success.
            for i in np.unique(np.linspace(0,len(frames)-1,min(4,len(frames))).astype(int)):
                pool.append(dict(data={k:v[i] for k,v in data.items()},observation=frames[i]['observation'],
                    case=row['case'],step=record['steps'],iteration=record['iteration'],
                    episode=row['episode'],frame=int(i),successful_episode=bool(row['double_capture'])))
        a=np.asarray(actions);p=np.asarray(physical)
        trajectory.append(dict(steps=record['steps'],iteration=record['iteration'],score=record['score'],
            gradient_updates=record['gradient_updates'],replay_size=record['replay_size'],
            active_decisions=active_count,saturated_action_components=float(np.mean(np.abs(a)>.99)),
            near_zero_intent=float(np.mean(np.max(np.abs(a),axis=1)<.01)),
            near_zero_physical=float(np.mean(np.max(np.abs(p),axis=1)<.01)),
            safety_shortened_fraction=float(np.sum(short)/active_count),
            any_target_currently_visible_fraction=float(np.mean(visible)),
            training_rewards=stats(rewards),successful_episode_transition_fraction_this_batch=batch_success_steps/len(rewards),
            successful_episode_transition_fraction_in_cumulative_replay=cumulative_success_transitions/cumulative_transitions))
    assert cumulative_transitions==20568 and max(x['replay_size'] for x in trajectory)<cfg['replay_capacity']
    windows=[('early',0,6000),('proficient',6000,12000),('falling',12000,15000),('late',15000,22000)]
    selected=[];window_stats=[]
    for name,lo,hi in windows:
        eligible=[x for x in pool if lo<=x['step']<hi]
        ids=np.linspace(0,len(eligible)-1,min(192,len(eligible))).astype(int)
        selected.extend(dict(x,window=name) for i,x in enumerate(eligible) if i in set(ids))
        rows=[r for r in trajectory if lo<=r['steps']<hi]
        window_stats.append(dict(window=name,from_steps=lo,to_steps=hi,anchor_rows=len(ids),
            training_episodes=sum(r['score']['episodes'] for r in rows),
            training_doubles=sum(r['score']['double'] for r in rows),
            trajectory_records=rows))
    data={key:np.stack([x['data'][key] for x in selected]) for key in selected[0]['data']}
    np.savez_compressed(audit.out/'anchor_states.npz',**data)
    metadata=[{k:v for k,v in x.items() if k!='data'} for x in selected]
    save(audit.out/'anchor_metadata.json',metadata)
    save(audit.out/'training_timeline.json',dict(windows=window_stats,records=trajectory))
    return data,metadata,trajectory


@torch.no_grad()
def actor_values(model,obs):
    actions=[];mu=[];std=[]
    for i,actor in enumerate(model.actor):
        x=torch.as_tensor(obs[:,i],dtype=torch.float32)
        net=actor.actor;features=net.feature_extractor(x) if net.feature_extractor is not None else x
        z=net.net(features);m=net.mu_layer(z);s=net.log_std_layer(z).clamp(-20,2).exp()
        actions.append(actor.get_actions(x,stochastic=False).numpy());mu.append(m.numpy());std.append(s.numpy())
    return np.stack(actions,1),np.stack(mu,1),np.stack(std,1)


@torch.no_grad()
def q_values(model,central,action):
    x=torch.as_tensor(central,dtype=torch.float32);a=torch.as_tensor(action.reshape(len(action),4),dtype=torch.float32)
    q1=model.critic.critic(x,a).numpy()[:,0];q2=model.critic.critic2(x,a).numpy()[:,0]
    return q1,q2


def physical_actions(actions,metadata):
    actual=[];short=[];case_cache={}
    for a,row in zip(actions,metadata):
        case=case_cache.setdefault(row['case'],read(row['case']));obs=row['observation']
        moved,count=safe_action(np.asarray(obs['positions']),a,obs['active_p'],case['speed'],np.asarray(case['map']))
        actual.append(moved/case['speed']);short.append(count)
    return np.asarray(actual),np.asarray(short)


def checkpoint_comparison(audit,data,metadata):
    models={};states={};predictions={};out={}
    for seed in (29,17):
        folder=n.OUT/'runs'/f'HASAC_Curriculum_{seed}'/'phase_snapshots/phase_0/checkpoints'
        for path in sorted(folder.glob('*.pt'),key=lambda p:int(p.stem.split('_')[-1])):
            audit.protect(path);model,state=load_model(path)
            name=f'N{seed}_{state["steps"]}';models[name]=model;states[name]=state
            action,mu,std=actor_values(model,data['obs']);predictions[name]=action
            actual,short=physical_actions(action,metadata);mask=data['active'][...,0]>0
            q1,q2=q_values(model,data['central'],data['action'])
            subsets={}
            for window in ('all','early','proficient','falling','late'):
                keep=np.ones(len(mask),bool) if window=='all' else np.asarray([r['window']==window for r in metadata])
                m=mask[keep];a=action[keep][m];p=actual[keep][m]
                subsets[window]=dict(active_rows=int(m.sum()),latent_mu=stats(mu[keep][m]),latent_std=stats(std[keep][m]),
                    saturated_action_components=float(np.mean(np.abs(a)>.99)),near_zero_intent=float(np.mean(np.max(abs(a),axis=1)<.01)),
                    near_zero_physical=float(np.mean(np.max(abs(p),axis=1)<.01)),
                    safety_shortened_fraction=float(short[keep].sum()/m.sum()),
                    q_historical_actions=stats(np.minimum(q1,q2)[keep]),q_disagreement=stats(np.abs(q1-q2)[keep]))
            out[name]=dict(seed=seed,steps=state['steps'],gradient_updates=state['gradient_updates'],subsets=subsets)
            audit.status('checkpoint_analyzed',checkpoint=name)
    cross={}
    for name,model in models.items():
        cross[name]={other:stats(np.minimum(*q_values(model,data['central'],action))) for other,action in predictions.items()}
    save(audit.out/'checkpoint_comparison.json',dict(models=out,critic_actor_cross_scores=cross))
    return models,states,out


@torch.no_grad()
def expected_logp(model,obs,repeats=32):
    nrows=len(obs);logp=np.zeros(nrows)
    for i,actor in enumerate(model.actor):
        x=np.tile(obs[:,i],(repeats,1));torch.manual_seed(790+i)
        _,lp=actor.get_actions_with_logprobs(x)
        logp+=lp.numpy().reshape(repeats,nrows).mean(0)
    return logp


def on_policy_calibration(audit,models,states):
    output={}
    for name,model in models.items():
        seed=int(name.split('_')[0][1:]);state=states[name];rows=[]
        folder=n.OUT/'runs'/f'HASAC_Curriculum_{seed}'/'evaluation/episodes'
        for path in sorted(folder.glob('*.json.gz')):
            # Only the saved checkpoint's STATIC stochastic evaluations.
            key=path.name
            if f'_p0_' not in key or f'_eval_{state["steps"]}_p0' not in key or '_stochastic_' not in key:continue
            payload=audit.packet(path);row=payload['summary']
            assert row['opponent']=='fixed_D_kernel' and row['execution_mode']=='stochastic'
            case=audit.packet(row['case']);own=Controller(case['map'],case['speed'],row['seed'],None,uniform=True)
            for old in case['history'][:-1]:own.observe(old)
            with public_geometry_cache():
                for frame in payload['trace']:
                    own.act(frame['observation']);assert own.belief.signature()==frame['belief_signature']
                    own.samples[-1]['action']=np.asarray(frame['stats']['normalized_intent'],np.float32)
                    own.samples[-1]['blocked']=frame['stats']['safety_shortened']/max(1,sum(frame['observation']['active_p']))
            data=episode_arrays(own.samples,payload['trace'])
            action,_,_=actor_values(model,data['obs']);lp=expected_logp(model,data['obs'])
            value=float(data['reward'][-1,0]);returns=np.empty(len(lp));returns[-1]=value
            for t in range(len(lp)-2,-1,-1):
                value=float(data['reward'][t,0])+value-model.critic.alpha*lp[t+1];returns[t]=value
            q1,q2=q_values(model,data['central'],data['action']);q=np.minimum(q1,q2)
            # Deterministic current actions are diagnostic only; recorded
            # stochastic actions are what get compared with on-policy returns.
            rows.append(dict(case=row['case'],world=row['world'],decisions=len(lp),double=bool(row['double_capture']),
                first=bool(row['captures']),raw_utility=row['utility'],shaped_return=float(data['reward'].sum()),
                initial_soft_MC=float(returns[0]),initial_min_Q=float(q[0]),initial_Q_minus_MC=float(q[0]-returns[0]),
                all_frame_Q_minus_MC=stats(q-returns),expected_logp=stats(lp)))
        assert len(rows)==12,(name,len(rows))
        output[name]=dict(episodes=rows,initial_Q=stats([x['initial_min_Q'] for x in rows]),
            initial_soft_MC=stats([x['initial_soft_MC'] for x in rows]),
            initial_Q_minus_MC=stats([x['initial_Q_minus_MC'] for x in rows]),
            double_captures=sum(x['double'] for x in rows))
        save(audit.out/'on_policy_calibration.json',output)
        audit.status('on_policy_calibration',checkpoint=name,episodes=len(rows))
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);audit=Audit(args.output);torch.set_num_threads(1)
    started=time.time();cfg=n.verify(n.OUT)
    for path in [Path(__file__),n.OUT/'experiment.json',ROOT/'code/E45S_N29_COLLAPSE_AUDIT_PROTOCOL_20261102.md']:
        audit.protect(path)
    for section in ('source_hashes','protected_hashes','upstream_hashes','bank_hashes'):
        for path,value in cfg[section].items():audit.protect(path,value)
    audit.status('training_records')
    data,metadata,timeline=training_data(audit,cfg)
    audit.status('common_state_inference',anchor_rows=len(metadata))
    models,states,comparison=checkpoint_comparison(audit,data,metadata)
    calibration=on_policy_calibration(audit,models,states)
    assert all(sha(p)==value for p,value in audit.inputs.items())
    save(audit.out/'input_hashes.json',audit.inputs)
    result=dict(passed=True,optimizer_updates=0,new_environment_episodes=0,inputs_unchanged=True,
        protected_files=len(audit.inputs),anchor_rows=len(metadata),checkpoints=list(models),
        actor_critic_comparison=comparison,
        calibration={k:{field:v for field,v in value.items() if field!='episodes'} for k,value in calibration.items()},
        wall_seconds=time.time()-started,
        limitations=['Only 0/10k/20k checkpoints exist; latent policy/Q changes cannot be temporally located more finely without a new controlled replay experiment.',
        'Common-state cross-scores are descriptive and do not prove counterfactual action quality.',
        'Soft Monte Carlo uses 32 fresh policy samples per stored state for expected entropy, without inverse tanh of censored recorded actions.',
        'The six existing training cases are reused; frames and episodes are not independent training seeds.'])
    save(audit.out/'summary.json',result);audit.status('complete',wall_seconds=result['wall_seconds'])


if __name__=='__main__':main()
