"""Read-only on-policy soft-return diagnostic on fixed M evaluation traces."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import gzip
import json
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import run_e45m_hasac_task as m

OUT=m.ROOT/'current/e45_marl_macro_audit_20261101/hasac_value_calibration_20k'


class LoggingActor:
    def __init__(self,actor):self.actor=actor;self.logps=[]
    def get_actions(self,obs,stochastic=True):
        action,logp=self.actor.get_actions_with_logprobs(obs,stochastic=stochastic)
        self.logps.append(float(logp.item()))
        return action


def describe(q,g,mask):
    q=q[mask];g=g[mask]
    return dict(transitions=len(q),mean_q=float(q.mean()),mean_realized_soft_return=float(g.mean()),
                mean_error=float((q-g).mean()),rmse=float(np.sqrt(np.mean((q-g)**2))),
                positive_error_fraction=float(np.mean(q>g)))


def soft_returns(reward,logp,alpha):
    result=np.zeros(len(reward),np.float64);future=0.
    for t in range(len(reward)-1,-1,-1):
        result[t]=float(reward[t])+future
        future=result[t]-alpha*float(logp[t])
    return result


def main(minimum_steps=20000):
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    # Finite horizon: terminal Q is immediate reward; entropy starts at t+1.
    np.testing.assert_allclose(soft_returns([1,2],[-3,-4],.1),[3.4,2])
    records={}
    for folder in sorted(m.OUT.glob('runs/*')):
        history=m.j.read(folder/'evaluation.json');point=next(x for x in history if x['steps']>=minimum_steps)
        checkpoint=folder/'checkpoints'/f'steps_{point["steps"]}.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['experiment_sha256']==m.j.sha(m.OUT/'experiment.json')
        model=m.AuthorHASAC(obs_dim=845,device='cpu',seed=0,retirement_mask=True)
        for actor,weights in zip(model.actor,state['actors']):actor.actor.load_state_dict(weights);actor.actor.eval()
        for key in ('critic','critic2'):getattr(model.critic,key).load_state_dict(state[key]);getattr(model.critic,key).eval()
        rows=[m.j.read(p) for p in sorted((folder/'evaluation/cache').glob('*.json'))]
        rows=[r for r in rows if r['stage']==f'hasac_eval_steps{point["steps"]}' and r['execution_mode']=='stochastic']
        assert len(rows)==12,len(rows)
        episodes=[];values=[];returns=[];remaining=[];roots=[]
        for row in rows:
            assert m.j.sha(row['episode'])==row['episode_sha256']
            payload=json.loads(gzip.decompress(Path(row['episode']).read_bytes()));trace=payload['trace'];case=m.j.read(row['case'])
            logged=[LoggingActor(a) for a in model.actor]
            own=m.Controller(case['map'],case['speed'],row['seed'],logged,stochastic=True,policy_seed=row['policy_seed'])
            for obs in case['history'][:-1]:own.observe(obs)
            with torch.inference_mode(),m.public_geometry_cache():
                for t in trace:
                    action=own.act(t['observation']);np.testing.assert_array_equal(action,t['action'])
                    assert own.belief.signature()==t['belief_signature']
                data=m.episode_arrays(own.samples,trace)
                q=model.critic.get_values(data['central'],data['action'].reshape(len(trace),4)).numpy().reshape(-1)
            logp=np.asarray([a.logps for a in logged]).sum(0)
            assert len(logp)==len(trace)
            g=soft_returns(data['reward'].reshape(-1),logp,.001)
            np.testing.assert_allclose(g[-1],data['reward'][-1,0],atol=0,rtol=0)
            ep=dict(key=row['key'],source=row['episode'],episode_sha256=row['episode_sha256'],
                    steps=len(trace),captures=payload['summary']['captures'],
                    root_q=float(q[0]),root_realized_soft_return=float(g[0]),
                    terminal_q=float(q[-1]),terminal_reward=float(data['reward'][-1,0]),
                    raw_return=float(sum(t['reward'] for t in trace)),shaped_return=float(data['reward'].sum()),
                    initial_phi=own.samples[0]['phi'],illegal_intent_penalty=.05*sum(s['blocked'] for s in own.samples),
                    joint_entropy_mc_mean=float(-logp.mean()),
                    root_entropy_correction=float(g[0]-data['reward'].sum()))
            np.testing.assert_allclose(ep['shaped_return'],ep['raw_return']-ep['initial_phi']-ep['illegal_intent_penalty'],atol=2e-6,rtol=0)
            episodes.append(ep);values.extend(q.tolist());returns.extend(g.tolist())
            remaining.extend(range(len(trace),0,-1));roots.extend([True]+[False]*(len(trace)-1))
            print(folder.name,len(episodes),'/12 root Q/return',ep['root_q'],ep['root_realized_soft_return'],flush=True)
        q=np.asarray(values);g=np.asarray(returns);left=np.asarray(remaining);root=np.asarray(roots)
        records[folder.name]=dict(checkpoint=str(checkpoint),checkpoint_sha256=m.j.sha(checkpoint),
            alpha=.001,gamma=1.,episodes=episodes,statistics={
                name:describe(q,g,mask) for name,mask in [('all',left>0),('roots',root),('last_10',left<=10),('terminal',left==1)]})
        m.j.save(OUT/'partial.json',records)
    m.j.save(OUT/'audit.json',dict(scope='descriptive on-policy critic/realized soft-return comparison on reused training-situation evaluation, not independent testing',
        results=records,source_sha256=m.j.sha(Path(__file__)),all_actor_actions_and_beliefs_match=True,
        truth_used_only_by_audited_training_critic=True,training_unchanged=True,
        inference_limit='Realized future returns are noisy. Episode frames are correlated; no independent-frame confidence claim. Terminal targets have no future sampling uncertainty.'))
    print('COMPLETE',flush=True)


def watch():
    global OUT
    review=m.ROOT/'current/e45_marl_macro_audit_20261101/hasac_value_calibration_curve'
    config=m.j.read(m.OUT/'experiment.json');curve={}
    for threshold in (20000,40000,60000,80000,100000):
        OUT=review.parent/f'hasac_value_calibration_{threshold//1000}k'
        while True:
            waiting=[]
            for seed in config['seeds']:
                folder=m.OUT/'runs'/f'HASAC_{seed}';history=m.j.read(folder/'evaluation.json') if (folder/'evaluation.json').exists() else []
                ready=any(x['steps']>=threshold and (folder/'checkpoints'/f'steps_{x["steps"]}.pt').exists() for x in history)
                if ready:continue
                status=m.j.read(folder/'status.json');command=Path('/proc')/str(status['pid'])/'cmdline'
                if not command.exists() or b'run_e45m_hasac_task.py train' not in command.read_bytes().replace(b'\0',b' '):
                    raise RuntimeError(f'Source trainer {seed} unavailable before checkpoint {threshold}; no restart attempted')
                waiting.append(dict(seed=seed,pid=status['pid'],steps=status.get('steps')))
            if not waiting:break
            m.j.common.emit(review,'waiting_for_checkpoint',minimum_steps=threshold,verified_live_trainers=waiting)
            time.sleep(30)
        report=OUT/'audit.json'
        if not report.exists():main(threshold)
        data=m.j.read(report)
        for seed in config['seeds']:
            value=data['results'][f'HASAC_{seed}'];assert m.j.sha(value['checkpoint'])==value['checkpoint_sha256']
        curve[str(threshold)]=dict(report=str(report),report_sha256=m.j.sha(report),
                                  results={key:value['statistics'] for key,value in data['results'].items()})
        m.j.save(review/'curve.json',dict(scope='all registered M checkpoint thresholds, reused on-policy evaluation diagnostics; no selection or independent test',
            checkpoints=curve,source_sha256=m.j.sha(Path(__file__))))
        m.j.common.emit(review,'checkpoint_audited',minimum_steps=threshold)
    m.j.common.emit(review,'complete',checkpoints=len(curve))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--minimum-steps',type=int,default=20000)
    args=parser.parse_args()
    if args.watch:watch()
    else:
        OUT=OUT.parent/f'hasac_value_calibration_{args.minimum_steps//1000}k'
        main(args.minimum_steps)
