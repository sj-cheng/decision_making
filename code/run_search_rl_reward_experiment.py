"""Isolated pursuit reward study; never edits the original environment reward.

Only the pursuer actors/critic are updated. Evaluation and all MCTS use the
unchanged environment reward, so reward scales cannot inflate success metrics.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.optim import Adam
import train_search_rl as trainer
from run_search_rl_ablation import compose_initial, state_hash, probe, gaussian_kl
from learning.local_rl import PolicyBundle, atomic_torch_save, critic_features
from local_observation import TEAMS, curriculum_episode, make_episode, observe
from solvers.belief_mcts import LocalController, BeliefMCTS, public_model, enable_fast_model
from benchmark_local_obstacles import evaluation_task

ARMS = ('legacy', 'completion_bonus_v1', 'capture_floor_v1')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reward_increment(arm, before, after, time_value, horizon, legal):
    """Pursuer-only extra reward; invalid endings can never earn a bonus."""
    if arm not in ARMS:
        raise ValueError(arm)
    captured = before-after
    if not legal or captured <= 0:
        return 0.
    if arm == 'completion_bonus_v1':
        return 1. if before > 0 and after == 0 else 0.
    if arm == 'capture_floor_v1':
        return .5*captured*np.clip(time_value/horizon, 0., 1.)
    return 0.


def reward_rollout(task):
    """Same PPO rollout as trainer, instrumented before/after capture events."""
    seed = task['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    bundle = PolicyBundle(scratch=True); bundle.load_state_dict(trainer.unwire_weights(task['bundle']))
    opponent = PolicyBundle(scratch=True); opponent.load_state_dict(trainer.unwire_weights(task['opponent']))
    spec = curriculum_episode(seed, task['progress'])
    p, state = spec.restore(); enable_fast_model(p)
    role = task['role']
    own = LocalController(public_model(spec.obstacles), bundle, role, seed)
    cls = BeliefMCTS if task['search_opponent'] else LocalController
    kw = dict(simulations=task['opponent_budget'], depth=task['depth']) if task['search_opponent'] else {}
    other = cls(public_model(spec.obstacles), opponent, 1-role, seed+1, **kw)
    rows = []; totals = np.zeros(2); transitions = 0; first = None; second = None
    for step in range(80):
        if p.is_terminal(state):
            break
        obs = observe(p, state, role)
        chosen, info = own.act(obs, stochastic=True)
        opposing, _ = other.act(observe(p, state, 1-role))
        transitions += own.last_stats.get('transitions', 0)+other.last_stats.get('transitions', 0)
        action = np.zeros((8,1))
        for team, values in ((role,chosen),(1-role,opposing)):
            for local, robot in enumerate(TEAMS[team]):
                action[p.action_idxs[robot],0] = values[local]
        nxt = p.step(state, action, p.dt)
        reward = p.transition_reward(state,nxt)
        totals += [np.mean(reward[list(TEAMS[t]),0]) for t in range(2)]
        illegal = [any(not trainer._valid_robot(p,nxt,r) for r in TEAMS[t]) for t in range(2)]
        before = int(p.active_evader_count(state)); after = int(p.active_evader_count(nxt))
        now = float(nxt[p.time_idx,0])
        if after < before:
            if first is None: first = now
            if after == 0: second = now
        base = float(np.mean(reward[list(TEAMS[role]),0]))
        rows.append(dict(obs=info['features'], active=obs.active.astype(np.float32),
                         limits=obs.limits.astype(np.float32), raw=info['raw'].astype(np.float32),
                         old_logp=info['old_logp'], critic=critic_features(p,state,info['features']),
                         reward=base, done=bool(p.is_terminal(nxt)), legacy_reward=base,
                         reward_increments={a:reward_increment(a,before,after,now,p.tf,not any(illegal)) for a in ARMS}))
        state = nxt
    illegal = [any(not trainer._valid_robot(p,state,r) for r in TEAMS[t]) for t in range(2)]
    remaining = int(p.active_evader_count(state)); now = float(state[p.time_idx,0])
    stats = dict(seed=seed,role=role,steps=len(rows),double=float(remaining==0 and not any(illegal)),
                 survival=float(remaining>0 and now>=p.tf and not any(illegal)),illegal=illegal,
                 returns=totals.tolist(),transitions=transitions,first_capture_time=first,
                 second_capture_time=second,remaining_evaders=remaining,
                 legal_one_capture_timeout=bool(now>=p.tf and remaining==1 and not any(illegal)),
                 reward_totals={a:float(sum(r['legacy_reward']+r['reward_increments'][a] for r in rows)) for a in ARMS})
    return dict(rows=rows,spec=asdict(spec),stats=stats)


def relabel(results,arm):
    # Only rewards change; observations, actions and behavior logprobs do not.
    return [dict(result,rows=[dict(row,reward=row['legacy_reward']+row['reward_increments'][arm])
                             for row in result['rows']]) for result in results]


def planned_tasks(args,seed,offset,weights,initial_wire,anchor_wire):
    tasks=[]
    for k in range(args.batch):
        index=args.start_index+offset+k
        rng=np.random.default_rng(trainer.episode_seed(seed,503,1,index))
        draw=rng.random()
        tasks.append(dict(seed=trainer.episode_seed(seed,502,1,index),role=1,kind='ppo',
                          progress=index/args.horizon,bundle=weights,
                          opponent=initial_wire if .45<=draw<.9 else anchor_wire,
                          search_opponent=bool(draw>=.9),opponent_budget=args.opponent_budget,
                          depth=8,budget=1000,training_index=index,
                          opponent_kind='search' if draw>=.9 else 'history' if draw>=.45 else 'distilled'))
    return tasks


def collect(pool,tasks):
    if pool:
        return list(pool.map(reward_rollout,tasks,chunksize=1))
    saved=trainer.rng_state()
    try: return [reward_rollout(t) for t in tasks]
    finally: trainer.restore_rng(saved)


def evaluate(args,out,pool,checkpoints,split,phase,count,methods=('ppo',),roles=(1,)):
    folder=out/split; folder.mkdir(exist_ok=True)
    tasks=[]; manifest=[]
    for label,seed,path in checkpoints:
        for role in roles:
            for method in methods:
                for li,layout in enumerate(('two','four','random')):
                    for i in range(count):
                        spec=make_episode(trainer.episode_seed(20261022,phase+li,role,i),layout)
                        key=f'{label}_s{seed}_r{role}_{method}_{layout}_{i}'
                        checkpoint_hash=digest(path)
                        task=dict(spec=asdict(spec),checkpoint=str(path),reference=str(out/'distilled.pt'),
                                  role=role,method=method,opponent='distilled',budget=args.search_budget,depth=args.search_depth,
                                  key=key,label=label,diagnostic_seed=seed)
                        entry=dict(key=key,checkpoint_sha256=checkpoint_hash,episode_hash=spec.fingerprint,
                                   seed=seed,label=label,role=role,method=method,spec=asdict(spec))
                        manifest.append(entry)
                        cache=folder/(key+'.json')
                        if cache.exists():
                            prior=json.loads(cache.read_text())
                            if prior['checkpoint_sha256']!=checkpoint_hash:
                                raise ValueError(f'Stale evaluation cache: {cache}')
                        else: tasks.append((task,checkpoint_hash))
    trainer.write_json(folder/'manifest.json',manifest)
    iterator=pool.imap(evaluation_task,[t for t,_ in tasks],chunksize=1) if pool else map(evaluation_task,[t for t,_ in tasks])
    for i,((task,chash),row) in enumerate(zip(tasks,iterator),1):
        row.update(label=task['label'],diagnostic_seed=task['diagnostic_seed'],key=task['key'],checkpoint_sha256=chash)
        trainer.write_json(folder/(task['key']+'.json'),row)
        if i%20==0 or i==len(tasks):print(f'EVAL {split} {i}/{len(tasks)}',flush=True)
    rows=[json.loads((folder/(entry['key']+'.json')).read_text()) for entry in manifest]
    trainer.write_json(folder/'summary.json',summarize(rows))
    return rows


def summarize(rows):
    groups={}
    for r in rows:groups.setdefault((r['label'],r['diagnostic_seed'],r['role'],r['method']),[]).append(r)
    result=[]
    for (label,seed,role,method),rs in groups.items():
        own='pursuer' if role==1 else 'evader'
        result.append(dict(label=label,seed=seed,role=role,method=method,n=len(rs),
                           success=sum(r['double_capture' if role==1 else 'legal_survival'] for r in rs),
                           own_illegal=sum(r[own+'_illegal'] for r in rs),
                           own_collision=sum(r[own+'_obstacle_collision'] for r in rs),
                           one_capture_timeout=sum(r['steps']==80 and r['first_capture_time'] is not None and not r['double_capture']
                                                   and not(r['evader_illegal'] or r['pursuer_illegal']) for r in rs),
                           environment_return=float(np.mean([r[own+'_return'] for r in rs])),
                           mean_latency_ms=float(np.mean([r['tested_latency_ms'] for r in rs]))))
    return result


def validation_score(rows):
    legal=[r for r in rows if r['double_capture']]
    return (float(np.mean([r['double_capture'] for r in rows])),
            -float(np.mean([r['pursuer_illegal'] for r in rows])),
            -float(np.mean([r['capture_time'] for r in legal])) if legal else -80.)


def train_to(args,out,pool,arms,target,initial,anchor):
    contexts=[]
    for seed in args.seeds:
        for arm in arms:
            folder=out/f'seed{seed}'/arm; folder.mkdir(parents=True,exist_ok=True)
            bundle=initial.copy()
            actor=Adam([p for robot in TEAMS[1] for p in bundle.actors[robot].parameters()],lr=3e-5)
            critic=Adam(bundle.critics[1].parameters(),lr=3e-4)
            count=0
            path=folder/'training_state.pt'
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=False)
                bundle.load_state_dict(saved['bundle']);count=saved['count']
                actor.load_state_dict(saved['optimizers'][0]);critic.load_state_dict(saved['optimizers'][1])
            contexts.append(dict(seed=seed,arm=arm,folder=folder,bundle=bundle,actor=actor,critic=critic,count=count))
    initial_wire=trainer.wire_weights(initial.state_dict());anchor_wire=trainer.wire_weights(anchor.state_dict())
    for offset in range(0,target,args.batch):
        for seed in args.seeds:
            pending=[c for c in contexts if c['seed']==seed and c['count']==offset]
            if not pending:continue
            hashes=[state_hash(c['bundle'],1) for c in pending]
            shared=len(set(hashes))==1
            task_groups=[planned_tasks(args,seed,offset,trainer.wire_weights(c['bundle'].state_dict()),initial_wire,anchor_wire)
                         for c in (pending[:1] if shared else pending)]
            started=time.perf_counter();results=collect(pool,[t for group in task_groups for t in group])
            rollout_seconds=time.perf_counter()-started
            for i,c in enumerate(pending):
                chunk=results[:args.batch] if shared else results[i*args.batch:(i+1)*args.batch]
                tasks=task_groups[0 if shared else i]
                labeled=relabel(chunk,c['arm']);before=probe(c['bundle'],anchor,1,labeled)
                error=float((before['logp']-before['old_logp']).abs().max())
                if error>2e-3:raise RuntimeError(f'Off-policy rollout: {error}')
                update_seed=trainer.episode_seed(seed,504,1,offset)
                random.seed(update_seed);np.random.seed(update_seed);torch.manual_seed(update_seed)
                options=argparse.Namespace(anchor=0.,ppo_epochs=4,minibatch=256,clip=.1,entropy=.001)
                metrics=trainer.ppo_update(c['bundle'],anchor,1,labeled,c['actor'],c['critic'],options,progress=0.)
                after=probe(c['bundle'],anchor,1,labeled);c['count']+=args.batch
                c['bundle'].metadata.update(training_seed=seed,training_reward_id=c['arm'],
                    diagnostic_episodes=c['count'],frozen_evader=True,value_validated=False,
                    source='episode512_team_best',search_reward='legacy_environment')
                assert state_hash(c['bundle'],0)==state_hash(initial,0),'Frozen evader changed'
                snapshot=c['folder']/f'policy_ep{c["count"]}.pt'
                c['bundle'].save(snapshot);c['bundle'].save(c['folder']/'latest.pt')
                atomic_torch_save(dict(bundle=c['bundle'].state_dict(),count=c['count'],
                                      optimizers=[c['actor'].state_dict(),c['critic'].state_dict()]),c['folder']/'training_state.pt')
                record=dict(seed=seed,arm=c['arm'],episodes=c['count'],start_index=args.start_index+offset,
                    own_state_before=hashes[i],own_state_after=state_hash(c['bundle'],1),shared_rollouts=shared,
                    old_logp_error=error,rollout_seconds=rollout_seconds,post_update_kl=gaussian_kl(before,after),
                    search_episodes=sum(t['search_opponent'] for t in tasks),
                    actual_reward_mean=float(np.mean([r['reward'] for e in labeled for r in e['rows']])),**metrics)
                with (c['folder']/'metrics.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
                with (c['folder']/'episodes.jsonl').open('a') as handle:
                    for task,result in zip(tasks,chunk):
                        handle.write(json.dumps(dict(training_index=task['training_index'],opponent=task['opponent_kind'],
                                                    spec=result['spec'],stats=result['stats']))+'\n')
                print('TRAIN '+json.dumps(record),flush=True)
    return contexts


def checkpoints(out,arms,seeds,episodes):
    return [('initial',None,out/'initial.pt')]+[(arm,seed,out/f'seed{seed}'/arm/f'policy_ep{episodes}.pt') for seed in seeds for arm in arms]


def select_reward(rows,seeds):
    scores=[]
    for arm in ARMS[1:]:
        by_seed=[validation_score([r for r in rows if r['label']==arm and r['diagnostic_seed']==s]) for s in seeds]
        scores.append(dict(arm=arm,mean_success=float(np.mean([v[0] for v in by_seed])),
                           worst_success=min(v[0] for v in by_seed),mean_negative_illegal=float(np.mean([v[1] for v in by_seed]))))
    return sorted(scores,key=lambda r:(r['mean_success'],r['worst_success'],r['mean_negative_illegal'],r['arm']),reverse=True)


def paired(rows,baseline,candidate,seeds,role,method):
    metric='double_capture' if role==1 else 'legal_survival'
    old={(r['diagnostic_seed'],r['episode_hash']):r for r in rows if r['label']==baseline and r['role']==role and r['method']==method}
    changes=[]
    for seed in seeds:
        rs=sorted([r for r in rows if r['label']==candidate and r['diagnostic_seed']==seed and r['role']==role and r['method']==method],key=lambda r:r['episode_hash'])
        changes.append([r[metric]-old[(None if baseline=='initial' else seed,r['episode_hash'])][metric] for r in rs])
    changes=np.asarray(changes);rng=np.random.default_rng(654991);boot=[]
    if not changes.size:return None
    for _ in range(4000):
        si=rng.integers(len(seeds),size=len(seeds));ei=rng.integers(changes.shape[1],size=changes.shape[1])
        boot.append(float(changes[si][:,ei].mean()))
    return dict(baseline=baseline,candidate=candidate,role=role,method=method,delta=float(changes.mean()),
                per_seed_delta=changes.mean(1).tolist(),ci95=np.percentile(boot,[2.5,97.5]).tolist(),
                note='Paired scenes and sampling seeds; shared pretrained seed, exploratory intervals.')


def report(args,out,selected,short_rows,long_rows,search_rows):
    comparisons=[]
    for rs,method in [(long_rows,'ppo'),(search_rows,'search_ppo')]:
        for role in (0,1):
            for base in ('initial','legacy'):
                result=paired(rs,base,selected,args.seeds,role,method)
                if result:comparisons.append(result)
    result=dict(selected_reward=selected,short=summarize(short_rows),long=summarize(long_rows),
                search=summarize(search_rows),comparisons=comparisons,continue_formal=False,
                note='Reward-only pursuer study. Frozen evader. Longer training and full-budget MCTS completed; no automatic 50k run.')
    trainer.write_json(out/'results.json',result)
    text=['# 第二次捕获奖励实验','',
          '原第512局最佳为冻结对照；只训练追捕，逃逸策略/critic/归一化均冻结。网络、学习率3e-5、课程、对手混合和PPO参数相同。',
          '原训练环境、评估环境与所有搜索仍使用原奖励；只在PPO训练样本中改变追捕奖励。',
          'completion_bonus_v1：原奖励+合法双捕事件1.0；capture_floor_v1：将每次合法捕获的0.5*(1-t/80)补到0.5；legacy为原奖励。非法终局不给新增奖励。',
          f'短训练每种子/每组{args.short_episodes}局，按新验证种子选出 `{selected}`；随后原奖励和候选均连续续训到{args.long_episodes}局。',
          '所有组从同一份早期actor/critic/norm及重置Adam起步；较长阶段保留各自优化器状态。训练分母固定5000，固定蒸馏/早期历史/搜索对手45/45/10%。',
          '两个新采样种子共享同一预训练种子；本结果不等于独立从头训练的多seed正式结论。','']
    for title,key in [('短训练验证','short'),('较长训练独立确认','long'),('完整预算搜索确认','search')]:
        text += [f'## {title}','','|组|种子|角色|方法|成功|自身非法|只抓一个超时|原环境回报|','|---|---:|---:|---|---:|---:|---:|---:|']
        for r in result[key]:
            text.append(f'|{r["label"]}|{r["seed"]}|{r["role"]}|{r["method"]}|{r["success"]:g}/{r["n"]}|{r["own_illegal"]:g}|{r["one_capture_timeout"]}|{r["environment_return"]:.3f}|')
        text.append('')
    text += [f'角色1为追捕、角色0为逃逸；全部成功要求合法结束。搜索每次决策{args.search_budget}次模拟、深度{args.search_depth}、beta_value=0，搜索奖励对所有组保持原版。',
             '独立确认结果不参与重新选奖励；短训练选择已在确认前冻结。每角色两个采样种子复用相同场景，不能当作两倍数量的独立地图。',
             'continue_formal=false；本实验不会自动开启5万局训练。','',
             '机器可读完整结果见 results.json，选择记录见 selection.json，模型哈希和源码指纹见 experiment.json。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,key,title in zip(axes,('long','search'),('Longer PPO: pursuer success',f'MCTS {args.search_budget}/depth {args.search_depth}: pursuer success')):
        names=['initial','legacy',selected]
        for s in args.seeds:
            yy=[next(r['success']/r['n'] for r in result[key] if r['label']==name and r['seed']==(None if name=='initial' else s) and r['role']==1) for name in names]
            ax.plot(range(3),yy,'o-',label=f'sampling seed {s}')
        ax.set_xticks(range(3),names,rotation=15);ax.set_ylim(0,1.05);ax.grid(alpha=.2);ax.set_title(title);ax.legend()
    fig.tight_layout();fig.savefig(out/'reward_results.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=trainer.ROOT/'current/search_rl_v3_overlap/pilot_seed0')
    p.add_argument('--output',type=Path,default=trainer.ROOT/'current/search_rl_reward_20261022')
    p.add_argument('--seeds',type=int,nargs='+',default=[41,73]);p.add_argument('--workers',type=int,default=32)
    p.add_argument('--batch',type=int,default=64);p.add_argument('--short-episodes',type=int,default=128)
    p.add_argument('--long-episodes',type=int,default=1024);p.add_argument('--start-index',type=int,default=1024)
    p.add_argument('--horizon',type=int,default=5000);p.add_argument('--opponent-budget',type=int,default=200)
    p.add_argument('--validation-per-layout',type=int,default=20)
    p.add_argument('--confirmation-per-layout',type=int,default=40)
    p.add_argument('--search-per-layout',type=int,default=4)
    p.add_argument('--search-budget',type=int,default=1000);p.add_argument('--search-depth',type=int,default=8)
    args=p.parse_args()
    if (min(args.batch,args.short_episodes,args.long_episodes,args.workers,args.validation_per_layout,args.confirmation_per_layout,args.search_per_layout)<=0
        or args.short_episodes%args.batch or args.long_episodes%args.batch or args.long_episodes<args.short_episodes
        or args.start_index+args.long_episodes>args.horizon or len(set(args.seeds))!=len(args.seeds)):
        p.error('Invalid size, seed list, or fixed curriculum window')
    args.source=args.source.resolve();args.output=args.output.resolve();out=args.output;out.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    signature=dict(config=config,source={name:digest(args.source/name) for name in ('best_role0.pt','best_role1.pt','distilled.pt','ppo_final.pt')},
                   script_sha256=digest(__file__),helpers_sha256=digest(Path(__file__).with_name('run_search_rl_ablation.py')),
                   base_implementation=trainer.implementation_hash())
    if (out/'experiment.json').exists() and json.loads((out/'experiment.json').read_text())!=signature:
        raise ValueError('Different experiment/config/source in output; use a new directory')
    trainer.write_json(out/'experiment.json',signature)
    torch.set_num_threads(1);random.seed(20261022);np.random.seed(20261022);torch.manual_seed(20261022)
    initial=compose_initial(args.source);anchor=PolicyBundle.load(args.source/'distilled.pt')
    if not(out/'initial.pt').exists():initial.save(out/'initial.pt')
    if not(out/'distilled.pt').exists():anchor.save(out/'distilled.pt')
    started=time.perf_counter()
    def status(phase,**kw):
        trainer.write_json(out/'status.json',dict(phase=phase,updated_unix=time.time(),**kw));print('STATUS',phase,kw,flush=True)
    with mp.get_context('spawn').Pool(args.workers,initializer=trainer.worker_init) if args.workers>1 else trainer._NoPool() as pool:
        status('short_reward_training')
        train_to(args,out,pool,ARMS,args.short_episodes,initial,anchor)
        status('short_reward_validation')
        short=evaluate(args,out,pool,checkpoints(out,ARMS,args.seeds,args.short_episodes),
                       'short_validation',600,args.validation_per_layout)
        ranking=select_reward(short,args.seeds);selected=ranking[0]['arm']
        selection=dict(selected=selected,ranking=ranking,chosen_before_long_confirmation=True)
        if (out/'selection.json').exists() and json.loads((out/'selection.json').read_text())!=selection:raise ValueError('Selection changed')
        trainer.write_json(out/'selection.json',selection)
        status('longer_paired_training',selected=selected)
        arms=('legacy',selected)
        train_to(args,out,pool,arms,args.long_episodes,initial,anchor)
        status('longer_policy_confirmation')
        long=evaluate(args,out,pool,checkpoints(out,arms,args.seeds,args.long_episodes),
                      'long_confirmation',610,args.confirmation_per_layout,roles=(0,1))
        status('full_budget_search_confirmation',budget=args.search_budget,depth=args.search_depth,selected=selected)
        search=evaluate(args,out,pool,checkpoints(out,arms,args.seeds,args.long_episodes),
                        'search_confirmation',620,args.search_per_layout,methods=('search_ppo',),roles=(0,1))
    report(args,out,selected,short,long,search)
    assert all(digest(args.source/name)==value for name,value in signature['source'].items())
    status('complete',selected=selected,elapsed_seconds=time.perf_counter()-started,continue_formal=False)


if __name__=='__main__':main()
