"""E45-T: one-factor critic-lr continuation on unchanged N static tasks."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import argparse
import ctypes
import fcntl
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import run_e45n_hasac_curriculum as n
from learning.e45p_execution_precision import environment_action

m=n.m;j=n.j;ROOT=n.ROOT
OUT=ROOT/'current/e45t_critic_lr_stability_20261102'
SMOKE=ROOT/'current/e45t_critic_lr_stability_smoke_20261102'
ARMS=('original','slow_critic')


def register(root,smoke=False):
    root=Path(root).resolve()
    if (root/'experiment.json').exists():return verify(root)
    root.mkdir(parents=True,exist_ok=True);parent=n.verify(n.OUT)
    audit=ROOT/'current/e45s_n29_collapse_20261102'
    assert j.read(audit/'summary.json')['passed']
    cfg=dict(parent)
    cfg.update(experiment='E45-T',registered_at=time.strftime('%FT%T%z'),smoke=smoke,
        arms=list(ARMS),phase=parent['phases'][0],additional_online_steps=1 if smoke else 5000,
        evaluation_every=1000,batch_episodes=12,update_batch=256,smoke_new_updates=2,
        critic_lr_by_arm=dict(original=5e-4,slow_critic=1e-4),actor_lr=5e-4,
        warmup_steps=0,updates_per_transition=1,parent_experiment_sha256=j.sha(n.OUT/'experiment.json'),
        hypothesis='N29 large on-policy soft-Q overestimation motivates testing lower critic step size; causal claim not assumed',
        intervention='ONLY critic optimizer lr and matching critic_lr metadata; all actor/reward/alpha/gamma/replay/UTD/architecture unchanged',
        sources={},source_selection='per-seed FIRST stored static checkpoint >=10k, not best selected checkpoint',
        task_scope='same six N INITIAL-visible stationary-target cases with real sensor; not fully observable or new generalization test',
        audit_summary=str(audit/'summary.json'),audit_summary_sha256=j.sha(audit/'summary.json'),
        evaluation='static deterministic6 + stochastic12 at0/1k/2k/3k/4k/5k full-batch crossings; no hidden probes or teacher',
        selection='all checkpoints and both seeds retained, no best-checkpoint substitution or automatic budget extension',
        budget='5000 NEW transitions/run x4; full12-episode overshoot disclosed; parent cost and smoke separately accounted',
        maximum_wall_seconds=4*3600)
    for seed in cfg['seeds']:
        folder=n.OUT/'runs'/f'HASAC_Curriculum_{seed}'
        candidates=sorted((folder/'phase_snapshots/phase_0/checkpoints').glob('steps_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
        path=next(p for p in candidates if int(p.stem.split('_')[-1])>=10000)
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['experiment_sha256']==cfg['parent_experiment_sha256']
        assert j.sha(state['replay_path'])==state['replay_sha256']
        source=dict(path=str(path),sha256=j.sha(path),replay_path=state['replay_path'],
            replay_sha256=state['replay_sha256'],steps=state['steps'],iteration=state['iteration'],
            gradient_updates=state['gradient_updates'])
        cfg['sources'][str(seed)]=source
        ref=root/'parents'/f'seed_{seed}';j.save(ref/'latest.json',source)
        del state
    protected=dict(parent['protected_hashes']);protected.update(parent['source_hashes'])
    path=ROOT/'code/learning/e45p_execution_precision.py';protected[str(path)]=j.sha(path)
    cfg['protected_hashes']=protected
    files=[Path(__file__).resolve(),ROOT/'code/audit_e45t_smoke.py',ROOT/'code/E45T_CRITIC_LR_PROTOCOL_20261102.md']
    cfg['source_hashes']={str(p):j.sha(p) for p in files}
    j.save(root/'experiment.json',cfg)
    for path in files:
        dst=root/'source_snapshot'/path.name;dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(path,dst)
    return verify(root)


def verify(root):
    cfg=j.read(Path(root)/'experiment.json')
    for section in ('source_hashes','protected_hashes','upstream_hashes','bank_hashes'):
        for path,value in cfg[section].items():
            if j.sha(path)!=value:raise RuntimeError('Frozen file changed: '+path)
    assert j.sha(n.OUT/'experiment.json')==cfg['parent_experiment_sha256']
    assert j.sha(cfg['audit_summary'])==cfg['audit_summary_sha256']
    for source in cfg['sources'].values():
        assert j.sha(source['path'])==source['sha256']
        assert j.sha(source['replay_path'])==source['replay_sha256']
    return cfg


def factory(case,task):
    own=m.factory(case,task);original=own.act
    def act(obs,fixed_simulations=None):
        action=original(obs,fixed_simulations);actual,error=environment_action(action,case['speed'])
        own.last_stats['environment_speed_roundoff_correction']=error
        return actual
    own.act=act;return own


def initialize_worker(folder,queue,parent):
    m.init_worker(folder,queue,parent);j.common.base.make_controller=factory


def pool_for(folder,cores):
    context=m.mp.get_context('spawn');queue=context.Queue()
    for cpu in cores:queue.put(cpu)
    return ProcessPoolExecutor(max_workers=len(cores),mp_context=context,initializer=initialize_worker,
        initargs=(str(folder),queue,os.getpid())),queue


def evaluate(pool,cfg,seed,model,steps):
    result=dict(steps=steps);weights=m.actor_weights(model)
    for mode in ('deterministic','stochastic'):
        repeat=1 if mode=='deterministic' else 2
        jobs=[dict(actors=weights,training=False,task=n.task_for(e,w,env,seed,cfg['phase'],mode,
            m.l.seed_for('N_eval',seed,e['id'],w,r),f'T_eval_{steps}'))
            for e,w,env in m.l.situations(cfg['entries']) for r in range(repeat)]
        result[mode]=m.l.score([v['summary'] for v in pool.map(m.rollout,jobs)])
    return result


def train(args):
    root=Path(args.output);cfg=verify(root);source=cfg['sources'][str(args.seed)]
    folder=root/'runs'/f'{args.arm}_{args.seed}';folder.mkdir(exist_ok=True,parents=True)
    lock=(folder/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cores=list(map(int,args.cpus.split(',')));os.sched_setaffinity(0,{cores[-1]});torch.set_num_threads(1)
    torch.cuda.set_device(args.device);m.configure(folder)
    model=m.AuthorHASAC(obs_dim=845,device=args.device,seed=args.seed,retirement_mask=True)
    replay=m.Replay(cfg['replay_capacity'],args.seed);cfg_sha=j.sha(root/'experiment.json')
    history=[];steps=0;pool=None;queue=None;started=time.time()
    try:
        if (folder/'latest.json').exists():steps,iteration,history=m.restore(folder,model,replay,cfg_sha)
        else:
            old_steps,iteration,_=m.restore(root/'parents'/f'seed_{args.seed}',model,replay,cfg['parent_experiment_sha256'])
            assert old_steps==source['steps'] and model.runner.total_it==source['gradient_updates']
        critic_lr=cfg['critic_lr_by_arm'][args.arm]
        for group in model.critic.critic_optimizer.param_groups:group['lr']=critic_lr
        model.critic.critic_lr=critic_lr
        assert all(a.actor_optimizer.param_groups[0]['lr']==cfg['actor_lr'] for a in model.actor)
        pool,queue=pool_for(folder,cores[:-1])
        if not history:
            j.common.emit(folder,'initial_evaluation',steps=0,critic_lr=critic_lr)
            history.append(evaluate(pool,cfg,args.seed,model,0));j.save(folder/'evaluation.json',history)
            m.checkpoint(folder,model,replay,0,iteration,history,cfg_sha)
        situations=m.l.situations(cfg['entries'])
        while steps<cfg['additional_online_steps']:
            weights=m.actor_weights(model);before=steps;begin=time.time()
            order=np.tile(np.arange(len(situations)),2)
            np.random.default_rng(m.l.seed_for('N_order',args.seed,iteration)).shuffle(order)
            jobs=[]
            for i,index in enumerate(order):
                e,w,env=situations[index]
                task=n.task_for(e,w,env,args.seed,cfg['phase'],'stochastic',m.l.seed_for('N_actions',args.seed,iteration,i),
                    f'T_train_iter{iteration}_i{i}',uniform=False)
                jobs.append(dict(actors=weights,task=task,training=True))
            results=list(pool.map(m.rollout,jobs))
            for result in results:replay.add(result['arrays']);steps+=len(result['arrays']['obs'])
            target=source['gradient_updates']+(cfg['smoke_new_updates'] if cfg['smoke'] else steps)
            while model.runner.total_it<target:model.update(replay.sample(cfg['update_batch']))
            assert model.runner.total_it==target
            iteration+=1;rows=[x['summary'] for x in results]
            record=dict(steps=steps,iteration=iteration,parent_steps=source['steps'],
                total_steps=source['steps']+steps,new_gradient_updates=target-source['gradient_updates'],
                gradient_updates=target,critic_lr=critic_lr,actor_lr=cfg['actor_lr'],replay_size=replay.size,
                episodes=rows,score=m.l.score(rows),seconds=time.time()-begin)
            j.save(folder/'updates'/f'{iteration:05d}.json',record)
            if before//cfg['evaluation_every']!=steps//cfg['evaluation_every'] or steps>=cfg['additional_online_steps']:
                j.common.emit(folder,'evaluation',steps=steps,new_gradient_updates=target-source['gradient_updates'])
                history.append(evaluate(pool,cfg,args.seed,model,steps));j.save(folder/'evaluation.json',history)
                m.checkpoint(folder,model,replay,steps,iteration,history,cfg_sha)
            j.common.emit(folder,'training',steps=steps,new_gradient_updates=target-source['gradient_updates'],score=record['score'])
        verify(root);j.save(folder/'decision.json',dict(status='complete',steps=steps,history=history,
            source=source,new_gradient_updates=model.runner.total_it-source['gradient_updates'],
            critic_lr=critic_lr,wall_seconds=time.time()-started))
        j.common.emit(folder,'complete',steps=steps,new_gradient_updates=model.runner.total_it-source['gradient_updates'])
    except BaseException as exc:
        j.common.emit(folder,'failed',steps=steps,error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        if queue is not None:queue.close()
        lock.close()


def report(root):
    root=Path(root);cfg=j.read(root/'experiment.json')
    status=j.read(root/'status.json') if (root/'status.json').exists() else {}
    lines=['E45-T：静态任务稳定性，唯一因素 Critic lr 5e-4 / 1e-4',time.strftime('%FT%T%z'),
        '状态：'+status.get('stage','registered'),
        '从各自原N首个10k检查点完整恢复；Actor lr/奖励/温度/回放均不变。','',
        '方法/种子 | 状态 | 新步数 | 新更新 | 初始确定性 | 最新确定性 | 最新随机']
    md=['# E45-T：Critic学习率单因素短对照','',
        '固定原N初始可见、目标静止的六个训练情形；不是原隐藏Root主任务成功或新泛化测试。','',
        '|组别|种子|新增步|新增更新|确定性双捕走势（每次6局）|最新随机双捕|状态|',
        '|---|---:|---:|---:|---|---|---|']
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            folder=root/'runs'/f'{arm}_{seed}'
            s=j.read(folder/'status.json') if (folder/'status.json').exists() else {}
            history=j.read(folder/'evaluation.json') if (folder/'evaluation.json').exists() else []
            initial=det=sto=curve='—'
            if history:
                d=history[-1]['deterministic'];v=history[-1]['stochastic'];d0=history[0]['deterministic']
                initial=f"{d0['double']}/{d0['episodes']}";det=f"{d['double']}/{d['episodes']}";sto=f"{v['double']}/{v['episodes']}"
                curve=' → '.join(str(x['deterministic']['double']) for x in history)
            stage=s.get('stage','queued');steps=s.get('steps',0);updates=s.get('new_gradient_updates',0)
            lines.append(f'{arm}/{seed} | {stage} | {steps} | {updates} | {initial} | {det} | {sto}')
            md.append(f'|{arm}|{seed}|{steps}|{updates}|{curve}|{sto}|{stage}|')
    lines+=['','正式每路5k新步，每1k评估；整批超出保留。诊断，不按得分追加预算。',str(root)]
    md+=['','四路完整保留父Actor/Critic/目标网络/Adam/Replay/RNG，只改slow_critic的Critic学习率。',
        '完整曲线与实际预算见各运行evaluation.json、decision.json；所有评估为纯Actor，无教师。']
    for name,content in [('LIVE.txt',lines),('REPORT.md',md)]:
        temp=root/(name+'.tmp');temp.write_text('\n'.join(content)+'\n');temp.replace(root/name)


def pipeline(args):
    root=Path(args.output).resolve();cfg=register(root,args.smoke)
    lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();children=[]
    try:
        if not cfg['smoke']:
            verify(SMOKE);audit=j.read(SMOKE/'smoke_audit.json')
            assert audit['passed'] and audit['runner_sha256']==j.sha(Path(__file__).resolve())
        combos=[(arm,seed) for arm in cfg['arms'] for seed in cfg['seeds']]
        cores=j.choose_cpus(len(combos)*(args.workers+1));assert len(cores)==len(combos)*(args.workers+1)
        allocation=[]
        for index,(arm,seed) in enumerate(combos):
            folder=root/'runs'/f'{arm}_{seed}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'decision.json').exists():continue
            used=cores[index*(args.workers+1):(index+1)*(args.workers+1)]
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'train','--output',str(root),
                '--seed',str(seed),'--arm',arm,'--device',args.device,'--cpus',','.join(map(str,used)),
                '--parent',str(os.getpid())]
            with (folder/'run.log').open('a') as log:child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            children.append(child);allocation.append(dict(arm=arm,seed=seed,pid=child.pid,cores=used,device=args.device))
        j.save(root/'execution.json',dict(allocation=allocation,workers=args.workers,started_unix=started))
        j.common.emit(root,'running',pids=[c.pid for c in children]);report(root)
        while any(c.poll() is None for c in children):
            report(root)
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('T child failed, inspect preserved logs')
            if time.time()-started>cfg['maximum_wall_seconds']:raise RuntimeError('T wall cap reached')
            time.sleep(5)
        assert all(c.returncode==0 for c in children);verify(root)
        j.save(root/'decision.json',dict(status='complete',wall_seconds=time.time()-started,
            runs={f'{a}_{s}':j.read(root/'runs'/f'{a}_{s}'/'decision.json') for a,s in combos}))
        j.common.emit(root,'complete');report(root)
    except BaseException as exc:
        j.common.emit(root,'failed',error=repr(exc),traceback=traceback.format_exc());report(root);raise
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        lock.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['register','pipeline','train','report'])
    parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--workers',type=int,default=12);parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--seed',type=int,default=29);parser.add_argument('--arm',choices=ARMS,default='slow_critic')
    parser.add_argument('--cpus');parser.add_argument('--parent',type=int);args=parser.parse_args()
    if args.parent:
        ctypes.CDLL(None).prctl(1,signal.SIGTERM)
        if os.getppid()!=args.parent:raise RuntimeError('Parent exited')
    if args.command=='register':register(args.output,args.smoke)
    elif args.command=='pipeline':pipeline(args)
    elif args.command=='train':train(args)
    else:report(args.output)


if __name__=='__main__':main()
