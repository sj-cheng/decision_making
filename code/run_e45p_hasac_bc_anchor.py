"""E45-P: bounded same-actor BC diagnosis, then gated replay +/- BC RL."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import ctypes
import fcntl
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import run_e45o_hasac_demonstration as o
from learning.e45m_author_hasac import AuthorHASAC
from learning.e45m_task_adapter import Replay
from learning.e45o_demonstration_replay import DemonstrationReplay
from learning.e45p_actor_supervision import public_teacher,initialize_bc_std,bc_step,install_anchor,fit_metrics
j=o.j
ROOT=o.ROOT
OUT=ROOT/'current/e45p_hasac_bc_anchor_20261101'


def register(root,smoke=False):
    root=Path(root).resolve()
    if (root/'experiment.json').exists():return verify(root)
    root.mkdir(parents=True,exist_ok=True)
    parent=o.verify(o.OUT)
    cfg=dict(parent)
    cfg.update(experiment='E45-P',registered_at=time.strftime('%FT%T%z'),smoke=smoke,
        entries=parent['entries'][:1] if smoke else parent['entries'],arms=['replay','replay_bc'],
        scope='same HASAC actor BC closed-loop diagnosis then conditional continued-supervision comparison',
        bc_updates=2 if smoke else 3000,bc_evaluation_every=1000,bc_initial_std=.15,
        bc_loss='mean latent MSE, atanh teacher action clipped 1e-4, active mask, full frozen teacher set',
        bc_actor_lr=5e-4,bc_selection='fixed terminal checkpoint; no best-checkpoint rescue',
        bc_gate=5,gate_scope='each original seed terminal deterministic >=5/6; both seeds required',
        smoke_bypasses_gate=bool(smoke),steps_per_model=1 if smoke else 20000,
        evaluation_every=5000,batch_episodes=2 if smoke else 12,
        replay_capacity=256 if smoke else 50000,update_batch=32 if smoke else 256,
        action_repeats=1 if smoke else 2,warmup_steps=0,updates_per_transition=1,
        anchor_weight=.1,anchor_schedule='constant, no annealing during this 20k diagnostic',
        anchor_application='full teacher public states, deterministic latent-mean auxiliary gradient added before same author Adam step',
        demonstration_file=str(o.OUT/'demonstrations/data.npz'),
        teacher_manifest=str(o.OUT/'demonstrations/manifest.json'),
        rl_initialization='same per-seed terminal BC actors; fresh matching actor optimizers and random author critic/targets',
        rl_sampling='current stochastic actor only, initial std .15, no random uniform warmup',
        selection='all fixed checkpoints and final; BC_ONLY terminal actor frozen',
        budget='3000 BC actor updates/seed plus historical 163 demo transitions; RL20k online/arm/seed plus same demos, episode batch overshoot disclosed',
        maximum_wall_seconds=12*3600)
    protected=dict(parent['protected_hashes']);protected.update(parent['source_hashes'])
    cfg['protected_hashes']=protected
    files=[Path(__file__).resolve(),ROOT/'code/learning/e45p_actor_supervision.py',
        ROOT/'code/tests/test_e45p_actor_supervision.py',ROOT/'code/E45P_HASAC_BC_ANCHOR_PROTOCOL_20261101.md',
        ROOT/'code/audit_e45p_smoke.py']
    cfg['source_hashes']={str(p):j.sha(p) for p in files}
    cfg['bank_hashes']={e['path']:j.sha(e['path']) for e in cfg['entries']}
    cfg['parent_O_experiment_sha256']=j.sha(o.OUT/'experiment.json')
    cfg['teacher_manifest_sha256']=j.sha(cfg['teacher_manifest'])
    j.save(root/'experiment.json',cfg)
    for p in files:
        dst=root/'source_snapshot'/p.relative_to(ROOT/'code');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    return verify(root)


def verify(root):
    root=Path(root);cfg=j.read(root/'experiment.json')
    for section in ('source_hashes','protected_hashes','upstream_hashes','bank_hashes'):
        for path,digest in cfg[section].items():
            if j.sha(path)!=digest:raise RuntimeError('Frozen artifact changed: '+path)
    assert j.sha(cfg['demonstration_file'])==cfg['demonstration_sha256']
    assert j.sha(cfg['teacher_manifest'])==cfg['teacher_manifest_sha256']
    return cfg


def load_data(cfg):
    with np.load(cfg['demonstration_file']) as z:return {key:z[key].copy() for key in z.files}


def gate_passed(history,cfg):
    row=history[-1]['deterministic']
    return row['episodes']==6 and row['double']>=cfg['bc_gate']


def setup(args,folder):
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cores=list(map(int,args.cpus.split(',')));os.sched_setaffinity(0,{cores[-1]});torch.set_num_threads(1)
    torch.cuda.set_device(args.device);o.configure(folder)
    model=AuthorHASAC(obs_dim=845,device=args.device,seed=args.seed,retirement_mask=True)
    pool,q=o.pool_for(folder,cores[:-1])
    return lock,model,pool,q


def bc_train(args):
    root=Path(args.output);cfg=verify(root);folder=root/'bc'/f'HASAC_{args.seed}'
    lock,model,pool,q=setup(args,folder)
    initialize_bc_std(model,cfg['bc_initial_std']);teacher=public_teacher(load_data(cfg),model.device)
    replay=Replay(cfg['replay_capacity'],args.seed);cfg_sha=j.sha(root/'experiment.json')
    updates=iteration=0;history=[];started=time.time()
    try:
        if (folder/'latest.json').exists():updates,iteration,history=o.restore(folder,model,replay,cfg_sha)
        if not history:
            j.common.emit(folder,'bc_initial_evaluation',steps=0)
            row=o.evaluate(pool,cfg,args.seed,model,0);row['fit']=fit_metrics(model,teacher)
            history.append(row);j.save(folder/'evaluation.json',history)
            o.checkpoint(folder,model,replay,0,0,history,cfg_sha)
        while updates<cfg['bc_updates']:
            bc_step(model,teacher);updates+=1
            if updates%100==0:
                j.common.emit(folder,'bc_fitting',steps=updates,fit=fit_metrics(model,teacher))
            if updates%cfg['bc_evaluation_every']==0 or updates==cfg['bc_updates']:
                j.common.emit(folder,'bc_evaluation',steps=updates)
                row=o.evaluate(pool,cfg,args.seed,model,updates);row['fit']=fit_metrics(model,teacher)
                history.append(row);j.save(folder/'evaluation.json',history)
                o.checkpoint(folder,model,replay,updates,updates,history,cfg_sha)
        passed=gate_passed(history,cfg)
        verify(root)
        j.save(folder/'decision.json',dict(status='complete',bc_updates=updates,history=history,
            gate_passed=passed,smoke_gate_bypass=cfg['smoke'],wall_seconds=time.time()-started,
            bc_only_checkpoint=j.read(folder/'latest.json'),actor_updates_per_seed=updates*2,
            closed_loop_scope='training tasks only; no generalization claim'))
        j.common.emit(folder,'bc_complete' if passed else 'bc_gate_failed',steps=updates,gate_passed=passed)
    except BaseException as exc:
        j.common.emit(folder,'failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True);q.close();lock.close()


def rl_train(args):
    root=Path(args.output);cfg=verify(root);folder=root/'runs'/f'{args.arm}_{args.seed}'
    bc=root/'bc'/f'HASAC_{args.seed}';bc_decision=j.read(bc/'decision.json')
    assert bc_decision['gate_passed'] or cfg['smoke']
    # No successful-seed substitution and no partial-seed second stage.
    assert cfg['smoke'] or all(j.read(root/'bc'/f'HASAC_{s}'/'decision.json')['gate_passed'] for s in cfg['seeds'])
    lock,model,pool,q=setup(args,folder);data=load_data(cfg);teacher=public_teacher(data,model.device)
    replay=DemonstrationReplay(data,cfg['replay_capacity'],args.seed,cfg['demonstration_fraction'])
    cfg_sha=j.sha(root/'experiment.json');steps=iteration=0;history=[];started=time.time()
    handles=[]
    try:
        if (folder/'latest.json').exists():steps,iteration,history=o.restore(folder,model,replay,cfg_sha)
        else:
            pointer=bc_decision['bc_only_checkpoint'];assert j.sha(pointer['path'])==pointer['sha256']
            saved=torch.load(pointer['path'],map_location=model.device,weights_only=False)
            for actor,state in zip(model.actor,saved['actors']):actor.actor.load_state_dict(state)
        weight=cfg['anchor_weight'] if args.arm=='replay_bc' else 0.
        handles,anchor=install_anchor(model,teacher,weight);anchor['calls']=2*model.runner.total_it if weight else 0
        if not history:
            history.append(o.evaluate(pool,cfg,args.seed,model,0));j.save(folder/'evaluation.json',history)
            o.checkpoint(folder,model,replay,0,0,history,cfg_sha)
        situations=o.l.situations(cfg['entries'])
        while steps<cfg['steps_per_model']:
            begin=time.time();weights=o.actor_weights(model)
            order=np.tile(np.arange(len(situations)),int(np.ceil(cfg['batch_episodes']/len(situations))))[:cfg['batch_episodes']]
            np.random.default_rng(o.l.seed_for('HASAC_order',args.seed,iteration)).shuffle(order)
            jobs=[]
            for i,index in enumerate(order):
                e,w,env=situations[index]
                task=o.task_for(e,w,env,args.seed,'stochastic',o.l.seed_for('HASAC_actions',args.seed,iteration,i),
                    f'train_iter{iteration}_i{i}',uniform=False)
                jobs.append(dict(actors=weights,task=task,training=True))
            results=list(pool.map(o.rollout,jobs));before=steps
            for result in results:replay.add(result['arrays']);steps+=len(result['arrays']['obs'])
            target_updates=cfg['smoke_updates'] if cfg['smoke'] else steps
            while model.runner.total_it<target_updates:model.update(replay.sample(cfg['update_batch']))
            iteration+=1;rows=[r['summary'] for r in results]
            record=dict(steps=steps,demonstration_steps=cfg['demonstration_steps'],
                total_accounted_steps=steps+cfg['demonstration_steps'],bc_updates=cfg['bc_updates'],
                iteration=iteration,arm=args.arm,gradient_updates=model.runner.total_it,
                replay_size=replay.size,anchor=dict(anchor),fit=fit_metrics(model,teacher),
                episodes=rows,score=o.l.score(rows),seconds=time.time()-begin)
            j.save(folder/'updates'/f'{iteration:05d}.json',record)
            if before//cfg['evaluation_every']!=steps//cfg['evaluation_every'] or steps>=cfg['steps_per_model']:
                history.append(o.evaluate(pool,cfg,args.seed,model,steps));j.save(folder/'evaluation.json',history)
                o.checkpoint(folder,model,replay,steps,iteration,history,cfg_sha)
            j.common.emit(folder,'training',steps=steps,gradient_updates=model.runner.total_it,score=record['score'],anchor=anchor)
        verify(root);j.save(folder/'decision.json',dict(status='complete',steps=steps,history=history,
            gradient_updates=model.runner.total_it,anchor=anchor,bc_updates=cfg['bc_updates'],
            demonstration_steps=cfg['demonstration_steps'],total_accounted_steps=steps+cfg['demonstration_steps'],
            initialization=bc_decision['bc_only_checkpoint'],wall_seconds=time.time()-started))
        j.common.emit(folder,'complete',steps=steps,gradient_updates=model.runner.total_it)
    except BaseException as exc:
        j.common.emit(folder,'failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        for handle in handles:handle.remove()
        pool.shutdown(wait=True,cancel_futures=True);q.close();lock.close()


def report(root):
    root=Path(root);cfg=j.read(root/'experiment.json');state=j.read(root/'status.json') if (root/'status.json').exists() else {}
    lines=['E45-P：同架构BC闭环 -> 回放 / 回放+持续BC',time.strftime('%FT%T%z'),
        '状态：'+state.get('stage','registered'),
        '先固定终点BC门槛，两个种子均>=5/6才启动正式RL；评估全部关闭教师。','',
        '阶段/种子 | 状态 | 更新/在线步 | 最近确定性双捕 | 最近随机双捕']
    for phase in ['bc']+cfg['arms']:
        for seed in cfg['seeds']:
            folder=root/'bc'/f'HASAC_{seed}' if phase=='bc' else root/'runs'/f'{phase}_{seed}'
            s=j.read(folder/'status.json') if (folder/'status.json').exists() else {}
            h=j.read(folder/'evaluation.json') if (folder/'evaluation.json').exists() else []
            tail='尚无评估'
            if h:
                a=h[-1]['deterministic'];b=h[-1]['stochastic']
                tail=f"{a['double']}/{a['episodes']} | {b['double']}/{b['episodes']}"
            lines.append(f"{phase}/{seed} | {s.get('stage','queued')} | {s.get('steps',0)} | {tail}")
    tmp=root/'LIVE.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(root/'LIVE.txt')


def final_report(root,cfg,bc,rl_completed=False):
    lines=['# E45-P：BC闭环与持续监督对照', '',
        '固定训练情形的开发诊断，不是独立泛化测试。BC_ONLY检查点冻结保留。', '',
        '|种子|BC更新/actor|终点潜变量MSE（两actor）|终点动作MSE|确定性双捕|随机双捕|终点门槛|',
        '|---|---:|---|---|---|---|---|']
    for seed,row in bc.items():
        h=row['history'][-1];a=h['deterministic'];b=h['stochastic']
        lines.append(f"|{seed}|{row['bc_updates']}|{h['fit']['latent_mse']}|{h['fit']['action_mse']}|{a['double']}/{a['episodes']}|{b['double']}/{b['episodes']}|{row['gate_passed']}|")
    lines+=['', 'BC使用同一845维公共输入、256x256双actor、163条教师转移；固定3000次全批更新/actor，沿用5e-4 Adam，std初始化固定0.15。',
        '训练内门槛为每个种子终点确定性>=5/6，全部原种子保留，不选中途最好模型。']
    if rl_completed:
        lines+=['', '|方法|种子|在线步|更新数|终点确定性双捕|终点随机双捕|', '|---|---|---:|---:|---|---|']
        for arm in cfg['arms']:
            for seed in cfg['seeds']:
                d=j.read(root/'runs'/f'{arm}_{seed}'/'decision.json');h=d['history'][-1];a=h['deterministic'];b=h['stochastic']
                lines.append(f"|{arm}|{seed}|{d['steps']}|{d['gradient_updates']}|{a['double']}/{a['episodes']}|{b['double']}/{b['episodes']}|")
        lines+=['', '两RL臂同BC初始化、同新建优化器/critic、同25%示范回放，持续监督臂lambda=0.1不退火。',
            '所有固定检查点见各运行evaluation.json，不能用单个最好检查点代替稳定性结论。']
    else:
        lines+=['', 'BC闭环门槛未跨两个种子通过，第二步未启动，属于能力门槛失败，不是软件失败。',
            '若监督拟合好而闭环弱，优先检查偏离教师轨迹后的状态覆盖/误差累积，不能据此直接认定网络容量不足。']
    if cfg['smoke']:lines+=['','本目录是工程smoke，显式绕过学习门槛，不能用于能力结论。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run_children(root,cfg,args,phase,started):
    combinations=[('bc',s) for s in cfg['seeds']] if phase=='bc' else [(a,s) for a in cfg['arms'] for s in cfg['seeds']]
    cores=j.choose_cpus(len(combinations)*(args.workers+1));assert len(cores)==len(combinations)*(args.workers+1)
    children=[];allocation=[]
    try:
        for i,(arm,seed) in enumerate(combinations):
            folder=root/'bc'/f'HASAC_{seed}' if phase=='bc' else root/'runs'/f'{arm}_{seed}'
            folder.mkdir(parents=True,exist_ok=True)
            if (folder/'decision.json').exists():continue
            use=cores[i*(args.workers+1):(i+1)*(args.workers+1)]
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'bc' if phase=='bc' else 'rl',
                '--output',str(root),'--seed',str(seed),'--arm',arm,'--device',args.device,
                '--cpus',','.join(map(str,use)),'--parent',str(os.getpid())]
            with (folder/'run.log').open('a') as log:child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            children.append(child);allocation.append(dict(arm=arm,seed=seed,pid=child.pid,cores=use,device=args.device))
        j.save(root/f'execution_{phase}.json',dict(allocation=allocation,workers_per_model=args.workers))
        while any(child.poll() is None for child in children):
            report(root)
            if any(child.poll() not in (None,0) for child in children):raise RuntimeError('An E45-P child failed; inspect retained logs')
            if time.time()-started>cfg['maximum_wall_seconds']:raise RuntimeError('Registered wall cap reached')
            time.sleep(5)
        assert all(child.returncode==0 for child in children)
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()


def pipeline(args):
    root=Path(args.output).resolve();cfg=register(root,args.smoke)
    lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time()
    try:
        if not cfg['smoke']:
            smoke=ROOT/'current/e45p_hasac_bc_anchor_smoke_20261101'
            verify(smoke);assert j.read(smoke/'smoke_audit.json')['passed']
        j.common.emit(root,'bc_running');run_children(root,cfg,args,'bc',started)
        bc={str(s):j.read(root/'bc'/f'HASAC_{s}'/'decision.json') for s in cfg['seeds']}
        passed=all(d['gate_passed'] for d in bc.values());run_rl=passed or cfg['smoke']
        j.save(root/'bc_gate.json',dict(passed=passed,smoke_bypass=cfg['smoke'],seeds=bc))
        if run_rl:
            j.common.emit(root,'rl_running');run_children(root,cfg,args,'rl',started)
        verify(root);final_report(root,cfg,bc,run_rl)
        status='complete' if run_rl else 'bc_gate_failed'
        j.save(root/'decision.json',dict(status=status,bc_gate_passed=passed,rl_started=run_rl,
            smoke_bypass=cfg['smoke'],wall_seconds=time.time()-started))
        j.common.emit(root,status);report(root)
    except BaseException as exc:
        j.common.emit(root,'failed',error=repr(exc),traceback=traceback.format_exc());report(root);raise
    finally:lock.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['register','pipeline','bc','rl','report'])
    parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--workers',type=int,default=12);parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--seed',type=int,default=17);parser.add_argument('--arm',default='replay_bc')
    parser.add_argument('--cpus');parser.add_argument('--parent',type=int);args=parser.parse_args()
    if args.parent:
        ctypes.CDLL(None).prctl(1,signal.SIGTERM)
        if os.getppid()!=args.parent:raise RuntimeError('Coordinator exited')
    if args.command=='register':register(args.output,args.smoke)
    elif args.command=='pipeline':pipeline(args)
    elif args.command=='bc':bc_train(args)
    elif args.command=='rl':rl_train(args)
    else:report(args.output)


if __name__=='__main__':main()
