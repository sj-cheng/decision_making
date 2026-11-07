"""E45-R: paired continuation from full O states, replay +/- original-label BC."""
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
from learning.e45o_demonstration_replay import DemonstrationReplay
from learning.e45p_actor_supervision import public_teacher,install_anchor,fit_metrics
from learning.e45p_execution_precision import pool_for

j=o.j
ROOT=o.ROOT
OUT=ROOT/'current/e45r_hasac_continuation_20261102'
SMOKE=ROOT/'current/e45r_hasac_continuation_smoke_20261102'
ARMS=('replay','replay_bc')


def checked_data(path):
    with np.load(path,allow_pickle=False) as packet:
        data={key:packet[key].copy() for key in packet.files}
    assert len(data['obs'])==163
    assert np.isfinite(data['obs']).all() and np.isfinite(data['action']).all()
    assert np.max(np.abs(data['action']))<=1
    np.testing.assert_array_equal(data['obs'][:,:,-1:],data['active'])
    assert np.max(np.abs(data['action'][data['active'][...,0]==0]),initial=0)==0
    for actor in range(2):
        known={}
        for obs,action,active in zip(data['obs'][:,actor],data['action'][:,actor],data['active'][:,actor,0]):
            if not active:continue
            key=obs.tobytes()
            if key in known and np.max(np.abs(known[key]-action))>1e-5:
                raise RuntimeError('Conflicting active teacher labels; refuse single-action regression')
            known[key]=action
    return data


def register(root,smoke=False):
    root=Path(root).resolve()
    if (root/'experiment.json').exists():return verify(root)
    root.mkdir(parents=True,exist_ok=True)
    parent=o.verify(o.OUT)
    audit=ROOT/'current/e45q_teacher_label_audit_20261102'
    assert j.read(audit/'decision.json')['status']=='diagnosis_complete'
    assert j.read(audit/'summary.json')['inputs_unchanged']
    data=checked_data(o.OUT/'demonstrations/data.npz')
    cfg=dict(parent)
    cfg.update(experiment='E45-R',registered_at=time.strftime('%FT%T%z'),smoke=smoke,
        arms=list(ARMS),entries=parent['entries'][:1] if smoke else parent['entries'],
        additional_online_steps=1 if smoke else 20000,evaluation_every=5000,
        batch_episodes=2 if smoke else 12,update_batch=32 if smoke else 256,
        action_repeats=1 if smoke else 2,smoke_new_updates=2,
        warmup_steps=0,updates_per_transition=1,anchor_weight=.1,
        anchor_schedule='constant .1, no BC pretraining or eligibility gate',
        anchor_application='original 163 public labels only; full-batch latent mean MSE; active mask; same author Adam step',
        initialization='per-seed O guided FINAL full actor/critic/targets/optimizers/replay/RNG/iteration, never best checkpoint',
        supervision='immutable original O labels only; no Q correction data or new online teacher',
        teacher_definition='historical executed GEOM_SEARCH actions retained as a fixed dataset, not claimed globally deterministic',
        demonstration_file=str(o.OUT/'demonstrations/data.npz'),
        teacher_manifest=str(o.OUT/'demonstrations/manifest.json'),
        teacher_manifest_sha256=j.sha(o.OUT/'demonstrations/manifest.json'),
        teacher_label_audit=str(audit/'decision.json'),teacher_label_audit_sha256=j.sha(audit/'decision.json'),
        teacher_label_audit_summary=str(audit/'summary.json'),teacher_label_audit_summary_sha256=j.sha(audit/'summary.json'),
        parent_O_experiment_sha256=j.sha(o.OUT/'experiment.json'),
        budget='20k NEW online transitions/arm/seed, full episode-batch overshoot reported; parent training and reused teacher cost separately disclosed',
        selection='all initial/fixed/final checkpoints, deterministic and stochastic; no success threshold, no best-checkpoint rescue',
        execution='P audited float64 environment speed-roundoff mapping, <=2e-7; no action teacher or assignment',
        scope='same six repeatedly used training situations; development trend only, not independent generalization',
        maximum_wall_seconds=12*3600)
    sources={}
    for seed in cfg['seeds']:
        folder=o.OUT/'runs'/f'guided_{seed}'
        pointer=j.read(folder/'latest.json')
        assert j.sha(pointer['path'])==pointer['sha256']
        state=torch.load(pointer['path'],map_location='cpu',weights_only=False)
        assert state['experiment_sha256']==cfg['parent_O_experiment_sha256']
        assert j.sha(state['replay_path'])==state['replay_sha256']
        sources[str(seed)]=dict(folder=str(folder),pointer=pointer,
            latest_sha256=j.sha(folder/'latest.json'),replay_path=state['replay_path'],
            replay_sha256=state['replay_sha256'],online_steps=state['steps'],
            gradient_updates=state['gradient_updates'],iteration=state['iteration'])
        del state
    cfg['sources']=sources
    protected=dict(parent['protected_hashes']);protected.update(parent['source_hashes'])
    for path in [ROOT/'code/learning/e45p_actor_supervision.py',ROOT/'code/learning/e45p_execution_precision.py']:
        protected[str(path)]=j.sha(path)
    cfg['protected_hashes']=protected
    files=[Path(__file__).resolve(),ROOT/'code/audit_e45r_smoke.py',
        ROOT/'code/E45R_HASAC_CONTINUATION_PROTOCOL_20261102.md']
    cfg['source_hashes']={str(path):j.sha(path) for path in files}
    j.save(root/'experiment.json',cfg)
    for path in files:
        dst=root/'source_snapshot'/path.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
    return verify(root)


def verify(root):
    cfg=j.read(Path(root)/'experiment.json')
    for section in ('source_hashes','protected_hashes','upstream_hashes','bank_hashes'):
        for path,expected in cfg[section].items():
            if j.sha(path)!=expected:raise RuntimeError('Frozen artifact changed: '+path)
    assert j.sha(o.OUT/'experiment.json')==cfg['parent_O_experiment_sha256']
    for key,hash_key in [('demonstration_file','demonstration_sha256'),('teacher_manifest','teacher_manifest_sha256'),
        ('teacher_label_audit','teacher_label_audit_sha256'),('teacher_label_audit_summary','teacher_label_audit_summary_sha256')]:
        assert j.sha(cfg[key])==cfg[hash_key]
    for source in cfg['sources'].values():
        assert j.sha(Path(source['folder'])/'latest.json')==source['latest_sha256']
        assert j.sha(source['pointer']['path'])==source['pointer']['sha256']
        assert j.sha(source['replay_path'])==source['replay_sha256']
    return cfg


def new_replay(cfg,seed):
    return DemonstrationReplay(checked_data(cfg['demonstration_file']),cfg['replay_capacity'],seed,cfg['demonstration_fraction'])


def evaluate(pool,cfg,seed,model,steps):
    # Only changes output tag; evaluation actions have no teacher/anchor path.
    weights=o.actor_weights(model);out=dict(steps=steps)
    for mode in ('deterministic','stochastic'):
        repeats=1 if mode=='deterministic' else cfg['action_repeats']
        jobs=[dict(actors=weights,training=False,task=o.task_for(e,w,env,seed,mode,
            o.l.seed_for('HASAC_eval',seed,e['id'],w,r),f'R_eval_steps{steps}'))
            for e,w,env in o.l.situations(cfg['entries']) for r in range(repeats)]
        out[mode]=o.l.score([result['summary'] for result in pool.map(o.rollout,jobs)])
    return out


def train(args):
    root=Path(args.output);cfg=verify(root);folder=root/'runs'/f'{args.arm}_{args.seed}'
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cores=list(map(int,args.cpus.split(',')));os.sched_setaffinity(0,{cores[-1]})
    torch.set_num_threads(1);torch.cuda.set_device(args.device);o.configure(folder)
    model=AuthorHASAC(obs_dim=845,device=args.device,seed=args.seed,retirement_mask=True)
    replay=new_replay(cfg,args.seed);source=cfg['sources'][str(args.seed)]
    cfg_sha=j.sha(root/'experiment.json');history=[];steps=0;handles=[];pool=None;queue=None
    started=time.time()
    try:
        if (folder/'latest.json').exists():
            steps,iteration,history=o.restore(folder,model,replay,cfg_sha)
        else:
            inherited_steps,iteration,_=o.restore(Path(source['folder']),model,replay,cfg['parent_O_experiment_sha256'])
            assert inherited_steps==source['online_steps'] and model.runner.total_it==source['gradient_updates']
        teacher=public_teacher(checked_data(cfg['demonstration_file']),model.device)
        weight=cfg['anchor_weight'] if args.arm=='replay_bc' else 0.
        handles,anchor=install_anchor(model,teacher,weight)
        anchor['calls']=2*(model.runner.total_it-source['gradient_updates']) if weight else 0
        pool,queue=pool_for(folder,cores[:-1])
        if not history:
            j.common.emit(folder,'initial_evaluation',steps=0,source_online_steps=source['online_steps'])
            history.append(evaluate(pool,cfg,args.seed,model,0));j.save(folder/'evaluation.json',history)
            o.checkpoint(folder,model,replay,0,iteration,history,cfg_sha)
            j.save(folder/'initialization.json',dict(source=source,full_state_restored=True,
                actor_lr=[a.actor_optimizer.param_groups[0]['lr'] for a in model.actor],
                critic_lr=model.critic.critic_optimizer.param_groups[0]['lr'],
                initial_gradient_updates=model.runner.total_it,anchor_weight=weight,
                optimizer_or_std_reset=False,precision_execution_amendment=True))
        situations=o.l.situations(cfg['entries'])
        while steps<cfg['additional_online_steps']:
            begin=time.time();weights=o.actor_weights(model)
            order=np.tile(np.arange(len(situations)),int(np.ceil(cfg['batch_episodes']/len(situations))))[:cfg['batch_episodes']]
            np.random.default_rng(o.l.seed_for('HASAC_order',args.seed,iteration)).shuffle(order)
            jobs=[]
            for index,which in enumerate(order):
                e,w,env=situations[which]
                task=o.task_for(e,w,env,args.seed,'stochastic',o.l.seed_for('HASAC_actions',args.seed,iteration,index),
                    f'R_train_iter{iteration}_i{index}',uniform=False)
                jobs.append(dict(actors=weights,task=task,training=True))
            results=list(pool.map(o.rollout,jobs));before=steps
            for result in results:replay.add(result['arrays']);steps+=len(result['arrays']['obs'])
            target=source['gradient_updates']+(cfg['smoke_new_updates'] if cfg['smoke'] else steps*cfg['updates_per_transition'])
            while model.runner.total_it<target:model.update(replay.sample(cfg['update_batch']))
            assert model.runner.total_it==target
            assert anchor['calls']==(2*(target-source['gradient_updates']) if weight else 0)
            iteration+=1;rows=[result['summary'] for result in results]
            record=dict(steps=steps,iteration=iteration,arm=args.arm,
                gradient_updates=model.runner.total_it,new_gradient_updates=target-source['gradient_updates'],
                parent_online_steps=source['online_steps'],total_online_steps=source['online_steps']+steps,
                reused_teacher_transitions=cfg['demonstration_steps'],new_teacher_queries=0,
                replay_size=replay.size,anchor=dict(anchor),fit=fit_metrics(model,teacher),
                episodes=rows,score=o.l.score(rows),seconds=time.time()-begin)
            j.save(folder/'updates'/f'{iteration:05d}.json',record)
            if before//cfg['evaluation_every']!=steps//cfg['evaluation_every'] or steps>=cfg['additional_online_steps']:
                j.common.emit(folder,'evaluation',steps=steps,new_gradient_updates=target-source['gradient_updates'])
                history.append(evaluate(pool,cfg,args.seed,model,steps));j.save(folder/'evaluation.json',history)
                o.checkpoint(folder,model,replay,steps,iteration,history,cfg_sha)
            j.common.emit(folder,'training',steps=steps,new_gradient_updates=target-source['gradient_updates'],score=record['score'],anchor=anchor)
        verify(root)
        j.save(folder/'decision.json',dict(status='complete',steps=steps,history=history,source=source,
            new_gradient_updates=model.runner.total_it-source['gradient_updates'],
            gradient_updates=model.runner.total_it,anchor=anchor,wall_seconds=time.time()-started,
            new_teacher_queries=0,new_bc_pretraining_updates=0,convergence_claim=False))
        j.common.emit(folder,'complete',steps=steps,new_gradient_updates=model.runner.total_it-source['gradient_updates'])
    except BaseException as exc:
        j.common.emit(folder,'failed',steps=steps,error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        for handle in handles:handle.remove()
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
        if queue is not None:queue.close()
        lock.close()


def report(root):
    root=Path(root);cfg=j.read(root/'experiment.json')
    state=j.read(root/'status.json') if (root/'status.json').exists() else {}
    lines=['E45-R：O完整状态继续训练，示范回放 / 示范回放+持续BC',time.strftime('%FT%T%z'),
        '状态：'+state.get('stage','registered'),
        '仅原始163条标签；无Q纠错；不重置critic/优化器/std；纯Actor评估。','',
        '方法/种子 | 状态 | 新在线步 | 新RL更新 | 确定性双捕 | 随机双捕']
    md=['# E45-R：完整HASAC状态继续训练对照','',
        '六个反复使用的训练情形；仅开发趋势，不是独立泛化测试。','',
        '两臂从同一种子的O guided最终完整状态分叉，保留actor/critic/targets/优化器/replay/RNG。',
        '共同保留25%原始示范回放；仅replay_bc增加lambda=0.1原始163条公共标签潜变量MSE；无Q纠错或在线教师。','',
        '|方法|种子|新增在线步|新增RL更新|初始确定性双捕|最新确定性双捕|最新随机双捕|状态|',
        '|---|---:|---:|---:|---|---|---|---|']
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            folder=root/'runs'/f'{arm}_{seed}'
            status=j.read(folder/'status.json') if (folder/'status.json').exists() else {}
            h=j.read(folder/'evaluation.json') if (folder/'evaluation.json').exists() else []
            a=b=initial='—'
            if h:
                d=h[-1]['deterministic'];s=h[-1]['stochastic'];d0=h[0]['deterministic']
                a=f"{d['double']}/{d['episodes']}";b=f"{s['double']}/{s['episodes']}";initial=f"{d0['double']}/{d0['episodes']}"
            stage=status.get('stage','queued');steps=status.get('steps',0);updates=status.get('new_gradient_updates',0)
            lines.append(f'{arm}/{seed} | {stage} | {steps} | {updates} | {a} | {b}')
            md.append(f'|{arm}|{seed}|{steps}|{updates}|{initial}|{a}|{b}|{stage}|')
    lines+=['','每路约20k新增步，每5k评估，整批超出如实记录；不挑最佳检查点。',str(root)]
    md+=['','原训练/示范搜索成本为共同历史成本，另列于experiment.json及各运行source；本次新增教师查询0。',
         '完整曲线见各运行evaluation.json；低分不构成软件失败，也不自动追加预算。',
         '20k步用于决定是否扩大开发，不用于证明算法不可能成功。']
    for name,content in [('LIVE.txt',lines),('REPORT.md',md)]:
        temp=root/(name+'.tmp');temp.write_text('\n'.join(content)+'\n');temp.replace(root/name)


def pipeline(args):
    root=Path(args.output).resolve();cfg=register(root,args.smoke)
    lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    children=[];started=time.time()
    try:
        if not cfg['smoke']:
            verify(SMOKE);audit=j.read(SMOKE/'smoke_audit.json');assert audit['passed']
            assert audit['runner_sha256']==j.sha(Path(__file__).resolve())
        combos=[(arm,seed) for arm in cfg['arms'] for seed in cfg['seeds']]
        cores=j.choose_cpus(len(combos)*(args.workers+1));assert len(cores)==len(combos)*(args.workers+1)
        allocation=[]
        for i,(arm,seed) in enumerate(combos):
            folder=root/'runs'/f'{arm}_{seed}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'decision.json').exists():continue
            use=cores[i*(args.workers+1):(i+1)*(args.workers+1)]
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'train','--output',str(root),
                '--seed',str(seed),'--arm',arm,'--device',args.device,'--cpus',','.join(map(str,use)),
                '--parent',str(os.getpid())]
            with (folder/'run.log').open('a') as log:child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            children.append(child);allocation.append(dict(arm=arm,seed=seed,pid=child.pid,cores=use,device=args.device))
        j.save(root/'execution.json',dict(allocation=allocation,workers_per_model=args.workers,started_unix=started))
        j.common.emit(root,'running',pids=[c.pid for c in children]);report(root)
        while any(child.poll() is None for child in children):
            report(root)
            if any(child.poll() not in (None,0) for child in children):raise RuntimeError('R trainer failed; inspect preserved logs')
            if time.time()-started>cfg['maximum_wall_seconds']:raise RuntimeError('Registered wall cap reached')
            time.sleep(5)
        assert all(child.returncode==0 for child in children)
        verify(root)
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
    parser.add_argument('--workers',type=int,default=12);parser.add_argument('--device',default='cuda:2')
    parser.add_argument('--seed',type=int,default=17);parser.add_argument('--arm',choices=ARMS,default='replay_bc')
    parser.add_argument('--cpus');parser.add_argument('--parent',type=int);args=parser.parse_args()
    if args.parent:
        ctypes.CDLL(None).prctl(1,signal.SIGTERM)
        if os.getppid()!=args.parent:raise RuntimeError('Coordinator exited')
    if args.command=='register':register(args.output,args.smoke)
    elif args.command=='pipeline':pipeline(args)
    elif args.command=='train':train(args)
    else:report(args.output)


if __name__=='__main__':main()
