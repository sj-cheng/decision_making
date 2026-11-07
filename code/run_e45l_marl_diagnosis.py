"""E45-L: original recurrent MAPPO, fixed hidden-task learning diagnosis.

No algorithm invention: reuse frozen J actor, critic and author PPO. Separate
environment and action RNG; no validation/test histories or assignments.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
import fcntl
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import run_e45k_main_task_learning as k
from e45g_training_acceleration import public_geometry_cache

j=k.j
ROOT=k.ROOT
OUT=ROOT/'current/e45l_marl_diagnosis_20261101'
FOLDER=None
OLD_FACTORY=j.factory


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:4],'little')%2**31


def situations(entries):
    return [(e,w,seed_for('E45L_environment',e['id'],w)) for e in entries for w in ('AB','BA')]


def configure(folder):
    global FOLDER
    FOLDER=Path(folder);k.configure(folder);j.common.base.make_controller=factory


def factory(case,task):
    own=OLD_FACTORY(case,task)
    if task['arm']=='RMAPPO':
        own.deterministic=task.get('execution_mode','deterministic')=='deterministic'
        # This RNG affects action sampling only; target/search/belief RNG uses
        # the separately fixed task seed. Prefix warmup is deterministic.
        torch.manual_seed(task['policy_seed'])
    return own


def init_worker(folder,queue,parent):
    ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    if os.getppid()!=parent:raise RuntimeError('Parent exited')
    os.sched_setaffinity(0,{queue.get()});torch.set_num_threads(1)
    configure(folder);j.common.base.PARENT=parent


def pool_for(folder,cores):
    ctx=mp.get_context('spawn');q=ctx.Queue()
    for c in cores:q.put(c)
    return ProcessPoolExecutor(max_workers=len(cores),mp_context=ctx,initializer=init_worker,
                               initargs=(str(folder),q,os.getpid())),q


def register(root,smoke=False):
    root=Path(root)
    if (root/'experiment.json').exists():return verify(root)
    root.mkdir(parents=True,exist_ok=True)
    k.verify(k.DEFAULT_OUT)
    previous=j.read(k.JROOT/'experiment.json');bank=j.read(k.JROOT/'bank/manifest.json')
    # Selection uses metadata/order alone; no controller scores are read.
    entries=[sorted([e for e in bank['train'] if e['family']==f],key=lambda e:e['id'])[0]
             for f in sorted({e['family'] for e in bank['train']})]
    assert len(entries)==3 and all(e['split']=='train' for e in entries)
    if smoke:entries=entries[:1]
    files=[Path(__file__).resolve(),ROOT/'code/tests/test_e45l_marl_diagnosis.py',
           ROOT/'code/E45L_MARL_DIAGNOSIS_PROTOCOL_20261101.md',ROOT/'code/run_e45k_main_task_learning.py']
    protected=dict(j.read(k.DEFAULT_OUT/'experiment.json')['source_hashes'])
    protected.update(j.read(k.DEFAULT_OUT/'experiment.json')['protected_hashes'])
    starts={str(s):str(k.JROOT/f'learning/RMAPPO_{s}/latest.pt') for s in (17,29)}
    cfg=dict(experiment='E45-L',registered_at=time.strftime('%FT%T%z'),smoke=smoke,
             entries=entries,seeds=[17,29],initializations=['random','J_latest'],
             steps_per_model=1 if smoke else 100000,batch_episodes=2 if smoke else 24,
             evaluation_every=20000,policy_repeats=1 if smoke else 2,
             opponent='rootpuct',task_situations=len(entries)*2,starts=starts,
             lr_start=3e-4,lr_end=3e-5,entropy=.01,critic_only_updates=0 if smoke else 3,
             reward='unchanged J reward: double capture + terminal-zero potential - illegal-intent cost',
             optimizer='fresh Adam in both initialization arms; weights are the intervention',
             settings='author J: 64 GRU, 844 public inputs, gamma1 lambda.95, epochs5 minibatch2 chunk10',
             selection='no checkpoint selected; report all fixed checkpoints and final',
             capability_target=.8,target_scope='per-seed deterministic fixed training situations; diagnostic only',
             positive_control='GEOM_SEARCH public no-assignment joint search, 200ms',
             held_out_test_used=False,validation_used=False,maximum_wall_seconds=8*3600,
             source_hashes={str(p):j.sha(p) for p in files},protected_hashes=protected,
             bank_hashes={e['path']:j.sha(e['path']) for e in entries})
    j.save(root/'experiment.json',cfg)
    import shutil
    for p in files:
        target=root/'source_snapshot'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    return verify(root)


def verify(root):
    cfg=j.read(Path(root)/'experiment.json')
    for sec in ('source_hashes','protected_hashes','bank_hashes'):
        for p,h in cfg[sec].items():
            if j.sha(p)!=h:raise RuntimeError('Frozen file changed '+p)
    return cfg


def make_task(entry,world,env_seed,train_seed,stage,mode,policy_seed,checkpoint=None,arm='RMAPPO',suffix=''):
    t=j.request(entry,world,'rootpuct',arm,train_seed,stage,checkpoint,episode_seed=env_seed)
    t.update(execution_mode=mode,policy_seed=int(policy_seed))
    t['key']+=f'_L_{mode}_actionseed{policy_seed}_{suffix}'
    return t


def evaluate(task):
    try:
        payload,_=j.common.base.episode(task)
        assert payload['summary']['public_replay_exact']
        j.save(FOLDER/'evaluation/episodes'/(task['key']+'.json'),payload)
        row=payload['summary'];j.save(FOLDER/'evaluation/cache'/(task['key']+'.json'),row)
        return row
    except BaseException as exc:
        j.save(FOLDER/'failures'/(task['key']+'.json'),dict(task=task,error=repr(exc),traceback=traceback.format_exc()))
        raise


def train_collect(job):
    # k.collect performs full rollout, exact replay, GAE input extraction, and
    # compressed trajectory persistence. Our factory resets action RNG only.
    return k.collect(job)


def score(rows):
    return dict(episodes=len(rows),double=sum(bool(r['double_capture']) for r in rows),
                first=sum(r['captures']>0 for r in rows),
                utility=float(np.mean([r['utility'] for r in rows])))


def evaluate_policy(pool,cfg,seed,checkpoint,steps):
    result=dict(steps=steps,checkpoint=str(checkpoint),sha256=j.sha(checkpoint))
    for mode in ('deterministic','stochastic'):
        repeats=1 if mode=='deterministic' else cfg['policy_repeats']
        tasks=[make_task(e,w,env,seed,'diagnosis_evaluation',mode,seed_for('eval_actions',seed,e['id'],w,r),
                         checkpoint,suffix=f'steps{steps}')
               for e,w,env in situations(cfg['entries']) for r in range(repeats)]
        result[mode]=score(list(pool.map(evaluate,tasks)))
    return result


def save_model(folder,model,steps,update,history,cfg_sha,checkpoint=False):
    path=folder/'checkpoints'/f'steps_{steps}.pt' if checkpoint else folder/'latest.pt'
    path.parent.mkdir(parents=True,exist_ok=True)
    data=dict(actor=model.actor.state_dict(),critic=model.critic.state_dict(),
              actor_optimizer=model.actor_optimizer.state_dict(),critic_optimizer=model.critic_optimizer.state_dict(),
              steps=steps,update=update,history=history,experiment_sha256=cfg_sha,
              torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(model.device))
    temp=path.with_suffix('.tmp');torch.save(data,temp);temp.replace(path);return path


def train(args):
    root=Path(args.output);cfg=verify(root);folder=root/'runs'/f'{args.initialization}_{args.seed}'
    folder.mkdir(parents=True,exist_ok=True);lock=(folder/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cores=list(map(int,args.cpus.split(',')));os.sched_setaffinity(0,{cores[-1]});torch.set_num_threads(1)
    torch.cuda.set_device(args.device);configure(folder);torch.manual_seed(args.seed)
    model,trainer=j.policy('RMAPPO',args.device);steps=update=0;history=[]
    cfg_sha=j.sha(root/'experiment.json')
    resume=folder/'latest.pt'
    if resume.exists():
        state=torch.load(resume,map_location=model.device,weights_only=False)
        assert state['experiment_sha256']==cfg_sha
        for key in ('actor','critic','actor_optimizer','critic_optimizer'):getattr(model,key).load_state_dict(state[key])
        steps,update,history=state['steps'],state['update'],state['history']
        torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu(),model.device)
    elif args.initialization=='J_latest':
        state=torch.load(cfg['starts'][str(args.seed)],map_location=model.device,weights_only=False)
        model.actor.load_state_dict(state['actor']);model.critic.load_state_dict(state['critic'])
    pool,q=pool_for(folder,cores[:-1]);started=time.time()
    try:
        if not history:
            j.common.emit(folder,'initial_evaluation',steps=0)
            path=save_model(folder,model,0,0,history,cfg_sha,True)
            history.append(evaluate_policy(pool,cfg,args.seed,path,0));j.save(folder/'evaluation.json',history)
            save_model(folder,model,0,0,history,cfg_sha)
        taskspec=situations(cfg['entries'])
        while steps<cfg['steps_per_model']:
            trainer.prep_rollout();weights={key:v.detach().cpu() for key,v in model.actor.state_dict().items()}
            jobs=[];rng=np.random.default_rng(seed_for('order',args.seed,update))
            # Balanced fixed situations, reshuffled each update. Environment
            # seeds are fixed; action seeds change with every sampled episode.
            order=np.tile(np.arange(len(taskspec)),int(np.ceil(cfg['batch_episodes']/len(taskspec))))[:cfg['batch_episodes']]
            rng.shuffle(order)
            for i,index in enumerate(order):
                e,w,env=taskspec[index]
                task=make_task(e,w,env,args.seed,'training','stochastic',seed_for('train_actions',args.seed,update,i),suffix=f'u{update}_i{i}')
                jobs.append(dict(actor=weights,task=task,variant='original_reward'))
            begin=time.time();episodes=list(pool.map(train_collect,jobs));buffer=j.finish_episodes(episodes,model)
            diagnostics=k.value_diagnostics(episodes);frac=min(1.,steps/cfg['steps_per_model'])
            lr=cfg['lr_start']*(1-frac)+cfg['lr_end']*frac
            for optim in (model.actor_optimizer,model.critic_optimizer):
                for group in optim.param_groups:group['lr']=lr
            trainer.entropy_coef=cfg['entropy'];trainer.prep_training()
            info=trainer.train(buffer,update_actor=update>=cfg['critic_only_updates'])
            info={key:float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for key,v in info.items()}
            assert all(np.isfinite(v) for v in info.values())
            before=steps;steps+=sum(len(e['obs']) for e in episodes);update+=1
            row=dict(steps=steps,update=update,seconds=time.time()-begin,lr=lr,
                     episodes=[e['summary'] for e in episodes],score=score([e['summary'] for e in episodes]),
                     **diagnostics,**info)
            j.save(folder/'updates'/f'{update:05d}.json',row)
            if before//cfg['evaluation_every']!=steps//cfg['evaluation_every'] or steps>=cfg['steps_per_model']:
                path=save_model(folder,model,steps,update,history,cfg_sha,True)
                j.common.emit(folder,'evaluation',steps=steps,update=update)
                history.append(evaluate_policy(pool,cfg,args.seed,path,steps));j.save(folder/'evaluation.json',history)
            save_model(folder,model,steps,update,history,cfg_sha)
            j.common.emit(folder,'training',steps=steps,update=update,score=row['score'])
        verify(root);j.save(folder/'decision.json',dict(status='complete',steps=steps,updates=update,history=history,
                                                      wall_seconds=time.time()-started))
        j.common.emit(folder,'complete',steps=steps)
    except BaseException as exc:
        j.common.emit(folder,'failed',steps=steps,error=repr(exc),traceback=traceback.format_exc());raise
    finally:pool.shutdown(wait=True,cancel_futures=True);q.close();lock.close()


def report(root):
    root=Path(root);cfg=j.read(root/'experiment.json')
    lines=['E45-L 真实隐藏主任务诊断（不是论文测试结果）',time.strftime('%FT%T%z'),'',
           '随机初始化 vs J权重；作者RMAPPO；固定RootPUCT3000；无任务指派',
           '初始化 / 种子 | 阶段 | 新增步 | 最后评估步 | 确定性双捕 | 随机双捕','']
    for init in cfg['initializations']:
        for seed in cfg['seeds']:
            f=root/'runs'/f'{init}_{seed}'
            s=j.read(f/'status.json') if (f/'status.json').exists() else {}
            h=j.read(f/'evaluation.json') if (f/'evaluation.json').exists() else []
            tail='尚未完成初评'
            if h:
                r=h[-1];d=r['deterministic'];st=r['stochastic']
                tail=f'{r["steps"]} | {d["double"]}/{d["episodes"]} | {st["double"]}/{st["episodes"]}'
            lines.append(f'{init} / {seed} | {s.get("stage","queued")} | {s.get("steps",0)} | {tail}')
    p=root/'positive_control/summary.json'
    if p.exists():
        d=j.read(p);lines+=['',f'GEOM_SEARCH正向控制：{d["double"]}/{d["episodes"]} 双捕（200ms开发负载）']
    lines+=['','完整记录：'+str(root),'不要求基线输给主方法；小任务掌握不等于泛化/强基线验收通过。']
    temp=root/'LIVE.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(root/'LIVE.txt')


def pipeline(args):
    root=Path(args.output);cfg=register(root,args.smoke)
    lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cores=j.choose_cpus(4*(args.workers+1));children=[];started=time.time()
    try:
        # Full public search controller confirms fixed training tasks are
        # reachable under the same finite-horizon physical rules.
        configure(root/'positive_control');pool,q=pool_for(root/'positive_control',cores[:args.workers])
        try:
            tasks=[make_task(e,w,env,0,'positive_control','deterministic',0,arm='GEOM_SEARCH') for e,w,env in situations(cfg['entries'])]
            rows=list(pool.map(evaluate,tasks));j.save(root/'positive_control/summary.json',score(rows))
        finally:pool.shutdown(wait=True);q.close()
        for i,(init,seed) in enumerate((a,s) for a in cfg['initializations'] for s in cfg['seeds']):
            f=root/'runs'/f'{init}_{seed}';f.mkdir(parents=True,exist_ok=True)
            if (f/'decision.json').exists():continue
            use=cores[i*(args.workers+1):(i+1)*(args.workers+1)]
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'train','--output',str(root),
                 '--initialization',init,'--seed',str(seed),'--device',f'cuda:{2+i%2}',
                 '--cpus',','.join(map(str,use)),'--parent',str(os.getpid())]
            with (f/'run.log').open('a') as log:children.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT))
        j.save(root/'execution.json',dict(pids=[c.pid for c in children],cores=cores))
        j.common.emit(root,'running',pids=[c.pid for c in children])
        while any(c.poll() is None for c in children):
            report(root)
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Trainer failed; inspect preserved logs')
            if time.time()-started>cfg['maximum_wall_seconds']:raise RuntimeError('Wall cap reached')
            time.sleep(10)
        assert all(c.returncode==0 for c in children);verify(root);report(root)
        j.save(root/'decision.json',dict(status='complete',wall_seconds=time.time()-started))
        j.common.emit(root,'complete')
    except BaseException as exc:
        j.common.emit(root,'failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            try:c.wait(timeout=20)
            except subprocess.TimeoutExpired:c.kill();c.wait()
        lock.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','pipeline','train','report'])
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--smoke',action='store_true')
    p.add_argument('--workers',type=int,default=12);p.add_argument('--initialization',choices=['random','J_latest'])
    p.add_argument('--seed',type=int,default=17);p.add_argument('--device',default='cuda:2');p.add_argument('--cpus');p.add_argument('--parent',type=int)
    args=p.parse_args()
    if args.parent:
        ctypes.CDLL(None).prctl(1,signal.SIGTERM)
        if os.getppid()!=args.parent:raise RuntimeError('Coordinator exited')
    if args.command=='register':register(args.output,args.smoke)
    elif args.command=='pipeline':pipeline(args)
    elif args.command=='train':train(args)
    else:report(args.output)


if __name__=='__main__':main()
