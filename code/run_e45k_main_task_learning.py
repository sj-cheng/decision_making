"""Bounded, paired hidden-task RMAPPO reward-credit development experiment.

Only C/J training and validation histories are used. Original artifacts and
author PPO code are read-only. No test benchmark is generated or scored.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import ctypes
import fcntl
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import run_e45j_repaired_benchmark as j
from learning.e45j_support125 import install
from e45g_training_acceleration import public_geometry_cache

ROOT = Path(__file__).resolve().parents[1]
JROOT = ROOT / 'current/e45j_repaired_benchmark_20261031'
DEFAULT_OUT = ROOT / 'current/e45k_main_task_learning_20261031'
BASE_SEED = 947220261031
VARIANTS = ('original_reward', 'capture_credit')
FOLDER = None


def rewards(raw, phi, blocked, new_captures, variant):
    """Per-capture training objective; evaluation always retains original utility."""
    raw = np.asarray(raw, dtype=float)
    events = np.asarray(new_captures, dtype=float)
    if variant == 'original_reward':
        base = raw
    elif variant == 'capture_credit':
        # raw = full-team-success indicator - .0025. Sum of event reward is
        # 1 for double capture, .5 for single capture, 0 for no capture.
        base = .5 * events - .0025
    else:
        raise ValueError(variant)
    return j.training_rewards(base, phi, blocked)


def select_entries(entries):
    buckets = defaultdict(list)
    for e in entries:
        case = j.read(e['path'])
        buckets[e['family'], e['pattern'], case['speed']].append(e)
    # Two pre-existing training histories in every family x entry x speed cell.
    chosen = [e for key in sorted(buckets) for e in sorted(buckets[key], key=lambda e:e['id'])[:2]]
    assert len(chosen) == 36 and all(e['split'] == 'train' for e in chosen)
    probes = [sorted(buckets[key], key=lambda e:e['id'])[0] for key in sorted(buckets)]
    return chosen, probes


def schedule(entries, seed, update, count=24):
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, seed, update]))
    jobs = []
    for index in range(count):
        e = entries[int(rng.integers(len(entries)))]
        jobs.append((e, 'AB' if rng.random()<.5 else 'BA',
                     'fixed_D_kernel' if index % 4 == 0 else 'rootpuct', int(rng.integers(2**31))))
    return jobs


def register(root, smoke=False):
    root = Path(root)
    if (root/'experiment.json').exists():
        return verify(root)
    root.mkdir(parents=True, exist_ok=True)
    previous = j.read(JROOT/'experiment.json')
    manifest = j.read(JROOT/'bank/manifest.json')
    train, probes = select_entries(manifest['train'])
    validation = manifest['validation']
    if smoke:
        train, probes, validation = train[:2], probes[:1], validation[:1]
    source_files = [Path(__file__).resolve(), ROOT/'code/tests/test_e45k_main_task_learning.py',
                    ROOT/'code/E45K_MAIN_TASK_LEARNING_PROTOCOL_20261031.md',
                    ROOT/'code/learning/e45j_repaired_mappo.py',
                    ROOT/'code/learning/e45i_official_mappo.py',
                    ROOT/'code/learning/e45j_support125.py',
                    ROOT/'code/run_e45i_baselines.py', ROOT/'code/run_e45b_role_free.py']
    protected = dict(previous['source_hashes'])
    protected.update(previous['protected_hashes'])
    protected.update(previous['upstream_hashes'])
    starts = {}
    for seed in (17, 29):
        path = JROOT/f'learning/RMAPPO_{seed}/latest.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['steps'] >= 500000
        starts[str(seed)] = dict(path=str(path), sha256=j.sha(path), prior_steps=saved['steps'])
        protected[str(path)] = j.sha(path)
    cfg = dict(experiment='E45-K1', registered_at=time.strftime('%FT%T%z'), smoke=smoke,
               scope='hidden-task learnability development, not an independent test benchmark',
               variants=list(VARIANTS), seeds=[17,29], additional_steps_per_run=1 if smoke else 100000,
               batch_episodes=2 if smoke else 24, validation_every=25000,
               train_entries=train, train_probe_entries=probes, validation_entries=validation,
               training_root_fraction=.75, starts=starts, lr_start=1e-4, lr_end=3e-5,
               entropy_start=.002, entropy_end=.0005, critic_only_initial_updates=3 if not smoke else 0,
               original_task_scoring=True, held_out_test_used=False,
               capability_gate=dict(train_root_capture_each_seed=.5, validation_root_capture_mean=.25,
                                    meaning='development capability only; no strong-baseline or convergence claim'),
               source_hashes={str(p):j.sha(p) for p in source_files}, protected_hashes=protected,
               bank_hashes={e['path']:j.sha(e['path']) for e in train+validation},
               parent_j_experiment_sha256=j.sha(JROOT/'experiment.json'))
    j.save(root/'experiment.json', cfg)
    for p in source_files:
        dst=root/'source_snapshot'/p.relative_to(ROOT/'code');dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dst)
    shutil.copy2(ROOT/'code/E45K_MAIN_TASK_LEARNING_PROTOCOL_20261031.md',root/'PROTOCOL.md')
    return verify(root)


def verify(root):
    cfg=j.read(Path(root)/'experiment.json')
    for section in ('source_hashes','protected_hashes','bank_hashes'):
        for path,digest in cfg[section].items():
            if j.sha(path)!=digest:
                raise RuntimeError('Frozen artifact changed: '+path)
    return cfg


def configure(folder):
    global FOLDER
    FOLDER=Path(folder);j.configure(FOLDER);j.common.BASE_SEED=BASE_SEED
    install(FOLDER/'support125_evidence')


def worker_init(folder, queue, parent):
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid()!=parent:raise RuntimeError('Trainer exited')
    os.sched_setaffinity(0,{queue.get()});torch.set_num_threads(1)
    configure(folder);j.common.base.PARENT=parent


def pool_for(folder, cpus):
    context=mp.get_context('spawn');queue=context.Queue()
    for c in cpus:queue.put(c)
    return ProcessPoolExecutor(max_workers=len(cpus),mp_context=context,initializer=worker_init,
                               initargs=(str(folder),queue,os.getpid())),queue


def collect(job):
    try:
        model,_=j.policy('RMAPPO');model.actor.load_state_dict(job['actor']);model.actor.eval()
        j.ROLLOUT_ACTOR=model.actor;torch.manual_seed(job['task']['seed'])
        with public_geometry_cache():payload,_=j.common.base.episode(job['task'])
        own=j.CAPTURE;trace=payload['trace'];samples=own.samples
        assert len(trace)==len(samples) and payload['summary']['public_replay_exact']
        ep={k:np.stack([r[k] for r in samples]) for k in ('obs','critic_public','actor_hidden','raw','logp','active')}
        ep['privileged']=np.asarray([r['state'][:4] for r in trace],np.float32)/10
        raw=[r['reward'] for r in trace];events=[r['new_captures'] for r in trace]
        phi=[r['phi'] for r in samples];blocked=[r['blocked_fraction'] for r in samples]
        ep['reward']=rewards(raw,phi,blocked,events,job['variant'])
        # Persist complete original-scoring trajectories, not only successful runs.
        data=gzip.compress(json.dumps(payload,ensure_ascii=False,allow_nan=False).encode(),mtime=0)
        path=FOLDER/'training/episodes'/(job['task']['key']+'.json.gz');path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.tmp');temp.write_bytes(data);temp.replace(path)
        ep['summary']={k:payload['summary'][k] for k in
            ('key','case','world','opponent','seed','decisions','utility','double_capture','captures','first_capture','public_replay_exact')}
        actions=np.array([r['action'] for r in trace]);active=ep['active'][...,0]
        ep['summary'].update(blocked_fraction=float(np.mean(blocked)),
            active_action_norm=float((np.linalg.norm(actions,axis=-1)*active).sum()/max(1,active.sum())),
            training_return=float(ep['reward'].sum()), original_return=float(sum(raw)),
            episode=str(path),episode_sha256=hashlib.sha256(data).hexdigest())
        return ep
    except BaseException as exc:
        j.save(FOLDER/'failures'/(job['task']['key']+'.json'),dict(task=job['task'],error=repr(exc),traceback=traceback.format_exc()))
        raise


def evaluate(task):
    return j.common.evaluate_job(task)


def metrics(rows):
    result={}
    for opp in ('fixed_D_kernel','rootpuct'):
        rr=[r for r in rows if r['opponent']==opp]
        if not rr:continue
        result[opp]=dict(episodes=len(rr),captures=sum(r['double_capture'] for r in rr),
            double_capture=float(np.mean([r['double_capture'] for r in rr])),
            first_capture=float(np.mean([r['captures']>0 for r in rr])),
            utility=float(np.mean([r['utility'] for r in rr])),
            completion_cost=float(np.mean([r.get('completion_cost',r['decisions']) for r in rr])))
    return result


def save_checkpoint(folder,model,seed,variant,steps,update,history,cfg_sha,latest=False):
    path=folder/'latest.pt' if latest else folder/'checkpoints'/f'steps_{steps}.pt'
    path.parent.mkdir(parents=True,exist_ok=True)
    obj=dict(actor={k:v.detach().cpu() for k,v in model.actor.state_dict().items()},
             critic={k:v.detach().cpu() for k,v in model.critic.state_dict().items()},
             actor_optimizer=model.actor_optimizer.state_dict(),critic_optimizer=model.critic_optimizer.state_dict(),
             steps=steps,update=update,seed=seed,variant=variant,history=history,experiment_sha256=cfg_sha,
             torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(model.device) if model.device.type=='cuda' else None)
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path);return path


def validate(pool,cfg,seed,path,steps):
    result=dict(steps=steps,checkpoint=str(path),checkpoint_sha256=j.sha(path))
    for tag,entries,worlds in [('train_probe',cfg['train_probe_entries'],['AB']),
                               ('validation',cfg['validation_entries'],['AB','BA'])]:
        tasks=[j.request(e,w,o,'RMAPPO',seed,tag,path) for e in entries for w in worlds
               for o in ('fixed_D_kernel','rootpuct')]
        rows=list(pool.map(evaluate,tasks));result[tag]=metrics(rows)
        assert all(r['public_replay_exact'] for r in rows)
    result['selection_utility']=float(np.mean([r['utility'] for r in rows]))
    return result


def value_diagnostics(episodes):
    values=np.concatenate([e['values'][...,0][e['active'][...,0]>0] for e in episodes])
    returns=np.concatenate([e['returns'][...,0][e['active'][...,0]>0] for e in episodes])
    var=float(np.var(returns))
    return dict(value_rmse=float(np.sqrt(np.mean((returns-values)**2))),
                explained_variance=float(1-np.var(returns-values)/var) if var>1e-10 else None,
                advantage_mean=float(np.mean(returns-values)),advantage_std=float(np.std(returns-values)))


def train(args):
    root=Path(args.output);cfg=verify(root);folder=root/'runs'/f'{args.variant}_{args.seed}'
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpus=list(map(int,args.cpus.split(',')));os.sched_setaffinity(0,{cpus[-1]});torch.set_num_threads(1)
    if args.device.startswith('cuda'):torch.cuda.set_device(torch.device(args.device))
    configure(folder);torch.manual_seed(args.seed);model,trainer=j.policy('RMAPPO',args.device)
    cfg_sha=j.sha(root/'experiment.json');resume=folder/'latest.pt'
    src=resume if resume.exists() else Path(cfg['starts'][str(args.seed)]['path'])
    state=torch.load(src,map_location=model.device,weights_only=False)
    for k in ('actor','critic','actor_optimizer','critic_optimizer'):getattr(model,k).load_state_dict(state[k])
    if resume.exists():
        assert state['experiment_sha256']==cfg_sha
        steps,update,history=state['steps'],state['update'],state['history']
        torch.set_rng_state(state['torch_rng'].cpu())
        if state['cuda_rng'] is not None:torch.cuda.set_rng_state(state['cuda_rng'].cpu(),model.device)
    else:
        steps,update,history=0,0,[]
    pool,queue=pool_for(folder,cpus[:-1]);started=time.time()
    try:
        if not history:
            j.common.emit(folder,'initial_validation',steps=0)
            path=save_checkpoint(folder,model,args.seed,args.variant,steps,update,history,cfg_sha)
            history.append(validate(pool,cfg,args.seed,path,0));j.save(folder/'validation.json',history)
            j.save(folder/'selected.json',history[-1])
            save_checkpoint(folder,model,args.seed,args.variant,steps,update,history,cfg_sha,True)
        while steps<cfg['additional_steps_per_run']:
            begin=time.time();trainer.prep_rollout()
            weights={k:v.detach().cpu() for k,v in model.actor.state_dict().items()}
            jobs=[]
            for e,w,o,seed in schedule(cfg['train_entries'],args.seed,update,cfg['batch_episodes']):
                task=j.request(e,w,o,'RMAPPO',args.seed,'training',episode_seed=seed)
                jobs.append(dict(actor=weights,task=task,variant=args.variant))
            episodes=list(pool.map(collect,jobs));rollout_seconds=time.time()-begin
            buffer=j.finish_episodes(episodes,model);diag=value_diagnostics(episodes)
            frac=min(1.,steps/cfg['additional_steps_per_run'])
            lr=cfg['lr_start']*(1-frac)+cfg['lr_end']*frac
            for optimizer in (model.actor_optimizer,model.critic_optimizer):
                for g in optimizer.param_groups:g['lr']=lr
            trainer.entropy_coef=cfg['entropy_start']*(1-frac)+cfg['entropy_end']*frac
            trainer.prep_training();info=trainer.train(buffer,update_actor=update>=cfg['critic_only_initial_updates'])
            info={k:float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k,v in info.items()}
            if not all(np.isfinite(v) for v in info.values()):raise RuntimeError('Nonfinite PPO update')
            before=steps;steps+=sum(len(e['obs']) for e in episodes);update+=1
            row=dict(update=update,steps=steps,learning_rate=lr,actor_updated=update>cfg['critic_only_initial_updates'],
                     rollout_seconds=rollout_seconds,total_seconds=time.time()-begin,
                     by_opponent=metrics([e['summary'] for e in episodes]),
                     blocked_fraction=float(np.mean([e['summary']['blocked_fraction'] for e in episodes])),
                     episodes=[e['summary'] for e in episodes],**diag,**info)
            j.save(folder/'updates'/f'{update:05d}.json',row)
            if before//cfg['validation_every']!=steps//cfg['validation_every'] or steps>=cfg['additional_steps_per_run']:
                path=save_checkpoint(folder,model,args.seed,args.variant,steps,update,history,cfg_sha)
                j.common.emit(folder,'validation',steps=steps,update=update)
                history.append(validate(pool,cfg,args.seed,path,steps));j.save(folder/'validation.json',history)
                chosen=max(history,key=lambda r:r['selection_utility']);j.save(folder/'selected.json',chosen)
            save_checkpoint(folder,model,args.seed,args.variant,steps,update,history,cfg_sha,True)
            j.common.emit(folder,'training',steps=steps,update=update,by_opponent=row['by_opponent'],
                          blocked_fraction=row['blocked_fraction'],value_rmse=diag['value_rmse'])
        verify(root)
        done=dict(status='complete',steps=steps,updates=update,seed=args.seed,variant=args.variant,
                  selected=j.read(folder/'selected.json'),wall_seconds=time.time()-started,
                  convergence_claim=False,held_out_test_used=False)
        j.save(folder/'decision.json',done);j.common.emit(folder,'complete',steps=steps)
    except BaseException as exc:
        j.common.emit(folder,'failed',steps=steps,update=update,error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True);queue.close();lock.close()


def report(root):
    root=Path(root);cfg=j.read(root/'experiment.json');records=[]
    lines=['# E45-K1 主任务可学习性开发进度','',
           '仅36训练历史与原12验证历史；两奖励×两已有种子续训。不是新测试benchmark。', '',
           '|版本|种子|状态|新增步数|所选步|验证Root双捕|训练探针Root双捕|验证固定D双捕|',
           '|---|---:|---|---:|---:|---:|---:|---:|']
    for variant in cfg['variants']:
        for seed in cfg['seeds']:
            f=root/'runs'/f'{variant}_{seed}'
            status=j.read(f/'status.json') if (f/'status.json').exists() else {}
            chosen=j.read(f/'selected.json') if (f/'selected.json').exists() else None
            r=dict(variant=variant,seed=seed,status=status.get('stage','queued'),steps=status.get('steps',0),selected=chosen)
            records.append(r)
            if chosen:
                lines.append(f'|{variant}|{seed}|{r["status"]}|{r["steps"]}|{chosen["steps"]}|{chosen["validation"]["rootpuct"]["double_capture"]:.2%}|{chosen["train_probe"]["rootpuct"]["double_capture"]:.2%}|{chosen["validation"]["fixed_D_kernel"]["double_capture"]:.2%}|')
            else:lines.append(f'|{variant}|{seed}|{r["status"]}|{r["steps"]}|—|—|—|—|')
    complete=all(r['status']=='complete' for r in records)
    result=dict(updated_at=time.strftime('%FT%T%z'),complete=complete,runs=records)
    if complete:
        gates={}
        for variant in cfg['variants']:
            rr=[r['selected'] for r in records if r['variant']==variant]
            gates[variant]=dict(train_gate=all(r['train_probe']['rootpuct']['double_capture']>=.5 for r in rr),
                validation_gate=float(np.mean([r['validation']['rootpuct']['double_capture'] for r in rr]))>=.25)
        result['capability_gates']=gates
        lines+=['','开发门槛只判断主任务能力，不是强MARL验收或收敛证明。', '',json.dumps(gates,ensure_ascii=False)]
    lines+=['','[注册配置](experiment.json) · [协议](PROTOCOL.md) · [机器状态](review_status.json)',
            '', '每捕获奖励会改变未双捕轨迹的训练目标；最终验证仍按原始效用选模。没有人为分配目标或角色。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');j.save(root/'review_status.json',result)
    return result


def pipeline(args):
    root=Path(args.output);cfg=register(root,args.smoke)
    lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpus=j.choose_cpus(4*(args.workers+1));assert len(cpus)==4*(args.workers+1)
    children=[];execution=[];started=time.time()
    try:
        for index,(variant,seed) in enumerate((v,s) for v in cfg['variants'] for s in cfg['seeds']):
            f=root/'runs'/f'{variant}_{seed}';f.mkdir(parents=True,exist_ok=True)
            if (f/'decision.json').exists():continue
            use=cpus[index*(args.workers+1):(index+1)*(args.workers+1)]
            device=('cuda:2','cuda:3')[index%2] if args.device=='cuda' else 'cpu'
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'train','--output',str(root),
                     '--variant',variant,'--seed',str(seed),'--device',device,'--cpus',','.join(map(str,use)),
                     '--parent',str(os.getpid())]
            stream=(f/'run.log').open('a');child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT);stream.close()
            children.append(child);execution.append(dict(variant=variant,seed=seed,pid=child.pid,cpus=use,device=device))
        j.save(root/'execution.json',dict(runs=execution,other_jobs_terminated=False,physical_core_count=len(cpus)))
        j.common.emit(root,'running',runs=execution)
        while any(c.poll() is None for c in children):
            report(root)
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('A K1 trainer failed; see preserved run log')
            if time.time()-started>8*3600:raise RuntimeError('Registered 8h wall-time cap reached')
            time.sleep(15)
        if any(c.returncode!=0 for c in children):raise RuntimeError('A K1 trainer failed')
        verify(root);result=report(root);assert result['complete']
        j.save(root/'decision.json',result);j.common.emit(root,'complete',wall_seconds=time.time()-started)
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
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['register','pipeline','train','report'])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT);parser.add_argument('--variant',choices=VARIANTS)
    parser.add_argument('--seed',type=int,default=17);parser.add_argument('--device',default='cuda')
    parser.add_argument('--cpus');parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--parent',type=int);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.parent:
        ctypes.CDLL(None).prctl(1,signal.SIGTERM)
        if os.getppid()!=args.parent:raise RuntimeError('Coordinator exited before trainer startup')
    if args.command=='register':register(args.output,args.smoke)
    elif args.command=='pipeline':pipeline(args)
    elif args.command=='train':train(args)
    else:print(json.dumps(report(args.output),ensure_ascii=False))


if __name__=='__main__':main()
