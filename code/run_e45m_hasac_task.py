"""Author HASAC on the real public-belief pursuit task, frozen development run."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
import fcntl
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import signal
import sys
import time
import traceback
import numpy as np
import torch
import run_e45l_marl_diagnosis as l
from learning.e45m_author_hasac import AuthorHASAC,UPSTREAM
from learning.e45m_task_adapter import Controller,Replay,episode_arrays
from e45g_training_acceleration import public_geometry_cache

j=l.j
ROOT=l.ROOT
OUT=ROOT/'current/e45m_hasac_task_20261101'
FOLDER=None
WORKER_MODEL=None
CURRENT=None


def register(root,smoke=False):
    root=Path(root)
    if (root/'experiment.json').exists():return verify(root)
    root.mkdir(parents=True,exist_ok=True)
    previous=l.verify(l.OUT)
    entries=previous['entries'][:1] if smoke else previous['entries']
    provenance=j.read(ROOT/'current/e45_marl_macro_audit_20261101/harl_upstream.json')
    files=[Path(__file__).resolve(),ROOT/'code/learning/e45m_author_hasac.py',
           ROOT/'code/learning/e45m_task_adapter.py',ROOT/'code/tests/test_e45m_hasac_task.py',
           ROOT/'code/E45M_HASAC_TASK_PROTOCOL_20261101.md']
    protected=dict(previous['source_hashes']);protected.update(previous['protected_hashes'])
    upstream=dict(provenance['source_hashes'])
    yaml_path=UPSTREAM/'harl/configs/algos_cfgs/hasac.yaml';upstream[str(yaml_path)]=j.sha(yaml_path)
    cfg=dict(experiment='E45-M',scope='author HASAC hidden-task learning development, not independent benchmark',
             registered_at=time.strftime('%FT%T%z'),smoke=smoke,entries=entries,seeds=[17,29],
             steps_per_model=1 if smoke else 100000,batch_episodes=2 if smoke else 12,
             evaluation_every=20000,action_repeats=1 if smoke else 2,warmup_steps=0 if smoke else 2000,
             replay_capacity=256 if smoke else 50000,update_batch=32 if smoke else 256,
             updates_per_transition=1,smoke_updates=2,actor_obs=845,critic_obs=858,
             architecture='two independent author feedforward 256x256 squashed Gaussian actors, shared twin Q; explicit public history in input',
             algorithm='HARL HASAC unchanged sequential actor and twin critic update',
             upstream_url=provenance['url'],upstream_commit=provenance['commit'],
             lr=5e-4,critic_lr=5e-4,alpha=.001,auto_alpha=False,polyak=.005,
             gamma=1.,n_step=1,use_valuenorm=False,use_proper_time_limits=False,
             reward='original J: raw double capture utility + terminal-zero public potential - .05 blocked intent',
             lifecycle='retired actor action/entropy zero; active actor mask; first capture does not terminate team',
             replay_action='tanh desired normalized action before fixed physical safety mapping',
             selection='none: all scheduled training-only evaluations and terminal checkpoint reported',
             target=.8,target_scope='at least5/6 deterministic training situations per seed, stability needs confirmation',
             held_out_test_used=False,validation_used=False,maximum_wall_seconds=8*3600,
             source_hashes={str(p):j.sha(p) for p in files},protected_hashes=protected,
             upstream_hashes=upstream,bank_hashes={e['path']:j.sha(e['path']) for e in entries},
             parent_L_experiment_sha256=j.sha(l.OUT/'experiment.json'))
    j.save(root/'experiment.json',cfg)
    import shutil
    for p in files:
        dst=root/'source_snapshot'/p.relative_to(ROOT/'code');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    return verify(root)


def verify(root):
    cfg=j.read(Path(root)/'experiment.json')
    for section in ('source_hashes','protected_hashes','upstream_hashes','bank_hashes'):
        for path,digest in cfg[section].items():
            if j.sha(path)!=digest:raise RuntimeError('Frozen artifact changed: '+path)
    return cfg


def configure(folder):
    global FOLDER
    FOLDER=Path(folder);l.k.configure(folder);j.common.base.make_controller=factory


def factory(case,task):
    global CURRENT
    if task['arm']!='HASAC':raise ValueError('Unexpected controller request')
    CURRENT=Controller(case['map'],case['speed'],task['seed'],WORKER_MODEL.actor,
                       stochastic=task['execution_mode']=='stochastic',uniform=task.get('uniform',False),
                       policy_seed=task['policy_seed'])
    return CURRENT


def init_worker(folder,queue,parent):
    global WORKER_MODEL
    ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    if os.getppid()!=parent:raise RuntimeError('Trainer exited')
    os.sched_setaffinity(0,{queue.get()});torch.set_num_threads(1)
    configure(folder);j.common.base.PARENT=parent
    WORKER_MODEL=AuthorHASAC(obs_dim=845,device='cpu',seed=0,retirement_mask=True)


def pool_for(folder,cores):
    ctx=mp.get_context('spawn');q=ctx.Queue()
    for c in cores:q.put(c)
    return ProcessPoolExecutor(max_workers=len(cores),mp_context=ctx,initializer=init_worker,
        initargs=(str(folder),q,os.getpid())),q


def task_for(entry,world,env_seed,seed,mode,policy_seed,tag,uniform=False):
    task=j.common.request(entry,world,'rootpuct','HASAC',seed,'hasac_'+tag,episode_seed=env_seed)
    task.update(method='HASAC'+str(seed),execution_mode=mode,policy_seed=policy_seed,uniform=uniform)
    task['key']+=f'_HASAC_{seed}_{mode}_a{policy_seed}_{tag}'
    return task


def rollout(job):
    task=job['task']
    try:
        for actor,weights in zip(WORKER_MODEL.actor,job['actors']):actor.actor.load_state_dict(weights);actor.actor.eval()
        with public_geometry_cache():payload,_=j.common.base.episode(task)
        assert payload['summary']['public_replay_exact']
        assert payload['summary']['reason'] in ('timeout','double_capture')
        stage='training' if job['training'] else 'evaluation'
        folder=FOLDER/stage;folder.mkdir(exist_ok=True,parents=True)
        path=folder/'episodes'/(task['key']+'.json.gz');path.parent.mkdir(exist_ok=True,parents=True)
        data=gzip.compress(json.dumps(payload,ensure_ascii=False,allow_nan=False).encode(),mtime=0)
        temp=path.with_suffix('.tmp');temp.write_bytes(data);temp.replace(path)
        row=payload['summary']|dict(episode=str(path),episode_sha256=j.sha(path))
        j.save(folder/'cache'/(task['key']+'.json'),row)
        arrays=episode_arrays(CURRENT.samples,payload['trace'])
        if job['training']:
            archive=folder/'replay_episodes'/(task['key']+'.npz');archive.parent.mkdir(exist_ok=True,parents=True)
            np.savez_compressed(archive,**arrays)
            row.update(replay_episode=str(archive),replay_sha256=j.sha(archive),
                       training_return=float(arrays['reward'].sum()),
                       blocked_fraction=float(np.mean([s['blocked'] for s in CURRENT.samples])))
        return dict(summary=row,arrays=arrays if job['training'] else None)
    except BaseException as exc:
        j.save(FOLDER/'failures'/(task['key']+'.json'),dict(task=task,error=repr(exc),traceback=traceback.format_exc()));raise


def actor_weights(model):
    return [{k:v.detach().cpu() for k,v in a.actor.state_dict().items()} for a in model.actor]


def evaluate(pool,cfg,seed,model,steps):
    weights=actor_weights(model);out=dict(steps=steps)
    for mode in ('deterministic','stochastic'):
        n=1 if mode=='deterministic' else cfg['action_repeats']
        jobs=[dict(actors=weights,training=False,task=task_for(e,w,env,seed,mode,
              l.seed_for('HASAC_eval',seed,e['id'],w,r),f'eval_steps{steps}'))
              for e,w,env in l.situations(cfg['entries']) for r in range(n)]
        rows=[x['summary'] for x in pool.map(rollout,jobs)];out[mode]=l.score(rows)
    return out


def checkpoint(folder,model,replay,steps,iteration,history,cfg_sha):
    path=folder/'checkpoints'/f'steps_{steps}.pt';path.parent.mkdir(exist_ok=True,parents=True)
    replay_path=path.with_suffix('.npz')
    if replay.size:replay.save(replay_path)
    c=model.critic
    state=dict(actors=[a.actor.state_dict() for a in model.actor],actor_optimizers=[a.actor_optimizer.state_dict() for a in model.actor],
        critic=c.critic.state_dict(),critic2=c.critic2.state_dict(),target_critic=c.target_critic.state_dict(),
        target_critic2=c.target_critic2.state_dict(),critic_optimizer=c.critic_optimizer.state_dict(),
        steps=steps,iteration=iteration,gradient_updates=model.runner.total_it,history=history,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(model.device),numpy_rng=np.random.get_state(),
        replay_rng=replay.rng.bit_generator.state,replay_resamples=replay.resamples,experiment_sha256=cfg_sha,
        replay_path=str(replay_path) if replay.size else None,replay_sha256=j.sha(replay_path) if replay.size else None)
    temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)
    j.save(folder/'latest.json',dict(path=str(path),sha256=j.sha(path),steps=steps))
    return state


def restore(folder,model,replay,cfg_sha):
    pointer=j.read(folder/'latest.json');assert j.sha(pointer['path'])==pointer['sha256']
    state=torch.load(pointer['path'],map_location=model.device,weights_only=False)
    assert state['experiment_sha256']==cfg_sha
    for a,w,o in zip(model.actor,state['actors'],state['actor_optimizers']):a.actor.load_state_dict(w);a.actor_optimizer.load_state_dict(o)
    for key in ('critic','critic2','target_critic','target_critic2','critic_optimizer'):getattr(model.critic,key).load_state_dict(state[key])
    model.runner.total_it=state['gradient_updates']
    torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu(),model.device)
    np.random.set_state(state['numpy_rng'])
    if state['replay_path']:
        assert j.sha(state['replay_path'])==state['replay_sha256'];replay.load(state['replay_path'],state['replay_rng'])
    replay.resamples=state['replay_resamples']
    return state['steps'],state['iteration'],state['history']


def train(args):
    root=Path(args.output);cfg=verify(root);folder=root/'runs'/f'HASAC_{args.seed}';folder.mkdir(exist_ok=True,parents=True)
    lock=(folder/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cores=list(map(int,args.cpus.split(',')));os.sched_setaffinity(0,{cores[-1]});torch.set_num_threads(1)
    torch.cuda.set_device(args.device);configure(folder)
    model=AuthorHASAC(obs_dim=845,device=args.device,seed=args.seed,retirement_mask=True)
    replay=Replay(cfg['replay_capacity'],args.seed);cfg_sha=j.sha(root/'experiment.json')
    steps=iteration=0;history=[]
    if (folder/'latest.json').exists():steps,iteration,history=restore(folder,model,replay,cfg_sha)
    pool,q=pool_for(folder,cores[:-1]);started=time.time()
    try:
        if not history:
            j.common.emit(folder,'initial_evaluation',steps=0)
            history.append(evaluate(pool,cfg,args.seed,model,0));j.save(folder/'evaluation.json',history)
            checkpoint(folder,model,replay,steps,iteration,history,cfg_sha)
        situations=l.situations(cfg['entries'])
        while steps<cfg['steps_per_model']:
            begin=time.time();weights=actor_weights(model)
            order=np.tile(np.arange(len(situations)),int(np.ceil(cfg['batch_episodes']/len(situations))))[:cfg['batch_episodes']]
            np.random.default_rng(l.seed_for('HASAC_order',args.seed,iteration)).shuffle(order)
            jobs=[]
            for i,index in enumerate(order):
                e,w,env=situations[index]
                task=task_for(e,w,env,args.seed,'stochastic',l.seed_for('HASAC_actions',args.seed,iteration,i),
                              f'train_iter{iteration}_i{i}',uniform=steps<cfg['warmup_steps'])
                jobs.append(dict(actors=weights,task=task,training=True))
            results=list(pool.map(rollout,jobs));before=steps
            for r in results:replay.add(r['arrays']);steps+=len(r['arrays']['obs'])
            target_updates=max(0,steps-cfg['warmup_steps'])*cfg['updates_per_transition']
            if cfg['smoke']:target_updates=cfg['smoke_updates']
            qdiag=None
            while model.runner.total_it<target_updates:
                batch=replay.sample(cfg['update_batch']);model.update(batch)
            if model.runner.total_it:
                with torch.no_grad():
                    batch=replay.sample(cfg['update_batch'])
                    value=model.critic.get_values(batch[0],batch[2].transpose(1,0,2).reshape(len(batch[0]),4)).cpu().numpy()
                qdiag=dict(mean=float(value.mean()),std=float(value.std()),minimum=float(value.min()),maximum=float(value.max()))
            iteration+=1;rows=[r['summary'] for r in results]
            record=dict(steps=steps,iteration=iteration,gradient_updates=model.runner.total_it,
                        replay_size=replay.size,replay_resamples=replay.resamples,
                        episodes=rows,score=l.score(rows),q=qdiag,seconds=time.time()-begin)
            j.save(folder/'updates'/f'{iteration:05d}.json',record)
            if before//cfg['evaluation_every']!=steps//cfg['evaluation_every'] or steps>=cfg['steps_per_model']:
                j.common.emit(folder,'evaluation',steps=steps,gradient_updates=model.runner.total_it)
                history.append(evaluate(pool,cfg,args.seed,model,steps));j.save(folder/'evaluation.json',history)
                checkpoint(folder,model,replay,steps,iteration,history,cfg_sha)
            j.common.emit(folder,'training',steps=steps,iteration=iteration,gradient_updates=model.runner.total_it,score=record['score'],q=qdiag)
        verify(root);j.save(folder/'decision.json',dict(status='complete',steps=steps,history=history,
             gradient_updates=model.runner.total_it,wall_seconds=time.time()-started,convergence_claim=False))
        j.common.emit(folder,'complete',steps=steps)
    except BaseException as exc:
        j.common.emit(folder,'failed',steps=steps,error=repr(exc),traceback=traceback.format_exc());raise
    finally:pool.shutdown(wait=True,cancel_futures=True);q.close();lock.close()


def report(root):
    root=Path(root);cfg=j.read(root/'experiment.json')
    state=j.read(root/'status.json') if (root/'status.json').exists() else {}
    lines=['E45-M 作者HASAC真实隐藏任务开发',time.strftime('%FT%T%z'),
           'RootPUCT3000；无目标指派；独立256x256 actor、双Q、作者顺序更新',
           '协调器：'+state.get('stage','starting'),'',
           '种子 | 阶段 | 新增步 | 梯度更新 | 最后评估步 | 确定性双捕 | 随机双捕']
    for seed in cfg['seeds']:
        f=root/'runs'/f'HASAC_{seed}';s=j.read(f/'status.json') if (f/'status.json').exists() else {}
        h=j.read(f/'evaluation.json') if (f/'evaluation.json').exists() else []
        tail='等待初评'
        if h:
            r=h[-1];a=r['deterministic'];b=r['stochastic']
            tail=f'{r["steps"]} | {a["double"]}/{a["episodes"]} | {b["double"]}/{b["episodes"]}'
        lines.append(f'{seed} | {s.get("stage","queued")} | {s.get("steps",0)} | {s.get("gradient_updates","—")} | {tail}')
    lines+=['','本轮为训练内能力开发，不是新测试benchmark；不预设胜者。',str(root)]
    temp=root/'LIVE.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(root/'LIVE.txt')


def l_is_live():
    p=l.OUT/'status.json'
    if not p.exists():return False
    pid=j.read(p).get('pid');proc=Path(f'/proc/{pid}/cmdline')
    if not proc.exists():return False
    data=proc.read_bytes().replace(b'\0',b' ')
    return b'run_e45l_marl_diagnosis.py pipeline' in data


def pipeline(args):
    root=Path(args.output);cfg=register(root,args.smoke)
    lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    children=[];started=time.time()
    try:
        if args.wait_for_l:
            j.common.emit(root,'waiting_for_L_resource_release')
            while l_is_live():
                report(root)
                if time.time()-started>cfg['maximum_wall_seconds']:raise RuntimeError('Resource wait exceeded wall cap')
                time.sleep(10)
            if j.read(l.OUT/'status.json')['stage']!='complete':raise RuntimeError('L exited without completion; inspect before expanding work')
        cores=j.choose_cpus(2*(args.workers+1))
        for i,seed in enumerate(cfg['seeds']):
            f=root/'runs'/f'HASAC_{seed}';f.mkdir(parents=True,exist_ok=True)
            if (f/'decision.json').exists():continue
            use=cores[i*(args.workers+1):(i+1)*(args.workers+1)]
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'train','--output',str(root),
                 '--seed',str(seed),'--device','cuda:1','--cpus',','.join(map(str,use)),'--parent',str(os.getpid())]
            with (f/'run.log').open('a') as log:children.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT))
        j.save(root/'execution.json',dict(pids=[p.pid for p in children],cores=cores,gpu=1))
        j.common.emit(root,'running',pids=[p.pid for p in children])
        while any(c.poll() is None for c in children):
            report(root)
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('A HASAC trainer failed; preserved logs available')
            if time.time()-started>cfg['maximum_wall_seconds']:raise RuntimeError('Registered wall cap reached')
            time.sleep(10)
        assert all(c.returncode==0 for c in children);verify(root)
        j.save(root/'decision.json',dict(status='complete',wall_seconds=time.time()-started));j.common.emit(root,'complete');report(root)
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
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','pipeline','train','report'])
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--smoke',action='store_true')
    p.add_argument('--workers',type=int,default=8);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--device',default='cuda:1');p.add_argument('--cpus');p.add_argument('--parent',type=int)
    p.add_argument('--wait-for-l',action='store_true');args=p.parse_args()
    if args.parent:
        ctypes.CDLL(None).prctl(1,signal.SIGTERM)
        if os.getppid()!=args.parent:raise RuntimeError('Coordinator exited')
    if args.command=='register':register(args.output,args.smoke)
    elif args.command=='pipeline':pipeline(args)
    elif args.command=='train':train(args)
    else:report(args.output)


if __name__=='__main__':main()
