"""E45-O: full-trajectory public-search demonstrations in author HASAC replay."""
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
import run_e45m_hasac_task as m
from learning.e45o_demonstration_replay import DemonstrationReplay
from learning.e45g_ppo import safe_action
from learning.e45m_author_hasac import AuthorHASAC,UPSTREAM
from learning.e45m_task_adapter import Controller,Replay,episode_arrays
from e45g_training_acceleration import public_geometry_cache

j=l.j
ROOT=l.ROOT
OUT=ROOT/'current/e45o_hasac_demonstration_20261101'
FOLDER=None
WORKER_MODEL=None
CURRENT=None


def register(root,smoke=False):
    root=Path(root)
    if (root/'experiment.json').exists():return verify(root)
    root.mkdir(parents=True,exist_ok=True)
    parent=m.verify(m.OUT)
    cfg=dict(parent)
    cfg.update(experiment='E45-O', registered_at=time.strftime('%FT%T%z'), smoke=smoke,
        scope='protected GEOM_SEARCH demonstration replay vs original HASAC; training-task pilot only',
        entries=parent['entries'][:1] if smoke else parent['entries'],
        arms=['baseline','guided'], steps_per_model=1 if smoke else 50000,
        batch_episodes=2 if smoke else 12, evaluation_every=10000,
        action_repeats=1 if smoke else 2, warmup_steps=0 if smoke else 2000,
        replay_capacity=256 if smoke else 50000, update_batch=32 if smoke else 256,
        demonstration_fraction=.25, maximum_wall_seconds=12*3600,
        selection='none: report every fixed training evaluation and final checkpoint',
        teacher='all existing L GEOM_SEARCH controls for the selected training entries, no success filtering',
        budget='50k total transitions/model: baseline online; guided online + historical demonstration transitions, complete batch overshoot reported',
        algorithm='unchanged HARL HASAC losses and sequential updates; guided changes replay distribution only')
    manifest=prepare_demonstrations(root,cfg)
    cfg['demonstration_steps']=manifest['transitions']
    cfg['demonstration_sha256']=j.sha(root/'demonstrations/data.npz')
    cfg['demonstration_manifest_sha256']=j.sha(root/'demonstrations/manifest.json')
    protected=dict(parent['protected_hashes']);protected.update(parent['source_hashes'])
    cfg['protected_hashes']=protected
    files=[Path(__file__).resolve(),ROOT/'code/learning/e45o_demonstration_replay.py',
           ROOT/'code/tests/test_e45o_demonstration_replay.py',ROOT/'code/E45O_HASAC_DEMONSTRATION_PROTOCOL_20261101.md',
           ROOT/'code/audit_e45o_demonstration_smoke.py']
    cfg['source_hashes']={str(p):j.sha(p) for p in files}
    cfg['bank_hashes']={e['path']:j.sha(e['path']) for e in cfg['entries']}
    cfg['parent_M_experiment_sha256']=j.sha(m.OUT/'experiment.json')
    j.save(root/'experiment.json',cfg)
    import shutil
    for p in files:
        dst=root/'source_snapshot'/p.relative_to(ROOT/'code');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    return verify(root)


def prepare_demonstrations(root,cfg):
    folder=root/'demonstrations';folder.mkdir(exist_ok=True)
    configure(folder/'public_reconstruction')
    chosen={e['id'] for e in cfg['entries']};episodes=[];arrays=[];singles=0;max_error=0.
    source=l.OUT/'positive_control/evaluation'
    for path in sorted((source/'cache').glob('*.json')):
        row=j.read(path)
        if row['group'] not in chosen:continue
        payload_path=source/'episodes'/path.name
        payload=j.read(payload_path);case=j.read(row['case'])
        assert payload['summary']==row and row['split']=='train' and row['arm']=='GEOM_SEARCH'
        assert row['public_replay_exact'] and row['reason'] in ('timeout','double_capture')
        assert not any(t['stats'].get('assignment_used',False) or t['stats'].get('roles_used',False) for t in payload['trace'])
        own=Controller(case['map'],case['speed'],row['seed'],actors=None,uniform=True)
        for obs in case['history'][:-1]:own.observe(obs)
        with public_geometry_cache():
            for t in payload['trace']:
                own.act(t['observation'])
                assert own.belief.signature()==t['belief_signature']
                # Explicit canonical legal intent, NOT a claim to recover a
                # search controller's pre-projection candidate. Verify remap.
                intent=(np.asarray(t['action'])/case['speed']).astype(np.float32)
                physical,changed=safe_action(np.asarray(t['observation']['positions']),intent,
                    t['observation']['active_p'],case['speed'],np.asarray(case['map']))
                error=float(np.max(np.abs(physical-np.asarray(t['action']))));max_error=max(max_error,error)
                if error>2e-6:raise RuntimeError('Teacher canonical intent failed physical remap: '+str(error))
                own.samples[-1].update(action=intent,blocked=float(changed/max(1,sum(t['observation']['active_p']))))
        data=episode_arrays(own.samples,payload['trace'])
        for i,t in enumerate(payload['trace'][:-1]):
            if t['new_captures']==1:
                assert data['done'][i,0]==0 and data['active'][i+1].sum()==1
                singles+=1
        archive=folder/(row['group']+'_'+row['world']+'.npz');np.savez_compressed(archive,**data)
        arrays.append(data)
        episodes.append(dict(group=row['group'],world=row['world'],seed=row['seed'],
            case=row['case'],case_sha256=j.sha(row['case']),source=str(payload_path),source_sha256=j.sha(payload_path),
            archive=str(archive),archive_sha256=j.sha(archive),transitions=len(data['obs']),
            double_capture=row['double_capture'],teacher_online_ms=sum(row['latencies_ms']),
            simulated_transitions=row['simulated_transitions']))
    assert {(e['group'],e['world']) for e in episodes}=={(e['id'],w) for e in cfg['entries'] for w in ('AB','BA')}
    np.savez_compressed(folder/'data.npz',**{key:np.concatenate([a[key] for a in arrays]) for key in arrays[0]})
    manifest=dict(episodes=episodes,transitions=sum(e['transitions'] for e in episodes),
        first_capture_continuations=singles,maximum_physical_remap_error=max_error,
        teacher_online_ms=sum(e['teacher_online_ms'] for e in episodes),
        teacher_simulated_transitions=sum(e['simulated_transitions'] for e in episodes),
        reused_historical_training_controls=True,new_teacher_queries=0,success_filtering=False,
        intent_semantics='canonical legal physical action/speed, float32 remap checked <=2e-6',
        information='actor inputs reconstructed from public observations only; hidden coordinates in critic only')
    j.save(folder/'manifest.json',manifest)
    return manifest


def verify(root):
    cfg=j.read(Path(root)/'experiment.json')
    for section in ('source_hashes','protected_hashes','upstream_hashes','bank_hashes'):
        for path,digest in cfg[section].items():
            if j.sha(path)!=digest:raise RuntimeError('Frozen artifact changed: '+path)
    assert j.sha(Path(root)/'demonstrations/data.npz')==cfg['demonstration_sha256']
    assert j.sha(Path(root)/'demonstrations/manifest.json')==cfg['demonstration_manifest_sha256']
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
    root=Path(args.output);cfg=verify(root);folder=root/'runs'/f'{args.arm}_{args.seed}';folder.mkdir(exist_ok=True,parents=True)
    lock=(folder/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cores=list(map(int,args.cpus.split(',')));os.sched_setaffinity(0,{cores[-1]});torch.set_num_threads(1)
    torch.cuda.set_device(args.device);configure(folder)
    model=AuthorHASAC(obs_dim=845,device=args.device,seed=args.seed,retirement_mask=True)
    cfg_sha=j.sha(root/'experiment.json')
    demo_steps=cfg['demonstration_steps'] if args.arm=='guided' else 0
    if args.arm=='guided':
        with np.load(root/'demonstrations/data.npz') as data:
            demos={key:data[key].copy() for key in data.files}
        replay=DemonstrationReplay(demos,cfg['replay_capacity'],args.seed,cfg['demonstration_fraction'])
    else:replay=Replay(cfg['replay_capacity'],args.seed)
    online_limit=cfg['steps_per_model'] if cfg['smoke'] else cfg['steps_per_model']-demo_steps
    steps=iteration=0;history=[]
    if (folder/'latest.json').exists():steps,iteration,history=restore(folder,model,replay,cfg_sha)
    pool,q=pool_for(folder,cores[:-1]);started=time.time()
    try:
        if not history:
            j.common.emit(folder,'initial_evaluation',steps=0)
            history.append(evaluate(pool,cfg,args.seed,model,0));j.save(folder/'evaluation.json',history)
            checkpoint(folder,model,replay,steps,iteration,history,cfg_sha)
        situations=l.situations(cfg['entries'])
        while steps<online_limit:
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
            target_updates=max(0,steps+demo_steps-cfg['warmup_steps'])*cfg['updates_per_transition']
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
            record=dict(steps=steps,total_accounted_steps=steps+demo_steps,demonstration_steps=demo_steps,arm=args.arm,iteration=iteration,gradient_updates=model.runner.total_it,
                        replay_size=replay.size,replay_resamples=replay.resamples,
                        episodes=rows,score=l.score(rows),q=qdiag,seconds=time.time()-begin)
            j.save(folder/'updates'/f'{iteration:05d}.json',record)
            if before//cfg['evaluation_every']!=steps//cfg['evaluation_every'] or steps>=online_limit:
                j.common.emit(folder,'evaluation',steps=steps,gradient_updates=model.runner.total_it)
                history.append(evaluate(pool,cfg,args.seed,model,steps));j.save(folder/'evaluation.json',history)
                checkpoint(folder,model,replay,steps,iteration,history,cfg_sha)
            j.common.emit(folder,'training',steps=steps,iteration=iteration,gradient_updates=model.runner.total_it,score=record['score'],q=qdiag)
        verify(root);j.save(folder/'decision.json',dict(status='complete',steps=steps,total_accounted_steps=steps+demo_steps,demonstration_steps=demo_steps,arm=args.arm,history=history,
             gradient_updates=model.runner.total_it,wall_seconds=time.time()-started,convergence_claim=False))
        j.common.emit(folder,'complete',steps=steps)
    except BaseException as exc:
        j.common.emit(folder,'failed',steps=steps,error=repr(exc),traceback=traceback.format_exc());raise
    finally:pool.shutdown(wait=True,cancel_futures=True);q.close();lock.close()


def report(root):
    root=Path(root);cfg=j.read(root/'experiment.json')
    status=j.read(root/'status.json') if (root/'status.json').exists() else {}
    lines=['E45-O 示范回放 HASAC 对照（纯Actor评估）',time.strftime('%FT%T%z'),
        '状态：'+status.get('stage','starting'),
        f"教师：{cfg['demonstration_steps']}条转移；guided每批25%；教师成本计入预算",'',
        '组别/种子 | 状态 | 在线步 | 更新数 | 最近确定性双捕 | 最近随机双捕']
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            f=root/'runs'/f'{arm}_{seed}';s=j.read(f/'status.json') if (f/'status.json').exists() else {}
            h=j.read(f/'evaluation.json') if (f/'evaluation.json').exists() else []
            tail='尚无评估'
            if h:
                a=h[-1]['deterministic'];b=h[-1]['stochastic']
                tail=f"{a['double']}/{a['episodes']} | {b['double']}/{b['episodes']}"
            lines.append(f"{arm}/{seed} | {s.get('stage','queued')} | {s.get('steps',0)} | {s.get('gradient_updates','—')} | {tail}")
    lines+=['','全部是固定训练情形的开发评估，不能作为泛化结论。',str(root)]
    tmp=root/'LIVE.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(root/'LIVE.txt')


def pipeline(args):
    root=Path(args.output);cfg=register(root,args.smoke)
    lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    children=[];started=time.time()
    try:
        combinations=[(arm,seed) for arm in cfg['arms'] for seed in cfg['seeds']]
        cores=j.choose_cpus(len(combinations)*(args.workers+1))
        if len(cores)!=len(combinations)*(args.workers+1):raise RuntimeError('Insufficient physical CPU cores')
        for i,(arm,seed) in enumerate(combinations):
            f=root/'runs'/f'{arm}_{seed}';f.mkdir(parents=True,exist_ok=True)
            if (f/'decision.json').exists():continue
            use=cores[i*(args.workers+1):(i+1)*(args.workers+1)]
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'train','--output',str(root),
                 '--seed',str(seed),'--arm',arm,'--device',args.device,'--cpus',','.join(map(str,use)),'--parent',str(os.getpid())]
            with (f/'run.log').open('a') as log:children.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT))
        j.save(root/'execution.json',dict(pids=[p.pid for p in children],cores=cores,device=args.device,combinations=combinations))
        j.common.emit(root,'running',pids=[p.pid for p in children])
        while any(c.poll() is None for c in children):
            report(root)
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('An E45-O trainer failed; inspect preserved logs')
            if time.time()-started>cfg['maximum_wall_seconds']:raise RuntimeError('Registered wall cap reached')
            time.sleep(10)
        assert all(c.returncode==0 for c in children);verify(root)
        result=dict(status='complete',wall_seconds=time.time()-started,runs={})
        for arm,seed in combinations:result['runs'][f'{arm}_{seed}']=j.read(root/'runs'/f'{arm}_{seed}'/'decision.json')
        j.save(root/'decision.json',result);j.common.emit(root,'complete');report(root)
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
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--arm',choices=['baseline','guided'],default='guided')
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
