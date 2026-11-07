"""Author HASAC with a fixed task curriculum; algorithm/update unchanged.

Isolates cold-start exploration from algorithm suitability. Every phase uses
actual sensor observations, no assignment, no expert actions, and fresh replay
after a change of opponent dynamics. Original M remains the from-scratch arm.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
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
import run_e45m_hasac_task as m

j=m.j
ROOT=m.ROOT
OUT=ROOT/'current/e45n_hasac_curriculum_20261101'
PHASES=(dict(name='static_visible',variant='static',opponent='fixed_D_kernel',steps=20000),
        dict(name='moving_visible',variant='moving',opponent='fixed_D_kernel',steps=30000),
        dict(name='hidden_root',variant=None,opponent='rootpuct',steps=100000))


def phase_entry(entry,phase):
    variant=phase['variant']
    return entry if variant is None else entry|dict(id=entry['id']+'_'+variant,path=entry['curriculum'][variant],suite='curriculum')


def gate_passed(evaluation,smoke=False):
    d=evaluation['deterministic']
    return bool(smoke or d['double']/d['episodes']>=.8)


def register(root,smoke=False):
    root=Path(root)
    if (root/'experiment.json').exists():return verify(root)
    source=m.verify(m.OUT);root.mkdir(parents=True,exist_ok=True)
    entries=source['entries'][:1] if smoke else source['entries']
    phases=[p|dict(steps=1 if smoke else p['steps']) for p in PHASES]
    files=[Path(__file__).resolve(),ROOT/'code/tests/test_e45n_hasac_curriculum.py',
           ROOT/'code/E45N_HASAC_CURRICULUM_PROTOCOL_20261101.md']
    cfg={k:v for k,v in source.items() if k not in ('source_hashes','protected_hashes','bank_hashes')}
    protected=dict(source['source_hashes']);protected.update(source['protected_hashes'])
    bank={phase_entry(e,p)['path']:j.sha(phase_entry(e,p)['path']) for e in entries for p in phases}
    cfg.update(experiment='E45-N',scope='author HASAC task curriculum development, not a new algorithm or final benchmark',
        registered_at=time.strftime('%FT%T%z'),smoke=smoke,entries=entries,seeds=[17] if smoke else [17,29],
        phases=phases,steps_per_model=sum(p['steps'] for p in phases),
        batch_episodes=2 if smoke else 12,replay_capacity=512 if smoke else 50000,
        update_batch=32 if smoke else 256,warmup_steps=0 if smoke else 2000,
        phase_evaluation_every=10000,action_repeats=1 if smoke else 2,
        replay_reset_at_every_phase=True,optimizer_reset_at_phase=False,smoke_gate_bypass=smoke,
        gate='phase-end deterministic >=80%; visible gate failure stops that seed, not a software failure',
        hidden_probe='same original hidden6 training situations at every checkpoint, deterministic; no selection',
        budget_disclosure='up to50k visible course +100k hidden Root per model, versus M100k hidden only',
        maximum_wall_seconds=12*3600,parent_M_experiment_sha256=j.sha(m.OUT/'experiment.json'),
        source_hashes={str(p):j.sha(p) for p in files},protected_hashes=protected,bank_hashes=bank)
    j.save(root/'experiment.json',cfg)
    for p in files:
        dst=root/'source_snapshot'/p.name;dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,dst)
    return verify(root)


def verify(root):
    cfg=j.read(Path(root)/'experiment.json')
    for section in ('source_hashes','protected_hashes','upstream_hashes','bank_hashes'):
        for path,digest in cfg[section].items():
            if j.sha(path)!=digest:raise RuntimeError('Frozen artifact changed: '+path)
    return cfg


def task_for(entry,world,env,seed,phase,mode,action_seed,tag,uniform=False):
    e=phase_entry(entry,phase)
    t=j.common.request(e,world,phase['opponent'],'HASAC',seed,'hasac_N_'+tag,episode_seed=env)
    t.update(method='HASAC_Curriculum'+str(seed),execution_mode=mode,policy_seed=action_seed,uniform=uniform)
    t['key']+=f'_N{seed}_{phase["name"]}_{mode}_a{action_seed}_{tag}'
    return t


def evaluate(pool,cfg,seed,model,total_steps,phase_index,phase_steps):
    weights=m.actor_weights(model);phase=cfg['phases'][phase_index]
    result=dict(phase_index=phase_index,phase=phase['name'],steps=total_steps,phase_steps=phase_steps)
    for mode in ('deterministic','stochastic'):
        repeats=1 if mode=='deterministic' else cfg['action_repeats']
        jobs=[dict(actors=weights,training=False,task=task_for(e,w,env,seed,phase,mode,
              m.l.seed_for('N_eval',seed,e['id'],w,r),f'eval_{total_steps}_p{phase_index}'))
              for e,w,env in m.l.situations(cfg['entries']) for r in range(repeats)]
        result[mode]=m.l.score([x['summary'] for x in pool.map(m.rollout,jobs)])
    if phase['variant'] is None:result['hidden_probe']=dict(result['deterministic'])
    else:
        hidden=cfg['phases'][-1]
        jobs=[dict(actors=weights,training=False,task=task_for(e,w,env,seed,hidden,'deterministic',
             m.l.seed_for('N_hidden_probe',seed,e['id'],w),f'probe_{total_steps}_p{phase_index}'))
             for e,w,env in m.l.situations(cfg['entries'])]
        result['hidden_probe']=m.l.score([x['summary'] for x in pool.map(m.rollout,jobs)])
    return result


def checkpoint(folder,model,replay,steps,iteration,history,cfg_sha,phase_index):
    # Separate phase directories preserve equal-step endpoints and next phase
    # starts as distinct snapshots. The top pointer commits one full bundle.
    phase_folder=folder/'phase_snapshots'/f'phase_{phase_index}'
    m.checkpoint(phase_folder,model,replay,steps,iteration,history,cfg_sha)
    pointer=j.read(phase_folder/'latest.json')|dict(phase_index=phase_index,
        phase_steps=history[-1]['phase_steps'],replay_rng=replay.rng.bit_generator.state)
    j.save(folder/'latest.json',pointer)


def train(args):
    root=Path(args.output);cfg=verify(root);folder=root/'runs'/f'HASAC_Curriculum_{args.seed}';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cores=list(map(int,args.cpus.split(',')));os.sched_setaffinity(0,{cores[-1]});torch.set_num_threads(1)
    torch.cuda.set_device(args.device);m.configure(folder)
    model=m.AuthorHASAC(obs_dim=845,device=args.device,seed=args.seed,retirement_mask=True)
    replay=m.Replay(cfg['replay_capacity'],args.seed);cfg_sha=j.sha(root/'experiment.json')
    steps=iteration=0;history=[];start_phase=0;resume_phase_steps=0
    if (folder/'latest.json').exists():
        steps,iteration,history=m.restore(folder,model,replay,cfg_sha)
        start_phase=history[-1]['phase_index'];resume_phase_steps=history[-1]['phase_steps']
        # The common bridge has no replay arrays to load at an empty phase
        # start; restore its RNG explicitly from the committed top pointer.
        replay.rng.bit_generator.state=j.read(folder/'latest.json')['replay_rng']
    pool,q=m.pool_for(folder,cores[:-1]);started=time.time();outcome='complete'
    gates=j.read(folder/'gates.json') if (folder/'gates.json').exists() else []
    gates=[g for g in gates if g['phase']<start_phase]
    try:
        for index in range(start_phase,len(cfg['phases'])):
            phase=cfg['phases'][index];phase_steps=resume_phase_steps if index==start_phase else 0
            if not history or history[-1]['phase_index']!=index:
                # Dynamics have changed. Retain learned actor/critic and
                # optimizers, never mix old-MDP samples into the new replay.
                replay=m.Replay(cfg['replay_capacity'],m.l.seed_for('N_replay',args.seed,index))
                j.common.emit(folder,'initial_phase_evaluation',steps=steps,phase=index,phase_steps=0)
                history.append(evaluate(pool,cfg,args.seed,model,steps,index,0));j.save(folder/'evaluation.json',history)
                checkpoint(folder,model,replay,steps,iteration,history,cfg_sha,index)
            initial_gradient_updates=model.runner.total_it
            # Resumed update target is relative to the current replay phase.
            cases=m.l.situations(cfg['entries'])
            while phase_steps<phase['steps']:
                begin=time.time();weights=m.actor_weights(model)
                order=np.tile(np.arange(len(cases)),int(np.ceil(cfg['batch_episodes']/len(cases))))[:cfg['batch_episodes']]
                np.random.default_rng(m.l.seed_for('N_order',args.seed,iteration)).shuffle(order)
                jobs=[]
                for i,s in enumerate(order):
                    e,w,env=cases[s]
                    task=task_for(e,w,env,args.seed,phase,'stochastic',m.l.seed_for('N_actions',args.seed,iteration,i),
                                  f'train_iter{iteration}_i{i}',uniform=steps<cfg['warmup_steps'])
                    jobs.append(dict(actors=weights,task=task,training=True))
                rows=list(pool.map(m.rollout,jobs));before=phase_steps;new_steps=0
                for row in rows:replay.add(row['arrays']);new_steps+=len(row['arrays']['obs'])
                steps+=new_steps;phase_steps+=new_steps
                if cfg['smoke']:target=initial_gradient_updates+cfg['smoke_updates']
                else:
                    # Count global warmup once, not again after every switch.
                    warmup_before=max(0,cfg['warmup_steps']-(steps-new_steps))
                    warmup_now=min(new_steps,warmup_before)
                    target=model.runner.total_it+new_steps-warmup_now
                while model.runner.total_it<target:model.update(replay.sample(cfg['update_batch']))
                iteration+=1
                record=dict(steps=steps,phase_index=index,phase=phase['name'],phase_steps=phase_steps,
                            iteration=iteration,gradient_updates=model.runner.total_it,replay_size=replay.size,
                            replay_resamples=replay.resamples,episodes=[r['summary'] for r in rows],
                            score=m.l.score([r['summary'] for r in rows]),seconds=time.time()-begin)
                j.save(folder/'updates'/f'{iteration:05d}.json',record)
                if before//cfg['phase_evaluation_every']!=phase_steps//cfg['phase_evaluation_every'] or phase_steps>=phase['steps']:
                    j.common.emit(folder,'evaluation',steps=steps,phase=index,phase_steps=phase_steps)
                    history.append(evaluate(pool,cfg,args.seed,model,steps,index,phase_steps));j.save(folder/'evaluation.json',history)
                    checkpoint(folder,model,replay,steps,iteration,history,cfg_sha,index)
                j.common.emit(folder,'training',steps=steps,phase=index,phase_steps=phase_steps,
                              gradient_updates=model.runner.total_it,score=record['score'])
            passed=gate_passed(history[-1],cfg['smoke'])
            gates.append(dict(phase=index,name=phase['name'],passed=passed,smoke_bypass=cfg['smoke'],
                              deterministic=history[-1]['deterministic']))
            j.save(folder/'gates.json',gates)
            if phase['variant'] is not None and not passed:
                outcome='gate_failed';break
            if index<len(cfg['phases'])-1:
                j.common.emit(folder,'phase_advanced',steps=steps,phase=index+1,replay_will_reset=True)
        verify(root);j.save(folder/'decision.json',dict(status=outcome,steps=steps,gates=gates,history=history,
             gradient_updates=model.runner.total_it,wall_seconds=time.time()-started,convergence_claim=False))
        j.common.emit(folder,outcome,steps=steps,phase=history[-1]['phase_index'])
    except BaseException as exc:
        j.common.emit(folder,'failed',steps=steps,error=repr(exc),traceback=traceback.format_exc());raise
    finally:pool.shutdown(wait=True,cancel_futures=True);q.close();lock.close()


def report(root):
    root=Path(root);cfg=j.read(root/'experiment.json')
    status=j.read(root/'status.json') if (root/'status.json').exists() else {}
    lines=['E45-N 作者HASAC：真实可见→隐藏课程',time.strftime('%FT%T%z'),
           '同作者HASAC与J奖励；无任务分配；动力学换段后清空回放',
           '协调器：'+status.get('stage','starting'),'',
           '种子 | 阶段 | 全部新增步 | 阶段步 | 最新阶段双捕 | 原隐藏任务探针']
    for seed in cfg['seeds']:
        f=root/'runs'/f'HASAC_Curriculum_{seed}'
        s=j.read(f/'status.json') if (f/'status.json').exists() else {}
        h=j.read(f/'evaluation.json') if (f/'evaluation.json').exists() else []
        tail='等待初评'
        if h:
            d=h[-1]['deterministic'];p=h[-1]['hidden_probe'];tail=f'{d["double"]}/{d["episodes"]} | {p["double"]}/{p["episodes"]}'
        lines.append(f'{seed} | {s.get("stage","queued")} (phase{s.get("phase","—")}) | {s.get("steps",0)} | {s.get("phase_steps","—")} | {tail}')
    lines+=['','课程额外最多50k/模型，之后隐藏Root100k/模型；预算与M分开披露。',
            '可见阶段门槛失败就停止该种子，保留失败，不虚报进入主任务。',str(root)]
    temp=root/'LIVE.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(root/'LIVE.txt')


def parent_m_live():
    p=m.OUT/'status.json'
    if not p.exists():return False
    cmd=Path('/proc/'+str(j.read(p).get('pid'))+'/cmdline')
    return cmd.exists() and b'run_e45m_hasac_task.py pipeline' in cmd.read_bytes().replace(b'\0',b' ')


def pipeline(args):
    root=Path(args.output);cfg=register(root,args.smoke)
    lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    children=[];started=time.time()
    try:
        if args.wait_for_m:
            j.common.emit(root,'waiting_for_M_resource_release')
            while parent_m_live():
                report(root)
                if time.time()-started>cfg['maximum_wall_seconds']:raise RuntimeError('Resource wait exceeded cap')
                time.sleep(10)
            if j.read(m.OUT/'status.json')['stage']!='complete':raise RuntimeError('M did not complete; inspect software before continuing')
        cores=j.choose_cpus(len(cfg['seeds'])*(args.workers+1))
        for i,seed in enumerate(cfg['seeds']):
            f=root/'runs'/f'HASAC_Curriculum_{seed}';f.mkdir(exist_ok=True,parents=True)
            if (f/'decision.json').exists():continue
            use=cores[i*(args.workers+1):(i+1)*(args.workers+1)]
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'train','--output',str(root),
                 '--seed',str(seed),'--device','cuda:1','--cpus',','.join(map(str,use)),'--parent',str(os.getpid())]
            with (f/'run.log').open('a') as log:children.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT))
        j.save(root/'execution.json',dict(pids=[c.pid for c in children],cores=cores))
        j.common.emit(root,'running',pids=[c.pid for c in children])
        while any(c.poll() is None for c in children):
            report(root)
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Curriculum trainer software failure')
            if time.time()-started>cfg['maximum_wall_seconds']:raise RuntimeError('Registered wall cap reached')
            time.sleep(10)
        assert all(c.returncode==0 for c in children);verify(root)
        decisions=[j.read(root/'runs'/f'HASAC_Curriculum_{seed}'/'decision.json') for seed in cfg['seeds']]
        j.save(root/'decision.json',dict(status='complete',runs=decisions,wall_seconds=time.time()-started))
        j.common.emit(root,'complete',outcomes=[d['status'] for d in decisions]);report(root)
    except BaseException as exc:
        j.common.emit(root,'failed',error=repr(exc),traceback=traceback.format_exc());report(root);raise
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
    p.add_argument('--workers',type=int,default=8);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--device',default='cuda:1');p.add_argument('--cpus');p.add_argument('--parent',type=int)
    p.add_argument('--wait-for-m',action='store_true');args=p.parse_args()
    if args.parent:
        ctypes.CDLL(None).prctl(1,signal.SIGTERM)
        if os.getppid()!=args.parent:raise RuntimeError('Coordinator exited')
    if args.command=='register':register(args.output,args.smoke)
    elif args.command=='pipeline':pipeline(args)
    elif args.command=='train':train(args)
    else:report(args.output)


if __name__=='__main__':main()
