"""Resumable engineering/pilot driver. Formal dispatch stays gated on audited pilots."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['TZ']='Asia/Shanghai'
import argparse
import collections
import concurrent.futures
import fcntl
import hashlib
import json
import multiprocessing
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'current/large_pursuit_v2_20261108'
PYTHON=sys.executable


def read(path):return json.loads(Path(path).read_text())

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));os.replace(temporary,path)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def event(out,kind,**fields):
    row=dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),event=kind,**fields)
    with open(out/'events.jsonl','a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)


def resources():
    available=os.sched_getaffinity(0);groups=collections.defaultdict(list)
    text=subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET'],text=True)
    for line in text.splitlines():
        if line.startswith('#'):continue
        cpu,core,socket=map(int,line.split(','))
        if cpu in available:groups[socket,core].append(cpu)
    physical=[sorted(v) for _,v in sorted(groups.items())]
    reserved=[cpu for group in physical[:10] for cpu in group]
    if len(reserved)<20 or len(physical)<24:raise RuntimeError('Cannot reserve required CPUs')
    training=[group[0] for group in physical[10:-4]]
    timing=[group[0] for group in physical[-4:]]
    idle=[cpu for group in physical[-4:] for cpu in group[1:]]
    bulk=[group[0] for group in physical[10:-4]]
    bulk_smt=[cpu for group in physical[10:-4] for cpu in group[1:]]
    # Use one hardware thread per physical core initially; expand only if measured faster.
    return dict(reserved=reserved,training=training,timing=timing,timing_siblings_idle=idle,
                bulk=bulk,bulk_smt=bulk_smt,gpu_preference=[3,2,1],forbidden_gpu=[0],total_logical=len(available),
                physical_cores=len(physical),one_thread_per_core=True)


def worker_init(cpus):
    os.sched_setaffinity(0,set(cpus));os.environ['CUDA_VISIBLE_DEVICES']=''
    signal.signal(signal.SIGINT,signal.SIG_IGN)


def map_job(job):
    from large_pursuit.core import generate_case
    path=Path(job['file'])
    if path.exists():return read(path)
    for attempt in range(1000):
        seed=job['seed']+attempt*10000019
        try:case=generate_case(job['n'],job['family'],seed,job['obstacles']);break
        except ValueError:continue
    else:raise RuntimeError('Map generation exhausted geometry-only attempts')
    case.update(id=job['id'],split=job['split'],suite=job['suite'],attempts=attempt+1)
    write(path,case);return {k:v for k,v in case.items() if k not in ('boxes','initials')}


def build_bank(out,layout,workers):
    from large_pursuit.core import FAMILIES
    folder=out/'bank';manifest=folder/'manifest.json'
    from large_pursuit.provenance import sha
    generation=dict(schema='large-pursuit-v2',core=sha(ROOT/'code/large_pursuit/core.py'),geometry=sha(ROOT/'code/large_pursuit/geometry.py'))
    generation_path=folder/'generation.json'
    if generation_path.exists() and read(generation_path)!=generation:raise RuntimeError('Map generator changed; use a new bank version')
    write(generation_path,generation)
    if manifest.exists():return read(manifest)
    # Engineering and pilot maps have disjoint namespaces from all formal splits.
    splits=dict(engineering=1,pilot_train=4,pilot_validation=2,train=40,validation=8,sealed_ID=8,sealed_OOD=8)
    tasks=[];entries=[];namespace=90820261108
    for n in (5,16):
        for split,per_family in splits.items():
            for index in range(per_family):
                for f,family in enumerate(FAMILIES):
                    count=(24,32,40)[index%2] if n==5 else (80,100,120)[index%2]
                    if split=='engineering':count=((24,32,40) if n==5 else (80,100,120))[f%3]
                    if split=='sealed_OOD':count=40 if n==5 else 120
                    identifier=f'{n}v{n}_{split}_{family}_{index:03d}'
                    relative=f'{n}v{n}/{split}/{identifier}.json';seed=namespace+n*1000000+list(splits).index(split)*100000+f*1000+index
                    task=dict(file=str(folder/relative),n=n,family=family,seed=seed,obstacles=count,
                              id=identifier,split=split,suite='OOD' if split=='sealed_OOD' else 'ID')
                    tasks.append(task);entries.append(dict(path=relative,n=n,split=split,id=identifier,family=family,suite=task['suite']))
    event(out,'bank_started',maps=len(tasks),workers=workers)
    context=multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=worker_init,initargs=(layout['bulk'],)) as pool:
        pending={pool.submit(map_job,job):job for job in tasks}
        completed=0
        for future in concurrent.futures.as_completed(pending):
            future.result();completed+=1
            write(out/'progress.json',dict(stage='map_generation',completed=completed,total=len(tasks),pid=os.getpid()))
            if completed%100==0:event(out,'bank_progress',completed=completed,total=len(tasks))
    geometries={};hashes={}
    for entry in entries:
        path=folder/entry['path'];case=read(path)
        key=digest(sorted(case['boxes']))
        if key in geometries:raise RuntimeError(f'Duplicate map: {entry["id"]} and {geometries[key]}')
        geometries[key]=entry['id'];entry['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();hashes[entry['id']]=entry['sha256']
    config=dict(version=2,generation=generation,entries=entries,splits_per_family=splits,namespace=namespace,
        id='obstacle counts 24/32 or 80/100, seven tiled rectangle families',
        ood='held-out density 40 or 120 obstacles across the same seven layout families',
        selection='geometry legality, exact union occupancy [2%,40%], exact continuous rectangle-cell connectivity and certified births, spawn distance >=3m only',
        test_seal='sealed_ID and sealed_OOD must not be read by calibration, teacher collection, training or checkpoint selection')
    write(manifest,config);event(out,'bank_complete',maps=len(entries));return config


def calibration_job(job):
    from large_pursuit.core import Config,World,PublicMap,from_initial
    from large_pursuit.methods import Controller,candidate_pool,Network,kernel_action
    from large_pursuit.training import teacher_labels
    from large_pursuit.motion import MotionKernel
    import torch
    case=read(job['case']);public=PublicMap(Config.scale(case['n']),case['boxes']);world=World(public.cfg,case['boxes']);state=from_initial(case,0)
    rng=np.random.default_rng(job['seed']);motion=MotionKernel(public,rng);controller=Controller(public,job['method'],job['seed'],simulations=job['simulations'],depth=8)
    rows=[]
    for _ in range(job['steps']):
        obs=world.observe(state);action=controller.act(obs);rows.append(controller.last_stats)
        state=world.transition(state,action,motion.actions(state))
        if world.done(state):break
    obs=world.observe(state);controller.belief.update(obs)
    pool=candidate_pool(public,state.p,controller.belief.estimates(),state.alive,rng)
    start=time.perf_counter();teacher_labels(public,obs,controller.belief,pool,rng,4,8);teacher_ms=1000*(time.perf_counter()-start)
    return dict(n=case['n'],method=job['method'],simulations=job['simulations'],latencies_ms=[r['elapsed_ms'] for r in rows],
                teacher_query_ms=teacher_ms,teacher_simulations=128,teacher_depth=8,
                completed_simulations=sum(r['simulations'] for r in rows),environment_team_steps=len(rows),
                elapsed_ms=sum(r['elapsed_ms'] for r in rows))


def cpu_job(job):
    if job.get('cpu') is not None:os.sched_setaffinity(0,{job['cpu']})
    from large_pursuit.provenance import bind_job
    identity=bind_job(job['folder'],job)
    result_path=Path(job['folder'])/'result.json'
    if result_path.exists():return read(result_path)
    if job['kind']=='reference_profile':
        from large_pursuit.core import Config,PublicMap,World,from_initial
        from large_pursuit.rootpuct import rootpuct_evader
        case=read(job['case']);public=PublicMap(Config.scale(case['n']),case['boxes']);state=from_initial(case,0)
        started=time.perf_counter();action,stats=rootpuct_evader(public,state,np.random.default_rng(job['seed']),job['simulations'],100,return_stats=True)
        elapsed=time.perf_counter()-started
        if action.shape!=(case['n'],2) or np.max(np.abs(action))>1+1e-8:raise RuntimeError('RootPUCT action contract failed')
        nxt=World(public.cfg,public.boxes).transition(state,np.zeros_like(action),action)
        if not np.isfinite(nxt.e).all():raise RuntimeError('RootPUCT transition not finite')
        if stats['pursuer_expansions']<1:raise RuntimeError('RootPUCT did not expand the opponent team')
        result=dict(n=case['n'],seconds=elapsed,action_max=float(np.abs(action).max()),**stats)
    elif job['kind']=='calibration':result=calibration_job(job)
    elif job['kind']=='teacher_collect':
        from large_pursuit.training import collect_teacher_packets
        result=collect_teacher_packets(job)
    elif job['kind']=='teacher_label':
        from large_pursuit.training import label_teacher_packet
        result=label_teacher_packet(job)
    elif job['kind']=='dataset':
        from large_pursuit.training import make_dataset
        result=make_dataset(job)
    elif job['kind']=='evaluate':
        from large_pursuit.evaluation import evaluate
        result=evaluate(job)
    else:raise ValueError(job['kind'])
    result['identity']=identity['digest']
    write(result_path,result);return result


def cpu_batch(out,jobs,cpus,workers,stage):
    from large_pursuit.provenance import bind_job
    for job in jobs:bind_job(job['folder'],job)
    missing=[job for job in jobs if not (Path(job['folder'])/'result.json').exists()]
    if not missing:return [read(Path(j['folder'])/'result.json') for j in jobs]
    event(out,'stage_started',stage=stage,jobs=len(missing),workers=workers)
    context=multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=worker_init,initargs=(cpus,)) as pool:
        active={};iterator=iter(missing);free=list(cpus[:workers]);finished=0
        while True:
            while free:
                try:job=next(iterator)
                except StopIteration:break
                cpu=free.pop();submitted=dict(job,cpu=cpu);active[pool.submit(cpu_job,submitted)]=(cpu,job)
            if not active:break
            done,_=concurrent.futures.wait(active,timeout=20,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                cpu,job=active.pop(future);free.append(cpu)
                try:future.result()
                except Exception:
                    write(Path(job['folder'])/'failure.json',dict(traceback=traceback.format_exc(),job=job));raise
                finished+=1
            write(out/'progress.json',dict(stage=stage,completed=finished+len(jobs)-len(missing),total=len(jobs),active=len(active),pid=os.getpid()))
    return [read(Path(j['folder'])/'result.json') for j in jobs]


def gpu_free():
    value=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(a):int(b) for row in value.splitlines() for a,b in [row.split(',')]}


def gpu_batch(out,jobs,layout,max_jobs,stage):
    from large_pursuit.provenance import bind_job
    for job in jobs:bind_job(job['folder'],job)
    missing=collections.deque(j for j in jobs if not (Path(j['folder'])/'result.json').exists());active={}
    initial_free=gpu_free();reserve_mb=1024
    event(out,'gpu_stage_started',stage=stage,jobs=len(missing),max_jobs=max_jobs,reserve_mb_per_job=reserve_mb)
    try:
        while missing or active:
            for pid,(proc,handle,job) in list(active.items()):
                code=proc.poll()
                if code is None:continue
                handle.close();del active[pid]
                if code:
                    write(Path(job['folder'])/'failure.json',dict(exit_code=code,job=job))
                    raise RuntimeError(f'Training failed: {job["folder"]}; inspect console.log')
                if not (Path(job['folder'])/'result.json').exists():raise RuntimeError('Training exited without result')
                event(out,'training_completed',stage=stage,folder=job['folder'],pid=pid)
            while missing and len(active)<max_jobs:
                memory=gpu_free();counts=collections.Counter(v[2]['gpu'] for v in active.values())
                pending_folder=Path(missing[0]['folder'])
                resume=read(pending_folder/'job.json').get('gpu') if (pending_folder/'latest.pt').exists() and (pending_folder/'job.json').exists() else None
                preference=(resume,) if resume is not None else (3,2,1)
                gpu=next((g for g in preference if g in (3,2,1) and min(memory.get(g,0),initial_free.get(g,0)-counts[g]*reserve_mb)>reserve_mb+2048),None)
                if gpu is None:break
                used={item[2]['cpus'][0] for item in active.values()};cpu=next(c for c in layout['training'] if c not in used)
                job=dict(missing.popleft(),gpu=gpu,cpus=[cpu]);folder=Path(job['folder']);folder.mkdir(parents=True,exist_ok=True)
                write(folder/'job.json',job);handle=open(folder/'console.log','a')
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
                proc=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'--train-job',str(folder/'job.json')],env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                active[proc.pid]=(proc,handle,job);event(out,'training_started',stage=stage,n=job['n'],branch=job['branch'],seed=job['seed'],gpu=gpu,cpu=cpu,pid=proc.pid)
                # Startup memory reservation prevents launches oversubscribing a busy card.
                time.sleep(1)
            write(out/'progress.json',dict(stage=stage,queued=len(missing),active=[dict(pid=pid,folder=item[2]['folder'],gpu=item[2]['gpu']) for pid,item in active.items()],pid=os.getpid()))
            time.sleep(5)
    finally:
        # Avoid unowned training if the stage fails. Checkpointed jobs can resume.
        for proc,handle,job in active.values():
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            handle.close()
    return [read(Path(j['folder'])/'result.json') for j in jobs]


def train_job(out,phase,n,branch,seed,role='pursuer',round_index=0,**fields):
    pilot=phase=='pilot'
    base=dict(kind='train',bank=str(out/'bank'),n=n,branch=branch,seed=seed,role=role,
        folder=str(out/phase/f'r{round_index}_{role}_{n}_{branch}_{seed}'),
        train_split='pilot_train' if pilot else 'train',validation_split='pilot_validation' if pilot else 'validation',
        validation_maps=14 if pilot else 56,validation_every=2048 if pilot else 32768,
        team_steps=8192 if pilot else 262144,rollout=256,minibatch=64,ppo_epochs=4,
        supervised_updates=128 if pilot else 2048,auxiliary_updates=1 if role=='pursuer' else 0,
        online_teacher_stride=256 if role=='pursuer' else 0,teacher_particles=4,teacher_depth=8 if pilot else 32,
        collection_simulations=2 if pilot else 16,validation_simulations=2 if pilot else 64,search_depth=4 if pilot else 8,
        opponent_simulations=2 if pilot else 64,opponents=[],round_index=round_index,select_final=phase=='independent')
    base.update(fields);return base


def protocol(out,layout):
    from large_pursuit.protocol import method_protocol
    document=method_protocol()
    document.update(source_plan=str(ROOT/'计划/大规模持续追击_5v5与16v16重训及强基线评测计划.md'),
        plan_sha256=hashlib.sha256((ROOT/'计划/大规模持续追击_5v5与16v16重训及强基线评测计划.md').read_bytes()).hexdigest(),
        scales={'5v5':dict(size=40,horizon=160,obstacles=[24,32,40]),'16v16':dict(size=80,horizon=320,obstacles=[80,100,120])},
        seeds=[17,29,43],independent_evader_seeds=[61,73,89],
        task=dict(speed_per_axis=1,sensing=6,capture_strict_radius=1,line_of_sight=True,pursuers_persistent=True,multi_capture=True,
            robot_intercollision=False,swept_wall_contact=True,simultaneous_actions=True,initial_positions_announced_once=True,evader_full_state=True),
        splits=dict(train=280,validation=56,sealed_ID=56,sealed_OOD=56,initials_per_map=2),
        budgets=dict(simulations=[16,64],milliseconds=[200,1000],scope='observation update through action return',direct='actual elapsed time'),
        pilot=dict(seed=101,evader_seed=107,steps=8192,teacher_queries_per_scale=32),
        resources=layout,formal_budget='not frozen until repair, calibration and fresh pilot evidence pass',
        ood_scope='held-out obstacle density 40/120, seven existing layout families; no claim of unseen-family generalization')
    path=out/'protocol.draft.json'
    if path.exists() and read(path)!=json.loads(json.dumps(document)):raise RuntimeError('Protocol changed inside a versioned run')
    write(path,document)


def pilot_report(out,calibration,results):
    summaries=[];finite=True;has_progress=True
    for result in results:
        folder=Path(result['latest']).parent
        curves=[json.loads(row) for row in (folder/'curve.jsonl').read_text().splitlines()]
        losses=np.array([r['loss'] for r in curves]);checkpoints=[r for r in curves if r['validation'] is not None]
        healthy=bool(np.isfinite(losses).all() and len(checkpoints)>=3)
        improvement=checkpoints[-1]['validation']-checkpoints[0]['validation'] if checkpoints else None
        summaries.append(dict(n=result['n'],branch=result['branch'],seed=result['seed'],healthy=healthy,
            first_validation=checkpoints[0]['validation'] if checkpoints else None,last_validation=checkpoints[-1]['validation'] if checkpoints else None,
            validation_improvement=improvement,mean_steps_per_second=float(np.mean([r['steps_per_second'] for r in curves])),
            peak_gpu_mb=max(r['peak_gpu_mb'] for r in curves),teacher_records=len(__import__('torch').load(folder/'online_teacher.pt',weights_only=False)),
            training_result=result))
        finite &= healthy
    report=dict(status='requires_scientific_audit',calibration=calibration,learning_curves=summaries,
        engineering_passed=True,numerically_healthy=finite,
        formal_budget_frozen=False,
        outstanding=['Verify legacy fixed motion kernel and RootPUCT equivalence or document justified replacements.',
            'Audit open-loop MCTS tree belief/value consistency and learned evader prediction diagnostics.',
            'Run common frozen-opponent alternating pilot, including independent teacher labels and validation for every branch.',
            'Calibrate isolated-core timing at 1/2/4 workers with bulk collection paused.',
            'Inspect learning curves before fixing team steps, teacher queries, rounds, update limits and selection rules.'])
    write(out/'pilot_report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--workers',type=int,default=84);parser.add_argument('--gpu-jobs',type=int,default=24)
    parser.add_argument('--train-job',type=Path)
    args=parser.parse_args()
    if args.train_job:
        from large_pursuit.training import train
        train(read(args.train_job));return
    from large_pursuit.acceptance import require_repair
    require_repair(args.output)
    subprocess.run([PYTHON,str(ROOT/'code/run_large_pursuit_full.py'),'--output',str(args.output),
        '--workers',str(args.workers),'--gpu-jobs',str(args.gpu_jobs)],check=True)

if __name__=='__main__':main()
