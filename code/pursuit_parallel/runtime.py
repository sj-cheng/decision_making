from __future__ import annotations
import collections
import concurrent.futures
import copy
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from large_pursuit import training as canonical
from large_pursuit.provenance import ROOT,sha,sources

SERIAL_TEACHER=canonical.teacher_labels
SERIAL_VALIDATION=canonical.validation


def runtime_sources():
    paths=list((ROOT/'code/pursuit_parallel').glob('*.py'))+[
        ROOT/'code/run_large_pursuit_parallel.py',ROOT/'code/tests/test_pursuit_parallel.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def verify_receipt(root):
    path=Path(root)/'execution/parallel_equivalence.json'
    receipt=json.loads(path.read_text())
    if receipt.get('status')!='passed' or receipt['algorithm_sources']!=sources() or receipt['runtime_sources']!=runtime_sources():
        raise RuntimeError('Missing/stale parallel execution equivalence proof')
    return receipt


def worker_init(cpus):
    index=multiprocessing.current_process()._identity[-1]-1
    os.sched_setaffinity(0,{cpus[index%len(cpus)]})
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def teacher_chunk(cfg,boxes,obs,posterior,pool,rng_state,particles,depth):
    # Call the unchanged scientific teacher on a disjoint candidate slice.
    # Each slice receives exactly the same pre-draw RNG state and root seeds.
    from large_pursuit.core import Config,PublicMap
    key=hashlib.sha256(np.asarray(boxes).tobytes()+repr(cfg).encode()).hexdigest()
    if not hasattr(teacher_chunk,'maps'):teacher_chunk.maps={}
    if key not in teacher_chunk.maps:
        if len(teacher_chunk.maps)>=12:teacher_chunk.maps.pop(next(iter(teacher_chunk.maps)))
        teacher_chunk.maps[key]=PublicMap(Config(**cfg),boxes)
    public=teacher_chunk.maps[key]
    belief=canonical.restore_belief(public,posterior)
    rng=np.random.default_rng();rng.bit_generator.state=rng_state
    return SERIAL_TEACHER(public,obs,belief,pool,rng,particles,depth)


def make_teacher(executor,workers):
    def parallel_teacher(public,obs,belief,pool,rng,particles=4,depth=32):
        state=copy.deepcopy(rng.bit_generator.state)
        # Advance the parent's generator once, exactly as the canonical call.
        rng.integers(2**31,size=particles)
        posterior=canonical.belief_state(belief)
        chunks=[part for part in np.array_split(pool,min(workers,len(pool))) if len(part)]
        pending=[executor.submit(teacher_chunk,public.cfg.__dict__,public.boxes,obs,posterior,
                                 part,state,particles,depth) for part in chunks]
        return np.concatenate([future.result() for future in pending])
    return parallel_teacher


def trace_update(digest,state,pa,ea,nxt):
    for value in (state.p,state.e,state.alive,pa,ea,nxt.p,nxt.e,nxt.alive):
        digest.update(np.ascontiguousarray(value).tobytes())


def _model_on_device(state_dict,device,root,gpu):
    model=canonical.Network();model.load_state_dict(state_dict);model.eval()
    if not str(device).startswith('cuda'):return model.to(device)
    # Serialize admission and actual CUDA initialization. A chunk releases its
    # context when its short-lived process exits, so waiting cannot pin an idle
    # persistent pool's GPU memory indefinitely.
    lock_path=Path(root)/'execution/cuda_initialization.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    while True:
        with open(lock_path,'w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}',
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=1536:
                model=model.to(device);torch.cuda.synchronize();return model
        time.sleep(1)


def validation_chunk(chunk,job,state_dict,device,root):
    from large_pursuit.core import World,from_initial
    from large_pursuit.methods import Controller,MotionKernel,evader_action
    model=_model_on_device(state_dict,device,root,job.get('gpu',3))
    specs=canonical.load_specs(job.get('opponents') or [dict(method='AG' if job['role']=='evader' else 'D_kernel')],'cpu')
    rows=[]
    for index,item in chunk:
        case,public=canonical.read_case(item['case']);state=from_initial(case,item['initial']);world=World(public.cfg,public.boxes)
        spec=specs[item['opponent_index']];rng=np.random.default_rng(700_000+job['n']*1000+index)
        if job['role']=='pursuer':
            ctrl=Controller(public,canonical.method_for(job['branch']),17,model,job['validation_simulations'],depth=job['search_depth'])
        else:ctrl=Controller(public,spec['method'],17,spec['model'],job['validation_simulations'],depth=job['search_depth'])
        motion=MotionKernel(public,rng);digest=hashlib.sha256();started=time.perf_counter()
        while not world.done(state):
            pa=ctrl.act(world.observe(state))
            if job['role']=='evader':ea=evader_action(public,state,rng,model)
            elif spec['model'] is not None:ea=evader_action(public,state,rng,spec['model'])
            else:ea=motion.actions(state)
            nxt=world.transition(state,pa,ea);trace_update(digest,state,pa,ea,nxt);state=nxt
        score=1-float(state.alive.mean()) if job['role']=='pursuer' else float(state.alive.mean())
        label=spec.get('method','D_kernel')
        if job['role']=='pursuer' and spec.get('checkpoint'):label+=f"_s{spec.get('seed')}_r{spec.get('round',0)}"
        rows.append((index,dict(case=case['id'],initial=item['initial'],opponent=label,score=score),
                     dict(trajectory_sha256=digest.hexdigest(),decisions=state.t,seconds=time.perf_counter()-started)))
    return rows


def parallel_validation(model,job,device,cpus,workers,root):
    paths=canonical.entries(job['bank'],job['n'],job['validation_split'])
    if job.get('verification'):paths=paths[:job.get('validation_maps',1)]
    elif len(paths)!=job['validation_maps']:raise ValueError('Validation bank does not match frozen map count')
    # Preserve the canonical parent's model construction/RNG consumption too.
    specs=canonical.load_specs(job.get('opponents') or [dict(method='AG' if job['role']=='evader' else 'D_kernel')],'cpu')
    plan=canonical.validation_plan(paths,specs);model.eval()
    canonical.write_json(Path(job['folder'])/'validation_matrix.json',dict(scenarios=plan,opponents=job.get('opponents',[]),
        rule='all maps x two initials; balanced methods and cycling seed/history snapshots; worst opponent mean then overall mean'))
    state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    workers=min(workers,len(plan));chunks=[list(enumerate(plan))[i::workers] for i in range(workers)]
    started=time.perf_counter()
    context=multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=context,
            initializer=worker_init,initargs=(cpus,),max_tasks_per_child=1) as executor:
        pending=[executor.submit(validation_chunk,chunk,job,state_dict,str(device),str(root)) for chunk in chunks]
        rows=sorted([row for future in pending for row in future.result()],key=lambda row:row[0])
    groups=collections.defaultdict(list);episodes=[]
    for _,episode,_ in rows:groups[episode['opponent']].append(episode['score']);episodes.append(episode)
    means={key:float(np.mean(value)) for key,value in groups.items()}
    result=dict(score=min(means.values()),mean=float(np.mean([e['score'] for e in episodes])),groups=means,episodes=episodes)
    canonical.write_json(Path(job['folder'])/'execution'/f'validation_{time.time_ns()}.json',
        dict(workers=workers,cpus=cpus,seconds=time.perf_counter()-started,rows=rows,
             runtime_sources=runtime_sources(),timing_claim=False))
    return result


def run_training(job,config):
    root=Path(config['root']);verify_receipt(root)
    canonical.write_json(Path(job['folder'])/'execution/runtime.json',dict(
        configuration=config,algorithm_sources=sources(),runtime_sources=runtime_sources(),
        preserves='candidate order, root seeds, parent RNG consumption, validation scenarios and reduction order',
        pid=os.getpid(),time=time.strftime('%Y-%m-%d %H:%M:%S')))
    if len(config['cpus'])<=1:return canonical.train(job)
    context=multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=config['teacher_workers'],mp_context=context,
            initializer=worker_init,initargs=(config['cpus'],)) as teachers:
        canonical.teacher_labels=make_teacher(teachers,config['teacher_workers'])
        canonical.validation=lambda model,spec,device:parallel_validation(
            model,spec,device,config['cpus'],config['validation_workers'],root)
        try:return canonical.train(job)
        finally:canonical.teacher_labels=SERIAL_TEACHER;canonical.validation=SERIAL_VALIDATION
