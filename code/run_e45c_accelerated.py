"""Execution-only E45-C extension: parallel independent frozen trainer jobs.

CPU episode concurrency/budgets and every scientific function stay unchanged.
The original learner is derived with two guarded scheduler substitutions.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import argparse
import concurrent.futures as futures
import ctypes
import inspect
import json
import multiprocessing as mp
import signal
import time
from pathlib import Path
import numpy as np
import torch
import run_e45c_budget as r
from run_e44_parallel import core_topology

EXECUTION_NAME='EXECUTION_ACCELERATION_20261028.json'
TRAINERS=9
SLOTS=[]
ORIGINAL_POOL=r.Pool
ORIGINAL_LEARN=r.learn


def derive_learner(source,trainers):
    replacements=(('gpu=Pool(2,True)',f'gpu=Pool({trainers},True)'),
                  ('(True,gpu_queues,4,gpu)',f'(True,gpu_queues,{2*trainers},gpu)'))
    for old,new in replacements:
        if source.count(old)!=1:raise RuntimeError('Frozen scheduler shape changed: '+old)
        source=source.replace(old,new)
    return source


def verify_execution(root):
    root=Path(root);config=r.read(root/EXECUTION_NAME)
    assert config['original_experiment_sha256']==r.sha(root/'experiment.json')
    assert config['cpu_workers']==40
    for p,h in config['source_hashes'].items():
        if r.sha(p)!=h:raise RuntimeError('Frozen acceleration extension changed: '+p)
    for p,h in config['bound_artifacts'].items():
        if r.sha(p)!=h:raise RuntimeError('Bound acceleration artifact changed: '+p)
    return config


class OneSlot:
    def __init__(self,value):self.value=value
    def get(self):return self.value


def init_worker(root,smoke,queue,parent,training):
    verify_execution(root)
    slot=queue.get()
    if training:
        gpu,cpu=slot;os.sched_setaffinity(0,{cpu});slot=gpu
    r.child_init(root,smoke,OneSlot(slot),parent,training)


class AcceleratedPool(ORIGINAL_POOL):
    def __init__(self,count,training=False):
        self.training=training;ctx=mp.get_context('spawn');self.queue=ctx.Queue()
        values=SLOTS[:count] if training else r.CPUS[:count]
        if len(values)!=count:raise RuntimeError('Insufficient registered slots')
        for value in values:self.queue.put(value)
        self.executor=futures.ProcessPoolExecutor(max_workers=count,mp_context=ctx,
            initializer=init_worker,initargs=(str(r.OUT),r.SMOKE,self.queue,os.getpid(),training))


def install(root):
    global TRAINERS,SLOTS
    config=verify_execution(root);TRAINERS=config['gpu_trainers'];SLOTS=config['training_slots']
    r.WORKERS=40;r.OUT=Path(root);r.SMOKE=False
    execution=r.read(r.OUT/'execution.json');assert execution['workers']==40
    r.CPUS=execution['physical_cores']
    source=derive_learner(inspect.getsource(ORIGINAL_LEARN),TRAINERS)
    assert source==(r.OUT/'execution_acceleration/learner_snapshot.py').read_text()
    r.Pool=AcceleratedPool
    exec(compile(source,str(r.OUT/'execution_acceleration/learner_snapshot.py'),'exec'),r.__dict__)


def benchmark_init(root,queue,parent):
    ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    if os.getppid()!=parent:raise RuntimeError('Benchmark coordinator exited')
    gpu,cpu=queue.get();os.sched_setaffinity(0,{cpu});torch.set_num_threads(1)
    r.base.PARENT=parent;r.base.DEVICE=f'cuda:{gpu}';torch.cuda.set_device(gpu)
    torch.use_deterministic_algorithms(True);r.configure(root)


def benchmark_job(task):return r.base.train_job(task)


def benchmark(root):
    import shutil
    root=Path(root);root.mkdir(parents=True,exist_ok=True);r.verify()
    source=r.level('x20');execution=r.read(r.OUT/'execution.json')
    cores=core_topology();used={key for key,cpus in cores.items() if set(cpus)&set(execution['physical_cores'])}
    eligible=[cpu for cpu in r.choose_cpus(len(cores)) if next(key for key,v in cores.items() if cpu in v) not in used]
    assert len(eligible)>=9
    results={}
    for workers in (2,9):
        output=root/f'workers_{workers}';output.mkdir(exist_ok=True)
        r.save(output/'experiment.json',dict(epochs_per_round=1,max_states_per_episode=32,engineering_benchmark_only=True))
        for round_ in (8,9,10):
            dest=output/f'round_{round_:02d}';dest.mkdir(exist_ok=True)
            shutil.copyfile(source/f'round_{round_:02d}/collection_results.json',dest/'collection_results.json')
        dest=output/'round_09/models';dest.mkdir(exist_ok=True)
        for kind in r.KINDS:
            for seed in r.SEEDS:
                shutil.copyfile(source/f'round_09/models/{kind}_{seed}.pt',dest/f'{kind}_{seed}.pt')
        devices=[2,3] if workers==2 else [1,2,3]*3
        slots=list(zip(devices,eligible[:workers]));ctx=mp.get_context('spawn');queue=ctx.Queue()
        for slot in slots:queue.put(slot)
        tasks=[dict(kind=kind,seed=seed,round=10) for seed in r.SEEDS for kind in r.KINDS]
        started=time.perf_counter()
        with futures.ProcessPoolExecutor(workers,mp_context=ctx,initializer=benchmark_init,initargs=(str(output),queue,os.getpid())) as pool:
            records=list(pool.map(benchmark_job,tasks))
        wall=time.perf_counter()-started;queue.close()
        results[str(workers)]=dict(workers=workers,wall_seconds=wall,training_seconds=[x['seconds'] for x in records],
            model_hashes={f'{x["kind"]}_{x["seed"]}':x['model_sha256'] for x in records},slots=slots)
        r.save(root/'benchmark_progress.json',results)
        print(json.dumps(dict(workers=workers,wall_seconds=wall,mean_training_seconds=float(np.mean([x['seconds'] for x in records])))),flush=True)
    a,b=results['2'],results['9'];exact=a['model_hashes']==b['model_hashes']
    # The NPZ zip container embeds timestamps, so compare model arrays rather
    # than only archive bytes when different wall times yield different SHA.
    array_exact=True;max_difference=0.
    for key in a['model_hashes']:
        with np.load(root/f'workers_2/round_10/models/{key}.npz') as left,np.load(root/f'workers_9/round_10/models/{key}.npz') as right:
            assert left.files==right.files
            for name in left.files:
                array_exact &= np.array_equal(left[name],right[name])
                if left[name].dtype.kind not in ('U','S'):max_difference=max(max_difference,float(np.max(abs(left[name]-right[name]))))
    result=dict(conditions=results,wall_speedup=a['wall_seconds']/b['wall_seconds'],model_archives_identical=exact,
        model_arrays_exact=bool(array_exact),max_model_difference=max_difference,
        accepted=bool(array_exact and b['wall_seconds']<a['wall_seconds']),
        scope='Nine independent original trainer jobs, same existing train/validation data and previous checkpoints, ONE epoch; engineering only, no formal model substitution',
        validation_outcomes_used_for_parallelism=False)
    r.save(root/'benchmark.json',result);print(json.dumps({k:v for k,v in result.items() if k!='conditions'}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('pipeline','benchmark'));parser.add_argument('--root',type=Path,default=r.OUT);parser.add_argument('--benchmark-root',type=Path)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.mode=='benchmark':
        if args.benchmark_root is None:parser.error('--benchmark-root required')
        benchmark(args.benchmark_root);return
    r.OUT=args.root.resolve();r.verify();install(r.OUT)
    r.save(r.OUT/'execution_acceleration/resume.json',dict(pid=os.getpid(),tmux='train:E45C',cpu_workers=40,
        gpu_trainers=TRAINERS,training_slots=SLOTS,timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    r.pipeline()


if __name__=='__main__':main()
