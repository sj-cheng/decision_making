"""Execution-only E45-O amendment: 12 rollout workers/run, original GPU1.

The frozen scientific runner, batch size, RNG seeds, replay and author losses
remain unchanged. Only process allocation and CUDA device assignment change.
"""
import argparse
import ctypes
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import run_e45o_hasac_demonstration as o


def pipeline():
    root=o.OUT;cfg=o.verify(root)
    assert o.j.read(root.parent/'e45o_hasac_demonstration_smoke_20261101/smoke_audit.json')['passed']
    lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    amendment=o.j.read(root/'parallel_execution_amendment.json')
    assert o.j.sha(Path(__file__).resolve())==amendment['runner_sha256']
    assert o.j.sha(root/'experiment.json')==amendment['experiment_sha256']
    combinations=[(arm,seed) for arm in cfg['arms'] for seed in cfg['seeds']]
    devices=['cuda:1']*4
    cores=o.j.choose_cpus(52)
    assert len(cores)==52
    children=[];started=time.time()
    try:
        allocation=[]
        for i,((arm,seed),device) in enumerate(zip(combinations,devices)):
            folder=root/'runs'/f'{arm}_{seed}'
            use=cores[i*13:(i+1)*13]
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'train',
                '--arm',arm,'--seed',str(seed),'--device',device,
                '--cpus',','.join(map(str,use)),'--parent',str(os.getpid())]
            if (folder/'decision.json').exists():continue
            with (folder/'run.log').open('a') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            children.append(child)
            allocation.append(dict(arm=arm,seed=seed,device=device,cores=use,pid=child.pid,workers=12))
        o.j.save(root/'execution.json',dict(pids=[p.pid for p in children],cores=cores,
            allocation=allocation,execution_only_amendment=True))
        o.j.common.emit(root,'running_parallel',pids=[p.pid for p in children],workers_per_model=12)
        while any(child.poll() is None for child in children):
            o.report(root)
            if any(child.poll() not in (None,0) for child in children):
                raise RuntimeError('Parallel trainer failed; inspect retained logs')
            # The initial launch wall-clock remains part of the registered cap.
            elapsed=time.time()-amendment['original_launch_unix']
            if elapsed>cfg['maximum_wall_seconds']:
                raise RuntimeError('Original experiment wall cap reached')
            time.sleep(10)
        assert all(child.returncode==0 for child in children)
        o.verify(root)
        result=dict(status='complete',parallel_wall_seconds=time.time()-started,runs={})
        for arm,seed in combinations:
            result['runs'][f'{arm}_{seed}']=o.j.read(root/'runs'/f'{arm}_{seed}'/'decision.json')
        o.j.save(root/'decision.json',result);o.j.common.emit(root,'complete');o.report(root)
        from analyze_e45o_demonstration import analyze
        analyze(root)
    except BaseException as exc:
        o.j.common.emit(root,'failed',error=repr(exc),traceback=traceback.format_exc());o.report(root)
        raise
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['pipeline','train'])
    parser.add_argument('--arm',choices=['baseline','guided']);parser.add_argument('--seed',type=int)
    parser.add_argument('--device');parser.add_argument('--cpus');parser.add_argument('--parent',type=int)
    args=parser.parse_args();args.output=o.OUT
    if args.parent:
        ctypes.CDLL(None).prctl(1,signal.SIGTERM)
        if os.getppid()!=args.parent:raise RuntimeError('Coordinator exited')
    if args.command=='train':o.train(args)
    else:pipeline()
