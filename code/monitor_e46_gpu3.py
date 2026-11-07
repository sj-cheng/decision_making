"""A visible, low-overhead GPU3 lifetime marker for the E46 pipeline.

Holds a tiny CUDA allocation; does no training or GPU busy-loop computation.
Disappears after the registered coordinator/launcher stop. Terminal status is
kept on disk so completion and abnormal termination remain distinguishable.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'current/e46_geometry_teacher_20261105'
CST = datetime.timezone(datetime.timedelta(hours=8))


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write(path, data):
    path=Path(path)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    os.replace(tmp,path)


def identity(pid, required_token):
    try:
        directory=Path('/proc')/str(int(pid))
        stat=(directory/'stat').read_text()
        fields=stat[stat.rfind(')')+2:].split()
        command=(directory/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if fields[0]=='Z' or required_token not in command:
            return None
        return fields[19]
    except (FileNotFoundError, ProcessLookupError, PermissionError, TypeError, ValueError):
        return None


def inspect(output, remembered):
    progress=read(output/'progress.json',{})
    launch=read(output/'launch.json',{})
    coordinator=progress.get('pid',launch.get('coordinator_pid'))
    coordinator_identity=identity(coordinator,'run_e46_geometry_teacher.py')
    if coordinator_identity is not None:
        previous=remembered.setdefault(coordinator,coordinator_identity)
        if previous!=coordinator_identity:
            coordinator_identity=None  # A recycled PID is not the original run.
    pane=str(launch.get('pane','')).split()
    launcher=int(pane[-1]) if pane and pane[-1].isdigit() else None
    launcher_alive=identity(launcher,'launch_e46_geometry_teacher.sh') is not None
    alive=coordinator_identity is not None or launcher_alive
    finished=progress.get('phase')=='finished' and (output/'final/COMPLETE.json').exists()
    phase=progress.get('phase','unknown')
    if alive:
        state='FINISHING' if finished else ('STOPPING' if phase=='stopped_incomplete' else 'RUNNING')
    else:
        state='COMPLETED' if finished else 'STOPPED_INCOMPLETE'
    stages=progress.get('stages',{})
    totals={key:sum(v.get(key,0) for v in stages.values()) for key in ('done','running','pending','failed')}
    now=datetime.datetime.now(CST).isoformat(timespec='seconds')
    return dict(state=state,pipeline_alive=alive,coordinator_pid=coordinator,
        coordinator_alive=coordinator_identity is not None,launcher_pid=launcher,
        launcher_alive=launcher_alive,progress_phase=phase,progress_updated=progress.get('time'),
        checked_cst=now,jobs=totals,gpu_training_jobs=progress.get('active_gpu_jobs',[]),
        saved_models=len(list((output/'training').glob('*/result.json'))),
        training_failures=progress.get('training_failures',[]),completion_marker=finished,
        monitor_pid=os.getpid(),monitor_gpu_physical=3,
        gpu_workload='CUDA lifetime marker only; no busy loop or training')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--interval',type=float,default=5.)
    parser.add_argument('--once',action='store_true',help='Inspect once without allocating GPU memory')
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_dir():
        raise SystemExit('E46 output directory does not exist')
    remembered={}
    state=inspect(output,remembered)
    if args.once:
        print(json.dumps(state,ensure_ascii=False,indent=2))
        return 0 if state['pipeline_alive'] or state['completion_marker'] else 2
    # One visible marker per E46 output directory, even after accidental relaunch.
    import fcntl
    lock=open(output/'gpu3_monitor.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not state['pipeline_alive']:
        write(output/'gpu3_monitor_status.json',state)
        return 0 if state['completion_marker'] else 2
    resources=read(output/'resources.json',{})
    candidates=resources.get('training',[])
    available=os.sched_getaffinity(0)
    cpu=12 if 12 in candidates and 12 in available else next((c for c in candidates if c in available),None)
    if cpu is not None:os.sched_setaffinity(0,{cpu})
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[key]='1'
    uuid=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    os.environ['CUDA_VISIBLE_DEVICES']=uuid  # Physical GPU3 only, regardless of CUDA device ordering.
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    allocation=torch.empty(1024,dtype=torch.uint8,device='cuda:0')
    torch.cuda.synchronize()
    stop=[False]
    def request_stop(*_):stop[0]=True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    while not stop[0]:
        state=inspect(output,remembered)
        state.update(gpu_uuid=uuid,cpu_affinity=sorted(os.sched_getaffinity(0)),
                     torch_reserved_bytes=torch.cuda.memory_reserved(0))
        title=(f'E46_MONITOR | {state["state"]} | done={state["jobs"]["done"]} '
               f'pending={state["jobs"]["pending"]} '
               f'gpu_train={len(state["gpu_training_jobs"])} '
               f'failed={state["jobs"]["failed"]+len(state["training_failures"])}')
        write(output/'gpu3_monitor_status.json',state)
        print(state['checked_cst'],title,flush=True)
        if not state['pipeline_alive']:
            break
        deadline=time.monotonic()+max(1.,args.interval)
        while not stop[0] and time.monotonic()<deadline:
            time.sleep(min(.5,max(0.,deadline-time.monotonic())))
    if stop[0]:
        state.update(state='MONITOR_STOPPED_MANUALLY',checked_cst=datetime.datetime.now(CST).isoformat(timespec='seconds'))
        write(output/'gpu3_monitor_status.json',state)
    del allocation
    return 0 if state.get('completion_marker') or stop[0] else 2


if __name__=='__main__':
    raise SystemExit(main())
