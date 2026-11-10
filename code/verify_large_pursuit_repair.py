"""Repair tests only. Writes a release receipt only after all checks actually pass."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['TZ']='Asia/Shanghai'
import argparse
import shutil
import signal
import traceback
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'current/large_pursuit_v2_20261108'


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2));temporary.replace(path)


def make_fixtures(out):
    from large_pursuit.core import generate_case
    from large_pursuit.provenance import sha
    bank=out/'verification/bank';entries=[]
    for n in (5,16):
        for split in ('train','validation'):
            relative=f'{n}_{split}.json';path=bank/relative
            if not path.exists():
                case=generate_case(n,'post_field',714201+n+(split=='validation'),40 if n==5 else 120)
                case.update(id=f'verification_{n}_{split}',suite='ID',split=split);write(path,case)
            entries.append(dict(n=n,split=split,id=f'verification_{n}_{split}',path=relative,sha256=sha(path)))
    write(bank/'manifest.json',dict(schema='large-pursuit-v2',scope='engineering fixtures only; no formal or sealed maps',entries=entries))


def proof_job(path):
    import torch
    import large_pursuit.training as training
    from large_pursuit.semantics import SCHEMA
    base=json.loads(Path(path).read_text());folder=Path(base.pop('proof_folder'))
    # Validation logic has separate behavioral/matrix tests. This proof isolates
    # exact simulation/controller/optimizer/RNG restoration without a full validation sweep.
    training.validation=lambda *a,**k:dict(score=0.,mean=0.,groups={'functional_stub':0.},episodes=[])
    uninterrupted=dict(base,folder=str(folder/'uninterrupted'))
    resumed=dict(base,folder=str(folder/'resumed'))
    training.train(uninterrupted)
    try:training.train(dict(resumed,interrupt_after_rollouts=1))
    except InterruptedError:pass
    training.train(resumed)
    x=torch.load(Path(uninterrupted['folder'])/'latest.pt',map_location='cpu',weights_only=False)
    y=torch.load(Path(resumed['folder'])/'latest.pt',map_location='cpu',weights_only=False)
    def equal(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
        import dataclasses
        if dataclasses.is_dataclass(a):return equal(a.__dict__,b.__dict__)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(v,w) for v,w in zip(a,b))
        import numpy as np
        if isinstance(a,np.ndarray):return np.array_equal(a,b,equal_nan=True)
        return a==b
    checks={key:equal(x[key],y[key]) for key in ('model','optimizer','environment','steps','updates','teacher_queries','planner_simulations','opponent_samples','numpy_rng','torch_rng','cuda_rng')}
    if not all(checks.values()):raise AssertionError('Exact checkpoint proof failed: '+str(checks))
    report=dict(schema=SCHEMA,n=base['n'],role=base['role'],device='cuda' if torch.cuda.is_available() else 'cpu',gpu=base['gpu'],checks=checks,
        scope='exact training, optimizer, physical episode, public posterior, controller history and CPU/CUDA RNG; validation matrix separately tested',torch_version=torch.__version__)
    write(folder/'proof.json',report)


def cleanup_proofs(out):
    """Remove only our own stale functional-test children before archiving paths."""
    script=Path(__file__).resolve();victims=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            if entry.stat().st_uid!=os.getuid():continue
            argv=(entry/'cmdline').read_bytes().decode().split('\0')
            if len(argv)<4 or Path(argv[1]).resolve()!=script or '--proof-job' not in argv:continue
            job=Path(argv[argv.index('--proof-job')+1]).resolve()
            if not job.is_relative_to(out/'verification'):continue
            pid=int(entry.name)
            if pid==os.getpid():continue
            os.kill(pid,signal.SIGTERM);victims.append(pid)
        except (OSError,ValueError,IndexError):continue
    for _ in range(30):
        alive=[]
        for pid in victims:
            try:
                status=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
                if status!='Z':alive.append(pid)
            except OSError:pass
        if not alive:return
        time.sleep(.1)
    raise RuntimeError('Functional-test child did not exit: '+str(alive))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--proof-job',type=Path)
    args=parser.parse_args()
    if args.proof_job:proof_job(args.proof_job);return
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cleanup_proofs(out)
    def terminate(*_):
        cleanup_proofs(out)
        raise KeyboardInterrupt('Repair verification interrupted')
    signal.signal(signal.SIGTERM,terminate)
    from run_large_pursuit import resources
    unit_layout=resources()
    os.sched_setaffinity(0,set(unit_layout['bulk']+unit_layout.get('bulk_smt',[])))
    write(out/'audit_hold.json',dict(reason='Repair verification in progress; no experiment dispatch yet'))
    from large_pursuit.acceptance import acceptance_sources
    before=acceptance_sources();manifest=out/'verification/source_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=before:
        old=json.loads(manifest.read_text());name=hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest()[:12]
        target=out/'verification_attempts'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():target=target.with_name(name+'_'+str(int(time.time())))
        shutil.move(str(out/'verification'),str(target))
        if (out/'closed_loop_acceptance.json').exists():shutil.move(str(out/'closed_loop_acceptance.json'),str(target/'closed_loop_acceptance.json'))
    write(manifest,before)
    write(out/'verification_coordinator.json',dict(pid=os.getpid(),status='running'))
    write(out/'progress.json',dict(stage='repair_verification',pid=os.getpid(),running=True))
    logs=out/'verification/logs';logs.mkdir(parents=True,exist_ok=True)
    print(time.strftime('%Y-%m-%d %H:%M:%S')+' repair verification started; experiments remain gated',flush=True)
    for script in ('test_large_pursuit.py','test_large_pursuit_semantics.py'):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONPATH']=str(ROOT/'code')
        result=subprocess.run([sys.executable,str(ROOT/'code/tests'/script)],env=env,capture_output=True,text=True)
        (logs/(script+'.log')).write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('Repair regression failed: '+script)
        print('PASS '+script,flush=True)
    make_fixtures(out)
    from run_large_pursuit import train_job,resources
    layout=unit_layout;cpus=layout['training'];jobs=[]
    # GPU0 is never visible in CUDA verification. Same physical GPU preference as plan.
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    available={int(a):int(b) for line in memory.splitlines() for a,b in [line.split(',')]}
    gpu=next((g for g in (3,2,1) if available.get(g,0)>2048),None)
    if gpu is None:raise RuntimeError('No permitted GPU available for CUDA restoration proof')
    for device in ('cpu','cuda'):
        for n in (5,16):
            for role in ('pursuer','evader'):
                folder=out/'verification'/f'{device}_{n}_{role}'
                base=train_job(out,'pilot',n,'joint' if role=='pursuer' else 'evader',619,role=role,
                    bank=str(out/'verification/bank'),train_split='train',validation_split='validation',verification=True,
                    team_steps=32,rollout=16,minibatch=16,ppo_epochs=2,validation_every=16,validation_maps=1,
                    supervised_updates=0,auxiliary_updates=0,online_teacher_stride=0,collection_simulations=1,
                    opponent_simulations=1,validation_simulations=1,search_depth=2,gpu=gpu,cpus=[cpus[len(jobs)%len(cpus)]])
                base.pop('folder');base['proof_folder']=str(folder)
                path=folder/'proof_job.json';write(path,base)
                jobs.append((device,path,folder))
    active=[];pending=list(jobs)
    while pending or active:
        while pending and len(active)<4:
            device,path,folder=pending.pop(0);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu) if device=='cuda' else '';env['PYTHONPATH']=str(ROOT/'code')
            handle=open(logs/(folder.name+'.log'),'a')
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--proof-job',str(path)],env=env,stdout=handle,stderr=subprocess.STDOUT)
            active.append((process,handle,folder));print('checkpoint proof started: '+folder.name+' pid='+str(process.pid),flush=True)
        for process,handle,folder in list(active):
            status=process.poll()
            if status is None:continue
            handle.close();active.remove((process,handle,folder))
            if not status:print('checkpoint proof passed: '+folder.name,flush=True)
            if status:
                for other,other_handle,_ in active:
                    if other.poll() is None:other.terminate()
                    other.wait(timeout=15);other_handle.close()
                raise RuntimeError('Restoration proof failed: '+str(folder))
        time.sleep(2)
    print('32 closed-loop engineering episodes started',flush=True)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONPATH']=str(ROOT/'code')
    with open(logs/'closed_loop.log','a') as handle:
        result=subprocess.run([sys.executable,str(ROOT/'code/verify_large_pursuit.py'),'--output',str(out)],env=env,stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Closed-loop engineering acceptance failed')
    proofs=[json.loads((folder/'proof.json').read_text()) for _,_,folder in jobs]
    closed=json.loads((out/'closed_loop_acceptance.json').read_text())
    if len(proofs)!=8 or not closed['passed'] or closed['sealed_used']:raise RuntimeError('Incomplete repair evidence')
    if acceptance_sources()!=before:raise RuntimeError('Code changed during repair verification')
    receipt=dict(schema='large-pursuit-v2',status='passed',sources=before,closed_blockers=['B%02d'%i for i in range(1,11)],
        tests=['environment acceptance','behavior/mathematical semantics','8 CPU/CUDA role/scale checkpoint proofs','32 scale-specific closed-loop engineering episodes'],
        restoration=proofs,closed_loop=closed,formal_started=False,old_pilot_models_reused=False)
    write(out/'repair_acceptance.json',receipt)
    (out/'audit_hold.json').unlink()
    write(out/'verification_coordinator.json',dict(pid=os.getpid(),status='passed'))
    write(out/'progress.json',dict(stage='repair_verified',running=False))
    print('Repair acceptance PASSED; fresh experiments may now start.',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        # The audit hold remains in place; nothing dispatches formal training.
        out=Path(sys.argv[sys.argv.index('--output')+1]) if '--output' in sys.argv else DEFAULT
        if '--proof-job' not in sys.argv:
            write(out/'verification_failure.json',dict(traceback=traceback.format_exc()))
            write(out/'progress.json',dict(stage='repair_verification_failed',running=False))
        raise
