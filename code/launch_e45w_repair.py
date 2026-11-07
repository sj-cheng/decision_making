"""Supervise the W repair while the four unaffected original learners continue."""
import os
import signal
import subprocess
import sys
import time
import json
import hashlib
import fcntl
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'current/e45w_joint_ppo_and_planning_20261103'
STATE=OUT/'repair_coordinator'
PYTHON=OUT/'venv/bin/python'
POOLS={'B_17':(1,list(range(0,8))), 'C_17':(1,list(range(8,16))),
       'B_29':(2,list(range(16,24))), 'C_29':(2,list(range(24,32))),
       'B_43':(3,list(range(32,40))), 'C_43':(3,list(range(40,48)))}


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def terminate_failed(status):
    root=int(status['pid']); proc=Path('/proc')/str(root)
    if not proc.exists():return
    command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if not any(s in command for s in ('code/run_e45w_joint_training.py','code/run_e45w_repaired_training.py')):
        raise RuntimeError('Refuse to terminate unrelated process')
    rows={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            ppid=int(next(x for x in (p/'status').read_text().splitlines() if x.startswith('PPid:')).split()[1])
            rows[int(p.name)]=ppid
        except (OSError,StopIteration):pass
    ids={root}
    while True:
        expanded=ids|{pid for pid,ppid in rows.items() if ppid in ids}
        if expanded==ids:break
        ids=expanded
    for pid in [root]+sorted(ids-{root}):
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass


def main():
    STATE.mkdir(parents=True,exist_ok=True)
    lock=(STATE/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (STATE/'registry.json').exists():raise RuntimeError('Refuse duplicate repair coordinator')
    sources=[ROOT/'code'/name for name in ('learning/e45w_support_recovery.py',
        'run_e45w_repaired_training.py','run_e45w_repaired_planning.py',
        'run_e45w_repaired_evaluation.py','launch_e45w_repair.py',
        'watch_e45w_repair.py','tests/test_e45w_support_recovery.py',
        'tests/fixtures/e45w_fine_support_public.json')]
    frozen={str(p):sha(p) for p in sources}
    write(STATE/'source_manifest.json',dict(files=frozen,revision='W-support-r1',started=now(),
        original_manifest_sha256=sha(OUT/'component_source_manifest.json')))
    def integrity():
        for p,h in frozen.items():assert sha(p)==h,p
        for p,h in read(OUT/'component_source_manifest.json')['files'].items():assert sha(p)==h,p
    integrity()
    registry={name:dict(directory=str(OUT/'training'/name),attempt=0) for name in POOLS}
    handles={}; streams=[]
    wall_dir=OUT/'planning_wall_repair1'
    write(wall_dir/'repair_protocol.json',dict(revision='W-support-r1',
        description='Complete new wall run; original G succeeds identically; finer recovery only after genuine exhaustion',
        recovery_sources=frozen,old_results_preserved=True,old_successes_not_spliced=True))
    def start(command,log,cpus):
        stream=Path(log).open('a');streams.append(stream)
        env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):env[key]='1'
        return subprocess.Popen(['taskset','-c',','.join(map(str,cpus)),str(PYTHON),'-u']+command,
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env)
    wall=start(['code/run_e45w_repaired_planning.py','--mode','wall',
        '--workers','8','--cpus','48,49,50,51,52,53,54,55','--output',str(wall_dir),
        '--calibration',str(OUT/'planning_calibration_repair1')],OUT/'logs/planning_wall_repair1.log',list(range(48,56)))
    final=None;unlocked=False
    while True:
        integrity()
        for name,row in registry.items():
            d=Path(row['directory']);s=read(d/'status.json') if (d/'status.json').exists() else None
            if s and s['status']=='failed' and row['attempt']<2:
                terminate_failed(s)
                if name in handles:handles[name].wait(timeout=15)
                gpu,cpus=POOLS[name];attempt=row['attempt']+1
                label=f'{name}_repair{attempt}';newdir=OUT/'repaired_training'/label
                cmd=['code/run_e45w_repaired_training.py','--source',str(d),
                     '--run-name',label,'--device',f'cuda:{gpu}',
                     '--cpus',','.join(map(str,cpus)),'--eval-workers','8']
                process=start(cmd,OUT/'logs'/f'{label}.log',cpus)
                handles[name]=process
                row.update(directory=str(newdir),attempt=attempt,pid=process.pid,source=str(d),
                           started=now(),command=cmd,process_failed=False)
                row.pop('terminal_error',None)
                print('REPAIR START',name,process.pid,'from',d,flush=True)
                s=None
            row['status']=s
            if name in handles and handles[name].poll() is not None:
                row['exit_code']=handles[name].returncode
                if not s or s.get('status')!='complete':
                    row['process_failed']=True
                    if not s or s.get('status')!='failed':
                        row['terminal_error']='Repair process exited without completed training; inspect preserved log'
        write(STATE/'registry.json',registry)
        all_complete=all(r.get('status',{} ) and r['status']['status']=='complete' for r in registry.values())
        if all_complete and not unlocked:
            write(OUT/'test_scores_unlock_repair1.json',dict(unlocked=True,all_six_complete=True,
                registry_sha256=sha(STATE/'registry.json'),revision='W-support-r1',timestamp=now()))
            unlocked=True
        if all_complete and wall.poll() is not None and final is None:
            final=start(['code/run_e45w_repaired_evaluation.py','--workers','40'],
                OUT/'logs/final_evaluation_repair1.log',list(range(40)))
            print('FINAL START',final.pid,flush=True)
        wall_status=read(wall_dir/'status.json') if (wall_dir/'status.json').exists() else None
        final_status=read(OUT/'final_evaluation_repair1/status.json') if (OUT/'final_evaluation_repair1/status.json').exists() else None
        data=dict(timestamp=now(),state='running',registry=registry,wall_pid=wall.pid,
            wall_exit_code=wall.poll(),wall_status=wall_status,final_pid=final.pid if final else None,
            final_status=final_status,test_scores_unlocked=unlocked)
        if final and final.poll() is not None:
            data.update(state='finished',final_exit_code=final.returncode)
            write(STATE/'progress.json',data);break
        exhausted=[name for name,r in registry.items() if r.get('terminal_error') or
            (r.get('process_failed') and r['attempt']>=2)]
        if exhausted:
            data.update(state='repair_failure_requires_review',failed=exhausted)
        write(STATE/'progress.json',data)
        time.sleep(10)


if __name__=='__main__':main()
