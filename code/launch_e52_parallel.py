"""tmux-owned dependency scheduler: independent experiments overlap teacher work.

Reuses pilot episodes in formal aggregation; does not restart an observed job.
A failed pilot gate blocks new neural formal runs and retains diagnostic data.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import tarfile

from run_e52_experiments import summarize

ROOT=Path(__file__).resolve().parents[1]
MODES=['e50','geometry','nar','nae','both']


def atomic(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,ensure_ascii=False));temp.replace(path)


class Pipeline:
    def __init__(self,out,teacher_workers,independent_workers,neural_workers):
        self.out=out;self.lock=threading.Lock();self.state={};self.handles={}
        self.tw,self.iw,self.nw=teacher_workers,independent_workers,neural_workers
        self.cases=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'current/e50_deployment_audit_20261117/formal_random_cases').glob('*/*.json'))
        self.pilot=[p for group in ('wall','corner','sparse') for p in sorted(p for p in self.cases if Path(p).parent.name==group)[:3]]
        self.ckpt=out/'frozen_nar_nae.pt'

    def save_state(self):
        # Caller holds self.lock.
        atomic(self.out/'status.json',dict(supervisor_pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),stages=self.state))

    def command(self,name,args):
        log=self.out/f'{name}.log'
        cmd=[sys.executable,'-u',*map(str,args)]
        with log.open('a',buffering=1) as stream:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            with self.lock:
                self.handles[name]=proc
                self.state[name]=dict(state='running',pid=proc.pid,command=cmd,log=str(log),started_at=time.time())
                self.save_state()
            print(f'START {name}: pid={proc.pid}; {log}',flush=True)
            rc=proc.wait()
        with self.lock:
            self.state[name].update(state='complete' if rc==0 else 'failed',exit_code=rc,finished_at=time.time())
            self.save_state()
        print(f'END {name}: exit={rc}',flush=True)
        if rc: raise RuntimeError(f'{name} failed (exit {rc}); inspect {log}')

    def experiment(self,name,stage,modes,workers,*,exclude=(),communication=False):
        args=['code/run_e52_experiments.py','--stage',stage,'--modes',*modes,'--workers',workers,'--output',self.out/name]
        if any(m in ('nar','nae','both') for m in modes):args+=['--checkpoint',self.ckpt]
        if exclude:args+=['--exclude-cases',*exclude]
        if communication:args+=['--delay',2,'--drop-cycle',2,'--drop-cycle',3]
        self.command(name,args)

    def teacher(self):
        self.experiment('teacher','teacher',['teacher'],self.tw)
        logs=[Path(row['run']) for row in json.loads((self.out/'teacher/summary.json').read_text())['rows']]
        logs=sorted(logs)
        if len(set(logs))!=len(logs):raise RuntimeError('duplicate teacher run records')
        if len(logs)!=30:raise RuntimeError(f'expected 30 teacher records, got {len(logs)}')
        # Failed native searches/collisions must be examined before training labels.
        self.command('teacher_gate',['code/check_e52_results.py',self.out/'teacher/summary.json'])
        self.command('training',['code/run_e52_train_nar_nae.py','--logs',*logs,'--output',self.ckpt,'--epochs',200])
        # Verify the frozen checkpoint's initial-state split before any test run.
        metadata=json.loads(self.ckpt.with_suffix('.json').read_text())
        from run_e52_train_nar_nae import initial_id
        test_ids={initial_id(json.loads((ROOT/p).read_text())) for p in self.cases}
        if test_ids & set(metadata['train_initial_ids']+metadata['validation_initial_ids']):
            raise RuntimeError('training/validation overlap formal initial states')
        atomic(self.out/'frozen_checkpoint.json',dict(path=str(self.ckpt),sha256=hashlib.sha256(self.ckpt.read_bytes()).hexdigest(),metadata=metadata))

    def independent(self):
        self.experiment('independent_formal','formal',['e50','geometry'],self.iw)
        self.experiment('independent_communication','pilot',['e50','geometry'],18,communication=True)

    def merge(self,name,parts,stage,cases):
        rows=[]
        for part in parts:
            data=json.loads((self.out/part/'summary.json').read_text())
            rows.extend(r for r in data['rows'] if r['case'] in cases)
        keys=[(r['case'],r['mode']) for r in rows]
        expected={(p,m) for p in cases for m in MODES}
        if len(set(keys))!=len(keys) or set(keys)!=expected:
            raise RuntimeError(f'{name}: incomplete/duplicate pairing ({len(rows)} vs {len(expected)})')
        target=self.out/name;target.mkdir(exist_ok=True)
        manifest=dict(stage=stage,cases=cases,cycles=160,modes=MODES,jobs=[r['run'] for r in rows],
            checkpoint=str(self.ckpt),checkpoint_sha256=hashlib.sha256(self.ckpt.read_bytes()).hexdigest(),
            reused_pilot=True,paired_seed=17,rootpuct_simulations=20,environment_horizon=160)
        summarize(target,manifest,rows)
        return rows

    def diagnostic_analysis(self):
        records=[]
        for summary in self.out.glob('*/summary.json'):
            for row in json.loads(summary.read_text())['rows']:
                if row['collision_cycles'] or row['minco_infeasible'] or row['rootpuct_failures'] or row['terminal_opponent_skips'] or row['lifecycle_errors']:
                    data=json.loads(Path(row['run']).read_text())
                    examples=[x for x in data['logs'] if x['pursuer_collision'] or any(not m['feasible'] for m in x['minco'].values())][:3]
                    records.append(dict(summary=str(summary),result=row,examples=examples))
        atomic(self.out/'failure_analysis.json',dict(invalid_episodes=records,note='Preserved experiment evidence; neural formal stage will not start after a failed pilot gate.'))

    def run(self):
        # Two independent branches: initially 30 teacher + 24 evaluation workers.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as threads:
            teacher=threads.submit(self.teacher);independent=threads.submit(self.independent)
            teacher.result()
            # Frozen-network communication is independent of the clean pilot.
            communication=threads.submit(self.experiment,'neural_communication','pilot',['nar','nae','both'],self.nw,communication=True)
            self.experiment('neural_pilot','pilot',['nar','nae','both'],self.nw)
            independent.result()
            self.merge('pilot',['independent_formal','neural_pilot'],'pilot',self.pilot)
            try:
                self.command('pilot_gate',['code/check_e52_results.py',self.out/'pilot/summary.json'])
            except RuntimeError:
                communication.result();self.merge('communication',['independent_communication','neural_communication'],'communication',self.pilot)
                self.diagnostic_analysis();raise
            self.experiment('neural_formal_remaining','formal',['nar','nae','both'],self.nw,exclude=self.pilot)
            communication.result()
        rows=self.merge('formal',['independent_formal','neural_pilot','neural_formal_remaining'],'formal',self.cases)
        self.merge('communication',['independent_communication','neural_communication'],'communication',self.pilot)
        from make_e50_distributed_gif import render
        for group in ('wall','corner','sparse'):
            record=next(r for r in rows if r['mode']=='both' and Path(r['case']).parent.name==group)
            render(record['run'],record['case'],self.out/'formal'/f'{group}_both.gif')
        with self.lock:
            self.state['pipeline']=dict(state='complete',finished_at=time.time());self.save_state()
        self.diagnostic_analysis()
        print('All scheduled stages completed. Acceptance gates remain in formal/summary.json.',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--teacher-workers',type=int,default=30);p.add_argument('--independent-workers',type=int,default=24)
    p.add_argument('--neural-workers',type=int,default=27);a=p.parse_args()
    os.chdir(ROOT);out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'status.json').exists():raise RuntimeError('existing production state: inspect before restarting; no automatic reruns')
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'code').rglob('*.py') if '__pycache__' not in p.parts}
    atomic(out/'source_manifest.json',dict(files=sources,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    with tarfile.open(out/'source_snapshot.tar.gz','w:gz') as archive:
        for path in sources:archive.add(ROOT/path,arcname=path)
    pipeline=Pipeline(out,a.teacher_workers,a.independent_workers,a.neural_workers)
    atomic(out/'schedule.json',dict(teacher_episodes=30,formal_episodes=150,communication_episodes=45,
        pilot_episodes_reused_in_formal=45,unique_total_episodes=225,teacher_workers=a.teacher_workers,
        independent_workers=a.independent_workers,neural_workers=a.neural_workers,
        initial_peak_workers=a.teacher_workers+a.independent_workers,dependent_peak_workers=2*a.neural_workers,
        dependencies='teacher -> training -> neural pilot/communication; pilot gate -> remaining neural formal',
        tmux_session='train',source_manifest=str(out/'source_manifest.json')))
    try:pipeline.run()
    except BaseException:
        with pipeline.lock:
            pipeline.state['pipeline']=dict(state='failed',error=traceback.format_exc(),finished_at=time.time());pipeline.save_state()
        pipeline.diagnostic_analysis();raise


if __name__=='__main__':main()
