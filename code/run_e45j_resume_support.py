"""J amendment01 coordinator; original registered runner stays immutable."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
import argparse
import ctypes
import fcntl
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
import run_e45j_repaired_benchmark as j
from learning.e45j_support125 import install,TeamBelief125

ORIGINAL_CONFIGURE=j.configure
ROOT=j.ROOT;OUT=j.OUT
FILES=('run_e45j_resume_support.py','learning/e45j_support125.py','tests/test_e45j_support125.py','E45J_SUPPORT125_AMENDMENT_20261031.md','review_e45j_repaired_benchmark.py')


def configure(output):
    ORIGINAL_CONFIGURE(output);install(Path(output)/'support125_evidence')


j.configure=configure


def verify():
    j.verify();d=j.read(OUT/'amendment_01.json')
    assert d['original_experiment_sha256']==j.sha(OUT/'experiment.json')
    for p,h in d['source_hashes'].items():
        if j.sha(p)!=h:raise RuntimeError('Amendment source changed '+p)
    return d


def public_replay(row):
    packet=j.read(row['episode']);case=j.read(row['case']);b=TeamBelief125(case['map'],row['seed'])
    for o in case['history'][:-1]:b.update(o)
    for frame in packet['trace']:
        b.update(frame['observation'])
        if b.signature()!=frame['belief_signature']:raise RuntimeError('Existing fixed trace changed under amendment '+row['key'])
    return dict(key=row['key'],frames=len(packet['trace']),episode_sha256=j.sha(row['episode']))


def prove():
    configure(OUT);j.verify();rows=[j.read(p) for p in sorted((OUT/'test/cache').glob('*.json')) if not p.name.endswith('.failure.json')]
    assert len(rows)==336;pool,queue=j.pool_for(j.choose_cpus(12))
    try:proof=list(pool.map(public_replay,rows))
    finally:pool.shutdown(wait=True,cancel_futures=True);queue.close()
    j.save(OUT/'support125_existing_trace_proof.json',dict(episodes=len(proof),frames=sum(r['frames'] for r in proof),
        exact_public_signatures=True,receipts=proof))
    print('All336 saved fixed-controller trajectories remain bit-identical under amended public replay',flush=True)


def fault_replay():
    configure(OUT);f=OUT/'learning/MAPPO_APF_29';saved=torch.load(j.read(f/'resume.json')['checkpoint'],map_location='cpu',weights_only=False)
    rng=np.random.default_rng();rng.bit_generator.state=saved['numpy_rng'];entries=j.read(OUT/'bank/manifest.json')['train'];chosen=None
    for _ in range(24):
        e=entries[int(rng.integers(len(entries)))];world='AB' if rng.random()<.5 else 'BA'
        opp='rootpuct' if rng.random()<.5 else 'fixed_D_kernel'
        req=j.request(e,world,opp,'MAPPO_APF',29,'training',episode_seed=rng.integers(2**31))
        if req['seed']==523697279:chosen=req
    if chosen is None:raise RuntimeError('Cannot reconstruct original failed task from saved RNG')
    ep=j.rollout(dict(kind='MAPPO_APF',actor=saved['actor'],request=chosen))
    assert ep['summary']['public_replay_exact']
    j.save(OUT/'support125_fault_replay.json',dict(request=chosen,checkpoint_sha256=j.sha(j.read(f/'resume.json')['checkpoint']),
        steps=len(ep['obs']),summary=ep['summary'],public_replay_exact=True,discarded_or_resampled=False))
    print('Original failing rollout completed, same RNG/case/opponent/actor, exact public replay',flush=True)


def register():
    j.verify()
    if (OUT/'amendment_01.json').exists():verify();return
    assert j.read(OUT/'support125_existing_trace_proof.json')['exact_public_signatures']
    assert j.read(OUT/'support125_fault_replay.json')['public_replay_exact']
    archive=OUT/'interruption_01';archive.mkdir(exist_ok=True);runs=[]
    for name in ('status.json','execution.json','REPORT.md','review_status.json','pipeline.log','events.jsonl'):
        p=OUT/name
        if p.exists():shutil.copy2(p,archive/name)
    for kind in j.RL:
        for seed in j.SEEDS:
            f=OUT/'learning'/f'{kind}_{seed}';dest=archive/'learning'/f'{kind}_{seed}';dest.mkdir(parents=True,exist_ok=True)
            for name in ('latest.pt','resume.json','selected.json','initial.json','status.json','run.log','events.jsonl','validation.json','gates.json'):
                p=f/name
                if p.exists():shutil.copy2(p,dest/name)
            saved=torch.load(f/'latest.pt',map_location='cpu',weights_only=False)
            assert saved['steps']==j.read(f/'resume.json')['steps']
            updates=sorted((f/'updates').glob('*.json'))
            assert j.read(updates[-1])['steps']==saved['steps'],'Uncommitted update needs separate archival before resume'
            runs.append(dict(kind=kind,seed=seed,steps=saved['steps'],update=saved['update'],phase=saved['curriculum']['phase'],
                original_checkpoint_sha256=j.sha(f/'latest.pt'),archive_checkpoint=str(dest/'latest.pt')))
    data=dict(amendment='E45-J-01',registered_at=time.strftime('%FT%T%z'),original_experiment_sha256=j.sha(OUT/'experiment.json'),
        scope='public-only 0.125 support recovery after existing 0.5/0.25 failure; same interface for every arm and scoring replay',
        discretization_changed=True,earlier_training_steps=sum(r['steps'] for r in runs),resume_points=runs,
        unchanged_budget=500000,formal_episodes=5040,existing_fixed_episodes_reused=336,
        source_hashes={str(ROOT/'code'/n):j.sha(ROOT/'code'/n) for n in FILES})
    j.save(OUT/'amendment_01.json',data)
    for name in FILES:
        p=OUT/'amendment_01_source'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'code'/name,p)
    shutil.copy2(ROOT/'code/E45J_SUPPORT125_AMENDMENT_20261031.md',OUT/'AMENDMENT_01.md')
    print(json.dumps(data,ensure_ascii=False),flush=True)


def pipeline():
    verify();configure(OUT);lock=(OUT/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpus=j.choose_cpus(54);assert len(cpus)==54;children=[];schedule=[]
    try:
        for index,(kind,seed) in enumerate((k,s) for k in j.RL for s in j.SEEDS):
            folder=OUT/'learning'/f'{kind}_{seed}';allocated=cpus[index*7:(index+1)*7];device=('cuda:2','cuda:3')[index%2]
            if (folder/'decision.json').exists():continue
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'train','--algorithm',kind,'--seed',str(seed),
                '--device',device,'--cpus',','.join(map(str,allocated)),'--parent',str(os.getpid())]
            stream=(folder/'run.log').open('a');child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT);stream.close()
            children.append(child);schedule.append(dict(kind=kind,seed=seed,pid=child.pid,cpus=allocated,device=device))
        j.save(OUT/'execution.json',dict(training=schedule,sampler_cores=36,parent_cores=6,evaluation_workers=12,
            phases_separate=True,other_jobs_terminated=False,amendment='01'))
        j.emit('training_resumed_after_support_repair',runs=schedule,amendment_sha256=j.sha(OUT/'amendment_01.json'))
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Amended J trainer failed; inspect logs')
            time.sleep(10)
        if any(c.returncode!=0 for c in children):raise RuntimeError('Amended trainer exit failure')
        verify();fixed=[j.read(p) for p in sorted((OUT/'test/cache').glob('*.json')) if not p.name.endswith('.failure.json') and j.read(p).get('arm') in ('APF','APF_I')]
        assert len(fixed)==336
        remaining=tuple(a for a in j.MAIN+j.AUX if a not in ('APF','APF_I'))
        learned=j.common.evaluate(j.tasks_for(remaining),cpus[:12],'remaining_benchmark');rows=fixed+learned
        j.summarize([r for r in rows if r['arm'] in j.MAIN],'nine_family_benchmark')
        j.summarize(rows,'all_controls_benchmark');j.benchmark_statistics(rows);verify()
        decisions=[j.read(OUT/'learning'/f'{k}_{s}'/'decision.json') for k in j.RL for s in j.SEEDS]
        assert len(rows)==5040
        j.save(OUT/'decision.json',dict(status='complete',episodes=len(rows),expected=5040,main_families=9,total_families=14,
            gate_failed_runs=sum(d['learning_gate_failed'] for d in decisions),training_steps=sum(d['steps'] for d in decisions),
            convergence_claim=False,amendment_sha256=j.sha(OUT/'amendment_01.json'),common_public_filter='J-01 adaptive 0.125 fallback'))
        j.emit('complete',episodes=len(rows),training_steps=sum(d['steps'] for d in decisions),amendment='01')
    except BaseException:
        for c in children:
            if c.poll() is None:c.terminate()
        raise
    finally:lock.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prove','fault_replay','register','pipeline','train'))
    p.add_argument('--algorithm',choices=j.RL,default='RMAPPO');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--device',default='cuda:2');p.add_argument('--cpus',default='');p.add_argument('--parent',type=int)
    args=p.parse_args();torch.set_num_threads(1);configure(OUT)
    if args.parent:
        ctypes.CDLL(None).prctl(1,signal.SIGTERM)
        if os.getppid()!=args.parent:raise RuntimeError('Resume coordinator exited')
    try:
        if args.command=='prove':prove()
        elif args.command=='fault_replay':fault_replay()
        elif args.command=='register':register()
        elif args.command=='train':verify();j.train(args)
        else:pipeline()
    except BaseException as exc:
        j.common.emit(OUT/'learning'/f'{args.algorithm}_{args.seed}' if args.command=='train' else OUT,'failed',error=repr(exc));raise


if __name__=='__main__':main()
