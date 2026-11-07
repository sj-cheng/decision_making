"""Evaluation-only amendment: exact frozen recovery exception is a failure.

The original algorithms, runner, scenarios, parameters and task seeds stay
unchanged. Original request 46 is registered from its saved evidence, never
executed again as an evaluation request. Verification replay is separate.
"""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[_k]='1'
import argparse
import collections
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
import torch
import run_l45_occluded_comparison as base

MESSAGE='No lawful belief recovery support'
ORIGINAL_KEY='long_wall_H4_v1.3_3720602974_B_s1_G'


def rng_globals():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),
                cuda='not_used_by_frozen_algorithms; no_CUDA_state_initialized')


def recognized(error):
    if type(error) is not RuntimeError or str(error)!=MESSAGE:return False
    frame=error.__traceback__
    while frame.tb_next is not None:frame=frame.tb_next
    return Path(frame.tb_frame.f_code.co_filename).resolve()==base.ROOT/'code/solvers/local_pursuit.py' and frame.tb_frame.f_code.co_name=='update'


def fingerprint_controller(controller):
    return hashlib.sha256(controller['particles'].tobytes()+controller['filter_rng'].tobytes()+controller['rng'].tobytes()+
        json.dumps(controller['observations'],sort_keys=True).encode()+str(controller['recoveries']).encode()).hexdigest()


def commit(task,output,state,trace,cost,failure=None,final=None,wall_seconds=None):
    row=base.metrics(task,state,trace,cost)
    if failure is not None:
        row.update(capture=0,failure='estimator_recovery_failure',method_execution_failure=1,
                   completion_cost=80-row['handoff_step'],failure_step=int(state[8,0]),
                   exception=MESSAGE,post_exception_state_captured=failure['post'] is not None,
                   failed_decision_wall_ms=failure.get('failed_decision_wall_ms'),
                   original_failure_registered_without_rerun=bool(failure.get('registered_original')))
    else:row['method_execution_failure']=0
    row['wall_seconds']=wall_seconds
    path=output/'episodes'/(task['key']+'.pt')
    base.save_torch(path,dict(task=task,trajectory=trace,final=final,failure=failure))
    row.update(episode_path=str(path),episode_sha256=base.sha(path))
    base.put(output/'cache'/(task['key']+'.json'),row)
    return row


def register_original(output):
    task=next(t for t in base.tasks() if t['key']==ORIGINAL_KEY)
    existing=base.cached(task,output)
    if existing is not None:
        if existing['failure']!='estimator_recovery_failure':raise RuntimeError('Original failure registration changed')
        return existing
    pending=output/'pending'/(ORIGINAL_KEY+'.pt')
    saved=base.load_torch(pending)
    audit=json.loads((output/'failure_audit.json').read_text())
    if audit['key']!=ORIGINAL_KEY or int(saved['state'][8,0])!=22 or len(saved['trace'])!=13:raise RuntimeError('Original evidence mismatch')
    # Registration reads only existing state; no setup, act, update or search.
    observation=base.Observation(22,saved['state'][4:6,0].copy(),None,58)
    failure=dict(exception=MESSAGE,registered_original=True,observation=observation,
        pre=copy.deepcopy(saved),post=None,
        active_scientific_random_states='pre.controller.rng; pre.controller.filter_rng; pre.opponent.solver/problem',
        post_state_availability='original process did not persist it; any later reconstruction is labeled verification-derived',
        unused_global_rng_availability='not recorded by original runner; frozen algorithms use explicit saved scientific streams',
        original_traceback=(output/'run.log').read_text().split('Traceback (most recent call last):')[-1],
        original_checkpoint_sha256=base.sha(pending),original_failure_audit_sha256=base.sha(output/'failure_audit.json'))
    return commit(task,output,saved['state'],saved['trace'],None,failure=failure,final=copy.deepcopy(saved),wall_seconds=None)


def episode(task,output):
    p,state,own,other,cost=base.setup(task);trace=[]
    pending=output/'pending'/(task['key']+'.pt')
    if pending.exists():
        saved=base.load_torch(pending);state=saved['state'];own.restore(saved['controller']);other.set_rng_state(p,saved['opponent']);trace=saved['trace']
    started=time.perf_counter();failure=None
    while not p.is_terminal(state):
        before=base.snapshot(state,own,other,p,[]);before['global_rng']=rng_globals()
        decision_begin=time.perf_counter()
        try:
            nxt,record=base.step(p,state,own,other)
        except Exception as error:
            if not recognized(error):raise
            failed_ms=(time.perf_counter()-decision_begin)*1000
            # Own update failed before the actual opponent or world advanced.
            after=base.snapshot(state,own,other,p,[]);after['global_rng']=rng_globals()
            if before['opponent']!=after['opponent']:raise RuntimeError('Opponent advanced during estimator exception')
            measurement=own.public.observe(int(state[8,0]),state[4:6,0],state[:2,0])
            failure=dict(exception=MESSAGE,registered_original=False,observation=measurement,
                pre=before,post=after,traceback=traceback.format_exc(),
                failed_decision_wall_ms=failed_ms,
                active_scientific_random_states='pre/post.controller.rng/filter_rng and opponent.solver/problem',
                global_random_states='pre/post.global_rng python,numpy,torch_cpu; CUDA unused')
            break
        state=nxt;trace.append(record)
        base.save_torch(pending,base.snapshot(state,own,other,p,trace))
    return commit(task,output,state,trace,cost,failure=failure,final=base.snapshot(state,own,other,p,[]),wall_seconds=time.perf_counter()-started)


def verify_frozen(output):
    record=json.loads((output/'experiment.json').read_text())
    if base.sha(base.__file__)!=record['runner_sha256']:raise RuntimeError('Original runner changed')
    if any(base.sha(path)!=value for path,value in record['algorithm_sources'].items()):raise RuntimeError('Frozen algorithm changed')
    for name in ('frozen.json','DELIVERY_SHA256.json'):
        lock=json.loads((base.SOURCE/name).read_text())
        if any(base.sha(base.SOURCE/path)!=value for path,value in lock['files'].items()):raise RuntimeError('Frozen scenarios changed')
    if json.loads((output/'requests.json').read_text())!=base.tasks():raise RuntimeError('Request list or seeds changed')


def prepare(output):
    verify_frozen(output)
    path=output/'continuation_protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['framework_sha256']!=base.sha(__file__):raise RuntimeError('Continuation framework changed')
        return
    archive=output/'pause_archive';archive.mkdir(exist_ok=True)
    for name in ('REPORT.md','status.json','failure_audit.json'):
        (archive/name).write_bytes((output/name).read_bytes())
    cached={str(p.relative_to(output)):base.sha(p) for p in (output/'cache').glob('*.json')}
    if len(cached)!=45:raise RuntimeError('Expected exactly original 45 completed requests')
    base.put(path,dict(evaluation_only_amendment=True,authorization='User: frozen estimator recovery exception counts as method failure; retain original request 46 without rerun',
        exception_whitelist=dict(type='RuntimeError',message=MESSAGE,source='solvers/local_pursuit.py::Controller.update'),
        unknown_exceptions='pause and propagate',cost='original full remaining steps from handoff; capture=0',
        scientific_algorithms_changed=False,parameters_changed=False,scenarios_changed=False,
        original_experiment_sha256=base.sha(output/'experiment.json'),original_cached_rows=cached,
        original_request46_checkpoint_sha256=base.sha(output/'pending'/(ORIGINAL_KEY+'.pt')),
        framework_sha256=base.sha(__file__),bootstrap_unit='24 public history groups',draws=10000,seed=61020261025,
        original_post_exception_state='not captured; retained as unavailable, verification reconstruction separate'))


def evaluate(output):
    prepare(output);register_original(output)
    rows=[];started=time.perf_counter();reused=0
    for i,task in enumerate(base.tasks()):
        row=base.cached(task,output)
        if row is not None:reused+=1
        else:
            try:row=episode(task,output)
            except Exception as error:
                base.put(output/'status.json',dict(stage='paused_unknown_exception',completed=i,total=384,key=task['key'],error=repr(error),pid=None));raise
        rows.append(row)
        status=dict(stage='evaluate',completed=i+1,total=384,reused=reused,
            estimator_failures=sum(r['failure']=='estimator_recovery_failure' for r in rows),
            current=task['key'],elapsed_seconds=time.perf_counter()-started,pid=os.getpid())
        base.put(output/'status.json',status);print(json.dumps(status),flush=True)
    base.put(output/'results.json',rows)
    validate_records(output)


def validate_records(output):
    tasks=base.tasks();rows=json.loads((output/'results.json').read_text());idx={r['key']:r for r in rows}
    if len(rows)!=384 or len(idx)!=384 or set(idx)!={t['key'] for t in tasks}:raise RuntimeError('Missing or duplicate samples')
    failure_count=0
    for task in tasks:
        row=base.cached(task,output);record=base.load_torch(row['episode_path'])
        if row['failure']=='estimator_recovery_failure':
            failure_count+=1
            if row['capture']!=0 or row['completion_cost']!=80-row['handoff_step'] or record['failure'] is None:raise RuntimeError('Bad failure scoring')
            if record['failure']['pre']['opponent']!=record['final']['opponent']:raise RuntimeError('Failure advanced opponent RNG')
        if task['key']==ORIGINAL_KEY and not row.get('original_failure_registered_without_rerun'):raise RuntimeError('Original failure was rerun')
    protocol=json.loads((output/'continuation_protocol.json').read_text())
    if any(base.sha(output/path)!=value for path,value in protocol['original_cached_rows'].items()):raise RuntimeError('Original successful/timeout records changed')
    base.put(output/'integrity.json',dict(passed=True,requests=384,public_histories=24,all_failure_samples_retained=True,
        estimator_failures=failure_count,original45_unchanged=True,original46_registered_without_rerun=True))


def verify_task(task,output):
    row=base.cached(task,output);saved=base.load_torch(row['episode_path']);p,state,own,other,_=base.setup(task)
    for original in saved['trajectory']:
        nxt,actual=base.step(p,state,own,other,fixed=original['own']['simulations'])
        for name in ('action','next_state','particles','filter_rng','search_rng'):
            if not np.array_equal(original[name],actual[name]):raise RuntimeError('Replay differs '+task['key']+' '+name)
        if actual['opponent_rng']!=original['opponent_rng']:raise RuntimeError('Opponent RNG differs')
        state=nxt
    reproduced=False
    if row['failure']=='estimator_recovery_failure':
        failure=saved['failure'];pre=failure['pre']
        if not np.array_equal(state,pre['state']) or fingerprint_controller(own.snapshot())!=fingerprint_controller(pre['controller']):raise RuntimeError('Failure prefix differs')
        if 'global_rng' in pre:
            random.setstate(pre['global_rng']['python']);np.random.set_state(pre['global_rng']['numpy']);torch.set_rng_state(pre['global_rng']['torch_cpu'])
        try:base.step(p,state,own,other)
        except Exception as error:
            if not recognized(error):raise
            reproduced=True
        if not reproduced:raise RuntimeError('Recorded estimator exception did not reproduce')
        after=base.snapshot(state,own,other,p,[])
        after['global_rng']=rng_globals()
        if failure['post'] is not None and fingerprint_controller(after['controller'])!=fingerprint_controller(failure['post']['controller']):raise RuntimeError('Post-exception RNG/particles differ')
        if after['opponent']!=pre['opponent']:raise RuntimeError('Replay exception advanced opponent')
        if failure['post'] is not None:
            original_global=failure['post']['global_rng'];actual_global=after['global_rng']
            if original_global['python']!=actual_global['python'] or not torch.equal(original_global['torch_cpu'],actual_global['torch_cpu']):raise RuntimeError('Global RNG differs')
            aa,bb=original_global['numpy'],actual_global['numpy']
            if aa[0]!=bb[0] or not np.array_equal(aa[1],bb[1]) or aa[2:]!=bb[2:]:raise RuntimeError('NumPy RNG differs')
        base.save_torch(output/'failure_reproduction'/(task['key']+'.pt'),dict(verification_derived=True,original_post_state_was_available=failure['post'] is not None,post=after))
    return dict(key=task['key'],passed=True,completed_steps=len(saved['trajectory']),estimator_failure_reproduced=reproduced,
                original46_not_reevaluated=(task['key']==ORIGINAL_KEY))


def replay(output,workers=8):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    validate_records(output);checks=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(verify_task,t,output) for t in base.tasks()]
        for job in as_completed(jobs):
            checks.append(job.result());base.put(output/'status.json',dict(stage='replay',completed=len(checks),total=384,workers=workers,pid=os.getpid()))
            print(json.dumps(dict(stage='replay',completed=len(checks),total=384)),flush=True)
    base.put(output/'replay.json',dict(passed=True,episodes=sorted(checks,key=lambda r:r['key']),
        fixed_recorded_simulations=True,estimator_failures_reproduced=sum(c['estimator_failure_reproduced'] for c in checks),
        all_failure_reproductions_match=True,workers=workers,original46_registration_preserved=True))


def report(output):
    base.report(output)
    rows=json.loads((output/'results.json').read_text());index={(r['history_id'],r['world'],r['stream'],r['method']):r for r in rows}
    histories=sorted({r['history_id'] for r in rows});rng=np.random.default_rng(61020261025);draws=rng.integers(0,24,(10000,24))
    rates={};failure_comparisons={}
    for m in base.METHODS:
        vals=np.array([np.mean([index[(h,w,s,m)]['failure']=='estimator_recovery_failure' for w in ('A','B') for s in (0,1)]) for h in histories])
        rates[m]=dict(count=sum(r['method']==m and r['failure']=='estimator_recovery_failure' for r in rows),rate=float(vals.mean()),ci95=np.quantile(vals[draws].mean(1),[.025,.975]).tolist())
    for a,b in (('G','L'),('T','G'),('V','T'),('V','G')):
        vals=np.array([np.mean([(index[(h,w,s,a)]['failure']=='estimator_recovery_failure')-(index[(h,w,s,b)]['failure']=='estimator_recovery_failure') for w in ('A','B') for s in (0,1)]) for h in histories])
        failure_comparisons[a+'-'+b]=dict(delta=float(vals.mean()),ci95=np.quantile(vals[draws].mean(1),[.025,.975]).tolist())
    base.put(output/'estimator_reliability.json',dict(rates=rates,paired_failure_rate_differences=failure_comparisons,
        interpretation='All-sample capture differences include execution reliability and closed-loop tracking/planning. Conditional non-crash subsets are selected outcomes and do not isolate causal search contributions.'))
    text=(output/'REPORT.md').read_text()
    text+='\n**评估框架修订与估计器可靠性。**\n\n只新增异常记分，不修改估计器或搜索。已知恢复异常计捕获失败，成本仍为接管时完整剩余步数；其他异常不吞掉。第46个请求按原证据登记，没有重新评估，异常后缺失状态明确标为未采集，重放推导状态单独保存。\n\n|方法|恢复失败次数/96|失败率95%区间|\n|---|---:|---|\n'
    for m,v in rates.items():text+=f'|{m}|{v["count"]}/96|[{100*v["ci95"][0]:.2f}%, {100*v["ci95"][1]:.2f}%]|\n'
    text+='\n可靠性与搜索贡献须区分：全样本收益衡量整套方法的实际效果，包括恢复失败。G/T/V共用估计器，但行动改变观测历史，所以各自失败率必须实测。删除崩溃样本后的比较不能作为搜索或新候选的独立因果贡献。\n'
    text+='\n重放既核验正常前缀，也逐次尝试复现记录的恢复异常；详情见replay.json和failure_reproduction。本文是固定困难场景开发结果，尚非独立确认。\n'
    (output/'REPORT.md').write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('resume','replay','report','register'))
    parser.add_argument('--output',type=Path,default=base.OUTPUT);args=parser.parse_args();output=args.output.resolve()
    with (base.ROOT/'current/.l45_local_pursuit_serial.lock').open('a+') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another timed evaluation is active')
        prepare(output)
        if args.command=='register':register_original(output);return
        if args.command=='report':report(output);return
        base.warmup()
        if args.command=='replay':replay(output);return
        try:evaluate(output);replay(output);report(output)
        except Exception as error:
            status=json.loads((output/'status.json').read_text())
            status.update(stage='paused_unknown_exception',error=repr(error),pid=None)
            base.put(output/'status.json',status);raise


if __name__=='__main__':main()
