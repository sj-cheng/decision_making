"""E45-E: frozen Single/FACT/Joint proposal comparison with learned value off."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
os.environ.setdefault('TZ','Asia/Shanghai')
import argparse
import collections
import concurrent.futures as futures
import ctypes
import fcntl
import hashlib
import json
import multiprocessing as mp
import re
import signal
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import run_e45c_budget as old
import run_e45b_role_free as base
from e45c_belief_diagnostics import install, SUPPORT_ERRORS
from e45c_capture_audit import audit_simulated
from solvers.e45b_joint_nte import Controller
from run_e44_parallel import choose_cpus

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'current/e45c_budget_scaling_20261028'
OUT=ROOT/'current/e45e_value_off_20261029'
SEEDS=(17,29,43)
ARMS=('SINGLE_PRIOR','FACT_PRIOR','JOINT_PRIOR')
NEW_ARMS=ARMS[:2]
MODEL_KINDS=dict(zip(ARMS,('single','factorized','joint')))
BASE_SEED=945620261029
CPUS=[]
SMOKE=False
SOURCES=('run_e45e_value_off.py','E45E_VALUE_OFF_PROTOCOL_20261029.md','tests/test_e45e_value_off.py')








def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):return old.save(Path(path),data)
def family(row):return row['method'].rstrip('0123456789')


def factory(case,task):
    arm=family(task)
    if arm not in ARMS:raise ValueError(arm)
    return Controller(case['map'],case['speed'],task['seed'],task['checkpoint'],
        use_policy=True,use_value=False,direct=False,budget_ms=task['budget_ms'])


def configure():
    base.OUT=OUT;base.SMOKE=False;base.make_controller=factory
    base.audit_simulated=audit_simulated
    install(OUT/'belief_failures')


def emit(stage,**values):
    row=dict(pid=os.getpid(),timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),stage=stage,**values)
    save(OUT/'status.json',row)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)


def verify():
    config=read(OUT/'experiment.json')
    for section in ('source_hashes','protected_hashes','bank_hashes'):
        for path,digest in config[section].items():
            if sha(path)!=digest:raise RuntimeError('Frozen E45-E artifact changed: '+path)
    return config


def prepare():
    if (OUT/'experiment.json').exists():return verify()
    OUT.mkdir(parents=True,exist_ok=True);old.verify()
    previous=read(SOURCE/'experiment.json')
    protected=previous['protected_hashes'].copy()
    for name in ('experiment.json','models_frozen.json','bank/manifest.json',
                 'evaluation/results_with_engineering_failures.json'):
        protected[str(SOURCE/name)]=sha(SOURCE/name)
    for name in ('REPAIR_CAPTURE_AUDIT_20261028.json','REPAIR_SUPPORT_DIAGNOSTICS_20261028.json'):
        repair=read(SOURCE/name);protected.update(repair['extra_hashes'])
        protected.update({p:v['repaired_sha256'] for p,v in repair['overrides'].items()})
        protected[str(SOURCE/name)]=sha(SOURCE/name)
    for p in previous['source_hashes']:protected.setdefault(p,sha(p))
    frozen=read(SOURCE/'models_frozen.json')['files'];models={}
    for arm,kind in MODEL_KINDS.items():
        models[arm]={}
        for seed in SEEDS:
            path=str(SOURCE/f'learning/x20/round_12/models/{kind}_{seed}.npz')
            if sha(path)!=frozen[path]:raise RuntimeError('Checkpoint changed: '+path)
            models[arm][str(seed)]=path;protected[path]=frozen[path]
    manifest=read(SOURCE/'bank/manifest.json')['entries']
    validation=[e for e in manifest if e['split']=='validation']
    entries=[e for e in manifest if e['split']=='test']
    if SMOKE:entries=[next(e for e in validation if e['family']==f) for f in ('long_wall','wall_corner','multi_route')]
    for e in validation:protected[e['path']]=sha(e['path'])
    save(OUT/'bank/manifest.json',dict(entries=entries,selection='Unchanged E45-C public history bank; development ablation, no outcome-based selection'))
    references=[];reference_roots={};artifacts={}
    if not SMOKE:
        results=read(SOURCE/'evaluation/results_with_engineering_failures.json')
        results=[r for r in results if r['_output']==str(SOURCE/'learning/x20') and r['method'].startswith('PRIOR')]
        if len(results)!=432:raise RuntimeError('Expected 432 original x20 PRIOR games')
        by_group={e['id']:e for e in entries}
        for row in results:
            if not row.get('valid_episode',True):raise RuntimeError('Invalid reference PRIOR cell')
            seed=int(row['method'][5:]);e=by_group[row['group']]
            expected=old.request(e,row['world'],row['opponent'],row['method'],'evaluation',SOURCE/'learning/x20',model_round=12)
            for field in ('key','case','seed','world','opponent','checkpoint','budget_ms'):
                if row[field]!=expected[field]:raise RuntimeError('Reference task mismatch: '+field)
            resolved=Path(row['_output'])/f'round_{row["model_round"]:02d}/models/joint_{seed}.npz'
            if str(resolved)!=models['JOINT_PRIOR'][str(seed)]:raise RuntimeError('Implicit checkpoint path mismatch')
            payload=read(row['episode']);digest=sha(row['episode'])
            if digest!=row['episode_sha256']:raise RuntimeError('Reference episode hash mismatch')
            cache=SOURCE/'learning/x20/evaluation/cache'/(row['key']+'.json')
            if read(cache)!=row:raise RuntimeError('Original reference cache mismatch')
            artifacts[str(cache)]=sha(cache);artifacts[row['episode']]=digest
            key=f'{row["group"]}_{row["world"]}_{row["opponent"]}_JOINT_PRIOR{seed}_r0'
            reference_roots[key]=dict(simulations=payload['trace'][0]['stats'].get('simulations',0),family=row['family'])
            references.append(row|dict(method=f'JOINT_PRIOR{seed}',key=key,origin='E45-C x20 cached PRIOR',original_key=row['key']))
    save(OUT/'reference/joint_prior.json',references)
    save(OUT/'reference/root_metrics.json',reference_roots)
    save(OUT/'reference/integrity.json',dict(files=artifacts,verified=len(references)))
    for name in ('joint_prior.json','root_metrics.json','integrity.json'):
        protected[str(OUT/'reference'/name)]=sha(OUT/'reference'/name)
    config=dict(experiment='E45-E',smoke=SMOKE,base_seed=BASE_SEED,seeds=list(SEEDS),arms=list(ARMS),new_arms=list(NEW_ARMS),
        checkpoints=models,test_groups=len(entries),worlds=['AB','BA'],opponents=list(base.OPPONENTS),
        formal_tasks=len(entries)*2*2*3*2,reference_tasks=len(references),online_budget_ms=200,training=False,performance_gate=False,
        primary='JOINT_PRIOR minus SINGLE_PRIOR / FACT_PRIOR for both opponents; four simultaneous 95% intervals, public-history grouped',
        cache_rule='72 paired first-root checks: overall throughput ratio in [.85,1/.85], each family [.65,1/.65]; otherwise rerun all 432 JOINT_PRIOR games in mixed batch',
        source_hashes={str(ROOT/'code'/p):sha(ROOT/'code'/p) for p in SOURCES},protected_hashes=protected,
        bank_hashes={e['path']:sha(e['path']) for e in entries}|{str(OUT/'bank/manifest.json'):sha(OUT/'bank/manifest.json')},validation_entries=validation)
    save(OUT/'experiment.json',config)
    (OUT/'PROTOCOL.md').write_bytes((ROOT/'code/E45E_VALUE_OFF_PROTOCOL_20261029.md').read_bytes())
    for p in SOURCES:
        target=OUT/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/'code'/p).read_bytes())
    return config


def request(entry,world,opponent,arm,seed,stage,rep=0):
    config=read(OUT/'experiment.json')
    original_method={'SINGLE_PRIOR':'SINGLE','FACT_PRIOR':'FACT','JOINT_PRIOR':'PRIOR'}[arm]+str(seed)
    task=old.request(entry,world,opponent,original_method,stage,OUT,model_round=12,
                     checkpoint=config['checkpoints'][arm][str(seed)])
    task.update(key=f'{entry["id"]}_{world}_{opponent}_{arm}{seed}_r{rep}',method=f'{arm}{seed}')
    if rep:task['seed']=int(np.random.SeedSequence([task['seed'],rep]).generate_state(1)[0])
    return task


def invalid(task,failure):
    message=failure['error']
    if failure['task']!=task or not any(msg in message for msg in SUPPORT_ERRORS):
        raise RuntimeError('Unresolved execution failure: '+message)
    match=re.search(r'public_history_evidence=([^\s\x27\x22;)]+)',message)
    if not match:raise RuntimeError('Support failure without public history evidence')
    evidence=Path(match.group(1)).with_suffix('.json')
    if not evidence.resolve().is_relative_to((OUT/'belief_failures').resolve()):raise RuntimeError('Bad evidence path')
    proof=read(evidence)
    row=dict(**task,valid_episode=False,reason='belief_support_exhaustion',double_capture=None,utility=None,
        completion_cost=None,decisions=None,first_capture=None,latencies_ms=[],cpu_ms=[],hidden_simulations=[],
        public_evidence=str(evidence),public_evidence_sha256=sha(evidence),arrays=proof['arrays'],
        arrays_sha256=sha(proof['arrays']),failure=message)
    save(OUT/task['stage']/'invalid'/(task['key']+'.json'),row)
    return row


def job(task):
    configure()
    _,cache,_=base.paths(task);failure=cache.with_name(cache.stem+'.failure.json')
    if failure.exists():return invalid(task,read(failure))
    try:return base.episode_job(task)
    except RuntimeError:
        if failure.exists() and any(msg in read(failure)['error'] for msg in SUPPORT_ERRORS):return invalid(task,read(failure))
        raise


def worker_init(output,cpus,parent):
    global OUT
    OUT=Path(output);torch.set_num_threads(1)
    ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    if os.getppid()!=parent:raise RuntimeError('Coordinator exited')
    os.sched_setaffinity(0,{cpus.get()});base.PARENT=parent
    verify();configure()
    validation=read(OUT/'experiment.json')['validation_entries']
    for f in ('long_wall','wall_corner','multi_route'):
        e=next(e for e in validation if e['family']==f);case=read(e['path'])
        for arm in ARMS:
            task=request(e,'AB','fixed_D_kernel',arm,17,'warmup')
            own=factory(case,task)
            for obs in case['history'][:-1]:own.observe(obs)
            own.act(case['history'][-1],fixed_simulations=2)
    audit_simulated(np.zeros((0,26)),np.zeros((0,2,2)),1.1)
    base.worker_status(stage='ready')


def run_tasks(tasks,workers,stage):
    pending_tasks=collections.deque();results=[]
    configure()
    for task in tasks:
        row=base.cached(task)
        if row is not None:results.append(row)
        else:pending_tasks.append(task)
    if not pending_tasks:return sorted(results,key=lambda x:x['key'])
    count=min(workers,len(pending_tasks));ctx=mp.get_context('spawn');queue=ctx.Queue()
    for cpu in CPUS[:count]:queue.put(cpu)
    pool=futures.ProcessPoolExecutor(max_workers=count,mp_context=ctx,initializer=worker_init,initargs=(str(OUT),queue,os.getpid()))
    running={};failed=True;last_report=0.
    try:
        while pending_tasks or running:
            if time.time()-read(OUT/'clock.json')['started_unix']>12*3600:raise RuntimeError('Registered 12h cap reached')
            while pending_tasks and len(running)<2*count:
                task=pending_tasks.popleft();running[pool.submit(job,task)]=task
            done,_=futures.wait(running,timeout=5,return_when=futures.FIRST_COMPLETED)
            for future in done:
                task=running.pop(future)
                try:results.append(future.result())
                except BaseException:save(OUT/'failed_task.json',task);raise
            if done or time.monotonic()-last_report>=15:
                emit(stage,completed=len(results),total=len(tasks),workers=count,
                    invalid=sum(not x.get('valid_episode',True) for x in results))
                last_report=time.monotonic()
        failed=False
    finally:
        processes=list((pool._processes or {}).values());pool.shutdown(wait=not failed,cancel_futures=failed)
        if failed:
            for p in processes:
                if p.is_alive():p.terminate()
            for p in processes:p.join(timeout=5)
        queue.close()
    return sorted(results,key=lambda x:x['key'])


def timing(rows):
    if any(not r.get('valid_episode',True) for r in rows):raise RuntimeError('Calibration support failure')
    wall=sum(sum(r['latencies_ms']) for r in rows);cpu=sum(sum(r['cpu_ms']) for r in rows)
    lat=[v for r in rows for v in r['latencies_ms']]
    return dict(cpu_wall=cpu/max(wall,1e-9),overrun_rate=np.mean(np.array(lat)>200),
        simulations=sum(sum(r['hidden_simulations']) for r in rows),
        by_arm={a:sum(sum(r['hidden_simulations']) for r in rows if family(r)==a) for a in ARMS})


def calibrate(maximum):
    path=OUT/'timing/decision.json'
    if path.exists():return read(path)['workers']
    if SMOKE:
        save(path,dict(workers=min(4,maximum),smoke=True));return min(4,maximum)
    validation=read(OUT/'experiment.json')['validation_entries']
    entries=[next(e for e in validation if e['family']==f) for f in ('long_wall','wall_corner','multi_route')]
    def tasks(stage):return [request(e,'AB',o,a,s,stage,rep)|dict(limit=2)
        for rep in range(2) for e in entries for o in base.OPPONENTS for s in SEEDS for a in ARMS]
    reference_tasks=tasks('calibration/serial');serial=[];affinity=os.sched_getaffinity(0)
    os.sched_setaffinity(0,{CPUS[0]})
    # Warm shared JIT paths on all three public geometries before reference.
    configure()
    for e in entries:
        case=read(e['path']);own=factory(case,request(e,'AB','fixed_D_kernel','JOINT_PRIOR',17,'warmup'))
        for obs in case['history'][:-1]:own.observe(obs)
        own.act(case['history'][-1],fixed_simulations=2)
    try:
        for task in reference_tasks:
            serial.append(job(task));emit('calibration_serial',completed=len(serial),total=len(reference_tasks),workers=1)
    finally:os.sched_setaffinity(0,affinity)
    ref=timing(serial);candidates=[]
    for n in sorted({maximum}|{v for v in (40,32,24,16,8,4) if v<maximum},reverse=True):
        rows=run_tasks(tasks(f'calibration/parallel_{n}'),n,'calibration_parallel')
        metrics=timing(rows);ratio=metrics['simulations']/max(1,ref['simulations'])
        arm_ratio=min(metrics['by_arm'][a]/max(1,ref['by_arm'][a]) for a in ARMS)
        passed=bool(metrics['cpu_wall']>=.85 and ratio>=.85 and arm_ratio>=.65 and metrics['overrun_rate']<=ref['overrun_rate']+.002)
        candidates.append(dict(workers=n,metrics=metrics,simulation_ratio=ratio,minimum_arm_ratio=arm_ratio,passed=passed))
        save(OUT/'timing/calibration.json',dict(reference=ref,candidates=candidates))
        if passed:
            save(path,dict(workers=n,reference=ref,candidates=candidates,outcomes_used=False));return n
    save(path,dict(workers=1,reference=ref,candidates=candidates,outcomes_used=False));return 1


def summarize(rows):
    cells=collections.defaultdict(list)
    for row in rows:cells[row['opponent'],family(row)].append(row)
    summaries=[]
    for (op,a),rr in sorted(cells.items()):
        good=[r for r in rr if r.get('valid_episode',True)];wins=sum(r['double_capture'] for r in good)
        lat=[t for r in good for t in r['latencies_ms']]
        summaries.append(dict(opponent=op,arm=a,planned=len(rr),valid=len(good),invalid=len(rr)-len(good),captures=wins,
            capture_rate_bounds=[wins/len(rr),(wins+len(rr)-len(good))/len(rr)],
            utility_valid=float(np.mean([r['utility'] for r in good])) if good else None,
            completion_cost_valid=float(np.mean([r['completion_cost'] for r in good])) if good else None,
            overrun_rate_valid=float(np.mean(np.array(lat)>200)) if lat else None,
            origin=sorted({r.get('origin','E45-E new evaluation') for r in rr})))
    groups=sorted({r['group'] for r in rows});comparisons=[(o,a) for o in base.OPPONENTS for a in NEW_ARMS]
    expected={(g,w,o,a,s) for g in groups for w in ('AB','BA') for o in base.OPPONENTS for a in ARMS for s in SEEDS}
    observed=[(r['group'],r['world'],r['opponent'],family(r),int(r['method'][len(family(r)):])) for r in rows]
    complete=len(observed)==len(expected) and set(observed)==expected and all(r.get('valid_episode',True) for r in rows)
    primary=None
    if complete and not SMOKE:
        means={(g,o,a):np.mean([r['utility'] for r in cells[o,a] if r['group']==g]) for g in groups for o in base.OPPONENTS for a in ARMS}
        delta=np.array([[means[g,o,'JOINT_PRIOR']-means[g,o,a] for o,a in comparisons] for g in groups])
        mean=delta.mean(0);rng=np.random.default_rng(BASE_SEED+991)
        bs=delta[rng.integers(len(groups),size=(10000,len(groups)))].mean(1)
        radius=float(np.quantile(np.max(np.abs(bs-mean),axis=1),.95))
        primary=[dict(opponent=o,contrast='JOINT_PRIOR - '+a,utility_difference=float(mean[i]),
             simultaneous95=[float(mean[i]-radius),float(mean[i]+radius)]) for i,(o,a) in enumerate(comparisons)]
    return dict(summaries=summaries,primary=primary,groups=len(groups),smoke=SMOKE,complete_cells=complete,
         primary_status='complete' if primary is not None else 'engineering_only' if SMOKE else 'withheld_incomplete_cells')


def report(rows=None,stats=None):
    status=read(OUT/'status.json')
    lines=['# E45-E：20×三类连续提议的Value-Off闭环消融','',
        '冻结Single、FACT、Joint各3个模型；统一关闭部署学习叶价值，保留联合MCTS、几何势和团队局部观测。无新增训练。','',
        '复用E45-C的36组公开历史、两隐藏世界和两对手。新增Single PRIOR、FACT PRIOR共864局；Joint PRIOR的432局按预定吞吐检查决定复用或同批重跑。这是已有测试历史上的补充机制比较。','',
        f'当前状态：`{status["stage"]}`；[实时进度](status.json) · [运行日志](run.log)','',
        '[协议](PROTOCOL.md) · [冻结配置](experiment.json) · [并发校准](timing/decision.json) · [旧PRIOR复用判定](timing/reference_decision.json)']
    if stats:
        lines+=['','|对手|提议|有效/计划|双捕|完成成本|团队效用|','|---|---|---:|---:|---:|---:|']
        for r in stats['summaries']:
            cost='null' if r['completion_cost_valid'] is None else f'{r["completion_cost_valid"]:.3f}'
            util='null' if r['utility_valid'] is None else f'{r["utility_valid"]:.5f}'
            lines.append(f'|{r["opponent"]}|{r["arm"]}|{r["valid"]}/{r["planned"]}|{r["captures"]}|{cost}|{util}|')
        lines+=['',f'主比较状态：`{stats["primary_status"]}`。36组公共历史配对bootstrap10000次，四项同时95%区间。']
        for r in stats['primary'] or []:
            lines.append(f'- {r["opponent"]}，{r["contrast"]}: {r["utility_difference"]:+.5f}，{r["simultaneous95"]}')
        lines+=['','[新跑逐局结果](new_results.json) · [含参考的完整比较结果](results.json) · [统计](statistics.json)']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def reference_check(workers):
    path=OUT/'timing/reference_decision.json'
    if path.exists():return read(path)
    if SMOKE:
        decision=dict(reuse=False,smoke=True);save(path,decision);return decision
    entries=read(OUT/'bank/manifest.json')['entries'];selected=[]
    for f in ('long_wall','wall_corner','multi_route'):
        selected.extend(sorted([e for e in entries if e['family']==f],key=lambda e:e['id'])[:2])
    tasks=[request(e,w,o,'JOINT_PRIOR',s,'reference_roots')|dict(limit=1)
           for e in selected for w in ('AB','BA') for o in base.OPPONENTS for s in SEEDS]
    rows=run_tasks(tasks,workers,'reference_roots');original=read(OUT/'reference/root_metrics.json')
    totals=collections.defaultdict(lambda:[0,0]);pairs=[]
    for row in rows:
        if not row.get('valid_episode',True):raise RuntimeError('Reference root support failure')
        current=read(row['episode'])['trace'][0]['stats'].get('simulations',0)
        old_count=original[row['key']]['simulations']
        totals['all'][0]+=current;totals['all'][1]+=old_count
        totals[row['family']][0]+=current;totals[row['family']][1]+=old_count
        pairs.append(dict(key=row['key'],current=current,reference=old_count))
    ratios={k:a/max(1,b) for k,(a,b) in totals.items()}
    reuse=.85<=ratios['all']<=1/.85 and all(.65<=ratios[f]<=1/.65 for f in ratios if f!='all')
    decision=dict(reuse=bool(reuse),roots=len(rows),ratios=ratios,pairs=pairs,
                  outcomes_used=False,scope='First-root throughput check; cache and new runs remain different wall-clock batches')
    save(path,decision);return decision


def main():
    global OUT,SMOKE,CPUS
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=40);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();SMOKE=args.smoke
    if args.output:OUT=args.output.resolve()
    elif SMOKE:OUT=OUT.with_name(OUT.name+'_smoke')
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'coordinator.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=prepare();verify();configure()
        if not (OUT/'clock.json').exists():save(OUT/'clock.json',dict(started_unix=time.time(),cap_seconds=12*3600))
        execution=OUT/'execution.json'
        CPUS=read(execution)['cpus'] if execution.exists() else choose_cpus(args.workers)
        if not CPUS:raise RuntimeError('No available physical cores')
        maximum=min(args.workers,len(CPUS))
        save(execution,dict(pid=os.getpid(),cpus=CPUS,max_workers=maximum,threads_per_worker=1,training=False,tmux='train:E45E'))
        try:
            emit('registered',total=config['formal_tasks'],max_workers=maximum);report()
            workers=calibrate(maximum);decision=reference_check(workers)
            entries=read(OUT/'bank/manifest.json')['entries']
            arms=NEW_ARMS if decision['reuse'] else ARMS
            tasks=[request(e,w,o,a,s,'evaluation') for e in entries for w in ('AB','BA') for o in base.OPPONENTS for s in SEEDS for a in arms]
            if SMOKE:tasks=[t|dict(limit=2,fixed=4) for t in tasks]
            save(OUT/'requests.json',tasks)
            np.random.default_rng(BASE_SEED+77).shuffle(tasks)
            emit('evaluation_start',total=len(tasks),workers=workers,reuse_joint_prior=decision['reuse']);report()
            new_rows=run_tasks(tasks,workers,'evaluation');save(OUT/'new_results.json',new_rows)
            rows=sorted(new_rows+(read(OUT/'reference/joint_prior.json') if decision['reuse'] else []),key=lambda r:r['key'])
            save(OUT/'results.json',rows);stats=summarize(rows);save(OUT/'statistics.json',stats);verify()
            integrity=read(OUT/'reference/integrity.json')
            for p,h in integrity['files'].items():
                if sha(p)!=h:raise RuntimeError('Original reference artifact changed: '+p)
            emit('complete',completed=len(new_rows),total=len(tasks),workers=workers,comparison_cells=len(rows),
                invalid=sum(not r.get('valid_episode',True) for r in rows),primary_status=stats['primary_status'])
            report(rows,stats)
            save(OUT/'decision.json',dict(status='complete',attempted_new=len(new_rows),reused=len(rows)-len(new_rows),
                primary_status=stats['primary_status'],training=False))
        except BaseException as exc:
            save(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
            emit('failed',error=repr(exc));report();raise


if __name__=='__main__':main()
