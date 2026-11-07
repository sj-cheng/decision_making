"""E45-D: one frozen-checkpoint shared-component ablation, resumable."""
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
from e45b_scenarios import make_case
from e45c_belief_diagnostics import install, SUPPORT_ERRORS
from e45c_capture_audit import audit_simulated
from learning.e45b_joint_network import Runtime
from solvers.e45b_joint_nte import Controller
from run_e44_parallel import choose_cpus

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'current/e45c_budget_scaling_20261028'
OUT=ROOT/'current/e45d_prior_correlation_20261029'
SEEDS=(17,29,43)
ARMS=('PRIOR','PRODUCT')
BASE_SEED=945520261029
CPUS=[]
SMOKE=False
SOURCES=('run_e45d_prior_correlation.py','E45D_PRIOR_PROTOCOL_20261029.md','tests/test_e45d_prior_correlation.py')


class ProductRuntime:
    """Independent component indices, exactly the same raw agent marginals."""
    def __init__(self,original):
        if original.kind!='joint':raise ValueError('Requires frozen Joint checkpoint')
        self.original=original
    def __call__(self,features):return self.original(features)
    def sample(self,parameters,rng):
        p,mu,logs=parameters
        components=rng.choice(len(p),size=2,p=p)
        noise=rng.standard_normal((2,2))
        return np.array([np.tanh(mu[k,2*i:2*i+2]+np.exp(logs[k,2*i:2*i+2])*noise[i])
            for i,k in enumerate(components)])


class ProductController(Controller):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.runtime=ProductRuntime(self.runtime)


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):return old.save(Path(path),data)
def family(row):return row['method'].rstrip('0123456789')


def factory(case,task):
    arm=family(task)
    if arm not in ARMS:raise ValueError(arm)
    cls=Controller if arm=='PRIOR' else ProductController
    return cls(case['map'],case['speed'],task['seed'],task['checkpoint'],
        use_policy=True,use_value=False,budget_ms=task['budget_ms'])


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
            if sha(path)!=digest:raise RuntimeError('Frozen E45-D artifact changed: '+path)
    return config


def prepare():
    if (OUT/'experiment.json').exists():return verify()
    OUT.mkdir(parents=True,exist_ok=True)
    old.verify()
    protected={str(SOURCE/'experiment.json'):sha(SOURCE/'experiment.json'),
        str(SOURCE/'models_frozen.json'):sha(SOURCE/'models_frozen.json')}
    previous=read(SOURCE/'experiment.json')
    protected.update(previous['protected_hashes'])
    for name in ('REPAIR_CAPTURE_AUDIT_20261028.json','REPAIR_SUPPORT_DIAGNOSTICS_20261028.json'):
        repair=read(SOURCE/name)
        protected.update(repair['extra_hashes'])
        protected.update({p:v['repaired_sha256'] for p,v in repair['overrides'].items()})
        protected[str(SOURCE/name)]=sha(SOURCE/name)
    for p in previous['source_hashes']:protected.setdefault(p,sha(p))
    models={s:str(SOURCE/f'learning/x20/round_12/models/joint_{s}.npz') for s in SEEDS}
    frozen=read(SOURCE/'models_frozen.json')['files']
    for path in models.values():
        assert sha(path)==frozen[path]
        protected[path]=frozen[path]
    def geometry(case):return hashlib.sha256(json.dumps(case['map'],sort_keys=True).encode()).hexdigest()
    known=set()
    for folder in (SOURCE/'bank',old.OLD/'bank'):
        for p in folder.glob('*/*.json'):known.add(geometry(read(p)))
    entries=[];rejected=[];seed=BASE_SEED+(1000000 if SMOKE else 0)
    for f in ('long_wall','wall_corner','multi_route'):
        accepted=collections.Counter();index=0;target=1 if SMOKE else 4
        while sum(accepted.values())<(1 if SMOKE else 12):
            if index>1000:raise RuntimeError('History generation exhausted')
            try:
                case=make_case(f,index,'test',seed)
                if not SMOKE and accepted[case['pattern']]>=target:index+=1;continue
                digest=geometry(case)
                if digest in known:raise ValueError('Duplicate geometry')
            except (AssertionError,ValueError) as exc:
                rejected.append(dict(family=f,index=index,error=repr(exc)));index+=1;continue
            known.add(digest);accepted[case['pattern']]+=1
            path=OUT/'bank/test'/(case['id']+'.json');save(path,case)
            entries.append(dict(id=case['id'],path=str(path),family=f,pattern=case['pattern'],split='test',source_index=index))
            index+=1
    validation=[e for e in read(SOURCE/'bank/manifest.json')['entries'] if e['split']=='validation']
    for e in validation:protected[e['path']]=sha(e['path'])
    save(OUT/'bank/manifest.json',dict(entries=entries,rejections=rejected,selection='public geometry and history legality only'))
    config=dict(experiment='E45-D',smoke=SMOKE,base_seed=seed,seeds=list(SEEDS),arms=list(ARMS),
        checkpoints=models,test_groups=len(entries),worlds=['AB','BA'],opponents=list(base.OPPONENTS),
        formal_tasks=len(entries)*2*2*3*2,online_budget_ms=200,training=False,performance_gate=False,
        primary='PRIOR minus PRODUCT raw team utility, grouped by public history; two-opponent simultaneous 95% intervals',
        source_hashes={str(ROOT/'code'/p):sha(ROOT/'code'/p) for p in SOURCES},protected_hashes=protected,
        bank_hashes={str(p):sha(p) for p in (OUT/'bank').rglob('*.json')},validation_entries=validation)
    save(OUT/'experiment.json',config)
    (OUT/'PROTOCOL.md').write_bytes((ROOT/'code/E45D_PRIOR_PROTOCOL_20261029.md').read_bytes())
    for p in SOURCES:
        target=OUT/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/'code'/p).read_bytes())
    return config


def request(entry,world,opponent,arm,seed,stage,rep=0):
    config=read(OUT/'experiment.json')
    random_seed=int(np.random.SeedSequence([config['base_seed'],int(sha(entry['path'])[:8],16),int(world=='BA'),rep]).generate_state(1)[0])
    return dict(key=f'{entry["id"]}_{world}_{opponent}_{arm}{seed}_r{rep}',case=entry['path'],group=entry['id'],
        family=entry['family'],pattern=entry['pattern'],split='validation' if stage.startswith('calibration') else 'test',
        world=world,opponent=opponent,method=f'{arm}{seed}',seed=random_seed,stage=stage,round=0,model_round=12,
        budget_ms=200.,_output=str(OUT),checkpoint=config['checkpoints'][str(seed)])


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
    worker_init_serial=True
    # Warm shared JIT paths on all three public geometries before reference.
    configure()
    for e in entries:
        case=read(e['path']);own=factory(case,request(e,'AB','fixed_D_kernel','PRIOR',17,'warmup'))
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
        summaries.append(dict(opponent=op,arm=a,planned=len(rr),valid=len(good),invalid=len(rr)-len(good),captures=wins,
            capture_rate_bounds=[wins/len(rr),(wins+len(rr)-len(good))/len(rr)],
            utility_valid=np.mean([r['utility'] for r in good]) if good else None,
            completion_cost_valid=np.mean([r['completion_cost'] for r in good]) if good else None,
            overrun_rate_valid=np.mean([t>200 for r in good for t in r['latencies_ms']]) if good else None))
    comparison=None
    if all(r.get('valid_episode',True) for r in rows) and not SMOKE:
        groups=sorted({r['group'] for r in rows});ops=base.OPPONENTS
        delta=np.array([[np.mean([r['utility'] for r in rows if r['group']==g and r['opponent']==o and family(r)=='PRIOR'])
            -np.mean([r['utility'] for r in rows if r['group']==g and r['opponent']==o and family(r)=='PRODUCT']) for o in ops] for g in groups])
        mean=delta.mean(0);rng=np.random.default_rng(BASE_SEED+991)
        bs=delta[rng.integers(len(groups),size=(10000,len(groups)))].mean(1)
        radius=float(np.quantile(np.max(np.abs(bs-mean),axis=1),.95))
        comparison=[dict(opponent=o,utility_difference=float(mean[i]),simultaneous95=[float(mean[i]-radius),float(mean[i]+radius)]) for i,o in enumerate(ops)]
    return dict(summaries=summaries,primary=comparison,groups=len({r['group'] for r in rows}),
        smoke=SMOKE,primary_status='complete' if comparison is not None else 'engineering_only' if SMOKE else 'withheld_incomplete_cells')


def report(rows=None,stats=None):
    status=read(OUT/'status.json')
    lines=['# E45-D：PRIOR联合采样机制对照','',
        '只比较原PRIOR与同Joint检查点的两车独立分量采样；3种子、36组新公共历史、两世界、两对手，共864局。200ms，不重新训练。','',
        f'当前状态：`{status["stage"]}`；进度见[status.json](status.json)，运行日志见[run.log](run.log)。','',
        '[协议](PROTOCOL.md) · [冻结配置](experiment.json) · [时延与并发](timing/decision.json)']
    if stats:
        lines+=['','|对手|方法|有效/计划|双捕|完成成本|团队效用|','|---|---|---:|---:|---:|---:|']
        for r in stats['summaries']:
            cost='null' if r['completion_cost_valid'] is None else f'{r["completion_cost_valid"]:.3f}'
            util='null' if r['utility_valid'] is None else f'{r["utility_valid"]:.5f}'
            lines.append(f'|{r["opponent"]}|{r["arm"]}|{r["valid"]}/{r["planned"]}|{r["captures"]}|{cost}|{util}|')
        if stats['primary']:
            lines+=['','PRIOR − PRODUCT，36公共历史组配对bootstrap、两项同时95%区间：','']
            for r in stats['primary']:lines.append(f'- {r["opponent"]}: {r["utility_difference"]:+.5f}，{r["simultaneous95"]}')
        lines+=['','[逐局结果](results.json) · [统计](statistics.json)']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    global OUT,SMOKE,CPUS
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=40);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();SMOKE=args.smoke
    if SMOKE:OUT=OUT.with_name(OUT.name+'_smoke')
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'coordinator.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=prepare();verify();configure()
        if not (OUT/'clock.json').exists():save(OUT/'clock.json',dict(started_unix=time.time(),cap_seconds=12*3600))
        execution=OUT/'execution.json'
        if execution.exists():CPUS=read(execution)['cpus']
        else:
            CPUS=choose_cpus(args.workers)
            if not CPUS:raise RuntimeError('No available physical cores')
        maximum=min(args.workers,len(CPUS))
        save(execution,dict(pid=os.getpid(),cpus=CPUS,max_workers=maximum,threads_per_worker=1,training=False,tmux='train:E45D'))
        try:
            emit('registered',total=config['formal_tasks'],max_workers=maximum);report()
            workers=calibrate(maximum)
            entries=read(OUT/'bank/manifest.json')['entries']
            tasks=[request(e,w,o,a,s,'evaluation') for e in entries for w in ('AB','BA') for o in base.OPPONENTS for s in SEEDS for a in ARMS]
            if SMOKE:tasks=[t|dict(limit=2,fixed=4) for t in tasks]
            save(OUT/'requests.json',tasks)
            # Outcome-independent shuffle keeps both arms mixed under the same load.
            np.random.default_rng(BASE_SEED+77).shuffle(tasks)
            rows=run_tasks(tasks,workers,'evaluation');save(OUT/'results.json',rows)
            stats=summarize(rows);save(OUT/'statistics.json',stats)
            verify()
            emit('complete',completed=len(rows),total=len(tasks),workers=workers,invalid=sum(not r.get('valid_episode',True) for r in rows),primary_status=stats['primary_status'])
            report(rows,stats)
            save(OUT/'decision.json',dict(status='complete',attempted=len(rows),primary_status=stats['primary_status'],training=False))
        except BaseException as exc:
            save(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
            emit('failed',error=repr(exc));report();raise


if __name__=='__main__':main()
