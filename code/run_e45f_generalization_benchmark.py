"""E45-F: independent benchmark and held-out obstacle-layout generalization."""
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
from solvers.e45_team_adapter import TeamController
from solvers.e45f_baselines import GeometryController,VoronoiController
from e45f_scenarios import ID_FAMILIES,OOD_FAMILIES,new_id_case,shifted_case,geometry_key
from run_e44_parallel import choose_cpus

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'current/e45c_budget_scaling_20261028'
OUT=ROOT/'current/e45f_generalization_benchmark_20261029'
SEEDS=(17,29,43)
ARMS=('SINGLE_PRIOR','FACT_PRIOR','JOINT_PRIOR','SINGLE_VALUE','SINGLE_DIRECT','U','AG','IG','GEOM_SEARCH','VOO_UCT')
LEARNED=ARMS[:5]
MODEL_KINDS=dict(zip(LEARNED,('single','factorized','joint','single','single')))
COMPARATORS=ARMS[1:]
SEARCH_ARMS=tuple(a for a in ARMS if a not in ('AG','IG','SINGLE_DIRECT'))
BASE_SEED=945720261029
CPUS=[]
SMOKE=False
SOURCES=('run_e45f_generalization_benchmark.py','e45f_scenarios.py','solvers/e45f_baselines.py','E45F_GENERALIZATION_PROTOCOL_20261029.md','tests/test_e45f_generalization.py')








def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):return old.save(Path(path),data)
def family(row):return row['method'].rstrip('0123456789')


def factory(case,task):
    arm=family(task);budget=task['budget_ms'];seed=task['seed']
    if arm in ('AG','IG'):own=TeamController(case['map'],case['speed'],seed,arm,budget_ms=budget)
    elif arm=='U':own=Controller(case['map'],case['speed'],seed,budget_ms=budget)
    elif arm in ('GEOM_SEARCH','VOO_UCT'):
        cls=GeometryController if arm=='GEOM_SEARCH' else VoronoiController
        own=cls(case['map'],case['speed'],seed,use_policy=False,use_value=False,budget_ms=budget)
    elif arm in LEARNED:
        own=Controller(case['map'],case['speed'],seed,task['checkpoint'],use_policy=True,
                       use_value=arm=='SINGLE_VALUE',direct=arm=='SINGLE_DIRECT',budget_ms=budget)
    else:raise ValueError(arm)
    # Preserve original behavior. Only declared finite representation errors
    # gain PUBLIC evidence and explicit invalid outcomes rather than killing unrelated jobs.
    known=('Candidate capacity exceeded','More important regions than the frozen sample budget')
    def guarded(original):
        def call(*args,**kwargs):
            try:return original(*args,**kwargs)
            except (RuntimeError,ValueError) as exc:
                if str(exc) not in known:raise
                stem=OUT/'belief_failures'/f'representation_{os.getpid()}_{time.time_ns()}'
                stem.parent.mkdir(parents=True,exist_ok=True)
                arrays=stem.with_suffix('.npz')
                np.savez_compressed(arrays,**{f'weights_{i}':est.weights for i,est in enumerate(own.belief.estimators) if est.weights is not None})
                save(stem.with_suffix('.json'),dict(error=str(exc),seed=seed,boxes=case['map'],
                    history=own.belief.public_history,arrays=str(arrays),representation_failure=True))
                raise RuntimeError(str(exc)+'; public_history_evidence='+str(stem)) from exc
        return call
    own.observe=guarded(own.observe);own.act=guarded(own.act)
    return own


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
            if sha(path)!=digest:raise RuntimeError('Frozen E45-F artifact changed: '+path)
    return config


def prepare():
    if (OUT/'experiment.json').exists():return verify()
    OUT.mkdir(parents=True,exist_ok=True);old.verify()
    previous=read(ROOT/'current/e45e_value_off_20261029/experiment.json')
    protected=previous['protected_hashes']|previous['source_hashes']
    protected[str(SOURCE/'bank/manifest.json')]=sha(SOURCE/'bank/manifest.json')
    frozen=read(SOURCE/'models_frozen.json')['files'];models={}
    for arm,kind in MODEL_KINDS.items():
        models[arm]={}
        for seed in SEEDS:
            path=str(SOURCE/f'learning/x20/round_12/models/{kind}_{seed}.npz')
            if sha(path)!=frozen[path]:raise RuntimeError('Frozen model changed: '+path)
            models[arm][str(seed)]=path;protected[path]=frozen[path]
    known=set()
    for directory in ('e45b_role_free_joint_nte_20261027','e45c_budget_scaling_20261028','e45d_prior_correlation_20261029'):
        for path in (ROOT/'current'/directory/'bank').glob('*/*.json'):
            known.add(geometry_key(read(path)['map']))
    entries=[];validation=[];rejections=[];namespace=BASE_SEED+(1000000 if SMOKE else 0)
    for suite,families in (('benchmark',ID_FAMILIES),('generalization',OOD_FAMILIES)):
        for f in families:
            accepted=collections.Counter();index=0;target=1 if SMOKE else 12
            while sum(accepted.values())<target:
                if index>1200:raise RuntimeError('Public scenario construction exhausted: '+f)
                try:
                    builder=new_id_case if suite=='benchmark' else shifted_case
                    case=builder(f,index,'test',namespace)
                    cell=(case['pattern'],case['speed'],case['density']) if suite=='generalization' else (case['pattern'],case['speed'])
                    quota=1 if suite=='generalization' else 2
                    if not SMOKE and accepted[cell]>=quota:index+=1;continue
                    digest=geometry_key(case['map'])
                    if digest in known:raise ValueError('Previously used geometry up to translation/quarter-turn and1e-4')
                except (ValueError,AssertionError) as exc:
                    rejections.append(dict(suite=suite,family=f,index=index,error=repr(exc)));index+=1;continue
                known.add(digest);accepted[cell]+=1
                path=OUT/'bank'/suite/(case['id']+'.json');save(path,case)
                entries.append(dict(id=case['id'],path=str(path),family=f,pattern=case['pattern'],split='test',
                    suite=suite,density=case['density'],source_index=index,geometry_summary=case['geometry_summary']))
                index+=1
            emit('prepare_bank',suite=suite,family=f,accepted=target)
    # Independent engineering/timing cases, never selected by controller outcomes.
    for f in ID_FAMILIES+OOD_FAMILIES:
        for index in range(100):
            try:
                builder=new_id_case if f in ID_FAMILIES else shifted_case
                # Dense OOD maps exercise the larger geometry cost during calibration.
                case=builder(f,2*index+(f in OOD_FAMILIES),'validation',namespace+777)
                digest=geometry_key(case['map'])
                if digest in known:continue
            except (ValueError,AssertionError):continue
            known.add(digest);path=OUT/'bank/validation'/(case['id']+'.json');save(path,case)
            validation.append(dict(id=case['id'],path=str(path),family=f,pattern=case['pattern'],split='validation',suite='calibration',density=case['density']))
            break
        else:raise RuntimeError('Could not construct calibration history '+f)
    manifest=dict(entries=entries,validation=validation,rejections=rejections,
        selection='Public geometry/legal causal history/indistinguishability/connectivity only; no estimator or controller success used')
    save(OUT/'bank/manifest.json',manifest)
    methods=[(a,s) for a in ARMS for s in (SEEDS if a in LEARNED else (0,))]
    config=dict(experiment='E45-F',smoke=SMOKE,base_seed=namespace,seeds=list(SEEDS),arms=list(ARMS),
        execution_arms=[dict(arm=a,seed=s) for a,s in methods],checkpoints=models,
        history_groups=dict(collections.Counter(e['suite'] for e in entries)),test_groups=len(entries),
        worlds=['AB','BA'],opponents=list(base.OPPONENTS),formal_tasks=len(entries)*2*2*len(methods),
        online_budget_ms=200,training=False,performance_gate=False,reuse_old_results=False,
        baseline_scope='Matched original UCT with geometric or VOO-style proposal; not complete VOMCPOW or original NTE replication',
        primary='Per suite, SINGLE_PRIOR minus nine comparators x two opponents; 18 simultaneous intervals, stratified public-history bootstrap10000',
        source_hashes={str(ROOT/'code'/p):sha(ROOT/'code'/p) for p in SOURCES},protected_hashes=protected,
        bank_hashes={str(p):sha(p) for p in (OUT/'bank').rglob('*.json')},validation_entries=validation)
    save(OUT/'experiment.json',config)
    (OUT/'PROTOCOL.md').write_bytes((ROOT/'code/E45F_GENERALIZATION_PROTOCOL_20261029.md').read_bytes())
    for p in SOURCES:
        target=OUT/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/'code'/p).read_bytes())
    gallery(entries)
    return config


def request(entry,world,opponent,arm,seed,stage,rep=0):
    config=read(OUT/'experiment.json')
    random_seed=int(np.random.SeedSequence([config['base_seed'],int(sha(entry['path'])[:8],16),int(world=='BA'),rep]).generate_state(1)[0])
    method=f'{arm}{seed}' if arm in LEARNED else arm
    return dict(key=f'{entry["id"]}_{world}_{opponent}_{method}_r{rep}',case=entry['path'],group=entry['id'],
        family=entry['family'],pattern=entry['pattern'],split=entry['split'],suite=entry['suite'],density=entry['density'],
        world=world,opponent=opponent,method=method,train_seed=seed,seed=random_seed,stage=stage,round=0,model_round=12,
        budget_ms=200.,_output=str(OUT),checkpoint=config['checkpoints'][arm][str(seed)] if arm in LEARNED else None)


def invalid(task,failure):
    message=failure['error']
    if failure['task']!=task or not any(msg in message for msg in SUPPORT_ERRORS+('Candidate capacity exceeded','More important regions than the frozen sample budget')):
        raise RuntimeError('Unresolved execution failure: '+message)
    match=re.search(r'public_history_evidence=([^\s\x27\x22;)]+)',message)
    if not match:raise RuntimeError('Support failure without public history evidence')
    evidence=Path(match.group(1)).with_suffix('.json')
    if not evidence.resolve().is_relative_to((OUT/'belief_failures').resolve()):raise RuntimeError('Bad evidence path')
    proof=read(evidence)
    row=dict(**task,valid_episode=False,reason='representation_failure' if proof.get('representation_failure') else 'belief_support_exhaustion',double_capture=None,utility=None,
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
        if failure.exists() and any(msg in read(failure)['error'] for msg in SUPPORT_ERRORS+('Candidate capacity exceeded','More important regions than the frozen sample budget')):return invalid(task,read(failure))
        raise


def worker_init(output,cpus,parent):
    global OUT
    OUT=Path(output);torch.set_num_threads(1)
    ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    if os.getppid()!=parent:raise RuntimeError('Coordinator exited')
    os.sched_setaffinity(0,{cpus.get()});base.PARENT=parent
    config=verify();configure()
    for entry in config['validation_entries']:
        case=read(entry['path']);own=factory(case,request(entry,'AB','fixed_D_kernel','SINGLE_PRIOR',17,'warmup'))
        for obs in case['history'][:-1]:own.observe(obs)
        own.act(case['history'][-1],fixed_simulations=2)
    entry=config['validation_entries'][0];case=read(entry['path'])
    for arm in ARMS:
        own=factory(case,request(entry,'AB','fixed_D_kernel',arm,17 if arm in LEARNED else 0,'warmup'))
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
    if any(not r.get('valid_episode',True) for r in rows):raise RuntimeError('Calibration representation failure')
    wall=sum(sum(r['latencies_ms']) for r in rows);cpu=sum(sum(r['cpu_ms']) for r in rows)
    lat=[t for r in rows for t in r['latencies_ms']]
    return dict(cpu_wall=cpu/max(wall,1e-9),overrun_rate=float(np.mean(np.array(lat)>200)),
        simulations=sum(sum(r['hidden_simulations']) for r in rows),
        by_arm={a:sum(sum(r['hidden_simulations']) for r in rows if family(r)==a) for a in SEARCH_ARMS})


def calibrate(maximum):
    path=OUT/'timing/decision.json'
    if path.exists():return read(path)['workers']
    if SMOKE:
        save(path,dict(workers=min(8,maximum),smoke=True));return min(8,maximum)
    config=read(OUT/'experiment.json');validation=config['validation_entries']
    def tasks(stage):return [request(e,'AB',o,m['arm'],m['seed'],stage)|dict(limit=2)
        for e in validation for o in base.OPPONENTS for m in config['execution_arms']]
    serial=[];affinity=os.sched_getaffinity(0);os.sched_setaffinity(0,{CPUS[0]});configure()
    try:
        for e in validation:
            case=read(e['path'])
            for arm in ARMS:
                own=factory(case,request(e,'AB','fixed_D_kernel',arm,17 if arm in LEARNED else 0,'warmup'))
                for obs in case['history'][:-1]:own.observe(obs)
                own.act(case['history'][-1],fixed_simulations=2)
        jobs=tasks('calibration/serial')
        for task in jobs:
            serial.append(job(task));emit('calibration_serial',completed=len(serial),total=len(jobs),workers=1)
    finally:os.sched_setaffinity(0,affinity)
    ref=timing(serial);candidates=[]
    for n in sorted({maximum}|{v for v in (48,40,32,24,16,8,4) if v<maximum},reverse=True):
        metrics=timing(run_tasks(tasks(f'calibration/parallel_{n}'),n,'calibration_parallel'))
        ratio=metrics['simulations']/max(1,ref['simulations'])
        arm_ratio=min((metrics['by_arm'][a]/ref['by_arm'][a] for a in SEARCH_ARMS if ref['by_arm'][a]>0),default=1.)
        passed=bool(metrics['cpu_wall']>=.85 and ratio>=.85 and arm_ratio>=.65 and metrics['overrun_rate']<=ref['overrun_rate']+.002)
        candidates.append(dict(workers=n,metrics=metrics,simulation_ratio=ratio,minimum_arm_ratio=arm_ratio,passed=passed))
        save(OUT/'timing/calibration.json',dict(reference=ref,candidates=candidates))
        if passed:
            save(path,dict(workers=n,reference=ref,candidates=candidates,outcomes_used=False));return n
    save(path,dict(workers=1,reference=ref,candidates=candidates,outcomes_used=False));return 1


def summarize(rows):
    cells=collections.defaultdict(list)
    for row in rows:cells[row['suite'],row['opponent'],family(row)].append(row)
    summaries=[];by_family=[]
    def describe(rr):
        good=[r for r in rr if r.get('valid_episode',True)];wins=sum(r['double_capture'] for r in good)
        lat=[t for r in good for t in r['latencies_ms']]
        return dict(planned=len(rr),valid=len(good),invalid=len(rr)-len(good),captures=wins,
            capture_rate_bounds=[wins/len(rr),(wins+len(rr)-len(good))/len(rr)],
            utility_valid=float(np.mean([r['utility'] for r in good])) if good else None,
            completion_cost_valid=float(np.mean([r['completion_cost'] for r in good])) if good else None,
            p95_ms=float(np.percentile(lat,95)) if lat else None,overrun_rate_valid=float(np.mean(np.array(lat)>200)) if lat else None)
    for (suite,op,a),rr in sorted(cells.items()):
        summaries.append(dict(suite=suite,opponent=op,arm=a,**describe(rr)))
        for f in sorted({r['family'] for r in rr}):
            by_family.append(dict(suite=suite,family=f,opponent=op,arm=a,**describe([r for r in rr if r['family']==f])))
    primaries={}
    for suite in sorted({r['suite'] for r in rows}):
        rr=[r for r in rows if r['suite']==suite];groups=sorted({r['group'] for r in rr})
        expected={(g,w,o,a,s) for g in groups for w in ('AB','BA') for o in base.OPPONENTS for a in ARMS for s in (SEEDS if a in LEARNED else (0,))}
        observed=[(r['group'],r['world'],r['opponent'],family(r),r['train_seed']) for r in rr]
        complete=len(observed)==len(expected) and set(observed)==expected and all(r.get('valid_episode',True) for r in rr)
        result=dict(status='engineering_only' if SMOKE else 'withheld_incomplete_cells',groups=len(groups),complete_cells=complete,comparisons=[])
        if complete and not SMOKE:
            comparisons=[(o,a) for o in base.OPPONENTS for a in COMPARATORS]
            means={(g,o,a):np.mean([r['utility'] for r in cells[suite,o,a] if r['group']==g]) for g in groups for o in base.OPPONENTS for a in ARMS}
            delta=np.array([[means[g,o,'SINGLE_PRIOR']-means[g,o,a] for o,a in comparisons] for g in groups]);mean=delta.mean(0)
            families={g:next(r['family'] for r in rr if r['group']==g) for g in groups};rng=np.random.default_rng(BASE_SEED+(1 if suite=='benchmark' else 2))
            bs=np.zeros((10000,len(comparisons)))
            for f in sorted(set(families.values())):
                ids=np.array([i for i,g in enumerate(groups) if families[g]==f])
                bs+=delta[ids[rng.integers(len(ids),size=(10000,len(ids)))]].sum(1)/len(groups)
            radius=float(np.quantile(np.max(np.abs(bs-mean),axis=1),.95))
            result.update(status='complete',comparisons=[dict(opponent=o,contrast='SINGLE_PRIOR - '+a,utility_difference=float(mean[i]),
                simultaneous95=[float(mean[i]-radius),float(mean[i]+radius)]) for i,(o,a) in enumerate(comparisons)])
        primaries[suite]=result
    return dict(summaries=summaries,by_family=by_family,primary=primaries,smoke=SMOKE,
        groups=len({r['group'] for r in rows}),primary_status='complete' if primaries and all(x['status']=='complete' for x in primaries.values()) else 'engineering_only' if SMOKE else 'incomplete')


def report(rows=None,stats=None):
    status=read(OUT/'status.json')
    lines=['# E45-F：新地图基线benchmark与障碍布局泛化','',
        '冻结20× round12的9个模型，无新增训练。36组原三类地图新实例benchmark，48组未训练布局组合泛化；84组×2世界×2对手×20执行臂=6720完整对局，旧结果不复用。','',
        '方法：Single PRIOR、FACT PRIOR、Joint PRIOR、Single Value-On、Single DIRECT（各3种子），U、AG、IG、同树几何提议、同树VOO风格提议。后两者是匹配适配，不称完整VOMCPOW或原NTE复现。','',
        f'当前阶段：`{status["stage"]}`；[实时进度](status.json) · [运行日志](run.log)','',
        '[协议](PROTOCOL.md) · [冻结配置](experiment.json) · [地图画廊](layout_gallery.png) · [覆盖审计](coverage_audit.json) · [并发校准](timing/decision.json)']
    if stats:
        lines+=['','|测试套|对手|方法|有效/计划|双捕|完成成本|效用|','|---|---|---|---:|---:|---:|---:|']
        for r in stats['summaries']:
            cost='null' if r['completion_cost_valid'] is None else f'{r["completion_cost_valid"]:.3f}'
            util='null' if r['utility_valid'] is None else f'{r["utility_valid"]:.5f}'
            lines.append(f'|{r["suite"]}|{r["opponent"]}|{r["arm"]}|{r["valid"]}/{r["planned"]}|{r["captures"]}|{cost}|{util}|')
        for suite,data in stats['primary'].items():
            lines+=['',f'## {suite}主比较：{data["status"]}','',f'{data["groups"]}组公共历史，按布局类别分层的组配对bootstrap10000次，18项同时95%区间。','']
            for r in data['comparisons']:lines.append(f'- {r["opponent"]}，{r["contrast"]}: {r["utility_difference"]:+.5f}，{r["simultaneous95"]}')
        lines+=['','[逐局结果](results.json) · [统计及逐布局结果](statistics.json)']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')





def gallery(entries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    chosen=[]
    for f in ID_FAMILIES+OOD_FAMILIES:
        ee=[e for e in entries if e['family']==f]
        for density in sorted({e['density'] for e in ee}):chosen.append(next(e for e in ee if e['density']==density))
    fig,axes=plt.subplots(int(np.ceil(len(chosen)/4)),4,figsize=(16,4.2*int(np.ceil(len(chosen)/4))),squeeze=False)
    for ax,e in zip(axes.flat,chosen):
        c=read(e['path'])
        for box in c['map']:
            b=np.array(box);ax.add_patch(Rectangle(b[:,0],*(b[:,1]-b[:,0]),color='.35'))
        s=np.array(c['worlds'][0]['initial'])[:8].reshape(4,2)
        ax.scatter(s[2:,0],s[2:,1],c='royalblue',marker='^',label='pursuers')
        ax.scatter(s[:2,0],s[:2,1],c='crimson',marker='x',label='hidden truth (plot only)')
        ax.set(xlim=(-10,10),ylim=(-10,10),aspect='equal',title=f'{e["family"]} / {e["density"]} / {len(c["map"])} obstacles')
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(chosen):]:ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'layout_gallery.png',dpi=140);plt.close(fig)
    save(OUT/'coverage_audit.json',dict(existing_coverage='Old tests were new instances of long_wall/wall_corner/multi_route, generally1-2 rectangles; no frozen-policy new-layout benchmark',
        missing_filled=['new obstacle-layout combinations','obstacle count3-6 and gap/shape changes','Single direct same checkpoint','matched geometric and adaptive continuous proposal search'],
        histories_by_suite=dict(collections.Counter(e['suite'] for e in entries)),
        layout_counts=dict(collections.Counter(e['family'] for e in entries)),
        obstacle_count_histogram=dict(collections.Counter(read(e['path'])['geometry_summary']['obstacles'] for e in entries)),
        snapshot_only_plot_uses_hidden_truth=True,hidden_truth_never_controller_input=True,
        scope='Controlled layout-combination shift with original causal scenario scaffolding; not arbitrary-map or unseen-dynamics generalization'))


def main():
    global OUT,SMOKE,CPUS
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=48);parser.add_argument('--smoke',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args();SMOKE=args.smoke
    if args.output:OUT=args.output.resolve()
    elif SMOKE:OUT=OUT.with_name(OUT.name+'_smoke')
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'coordinator.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=prepare();verify();configure()
        if not (OUT/'clock.json').exists():save(OUT/'clock.json',dict(started_unix=time.time(),cap_seconds=12*3600))
        execution=OUT/'execution.json';CPUS=read(execution)['cpus'] if execution.exists() else choose_cpus(args.workers)
        if not CPUS:raise RuntimeError('No available physical cores')
        maximum=min(args.workers,len(CPUS))
        save(execution,dict(pid=os.getpid(),cpus=CPUS,max_workers=maximum,threads_per_worker=1,training=False,tmux='train:E45F'))
        try:
            emit('registered',total=config['formal_tasks'],max_workers=maximum);report()
            workers=calibrate(maximum);entries=read(OUT/'bank/manifest.json')['entries']
            tasks=[request(e,w,o,m['arm'],m['seed'],'evaluation') for e in entries for w in ('AB','BA') for o in base.OPPONENTS for m in config['execution_arms']]
            if SMOKE:tasks=[t|dict(limit=2,fixed=4) for t in tasks]
            save(OUT/'requests.json',tasks);np.random.default_rng(BASE_SEED+77).shuffle(tasks)
            emit('evaluation_start',total=len(tasks),workers=workers);report()
            rows=run_tasks(tasks,workers,'evaluation');save(OUT/'results.json',rows)
            stats=summarize(rows);save(OUT/'statistics.json',stats);verify()
            for suite in ('benchmark','generalization'):
                save(OUT/suite/'results.json',[r for r in rows if r['suite']==suite])
            emit('complete',completed=len(rows),total=len(tasks),workers=workers,
                invalid=sum(not r.get('valid_episode',True) for r in rows),primary_status=stats['primary_status'])
            report(rows,stats);save(OUT/'decision.json',dict(status='complete',attempted=len(rows),primary_status=stats['primary_status'],training=False))
        except BaseException as exc:
            save(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
            emit('failed',error=repr(exc));report();raise


if __name__=='__main__':main()
