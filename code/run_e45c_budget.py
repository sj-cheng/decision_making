"""E45-C budget scaling, reusing the frozen E45-B scientific implementation."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[_key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('TZ','Asia/Shanghai')
import argparse
import collections
import concurrent.futures as futures
import ctypes
import fcntl
import hashlib
import multiprocessing as mp
import signal
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import run_e45b_role_free as base
from e45b_scenarios import make_case
from e45c_capture_audit import audit_simulated
from e45c_belief_diagnostics import install as install_belief_diagnostics
from run_frozen_d_closed_loop import save,sha
from solvers.e45b_joint_nte import Controller
from solvers.e45_team_adapter import TeamController
from learning.e45b_joint_network import KINDS,Network
from run_e44_parallel import choose_cpus

ROOT=base.ROOT
OLD=ROOT/'current/e45b_role_free_joint_nte_20261027'
OUT=ROOT/'current/e45c_budget_scaling_20261028'
BASE_SEED=945320261028
SEEDS=(17,29,43)
DOSES=('x1','x5','x20')
BUDGETS=(200,2000,10000)
SMOKE=False
WORKERS=40
CPUS=[]
WALL_CAP=48*3600
SOURCES=('run_e45c_budget.py','E45C_BUDGET_PROTOCOL_20261028.md','tests/test_e45c_budget.py',
    'e45c_capture_audit.py','tests/test_e45c_capture_audit.py','tests/fixtures/e45c_capture_pairing_v1.npz',
    'e45c_belief_diagnostics.py','tests/test_e45c_belief_diagnostics.py')


def read(path):return base.read(path)
def counts():return dict(zip(DOSES,(3,6,12) if SMOKE else (24,120,480)))
def rounds():return 2 if SMOKE else 12
def methods():return ('AG','U','JOINT17','PRIOR17','DIRECT17') if SMOKE else base.METHODS
def search_methods():return ('U','JOINT17','PRIOR17') if SMOKE else ('U',)+tuple(f'{m}{s}' for m in ('JOINT','PRIOR') for s in SEEDS)
def level(dose):return OUT/'learning'/dose
def entries(split,dose=None):
    rows=[r for r in read(OUT/'bank/manifest.json')['entries'] if r['split']==split]
    return [r for r in rows if r['family_index']<counts()[dose]//3] if split=='train' and dose else rows


def emit(stage,**values):
    import json
    row=dict(pid=os.getpid(),timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),stage=stage,**values)
    save(OUT/'status.json',row)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)


def verify():
    c=read(OUT/'experiment.json')
    override={}
    expected=c['source_hashes'].copy()
    for name in ('REPAIR_CAPTURE_AUDIT_20261028.json','REPAIR_SUPPORT_DIAGNOSTICS_20261028.json'):
        amendment=OUT/name
        if not amendment.exists():continue
        repair=read(amendment)
        assert repair['original_experiment_sha256']==sha(OUT/'experiment.json')
        if 'previous_amendment' in repair:
            assert sha(repair['previous_amendment'])==repair['previous_amendment_sha256']
        for p,item in repair['overrides'].items():
            assert expected[p]==item['original_sha256']
            expected[p]=item['repaired_sha256']
            override[p]=item['repaired_sha256']
        for p,h in repair['extra_hashes'].items():
            if sha(p)!=h:raise RuntimeError('Frozen E45-C repair artifact changed: '+p)
    for section in ('source_hashes','protected_hashes','bank_hashes'):
        for p,h in c[section].items():
            if section=='source_hashes':h=override.get(p,h)
            if sha(p)!=h:raise RuntimeError('Frozen E45-C artifact changed: '+p)
    return c


def make_controller(case,task):
    method=task['method'];budget=task['budget_ms'];seed=task['seed']
    if method in ('AG','IG'):return TeamController(case['map'],case['speed'],seed,method,budget_ms=budget)
    if method=='U':return Controller(case['map'],case['speed'],seed,budget_ms=budget)
    kind=next(k for k in ('VALUE_U','SINGLE','FACT','JOINT','DIRECT','PRIOR') if method.startswith(k))
    train_seed=int(method[len(kind):]);name=dict(SINGLE='single',FACT='factorized',JOINT='joint',DIRECT='joint',VALUE_U='joint',PRIOR='joint')[kind]
    path=Path(task['checkpoint']) if task.get('checkpoint') else base.model_file(name,train_seed,task['model_round'])
    return Controller(case['map'],case['speed'],seed,path,use_policy=kind!='VALUE_U',use_value=kind!='PRIOR',direct=kind=='DIRECT',budget_ms=budget)


def configure(path,smoke=None):
    base.OUT=Path(path);base.SMOKE=SMOKE if smoke is None else smoke
    base.BASE_SEED=BASE_SEED;base.make_controller=make_controller
    base.audit_simulated=audit_simulated
    install_belief_diagnostics(OUT/'belief_failures')


def prepare():
    if (OUT/'experiment.json').exists():return verify()
    OUT.mkdir(parents=True,exist_ok=True)
    original=base.OUT;base.OUT=OLD;old=base.verify();base.OUT=original
    known={hashlib.sha256(str(read(p)['map']).encode()).hexdigest() for p in (OLD/'bank').glob('*/*.json')}
    rows=[];rejected=[]
    namespace=int(hashlib.sha256(OUT.name.encode()).hexdigest()[:8],16) if SMOKE else 0
    for split,total in (('train',counts()['x20']),('validation',3 if SMOKE else 12),('test',3 if SMOKE else 36)):
        for family in ('long_wall','wall_corner','multi_route'):
            accepted=0;index=0
            while accepted<total//3:
                try:
                    case=make_case(family,index,split,BASE_SEED+namespace)
                    digest=hashlib.sha256(str(case['map']).encode()).hexdigest()
                    if digest in known:raise ValueError('Duplicate public geometry')
                except (AssertionError,ValueError) as exc:
                    rejected.append(dict(split=split,family=family,index=index,error=repr(exc)))
                    index+=1
                    if index>total*10:raise RuntimeError('Too many geometric rejections')
                    continue
                known.add(digest);path=OUT/'bank'/split/(case['id']+'.json');save(path,case)
                rows.append(dict(id=case['id'],path=str(path),family=family,pattern=case['pattern'],split=split,family_index=accepted,source_index=index))
                accepted+=1;index+=1
            emit('prepare_bank',split=split,family=family,accepted=accepted,rejections=len(rejected))
    save(OUT/'bank/manifest.json',dict(entries=rows,rejections=rejected,selection='public geometry and causal history only; no controller outcomes'))
    protected=old['protected_hashes'].copy()
    protected.update({p:sha(p) for p in old['source_hashes']})
    repair=read(OLD/'REPAIR_SENSOR_AUDIT_20261027.json')
    protected.update(repair['extra_hashes'])
    protected.update(read(OLD/'models_frozen.json')['files'])
    for name in ('experiment.json','decision.json','models_frozen.json','evaluation/results.json','evaluation/statistics.json','PROTOCOL.md','REPAIR_SENSOR_AUDIT_20261027.json'):
        protected[str(OLD/name)]=sha(OLD/name)
    c=dict(experiment='E45-C',smoke=SMOKE,base_seed=BASE_SEED,seeds=list(SEEDS),doses=list(DOSES),train_cases=counts(),validation_cases=3 if SMOKE else 12,test_cases=3 if SMOKE else 36,
        rounds=rounds(),epochs_per_round=2 if SMOKE else 30,max_states_per_episode=2 if SMOKE else 32,
        search_budget_ms=list(BUDGETS),methods=list(methods()),search_methods=list(search_methods()),train_decision_ms=400,online_decision_ms=200,
        source_hashes={str(ROOT/'code'/s):sha(ROOT/'code'/s) for s in SOURCES},protected_hashes=protected,
        bank_hashes={str(p):sha(p) for p in (OUT/'bank').rglob('*.json')},performance_gate=False,wall_cap_seconds=WALL_CAP,
        dose_scope='new maps/episodes/replay and optimizer updates at fixed epochs; not pure data scaling at fixed updates',
        architecture={k:sum(p.numel() for p in Network(k).parameters()) for k in KINDS})
    save(OUT/'experiment.json',c)
    (OUT/'PROTOCOL.md').write_bytes((ROOT/'code/E45C_BUDGET_PROTOCOL_20261028.md').read_bytes())
    for relative in SOURCES:
        path=OUT/'source_snapshot'/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/'code'/relative).read_bytes())
    for dose in DOSES:
        folder=level(dose);folder.mkdir(parents=True,exist_ok=True)
        save(folder/'experiment.json',dict(experiment='E45-C/'+dose,rounds=rounds(),epochs_per_round=c['epochs_per_round'],max_states_per_episode=c['max_states_per_episode'],
            parent_experiment_sha256=sha(OUT/'experiment.json'),train_cases=counts()[dose],smoke=SMOKE))
        configure(folder);base.initialize_models()
    configure(OUT)
    emit('registered',train_cases=counts(),rounds=rounds(),workers=WORKERS)
    report('协议已冻结，工程验收待执行。')
    return c


def request(entry,world,opponent,method,stage,output,round_=0,model_round=0,budget=200,checkpoint=None):
    wi=int(world=='BA')
    seed=int(np.random.SeedSequence([BASE_SEED,int(sha(entry['path'])[:8],16),wi,round_ if stage.startswith('collection') else 0]).generate_state(1)[0])
    return dict(key=f'{entry["id"]}_{world}_{opponent}_{method}',case=entry['path'],group=entry['id'],family=entry['family'],pattern=entry['pattern'],split=entry['split'],
        world=world,opponent=opponent,method=method,seed=seed,stage=stage,round=round_,model_round=model_round,budget_ms=float(budget),_output=str(output),checkpoint=str(checkpoint) if checkpoint else None)


def collection_tasks(dose,round_):
    tasks=[]
    for split in ('train','validation'):
        for i,e in enumerate(entries(split,dose)):
            for s,seed in enumerate(SEEDS):
                pairs=[(('AB','BA')[(e['family_index']+s+round_)%2], 'fixed_D_kernel' if round_<=4 else base.OPPONENTS[(e['family_index']+s+round_)%2])] if split=='train' else [(w,o) for w in ('AB','BA') for o in base.OPPONENTS]
                for world,opp in pairs:tasks.append(request(e,world,opp,f'JOINT{seed}','collection/'+split,level(dose),round_,round_-1,400))
    return tasks


def child_init(main,smoke,queue,parent,training):
    global OUT,SMOKE
    OUT=Path(main);SMOKE=smoke;base.PARENT=parent;torch.set_num_threads(1)
    ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    if os.getppid()!=parent:raise RuntimeError('Coordinator exited during init')
    slot=queue.get()
    if training:
        base.DEVICE=f'cuda:{slot}';torch.cuda.set_device(slot);torch.use_deterministic_algorithms(True)
    else:os.sched_setaffinity(0,{slot})
    verify()
    configure(OUT)
    if not training:
        for family in ('long_wall','wall_corner','multi_route'):
            e=next(e for e in entries('validation') if e['family']==family);case=read(e['path'])
            own=Controller(case['map'],case['speed'],317)
            for obs in case['history'][:-1]:own.observe(obs)
            own.act(case['history'][-1],fixed_simulations=2)
            audit_simulated(np.zeros((0,26)),np.asarray(case['map']),case['speed'])


def episode_job(task):
    configure(task['_output'])
    return base.episode_job(task)


def train_job(task):
    configure(task['_output'])
    return base.train_job(task)


class Pool:
    def __init__(self,count,training=False):
        self.training=training;ctx=mp.get_context('spawn');self.queue=ctx.Queue()
        for value in ([2,3][:count] if training else CPUS[:count]):self.queue.put(value)
        self.executor=futures.ProcessPoolExecutor(max_workers=count,mp_context=ctx,initializer=child_init,initargs=(str(OUT),SMOKE,self.queue,os.getpid(),training))
    def submit(self,task):return self.executor.submit(train_job if self.training else episode_job,task)
    def close(self,failed=False):
        processes=list((self.executor._processes or {}).values())
        self.executor.shutdown(wait=not failed,cancel_futures=failed)
        if failed:
            for p in processes:
                if p.is_alive():p.terminate()
            for p in processes:p.join(timeout=5)
        self.queue.close()


def run_jobs(stage,tasks,workers=None):
    tasks=list(tasks);results=[];remaining=collections.deque()
    for task in tasks:
        configure(task['_output']);cached=base.cached(task)
        if cached is None:remaining.append(task)
        else:results.append(cached)
    configure(OUT);emit(stage,completed=len(results),total=len(tasks),workers=min(workers or WORKERS,len(tasks)))
    if not remaining:return sorted(results,key=lambda r:r['key'])
    count=min(workers or WORKERS,len(remaining));pool=Pool(count);pending={};failed=True
    try:
        while remaining or pending:
            while remaining and len(pending)<2*count:
                task=remaining.popleft();pending[pool.submit(task)]=task
            done,_=futures.wait(pending,timeout=30,return_when=futures.FIRST_COMPLETED)
            for future in done:
                task=pending.pop(future)
                try:results.append(future.result())
                except BaseException:
                    save(OUT/'failed_task.json',task);raise
            emit(stage,completed=len(results),total=len(tasks),workers=count)
        failed=False
    finally:pool.close(failed)
    return sorted(results,key=lambda r:(r['stage'],r['key']))


def search_entries():
    rows=[r for r in read(OLD/'bank/manifest.json')['entries'] if r['split']=='validation']
    if SMOKE:rows=[next(e for e in rows if e['family']==f) for f in ('long_wall','wall_corner','multi_route')]
    return rows


def old_checkpoint(method):
    return OLD/'round_12/models'/f'joint_{int(method.removeprefix("JOINT").removeprefix("PRIOR"))}.npz' if method!='U' else None


def probe_tasks():
    return [request(e,'AB','rootpuct',method,f'search/probes/budget_{budget}',OUT,budget=budget,checkpoint=old_checkpoint(method))|dict(limit=1) for e in search_entries() for method in search_methods() for budget in BUDGETS]


def search_tasks():
    result=[]
    for e in search_entries():
        for w in ('AB','BA'):
            for opp in base.OPPONENTS:
                for budget in BUDGETS:
                    for m in search_methods():result.append(request(e,w,opp,m,f'search/closed_loop/budget_{budget}',OUT,budget=budget,checkpoint=old_checkpoint(m)))
                result.append(request(e,w,opp,'AG','search/closed_loop/budget_200',OUT))
    return result


def engineering():
    path=OUT/'engineering.json'
    if path.exists():return
    selected=[next(e for e in entries('validation') if e['family']==f) for f in ('long_wall','wall_corner','multi_route')]
    tasks=[request(e,'AB',o,m,'engineering',level('x1'),model_round=0)|dict(limit=2,fixed=4) for e in selected for o in base.OPPONENTS for m in ('AG','U','SINGLE17','FACT17','JOINT17','DIRECT17','VALUE_U17','PRIOR17')]
    rows=run_jobs('engineering',tasks);save(path,dict(passed=True,prefixes=len(rows),outcomes_used=False))


def finish_models(dose,round_,records):
    folder=level(dose)/f'round_{round_:02d}'
    assert len(records)==9
    save(folder/'training.json',dict(completed=True,round=round_,records=sorted(records,key=lambda r:(r['seed'],r['kind']))))
    save(folder/'models_manifest.json',dict(files={str(p):sha(p) for p in (folder/'models').iterdir() if p.is_file()}))


def verify_models(path):
    for p,h in read(path)['files'].items():
        if sha(p)!=h:raise RuntimeError('Frozen E45-C model changed: '+p)


def learn():
    cpu=Pool(WORKERS);gpu=Pool(2,True);active={};cpu_queues={};gpu_queues={};received={};fit_records={};state={};failed=True
    def advance(dose,r):
        while r<=rounds():
            folder=level(dose)/f'round_{r:02d}'
            if (folder/'models_manifest.json').exists():verify_models(folder/'models_manifest.json');r+=1;continue
            state[dose]=dict(round=r,phase='collect',completed=0)
            manifest=folder/'collection_results.json'
            if manifest.exists():
                for row in read(manifest):assert sha(row['episode'])==row['episode_sha256'] and sha(row['data'])==row['data_sha256']
                queue_train(dose,r);return
            tasks=collection_tasks(dose,r);received[dose]=[];cpu_queues[dose]=collections.deque()
            for task in tasks:
                configure(task['_output']);cached=base.cached(task)
                if cached is not None:received[dose].append(cached)
                else:cpu_queues[dose].append(task)
            state[dose].update(completed=len(received[dose]),total=len(tasks))
            configure(OUT)
            if not cpu_queues[dose]:finish_collection(dose,r)
            return
        state[dose]=dict(round=rounds(),phase='done');cpu_queues.pop(dose,None);gpu_queues.pop(dose,None)
    def queue_train(dose,r):
        fit_records[dose]=[];gpu_queues[dose]=collections.deque(dict(kind=k,seed=s,round=r,_output=str(level(dose))) for s in SEEDS for k in KINDS)
        state[dose]=dict(round=r,phase='training',completed=0,total=9)
    def finish_collection(dose,r):
        save(level(dose)/f'round_{r:02d}/collection_results.json',sorted(received[dose],key=lambda v:v['key']))
        cpu_queues.pop(dose,None);queue_train(dose,r)
    try:
        for dose in DOSES:advance(dose,1)
        while any(v['phase']!='done' for v in state.values()):
            for training,queues,limit,pool in ((False,cpu_queues,2*WORKERS,cpu),(True,gpu_queues,4,gpu)):
                # Small queues get latency priority; large jobs still fill idle cores.
                for dose in sorted(list(queues),key=lambda d:len(queues[d])):
                    while queues[dose] and sum(1 for meta in active.values() if meta[1]==training)<limit:
                        task=queues[dose].popleft();active[pool.submit(task)]=(dose,training,task)
            if not active:raise RuntimeError('Learning scheduler stalled')
            done,_=futures.wait(active,timeout=30,return_when=futures.FIRST_COMPLETED)
            for future in done:
                dose,training,task=active.pop(future)
                try:row=future.result()
                except BaseException:
                    save(OUT/'failed_task.json',task);raise
                r=state[dose]['round']
                if training:
                    fit_records[dose].append(row);state[dose]['completed']=len(fit_records[dose])
                    if len(fit_records[dose])==9:
                        finish_models(dose,r,fit_records[dose]);gpu_queues.pop(dose,None);advance(dose,r+1);report('学习经验曲线训练进行中。')
                else:
                    received[dose].append(row);state[dose]['completed']=len(received[dose])
                    if len(received[dose])==state[dose]['total']:finish_collection(dose,r)
            configure(OUT);emit('learning',doses=state,cpu_pending=sum(not m[1] for m in active.values()),gpu_pending=sum(m[1] for m in active.values()))
        failed=False
    finally:cpu.close(failed);gpu.close(failed);configure(OUT)
    files={str(p):sha(p) for dose in DOSES for p in (level(dose)/f'round_{rounds():02d}/models').glob('*')}
    save(OUT/'models_frozen.json',dict(final_models=27,rounds=rounds(),files=files))


def calibration_tasks(stage):
    tasks=[]
    for f in ('long_wall','wall_corner','multi_route'):
        e=next(e for e in entries('validation') if e['family']==f)
        for o in base.OPPONENTS:
            for dose in DOSES:
                for m in ('JOINT17','PRIOR17'):
                    tasks.append(request(e,'AB',o,m,stage,level(dose),model_round=rounds())|dict(limit=2))
            for m in ('AG','U'):tasks.append(request(e,'AB',o,m,stage,OUT)|dict(limit=2))
    return tasks


def timing(rows):
    wall=sum(sum(r['latencies_ms']) for r in rows);cpu=sum(sum(r['cpu_ms']) for r in rows);lat=[t for r in rows for t in r['latencies_ms']]
    samples=collections.defaultdict(int)
    for r in rows:
        if r['method']!='AG':samples[(r['_output'],r['method'])]+=sum(r['hidden_simulations'])
    return dict(cpu_wall=cpu/max(wall,1e-9),overrun_rate=float(np.mean(np.asarray(lat)>200)),p95_ms=float(np.percentile(lat,95)),simulations=sum(samples.values())),samples


def calibrate():
    path=OUT/'timing/decision.json'
    if path.exists():return read(path)['workers']
    if SMOKE:save(path,dict(workers=WORKERS,smoke=True));return WORKERS
    serial=[]
    for i,t in enumerate(calibration_tasks('calibration/serial')):
        serial.append(episode_job(t));emit('timing_serial',completed=i+1,total=len(calibration_tasks('calibration/serial')))
    configure(OUT);ref,per_ref=timing(serial);candidates=[]
    for n in sorted({WORKERS}|{n for n in (40,32,24,16,8,4) if n<WORKERS},reverse=True):
        rows=run_jobs('timing_parallel',calibration_tasks(f'calibration/parallel_{n}'),n);metrics,per=timing(rows)
        ratio=metrics['simulations']/max(1,ref['simulations']);minimum=min(per[k]/max(1,v) for k,v in per_ref.items())
        accepted=bool(metrics['cpu_wall']>=.85 and ratio>=.85 and minimum>=.65 and metrics['overrun_rate']<=ref['overrun_rate']+.002)
        candidates.append(dict(workers=n,metrics=metrics,simulation_ratio=ratio,min_arm_ratio=minimum,accepted=accepted))
        save(OUT/'timing/calibration.json',dict(reference=ref,candidates=candidates))
        if accepted:save(path,dict(workers=n,reference=ref,candidates=candidates,outcomes_used=False));return n
    save(path,dict(workers=1,reference=ref,candidates=candidates,outcomes_used=False));return 1


def evaluation_tasks():
    return [request(e,w,o,m,'evaluation',level(dose),model_round=rounds()) for dose in DOSES for e in entries('test') for w in ('AB','BA') for o in base.OPPONENTS for m in methods()]


def compare(rows,pairs,scope):
    groups=sorted({r['group'] for r in rows});matrix=[]
    def mean(g,opp,dose,method,budget=None):
        selected=[r['utility'] for r in rows if r['group']==g and r['opponent']==opp and (dose is None or r['_output']==str(level(dose)))
            and (r['method']==method if method in ('AG','IG','U') else r['method'].startswith(method)) and (budget is None or r['budget_ms']==budget)]
        if not selected:raise RuntimeError('Missing comparison cell')
        return np.mean(selected)
    for g in groups:
        matrix.append([mean(g,o,d1,m,b1)-mean(g,o,d0,m,b0) for o,m,d1,d0,b1,b0 in pairs])
    delta=np.asarray(matrix);center=delta.mean(0);rng=np.random.default_rng(BASE_SEED+774)
    bs=delta[rng.integers(len(groups),size=(10000,len(groups)))].mean(1);radius=float(np.quantile(np.max(abs(bs-center),axis=1),.95))
    return dict(scope=scope,groups=len(groups),comparisons=[dict(opponent=o,method=m,treatment_dose=d1,reference_dose=d0,treatment_budget=b1,reference_budget=b0,
        utility_difference=float(center[j]),simultaneous95=[float(center[j]-radius),float(center[j]+radius)]) for j,(o,m,d1,d0,b1,b0) in enumerate(pairs)])


def statistics(search,learning):
    search_pairs=[(o,m,None,None,b,200) for o in base.OPPONENTS for m in ('JOINT','PRIOR','U') for b in (2000,10000)]
    learning_pairs=[(o,m,d,'x1',None,None) for o in base.OPPONENTS for m in ('JOINT','PRIOR') for d in ('x5','x20')]
    result=dict(search=compare(search,search_pairs,'Development budget response on old validation histories'),learning=compare(learning,learning_pairs,'New held-out public histories; 8 primary simultaneous comparisons'),smoke=SMOKE,arms=[])
    cells=collections.defaultdict(list)
    for source,rows in (('search',search),('learning',learning)):
        for r in rows:cells[(source,r['_output'],r['opponent'],r['method'],r['budget_ms'])].append(r)
    for (source,output,opp,method,budget),rr in sorted(cells.items()):
        lat=[t for r in rr for t in r['latencies_ms']]
        result['arms'].append(dict(source=source,output=output,opponent=opp,method=method,budget_ms=budget,games=len(rr),double_capture=sum(r['double_capture'] for r in rr),utility=float(np.mean([r['utility'] for r in rr])),cost=float(np.mean([r['completion_cost'] for r in rr])),p95_ms=float(np.percentile(lat,95)),overruns=sum(r['online_overruns'] for r in rr)))
    save(OUT/'statistics.json',result);return result


def report(state):
    lines=['# E45-C：搜索与学习经验预算曲线','',state,'','[冻结协议](PROTOCOL.md) · [实时进度](status.json) · [配置](experiment.json) · [统计](statistics.json)','',
        '训练三档为24/120/480张嵌套地图，各三网络×三种子×12轮×30epoch；另测冻结E45-B的200ms/2s/10s搜索。无手工角色或任务分配。',
        '正式计划27648局完整采集、9072局新测试、528局开发搜索响应及126次根探针。隔离smoke的短前缀不计正式成果。','',
        '|经验档位|已完成轮数|已汇总采集局数|最终模型|','|---|---:|---:|---:|']
    if (OUT/'REPAIR_SUPPORT_DIAGNOSTICS_20261028.json').exists():
        lines[8:8]=['[2026-10-28 信念支持中断与修复记录](repair_support_v2/REPAIR.md)：空支持时用同一D运动核与实际团队传感器边界重算；正常路径不变。原中断缺少完整现场，是否同一机制尚未确认。','']
    for dose in DOSES:
        done=len(list(level(dose).glob('round_*/models_manifest.json')));n=sum(len(read(p)) for p in level(dose).glob('round_*/collection_results.json'))
        lines.append(f'|{dose}|{done}/{rounds()}|{n}|{9 if done==rounds() else 0}/9|')
    if (OUT/'statistics.json').exists():
        s=read(OUT/'statistics.json');lines+=['','|实验|方法|条件|比较|效用差|同时95%区间|','|---|---|---|---|---:|---|']
        for key in ('search','learning'):
            for r in s[key]['comparisons']:
                label=f'{r["treatment_dose"]}−{r["reference_dose"]}' if key=='learning' else f'{r["treatment_budget"]}−200ms'
                lines.append(f'|{key}|{r["method"]}|{r["opponent"]}|{label}|{r["utility_difference"]:+.4f}|{r["simultaneous95"]}|')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def pipeline():
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare();verify();clock=OUT/'clock.json'
    if not clock.exists():save(clock,dict(started_unix=time.time(),cap_seconds=WALL_CAP))
    elapsed=time.time()-read(clock)['started_unix'];remaining=int(WALL_CAP-elapsed)
    if remaining<=0:raise RuntimeError('Registered 48h wall budget exhausted')
    def stop(signum,*_):raise RuntimeError('Budget expired' if signum==signal.SIGALRM else 'Coordinator interrupted by signal '+str(signum))
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(remaining)
    try:
        engineering()
        probe=OUT/'search/probe_results.json'
        if not probe.exists():save(probe,run_jobs('root_budget_probes',probe_tasks()))
        learn();verify()
        concurrency=calibrate()
        search_path=OUT/'search/results.json'
        search=read(search_path) if search_path.exists() else run_jobs('search_budget_closed_loop',search_tasks(),concurrency)
        save(search_path,search)
        eval_path=OUT/'evaluation/results.json'
        evaluation=read(eval_path) if eval_path.exists() else run_jobs('learning_budget_test',evaluation_tasks(),concurrency)
        save(eval_path,evaluation)
        expected_search=len(search_tasks());expected_eval=len(evaluation_tasks())
        assert len(search)==expected_search and len(evaluation)==expected_eval
        if not SMOKE:assert all(r['reason'] in ('double_capture','timeout') for r in search+evaluation)
        statistics(search,evaluation);verify();verify_models(OUT/'models_frozen.json')
        save(OUT/'decision.json',dict(status='complete',smoke=SMOKE,trained_models=27,learning_test_games=len(evaluation),search_closed_loop_games=len(search),elapsed_seconds=time.time()-read(clock)['started_unix']))
        report('完整预算流水线完成。' if not SMOKE else '隔离工程smoke完成；不作方法收益结论。');emit('complete',learning_test_games=len(evaluation),search_games=len(search))
    except BaseException as exc:
        save(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),previous_status=read(OUT/'status.json')))
        report('执行中断，现场保留：'+repr(exc));emit('execution_failure',error=repr(exc));raise
    finally:signal.alarm(0)


def main():
    global OUT,WORKERS,SMOKE,CPUS
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','pipeline','smoke','status'));parser.add_argument('--output',type=Path);parser.add_argument('--workers',type=int,default=40);args=parser.parse_args()
    if args.output:OUT=args.output.resolve()
    WORKERS=args.workers;SMOKE=args.stage=='smoke';torch.set_num_threads(1)
    if args.stage=='status':print(read(OUT/'status.json'));return
    if args.stage=='prepare':prepare();return
    path=OUT/'execution.json';OUT.mkdir(parents=True,exist_ok=True)
    if path.exists():
        c=read(path);assert c['workers']==WORKERS;CPUS=c['physical_cores']
    else:
        CPUS=choose_cpus(WORKERS)
        if len(CPUS)<WORKERS:raise RuntimeError('Not enough distinct physical cores')
        save(path,dict(workers=WORKERS,physical_cores=CPUS,gpus=[2,3],pid=os.getpid(),timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    pipeline()


if __name__=='__main__':main()
