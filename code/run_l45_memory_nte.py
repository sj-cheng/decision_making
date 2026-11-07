"""Registered memory-conditioned NTE development experiment. Old runs are read-only."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
import collections
import copy
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
import run_l45_occluded_comparison as base
from continue_l45_occluded_comparison import recognized, rng_globals, fingerprint_controller
from memory_nte_scenarios import construct
from prepare_l45_occlusion_scenarios import store_case
from verify_l45_occlusion_scenarios import verify_case, geometry_tests

ROOT=Path(__file__).resolve().parent.parent
DEFAULT=ROOT/'current/l45_memory_nte_20261025'
SEED=62020261025
FAMILIES=('long_wall','wall_corner','multi_route')
put,sha,save,load=base.put,base.sha,base.save_torch,base.load_torch


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def configure(output):
    # Each worker has its own module namespace; the frozen runner file is untouched.
    base.SOURCE=output


def sources():
    files=[Path(__file__),ROOT/'code/memory_nte_scenarios.py',
        ROOT/'code/solvers/local_pursuit.py',Path(base.__file__),
        ROOT/'code/continue_l45_occluded_comparison.py',ROOT/'code/prepare_l45_occlusion_scenarios.py',
        ROOT/'code/verify_l45_occlusion_scenarios.py',ROOT/'code/problems/example8.py',
        ROOT/'code/solvers/root_audit_puct.py',ROOT/'code/solvers/c_puct.py']
    files+=list((ROOT/'code/cpp/build_root_audit').glob('bindings_root_audit*.so'))
    return {str(p):sha(p) for p in files}


def check_frozen(output):
    lock=json.loads((output/'experiment.json').read_text())
    for p,h in lock['sources'].items():
        if sha(p)!=h:raise RuntimeError('Frozen source changed: '+p)
    for p,h in lock['artifacts'].items():
        if sha(output/p)!=h:raise RuntimeError('Frozen artifact changed: '+p)
    return lock


def seed(*keys):return int(np.random.SeedSequence([SEED,*keys]).generate_state(1)[0])


def historical_geometry(output):
    """Inventory actual historical public maps, including unused and smoke banks."""
    result={};lengths=set()
    for path in (ROOT/'current').glob('**/controller_inputs/*.json'):
        if path.is_relative_to(output):continue
        value=json.loads(path.read_text())
        result[str(path)]=sha(path)
        for b in value.get('map',[]):
            lengths.add(round(max(b[0][1]-b[0][0],b[1][1]-b[1][0]),9))
    # Existing full-state banks have the original rectangular maps. Include all
    # JSON obstacle/map arrays without loading huge trajectory binary archives.
    for folder in (ROOT/'current').iterdir():
        if not folder.is_dir() or folder.resolve()==output:continue
        for pattern in ('**/scenarios.json','**/bank.json','**/cases.json','**/all_cases.json','**/diagnostic_bank.json'):
            for path in folder.glob(pattern):
                result[str(path)]=sha(path)
                def scan(v):
                    if isinstance(v,dict):
                        for k,x in v.items():
                            if k in ('map','obstacles') and isinstance(x,list):
                                for b in x:
                                    a=np.asarray(b)
                                    if a.shape==(2,2) and np.issubdtype(a.dtype,np.number):
                                        lengths.add(round(float(np.max(a[:,1]-a[:,0])),9))
                            if isinstance(x,(dict,list)):scan(x)
                    elif isinstance(v,list):
                        for x in v:scan(x)
                scan(json.loads(path.read_text()))
    return result,lengths


def prepare(output):
    if (output/'experiment.json').exists():
        check_frozen(output);return
    output.mkdir(parents=True,exist_ok=True)
    history,lengths=historical_geometry(output)
    entries=[];accepted=[];rejected=[];used=set();seeds={}
    # Distinct wall lengths, not just rigid transformations. Split is assigned
    # before construction and never depends on controller outcomes.
    for fi,family in enumerate(FAMILIES):
        for j in range(60):
            split='train' if j<32 else 'validation' if j<40 else 'test'
            H=4 if j%2==0 else 6
            speed=(1.1,1.3)[(j//2)%2]
            for attempt in range(100):
                s=seed(1,fi,j,attempt);rng=np.random.default_rng(s)
                anchor=(5.5,6.,6.5)[(j//2)%3] if H==4 else (8.,8.5)[(j//2)%2]
                height=anchor+float(rng.uniform(.015,.075))
                length=round(2*height,9)
                if length in lengths or length in used:continue
                try:
                    env,pub=construct(family,H,speed,s,height,j%4)
                    cert=verify_case(env,pub)
                except ValueError as error:
                    rejected.append(dict(family=family,index=j,seed=s,height=height,reason=str(error)));continue
                if not cert['passed']:
                    rejected.append(dict(family=family,index=j,seed=s,height=height,reasons=cert['reasons']));continue
                break
            else:raise RuntimeError('Cannot construct an accepted geometric instance')
            entry=store_case(output/'bank',env,pub)
            entry.update(split=split,family=family,geometry_id=digest(dict(family=family,height=height)),
                         teacher_check=(j<8),height=height)
            entries.append(entry);accepted.append(cert);used.add(length)
            seeds[entry['id']]=dict(controller=pub['controller_seeds'],opponent=[seed(2,fi,j,k) for k in range(2)])
    assert collections.Counter(e['split'] for e in entries)==dict(train=96,validation=24,test=60)
    assert len(used)==180 and sum(e['teacher_check'] for e in entries)==24
    put(output/'bank/manifest.json',dict(entries=entries))
    put(output/'bank/acceptance.json',dict(passed=True,geometry_tests=geometry_tests(),cases=accepted))
    put(output/'bank/rejected.json',rejected)
    put(output/'private_evaluation_seed_plan.json',dict(entries=seeds))
    put(output/'scenario_independence.json',dict(historical_files=history,historical_lengths=sorted(lengths),
        unique_new_wall_lengths=sorted(used),split_by='unique non-rigid base geometry before outcomes',
        scope='new wall lengths within three prescribed topologies; not arbitrary-map generalization'))
    tasklist=make_tasks(output)
    put(output/'teacher_check/requests.json',tasklist)
    artifacts={str(p.relative_to(output)):sha(p) for p in output.rglob('*.json')}
    protocol=dict(version=1,seed=SEED,development_groups=120,test_groups=60,train_groups=96,validation_groups=24,
        training_seeds=[0,1,2],teacher_check_requests=288,teacher_simulations=10000,student_simulations=1000,
        deadline_ms=50,opponent_simulations=3000,depth=100,particles=64,teacher_privileged_state=False,
        max_supervision_queries=1920,supervision_simulation_cap=19200000,
        teacher_check_budget='separate closed-loop prerequisite; not charged as training label queries',
        teacher_gate='(E>=T+3 and E>=G) or (E>=max(T,G) and E_cost<=0.9*min(T_cost,G_cost)); E_fail<=T_fail',
        no_training_until_teacher_passes=True,deployment_changed=False,
        sources=sources(),artifacts=artifacts)
    put(output/'experiment.json',protocol)
    put(output/'status.json',dict(stage='prepared',completed=0,total=288,pid=None))


def make_tasks(output):
    entries=json.loads((output/'bank/manifest.json').read_text())['entries']
    seeds=json.loads((output/'private_evaluation_seed_plan.json').read_text())['entries']
    result=[]
    for i,e in enumerate(e for e in entries if e['teacher_check']):
        for wi,w in enumerate(('A','B')):
            for s in range(2):
                order=('G','T','E');r=(i+wi+s)%3;order=order[r:]+order[:r]
                for m in order:
                    result.append(dict(key=f'{e["id"]}_{w}_s{s}_{m}',entry=e,world=w,stream=s,method=m,
                        opponent_seed=seeds[e['id']]['opponent'][s],environment_seed=seed(3,i,wi,s),folder='bank'))
    return result


def setup(task,output,initialize=True):
    configure(output)
    packet,environment=base.load_inputs(task['entry'])
    world=next(w for w in environment['worlds'] if w['id']==task['world'])
    p=base.original_model(packet['map'],packet['motion']['pursuer_axis_limit'])
    state=np.array(world['handoff_full_state'],float).reshape(-1,1)
    start=time.perf_counter()
    public=base.PublicMap(packet['map'],packet['motion']['pursuer_axis_limit'])
    own=base.Controller(public,'T' if task['method']=='E' else task['method'],packet['controller_seeds'][task['stream']],
        simulations=10000 if task['method']=='E' else 1000,deadline_ms=0.)
    own.initial_history=packet['history'][:-1];own.initial_cursor=0
    cost=dict(map_prepare_ms=1000*(time.perf_counter()-start))
    other=base.RootPUCT([None]*4,None,3000,0.,0.)
    other.seeded_state(p,task['opponent_seed'],task['environment_seed'])
    if initialize:assimilate(own)
    return p,state,own,other,cost


def assimilate(own):
    while own.initial_cursor<len(own.initial_history):
        own.update(base.obs(own.initial_history[own.initial_cursor]),np.zeros(2,np.int64))
        own.initial_cursor+=1


def snapshot(state,own,other,p,trace):
    saved=base.snapshot(state,own,other,p,trace);saved['global_rng']=rng_globals();return saved


def read_cached(task,output):
    directory=output/'teacher_check'
    row=base.cached(task,directory)
    if row is not None and row['task_fingerprint']!=digest(task):raise RuntimeError('Cached task mismatch')
    return row


def episode(task,output):
    directory=output/'teacher_check';pending=directory/'pending'/(task['key']+'.pt')
    # Resume never consumes earlier observations a second time.
    p,state,own,other,cost=setup(task,output,initialize=False);trace=[];failure=None
    if pending.exists():
        saved=load(pending);state=saved['state'];own.restore(saved['controller']);other.set_rng_state(p,saved['opponent']);trace=saved['trace']
    started=time.perf_counter()
    while not p.is_terminal(state):
        before=snapshot(state,own,other,p,[])
        try:
            assimilate(own)
            nxt,record=base.step(p,state,own,other,fixed=own.simulations)
        except Exception as error:
            if not recognized(error):raise
            after=snapshot(state,own,other,p,[])
            if before['opponent']!=after['opponent']:raise RuntimeError('Opponent advanced at own failure')
            failure=dict(exception=str(error),pre=before,post=after,traceback=traceback.format_exc())
            break
        state=nxt;trace.append(record);save(pending,snapshot(state,own,other,p,trace))
    row=base.metrics(task,state,trace,cost)
    row['method_execution_failure']=int(failure is not None)
    if failure:row.update(capture=0,failure='estimator_recovery_failure',completion_cost=80-row['handoff_step'])
    path=directory/'episodes'/(task['key']+'.pt')
    save(path,dict(task=task,trajectory=trace,failure=failure,final=snapshot(state,own,other,p,[])))
    row.update(episode_path=str(path),episode_sha256=sha(path),task_fingerprint=digest(task),wall_seconds=time.perf_counter()-started)
    put(directory/'cache'/(task['key']+'.json'),row)
    pending.unlink(missing_ok=True)
    return row


def worker(task,output):
    configure(output)
    return read_cached(task,output) or episode(task,output)


def preflight(output):
    check_frozen(output);configure(output)
    base.warmup()
    tasks=json.loads((output/'teacher_check/requests.json').read_text())
    checks=[]
    # Frozen G/T parity over real actions, posterior and both random streams.
    for method in ('G','T'):
        task=next(t for t in tasks if t['method']==method)
        p,state,a,other,_=setup(task,output)
        p2,state2,b,other2,_=base.setup(task)
        for _ in range(2):
            nxt,r=base.step(p,state,a,other,fixed=17)
            nxt2,r2=base.step(p2,state2,b,other2,fixed=17)
            for field in ('action','next_state','particles','filter_rng','search_rng'):
                assert np.array_equal(r[field],r2[field]),field
            assert r['opponent_rng']==r2['opponent_rng']
            state,state2=nxt,nxt2
        checks.append(dict(method=method,steps=2,fixed_simulations=17,passed=True))
    # Equal public input implies equal controllers, regardless of private world.
    for task in [t for t in tasks if t['method']=='T' and t['world']=='A']:
        p,state,a,other,_=setup(task,output)
        _,_,b,_,_=setup(dict(task,world='B'),output)
        assert fingerprint_controller(a.snapshot())==fingerprint_controller(b.snapshot())
    result=dict(passed=True,frozen_G_T_parity=checks,public_input_invariance_pairs=48,
        known_recovery_failures='retained with complete pre/post snapshots; unknown exceptions propagate',
        neural_checks='deferred until teacher passes; no networks loaded during teacher prerequisite')
    put(output/'preflight.json',result)
    return result


def teacher_check(output,workers):
    prepare(output)
    preflight(output)
    tasks=json.loads((output/'teacher_check/requests.json').read_text())
    rows=[];todo=[]
    for t in tasks:
        row=read_cached(t,output)
        if row is None:todo.append(t)
        else:rows.append(row)
    started=time.perf_counter();initial=len(rows)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(worker,t,output):t for t in todo}
        for future in as_completed(futures):
            try:row=future.result()
            except Exception as error:
                put(output/'status.json',dict(stage='paused_unknown_exception',error=repr(error),task=futures[future],pid=None))
                for f in futures:f.cancel()
                raise
            rows.append(row)
            status=dict(stage='teacher_check',completed=len(rows),total=288,workers=workers,pid=os.getpid(),
                elapsed_seconds=time.perf_counter()-started,committed_per_hour=(len(rows)-initial)*3600/max(.001,time.perf_counter()-started))
            put(output/'status.json',status);print(json.dumps(status),flush=True)
    rows=[read_cached(t,output) for t in tasks]
    put(output/'teacher_check/results.json',rows)
    result=gate(rows);put(output/'teacher_check/gate.json',result)
    report(output)
    put(output/'status.json',dict(stage='teacher_passed' if result['passed'] else 'stopped_teacher_gate',completed=288,total=288,pid=None,
        training_started=False,gate=result))
    return result


def gate(rows):
    if len(rows)!=288 or len({r['key'] for r in rows})!=288:raise RuntimeError('Incomplete teacher pairing')
    summary={}
    for m in ('G','T','E'):
        selected=[r for r in rows if r['method']==m]
        if len(selected)!=96:raise RuntimeError('Missing method rows')
        summary[m]=dict(captures=sum(r['capture'] for r in selected),cost=float(np.mean([r['completion_cost'] for r in selected])),
            execution_failures=sum(r['method_execution_failure'] for r in selected))
    g,t,e=(summary[m] for m in ('G','T','E'))
    capture=e['captures']>=t['captures']+3 and e['captures']>=g['captures']
    cost=e['captures']>=max(g['captures'],t['captures']) and e['cost']<=.9*min(g['cost'],t['cost'])
    reliable=e['execution_failures']<=t['execution_failures']
    return dict(passed=bool((capture or cost) and reliable),capture_gate=bool(capture),cost_gate=bool(cost),reliability_gate=bool(reliable),summary=summary)


def verify_task(task,output):
    configure(output);row=read_cached(task,output);record=load(row['episode_path'])
    p,state,own,other,_=setup(task,output,initialize=False)
    for original in record['trajectory']:
        assimilate(own)
        nxt,actual=base.step(p,state,own,other,fixed=original['own']['simulations'])
        for name in ('action','next_state','particles','filter_rng','search_rng'):
            if not np.array_equal(actual[name],original[name]):raise RuntimeError('Replay mismatch: '+task['key']+' '+name)
        if actual['opponent_rng']!=original['opponent_rng']:raise RuntimeError('Opponent RNG mismatch')
        state=nxt
    reproduced=False
    if record['failure']:
        try:
            assimilate(own)
            base.step(p,state,own,other,fixed=own.simulations)
        except Exception as error:
            if not recognized(error):raise
            reproduced=True
        if not reproduced:raise RuntimeError('Missing recorded failure')
        if fingerprint_controller(own.snapshot())!=fingerprint_controller(record['failure']['post']['controller']):raise RuntimeError('Post-failure state differs')
    return dict(key=task['key'],passed=True,steps=len(record['trajectory']),failure_reproduced=reproduced)


def replay(output,workers):
    check_frozen(output)
    tasks=json.loads((output/'teacher_check/requests.json').read_text())
    checks=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for r in pool.map(verify_task,tasks,[output]*len(tasks)):
            checks.append(r)
            put(output/'status.json',dict(stage='replay',completed=len(checks),total=len(tasks),pid=os.getpid()))
            print(json.dumps(dict(stage='replay',completed=len(checks),total=len(tasks))),flush=True)
    put(output/'teacher_check/replay.json',dict(passed=True,checks=checks))
    report(output)
    g=json.loads((output/'teacher_check/gate.json').read_text())
    put(output/'status.json',dict(stage='teacher_passed' if g['passed'] else 'stopped_teacher_gate',completed=288,total=288,
        replay_passed=True,pid=None,training_started=False))


def report(output):
    path=output/'teacher_check/results.json'
    if not path.exists():return
    rows=json.loads(path.read_text());g=gate(rows)
    idx={(r['history_id'],r['world'],r['stream'],r['method']):r for r in rows}
    histories=sorted({r['history_id'] for r in rows});rng=np.random.default_rng(SEED)
    draws=rng.integers(0,len(histories),(10000,len(histories)))
    stats={}
    for a,b in (('E','T'),('E','G'),('T','G')):
        stats[a+'-'+b]={}
        for field in ('capture','completion_cost','method_execution_failure'):
            delta=np.array([np.mean([idx[h,w,s,a][field]-idx[h,w,s,b][field] for w in ('A','B') for s in (0,1)]) for h in histories])
            stats[a+'-'+b][field]=dict(delta=float(delta.mean()),ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist())
    put(output/'teacher_check/statistics.json',dict(unit='24 public histories; both worlds and streams retained',seed=SEED,draws=10000,comparisons=stats))
    text='# 观测记忆增强 NTE：教师前置检查\n\n'
    text+='教师检查已完成 288 个请求。'+('教师达到预登记门槛。' if g['passed'] else '**教师未达到预登记门槛，按计划停止，不启动网络训练和正式测试。**')+'\n\n'
    text+='本阶段 G/T/E 均未使用神经网络；E 是同一冻结局部搜索运行 10,000 次模拟。不能据此判断神经网络或观测记忆无效。\n\n'
    text+='|方法|捕获数/96|捕获率|完成成本|估计器执行失败|\n|---|---:|---:|---:|---:|\n'
    for m,v in g['summary'].items():text+=f'|{m}|{v["captures"]}/96|{v["captures"]/96:.2%}|{v["cost"]:.3f}|{v["execution_failures"]}|\n'
    text+='\n|比较|捕获差百分点 [95% CI]|完成成本差 [95% CI]|\n|---|---|---|\n'
    for name,v in stats.items():
        a,b=v['capture'],v['completion_cost']
        text+=f'|{name}|{100*a["delta"]:+.2f} [{100*a["ci95"][0]:+.2f}, {100*a["ci95"][1]:+.2f}]|{b["delta"]:+.2f} [{b["ci95"][0]:+.2f}, {b["ci95"][1]:+.2f}]|\n'
    text+='\n捕获门槛：E 比 T 至少多捕获 3 局且不低于 G；或捕获不低于二者且完成成本比二者最小值低至少 10%；同时 E 执行失败不得多于 T。失败成本为原接管时剩余步数。\n'
    text+='\n按 24 组公共历史配对 bootstrap，保留组内两个隐藏初态、两个搜索流和所有方法，10,000 次，种子 62020261025。这里是开发教师检查，不是独立泛化确认。\n'
    text+='\n120 组开发历史及 60 组测试历史已按几何条件冻结。测试布局具有不同墙长，但仍属于三类既定拓扑。教师检查与训练标签预算分开计账；训练标签尚未产生。\n'
    text+='\n冻结的估计器、模拟对手和奖励保持不变，教师没有访问真实隐藏位置。高预算教师未改善时，只能说明当前搜索监督源尚不足以支撑本轮学习，不能单独定位估计、模拟对手或回报的因果责任。\n'
    replay_path=output/'teacher_check/replay.json'
    text+='\n重放：'+('全部请求通过。' if replay_path.exists() and json.loads(replay_path.read_text())['passed'] else '尚未完成。')+'\n'
    (output/'REPORT.md').write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','teacher_check','collect','train','evaluate','report','replay','pipeline'))
    parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--workers',type=int,default=80)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='prepare':prepare(out)
        elif args.command=='teacher_check':teacher_check(out,args.workers)
        elif args.command=='replay':replay(out,args.workers)
        elif args.command=='report':report(out)
        elif args.command=='pipeline':
            result=teacher_check(out,args.workers)
            replay(out,args.workers)
            if result['passed']:
                from memory_nte_learning import pipeline
                pipeline(out,args.workers)
        else:
            check_frozen(out)
            path=out/'teacher_check/gate.json'
            if not path.exists() or not json.loads(path.read_text())['passed']:
                raise RuntimeError('Teacher gate has not passed; downstream stages are prohibited')
            from memory_nte_learning import dispatch
            dispatch(args.command,out,args.workers)


if __name__=='__main__':main()
