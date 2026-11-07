"""Registered occlusion-branch neural PUCT experiment, with staged stopping."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import copy
import fcntl
import hashlib
import json
import multiprocessing
import platform
import random
import pickle
import struct
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
import run_l45_occluded_comparison as base
from continue_l45_occluded_comparison import rng_globals
from solvers.branch_support import GridMap,Support,SupportFailure,cell_id
from solvers.branch_nte import Controller,Search,expert_search
from solvers.local_pursuit import safe,Observation

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'current/l45_memory_nte_20261025'
DEFAULT=ROOT/'current/l45_branch_nte_20261025'
SEED=63020261025
FAMILIES=('long_wall','wall_corner','multi_route')
METHODS=('G','T','M','H','H-U','H-D')
put,sha,save,load=base.put,base.sha,base.save_torch,base.load_torch


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def seed(*keys):return int(np.random.SeedSequence([SEED,*keys]).generate_state(1)[0])


def entries():return json.loads((SOURCE/'bank/manifest.json').read_text())['entries']


def inputs(entry):return (json.loads((SOURCE/'bank'/entry['public']).read_text()),json.loads((SOURCE/'bank'/entry['environment']).read_text()))


def source_hashes():
    files=[Path(__file__),ROOT/'code/solvers/branch_support.py',ROOT/'code/solvers/branch_nte.py',
        ROOT/'code/learning/branch_nte.py',ROOT/'code/branch_nte_experiment.py',ROOT/'code/tests/test_l45_branch_nte.py',
        ROOT/'code/solvers/local_pursuit.py',ROOT/'code/learning/memory_nte.py',Path(base.__file__),
        ROOT/'code/solvers/root_audit_puct.py',ROOT/'code/solvers/c_puct.py',ROOT/'code/problems/example8.py',
        ROOT/'code/prepare_l45_occlusion_scenarios.py',ROOT/'code/verify_l45_occlusion_scenarios.py',Path(base.cpp.__file__)]
    return {str(p):sha(p) for p in files}


def check_frozen(output):
    lock=json.loads((output/'experiment.json').read_text())
    if lock['sources']!=source_hashes():raise RuntimeError('Frozen implementation fingerprint changed')
    for p,h in lock['artifacts'].items():
        if sha(p)!=h:raise RuntimeError('Frozen input changed: '+p)
    return lock


def prepare(output):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'experiment.json').exists():return check_frozen(output)
    invalid=ROOT/'current/l45_branch_nte_20261025/INVALIDATION.json'
    carry=json.loads(invalid.read_text())['teacher_queries_spent'] if invalid.exists() else 0
    original=json.loads((SOURCE/'experiment.json').read_text())
    for relative,h in original['artifacts'].items():
        if sha(SOURCE/relative)!=h:raise RuntimeError('Previous frozen scene changed')
    all_entries=entries();testids={e['id'] for e in all_entries if e['split']=='test'}
    contaminated=[]
    # Executed caches, rather than unused scenario manifests, indicate exposure.
    for parent in (ROOT/'current').iterdir():
        if not parent.is_dir() or parent.resolve()==output.resolve():continue
        for directory in parent.glob('**/cache'):
            for p in directory.glob('*.json'):
                if any(p.name.startswith(i+'_') for i in testids):contaminated.append(str(p))
    if contaminated:raise RuntimeError('Test histories already evaluated: '+str(contaminated[:4]))
    selected=[e for e in all_entries if e['teacher_check']]
    if len(selected)!=24 or any(e['split']!='train' for e in selected):raise RuntimeError('Bad teacher subset')
    put(output/'scenario_index.json',dict(source=str(SOURCE),entries=all_entries,teacher_histories=[e['id'] for e in selected],
        split_counts={s:sum(e['split']==s for e in all_entries) for s in ('train','validation','test')},
        test_executed_cache_matches=contaminated,test_information='geometry acceptance only; no control outcomes or labels'))
    artifacts={str(SOURCE/'bank/manifest.json'):sha(SOURCE/'bank/manifest.json'),str(SOURCE/'private_evaluation_seed_plan.json'):sha(SOURCE/'private_evaluation_seed_plan.json')}
    for e in all_entries:
        for k in ('public','environment'):artifacts[str(SOURCE/'bank'/e[k])]=sha(SOURCE/'bank'/e[k])
    if carry:artifacts[str(invalid)]=sha(invalid)
    protocol=dict(version=2,seed=SEED,methods=METHODS,grid_resolution=.25,branches=4,candidate_slots=16,
        neural_slots=8,uniform_slots=8,puct=1.5,depth=100,simulation_cap=1000,deadline_ms=50,p95_limit_ms=55,
        teacher_simulations=10000,teacher_queries=48,teacher_suffix_requests=192,teacher_continuation_simulations=128,
        total_query_cap=1536,total_teacher_simulation_cap=15360000,train_query_cap=1280-carry,validation_query_cap=256,
        inherited_invalid_queries=carry,active_query_cap=1536-carry,uniform_prior='uniform over deduplicated candidate nodes',
        collection_requests=480,screen_requests=576,screen_fixed_requests=96,test_requests=3360,
        seeds=[0,1,2],epochs=30,lr=.0003,batch=128,clip=.5,opponent_simulations=3000,
        privileged_expert=True,privileged_deployment=False,source=str(SOURCE),sources=source_hashes(),artifacts=artifacts,
        deployment_changed=False,scope='one pursuer/evader, known map, three established topologies; independent geometry instances',
        approximations=['conservative .25m outer support, uniform free-area sampling, not calibrated belief',
            'action/quantized-observation tree; local policy only receives sensor observations',
            'frozen local rollout retains its auxiliary 64-particle filter and simulated-history cutoff',
            'simulated evasive50/uniform50 opponent differs from real Search Only3000'],
        runtime=dict(python=sys.version,platform=platform.platform(),executable=sys.executable))
    put(output/'experiment.json',protocol)
    put(output/'status.json',dict(stage='prepared',pid=None,training_started=False,test_started=False))
    return protocol


_MAPS={}
def public_map(entry):
    packet,_=inputs(entry);key=digest(dict(map=packet['map'],speed=packet['motion']['pursuer_axis_limit']))
    if key not in _MAPS:_MAPS[key]=GridMap(packet['map'],packet['motion']['pursuer_axis_limit'])
    return _MAPS[key]


def make_task(entry,world,stream,method,stage,budget=128,training_seed=None):
    plan=json.loads((SOURCE/'private_evaluation_seed_plan.json').read_text())['entries'][entry['id']]
    uid=method if training_seed is None else f'{method}_seed{training_seed}'
    return dict(key=f'{entry["id"]}_{world}_s{stream}_{uid}',entry=entry,world=world,stream=stream,method=method,uid=uid,
        stage=stage,budget=budget,training_seed=training_seed,controller_seed=plan['controller'][stream],
        opponent_seed=plan['opponent'][stream],environment_seed=seed(1,plan['opponent'][stream],ord(world)))


def setup(task,output,initialize=True):
    random.seed(task['environment_seed']);np.random.seed(task['environment_seed']);torch.manual_seed(task['environment_seed'])
    packet,environment=inputs(task['entry']);world=next(w for w in environment['worlds'] if w['id']==task['world'])
    p=base.original_model(packet['map'],packet['motion']['pursuer_axis_limit'])
    state=np.array(world['handoff_full_state'],float).reshape(-1,1)
    started=time.perf_counter();map_key=digest(dict(map=packet['map'],speed=packet['motion']['pursuer_axis_limit']))
    cache_hit=map_key in _MAPS;public=public_map(task['entry']);policy=None
    if task['method'] in ('M','H','H-U','H-D'):
        from learning.branch_nte import Policy,Runtime
        path=output/'training'/f'{"M" if task["method"]=="M" else "H"}_seed{task["training_seed"]}'/'best.pt'
        if sha(path)!=task['model_sha256']:raise RuntimeError('Model changed')
        checkpoint=load(path);model=Policy();model.load_state_dict(checkpoint['model']);model.eval()
        policy=Runtime(model,public,'M' if task['method']=='M' else 'H')
    own=Controller(public,'T' if task['method'] in ('E-first','T-first') else task['method'],task['controller_seed'],policy=policy,
        simulations=1000,deadline_ms=50. if task['budget']=='time' else 0.)
    own.initial_history=packet['history'][:-1];own.initial_cursor=0
    other=base.RootPUCT([None]*4,None,3000,0.,0.)
    other.seeded_state(p,task['opponent_seed'],task['environment_seed'])
    cost=dict(map_prepare_ms=(time.perf_counter()-started)*1000,map_cache_hit=cache_hit)
    if initialize:assimilate(own)
    return p,state,own,other,cost


def assimilate(own):
    while own.initial_cursor<len(own.initial_history):
        own.update(base.obs(own.initial_history[own.initial_cursor]));own.initial_cursor+=1


def snapshot(state,own,other,p,trace):
    return dict(state=state.copy(),controller=own.snapshot(),opponent=other.get_rng_state(p),trace=trace,global_rng=rng_globals())


def step(p,state,own,other,fixed=None,forced=None):
    observation=own.public.observe(int(round(state[8,0])),state[4:6,0],state[:2,0])
    action=np.zeros((8,1))
    if forced is None:action[4:6,0]=own.act(observation,fixed_simulations=fixed)
    else:
        started=time.perf_counter();own.update(observation);action[4:6,0]=forced
        own.particles[:]=own.support.representative()
        own.last_stats=dict(simulations=0,latency_ms=(time.perf_counter()-started)*1000,geometry_queries=0,transitions=0,network_calls=0,
            source='frozen_first_action',root=[],recoveries=0,support_cells=int(own.support.mask.sum()),invalid_candidates=0)
    stamp=time.perf_counter();result=other.search(p,state.copy(),turn=0)
    action[:2,0]=np.asarray(result.best_action).reshape(-1)[:2]
    other_ms=(time.perf_counter()-stamp)*1000
    native=other._problem_wrapper_for(p);nxt=np.asarray(native.step(state,action,1.),float).reshape(-1,1)
    count=np.zeros(2,np.int64)
    return nxt,dict(state=state.copy(),action=action.copy(),next_state=nxt.copy(),observation=observation,
        particles=own.particles.copy(),search_rng=own.rng.copy(),filter_rng=own.filter_rng.copy(),own=copy.deepcopy(own.last_stats),
        support=own.support.mask.copy(),support_exact=own.support.exact.copy(),support_covered=own.support.covers(state[:2,0]),
        opponent_rng=other.get_rng_state(p),opponent_ms=other_ms,
        own_illegal=not safe(state[4:6,0],action[4:6,0],own.public.speed,own.public.boxes,count),
        opponent_illegal=not safe(state[:2,0],action[:2,0],1.,own.public.boxes,count),
        nearest_particle_error=float(np.min(np.linalg.norm(own.particles-state[:2,0],axis=1))))


def cached(task,output):
    path=output/task['stage']/'cache'/(task['key']+'.json')
    if not path.exists():return None
    row=json.loads(path.read_text())
    if row['task_hash']!=digest(task) or sha(row['episode_path'])!=row['episode_sha256']:raise RuntimeError('Damaged committed cache')
    return row


def play(task,output):
    row=cached(task,output)
    if row is not None:return row
    p,state,own,other,cost=setup(task,output,initialize=False);trace=[];failure=None
    folder=output/task['stage'];pending=folder/'pending'/(task['key']+'.pt')
    if pending.exists():
        saved=load(pending)
        if saved['task_hash']!=digest(task):raise RuntimeError('Pending task changed')
        state=saved['state'];own.restore(saved['controller']);other.set_rng_state(p,saved['opponent']);trace=saved['trace']
        globals_=saved['global_rng'];random.setstate(globals_['python']);np.random.set_state(globals_['numpy']);torch.set_rng_state(globals_['torch_cpu'])
    started=time.perf_counter()
    while not p.is_terminal(state):
        before=snapshot(state,own,other,p,[])
        try:
            assimilate(own)
            forced=np.asarray(task['first_action']) if not trace and 'first_action' in task else None
            nxt,record=step(p,state,own,other,fixed=None if task['budget']=='time' else task['budget'],forced=forced)
        except SupportFailure as error:
            failure=dict(exception=str(error),pre=before,post=snapshot(state,own,other,p,[]),traceback=traceback.format_exc());break
        state=nxt;trace.append(record)
        saved=snapshot(state,own,other,p,trace);saved['task_hash']=digest(task);save(pending,saved)
    base.SOURCE=SOURCE
    row=base.metrics(task,state,trace,cost)
    hidden=[r for r in trace if r['observation'].target is None]
    row.update(method_execution_failure=int(failure is not None),task_hash=digest(task),training_seed=task.get('training_seed'),uid=task.get('uid',task['method']),budget=task['budget'],
        network_calls=sum(r['own'].get('network_calls',0) for r in trace),support_covered=sum(r['support_covered'] for r in hidden),hidden_frames=len(hidden),
        invalid_candidates=sum(r['own']['invalid_candidates'] for r in trace),wall_seconds=time.perf_counter()-started)
    if failure:row.update(capture=0,failure='support_failure',completion_cost=80-row['handoff_step'])
    elif not row['capture'] and not row['own_illegal'] and not row['opponent_illegal']:
        row['failure']='never_reacquired' if row['first_reacquisition_after_handoff'] is None else 'reacquired_not_captured'
    row['reacquisition_cost']=row['first_reacquisition_after_handoff'] if row['first_reacquisition_after_handoff'] is not None else 80-row['handoff_step']
    path=folder/'episodes'/(task['key']+'.pt')
    save(path,dict(task=task,trajectory=trace,failure=failure,final=snapshot(state,own,other,p,[])))
    row.update(episode_path=str(path),episode_sha256=sha(path));put(folder/'cache'/(task['key']+'.json'),row);pending.unlink(missing_ok=True)
    return row


def execute(tasks,output,workers=8,serial=False):
    if not tasks:return []
    stage=tasks[0]['stage'];path=output/stage/'requests.json'
    if path.exists() and json.loads(path.read_text())!=tasks:raise RuntimeError('Request list changed')
    put(path,tasks);rows=[];started=time.perf_counter()
    def accept(row):
        rows.append(row);status=dict(stage=stage,completed=len(rows),total=len(tasks),pid=os.getpid(),serial=serial,
            elapsed_seconds=time.perf_counter()-started,failures=sum(r['method_execution_failure'] for r in rows))
        put(output/'status.json',status);print(json.dumps(status),flush=True)
    if serial:
        for task in tasks:accept(play(task,output))
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs=[pool.submit(play,t,output) for t in tasks]
            for f in as_completed(jobs):accept(f.result())
    rows=sorted(rows,key=lambda r:r['key']);put(output/stage/'results.json',rows);return rows


def teacher_query(context,output):
    key=context['key'];path=output/'labels'/(key+'.pt')
    if path.exists():
        lock=json.loads((output/'labels'/(key+'.json')).read_text())
        if sha(path)!=lock['sha256'] or lock['context_hash']!=digest(context):raise RuntimeError('Teacher label changed')
        return load(path)
    reservation=output/'query_reservations'/(key+'.json')
    if not reservation.exists():
        existing=list((output/'query_reservations').glob('*.json'))
        cap=json.loads((output/'experiment.json').read_text())['active_query_cap']
        if len(existing)>=cap:raise RuntimeError('Teacher query cap reached')
        put(reservation,dict(key=key,context_hash=digest(context),budget=10000,privileged=True,seed=int(key[:8],16)))
    elif json.loads(reservation.read_text())['context_hash']!=digest(context):raise RuntimeError('Teacher context changed')
    public=public_map(context['entry']);pending=output/'query_pending'/(key+'.pt');journal=output/'query_pending'/(key+'.journal')
    if pending.exists():search=load(pending)['search']
    else:
        own=Controller(public,'T',int(key[:8],16),deadline_ms=0.)
        for r in context['history']:own.update(base.obs(r))
        observation=base.obs(context['history'][-1]);search=Search(own,observation,np.asarray(context['target'],float));own.method='E'
        save(pending,dict(search=search,committed_simulations=0))
    # Compact, checksummed per-simulation deltas avoid replaying committed
    # simulations or rewriting the complete tree after every simulation.
    if journal.exists():
        valid_end=0
        with journal.open('rb') as handle:
            while True:
                header=handle.read(8)
                if not header:break
                if len(header)!=8:break
                size=struct.unpack('<Q',header)[0];checksum=handle.read(32);payload=handle.read(size)
                if len(payload)!=size:break
                if hashlib.sha256(payload).digest()!=checksum:raise RuntimeError('Teacher simulation journal corruption')
                delta=pickle.loads(payload)
                if delta['completed']>search.completed:search.apply_delta(delta)
                valid_end=handle.tell()
        if journal.stat().st_size!=valid_end:
            put(output/'query_pending'/(key+'.recovery.json'),dict(incomplete_tail_bytes=journal.stat().st_size-valid_end,committed_simulations=search.completed,max_uncommitted_replay=1))
            with journal.open('r+b') as handle:handle.truncate(valid_end)
    with journal.open('ab') as handle:
        def progress(s):
            payload=pickle.dumps(s.latest_delta,protocol=5)
            handle.write(struct.pack('<Q',len(payload))+hashlib.sha256(payload).digest()+payload);handle.flush();os.fsync(handle.fileno())
            if s.completed%1000==0:
                put(output/'query_progress.json',dict(key=key,committed_simulations=s.completed,pid=os.getpid()))
        action,roots=search.run(10000,progress=progress,start=search.completed)
    record=dict(context=context,action=action,root=roots,simulations=search.completed,transitions=int(search.count[1]),
        rng=search.c.rng.copy(),privileged=True,teacher_input='full training state at root; full simulated sensor',
        support_snapshot=dict(mask=search.root.support.mask,exact=search.root.support.exact),failure=None)
    save(path,record);put(output/'labels'/(key+'.json'),dict(sha256=sha(path),context_hash=digest(context),simulations=search.completed))
    pending.unlink(missing_ok=True);journal.unlink(missing_ok=True);return record


def teacher_check(output,workers):
    check_frozen(output)
    if not json.loads((output/'audit.json').read_text())['passed']:raise RuntimeError('Audit must pass')
    selected=[e for e in entries() if e['teacher_check']];tasks=[]
    for i,e in enumerate(selected):
        packet,environment=inputs(e);public=public_map(e)
        # T's first action depends only on the common history, not hidden world.
        path=output/'teacher_check/first_actions'/(e['id']+'.pt')
        if path.exists():
            if sha(path)!=json.loads(path.with_suffix('.json').read_text())['sha256']:raise RuntimeError('Frozen first action changed')
            first=load(path)
        else:
            own=Controller(public,'T',packet['controller_seeds'][0],deadline_ms=0.)
            for r in packet['history'][:-1]:own.update(base.obs(r))
            action=own.act(base.obs(packet['history'][-1]),fixed_simulations=128)
            first=dict(action=action,root=own.last_stats['root'],rng=own.rng.copy());save(path,first)
            put(path.with_suffix('.json'),dict(sha256=sha(path),simulations=128,public_history_hash=digest(packet['history'])))
        for world in environment['worlds']:
            context=dict(entry=e,history=packet['history'],target=world['handoff_full_state'][:2],world=world['id'],split='train')
            context['key']=digest(context);label=teacher_query(context,output)
            for stream in range(2):
                order=('T-first','E-first') if (i+stream)%2==0 else ('E-first','T-first')
                for m in order:
                    task=make_task(e,world['id'],stream,m,'teacher_check',128)
                    task['first_action']=(first['action'] if m=='T-first' else label['action']).tolist();task['label_key']=context['key']
                    tasks.append(task)
        put(output/'status.json',dict(stage='teacher_queries',completed=(i+1)*2,total=48,pid=os.getpid()))
        print(json.dumps(dict(stage='teacher_queries',completed=(i+1)*2,total=48)),flush=True)
    rows=execute(tasks,output,workers);result=teacher_gate(rows)
    put(output/'teacher_check/gate.json',result);report(output)
    return result


def teacher_gate(rows):
    if len(rows)!=192 or len({r['key'] for r in rows})!=192:raise RuntimeError('Incomplete teacher pairs')
    summary={}
    for m in ('T-first','E-first'):
        rs=[r for r in rows if r['method']==m]
        if len(rs)!=96:raise RuntimeError('Missing teacher method')
        summary[m]=dict(captures=sum(r['capture'] for r in rs),cost=float(np.mean([r['completion_cost'] for r in rs])),failures=sum(r['method_execution_failure'] for r in rs))
    t,e=summary['T-first'],summary['E-first']
    capture=e['captures']>=t['captures']+2 and e['cost']<=t['cost']
    cost=e['captures']>=t['captures'] and e['cost']<=.95*t['cost']
    reliable=e['failures']<=t['failures']
    return dict(passed=bool((capture or cost) and reliable),capture_gate=bool(capture),cost_gate=bool(cost),reliability_gate=bool(reliable),summary=summary)


def verify_task(task,output):
    row=cached(task,output)
    if row is None:raise RuntimeError('Missing replay pair')
    saved=load(row['episode_path']);p,state,own,other,_=setup(task,output,initialize=False)
    for i,original in enumerate(saved['trajectory']):
        assimilate(own)
        forced=np.asarray(task['first_action']) if i==0 and 'first_action' in task else None
        nxt,actual=step(p,state,own,other,fixed=original['own']['simulations'],forced=forced)
        for name in ('action','next_state','support','support_exact','search_rng','particles'):
            if not np.array_equal(actual[name],original[name],equal_nan=True):raise RuntimeError('Replay mismatch '+task['key']+' '+name)
        if actual['opponent_rng']!=original['opponent_rng']:raise RuntimeError('Opponent RNG mismatch')
        state=nxt
    reproduced=False
    if saved['failure']:
        try:
            assimilate(own);step(p,state,own,other,fixed=task['budget'] if task['budget']!='time' else 0)
        except SupportFailure:reproduced=True
        if not reproduced:raise RuntimeError('Recorded support failure did not reproduce')
    return dict(key=task['key'],passed=True,steps=len(saved['trajectory']),failure_reproduced=reproduced)


def replay(output,workers,stage='teacher_check'):
    check_frozen(output);tasks=json.loads((output/stage/'requests.json').read_text());checks=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(verify_task,t,output) for t in tasks]
        for f in as_completed(jobs):
            checks.append(f.result())
            put(output/'status.json',dict(stage='replay_'+stage,completed=len(checks),total=len(tasks),pid=os.getpid()))
    result=dict(passed=True,requests=len(checks),checks=checks)
    put(output/stage/'replay.json',result);return result


def statistics(rows,methods,comparisons):
    histories=sorted({r['history_id'] for r in rows});rng=np.random.default_rng(SEED)
    draws=rng.integers(0,len(histories),(10000,len(histories)));result={}
    # Seed averages stay inside each public-history block.
    for a,b in comparisons:
        item={}
        for field in ('capture','completion_cost','method_execution_failure','own_illegal','reacquisition_cost'):
            delta=[]
            for h in histories:
                av=[r[field] for r in rows if r['history_id']==h and r['method']==a]
                bv=[r[field] for r in rows if r['history_id']==h and r['method']==b]
                if not av or not bv:raise RuntimeError('Missing paired history')
                delta.append(np.mean(av)-np.mean(bv))
            delta=np.asarray(delta);item[field]=dict(delta=float(delta.mean()),ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist())
        result[a+'-'+b]=item
    return dict(unit='public history; hidden worlds, streams and training seeds retained within block',histories=len(histories),draws=10000,seed=SEED,comparisons=result)


def report(output):
    parts=['# 遮挡分支条件神经树扩展\n','本轮为分阶段开发与独立场景验证；当前部署推荐不变。\n']
    for stage in ('teacher_check','screen','screen_fixed','evaluate'):
        path=output/stage/'results.json'
        if not path.exists():continue
        rows=json.loads(path.read_text());methods=sorted({r['method'] for r in rows})
        comparisons=[('E-first','T-first')] if stage=='teacher_check' else [('H',m) for m in methods if m!='H']
        stats=statistics(rows,methods,comparisons);put(output/stage/'statistics.json',stats)
        parts+=['\n## '+stage+'\n','|方法|捕获/请求|完成成本|执行失败|平均/P95决策ms|\n|---|---:|---:|---:|---|\n']
        for m in methods:
            rs=[r for r in rows if r['method']==m];lat=[x for r in rs for x in r['latencies_ms']]
            parts.append(f'|{m}|{sum(r["capture"] for r in rs)}/{len(rs)}|{np.mean([r["completion_cost"] for r in rs]):.3f}|{sum(r["method_execution_failure"] for r in rs)}|{np.mean(lat) if lat else 0:.2f}/{np.percentile(lat,95) if lat else 0:.2f}|\n')
        for name,v in stats['comparisons'].items():
            a,b=v['capture'],v['completion_cost'];parts.append(f'\n{name}：捕获差 {100*a["delta"]:+.2f} 点，95%区间 [{100*a["ci95"][0]:+.2f}, {100*a["ci95"][1]:+.2f}]；完成成本差 {b["delta"]:+.3f}，区间 {b["ci95"]}。\n')
        gate=output/stage/'gate.json'
        if gate.exists():parts.append('\n预登记门槛：'+('通过。' if json.loads(gate.read_text())['passed'] else '**未通过，停止后续阶段。**')+'\n')
        parts.append('\n此处教师检查只比较第一步动作，后续统一为128次模拟T；不是每步使用专家的部署结果。\n' if stage=='teacher_check' else '\n')
        parts.append('\n|地图|方法|捕获/请求|完成成本|主要失败原因|\n|---|---|---:|---:|---|\n')
        from collections import Counter
        for family in FAMILIES:
            for m in methods:
                rs=[r for r in rows if r['family']==family and r['method']==m]
                if not rs:continue
                failures=Counter(r['failure'] for r in rs if not r['capture'])
                parts.append(f'|{family}|{m}|{sum(r["capture"] for r in rs)}/{len(rs)}|{np.mean([r["completion_cost"] for r in rs]):.3f}|{dict(failures)}|\n')
        if stage=='teacher_check':
            idx={(r['history_id'],r['world'],r['stream'],r['method']):r for r in rows}
            rescues=losses=0
            for r in rows:
                if r['method']!='E-first':continue
                ref=idx[r['history_id'],r['world'],r['stream'],'T-first']
                rescues+=int(r['capture']>ref['capture']);losses+=int(r['capture']<ref['capture'])
            parts.append(f'\n配对挽回 {rescues} 条、丢失 {losses} 条；统计单位仍为24组公共历史，不能把96条续跑当作96个独立场景。第一步动作计算耗时未计入续跑决策延迟；专家不是50ms可部署对照。\n')
    parts.append('\n## 结论边界\n\n旧64粒子估计已被共同的保守网格支持替代；跨历史胜率不能用于隔离新网络贡献。网格是近似外支持，均匀面积不是校准信念。共同rollout仍使用冻结的辅助粒子与恢复截止，模拟逃逸策略与真实3000次搜索对手不同。\n')
    if not (output/'training').exists():parts.append('\n尚未训练M/H，不能判断分支条件网络或神经PUCT有效或无效。\n')
    (output/'REPORT.md').write_text(''.join(parts))


def finalize(output):
    check_frozen(output)
    reservations=list((output/'query_reservations').glob('*.json'));labels=list((output/'labels').glob('*.pt'))
    inherited=json.loads((output/'experiment.json').read_text())['inherited_invalid_queries']
    if len(reservations)+inherited>1536 or len(labels)>len(reservations):raise RuntimeError('Teacher budget inconsistency')
    stages={}
    for stage in ('teacher_check','collection','screen','screen_fixed','evaluate'):
        path=output/stage/'results.json'
        if not path.exists():continue
        rows=json.loads(path.read_text());tasks=json.loads((output/stage/'requests.json').read_text())
        if len(rows)!=len(tasks) or len({r['key'] for r in rows})!=len(rows):raise RuntimeError('Incomplete committed stage')
        for task in tasks:
            if cached(task,output) is None:raise RuntimeError('Missing committed request')
        replay_path=output/stage/'replay.json'
        stages[stage]=dict(requests=len(tasks),cache_integrity=True,replay_passed=replay_path.exists() and json.loads(replay_path.read_text())['passed'])
    training=list((output/'training').glob('*/best.pt'))
    budget=dict(teacher_queries=len(reservations)+inherited,completed_teacher_queries=len(labels),teacher_simulations=10000*(len(labels)+inherited),
        invalid_implementation_queries=inherited,usable_teacher_queries=len(reservations),
        teacher_simulation_cap=15360000,teacher_queries_cap=1536,uncommitted_recovery='at most one simulation per interruption; separately evidenced in query_pending recovery files',
        trained_models=len(training),test_requests=stages.get('evaluate',{}).get('requests',0))
    put(output/'budget.json',budget)
    result=dict(passed=True,sources_and_scenes_unchanged=True,stages=stages,**budget,deployment_changed=False)
    put(output/'final_checks.json',result)
    from branch_nte_experiment import figures
    figures(output)
    report(output)
    hashes={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name not in ('run.lock','DELIVERY_SHA256.json') and '.tmp' not in p.name}
    put(output/'DELIVERY_SHA256.json',dict(files=hashes,sources=source_hashes()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','audit','teacher_check','collect','train','screen','evaluate','report','replay','pipeline'))
    parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--stage',default='teacher_check');args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.command=='prepare':prepare(out)
            elif args.command=='audit':
                from branch_nte_experiment import audit as run_audit
                run_audit(out)
            elif args.command=='teacher_check':teacher_check(out,args.workers)
            elif args.command=='report':check_frozen(out);report(out)
            elif args.command=='replay':replay(out,args.workers,args.stage)
            elif args.command=='pipeline':
                prepare(out)
                from branch_nte_experiment import audit as run_audit,dispatch
                run_audit(out);gate=teacher_check(out,args.workers);replay(out,args.workers)
                if gate['passed']:dispatch('pipeline',out,args.workers)
                else:
                    put(out/'status.json',dict(stage='stopped_teacher_gate',terminal=True,pid=None,completed=192,total=192,training_started=False,test_started=False))
                    report(out)
                finalize(out)
            else:
                from branch_nte_experiment import dispatch
                dispatch(args.command,out,args.workers)
        except Exception as error:
            put(out/'status.json',dict(stage='paused_unknown_exception',pid=None,error=repr(error),traceback=traceback.format_exc()))
            raise


if __name__=='__main__':main()
