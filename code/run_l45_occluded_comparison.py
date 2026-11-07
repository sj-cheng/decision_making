"""Frozen hard-scene comparison; 24 public histories are paired units.

Loads existing scenarios only. Never regenerates scenarios or changes models.
"""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_k]='1'
import argparse
import collections
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from solvers.local_pursuit import Controller,PublicMap,Observation
from solvers.root_audit_puct import RootPUCT,cpp
from prepare_l45_occlusion_scenarios import original_model
from solvers.local_pursuit import safe,warmup

torch.set_num_threads(1)

ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'current/l45_occlusion_scenarios_20261025'
OUTPUT=ROOT/'current/l45_occluded_comparison_20261025'
METHODS=('L','G','T','V')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(temp,path)


def obs(record):
    return Observation(record['step'],np.array(record['position'],float),None if record['target'] is None else np.array(record['target'],float),record['remaining'])


def public_controller(packet,method,stream):
    started=time.perf_counter()
    public=PublicMap(packet['map'],packet['motion']['pursuer_axis_limit'])
    built=time.perf_counter()
    controller=Controller(public,method,packet['controller_seeds'][stream],simulations=1000,deadline_ms=50.)
    count=np.zeros(2,np.int64)
    for measurement in packet['history'][:-1]:controller.update(obs(measurement),count)
    return controller,dict(map_prepare_ms=(built-started)*1000,history_assimilation_ms=(time.perf_counter()-built)*1000)


def preflight(output):
    output.mkdir(parents=True,exist_ok=True)
    for manifest_name in ('DELIVERY_SHA256.json','frozen.json'):
        manifest=json.loads((SOURCE/manifest_name).read_text())
        failed=[name for name,value in manifest['files'].items() if sha(SOURCE/name)!=value]
        if failed:raise RuntimeError('Frozen source artifact changed: '+str(failed))
    provenance=json.loads((SOURCE/'provenance.json').read_text())
    for path,value in provenance['original_files_unchanged'].items():
        if sha(path)!=value:raise RuntimeError('Original algorithm changed: '+path)
    entries=json.loads((SOURCE/'bank/manifest.json').read_text())['entries']
    seeds=json.loads((SOURCE/'private_evaluation_seed_plan.json').read_text())['entries']
    failures=[];checks=[]
    for entry in entries:
        packet=json.loads((SOURCE/'bank'/entry['public']).read_text())
        environment=json.loads((SOURCE/'bank'/entry['environment']).read_text())
        assert seeds[entry['id']]['controller']==packet['controller_seeds']
        for stream in range(2):
            objects={};item=dict(history_id=entry['id'],stream=stream,methods={},passed=True)
            for method in METHODS:
                try:
                    controller,cost=public_controller(packet,method,stream)
                    # A clone consumes the current observation solely to check
                    # full-history compatibility. Real execution will use the
                    # unmodified history[:-1] initializer once, then act once.
                    current=copy.deepcopy(controller)
                    current.update(obs(packet['history'][-1]),np.zeros(2,np.int64))
                    objects[method]=current
                    item['methods'][method]=dict(passed=True,**cost,recoveries=current.recoveries,
                        posterior_sha256=hashlib.sha256(current.particles.tobytes()+current.filter_rng.tobytes()).hexdigest())
                except Exception as error:
                    detail=dict(history_id=entry['id'],stream=stream,method=method,error=repr(error))
                    failures.append(detail);item['methods'][method]=dict(passed=False,error=repr(error));item['passed']=False
            if all(m in objects for m in ('G','T','V')):
                same=all(np.array_equal(objects['G'].particles,objects[m].particles) and np.array_equal(objects['G'].filter_rng,objects[m].filter_rng) for m in ('T','V'))
                item['common_estimates_identical']=same
                if not same:failures.append(dict(history_id=entry['id'],stream=stream,error='G/T/V posterior mismatch'));item['passed']=False
            for world in environment['worlds']:
                at=np.array(world['handoff_full_state'])
                if not np.array_equal(at[4:6],np.array(packet['history'][-1]['position'])):raise RuntimeError('Own pose mismatch')
                if packet['history'][-1]['target'] is not None:raise RuntimeError('Hidden handoff is exposed')
            checks.append(item)
    result=dict(passed=not failures,checked_public_histories=24,checked_history_stream_pairs=len(checks),
        initialization='update(history[:-1]); current observation processed once by act; clone used only in preflight',
        controller_initializers_receive_only_public_packets=True,algorithms_unchanged=True,
        requested_games=384,bootstrap_unit='public_history_group; retain both hidden worlds, all methods and both streams',
        bootstrap_draws=10000,bootstrap_seed=61020261025,
        statistics_override='Latest user instruction replaces prior six-template bootstrap; old frozen plan is preserved.',
        failures=failures,checks=checks,formal_evaluation_started=False)
    put(output/'preflight.json',result)
    put(output/'status.json',dict(stage='preflight_passed' if result['passed'] else 'paused_preflight_failure',
        completed=0,total=384,failures=len(failures),formal_evaluation_started=False))
    print(json.dumps(dict(passed=result['passed'],history_stream_pairs=len(checks),failures=len(failures)),ensure_ascii=False))
    return result


def load_inputs(entry,folder='bank'):
    return (json.loads((SOURCE/folder/entry['public']).read_text()),json.loads((SOURCE/folder/entry['environment']).read_text()))


def tasks():
    entries=json.loads((SOURCE/'bank/manifest.json').read_text())['entries']
    seeds=json.loads((SOURCE/'private_evaluation_seed_plan.json').read_text())['entries']
    result=[]
    for i,entry in enumerate(entries):
        for wi,world in enumerate(('A','B')):
            for stream in range(2):
                rotation=(i+wi+stream)%4
                for method in METHODS[rotation:]+METHODS[:rotation]:
                    opponent=seeds[entry['id']]['opponent'][stream]
                    result.append(dict(key=f'{entry["id"]}_{world}_s{stream}_{method}',entry=entry,world=world,
                        stream=stream,method=method,opponent_seed=opponent,
                        environment_seed=int(np.random.SeedSequence([61020261025,42,opponent]).generate_state(1)[0]),folder='bank'))
    assert len(result)==384
    return result


def save_torch(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    torch.save(value,temp);os.replace(temp,path)


def load_torch(path):return torch.load(path,weights_only=False,map_location='cpu')


def setup(task):
    packet,environment=load_inputs(task['entry'],task.get('folder','bank'))
    world=next(w for w in environment['worlds'] if w['id']==task['world'])
    p=original_model(packet['map'],packet['motion']['pursuer_axis_limit'])
    state=np.array(world['handoff_full_state'],float).reshape(-1,1)
    own,cost=public_controller(packet,task['method'],task['stream'])
    other=RootPUCT([None]*4,None,3000,0.,0.)
    other.seeded_state(p,task['opponent_seed'],task['environment_seed'])
    assert state[8,0]==packet['history'][-1]['step']
    return p,state,own,other,cost


def step(p,state,own,other,fixed=None):
    # The true state crosses only the sensor/environment boundary. Own act
    # receives Observation with target=None if the geometric sensor is empty.
    observation=own.public.observe(int(round(state[8,0])),state[4:6,0],state[:2,0])
    action=np.zeros((8,1));action[4:6,0]=own.act(observation,fixed_simulations=fixed)
    own_stats=copy.deepcopy(own.last_stats)
    stamp=time.perf_counter();result=other.search(p,state.copy(),turn=0)
    action[:2,0]=np.asarray(result.best_action).reshape(-1)[:2]
    other_ms=(time.perf_counter()-stamp)*1000
    native=other._problem_wrapper_for(p)
    nxt=np.asarray(native.step(state,action,1.),float).reshape(-1,1)
    own_illegal=not safe(state[4:6,0],action[4:6,0],own.public.speed,own.public.boxes,np.zeros(2,np.int64))
    other_illegal=not safe(state[:2,0],action[:2,0],1.,own.public.boxes,np.zeros(2,np.int64))
    return nxt,dict(state=state.copy(),action=action,next_state=nxt.copy(),observation=observation,
        particles=own.particles.copy(),filter_rng=own.filter_rng.copy(),search_rng=own.rng.copy(),own=own_stats,
        opponent_ms=other_ms,opponent_rng=other.get_rng_state(p),own_illegal=bool(own_illegal),opponent_illegal=bool(other_illegal),
        nearest_particle_error=float(np.min(np.linalg.norm(own.particles-state[:2,0],axis=1))))


def snapshot(state,own,other,p,trace):
    return dict(state=state.copy(),controller=own.snapshot(),opponent=other.get_rng_state(p),trace=trace)


def metrics(task,state,trace,init_cost):
    capture=state[9,0]<.5 and not any(t['own_illegal'] or t['opponent_illegal'] for t in trace)
    hidden=[t for t in trace if t['observation'].target is None]
    run=longest=0;first_visible=None;loss_events=0
    for i,t in enumerate(trace):
        if t['observation'].target is None:
            if run==0:loss_events+=1
            run+=1;longest=max(longest,run)
        else:
            if first_visible is None:first_visible=i
            run=0
    entry=task['entry'];_,env=load_inputs(entry,task.get('folder','bank'))
    lat=[t['own']['latency_ms'] for t in trace]
    return dict(key=task['key'],history_id=entry['id'],family=env['family'],required_blind_steps=env['required_blind_steps'],
        method=task['method'],world=task['world'],stream=task['stream'],capture=int(capture),
        own_illegal=int(any(t['own_illegal'] for t in trace)),opponent_illegal=int(any(t['opponent_illegal'] for t in trace)),
        failure='capture' if capture else 'own_illegal' if any(t['own_illegal'] for t in trace) else 'opponent_illegal' if any(t['opponent_illegal'] for t in trace) else 'timeout',
        suffix_steps=len(trace),completion_cost=len(trace) if capture else 80-env['handoff_step'],
        handoff_step=env['handoff_step'],final_step=float(state[8,0]),first_reacquisition_after_handoff=first_visible,
        longest_hidden=longest,hidden_steps=len(hidden),loss_events=loss_events,
        hidden_covered=sum(t['nearest_particle_error']<=1 for t in hidden) if task['method']!='L' else None,
        latencies_ms=lat,simulations=[t['own']['simulations'] for t in trace],
        transitions=sum(t['own']['transitions'] for t in trace),geometry_queries=sum(t['own']['geometry_queries'] for t in trace),
        sources=dict(collections.Counter(t['own']['source'] for t in trace)),initialization_cost=init_cost,
        trajectory_hash=hashlib.sha256(b''.join(t['state'].tobytes()+t['action'].tobytes()+t['next_state'].tobytes() for t in trace)).hexdigest())


def episode(task,output):
    p,state,own,other,cost=setup(task);trace=[]
    pending=output/'pending'/(task['key']+'.pt')
    if pending.exists():
        saved=load_torch(pending);state=saved['state'];own.restore(saved['controller']);other.set_rng_state(p,saved['opponent']);trace=saved['trace']
    started=time.perf_counter()
    while not p.is_terminal(state):
        state,record=step(p,state,own,other);trace.append(record)
        save_torch(pending,snapshot(state,own,other,p,trace))
    row=metrics(task,state,trace,cost);row['wall_seconds']=time.perf_counter()-started
    path=output/'episodes'/(task['key']+'.pt')
    save_torch(path,dict(task=task,trajectory=trace,final=snapshot(state,own,other,p,[])))
    row['episode_path']=str(path);row['episode_sha256']=sha(path)
    put(output/'cache'/(task['key']+'.json'),row)
    return row


def cached(task,output):
    path=output/'cache'/(task['key']+'.json')
    if not path.exists():return None
    row=json.loads(path.read_text())
    if row['key']!=task['key'] or sha(row['episode_path'])!=row['episode_sha256']:raise RuntimeError('Committed episode mismatch')
    return row


def freeze_run(output):
    config=dict(source=str(SOURCE),source_frozen_sha256=sha(SOURCE/'frozen.json'),
        sources_delivery_sha256=sha(SOURCE/'DELIVERY_SHA256.json'),requested_games=384,methods=list(METHODS),
        deadline_ms=50.,simulation_cap=1000,depth=100,opponent_simulations=3000,neural=False,training=False,
        bootstrap_unit='24 public histories; retain all worlds/methods/streams',bootstrap_draws=10000,
        bootstrap_seed=61020261025,bootstrap_sampling='paired resampling of 24 full history blocks',
        methods_unchanged=True,scenario_regeneration=False,independent_confirmation=False,
        algorithm_sources={str(p):sha(p) for p in (ROOT/'code/solvers/local_pursuit.py',ROOT/'code/solvers/root_audit_puct.py',ROOT/'code/solvers/c_puct.py',ROOT/'code/problems/example8.py',Path(cpp.__file__),ROOT/'code/prepare_l45_occlusion_scenarios.py')},
        runner_sha256=sha(__file__))
    path=output/'experiment.json'
    if path.exists():
        if json.loads(path.read_text())!=config:raise RuntimeError('Frozen execution configuration changed')
    else:put(path,config)
    put(output/'requests.json',tasks())


def profile(output):
    path=output/'profile.json'
    if path.exists():return json.loads(path.read_text())
    pilot=json.loads((SOURCE/'pilot/manifest.json').read_text())['entries'];chosen=[e for e in pilot if '_H4_' in e['id']]
    rows=[]
    for i,entry in enumerate(chosen):
        for method in METHODS:
            task=dict(key='engineering_'+entry['id']+'_'+method,entry=entry,world='A',stream=0,method=method,
                opponent_seed=int(np.random.SeedSequence([61020261025,99,i]).generate_state(1)[0]),
                environment_seed=6102501+i,folder='pilot')
            row=cached(task,output/'engineering') or episode(task,output/'engineering');rows.append(row)
            print(json.dumps(dict(stage='engineering',completed=len(rows),total=12,method=method,seconds=row['wall_seconds'])),flush=True)
    result=dict(passed=True,engineering_games=12,estimated_formal_seconds=sum(r['wall_seconds'] for r in rows)/12*384,rows=rows)
    put(path,result);return result


def evaluate(output):
    rows=[];started=time.perf_counter()
    for i,task in enumerate(tasks()):
        row=cached(task,output)
        try:row=row or episode(task,output)
        except Exception as error:
            put(output/'status.json',dict(stage='paused_execution_failure',completed=i,total=384,key=task['key'],error=repr(error),pid=None))
            raise
        rows.append(row)
        status=dict(stage='evaluate',completed=i+1,total=384,current=task['key'],elapsed_seconds=time.perf_counter()-started,pid=os.getpid())
        put(output/'status.json',status);print(json.dumps(status),flush=True)
    put(output/'results.json',rows);return rows


def replay(output):
    verified=[]
    for task in tasks():
        row=cached(task,output)
        if row is None:raise RuntimeError('Missing paired result')
        saved=load_torch(row['episode_path']);p,state,own,other,_=setup(task)
        for original in saved['trajectory']:
            nxt,actual=step(p,state,own,other,fixed=original['own']['simulations'])
            for name in ('action','next_state','particles','filter_rng','search_rng'):
                if not np.array_equal(original[name],actual[name]):raise RuntimeError('Replay mismatch: '+task['key']+' '+name)
            state=nxt
        verified.append(task['key'])
        put(output/'status.json',dict(stage='replay',completed=len(verified),total=384,pid=os.getpid()))
        print(json.dumps(dict(stage='replay',completed=len(verified),total=384)),flush=True)
    put(output/'replay.json',dict(passed=True,episodes=verified,fixed_recorded_simulations=True))


def aggregate(rows):
    latency=[v for r in rows for v in r['latencies_ms']];hidden=sum(r['hidden_steps'] for r in rows)
    return dict(games=len(rows),capture_rate=float(np.mean([r['capture'] for r in rows])),
        completion_cost=float(np.mean([r['completion_cost'] for r in rows])),
        failures=dict(collections.Counter(r['failure'] for r in rows)),
        reacquired_games=sum(r['first_reacquisition_after_handoff'] is not None for r in rows),
        first_reacquisition_mean=float(np.mean([r['first_reacquisition_after_handoff'] for r in rows if r['first_reacquisition_after_handoff'] is not None])) if any(r['first_reacquisition_after_handoff'] is not None for r in rows) else None,
        hidden_steps=hidden,coverage_1m=sum(r['hidden_covered'] or 0 for r in rows)/hidden if hidden and rows[0]['method']!='L' else None,
        latency_mean=float(np.mean(latency)),latency_p95=float(np.quantile(latency,.95)),
        simulations_mean=float(np.mean([v for r in rows for v in r['simulations']])),
        sources=dict(sum((collections.Counter(r['sources']) for r in rows),collections.Counter())))


def report(output):
    rows=json.loads((output/'results.json').read_text());histories=sorted({r['history_id'] for r in rows})
    index={(r['history_id'],r['world'],r['stream'],r['method']):r for r in rows}
    if len(histories)!=24 or len(index)!=384 or len(rows)!=384:raise RuntimeError('Missing/duplicated paired requests')
    comparisons={};rng=np.random.default_rng(61020261025);draws=rng.integers(0,24,size=(10000,24))
    for a,b in (('G','L'),('T','G'),('V','T'),('V','G')):
        result={}
        for metric in ('capture','completion_cost','own_illegal'):
            delta=np.array([np.mean([index[(h,w,s,a)][metric]-index[(h,w,s,b)][metric] for w in ('A','B') for s in (0,1)]) for h in histories])
            result[metric]=dict(delta=float(delta.mean()),ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),
                history_positive=int(np.sum(delta>0)),history_negative=int(np.sum(delta<0)))
        comparisons[a+'-'+b]=result
    summary={m:aggregate([r for r in rows if r['method']==m]) for m in METHODS}
    families={f:{m:aggregate([r for r in rows if r['method']==m and r['family']==f]) for m in METHODS} for f in ('long_wall','wall_corner','multi_route')}
    stats=dict(unit='24 public-history blocks, each includes both worlds/all methods/streams',bootstrap_draws=10000,bootstrap_seed=61020261025,
        summary=summary,by_family=families,comparisons=comparisons,development_only=True,independent_confirmation=False)
    put(output/'statistics.json',stats)
    lines=['困难场景L/G/T/V配对评估','',
        '完成384个请求，24组公共观测历史，每组保留A/B两个隐藏初态与两个随机流。全量固定模拟数重放通过。',
        '本轮为冻结困难模板上的开发实验，不是独立确认；没有新增训练、修订算法、调整参数或修改部署推荐。','',
        '|方法|合法捕获|完成成本（失败=原剩余步数）|重发现局数|单步mean/P95 ms|平均模拟数|',
        '|---|---:|---:|---:|---:|---:|']
    for m,v in summary.items():lines.append(f'|{m}|{v["capture_rate"]:.2%}|{v["completion_cost"]:.2f}|{v["reacquired_games"]}/{v["games"]}|{v["latency_mean"]:.3f}/{v["latency_p95"]:.3f}|{v["simulations_mean"]:.1f}|')
    lines+=['','|比较|捕获率差 pp [95% CI]|完成成本差 步 [95% CI]|','|---|---|---|']
    for name,v in comparisons.items():
        c,d=v['capture'],v['completion_cost'];lines.append(f'|{name}|{100*c["delta"]:+.2f} [{100*c["ci95"][0]:+.2f}, {100*c["ci95"][1]:+.2f}]|{d["delta"]:+.2f} [{d["ci95"][0]:+.2f}, {d["ci95"][1]:+.2f}]|')
    lines+=['','每次bootstrap抽取24个完整公共历史组，组内A/B、两流和所有方法共同保留；10000次，种子61020261025。旋转/平移不是独立地图泛化证据。','',
        '|地图|方法|捕获率|失败原因|重发现局数|隐藏覆盖率|','|---|---|---:|---|---:|---:|']
    for f,part in families.items():
        for m,v in part.items():lines.append(f'|{f}|{m}|{v["capture_rate"]:.2%}|{v["failures"]}|{v["reacquired_games"]}|{v["coverage_1m"] if v["coverage_1m"] is not None else "N/A"}|')
    lines+=['','所有方法从同一合法公共历史初始化；history[:-1]只吸收一次，当前观测在首次act中处理。原环境时钟保持第9/11步。',
        '控制器只读取局部传感器结果；隐藏真值由环境、信息占优的真实对手和事后评估使用。T/V仍是原50ms/1000模拟上限，模拟对手与真实3000次Search Only的差异保留。',
        '完整逐局指标见results.json，配对统计见statistics.json，轨迹/粒子/根统计见episodes，重放核验见replay.json。',
        '场景是事先冻结的困难几何模板，结果解释限于这套配置；尚未进行更多独立地图、对手或动力学验证。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    put(output/'status.json',dict(stage='complete',completed=384,total=384,replay_passed=True,report='REPORT.md',training=False,adoption=False,pid=None))
    return stats


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('preflight','pipeline','report','replay'))
    parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.command=='preflight':preflight(output);return
    with (ROOT/'current/.l45_local_pursuit_serial.lock').open('a+') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another local timing process is active')
        if args.command=='report':report(output);return
        warmup()
        if args.command=='replay':replay(output);return
        if not preflight(output)['passed']:raise RuntimeError('Preflight failed; formal run not started')
        freeze_run(output);profile(output);evaluate(output);replay(output);report(output)


if __name__=='__main__':main()
