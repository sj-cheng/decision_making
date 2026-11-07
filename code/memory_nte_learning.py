"""Conditional supervised stages. Every entry requires the frozen teacher gate.

Training and testing do not start merely by importing this module. Teacher-check
results and development labels have distinct budgets and directories.
"""
import os
import copy
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
import torch
import run_l45_memory_nte as run
from learning.memory_nte import Policy,features,observation_array,weighted_nll
from solvers.memory_nte import NeuralController


def authorize(output):
    run.check_frozen(output)
    path=output/'teacher_check/gate.json'
    if not path.exists() or not json.loads(path.read_text())['passed']:
        raise RuntimeError('Frozen teacher gate did not pass; no supervision/training/test is authorized')
    replay=output/'teacher_check/replay.json'
    if not replay.exists() or not json.loads(replay.read_text())['passed']:
        raise RuntimeError('Teacher replay must pass first')
    checks=output/'implementation_tests.json'
    if not checks.exists() or not json.loads(checks.read_text())['passed']:
        raise RuntimeError('Neural implementation checks must pass before training')
    files=[Path(__file__),run.ROOT/'code/learning/memory_nte.py',run.ROOT/'code/solvers/memory_nte.py']
    fingerprint={str(p):run.sha(p) for p in files}
    path=output/'learning_sources.json'
    if path.exists():
        if json.loads(path.read_text())!=fingerprint:raise RuntimeError('Learning implementation changed')
    else:run.put(path,fingerprint)


def model_path(output,round_,method,training_seed):
    return output/f'training/round{round_}/{method}_s{training_seed}/best.pt'


def make_controller(task,output):
    adapted=dict(task,method=task['method'] if task['method'] in ('G','T') else 'T')
    p,state,old,other,cost=run.setup(adapted,output,initialize=False)
    if task['method'] in ('N','H'):
        if run.sha(task['model_path'])!=task['model_sha256']:raise RuntimeError('Model fingerprint mismatch')
        checkpoint=run.load(Path(task['model_path']))
        model=Policy();model.load_state_dict(checkpoint['model']);model.eval()
        own=NeuralController(old.public,task['method'],task['controller_seed'],model.export(),deadline_ms=0.)
        own.initial_history=old.initial_history;own.initial_cursor=0
    else:own=old
    own.deadline_ms=50. if task['budget']=='time' else 0.
    return p,state,own,other,cost


def cache(task,output):
    path=output/task['stage']/'cache'/(task['key']+'.json')
    if not path.exists():return None
    value=json.loads(path.read_text())
    if value['task_fingerprint']!=run.digest(task) or run.sha(value['episode_path'])!=value['episode_sha256']:
        raise RuntimeError('Invalid episode cache')
    return value


def play(task,output):
    run.configure(output)
    cached=cache(task,output)
    if cached is not None:return cached
    stage=output/task['stage'];pending=stage/'pending'/(task['key']+'.pt')
    p,state,own,other,cost=make_controller(task,output)
    trace=[];failure=None
    if pending.exists():
        saved=run.load(pending)
        if saved['task_fingerprint']!=run.digest(task):raise RuntimeError('Pending task mismatch')
        state=saved['state'];own.restore(saved['controller']);other.set_rng_state(p,saved['opponent']);trace=saved['trace']
    fixed=None if task['budget']=='time' else 1000
    started=time.perf_counter()
    while not p.is_terminal(state):
        before=run.snapshot(state,own,other,p,[])
        try:
            run.assimilate(own)
            nxt,record=run.base.step(p,state,own,other,fixed=fixed)
        except Exception as error:
            if not run.recognized(error):raise
            failure=dict(exception=str(error),pre=before,post=run.snapshot(state,own,other,p,[]));break
        state=nxt;trace.append(record)
        saved=run.snapshot(state,own,other,p,trace);saved['task_fingerprint']=run.digest(task);run.save(pending,saved)
    row=run.base.metrics(task,state,trace,cost)
    row.update(uid=task['uid'],training_seed=task.get('training_seed'),budget=task['budget'],
        method_execution_failure=int(failure is not None),network_calls=sum(t['own'].get('network_calls',0) for t in trace),
        task_fingerprint=run.digest(task),wall_seconds=time.perf_counter()-started)
    if failure:row.update(capture=0,failure='estimator_recovery_failure',completion_cost=80-row['handoff_step'])
    path=stage/'episodes'/(task['key']+'.pt')
    run.save(path,dict(task=task,trajectory=trace,failure=failure,final=run.snapshot(state,own,other,p,[])))
    row.update(episode_path=str(path),episode_sha256=run.sha(path));run.put(stage/'cache'/(task['key']+'.json'),row)
    pending.unlink(missing_ok=True)
    return row


def make_task(output,e,world,stream,method,stage,budget='simulations',training_seed=None,round_=None):
    seedplan=json.loads((output/'private_evaluation_seed_plan.json').read_text())['entries'][e['id']]
    uid=method if training_seed is None else f'{method}_s{training_seed}'
    key=f'{e["id"]}_{world}_s{stream}_{uid}'
    task=dict(key=key,entry=e,world=world,stream=stream,method=method,uid=uid,stage=stage,budget=budget,
        training_seed=training_seed,controller_seed=seedplan['controller'][stream],
        opponent_seed=seedplan['opponent'][stream],environment_seed=run.seed(50,seedplan['opponent'][stream],ord(world)),folder='bank')
    if method in ('N','H'):
        path=model_path(output,round_,method,training_seed)
        task.update(model_path=str(path),model_sha256=run.sha(path))
    return task


def execute(tasks,output,workers,serial=False):
    if not tasks:return []
    stage=tasks[0]['stage'];path=output/stage/'requests.json'
    if path.exists():
        if json.loads(path.read_text())!=tasks:raise RuntimeError('Request manifest changed')
    else:run.put(path,tasks)
    rows=[]
    def accept(row):
        rows.append(row)
        status=dict(stage=stage,completed=len(rows),total=len(tasks),pid=os.getpid(),serial=serial)
        run.put(output/'status.json',status);print(json.dumps(status),flush=True)
    if serial:
        # No worker pool survives into actual wall-clock evaluation.
        for task in tasks:accept(play(task,output))
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for row in pool.map(play,tasks,[output]*len(tasks)):accept(row)
    run.put(output/stage/'results.json',rows)
    return rows


def select_contexts(rows,output,quota,source):
    manifest={e['id']:e for e in json.loads((output/'bank/manifest.json').read_text())['entries']}
    pools={}
    for row in rows:
        if row['uid']!=source:continue
        record=run.load(Path(row['episode_path']));task=record['task'];entry=manifest[row['history_id']]
        packet=json.loads((output/'bank'/entry['public']).read_text())
        history=copy.deepcopy(packet['history'][:-1])
        for frame in record['trajectory']:
            o=frame['observation']
            history.append(dict(step=o.step,position=o.position.tolist(),target=None if o.target is None else o.target.tolist(),remaining=o.remaining))
            context=dict(map=packet['map'],speed=packet['motion']['pursuer_axis_limit'],history=copy.deepcopy(history),
                history_id=entry['id'],split=entry['split'],family=entry['family'])
            identity=run.digest({k:context[k] for k in ('map','speed','history')})
            context['key']=identity
            group=(entry['split'],entry['family'],o.target is not None)
            pools.setdefault(group,{})[identity]=context
    # Fixed largest-remainder allocation, 80/20 train/validation, equal family
    # and visibility strata. Empty strata are reported; never outcome-replaced.
    groups=[(s,f,v) for s in ('train','validation') for f in run.FAMILIES for v in (False,True)]
    weights=np.array([(.8 if s=='train' else .2)/6 for s,f,v in groups])
    counts=np.floor(quota*weights).astype(int)
    for i in np.argsort(-(quota*weights-counts),kind='stable')[:quota-int(counts.sum())]:counts[i]+=1
    selected=[];shortfalls=[]
    for group,n in zip(groups,counts):
        candidates=sorted(pools.get(group,{}).values(),key=lambda c:run.digest([run.SEED,source,c['key']]))
        selected.extend(candidates[:n]);shortfalls.append(dict(stratum=list(group),requested=int(n),available=len(candidates),selected=min(int(n),len(candidates))))
    return selected,shortfalls


def label_query(context,output):
    from solvers.local_pursuit import simulation,guided
    path=output/'labels'/(context['key']+'.pt')
    if path.exists():return dict(key=context['key'],path=str(path),reused=True)
    reservation=output/'label_reservations'/(context['key']+'.json')
    if not reservation.exists():
        run.put(reservation,dict(key=context['key'],budget=10000,seed=int(context['key'][:8],16),
            recovery='resume the same query from committed simulation deltas; at most one uncommitted simulation is replayed'))
    public=run.base.PublicMap(context['map'],context['speed'])
    controller=run.base.Controller(public,'T',int(context['key'][:8],16),simulations=10000,deadline_ms=0.)
    initial=output/'label_reservations'/(context['key']+'.initial.pt')
    journal=output/'label_reservations'/(context['key']+'.jsonl')
    count=np.zeros(2,np.int64);before=None;failure=None
    try:
        if initial.exists():
            saved=run.load(initial);controller.restore(saved['controller']);count=saved['count']
        else:
            for item in context['history'][:-1]:
                before=controller.snapshot();controller.update(run.base.obs(item),np.zeros(2,np.int64))
            before=controller.snapshot()
            controller.update(run.base.obs(context['history'][-1]),count)
            run.save(initial,dict(controller=controller.snapshot(),count=count))
    except Exception as error:
        if not run.recognized(error):raise
        failure=dict(exception=str(error),pre=before,post=controller.snapshot())
        run.save(path,dict(context=context,actions=np.empty((0,2)),visits=np.empty(0),failure=failure,controller=controller.snapshot()))
        return dict(key=context['key'],path=str(path),reused=False,failed=True)
    observation=run.base.obs(context['history'][-1]);p=public
    roots,root_sources,n=guided(observation.position,controller.particles,p.speed,p.vertices,p.dist,p.edist,p.boxes,count,False)
    parents=np.full(10001,-1,np.int64);actions=np.zeros((10001,2));sources=np.zeros(10001,np.int64)
    visits=np.zeros(10001,np.int64);totals=np.zeros(10001);size=1;completed=0
    if journal.exists():
        valid_end=0
        with journal.open('rb') as f:
            while True:
                line=f.readline()
                if not line:break
                if not line.endswith(b'\n'):break
                row=json.loads(line)
                if row['simulation']!=completed+1:raise RuntimeError('Non-contiguous teacher simulation journal')
                for node,parent,action,source,visit,total in row['changes']:
                    parents[node]=parent;actions[node]=action;sources[node]=source;visits[node]=visit;totals[node]=total
                controller.rng[:]=row['rng'];count[:]=row['count'];size=row['size'];completed=row['simulation'];valid_end=f.tell()
        with journal.open('r+b') as f:f.truncate(valid_end)
    with journal.open('a') as f:
        for index in range(completed,10000):
            previous=visits.copy()
            size=simulation(observation.position,controller.particles,controller.last,np.asarray(controller.history).reshape(-1,2),
                p.speed,observation.remaining,p.vertices,p.dist,p.boxes,roots,root_sources,n,controller.rng,count,
                parents,actions,sources,visits,totals,size)
            changed=np.flatnonzero(visits!=previous)
            changes=[[int(i),int(parents[i]),actions[i].tolist(),int(sources[i]),int(visits[i]),float(totals[i])] for i in changed]
            f.write(json.dumps(dict(simulation=index+1,size=size,rng=controller.rng.tolist(),count=count.tolist(),changes=changes))+'\n');f.flush()
    root=np.flatnonzero(parents[:size]==0)
    if visits[root].sum()!=10000:raise RuntimeError('Incomplete teacher root visits')
    run.save(path,dict(context=context,actions=actions[root]/context['speed'],visits=visits[root].astype(float),
        failure=None,controller=controller.snapshot(),actual_committed_simulations=10000))
    return dict(key=context['key'],path=str(path),reused=False,failed=False)


def collect(output,workers,round_):
    authorize(output)
    stage=f'collection/round{round_}';manifest=output/stage/'labels.json'
    if manifest.exists():return
    entries=[e for e in json.loads((output/'bank/manifest.json').read_text())['entries'] if e['split']!='test']
    identities=[('G',None),('T',None)] if round_==1 else [(m,s) for m in ('N','H') for s in range(3)]
    tasks=[]
    for i,e in enumerate(entries):
        world=('A','B')[i%2];stream=(i//2)%2
        for m,s in identities:tasks.append(make_task(output,e,world,stream,m,stage,training_seed=s,round_=round_-1))
    rows=execute(tasks,output,workers)
    selected={};strata={}
    for m,s in identities:
        source=m if s is None else f'{m}_s{s}'
        contexts,counts=select_contexts(rows,output,480 if round_==1 else 160,source)
        strata[source]=counts
        for c in contexts:selected[c['key']]=c
    run.put(output/stage/'selection.json',dict(contexts=list(selected.values()),strata=strata,maximum_slots=960))
    # Independent processes work on distinct public-history keys.
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        labels=list(pool.map(label_query,selected.values(),[output]*len(selected)))
    if len(list((output/'label_reservations').glob('*.json')))>1920:raise RuntimeError('Label budget exceeded')
    run.put(manifest,labels)


def tensor_batch(items,memory):
    encoded=[features(np.array(x['context']['map'],np.float64),observation_array(x['context']['history']),
        len(x['context']['history']),x['context']['speed'],memory) for x in items]
    maps=max(max(len(x[0]) for x in encoded),1);actions=max(len(x['visits']) for x in items);n=len(items)
    b=np.zeros((n,maps,4),np.float32);bm=np.zeros((n,maps),np.float32)
    h=np.zeros((n,17,7),np.float32);hm=np.zeros((n,17),np.float32);p=np.zeros((n,7),np.float32)
    a=np.zeros((n,actions,2),np.float32);v=np.zeros((n,actions),np.float32)
    for i,(item,enc) in enumerate(zip(items,encoded)):
        b[i,:len(enc[0])]=enc[0];bm[i,:len(enc[0])]=1.;h[i],hm[i],p[i]=enc[1:]
        a[i,:len(item['actions'])]=item['actions'];v[i,:len(item['visits'])]=item['visits']
    return [torch.from_numpy(x) for x in (b,bm,h,hm,p,a,v)]


def train(output,round_):
    authorize(output);all_labels={}
    for r in range(1,round_+1):
        for row in json.loads((output/f'collection/round{r}/labels.json').read_text()):all_labels[row['key']]=row
    items=[run.load(Path(row['path'])) for row in all_labels.values()]
    items=[x for x in items if x['failure'] is None]
    training=[x for x in items if x['context']['split']=='train'];validation=[x for x in items if x['context']['split']=='validation']
    if not training or not validation:raise RuntimeError('Insufficient labeled train/validation data; no automatic replacements')
    for method in ('N','H'):
        for s in range(3):
            folder=model_path(output,round_,method,s).parent
            if (folder/'complete.json').exists():continue
            torch.manual_seed(s);model=Policy();opt=torch.optim.Adam(model.parameters(),lr=3e-4)
            rng=np.random.default_rng(run.seed(80,round_,s));start=0;best=float('inf');log=[]
            resume=folder/'last.pt'
            if resume.exists():
                previous=run.load(resume);model.load_state_dict(previous['model']);opt.load_state_dict(previous['optimizer'])
                start=previous['epoch']+1;best=previous['best'];log=previous['log'];rng.bit_generator.state=previous['rng'];torch.set_rng_state(previous['torch_rng'])
            elif round_>1:
                previous=run.load(model_path(output,round_-1,method,s));model.load_state_dict(previous['model']);opt.load_state_dict(previous['optimizer'])
            for epoch in range(start,30):
                model.train();permutation=rng.permutation(len(training));losses=[]
                for offset in range(0,len(training),128):
                    batch=tensor_batch([training[i] for i in permutation[offset:offset+128]],method=='H')
                    mu,lv=model(*batch[:5]);loss=weighted_nll(mu,lv,*batch[5:])
                    if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step();losses.append(float(loss.detach()))
                model.eval();total=0.
                with torch.no_grad():
                    for offset in range(0,len(validation),128):
                        part=validation[offset:offset+128];batch=tensor_batch(part,method=='H')
                        mu,lv=model(*batch[:5]);total+=float(weighted_nll(mu,lv,*batch[5:]))*len(part)
                score=total/len(validation);log.append(dict(epoch=epoch,training_nll=float(np.mean(losses)),validation_nll=score))
                improved=score<best;best=min(best,score)
                saved=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=best,log=log,
                    rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),method=method,seed=s,round=round_)
                if improved:run.save(folder/'best.pt',saved)
                run.save(resume,saved)
            run.put(folder/'complete.json',dict(epochs=30,training_samples=len(training),validation_samples=len(validation),
                best_validation_nll=best,best_sha256=run.sha(folder/'best.pt'),log=log))


def evaluate(output,workers):
    authorize(output)
    entries=[e for e in json.loads((output/'bank/manifest.json').read_text())['entries'] if e['split']=='test']
    configs=[('G',None),('T',None)]+[(m,s) for m in ('N','H') for s in range(3)]
    for budget in ('simulations','time'):
        tasks=[]
        for i,e in enumerate(entries):
            for wi,w in enumerate(('A','B')):
                for stream in (0,1):
                    r=(i+wi+stream)%8;order=configs[r:]+configs[:r]
                    for m,s in order:tasks.append(make_task(output,e,w,stream,m,'evaluation/'+budget,budget,s,2))
        assert len(tasks)==1920
        if budget=='time':
            # Compilation / inference warmup uses copies and an independent
            # observation; no state or RNG is shared with evaluated controllers.
            run.base.warmup()
            for m,s in configs:
                task=next(t for t in tasks if t['method']==m and t['training_seed']==s)
                _,_,c,_,_=make_controller(task,output)
                public=run.base.PublicMap([],1.3)
                clone=(NeuralController(public,m,run.seed(98),c.weights,deadline_ms=0.) if m in ('N','H')
                    else run.base.Controller(public,m,run.seed(98),deadline_ms=0.))
                clone.act(run.base.Observation(0,np.array([-7.123,7.321]),np.array([-3.234,6.987]),80),fixed_simulations=3)
        execute(tasks,output,workers,serial=(budget=='time'))
    checks=[]
    for budget in ('simulations','time'):
        tasks=json.loads((output/f'evaluation/{budget}/requests.json').read_text())
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            checks.extend(pool.map(verify_play,tasks,[output]*len(tasks)))
    run.put(output/'evaluation/replay.json',dict(passed=True,checks=checks))
    statistics(output)


def verify_play(task,output):
    run.configure(output);row=cache(task,output);record=run.load(Path(row['episode_path']))
    p,state,own,other,_=make_controller(task,output)
    for original in record['trajectory']:
        run.assimilate(own)
        nxt,actual=run.base.step(p,state,own,other,fixed=original['own']['simulations'])
        for name in ('action','next_state','particles','filter_rng','search_rng'):
            if not np.array_equal(actual[name],original[name]):raise RuntimeError('Neural replay mismatch: '+task['key']+' '+name)
        if actual['opponent_rng']!=original['opponent_rng']:raise RuntimeError('Opponent replay differs')
        state=nxt
    reproduced=False
    if record['failure']:
        try:
            run.assimilate(own);run.base.step(p,state,own,other,fixed=1000)
        except Exception as error:
            if not run.recognized(error):raise
            reproduced=True
        if not reproduced:raise RuntimeError('Estimator exception did not reproduce')
        if run.fingerprint_controller(own.snapshot())!=run.fingerprint_controller(record['failure']['post']['controller']):
            raise RuntimeError('Post-failure controller state mismatch')
    return dict(key=task['key'],budget=task['budget'],passed=True,steps=len(record['trajectory']),failure_reproduced=reproduced)


def statistics(output):
    summaries={};comparisons={}
    rng=np.random.default_rng(run.SEED);draws=rng.integers(0,60,(10000,60))
    for budget in ('simulations','time'):
        rows=json.loads((output/f'evaluation/{budget}/results.json').read_text())
        index={(r['history_id'],r['world'],r['stream'],r['uid']):r for r in rows}
        histories=sorted({r['history_id'] for r in rows})
        if len(rows)!=1920 or len(index)!=1920 or len(histories)!=60:raise RuntimeError('Incomplete evaluation pairing')
        summaries[budget]={};arrays={}
        for method in ('G','T','N','H'):
            uids=[method] if method in ('G','T') else [f'{method}_s{s}' for s in range(3)]
            arrays[method]={}
            for metric in ('capture','completion_cost','method_execution_failure','own_illegal'):
                arrays[method][metric]=np.array([np.mean([index[h,w,s,u][metric] for w in ('A','B') for s in (0,1) for u in uids]) for h in histories])
            selected=[r for r in rows if r['method']==method];lat=[v for r in selected for v in r['latencies_ms']]
            summaries[budget][method]={k:float(v.mean()) for k,v in arrays[method].items()}
            summaries[budget][method].update(latency_mean_ms=float(np.mean(lat)),latency_p95_ms=float(np.quantile(lat,.95)),
                per_seed={u:float(np.mean([r['capture'] for r in selected if r['uid']==u])) for u in uids},
                per_family={f:dict(capture=float(np.mean([r['capture'] for r in selected if r['family']==f])),
                    failures={reason:sum(r['failure']==reason for r in selected if r['family']==f) for reason in sorted({r['failure'] for r in selected})}) for f in run.FAMILIES})
        comparisons[budget]={};joint=[];points=[]
        for ref in ('N','T','G'):
            pair={}
            for metric in arrays['H']:
                delta=arrays['H'][metric]-arrays[ref][metric];boot=delta[draws].mean(1)
                pair[metric]=dict(delta=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
                if metric=='capture':joint.append(boot);points.append(delta.mean())
            comparisons[budget]['H-'+ref]=pair
        critical=float(np.quantile(np.max(np.stack(joint,1)-np.array(points),axis=1),.95))
        for ref,p in zip(('N','T','G'),points):comparisons[budget]['H-'+ref]['capture']['simultaneous_lower95']=float(p-critical)
    time_summary=summaries['time'];h=time_summary['H']
    same_direction={}
    for ref in ('N','T'):
        same_direction[ref]=sum(h['per_seed'][f'H_s{s}']>(time_summary[ref]['per_seed'][f'N_s{s}'] if ref=='N' else time_summary[ref]['capture']) for s in range(3))
    vs_search=all(h['capture']-time_summary[r]['capture']>=.03-1e-12 and same_direction[r]>=2 for r in ('N','T'))
    g=time_summary['G']
    vs_geometry=h['capture']-g['capture']>=.03-1e-12 or (h['capture']>=g['capture'] and h['completion_cost']<=.9*g['completion_cost'])
    reliability=all(h[k]-time_summary[r][k]<=.02+1e-12 for r in ('N','T','G') for k in ('method_execution_failure','own_illegal'))
    decision=dict(continue_development=bool(vs_search and vs_geometry and reliability),search_gate=bool(vs_search),geometry_gate=bool(vs_geometry),
        reliability_gate=bool(reliability),positive_seed_counts=same_direction,
        simultaneous_capture_lower_positive=all(v['capture']['simultaneous_lower95']>0 for v in comparisons['time'].values()),
        deployment_changed=False)
    run.put(output/'development_gate.json',decision)
    run.put(output/'statistics.json',dict(summaries=summaries,comparisons=comparisons,bootstrap_unit='60 public histories; average seeds within each unit',draws=10000,seed=run.SEED))
    text='# 观测记忆增强 NTE：三种子开发结果\n\n所有方法共享场景、估计和搜索条件；三个训练种子分别报告，未更换部署推荐。\n\n'
    for budget,summary in summaries.items():
        text+=f'\n**{budget}**\n\n|方法|捕获率|完成成本|执行失败率|平均/P95 ms|\n|---|---:|---:|---:|---|\n'
        for m,v in summary.items():text+=f'|{m}|{v["capture"]:.2%}|{v["completion_cost"]:.2f}|{v["method_execution_failure"]:.2%}|{v["latency_mean_ms"]:.3f}/{v["latency_p95_ms"]:.3f}|\n'
    text+='\n完整三项配对区间、同时下界、分地图及逐种子结果见 statistics.json。本轮是开发实验，不能替代独立确认。\n'
    text+='\n开发门槛：'+('通过，可另行登记独立验证。' if decision['continue_development'] else '未通过，本轮停止，不自动扩展训练或样本。')+'\n'
    (output/'REPORT.md').write_text(text)
    run.put(output/'status.json',dict(stage='completed',pid=None,training_seeds=[0,1,2],test_requests=3840,deployment_changed=False))


def pipeline(output,workers):
    authorize(output)
    for round_ in (1,2):collect(output,workers,round_);train(output,round_)
    evaluate(output,workers)


def dispatch(command,output,workers):
    authorize(output)
    if command=='collect':
        round_=2 if model_path(output,1,'H',2).exists() else 1
        collect(output,workers,round_)
    elif command=='train':train(output,2 if (output/'collection/round2/labels.json').exists() else 1)
    elif command=='evaluate':evaluate(output,workers)
    else:raise ValueError(command)
