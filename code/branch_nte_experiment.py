"""Audits and gated learning/evaluation for the registered branch-NTE run."""
import copy
import importlib.util
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import run_l45_branch_nte as run
from solvers.branch_support import Support,GridMap,cell_id
from solvers.branch_nte import Controller
from learning.branch_nte import Policy,Runtime,encoded,loss


def audit(output):
    run.check_frozen(output)
    path=output/'audit.json'
    if path.exists():
        result=json.loads(path.read_text())
        if not result['passed']:raise RuntimeError('Previous audit failed')
        return result
    spec=importlib.util.spec_from_file_location('branch_tests',run.ROOT/'code/tests/test_l45_branch_nte.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tests=module.run_unit_checks();coverage=[];invariance=[];frames=0
    # All bank histories and geometric witnesses are checked without evaluating
    # algorithms on test outcomes. This does not consume the sealed test set.
    for ei,e in enumerate(run.entries()):
        packet,environment=run.inputs(e);p=run.public_map(e);base=Support(p)
        for r in packet['history']:base.update(run.base.obs(r))
        for world in environment['worlds']:
            s=Support(p)
            for r,true in zip(packet['history'],world['history_states']):
                s.update(run.base.obs(r));frames+=1
                if not s.covers(true['evader']):raise RuntimeError('History true cell excluded: '+e['id'])
            for direction,trajectory in world['continuations'].items():
                branch=s.clone()
                for true in trajectory[1:]:
                    observation=p.observe(true['step'],np.array(true['pursuer']),np.array(true['evader']))
                    branch.update(observation);frames+=1
                    if not branch.covers(true['evader']):raise RuntimeError('Geometric witness excluded: '+e['id']+direction)
            coverage.append(dict(history=e['id'],world=world['id'],handoff_covered=True,upper_lower_witnesses_covered=True,cells=int(s.mask.sum())))
        if e['teacher_check']:
            a=Controller(p,'T',packet['controller_seeds'][0],deadline_ms=0.)
            b=Controller(p,'T',packet['controller_seeds'][0],deadline_ms=0.)
            for r in packet['history'][:-1]:a.update(run.base.obs(r));b.update(run.base.obs(r))
            aa=a.act(run.base.obs(packet['history'][-1]),fixed_simulations=3)
            bb=b.act(run.base.obs(packet['history'][-1]),fixed_simulations=3)
            assert np.array_equal(aa,bb) and np.array_equal(a.rng,b.rng) and a.last_stats['root']==b.last_stats['root']
            invariance.append(dict(history=e['id'],passed=True))
        if ei%20==0:print(json.dumps(dict(stage='audit_geometry',completed=ei+1,total=180)),flush=True)
    # Real trajectories from the completed predecessor, used only for support
    # diagnostics. Never compare its capture rate with the new search arms.
    historical_frames=0
    rows=json.loads((run.SOURCE/'teacher_check/results.json').read_text())
    for row in rows:
        e=next(e for e in run.entries() if e['id']==row['history_id']);packet,_=run.inputs(e)
        s=Support(run.public_map(e))
        for r in packet['history'][:-1]:s.update(run.base.obs(r))
        record=run.load(row['episode_path'])
        for frame in record['trajectory']:
            s.update(frame['observation']);historical_frames+=1
            if not s.covers(frame['state'][:2,0]):raise RuntimeError('Historical true cell excluded')
    # The same local network and search initialization sees no private state.
    p=GridMap([[[-1.,1.],[-4.,4.]]],1.1);torch.manual_seed(7);network=Policy();policy=Runtime(network,p,'H')
    small=[dict(step=0,position=[-3.,5.],target=[2.,5.],remaining=80),dict(step=1,position=[-3.,4.],target=None,remaining=79)]
    own=Controller(p,'H',123,policy,deadline_ms=0.)
    for r in small:own.update(run.base.obs(r))
    f,*_=own.support.branches();w,mu,lv=policy(own.records,f)
    assert np.isfinite(np.r_[w,mu.ravel(),lv.ravel()]).all()
    result=dict(passed=True,unit_checks=tests,bank_groups=180,worlds=360,geometric_frames=frames,
        historical_episodes=len(rows),historical_frames=historical_frames,all_true_cells_retained=True,
        hidden_world_initialization_pairs=invariance,coverage=coverage,
        scope='support and public-interface acceptance; no test control outcomes evaluated',
        ambiguity='outer cells may contain unreachable/inconsistent points; no calibrated-probability claim',
        training_started=False)
    run.put(path,result);return result


def authorize(output):
    run.check_frozen(output)
    for name in ('audit.json','teacher_check/gate.json','teacher_check/replay.json'):
        path=output/name
        if not path.exists() or not json.loads(path.read_text())['passed']:raise RuntimeError('Stage prerequisite failed: '+name)


def collection_tasks():
    return [run.make_task(e,w,0,m,'collection',128) for e in run.entries() if e['split']!='test' for w in ('A','B') for m in ('G','T')]


def context_from(task,history,target):
    context=dict(entry=task['entry'],history=history,target=np.asarray(target).tolist(),world=task['world'],split=task['entry']['split'])
    context['key']=run.digest(context)
    return context


def collect(output,workers):
    authorize(output);rows=run.execute(collection_tasks(),output,workers)
    if len(rows)!=480:raise RuntimeError('Incomplete collection')
    index=output/'data_index.json'
    if index.exists():contexts=json.loads(index.read_text())['contexts']
    else:
        existing=[]
        for path in sorted((output/'labels').glob('*.pt')):
            record=run.load(path);existing.append(record['context'])
        pools={};pilotkeys={c['key'] for c in existing}
        for row in rows:
            record=run.load(row['episode_path']);task=record['task'];packet,_=run.inputs(task['entry'])
            history=copy.deepcopy(packet['history'][:-1])
            for frame in record['trajectory']:
                o=frame['observation'];history.append(dict(step=o.step,position=o.position.tolist(),target=None if o.target is None else o.target.tolist(),remaining=o.remaining))
                context=context_from(task,copy.deepcopy(history),frame['state'][:2,0])
                if context['key'] in pilotkeys:continue
                group=(row['history_id'],context['key'])
                stratum=(task['entry']['split'],task['entry']['family'],o.target is not None)
                pools.setdefault(stratum,{})[context['key']]=context
        contexts=list(existing);shortfalls=[]
        protocol=json.loads((output/'experiment.json').read_text())
        for split,total in (('train',protocol['train_query_cap']),('validation',256)):
            already=sum(c['split']==split for c in contexts);remaining=total-already
            strata=[(split,f,v) for f in run.FAMILIES for v in (False,True)]
            for i,g in enumerate(strata):
                quota=remaining//6+int(i<remaining%6)
                candidates=sorted(pools.get(g,{}).values(),key=lambda c:run.digest([run.SEED,c['key']]))
                contexts+=candidates[:quota];shortfalls.append(dict(stratum=list(g),quota=quota,available=len(candidates),selected=min(quota,len(candidates))))
        if len({c['key'] for c in contexts})!=len(contexts):raise RuntimeError('Duplicate teacher context')
        run.put(index,dict(contexts=contexts,shortfalls=shortfalls,selection='map/visibility strata, fixed hash ordering; no outcomes',
            queries_requested=len(contexts),max_queries=protocol['active_query_cap']))
    for i,context in enumerate(contexts):
        run.teacher_query(context,output)
        run.put(output/'status.json',dict(stage='supervision',completed=i+1,total=len(contexts),pid=os.getpid()))
        if i%10==0:print(json.dumps(dict(stage='supervision',completed=i+1,total=len(contexts))),flush=True)
    run.put(output/'labels_manifest.json',dict(index_sha256=run.sha(index),labels={str(p):run.sha(p) for p in (output/'labels').glob('*.pt')}))
    inherited=json.loads((output/'experiment.json').read_text())['inherited_invalid_queries']
    run.put(output/'budget.json',dict(teacher_queries=len(list((output/'query_reservations').glob('*.json')))+inherited,
        invalid_implementation_queries=inherited,teacher_simulations=10000*(inherited+len(list((output/'labels').glob('*.pt')))),cap=15360000,
        collection_requests=480,teacher_first_action_queries_included=True,
        restart_policy='same reserved query, last atomic simulation checkpoint; uncommitted replay is recovery work, not new samples'))


def dataset(output):
    lock=json.loads((output/'labels_manifest.json').read_text())
    if lock['index_sha256']!=run.sha(output/'data_index.json') or any(run.sha(p)!=h for p,h in lock['labels'].items()):raise RuntimeError('Frozen supervision changed')
    contexts=json.loads((output/'data_index.json').read_text())['contexts'];values=[]
    for c in contexts:
        record=run.load(output/'labels'/(c['key']+'.pt'))
        if record['failure']:continue
        p=run.public_map(c['entry']);support=Support(p)
        for r in c['history']:support.update(run.base.obs(r))
        branches,assignment,_,_=support.branches();branch=int(assignment[cell_id(np.array(c['target']))])
        if branch<0:raise RuntimeError('Training state outside lawful public support')
        v=encoded(p,c['history'],branches)
        roots=record['root'];actions=np.zeros((16,2));visits=np.zeros(16)
        for i,r in enumerate(roots):actions[i]=np.array(r['action'])/p.speed;visits[i]=r['visits']
        v.update(actions=actions,visits=visits,branch=branch,split=c['split'],key=c['key']);values.append(v)
    return values


def batch(values):
    max_obstacles=max(len(v['obstacles']) for v in values)
    tensors={}
    for key in ('history','history_mask','public','branches','actions','visits'):
        tensors[key]=torch.tensor(np.stack([v[key] for v in values]),dtype=torch.float32)
    obstacles=np.zeros((len(values),max_obstacles,4));mask=np.zeros((len(values),max_obstacles))
    for i,v in enumerate(values):obstacles[i,:len(v['obstacles'])]=v['obstacles'];mask[i,:len(v['obstacles'])]=1
    tensors['obstacles']=torch.tensor(obstacles,dtype=torch.float32);tensors['map_mask']=torch.tensor(mask,dtype=torch.float32)
    tensors['branch']=torch.tensor([v['branch'] for v in values],dtype=torch.long)
    return tensors


def train_one(output,mode,training_seed,data):
    directory=output/'training'/f'{mode}_seed{training_seed}';last=directory/'last.pt';best=directory/'best.pt'
    train=[d for d in data if d['split']=='train'];validation=[d for d in data if d['split']=='validation']
    if not train or not validation:raise RuntimeError('Empty data split')
    torch.manual_seed(training_seed);model=Policy();optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    start=0;best_score=float('inf');history=[];fingerprint=run.sha(output/'data_index.json')
    if last.exists():
        saved=run.load(last)
        if saved['data_sha256']!=fingerprint:raise RuntimeError('Training data changed')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng'])
        start=saved['epoch'];best_score=saved['best_nll'];history=saved['history']
    for epoch in range(start,30):
        model.train();order=torch.randperm(len(train)).tolist();losses=[]
        for j in range(0,len(order),128):
            v=batch([train[k] for k in order[j:j+128]])
            logits,mu,lv=model(**{k:v[k] for k in ('obstacles','map_mask','history','history_mask','public','branches')},mode=mode)
            objective,_=loss(logits,mu,lv,v['actions'],v['visits'],v['branch'],mode)
            optimizer.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step();losses.append(float(objective.detach()))
        model.eval();scores=[]
        with torch.no_grad():
            for j in range(0,len(validation),128):
                v=batch(validation[j:j+128]);logits,mu,lv=model(**{k:v[k] for k in ('obstacles','map_mask','history','history_mask','public','branches')},mode=mode)
                _,nll=loss(logits,mu,lv,v['actions'],v['visits'],v['branch'],mode);scores.append((float(nll),len(v['branch'])))
        score=sum(a*n for a,n in scores)/sum(n for _,n in scores)
        if not np.isfinite(score):raise RuntimeError('Nonfinite training result')
        improved=score<best_score
        best_score=min(score,best_score);history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_mixture_nll=score))
        record=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),epoch=epoch+1,
            best_nll=best_score,history=history,data_sha256=fingerprint,mode=mode,training_seed=training_seed,
            parameter_count=sum(p.numel() for p in model.parameters()),train_records=len(train),validation_records=len(validation))
        if improved:run.save(best,record)
        run.save(last,record);run.put(directory/'history.json',history)
    return str(best)


def train(output,seeds=(0,)):
    authorize(output);data=dataset(output)
    for training_seed in seeds:
        for mode in ('M','H'):
            run.put(output/'status.json',dict(stage='train',mode=mode,training_seed=training_seed,pid=os.getpid()))
            train_one(output,mode,training_seed,data)


def add_model(task,output):
    if task['method'] not in ('G','T'):
        path=output/'training'/f'{"M" if task["method"]=="M" else "H"}_seed{task["training_seed"]}'/'best.pt'
        task['model_sha256']=run.sha(path)
    return task


def evaluation_tasks(output,split,seeds,stage,budget='time',fixed=False):
    selected=[e for e in run.entries() if e['split']==split]
    if fixed:
        selected=[e for family in run.FAMILIES for e in [x for x in selected if x['family']==family][:4]]
    configs=[('T',None),('M',0),('H',0),('H-U',0)] if fixed else [('G',None),('T',None)]+[(m,s) for m in ('M','H','H-U','H-D') for s in seeds]
    tasks=[]
    for i,e in enumerate(selected):
        for w in ('A','B'):
            for stream in (0,) if fixed else (0,1):
                rotation=(i+ord(w)+stream)%len(configs);order=configs[rotation:]+configs[:rotation]
                for method,s in order:tasks.append(add_model(run.make_task(e,w,stream,method,stage,budget,s),output))
    return tasks


def warmup(output):
    public=GridMap([[[-1.,1.],[-4.321,4.321]]],1.1)
    records=[dict(step=0,position=[-3.,5.321],target=[2.,5.321],remaining=80),dict(step=1,position=[-3.,4.321],target=None,remaining=79)]
    timings={}
    for method in run.METHODS:
        policy=None
        if method not in ('G','T'):
            mode='M' if method=='M' else 'H';model=Policy();model.load_state_dict(run.load(output/'training'/f'{mode}_seed0'/'best.pt')['model']);model.eval();policy=Runtime(model,public,mode)
        own=Controller(public,method,742,policy,deadline_ms=0.);own.update(run.base.obs(records[0]));started=time.perf_counter()
        own.act(run.base.obs(records[1]),fixed_simulations=2);timings[method]=(time.perf_counter()-started)*1000
    return dict(independent_geometry='wall_halfheight4.321',excluded=True,timings_ms=timings)


def summary(rows,method):
    rs=[r for r in rows if r['method']==method]
    return dict(capture=float(np.mean([r['capture'] for r in rs])),cost=float(np.mean([r['completion_cost'] for r in rs])),
        failures=float(np.mean([r['method_execution_failure'] for r in rs])),illegal=float(np.mean([r['own_illegal'] for r in rs])))


def screen_gate(rows):
    h=summary(rows,'H');comparisons={};passed=True
    for ref in ('M','T','G'):
        b=summary(rows,ref);cost=h['cost']<=.9*b['cost']
        if ref=='G':cost=cost or (h['capture']>=b['capture']+.03-1e-12 and h['cost']<=b['cost'])
        flags=dict(cost_or_capture=cost,capture_preserved=h['capture']>=b['capture']-.02-1e-12,
            failures=h['failures']<=b['failures']+.02+1e-12,illegal=h['illegal']<=b['illegal']+.02+1e-12)
        comparisons[ref]=dict(reference=b,checks=flags);passed=passed and all(flags.values())
    return dict(passed=bool(passed),candidate=h,comparisons=comparisons)


def serial_execute(tasks,output,stage):
    import fcntl
    with (run.ROOT/'current/.l45_local_pursuit_serial.lock').open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # No process pool survives here. Static maps and inference/JIT warmup
        # precede the serial measurements and are explicitly measured.
        started=time.perf_counter()
        for e in {t['entry']['id']:t['entry'] for t in tasks}.values():run.public_map(e)
        preparation=(time.perf_counter()-started)*1000
        run.put(output/stage/'warmup.json',dict(map_preprocessing_total_ms=preparation,**warmup(output)))
        return run.execute(tasks,output,serial=True)


def screen(output,workers):
    authorize(output)
    rows=serial_execute(evaluation_tasks(output,'validation',(0,),'screen'),output,'screen')
    if len(rows)!=576:raise RuntimeError('Incomplete screen')
    run.replay(output,workers,'screen')
    fixed=run.execute(evaluation_tasks(output,'validation',(0,),'screen_fixed',256,True),output,workers)
    if len(fixed)!=96:raise RuntimeError('Incomplete fixed-simulation check')
    run.replay(output,workers,'screen_fixed')
    gate=screen_gate(rows);run.put(output/'screen/gate.json',gate);run.report(output)
    return gate


def evaluate(output,workers):
    authorize(output)
    if not json.loads((output/'screen/gate.json').read_text())['passed']:raise RuntimeError('Screen gate failed')
    train(output,(1,2))
    tasks=evaluation_tasks(output,'test',(0,1,2),'evaluate')
    if len(tasks)!=3360:raise RuntimeError('Wrong sealed-test count')
    lock=output/'candidate_lock.json'
    record=dict(protocol_sha256=run.sha(output/'experiment.json'),requests_sha256=run.digest(tasks),
        models={str(p):run.sha(p) for p in sorted((output/'training').glob('*/best.pt'))},comparisons=['H-M','H-T','H-G','H-H-D'])
    if lock.exists() and json.loads(lock.read_text())!=record:raise RuntimeError('Candidate lock changed')
    run.put(lock,record)
    rows=serial_execute(tasks,output,'evaluate');run.replay(output,workers,'evaluate')
    stats=run.statistics(rows,run.METHODS,[('H',x) for x in ('M','T','G','H-D')])
    histories=sorted({r['history_id'] for r in rows});draws=np.random.default_rng(run.SEED).integers(0,60,(10000,60))
    improvements=[]
    for ref in ('M','T','G','H-D'):
        improvements.append([np.mean([r['completion_cost'] for r in rows if r['history_id']==h and r['method']==ref])-np.mean([r['completion_cost'] for r in rows if r['history_id']==h and r['method']=='H']) for h in histories])
    improvements=np.asarray(improvements).T;point=improvements.mean(0);boots=improvements[draws].mean(1)
    critical=float(np.quantile(np.max(boots-point,axis=1),.95));lower=point-critical
    comparisons={};h=summary(rows,'H');all_flags=[]
    for j,ref in enumerate(('M','T','G','H-D')):
        b=summary(rows,ref);same_direction=0;seeds=[]
        for s in (0,1,2):
            hv=np.mean([r['completion_cost'] for r in rows if r['method']=='H' and r['training_seed']==s])
            bv=np.mean([r['completion_cost'] for r in rows if r['method']==ref and (ref in ('G','T') or r['training_seed']==s)])
            seeds.append(float(bv-hv));same_direction+=int(hv<bv)
        flags=dict(cost10=h['cost']<=.9*b['cost'],simultaneous_lower_positive=lower[j]>0,
            capture_ci_preserved=stats['comparisons']['H-'+ref]['capture']['ci95'][0]>-.03,
            two_seeds=same_direction>=2,failures=h['failures']<=b['failures']+.02,illegal=h['illegal']<=b['illegal']+.02)
        comparisons[ref]=dict(checks={k:bool(v) for k,v in flags.items()},cost_gain=float(point[j]),simultaneous_lower=float(lower[j]),per_seed_cost_gain=seeds,reference=b)
        all_flags.append(all(flags.values()))
    lat=[v for r in rows if r['method']=='H' for v in r['latencies_ms']];p95=float(np.percentile(lat,95))
    result=dict(passed=bool(all(all_flags) and p95<=55),core_supported=bool(all(all_flags[:2])),whole_system_supported=bool(all(all_flags)),
        p95_ms=p95,time_target_met=p95<=55,comparisons=comparisons,simultaneous_critical=critical,candidate=h,
        statistical_scope='ordinary capture intervals and simultaneous four-comparison cost lower bounds; no joint guarantee across metric families')
    run.put(output/'evaluate/gate.json',result);run.report(output);return result


def dispatch(command,output,workers):
    authorize(output)
    if command=='collect':return collect(output,workers)
    if command=='train':return train(output)
    if command=='screen':return screen(output,workers)
    if command=='evaluate':return evaluate(output,workers)
    if command!='pipeline':raise ValueError(command)
    collect(output,workers);train(output);gate=screen(output,workers)
    if not gate['passed']:
        run.put(output/'status.json',dict(stage='stopped_screen_gate',terminal=True,pid=None,training_started=True,test_started=False));return
    result=evaluate(output,workers)
    run.put(output/'status.json',dict(stage='complete',terminal=True,pid=None,training_started=True,test_started=True,passed=result['passed'],deployment_changed=False))


def figures(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    folder=output/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,family in zip(axes,run.FAMILIES):
        e=next(e for e in run.entries() if e['family']==family and e['teacher_check']);packet,env=run.inputs(e);p=run.public_map(e);s=Support(p)
        for r in packet['history']:s.update(run.base.obs(r))
        ax.scatter(p.points[s.mask,0],p.points[s.mask,1],s=1,color='lightsteelblue',label='public reachable outer cells')
        for b in p.boxes:ax.add_patch(Rectangle((b[0,0],b[1,0]),b[0,1]-b[0,0],b[1,1]-b[1,0],color='0.3'))
        at=np.array(packet['history'][-1]['position']);path=np.array([r['position'] for r in packet['history']]);ax.plot(path[:,0],path[:,1],'k.-',label='observer history')
        branches,*_=s.branches()
        for f in branches:
            if f[11]>.5:ax.arrow(at[0],at[1],f[0],f[1],width=.035,color='tab:green')
        truth=np.array([w['handoff_full_state'][:2] for w in env['worlds']]);ax.scatter(truth[:,0],truth[:,1],marker='x',c='red',label='hidden truth (evaluation only)')
        ax.set(xlim=(-10,10),ylim=(-10,10),title=family,aspect='equal')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(folder/'public_support_branches.png',dpi=160);fig.savefig(folder/'public_support_branches.pdf');plt.close(fig)
    path=output/'teacher_check/results.json'
    if path.exists():
        rows=json.loads(path.read_text());fig,axes=plt.subplots(1,2,figsize=(8,3.5))
        for ax,field,title in zip(axes,('capture','completion_cost'),('Capture rate','Completion cost')):
            ax.bar(['T first action','Expert first action'],[np.mean([r[field] for r in rows if r['method']==m]) for m in ('T-first','E-first')],color=['0.5','tab:blue']);ax.set_title(title)
        fig.tight_layout();fig.savefig(folder/'teacher_first_action.png',dpi=160);plt.close(fig)
