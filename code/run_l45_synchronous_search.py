"""Fresh, gated evaluation of frozen l45 with synchronous continuous-action search."""
import os
for _var in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_var]='1'
import argparse
import contextlib
import copy
import fcntl
import io
import multiprocessing as mp
from pathlib import Path
import shutil
import threading
import time
import types
import numpy as np
import torch
torch.set_num_threads(1)
import run_l45_aq_once as base
import run_l45_deployment_consistency as research
from solvers.deployment_puct import DeploymentPUCT

o,s,v,f=base.o,base.s,base.v,base.f
ROOT,CODE=o.ROOT,o.CODE
SOURCE=research.DEFAULT
DEFAULT_OUTPUT=ROOT/'current/l45_synchronous_search_20261024'
SMOKE_OUTPUT=ROOT/'current/l45_synchronous_search_smoke_20261024'
lock_json,locked_read=base.lock_json,base.locked_read
SCIENCE=base.SCIENCE
SPLITS=base.SPLITS


def config(args):
    return dict(schema=1,smoke=args.smoke,training_episodes=0,teacher_queries=0,optimizer_updates=0,
        budget=3 if args.smoke else 1000,search_only_budget=3 if args.smoke else 3000,depth=100,
        candidate='C',rule='L_pursuit_synchronous_DUCT_joint_step_and_rollout; complete_O_evader',
        pursuer_betas=[.5,0.],evader_betas=[.5,.1],completion_bonus=0.,
        splits={n:[layout,1 if args.smoke else count] for n,(layout,count) in SPLITS.items()},
        scenario_namespace=5102026102401 if args.smoke else 5102026102402,
        bootstrap_seed=51020261024,bootstrap_draws=20000,
        comparisons=[dict(id='C-O',candidate='C',reference='O'),dict(id='C-L',candidate='C',reference='L')],
        request_caps=dict(validation=1350,confirmation=4500,evader_regression=1800,serial_timing=270),
        minimum_gain=.03,illegal_margin=.02,regression_margin=.03,no_automatic_expansion=True)


def fingerprints():
    paths=[Path(__file__),CODE/'L45_SYNCHRONOUS_SEARCH.md',CODE/'run_l45_aq_once.py']
    return dict(files={str(p):o.sha(p) for p in paths},inherited=research.fingerprints())


def prepare(args):
    out=args.output
    if out==ROOT or out==SOURCE or SOURCE in out.parents:raise ValueError('Use isolated output')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'experiment.lock.json').exists():verify(args);return locked_read(out/'scenarios.json')
    research.verify(research.parse_args([]))
    prior=locked_read(SOURCE/'results.json')
    if prior['historical_exact']!=99 or prior['recommended']!='old0:novalue':raise ValueError('Historical audit incomplete')
    source={str(SOURCE/name):o.sha(SOURCE/name) for name in ('experiment.json','results.json','replay.json')}
    for name in ('O','L'):
        dest=out/'models'/name;dest.mkdir(parents=True,exist_ok=True)
        for filename in o.MODEL_NAMES:
            origin=SOURCE/'models'/name/filename;source[str(origin)]=o.sha(origin)
            shutil.copyfile(origin,dest/filename)
    history=s.history_files(args);used=s.history_scenes(history);old_count=len(used);banks={}
    for phase,(name,(layout,count)) in enumerate(config(args)['splits'].items(),1900):
        cases=[];index=0
        while len(cases)<count:
            spec=o.make_scenario(o.scenario_seed(phase,config(args)['scenario_namespace'],index),layout);index+=1
            identity=s.previous.physical_hash(spec)
            if identity not in used:used.add(identity);cases.append(spec)
        banks[name]=cases
    record=dict(config=config(args),implementation=fingerprints(),source=source,
        history={str(p):o.sha(p) for p in history},models=registry(args))
    lock_json(out/'scenarios.json',banks)
    lock_json(out/'scenario_independence.json',dict(previous_physical_scenes=old_count,
        new_scenes=len(used)-old_count,overlap=0,history_reuse=False,all_phases_frozen=True))
    lock_json(out/'experiment.json',record)
    lock_json(out/'signature.json',dict(sha256=o.json_hash(dict(record=record,scenarios=o.sha(out/'scenarios.json')))))
    shutil.copyfile(CODE/'L45_SYNCHRONOUS_SEARCH.md',out/'PLAN.md')
    verify(args);return banks


def phase_pair(uid,role):
    if uid not in ('C','O','L'):raise ValueError('Unknown candidate')
    model='L' if uid=='L' or (uid=='C' and role==1) else 'O'
    return model,model


def make_own(problem,reg,uid,role,requested,seed):
    model=reg[phase_pair(uid,role)[0]]
    if o.model_fingerprint(model['models'])!=model['model_hash']:raise ValueError('Worker weights changed')
    paths,value_path=research.h.oracle_paths(str(ROOT),iteration=45,model_dir=model['models'])
    with contextlib.redirect_stdout(io.StringIO()):
        policies,value=research.h.get_oracles(problem,value_oracle_name='deterministic',value_oracle_path=value_path,
            policy_oracle_name='gaussian',policy_oracle_paths=paths)
    raw=DeploymentPUCT(policies,value,requested['search_budget'],requested['beta_policy'],
        requested['beta_value'],uid=='C' and role==1,requested['search_depth'])
    raw.solver_wrapper.set_seed(int(seed))
    masked=o.RoleMaskedSolver(raw,o.TEAMS[role])
    return masked


def effective(uid,role,reg):
    name=phase_pair(uid,role)[0]
    return dict(model=name,weights=reg[name]['model_hash'],
        algorithm='synchronous_DUCT' if uid=='C' and role==1 else 'legacy_PUCT')


def requests(args,uids,scenes,role=1,opponents=o.OPPONENTS,timing='parallel'):
    reg=registry(args);result=[]
    for uid in uids:
        behavior=effective(uid,role,reg)
        for spec in scenes:
            for opponent in opponents:
                identity=dict(protocol=o.sha(args.output/'experiment.json'),policy_id=o.json_hash(behavior),
                    episode_hash=spec['episode_hash'],role=role,opponent=opponent,timing_context=timing,
                    own_settings=settings(args,role),opponent_settings=settings(args,0),mode='search')
                key=o.json_hash(identity)
                task=dict(**identity,key=key,uid=uid,spec=spec,registry=reg,effective=behavior)
                result.append(dict(key=key,uid=uid,task=task))
    return result


def evaluate_one(task,legacy=False):
    p,state=o.restore_scenario(task['spec']);p.clear_closed_loop_cache()
    role,uid,reg=task['role'],task['uid'],task['registry']
    if legacy:
        if uid=='C':raise ValueError('No legacy C implementation')
        own=f.make_solver(p,reg[phase_pair(uid,role)[0]]['models'],'Search + Learning',role,
            task['own_settings'],task['spec']['search_seeds'][role])
        other=f.make_solver(p,reg['L']['models'],task['opponent'],1-role,
            task['opponent_settings'],task['spec']['search_seeds'][1-role])
    else:
        own=make_own(p,reg,uid,role,task['own_settings'],task['spec']['search_seeds'][role])
        other=base.private_solver(p,reg['L'],task['opponent'],1-role,
            task['opponent_settings'],task['spec']['search_seeds'][1-role])
    clock=time.perf_counter();state,stats,decisions,switching=simulate(p,state,own,other,role)
    row=metrics(p,state,stats)
    row.update({k:task[k] for k in ('key','policy_id','role','opponent','timing_context','mode')})
    row.update(uid=uid,episode_hash=task['spec']['episode_hash'],layout=task['spec']['layout'],
        search_seeds=task['spec']['search_seeds'],episode_wall_ms=(time.perf_counter()-clock)*1000.,
        actual_own=dict(task['own_settings'],effective=task['effective']),actual_opponent=other.actual_settings,
        switches=0,intervention_events=[],own_cost={} if legacy else own.solver.cost,
        own_mcts_calls=0 if legacy else own.solver.calls)
    if task['timing_context'] in ('warmup','single_process'):row['decision_latency_ms']=decisions
    return row


def read_cached(args,item):
    path=args.output/'cache'/(item['key']+'.json')
    if locked_read(args.output/'cache_checksums'/(item['key']+'.json'))['sha256']!=o.sha(path):raise ValueError('Cache file changed')
    row=o.read_json(path);digest=row.pop('content_sha256',None);task=item['task']
    if digest!=o.json_hash(row) or any(row[k]!=task[k] for k in ('key','policy_id','role','opponent','timing_context','mode')):
        raise ValueError('Cache identity changed')
    if row['episode_hash']!=task['spec']['episode_hash'] or row['search_seeds']!=task['spec']['search_seeds'] or \
       row['actual_own']!=dict(task['own_settings'],effective=task['effective']):raise ValueError('Unpaired scene/seed or role changed')
    requested=task['opponent_settings'];other=row['actual_opponent']
    if other['method']!=task['opponent']:raise ValueError('Opponent changed')
    if task['opponent']!='Policy Only':
        budget=requested['search_only_budget' if task['opponent']=='Search Only' else 'search_budget']
        beta=[0.,0.] if task['opponent']=='Search Only' else [requested['beta_policy'],requested['beta_value']]
        if other['budget']!=budget or other['depth']!=100 or not np.allclose([other['beta_policy'],other['beta_value']],beta):
            raise ValueError('Opponent search configuration changed')
    row['uid']=item['uid'];return row


def comparison(rows,reference,confidence=None):
    four=s.point_comparison(rows,'C',reference,'four');none=s.point_comparison(rows,'C',reference,'none')
    checks=s.performance_checks(four,none,gain=True)
    checks['no_obstacle_own_illegal']=none['own_illegal_delta']<=.02+1e-12
    if confidence is not None:checks['simultaneous_lower_positive']=confidence['simultaneous_lower95']>0
    return dict(four=four,none=none,checks=checks,passed=all(checks.values()))


def statistics(rows,draws=20000):
    indexed={uid:v.indexed_rows(rows,uid,'four',1) for uid in ('C','O','L')}
    keys=sorted(indexed['C']);scenes=sorted({k[0] for k in keys})
    if any(set(x)!=set(keys) for x in indexed.values()):raise ValueError('Missing statistical pairs')
    for scene in scenes:
        if {op for sc,op in keys if sc==scene}!=set(o.OPPONENTS):raise ValueError('Missing opponent stratum')
    differences=[];output=[]
    for ref in ('O','L'):
        if any(indexed['C'][k]['search_seeds']!=indexed[ref][k]['search_seeds'] for k in keys):raise ValueError('Statistical seed mismatch')
        delta={k:indexed['C'][k]['legal_double_capture']-indexed[ref][k]['legal_double_capture'] for k in keys}
        differences.append([np.mean([delta[k] for k in keys if k[0]==scene]) for scene in scenes])
        output.append(dict(id='C-'+ref,candidate='C',reference=ref,delta=float(np.mean(list(delta.values()))),
            n_scenes=len(scenes),n_games=len(keys),gained=sum(x>0 for x in delta.values()),lost=sum(x<0 for x in delta.values())))
    differences=np.asarray(differences);observed=differences.mean(1)
    rng=np.random.default_rng(51020261024);samples=np.empty((draws,2))
    for start in range(0,draws,256):
        stop=min(start+256,draws);indices=rng.integers(len(scenes),size=(stop-start,len(scenes)))
        samples[start:stop]=differences[:,indices].mean(-1).T
    correction=max(0.,float(np.quantile(np.max(samples-observed,axis=1),.95)))
    for i,row in enumerate(output):row.update(ci95=np.quantile(samples[:,i],[.025,.975]).tolist(),simultaneous_lower95=float(observed[i]-correction))
    return dict(comparisons=output,draws=draws,bootstrap_seed=51020261024,correction=correction,
        method='physical_scene_paired; max_centered_joint_one_sided_95')


def validate(args,banks,pool,status):
    rows=run_evaluation(args,'validation',requests(args,['C','O','L'],banks['validation_four']+banks['validation_none']),pool,status)
    comparisons={ref:comparison(rows,ref) for ref in ('O','L')}
    result=lock_json(args.output/'validation_result.json',dict(passed=all(x['passed'] for x in comparisons.values()) and not args.smoke,
        comparisons=comparisons,aggregates={uid:v.aggregate(rows,uid) for uid in ('C','O','L')}))
    if result['passed']:lock_json(args.output/'candidate_lock.json',dict(candidate='C',config=config(args),
        models=registry(args),implementation=fingerprints(),validation_sha256=o.sha(args.output/'validation_result.json')))
    return result


def check_candidate(args):
    if not locked_read(args.output/'validation_result.json')['passed']:raise ValueError('Validation failed; confirmation forbidden')
    expected=dict(candidate='C',config=config(args),models=registry(args),implementation=fingerprints(),
        validation_sha256=o.sha(args.output/'validation_result.json'))
    if locked_read(args.output/'candidate_lock.json')!=expected:raise ValueError('Candidate lock changed')


def confirm(args,banks,pool,status):
    check_candidate(args)
    rows=run_evaluation(args,'confirmation',requests(args,['C','O','L'],banks['confirm_four']+banks['confirm_none']),pool,status)
    stats=statistics(rows);comparisons={x['reference']:comparison(rows,x['reference'],x) for x in stats['comparisons']}
    return lock_json(args.output/'confirmation_result.json',dict(passed=all(x['passed'] for x in comparisons.values()),statistics=stats,comparisons=comparisons))


def guard(args,banks,pool,status):
    check_candidate(args)
    if not locked_read(args.output/'confirmation_result.json')['passed']:raise ValueError('Confirmation failed; regression forbidden')
    req=requests(args,['C','O','L'],banks['guard_four']+banks['guard_none'],role=0)
    rows=run_evaluation(args,'evader_regression',req,pool,status)
    comparisons={ref:{layout:s.point_comparison(rows,'C',ref,layout,role=0) for layout in ('four','none')} for ref in ('O','L')}
    passed=all(x['delta']>=-.03-1e-12 and x['own_illegal_delta']<=.02+1e-12 for part in comparisons.values() for x in part.values())
    return lock_json(args.output/'guard_result.json',dict(passed=passed,comparisons=comparisons,
        evader_complete_O=True,logical_requests=len(req),unique_requests=len({i['key'] for i in req})))


def timing(args,banks,status):
    if not locked_read(args.output/'guard_result.json')['passed']:raise ValueError('Regression failed; timing forbidden')
    if mp.active_children():raise ValueError('Exit parallel pool before timing')
    if (args.output/'timing_result.lock.json').exists():return locked_read(args.output/'timing_result.json')
    for item in requests(args,['C','O','L'],banks['warmup'],timing='warmup'):evaluate_one(item['task'])
    req=[]
    for i,spec in enumerate(banks['timing_four']):
        order=['C','O','L'];order=order[i%3:]+order[:i%3]
        for op in o.OPPONENTS:req+=requests(args,order,[spec],opponents=(op,),timing='single_process')
    rows=run_evaluation(args,'serial_timing',req,None,status);result={}
    for uid in ('C','O','L'):
        part=[r for r in rows if r['uid']==uid]
        decisions=np.concatenate([np.asarray(r['decision_latency_ms'])[:,0] for r in part])
        result[uid]=dict(pursuer_mean_ms=float(decisions.mean()),pursuer_p95_ms=float(np.percentile(decisions,95)),
            episode_mean_ms=float(np.mean([r['episode_wall_ms'] for r in part])),
            episode_p95_ms=float(np.percentile([r['episode_wall_ms'] for r in part],95)),episodes=len(part),
            own_cost={key:sum(r['own_cost'][key] for r in part) for key in part[0]['own_cost']})
    for row in result.values():row['ratio_to_O']=row['pursuer_mean_ms']/result['O']['pursuer_mean_ms']
    return lock_json(args.output/'timing_result.json',result)


def audit_one(task):
    traces=[];metrics_seen=[]
    for legacy in (True,False):
        p,state=o.restore_scenario(task['spec']);reg=task['registry'];role=task['role']
        if legacy:
            own=f.make_solver(p,reg['O']['models'],'Search + Learning',role,task['own_settings'],task['spec']['search_seeds'][role])
            other=f.make_solver(p,reg['L']['models'],task['opponent'],1-role,task['opponent_settings'],task['spec']['search_seeds'][1-role])
        else:
            own=make_own(p,reg,'O',role,task['own_settings'],task['spec']['search_seeds'][role])
            other=base.private_solver(p,reg['L'],task['opponent'],1-role,task['opponent_settings'],task['spec']['search_seeds'][1-role])
        trace=[];state,stats,_,_=simulate(p,state,own,other,role,trace=trace)
        traces.append(o.json_hash(trace));metrics_seen.append(science(metrics(p,state,stats)))
    if traces[0]!=traces[1] or metrics_seen[0]!=metrics_seen[1]:raise ValueError('Original full episode parity failed')
    return dict(layout=task['spec']['layout'],role=task['role'],opponent=task['opponent'],exact=True)


def audit(args,banks,status):
    if (args.output/'audit_result.lock.json').exists():return locked_read(args.output/'audit_result.json')
    tasks=[i['task'] for role in (0,1) for layout in ('four','none') for i in requests(args,['O'],banks['audit_'+layout],role)]
    probes=[]
    with mp.get_context('spawn').Pool(min(args.workers,12)) as pool:
        for row in pool.imap_unordered(audit_one,tasks):
            probes.append(row);status('audit',completed=len(probes),total=len(tasks))
    return lock_json(args.output/'audit_result.json',dict(passed=True,original_complete_episode_pairs=probes,
        sync_engineering_evidence=str(SOURCE/'audit_result.json'),no_actor_updates=True))


def load_recommended_solver(args,problem,role,seed):
    rec=locked_read(args.output/'recommended.json')
    if rec['implementation']!=fingerprints():raise ValueError('Recommendation implementation changed')
    for path,digest in rec['runtime_fingerprints'].items():
        if o.sha(path)!=digest:raise ValueError('Exported runtime changed')
    requested=rec['pursuer_settings'] if role else rec['evader_settings']
    return make_own(problem,rec['registry'],rec['uid'],role,requested,seed)


def replay(args,banks):
    rec=locked_read(args.output/'recommended.json');probes=[]
    for role in (0,1):
        spec=banks['warmup'][0];p,state=o.restore_scenario(spec)
        for phase in (0,1):
            current=state.copy()
            if phase:current[[p.active_idxs[0],p.active_idxs[2]],0]=0.
            a=make_own(p,registry(args),rec['uid'],role,settings(args,role),spec['search_seeds'][role])
            b=load_recommended_solver(args,p,role,spec['search_seeds'][role])
            if not np.array_equal(a.policy(p,current),b.policy(p,current)):raise ValueError('Export action replay failed')
            probes.append(dict(role=role,phase=phase,exact=True))
    original=requests(args,[rec['uid']],banks['warmup'],opponents=('Search Only',))[0]['task']
    exported=copy.deepcopy(original);exported['registry']=rec['registry']
    if science(evaluate_one(original))!=science(evaluate_one(exported)):raise ValueError('Export full episode failed')
    return lock_json(args.output/'replay.json',dict(passed=True,probes=probes,full_episode_exact=True,uid=rec['uid']))


def finish(args,banks):
    verify(args)
    validation,confirmation,regression,measurement=[optional(args,x) for x in ('validation_result','confirmation_result','guard_result','timing_result')]
    if validation is None:raise ValueError('Validation incomplete')
    for before,after in ((validation,confirmation),(confirmation,regression),(regression,measurement)):
        if before and before['passed'] and after is None:raise ValueError('Required downstream stage incomplete')
        if after is not None and (not before or not before['passed']):raise ValueError('Downstream without prerequisite')
    accepted=bool(regression and regression['passed'] and measurement and not args.smoke);uid='C' if accepted else 'O'
    exported={}
    for name,origin in registry(args).items():
        dest=args.output/'recommended/models'/name;dest.mkdir(parents=True,exist_ok=True)
        for filename in o.MODEL_NAMES:
            if not (dest/filename).exists():shutil.copyfile(Path(origin['models'])/filename,dest/filename)
        if o.model_fingerprint(dest)!=origin['model_hash']:raise ValueError('Exported weights changed')
        exported[name]=dict(models=str(dest),model_hash=origin['model_hash'])
    runtime={}
    for origin in [Path(__file__),CODE/'solvers/deployment_puct.py',CODE/'cpp/solvers/simultaneous_puct.hpp',
                   CODE/'cpp/build_deployment/build_manifest.json',*list((CODE/'cpp/build_deployment').glob('*.so'))]:
        dest=args.output/'recommended/runtime'/origin.name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(origin,dest)
        if o.sha(dest)!=o.sha(origin):raise ValueError('Runtime export changed')
        runtime[str(dest)]=o.sha(dest)
    lock_json(args.output/'recommended.json',dict(uid=uid,registry=exported,
        historical_uid=None if accepted else 'old0:novalue',pursuer_model='L' if accepted else 'O',evader_model='O',
        pursuer_search='synchronous_DUCT' if accepted else 'legacy_PUCT',evader_search='legacy_PUCT',
        pursuer_settings=settings(args),evader_settings=settings(args,0),quality_gate_passed=accepted,
        runtime_fingerprints=runtime,implementation=fingerprints(),loader='run_l45_synchronous_search.load_recommended_solver'))
    reason='smoke_only' if args.smoke else 'adopted' if accepted else 'validation_no_gain' if not validation['passed'] else \
        'confirmation_failed' if not confirmation['passed'] else 'guard_failed'
    result=dict(reason=reason,recommended=uid,quality_gate_passed=accepted,validation=validation,
        confirmation=confirmation,guard=regression,timing=measurement,training_episodes=0,teacher_queries=0,
        optimizer_updates=0,smoke=args.smoke,replay=replay(args,banks),continue_formal=False)
    lock_json(args.output/'results.json',result);report(args,banks,result);return result


def report(args,banks,result):
    rows=[read_cached(args,i) for i in requests(args,['C','O','L'],banks['validation_four']+banks['validation_none'])]
    lines=['冻结 l45 同步搜索新场景实验','',f"结束原因：{result['reason']}；推荐：{result['recommended']}。",
        'C 追捕用 L 的冻结权重及同步搜索，逃逸完整沿用 O；无新训练、教师标签或优化器更新。',
        '参考和三个对手保持原搜索。同步节点决策、转移、rollout 一起改变，不作单因素归因。','',
        '|配置|四障碍合法双捕|无障碍合法双捕|','|---|---:|---:|']
    for uid in ('C','O','L'):
        rates=[v.aggregate(rows,uid,layout)['success_rate'] for layout in ('four','none')]
        lines.append(f'|{uid}|{rates[0]:.2%}|{rates[1]:.2%}|')
    for ref,row in result['validation']['comparisons'].items():
        lines+=['',f"C 对 {ref}：四障碍 {row['four']['delta']*100:+.2f} 点；无障碍 {row['none']['delta']*100:+.2f} 点；门槛 {row['passed']}。",
            '四障碍逐对手增益：'+str(row['four']['per_opponent_delta']),
            '失败项：'+str([k for k,value in row['checks'].items() if not value])]
    for name in ('confirmation','guard'):lines+=['',name+'：'+('未执行' if result[name] is None else '通过' if result[name]['passed'] else '失败')]
    if result['confirmation']:lines+=['',str(result['confirmation']['statistics'])]
    lines+=['','单进程计时：'+str(result['timing']) if result['timing'] else '单进程计时：未执行，未知。',
        '相同模拟次数不代表环境调用或耗时相同；逐局成本见 cache，不根据并行吞吐宣称加速。',
        '任何效果门槛失败则保留 O；不改用其他同步组合、不扩大样本。单固定模型结果不代表跨种子稳定。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='pipeline',choices=['pipeline','prepare','audit','validate','confirm','guard','timing','report','replay','watch'])
    p.add_argument('--workers',type=int,default=80);p.add_argument('--output',type=Path)
    p.add_argument('--smoke',action='store_true');p.add_argument('--once',action='store_true')
    args=p.parse_args(argv);args.output=(args.output or (SMOKE_OUTPUT if args.smoke else DEFAULT_OUTPUT)).resolve()
    if args.workers<1:p.error('workers must be positive')
    return args


# Bind only generic, unchanged infrastructure to this experiment's namespace.
# No mutation of the historical module or its pipeline globals occurs.
for _name in ('settings','registry','verify','other_rng','initial_stats','advance','metrics','science',
              'simulate','run_evaluation','optional','watch'):
    _method=getattr(base,_name)
    globals()[_name]=types.FunctionType(_method.__code__,globals(),_name,_method.__defaults__,_method.__closure__)


def main(argv=None):
    args=parse_args(argv)
    if args.stage=='watch':return watch(args)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(pid=os.getpid(),phase='prepare',workers=args.workers,started=time.time())
        mutex=threading.Lock();stop=threading.Event()
        def status(phase,**fields):
            with mutex:
                if phase!=state['phase']:
                    state.pop('completed',None);state.pop('total',None);state['stage_started']=time.time()
                state.update(phase=phase,updated_unix=time.time(),**fields)
                children=[p.pid for p in mp.active_children() if p.is_alive()]
                threads={str(pid):len(list(Path('/proc',str(pid),'task').glob('*'))) for pid in children}
                state.update(actual_workers=len(children),worker_pids=children,worker_threads=threads,
                    committed_cache=len(list((args.output/'cache_checksums').glob('*.lock.json'))))
                o.write_json(args.output/'status.json',state)
        def heartbeat():
            while not stop.wait(5):
                with mutex:phase=state['phase']
                status(phase)
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:
            banks=prepare(args);status('prepared')
            if args.stage=='prepare':return
            if args.stage in ('pipeline','audit'):audit(args,banks,status)
            if args.stage=='audit':status('complete',reason='engineering_passed');return
            if not locked_read(args.output/'audit_result.json')['passed']:raise ValueError('Audit required')
            if args.stage=='confirm':check_candidate(args)
            if args.stage=='guard' and not locked_read(args.output/'confirmation_result.json')['passed']:
                raise ValueError('Confirmation failed; regression forbidden')
            if args.stage=='timing' and not locked_read(args.output/'guard_result.json')['passed']:
                raise ValueError('Regression failed; timing forbidden')
            if args.stage in ('pipeline','validate','confirm','guard'):
                with mp.get_context('spawn').Pool(args.workers) as pool:
                    if args.stage in ('pipeline','validate'):validation=validate(args,banks,pool,status)
                    else:validation=locked_read(args.output/'validation_result.json')
                    if args.stage in ('pipeline','confirm') and validation['passed']:
                        confirmation=confirm(args,banks,pool,status)
                    else:confirmation=optional(args,'confirmation_result')
                    if args.stage in ('pipeline','guard') and confirmation and confirmation['passed']:guard(args,banks,pool,status)
                if args.stage!='pipeline':status('complete',reason=args.stage+'_complete');return
            if args.stage in ('pipeline','timing') and optional(args,'guard_result') and optional(args,'guard_result')['passed']:timing(args,banks,status)
            if args.stage=='replay':replay(args,banks);status('complete',reason='replay_passed');return
            result=finish(args,banks)
            status('complete',reason=result['reason'],recommended=result['recommended'],elapsed_seconds=time.time()-state['started'])
        except BaseException as error:status('failed',reason=type(error).__name__+': '+str(error));raise
        finally:stop.set();thread.join(timeout=1)


if __name__=='__main__':main()
