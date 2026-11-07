"""Bounded deployment-consistency and scene-grouped action-scoring experiment."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import contextlib
import copy
import fcntl
import io
import multiprocessing as mp
from pathlib import Path
import platform
import shutil
import threading
import time
import numpy as np
import torch
torch.set_num_threads(1)
import run_l45_root_action_audit as r
from solvers.deployment_puct import DeploymentPUCT
from learning import deployment_scorer as scorer

o,h,s=r.o,r.h,r.s
ROOT,CODE=o.ROOT,o.CODE
DEFAULT=ROOT/'current/l45_deployment_consistency_20261024'
SMOKE=ROOT/'current/l45_deployment_consistency_smoke_20261024'
SOURCE=r.DEFAULT
lock_json,read_locked=r.lock_json,r.locked_read


def config(args):
    return dict(schema=1,smoke=args.smoke,budget=8 if args.smoke else 1000,
        opponent_budget=12 if args.smoke else 3000,depth=100,pursuer_betas=[.5,0.],
        opponent='Search Only',opponent_betas=[0.,0.],actor_updates=0,ppo_updates=0,
        new_teacher_queries=0,scorer='closed_form_ridge_or_stump; nested_5x4_scene_folds',
        scorer_methods=list(scorer.METHODS),thresholds=[0.,.05,.1,.2,'never'],
        minimum_gain=.03,require_both_pairwise_ci_positive=True,bootstrap_draws=20000,
        bootstrap_seed=scorer.SEED,source_scenes=99,continuation_requests=8 if args.smoke else 396,
        variants=['O_legacy','O_sync','L_legacy','L_sync'],
        sync='decoupled_UCT_progressive_widening; independent_same_state_selection; joint_step',
        sync_rollout='joint_safe_random; depth_minus_tree_path; physical_terminal_80',
        primary='L_legacy versus L_sync; conditional on historical O first-capture states',
        decision='exploratory_only; no_adoption_or_confirmation; stop_scorer_on_failed_feasibility',
        recommended='old0:novalue')


def fingerprints():
    paths=[Path(__file__),CODE/'learning/deployment_scorer.py',CODE/'solvers/deployment_puct.py',
        CODE/'cpp/solvers/simultaneous_puct.hpp',CODE/'build_deployment_search.py',
        CODE/'cpp/build_deployment/build_manifest.json']
    binaries=list((CODE/'cpp/build_deployment').glob('*.so'))
    if len(binaries)!=1:raise ValueError('Expected one private deployment extension')
    manifest=o.read_json(paths[-1])
    for name,value in manifest['generated'].items():
        if o.sha(CODE/'cpp/build_deployment/source'/name)!=value:raise ValueError('Private source changed')
    if manifest['binary']!=o.sha(binaries[0]):raise ValueError('Private binary changed')
    if manifest['builder']!=o.sha(CODE/'build_deployment_search.py') or \
       manifest['synchronous_source']!=o.sha(CODE/'cpp/solvers/simultaneous_puct.hpp'):
        raise ValueError('Rebuild private extension after source changes')
    return dict(files={str(p):o.sha(p) for p in paths+binaries},inherited=r.fingerprints(),
        platform=[platform.platform(),platform.python_version(),np.__version__,torch.__version__])


def prepare(args):
    if args.output==ROOT or args.output==SOURCE or SOURCE in args.output.parents:
        raise ValueError('Use an isolated output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'experiment.lock.json').exists():return verify(args)
    r.verify(r.parse_args([]))
    roots=read_locked(SOURCE/'extract/manifest.json')
    if len(roots)!=99:raise ValueError('Missing historical roots')
    history=s.history_files(args);used=s.history_scenes(history);historical=len(used)
    scenes={}
    for i,layout in enumerate(('four','none')):
        index=0
        while True:
            spec=o.make_scenario(o.scenario_seed(1800+i,50020261024,index),layout);index+=1
            physical=s.previous.physical_hash(spec)
            if physical not in used:used.add(physical);scenes[layout]=[spec];break
    lock_json(args.output/'scenarios.json',scenes)
    for name in ('O','L'):
        dest=args.output/'models'/name;dest.mkdir(parents=True,exist_ok=True)
        for model in o.MODEL_NAMES:shutil.copyfile(SOURCE/'models'/name/model,dest/model)
    source_paths=[SOURCE/'experiment.json',SOURCE/'bank.json',SOURCE/'extract/manifest.json',
        SOURCE/'branch/manifest.json',SOURCE/'paired_results.json']
    root_records=[]
    for task in roots:
        record=r.read_task(task)
        if len(record['candidates'])!=4:raise ValueError('Expected complete four-candidate source')
        root_records.append(dict(task=task,sha256=o.sha(task['result'])))
    if args.smoke:
        root_records=[]
        reg={n:dict(models=str(args.output/'models'/n),model_hash=o.model_fingerprint(args.output/'models'/n)) for n in ('O','L')}
        for layout,bank in scenes.items():
            spec=bank[0];p,state=o.restore_scenario(spec)
            state[[p.active_idxs[0],p.active_idxs[2]],0]=0.;state[p.time_idx,0]=77.
            own=make_solver(p,reg['O'],config(args),False,1,spec['search_seeds'][1])
            other=make_solver(p,reg['L'],config(args),False,0,spec['search_seeds'][0],True)
            own_before=own.get_rng_state(p);opponent_before=other.get_rng_state(p)
            own.policy(p,state,team=o.TEAMS[1])
            stats=h.initial_stats();stats.update(steps=77,first=77.,last=77.,capture_count=1)
            table=np.asarray(own.last_result.root_audit).copy()
            candidates=r.selected_candidates(table,p,state)
            root=dict(state=state,stats=stats,own_before=own_before,opponent_before=opponent_before,
                python_rng=o.rng_state(),root_table=table,candidates=candidates,
                task=dict(row=dict(spec=spec)))
            path=args.output/'smoke_roots'/(layout+'.pt');r.commit_pt(path,root)
            root_records.append(dict(task=dict(result=str(path),row=dict(spec=spec)),sha256=o.sha(path)))
    record=dict(config=config(args),implementation=fingerprints(),roots=root_records,
        models={n:dict(models=str(args.output/'models'/n),model_hash=o.model_fingerprint(args.output/'models'/n)) for n in ('O','L')},
        source={str(p):o.sha(p) for p in source_paths},history={str(p):o.sha(p) for p in history},
        scenarios_sha=o.sha(args.output/'scenarios.json'),historical_physical_scenes=historical)
    lock_json(args.output/'experiment.json',record)
    lock_json(args.output/'scenario_independence.json',dict(history_scenes=historical,new_engineering=2,
        overlap=0,diagnostic_reuse='all 99 recoverable O first-capture branches; explicit historical exception'))
    (args.output/'PLAN.md').write_text(PLAN)
    return verify(args)


def verify(args):
    record=read_locked(args.output/'experiment.json')
    if record['config']!=config(args) or record['implementation']!=fingerprints():
        raise ValueError('Protocol, implementation or platform changed')
    for path,digest in (record['source']|record['history']).items():
        if o.sha(path)!=digest:raise ValueError('Historical source changed: '+path)
    if o.sha(args.output/'scenarios.json')!=record['scenarios_sha']:raise ValueError('Scene bank changed')
    for model in record['models'].values():
        if o.model_fingerprint(model['models'])!=model['model_hash']:raise ValueError('Frozen model changed')
    for root in record['roots']:
        if o.sha(root['task']['result'])!=root['sha256']:raise ValueError('Historical root changed')
    return record


def make_solver(p,model,cfg,synchronous,role,seed=1,pure=False):
    if pure:policies,value=[None]*4,None
    else:
        paths,value_path=h.oracle_paths(str(ROOT),iteration=45,model_dir=model['models'])
        with contextlib.redirect_stdout(io.StringIO()):
            policies,value=h.get_oracles(p,value_oracle_name='deterministic',value_oracle_path=value_path,
                policy_oracle_name='gaussian',policy_oracle_paths=paths)
    solver=DeploymentPUCT(policies,value,cfg['opponent_budget'] if pure else cfg['budget'],
        0. if pure else .5,0. if pure or role else .1,synchronous,depth=cfg['depth'])
    solver.solver_wrapper.set_seed(int(seed));return solver


def load_root(item):
    if o.sha(item['task']['result'])!=item['sha256']:raise ValueError('Root fingerprint changed')
    return r.read_pt(item['task']['result'])


def items(args,record,kind):
    result=[]
    for root in record['roots']:
        for variant in (config(args)['variants'] if kind=='continuation' else ['semantic']):
            task=dict(kind=kind,variant=variant,root=root,config=config(args),models=record['models'],
                protocol=o.sha(args.output/'experiment.json'))
            task['key']=o.json_hash(task);task['result']=str(args.output/kind/(task['key']+'.pt'))
            result.append(task)
    return result


def read_task(task):
    row=r.read_pt(task['result'])
    if row['key']!=task['key'] or row['task']!={k:v for k,v in task.items() if k!='result'}:
        raise ValueError('Cache identity changed')
    return row


def semantic_one(task):
    root=load_root(task['root']);spec=root['task']['row']['spec']
    p,_=o.restore_scenario(spec);state=root['state'].copy()
    if p.active_evader_count(state)!=1 or p.active_pursuer_count(state)!=1:
        raise ValueError('Phase/capture markers disagree')
    other=make_solver(p,task['models']['L'],task['config'],False,0,pure=True)
    other.set_rng_state(p,root['opponent_before']);o.restore_rng(root['python_rng'])
    evader=other.policy(p,state,team=o.TEAMS[0])
    own=make_solver(p,task['models']['O'],task['config'],False,1)
    wrapper=own._problem_wrapper_for(p)
    records=[]
    # Evaluate every root arm, not only winners or failed trajectories.
    for row in np.asarray(root['root_table']):
        pursuit=row[6:].astype(float).reshape(-1,1)
        for robot in o.TEAMS[0]:pursuit[p.action_idxs[robot]]=0.
        joint=pursuit+evader
        actual=p.step(state,joint,p.dt)
        cpp=np.asarray(wrapper.step(state,joint,p.dt)).reshape(-1,1)
        positions=np.concatenate([p.state_idxs[i] for i in range(4)])
        if not np.allclose(actual[positions],cpp[positions],rtol=0,atol=5e-5) or \
           not np.array_equal(actual[p.active_idxs],cpp[p.active_idxs]) or \
           float(actual[p.time_idx,0])!=float(cpp[p.time_idx,0]):
            raise ValueError('C++/Python transition or capture mismatch')
        alone=np.asarray(wrapper.step(state,pursuit,p.dt)).reshape(-1,1)
        # The alternative order also advances two full ticks; it is diagnostic,
        # not a proposed implementation of a simultaneous transition.
        first=np.asarray(wrapper.step(state,evader,p.dt)).reshape(-1,1)
        alternating=np.asarray(wrapper.step(first,pursuit,p.dt)).reshape(-1,1)
        records.append(dict(ordinal=int(row[0]),selected=bool(row[5]),
            joint_capture=2-p.active_evader_count(actual),alone_capture=2-p.active_evader_count(alone),
            alternating_capture=2-p.active_evader_count(alternating),
            joint_illegal=list(map(bool,o.violations(p,actual))),alone_illegal=list(map(bool,o.violations(p,alone))),
            position_delta=float(np.max(np.abs(actual[positions]-alone[positions]))),
            joint_time=float(actual[p.time_idx,0]),alternating_time=float(alternating[p.time_idx,0])))
    return dict(key=task['key'],task={k:v for k,v in task.items() if k!='result'},records=records,
        physical_hash=s.previous.physical_hash(spec),root_state=state,evader_action=evader)


def continuation_one(task):
    root=load_root(task['root']);spec=root['task']['row']['spec'];p,_=o.restore_scenario(spec)
    name,mode=task['variant'].split('_')
    own=make_solver(p,task['models'][name],task['config'],mode=='sync',1)
    other=make_solver(p,task['models']['L'],task['config'],False,0,pure=True)
    own.set_rng_state(p,root['own_before']);other.set_rng_state(p,root['opponent_before'])
    o.restore_rng(root['python_rng'])
    state=root['state'].copy();stats=copy.deepcopy(root['stats']);trace=[];latencies=[]
    start=time.perf_counter()
    while not p.is_terminal(state):
        evader=other.policy(p,state,team=o.TEAMS[0])
        clock=time.perf_counter();pursuit=own.policy(p,state,team=o.TEAMS[1])
        latencies.append((time.perf_counter()-clock)*1000.)
        action=evader+pursuit;nxt=h.advance(p,state,action,stats)
        trace.append(dict(state=state.tolist(),action=action.tolist(),next_state=nxt.tolist()));state=nxt
        if stats['steps']>80:raise ValueError('Remaining horizon was reset')
    return dict(key=task['key'],task={k:v for k,v in task.items() if k!='result'},metrics=h.metrics(p,state,stats),
        state=state,trajectory=trace,own_rng=own.get_rng_state(p),opponent_rng=other.get_rng_state(p),
        own_cost=own.cost,other_cost=other.cost,own_calls=own.calls,other_calls=other.calls,
        decision_latency_ms=latencies,wall_ms=(time.perf_counter()-start)*1000.,
        physical_hash=s.previous.physical_hash(spec))


def run_tasks(args,stage,tasks,worker,pool,status):
    lock_json(args.output/stage/'manifest.json',tasks)
    missing=[];done=0
    for task in tasks:
        if Path(task['result']).with_suffix('.lock.json').exists():read_task(task);done+=1
        else:missing.append(task)
    status(stage,completed=done,total=len(tasks),cached=done)
    by_key={t['key']:t for t in tasks}
    iterator=pool.imap_unordered(worker,missing,chunksize=1) if pool else map(worker,missing)
    for row in iterator:
        r.commit_pt(by_key[row['key']]['result'],row);done+=1
        status(stage,completed=done,total=len(tasks),cached=len(tasks)-len(missing))
    return [read_task(task) for task in tasks]


def fixture(layout='none'):
    spec=o.make_scenario(500202610,layout);p,state=o.restore_scenario(spec)
    state[[p.active_idxs[0],p.active_idxs[2]],0]=0.;state[p.time_idx,0]=20.
    step=.6*min(p.action_lims[p.action_idxs[1][0],1],p.action_lims[p.action_idxs[3][0],1])
    state[p.state_idxs[3],0]=[0.,0.];state[p.state_idxs[1],0]=[p.desired_distance+.5*step,0.]
    joint=np.zeros((8,1));joint[p.action_idxs[1][0],0]=step;joint[p.action_idxs[3][0],0]=step
    pursuit=joint.copy();pursuit[p.action_idxs[1]]=0.
    return p,state,joint,pursuit


def audit(args,record):
    path=args.output/'audit_result.json'
    if path.with_suffix('.lock.json').exists():return read_locked(path)
    probes=[]
    for layout,scenes in read_locked(args.output/'scenarios.json').items():
        for role in (0,1):
            for phase in (0,1):
                p,state=o.restore_scenario(scenes[0])
                if phase:state[[p.active_idxs[0],p.active_idxs[2]],0]=0.
                # Full legacy action/root-stat/RNG parity at the configured budget.
                old=r.make_solver(p,record['models']['O'],'Search + Learning',role,
                    dict(budget=config(args)['budget'],opponent_budget=config(args)['opponent_budget']),True,73)
                new=make_solver(p,record['models']['O'],config(args),False,role,73)
                before=old.get_rng_state(p);new.set_rng_state(p,before)
                a=old.policy(p,state,team=o.TEAMS[role]);b=new.policy(p,state,team=o.TEAMS[role])
                if not np.array_equal(a,b) or old.get_rng_state(p)!=new.get_rng_state(p) or \
                   not np.array_equal(np.asarray(old.last_result.root_audit),np.asarray(new.last_result.root_audit)):
                    raise ValueError('Instrumented legacy changed search or RNG')
                sync=make_solver(p,record['models']['O'],config(args),True,role,73)
                initial=sync.get_rng_state(p);a=sync.policy(p,state,team=o.TEAMS[role]);after=sync.get_rng_state(p)
                sync.set_rng_state(p,initial);b=sync.policy(p,state,team=o.TEAMS[role])
                if not np.array_equal(a,b) or sync.get_rng_state(p)!=after:raise ValueError('Synchronous RNG restore changed action')
                for robot in range(4):
                    if robot not in o.TEAMS[role] or not p.is_active(state,robot):
                        if np.any(a[p.action_idxs[robot]]!=0):raise ValueError('Inactive/wrong-role action')
                probes.append(dict(layout=layout,role=role,phase=phase,legacy_exact=True,sync_restore=True,
                    step_calls=int(sync.last_result.step_calls),policy_evaluations=int(sync.last_result.policy_evaluations)))
    p,state,joint,pursuit=fixture()
    if not p.is_valid(state) or p.active_evader_count(p.step(state,joint,p.dt))!=1 or \
       p.active_evader_count(p.step(state,pursuit,p.dt))!=0:raise ValueError('Constructive semantic counterexample failed')
    value=dict(passed=True,probes=probes,counterexample=dict(valid=True,joint_capture=False,pursuit_only_capture=True),
        no_actor_updates=True,legacy_sources_unchanged=True)
    return lock_json(path,value)


def dataset(args,record):
    labels=read_locked(SOURCE/'paired_results.json');by_key={}
    for label in labels:
        key=(label['episode_hash'],label['candidate'],label['stream'])
        if key in by_key:raise ValueError('Duplicate paired label')
        by_key[key]=label
    if len(by_key)!=99*4*3:raise ValueError('Missing source pairing')
    # Check atomic suffix records against the protected flat label file.
    manifest=read_locked(SOURCE/'branch/manifest.json')
    if len(manifest)!=len(labels):raise ValueError('Missing suffix manifest')
    for task in manifest:
        row=r.read_task(task);key=(task['root_task']['row']['spec']['episode_hash'],task['candidate'],task['stream'])
        if row['metrics']['legal_double_capture']!=by_key[key]['legal_double_capture'] or \
           row['metrics']['pursuer_illegal']!=by_key[key]['pursuer_illegal']:
            raise ValueError('Flat labels differ from committed suffixes')
    result=[]
    for root_record in record['roots']:
        root=load_root(root_record);spec=root['task']['row']['spec'];p,_=o.restore_scenario(spec)
        state=root['state'];table=np.asarray(root['root_table']);c0,c1=root['candidates'][:2]
        active=[i for i in o.TEAMS[1] if p.is_active(state,i)]
        if len(active)!=1:raise ValueError('Expected one remaining pursuer')
        # Only the existing actor observation, public time/flags and root data.
        obs=np.asarray(p.policy_encoding(state,active[0]),float).reshape(-1)
        if obs.shape!=(270,):raise ValueError('Observation contract changed')
        dims=p.action_idxs[active[0]];half=(p.action_lims[dims,1]-p.action_lims[dims,0])/2
        a0=c0['action'][dims,0]/half;a1=c1['action'][dims,0]/half
        stats=[float(state[p.time_idx,0]/p.tf),c0['q'],c1['q'],c1['q']-c0['q'],
            np.log1p(c0['visits']),np.log1p(c1['visits']),float(len(table)),
            float(np.linalg.norm(a1-a0)),float(np.dot(a0,a1)),*a0,*a1]
        lidar=obs[8:264].reshape(64,4)
        compact=np.concatenate((stats,obs[:8],obs[264:],lidar.mean(axis=0),lidar.min(axis=0)))
        values=[];illegal=[]
        for stream in range(3):
            baseline=by_key[(spec['episode_hash'],0,stream)];alternative=by_key[(spec['episode_hash'],1,stream)]
            values.append(alternative['legal_double_capture']-baseline['legal_double_capture'])
            illegal.append(alternative['pursuer_illegal']-baseline['pursuer_illegal'])
        result.append(dict(physical_hash=s.previous.physical_hash(spec),episode_hash=spec['episode_hash'],
            compact=compact.tolist(),observation=np.concatenate((stats,obs)).tolist(),
            paired_delta=values,illegal_delta=illegal))
    if len(result)!=99 or len({x['physical_hash'] for x in result})!=99:raise ValueError('Scene grouping failed')
    lock_json(args.output/'scorer_dataset.json',result)
    return result


def summarize(args,semantics,continuations,fit_result):
    semantic_rows=[x for row in semantics for x in row['records']]
    semantic=dict(states=len(semantics),root_actions=len(semantic_rows),
        false_capture=int(sum(x['alone_capture']>x['joint_capture'] for x in semantic_rows)),
        missed_capture=int(sum(x['alone_capture']<x['joint_capture'] for x in semantic_rows)),
        capture_disagreement_scenes=int(sum(any(x['alone_capture']!=x['joint_capture'] for x in row['records']) for row in semantics)),
        selected_capture_disagreements=int(sum(x['selected'] and x['alone_capture']!=x['joint_capture'] for x in semantic_rows)),
        position_difference_actions=int(sum(x['position_delta']>1e-5 for x in semantic_rows)),
        illegal_disagreement_actions=int(sum(x['joint_illegal']!=x['alone_illegal'] for x in semantic_rows)),
        python_cpp_joint_transition_passed=True)
    groups={name:sorted([row for row in continuations if row['task']['variant']==name],key=lambda row:row['physical_hash'])
            for name in config(args)['variants']}
    summary={}
    for name,rows in groups.items():
        if len(rows)!=len(semantics):raise ValueError('Missing continuation pairing')
        costs={key:sum(x['own_cost'][key] for x in rows) for key in ('step_calls','policy_evaluations','value_evaluations')}
        summary[name]=dict(games=len(rows),successes=sum(x['metrics']['legal_double_capture'] for x in rows),
            success_rate=float(np.mean([x['metrics']['legal_double_capture'] for x in rows])),
            own_illegal=sum(x['metrics']['pursuer_illegal'] for x in rows),
            one_capture_timeout=sum(x['metrics']['one_capture_timeout'] for x in rows),
            own_mcts_calls=sum(x['own_calls'] for x in rows),cost=costs,
            parallel_wall_ms=sum(x['wall_ms'] for x in rows))
    comparisons={}
    for model in ('O','L'):
        a,b=groups[model+'_sync'],groups[model+'_legacy']
        if [x['physical_hash'] for x in a]!=[x['physical_hash'] for x in b]:raise ValueError('Unpaired model comparison')
        delta=np.asarray([x['metrics']['legal_double_capture']-y['metrics']['legal_double_capture'] for x,y in zip(a,b)],float)
        comparisons[model]=dict(delta=float(delta.mean()),gained=int((delta>0).sum()),lost=int((delta<0).sum()),
            ci95=scorer.paired_interval(delta),conditional_only=True,independent_confirmation=False)
    historical=0
    if not args.smoke:
        source_manifest=read_locked(SOURCE/'branch/manifest.json')
        originals={t['root_task']['row']['spec']['episode_hash']:t for t in source_manifest if t['candidate']==0 and t['stream']==0}
        for row in groups['O_legacy']:
            spec=row['task']['root']['task']['row']['spec'];old=r.read_task(originals[spec['episode_hash']])
            if h.science(row['metrics'])!=h.science(old['metrics']) or o.json_hash(row['trajectory'])!=o.json_hash(old['trajectory']) or \
               row['own_rng']!=old['own_rng'] or row['opponent_rng']!=old['opponent_rng']:
                raise ValueError('Complete O legacy continuation differs from historical cache')
            historical+=1
    result=dict(smoke=args.smoke,semantic=semantic,continuation=summary,comparisons=comparisons,
        historical_exact=historical,scorer=fit_result,recommended='old0:novalue',adopted=False,
        actor_updates=0,ppo_updates=0,continuation_requests=len(continuations),
        reason='smoke_only' if args.smoke else fit_result['reason'])
    lock_json(args.output/'results.json',result)
    lines=['部署一致性与动作选择可学习性实验','',
        '本轮为历史局面的条件诊断和探索性交叉验证；推荐继续为 old0:novalue。',
        'actor、PPO 更新均为 0；没有新场景效果确认，也没有导出评分器供部署。','',
        f"语义核验：{semantic['states']} 个首捕状态、{semantic['root_actions']} 个根动作。单方推进产生额外捕获 {semantic['false_capture']} 次、漏捕 {semantic['missed_capture']} 次，涉及 {semantic['capture_disagreement_scenes']} 个状态；原选中动作捕获分歧 {semantic['selected_capture_disagreements']} 次。",'',
        '|配置|条件合法双捕|自身非法|首捕后超时|环境转移调用|',
        '|---|---:|---:|---:|---:|']
    for name,row in summary.items():
        lines.append(f"|{name}|{row['successes']}/{row['games']} ({row['success_rate']:.2%})|{row['own_illegal']}|{row['one_capture_timeout']}|{row['cost']['step_calls']}|")
    lines+=['','所有配置共享 O 的历史首捕状态和对手随机流；L 结果不是 L 自身整局部署成功率。',
        '同步基线改变了节点决策和联合随机 rollout；条件效果不能单独归因于状态推进修正。',
        '模拟次数相同不代表转移次数或耗时相同；并行耗时只供审计，不作为单进程加速证据。','']
    for name,row in comparisons.items():
        lines.append(f"{name} 同步相对交替：{row['delta']*100:+.2f} 点，挽回 {row['gained']}、丢失 {row['lost']}，普通配对 95% 区间 {[round(x*100,2) for x in row['ci95']]} 点。")
    if fit_result:
        lines+=['',f"嵌套场景分组评分器：相对 A0 {fit_result['delta_vs_A0']*100:+.2f} 点，95% 区间 {[round(x*100,2) for x in fit_result['ci95_vs_A0']]}；相对恒选 AQ {fit_result['delta_vs_AQ']*100:+.2f} 点，95% 区间 {[round(x*100,2) for x in fit_result['ci95_vs_AQ']]}。",
            f"99 个物理场景、每场景 3 个配对随机流；外层共替换 {fit_result['replacements']} 个状态，挽回 {fit_result['gained']}、丢失 {fit_result['lost']} 个状态×流。",
            f"可学习性门槛：{'通过' if fit_result['passed'] else '未通过'}；{fit_result['reason']}。",
            '普通区间只描述探索结果，不能作为新场景独立确认或多方法同时置信保证。',
            '若门槛未过，停止该评分器分支；不追加数据、不改阈值、不转入 actor 训练。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


PLAN='''本轮执行协议（查看本轮结果之前锁定）

目标：先量化当前交替搜索与联合环境推进的差异，实现独立同步搜索基线；随后检查历史配对结果能否由部署信息预测。

模型 O、L 冻结。新 actor/PPO 更新、教师查询均为 0。正式每次追捕 1000 模拟、深度上限 100、βπ=.5、βv=0；真实对手保持原 Search Only 3000、βπ=βv=0；环境、奖励和剩余 80 步时限保持原配置。

同步基线：同节点双方各自用团队平均回报、UCT 和渐进拓宽选连续动作；选择过程不读取另一方本步的选择。联合动作仅推进一次。随机安全 rollout 同步推进，长度不超过搜索深度减已走树深，且服从物理终止。它是诊断基线，不宣称均衡收敛或新算法首创。

使用原根动作核验全部 99 个首捕后局面，包含原来成功和失败。逐根动作比较追捕单方推进、真实冻结对手联合推进与反向交替推进；真实对手当步动作只用于离线诊断，不输入树内决策。

续跑 O_legacy、O_sync、L_legacy、L_sync，最多 396 条。共享原 O 首捕状态和已保存双方随机流。O_legacy 必须与原 A0/stream0 完整动作、科学指标、双方最终 RNG 一致。L 续跑只回答相同 O 局面的条件问题，不作 L 自身整局效应。新场景效果验证不在本轮内。每项记录环境转移、策略/价值网络调用、真实决策数；不根据并行耗时宣称部署加速。

评分器只用既有 99 场景×2 动作×3 流配对结果；不新增标签。候选 A0/AQ，部署只替换首捕后一次。输入为原 actor 270 维观测、当时动作及根统计；无未来结果或隐藏对手身份。紧凑 Ridge α=10、全观测 Ridge α=100、单分裂树 min_leaf=12；内层 4 折选择方法和阈值 {0,.05,.1,.2,不替换}，外层 5 折评估。所有分区按物理场景，所有候选/流同组；拟合均值方差只用训练分区。

研发继续门槛：组外相对 A0 净增≥3点、普通 95% 配对场景 bootstrap 下界>0；相对恒选 AQ 净增>0 且下界>0；至少 3/5 外折增益>0；自身非法增加≤2点。bootstrap 20000 次、种子 50020261024。该门槛仅筛选是否值得登记新 L 数据，不代表独立确认。未过停止本学习方案，不追加数据、换阈值或训练 actor。通过时另行锁定新的 L 数据和正式验证协议，现有历史数据不得充当独立确认。

独立 smoke：两张地图各一新场景，构造剩余 3 步首捕状态，每搜索 8/12 模拟，共 8 条续跑。先通过原搜索、状态转移、捕获边界、角色/失活、随机恢复和缓存校验，再运行正式任务。正式 80 单线程 worker，独立 tmux。按语义任务和续跑任务原子提交，中断从上个已提交任务重放；源码、模型、平台、缓存变更或缺失配对时停止。报告和统计不修改历史结果；推荐始终 O，完成本轮不自动采用任何探索配置。
'''


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',nargs='?',default='pipeline',choices=['pipeline','prepare','audit','diagnose','pilot','fit','report','replay','watch'])
    parser.add_argument('--workers',type=int,default=80);parser.add_argument('--output',type=Path)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--once',action='store_true')
    args=parser.parse_args(argv)
    if args.workers<1:parser.error('workers must be positive')
    args.output=(args.output or (SMOKE if args.smoke else DEFAULT)).resolve();return args


def watch(args):
    while True:
        status=o.read_json(args.output/'status.json');status['pid_alive']=Path('/proc',str(status['pid'])).exists()
        print(__import__('json').dumps(status,ensure_ascii=False,indent=2),flush=True)
        if args.once or not status['pid_alive'] or status['stage'] in ('complete','failed'):return
        time.sleep(5)


def main(argv=None):
    args=parse_args(argv)
    if args.stage=='watch':return watch(args)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(pid=os.getpid(),stage='prepare',requested_workers=args.workers,started=time.time())
        mutex=threading.Lock();stop=threading.Event()
        def status(stage,**values):
            with mutex:
                if stage!=state['stage']:
                    state.pop('completed',None);state.pop('total',None);state.pop('cached',None)
                    state.pop('tasks_per_second',None)
                    state['stage_started']=time.time()
                state.update(stage=stage,updated=time.time(),**values)
                children=[p.pid for p in mp.active_children() if p.is_alive()]
                thread_counts={str(pid):len(list(Path('/proc',str(pid),'task').glob('*'))) for pid in children}
                state.update(actual_workers=len(children),worker_pids=children,worker_threads=thread_counts)
                if 'completed' in state:state['tasks_per_second']=state['completed']/max(.01,time.time()-state.get('stage_started',state['started']))
                o.write_json(args.output/'status.json',state)
        def heartbeat():
            while not stop.wait(5):
                with mutex:stage=state['stage']
                status(stage)
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:
            record=prepare(args);status('prepared')
            if args.stage=='prepare':return
            if args.stage in ('pipeline','audit'):audit(args,record)
            if args.stage=='audit':status('complete',reason='engineering_passed');return
            if not read_locked(args.output/'audit_result.json')['passed']:raise ValueError('Audit required')
            if args.stage in ('pipeline','diagnose','pilot'):
                with mp.get_context('spawn').Pool(args.workers) as pool:
                    if args.stage in ('pipeline','diagnose'):
                        run_tasks(args,'semantic',items(args,record,'semantic'),semantic_one,pool,status)
                    if args.stage in ('pipeline','pilot'):
                        run_tasks(args,'continuation',items(args,record,'continuation'),continuation_one,pool,status)
                if args.stage!='pipeline':status('complete',reason=args.stage+'_complete');return
            if args.stage in ('pipeline','fit') and not args.smoke:
                status('fit');data=dataset(args,record);fit_result=scorer.evaluate(data)
                lock_json(args.output/'scorer_result.json',fit_result)
            if args.stage=='fit':status('complete',reason=read_locked(args.output/'scorer_result.json')['reason']);return
            semantics=[read_task(t) for t in items(args,record,'semantic')]
            continuations=[read_task(t) for t in items(args,record,'continuation')]
            fit_result=None if args.smoke else read_locked(args.output/'scorer_result.json')
            result=summarize(args,semantics,continuations,fit_result)
            verify(args)
            lock_json(args.output/'replay.json',dict(passed=True,semantic=len(semantics),continuations=len(continuations),
                historical_exact=result['historical_exact'],cache_only=True))
            status('complete',reason=result['reason'],recommended=result['recommended'],
                elapsed_seconds=time.time()-state['started'])
        except BaseException as error:
            status('failed',reason=type(error).__name__+': '+str(error));raise
        finally:stop.set();thread.join(timeout=1)


if __name__=='__main__':main()
