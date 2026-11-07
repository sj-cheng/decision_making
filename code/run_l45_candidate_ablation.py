"""Six frozen root-proposal controls from identical real first-capture states."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
import contextlib
import copy
import fcntl
import hashlib
import io
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import threading
import time
import numpy as np
import torch
import run_l45_interception_guidance as old
from solvers.candidate_ablation import CandidateAblationPUCT,cpp

torch.set_num_threads(1)
o=old.o
ROOT,CODE=o.ROOT,o.CODE
SOURCE=old.DEFAULT_OUTPUT
DEFAULT_OUTPUT=ROOT/'current/l45_candidate_ablation_20261024'
SMOKE_OUTPUT=ROOT/'current/l45_candidate_ablation_smoke_20261024'
UIDS=('S','F','D','G','P','PR')
PRIMARY=('F-S','D-F','G-D','P-G','PR-P')
lock_json,locked_read=old.lock_json,old.locked_read
commit_row,read_cached,read_episode=old.commit_row,old.read_cached,old.read_episode

def config(args):
    return dict(schema=1,smoke=args.smoke,uids=list(UIDS),streams=[0,1],source=str(SOURCE),
        training_episodes=0,source_episode=2048,budget=3 if args.smoke else 1000,
        opponent_budget=3 if args.smoke else 3000,depth=100,pursuer_betas=[.5,0.],
        root_mixture=[.5,.25,.25],noise_fraction=.1,scope='all_99_S_first_capture_states',
        stream0='restore_all_original_rng',stream1='sha256_candidate_ablation_20261024_scene_side_low32',
        primary=list(PRIMARY),minimum_signal=.05,illegal_margin=.02,
        bootstrap_seed=56020261024,bootstrap_draws=10000,
        suffix_request_cap=24 if args.smoke else 1188,new_suffix_cap=24 if args.smoke else 891,
        historical_reuse_cap=0 if args.smoke else 297,development_only=True,
        recommended='old0:novalue',original_development_candidate='PR')

def fingerprints():
    paths=[Path(__file__),CODE/'build_candidate_ablation.py',CODE/'solvers/candidate_ablation.py',
        CODE/'cpp/solvers/root_candidate_ablation.hpp',CODE/'L45_CANDIDATE_ABLATION.md',
        CODE/'cpp/build_candidate_ablation/build_manifest.json',Path(cpp.__file__)]
    return dict(files={str(p):o.sha(p) for p in paths},frozen_parent=old.fingerprints())

def registry(args):
    p=args.output/'models/O'
    return {'O':dict(models=str(p),model_hash=o.model_fingerprint(p))}

def settings(args):
    c=config(args)
    return dict(search_budget=c['budget'],search_only_budget=c['opponent_budget'],
        search_depth=100,beta_policy=.5,beta_value=0.)

def effective(uid,reg):
    return dict(weights=reg['O']['model_hash'],algorithm='frozen_synchronous_DUCT',
        root_proposal=uid,rollout='guided_8' if uid=='PR' else 'original_random',
        activation='one_real_active_evader_and_pursuer',opponent='frozen_Search_Only')

def make_own(p,reg,uid,requested,seed):
    paths,value_path=old.previous.research.h.oracle_paths(str(ROOT),iteration=45,model_dir=reg['O']['models'])
    with contextlib.redirect_stdout(io.StringIO()):
        policies,value=old.previous.research.h.get_oracles(p,value_oracle_name='deterministic',value_oracle_path=value_path,
            policy_oracle_name='gaussian',policy_oracle_paths=paths)
    raw=CandidateAblationPUCT(policies,value,requested['search_budget'],.5,0.,uid,100)
    raw.solver_wrapper.set_seed(int(seed))
    return o.RoleMaskedSolver(raw,o.TEAMS[1])

def make_pair(task):
    p,state=o.restore_scenario(task['spec']);p.clear_closed_loop_cache()
    own=make_own(p,task['registry'],task['uid'],task['own_settings'],task['spec']['search_seeds'][1])
    other=old.previous.base.private_solver(p,task['registry']['O'],'Search Only',0,
        task['opponent_settings'],task['spec']['search_seeds'][0])
    return p,state,own,other

def stream_seed(scene,side):
    return int.from_bytes(hashlib.sha256(f'candidate_ablation_20261024|{scene}|{side}'.encode()).digest()[:8],'big')&0xffffffff

def restore(task):
    if o.sha(task['snapshot_path'])!=task['snapshot_sha256']:raise ValueError('First-capture snapshot changed')
    snap=o.load_torch(task['snapshot_path'])
    p,_,own,other=make_pair(task)
    own.solver.set_rng_state(p,snap['own_rng']);other.solver.set_rng_state(p,snap['opponent_rng'])
    o.restore_rng(snap['python_rng'])
    if task['stream']==1:
        own.solver.solver_wrapper.set_seed(task['spec']['search_seeds'][1])
        other.solver.solver_wrapper.set_seed(task['spec']['search_seeds'][0])
        seed=stream_seed(task['scene_id'],'python')
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    state=snap['state'].copy()
    if state.shape!=(53,1) or float(state[p.time_idx,0])!=snap['time'] or p.tf-snap['time']!=snap['remaining_time']:
        raise ValueError('Restored state or remaining time differs')
    p.clear_closed_loop_cache()
    return p,state,own,other,copy.deepcopy(snap['stats'])

def source_references():
    args=old.parse_args([]);banks=old.prepare(args)
    if old.locked_read(SOURCE/'results.json')['reason']!='development_passed':raise ValueError('Expected completed source')
    entries=old.requests(args,('S','P','PR'),banks['pilot_four'])
    result={}
    for item in entries:
        row,detail=old.read_episode(args,item)
        result.setdefault(row['episode_hash'],{})[item['uid']]=(item,row,detail)
    return banks,result

def smoke_cases(args):
    result=[]
    for i,layout in enumerate(('four','none')):
        spec=o.make_scenario(56261024+i,layout);p,state=o.restore_scenario(spec)
        state[[p.active_idxs[0],p.active_idxs[2]],0]=0.
        state[p.state_idxs[1],0]=[-5.,0.];state[p.state_idxs[3],0]=[5.,0.]
        state[p.time_idx,0]=76.;p._update_observation_memory(state,state,0.)
        task=dict(uid='S',spec=spec,registry=registry(args),own_settings=settings(args),opponent_settings=settings(args))
        _,_,own,other=make_pair(task)
        stats=old.previous.initial_stats();stats.update(steps=76,first=76.,last=76.,capture_count=1)
        snap=old.previous.snapshot(p,state,stats,own,other)
        result.append((spec,snap,{}))
    return result

def prepare(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if out==SOURCE or SOURCE in out.parents:raise ValueError('Use isolated output')
    if (out/'experiment.lock.json').exists():
        record=locked_read(out/'experiment.json')
        if record['config']!=config(args) or record['implementation']!=fingerprints() or record['models']!=registry(args):
            raise ValueError('Frozen experiment configuration changed')
        for p,sha in record['sources'].items():
            if o.sha(p)!=sha:raise ValueError('Frozen source changed: '+p)
        return locked_read(out/'cases.json')
    dest=out/'models/O';dest.mkdir(parents=True,exist_ok=True)
    for name in o.MODEL_NAMES:shutil.copyfile(SOURCE/'models/O'/name,dest/name)
    sources={str(SOURCE/'models/O'/n):o.sha(SOURCE/'models/O'/n) for n in o.MODEL_NAMES}
    all_cases=[];raw=[]
    if args.smoke:
        raw=smoke_cases(args)
    else:
        banks,refs=source_references()
        for spec in banks['pilot_four']:
            group=refs[spec['episode_hash']];snap=group['S'][2]['first_capture']
            eligible=bool(snap and snap['branch_available'])
            all_cases.append(dict(scene_id=spec['episode_hash'],eligible=eligible,spec=spec,
                fixed_if_no_suffix={u:{k:r[k] for k in old.previous.SCIENCE} for u,(_,r,_) in group.items()} if not eligible else {}))
            if not eligible:continue
            ref={}
            for u,(item,row,detail) in group.items():
                if not old.s.equal_tree(snap,detail['first_capture']):raise ValueError('Source first capture or RNG differs')
                metric=SOURCE/'cache'/(item['key']+'.json');artifact=Path(row['episode_record']['path'])
                sources[str(metric)]=o.sha(metric);sources[str(artifact)]=o.sha(artifact)
                ref[u]=dict(key=item['key'],metric=str(metric),record=str(artifact))
            raw.append((spec,snap,ref))
        if len(raw)!=99 or len(all_cases)!=100:raise ValueError('Expected all 99 eligible source states and 100 scenes')
        for name in ('experiment.json','scenarios.json','results.json','prefix_check.json'):
            sources[str(SOURCE/name)]=o.sha(SOURCE/name)
    cases=[]
    for spec,snap,ref in raw:
        path=out/'snapshots'/(spec['episode_hash']+'.pt')
        if not path.exists():o.save_torch(path,snap)
        stored=o.load_torch(path)
        if not old.s.equal_tree(stored,snap):raise ValueError('Existing snapshot differs')
        sources[str(path)]=o.sha(path)
        cases.append(dict(scene_id=spec['episode_hash'],spec=spec,snapshot_path=str(path),snapshot_sha256=o.sha(path),historical=ref))
    if args.smoke:all_cases=[dict(scene_id=c['scene_id'],eligible=True,spec=c['spec']) for c in cases]
    lock_json(out/'cases.json',cases);lock_json(out/'all_cases.json',all_cases)
    sources[str(out/'cases.json')]=o.sha(out/'cases.json');sources[str(out/'all_cases.json')]=o.sha(out/'all_cases.json')
    lock_json(out/'experiment.json',dict(config=config(args),implementation=fingerprints(),models=registry(args),sources=sources))
    shutil.copyfile(CODE/'L45_CANDIDATE_ABLATION.md',out/'PLAN.md')
    runtime=out/'runtime';runtime.mkdir(exist_ok=True)
    shutil.copyfile(Path(cpp.__file__),runtime/Path(cpp.__file__).name)
    shutil.copyfile(CODE/'cpp/build_candidate_ablation/build_manifest.json',runtime/'build_manifest.json')
    shutil.copytree(CODE/'cpp/build_candidate_ablation/source',runtime/'source',dirs_exist_ok=True)
    shutil.copyfile(CODE/'solvers/candidate_ablation.py',runtime/'candidate_ablation.py')
    return cases

def requests(args,cases):
    reg=registry(args);items=[]
    for case in cases:
        for stream in (0,1):
            spec=copy.deepcopy(case['spec'])
            if stream:
                spec['search_seeds']=[stream_seed(case['scene_id'],'evader'),stream_seed(case['scene_id'],'pursuer')]
                spec['episode_hash']=o.json_hash({k:v for k,v in spec.items() if k!='episode_hash'})
            for uid in UIDS:
                behavior=effective(uid,reg)
                identity=dict(protocol=o.sha(args.output/'experiment.json'),policy_id=o.json_hash(behavior),
                    episode_hash=spec['episode_hash'],scene_id=case['scene_id'],stream=stream,
                    snapshot_sha256=case['snapshot_sha256'],role=1,opponent='Search Only',timing_context='parallel',
                    own_settings=settings(args),opponent_settings=settings(args),mode='search')
                key=o.json_hash(identity)
                task=dict(**identity,key=key,uid=uid,spec=spec,registry=reg,effective=behavior,
                    snapshot_path=case['snapshot_path'],historical=case['historical'].get(uid) if stream==0 else None)
                items.append(dict(key=key,uid=uid,task=task))
    return items

def decorate(task,row,detail,decisions,elapsed,wall):
    row.update({k:task[k] for k in ('key','policy_id','role','opponent','timing_context','mode','scene_id','stream')})
    row.update(uid=task['uid'],episode_hash=task['spec']['episode_hash'],layout=task['spec']['layout'],
        search_seeds=task['spec']['search_seeds'],actual_own=dict(task['own_settings'],effective=task['effective']),
        actual_opponent=dict(method='Search Only',budget=task['opponent_settings']['search_only_budget'],depth=100,beta_policy=0.,beta_value=0.),
        own_cost={k:sum(r['cost'][k] for r in detail['roots']) for k in ('step_calls','policy_evaluations','value_evaluations')},
        own_mcts_calls=len(detail['trajectory']),episode_wall_ms=wall,deployment_episode_ms=elapsed,
        trajectory_sha256=old.previous.trajectory_digest(detail['trajectory']),decision_latency_ms=decisions,
        reused_historical=bool(task['historical']),inherited_prefix_steps=int(detail['initial']['stats']['steps']))
    row['_record']=dict(key=task['key'],scenario=task['spec'],effective=task['effective'],role=1,opponent='Search Only',
        scene_id=task['scene_id'],stream=task['stream'],**detail)
    return row

def evaluate_one(task):
    p,state,own,other,stats=restore(task);started=time.perf_counter()
    row,detail,decisions,elapsed=old.simulate(p,state,own,other,stats)
    return decorate(task,row,detail,decisions,elapsed,(time.perf_counter()-started)*1000)

def reuse_one(task):
    ref=task['historical'];row=o.read_json(ref['metric']);d=o.load_torch(ref['record'])
    snap=o.load_torch(task['snapshot_path']);start=snap['next_decision']
    if not old.s.equal_tree(snap,d['first_capture']):raise ValueError('Historical prefix state changed')
    detail=dict(initial=copy.deepcopy(snap),first_capture=None,trajectory=d['trajectory'][start:],
        roots=d['roots'][start:],final=d['final'],source_record=ref)
    decisions=row['decision_latency_ms'][start:]
    science={k:row[k] for k in old.previous.SCIENCE}
    return decorate(task,science,detail,decisions,sum(x[2] for x in decisions),0.)

def run(args,items,status):
    if len(items)>config(args)['suffix_request_cap']:raise ValueError('Suffix budget exceeded')
    lock_json(args.output/'pilot/manifest.json',[dict(key=i['key'],uid=i['uid'],scene_id=i['task']['scene_id'],stream=i['task']['stream']) for i in items])
    for item in items:
        if (args.output/'cache_checksums'/(item['key']+'.lock.json')).exists():read_cached(args,item)
        elif item['task']['historical']:commit_row(args,reuse_one(item['task']))
    missing=[i['task'] for i in items if not (args.output/'cache_checksums'/(i['key']+'.lock.json')).exists()]
    new_total=sum(not i['task']['historical'] for i in items)
    if new_total>config(args)['new_suffix_cap']:raise ValueError('New execution budget exceeded')
    done=len(items)-len(missing);started_done=done;clock=time.perf_counter()
    status('pilot',completed=done,total=len(items),historical_reused=sum(bool(i['task']['historical']) for i in items),
        new_total=new_total,pending_new=len(missing))
    with mp.get_context('spawn').Pool(min(args.workers,max(1,len(missing)))) as pool:
        o.write_json(args.output/'worker_snapshot.json',dict(pids=[p.pid for p in pool._pool],
            workers=len(pool._pool),compute_threads_per_worker=1,observed_unix=time.time()))
        for row in pool.imap_unordered(evaluate_one,missing,chunksize=1):
            commit_row(args,row);done+=1
            if done%5==0 or done==len(items):status('pilot',completed=done,total=len(items),
                throughput_per_second=(done-started_done)/max(.001,time.perf_counter()-clock))
    return [read_cached(args,i) for i in items]

def paired(rows):
    index={};scenes=sorted({r['scene_id'] for r in rows})
    for r in rows:
        key=(r['uid'],r['scene_id'],r['stream'])
        if key in index:raise ValueError('Duplicate scene/stream/method')
        index[key]=r
    if not scenes or len(index)!=len(scenes)*len(UIDS)*2:raise ValueError('Missing paired suffix')
    for scene in scenes:
        for stream in (0,1):
            seeds=[index[(u,scene,stream)]['search_seeds'] for u in UIDS]
            if any(x!=seeds[0] for x in seeds):raise ValueError('Unpaired search streams')
    return index,scenes

def analyze(rows,all_cases):
    index,scenes=paired(rows);rng=np.random.default_rng(56020261024)
    indices=rng.integers(0,len(scenes),size=(10000,len(scenes)))
    outcome={u:np.array([[index[(u,s,t)]['legal_double_capture'] for t in (0,1)] for s in scenes],float) for u in UIDS}
    names=list(PRIMARY)+[u+'-S' for u in ('D','G','P','PR')]+['F-P','D-P','G-P']
    diffs={name:(outcome[name.split('-')[0]]-outcome[name.split('-')[1]]).mean(1) for name in dict.fromkeys(names)}
    primary_draws=np.stack([diffs[n][indices].mean(1) for n in PRIMARY],1)
    centers=np.array([diffs[n].mean() for n in PRIMARY]);critical=float(np.quantile((primary_draws-centers).max(1),.95))
    stats={}
    for name,x in diffs.items():
        a,b=name.split('-');raw=outcome[a]-outcome[b]
        ia=np.array([[index[(a,s,t)]['pursuer_illegal'] for t in (0,1)] for s in scenes])
        ib=np.array([[index[(b,s,t)]['pursuer_illegal'] for t in (0,1)] for s in scenes])
        ci=np.quantile(x[indices].mean(1),[.025,.975]).tolist();delta=float(x.mean())
        cost=np.array([np.mean([index[(a,s,t)]['completion_cost']-index[(b,s,t)]['completion_cost'] for t in (0,1)]) for s in scenes])
        stats[name]=dict(delta=delta,ci95=ci,stream_delta=raw.mean(0).tolist(),
            gained_state_streams=int((raw>0).sum()),lost_state_streams=int((raw<0).sum()),
            own_illegal_delta=float((ia-ib).mean()),completion_cost_delta=float(cost.mean()),
            completion_cost_ci95=np.quantile(cost[indices].mean(1),[.025,.975]).tolist(),
            strong_signal=bool(delta>=.05-1e-12 and ci[0]>0 and (ia-ib).mean()<=.02+1e-12))
        if name in PRIMARY:stats[name]['simultaneous_lower95']=delta-critical
    summary={};full_n=len(all_cases)
    for u in UIDS:
        group=[index[(u,s,t)] for s in scenes for t in (0,1)]
        costs=[r['completion_cost'] for r in group]
        fixed=[c for c in all_cases if not c['eligible']]
        fixed_success=sum(c['fixed_if_no_suffix'].get(u,c['fixed_if_no_suffix']['S'])['legal_double_capture'] for c in fixed)
        summary[u]=dict(eligible_scenes=len(scenes),suffixes=len(group),
            conditional_success_rate=float(outcome[u].mean()),successes_by_stream=outcome[u].sum(0).astype(int).tolist(),
            full_scenes=full_n,full_success_rate=float((outcome[u].sum()+2*fixed_success)/(2*full_n)),
            own_illegal=sum(r['pursuer_illegal'] for r in group),opponent_illegal=sum(r['evader_illegal'] for r in group),
            mean_completion_cost=float(np.mean(costs)),
            full_mean_completion_cost=float((sum(costs)+sum(2*c['fixed_if_no_suffix'].get(u,c['fixed_if_no_suffix']['S'])['completion_cost'] for c in fixed))/(2*full_n)),
            mean_post_cost=float(np.mean([r['completion_cost']-r['first_capture_time'] for r in group])),
            successful_second_capture_mean=float(np.mean([r['capture_time']-r['first_capture_time'] for r in group if r['legal_double_capture']])) if any(r['legal_double_capture'] for r in group) else None,
            one_capture_timeouts=sum(r['one_capture_timeout'] for r in group),
            new_executions=sum(not r['reused_historical'] for r in group),
            historical_reused=sum(r['reused_historical'] for r in group))
    return dict(summary=summary,comparisons=stats,bootstrap_seed=56020261024,draws=10000,
        physical_scene_count=len(scenes),streams_per_scene=2,primary=list(PRIMARY),exploratory=True,
        simultaneous_critical=critical,recommended='old0:novalue',original_development_candidate='PR')

def diagnostics(args,items):
    result={u:dict(calls=0,step_calls=0,policy_evaluations=0,geometry_queries=0.,candidate_sources={},selected_sources={},
        proposal_fallbacks=0,noise_fallbacks=0,first_magnitudes=[],first_alignment=[],selected_magnitudes=[],selected_alignment=[],
        maximum_guided_prefix=0.,candidate_magnitudes=[],candidate_alignment=[],speed_groups={}) for u in UIDS}
    records=[]
    for item in items:
        row,d=read_episode(args,item);g=result[item['uid']];p,_=o.restore_scenario(d['scenario'])
        g['calls']+=len(d['roots'])
        for j,(root,step) in enumerate(zip(d['roots'],d['trajectory'])):
            state=step['state'];active=[r for r in p.pursuers if p.is_active(state,r)];evaders=[r for r in p.evaders if p.is_active(state,r)]
            if len(active)!=1 or len(evaders)!=1:raise ValueError('Suffix is not one on one')
            robot,evader=active[0],evaders[0];upper=p.action_lims[p.action_idxs[robot],1]
            delta=state[p.state_idxs[evader],0]-state[p.state_idxs[robot],0]
            act=step['action'][p.action_idxs[robot],0]
            magnitude=float(np.max(abs(act)/upper));alignment=float(act@delta/max(1e-12,np.linalg.norm(act)*np.linalg.norm(delta)))
            g['selected_magnitudes'].append(magnitude);g['selected_alignment'].append(alignment)
            if j==0:g['first_magnitudes'].append(magnitude);g['first_alignment'].append(alignment)
            for k in ('step_calls','policy_evaluations'):g[k]+=root['cost'][k]
            audit=root['guidance'];g['geometry_queries']+=audit.get('geometry_queries',0)
            g['proposal_fallbacks']+=audit.get('proposal_fallback',0);g['noise_fallbacks']+=audit.get('noise_fallback',0)
            g['maximum_guided_prefix']=max(g['maximum_guided_prefix'],audit.get('max_guided_prefix',0))
            source=str(int(audit.get('selected_source',-1)));g['selected_sources'][source]=g['selected_sources'].get(source,0)+1
            for source_id in (0,1,2):
                key=str(source_id);g['candidate_sources'][key]=g['candidate_sources'].get(key,0)+audit.get('root_source_'+key,0)
            for source_id,rootrow in zip(root['sources'],root['table']):
                if source_id!=2:continue
                candidate=rootrow[-p.action_dim:][p.action_idxs[robot]]
                g['candidate_magnitudes'].append(float(np.max(abs(candidate)/upper)))
                g['candidate_alignment'].append(float(candidate@delta/max(1e-12,np.linalg.norm(candidate)*np.linalg.norm(delta))))
        ratio=d['scenario']['pursuer_speed']/d['scenario']['evader_speed']
        label='<1.2' if ratio<1.2 else '>=1.2';sg=g['speed_groups'].setdefault(label,dict(suffixes=0,successes=0))
        sg['suffixes']+=1;sg['successes']+=row['legal_double_capture']
        records.append(dict(uid=item['uid'],scene_id=row['scene_id'],stream=row['stream'],success=row['legal_double_capture'],
            post_cost=row['completion_cost']-row['first_capture_time'],completion_cost=row['completion_cost'],
            speed_ratio=ratio,record=row['episode_record']['path'],reused=row['reused_historical']))
    for g in result.values():
        for k in ('first_magnitudes','first_alignment','selected_magnitudes','selected_alignment','candidate_magnitudes','candidate_alignment'):
            g[k+'_mean']=float(np.mean(g.pop(k))) if g[k] else None
    return result,records

def report(args,cases,items,rows):
    result=analyze(rows,locked_read(args.output/'all_cases.json'));diag,records=diagnostics(args,items)
    result['smoke']=args.smoke
    if args.smoke:
        for comparison in result['comparisons'].values():comparison['strong_signal']=False
    lock_json(args.output/'statistics.json',result);lock_json(args.output/'diagnostics.json',diag)
    lock_json(args.output/'paired_results.json',records)
    n=result['summary']['S']['full_scenes']
    lines=['首捕后根候选收益来源：六组开发实验' if not args.smoke else '隔离smoke：仅工程检查，不作效果结论','',
        '固定O权重、99个相同S首捕状态、每状态两个配对搜索流；全100口径保留原零捕场景。无新增训练。' if not args.smoke else '两个独立构造状态、六组、两流、预算3；仅检查分支和恢复。',
        'S原同步，F满幅神经均值方向，D直接追踪，G绕障碍追踪，P预测拦截，PR预测拦截+引导模拟。','',
        f'|组|流0成功/{n}|流1成功/{n}|全场景平均成功率|接管后成功率|自身非法/{2*len(cases)}|后段成本|全局完成成本|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for u in UIDS:
        g=result['summary'][u];a,b=g['successes_by_stream']
        lines.append(f'|{u}|{a}|{b}|{100*g["full_success_rate"]:.2f}%|{100*g["conditional_success_rate"]:.2f}%|{g["own_illegal"]}|{g["mean_post_cost"]:.2f}|{g["full_mean_completion_cost"]:.2f}|')
    lines+=['','主比较以99个物理状态为单位，先平均两流。10000次配对bootstrap，普通95%区间；同时下界共同覆盖五项主比较。','',
        '|比较|条件增益pp|普通95%区间pp|同时95%下界pp|挽回/丢失 状态×流|较强信号|',
        '|---|---:|---|---:|---:|---|']
    for name in PRIMARY:
        c=result['comparisons'][name];lo,hi=c['ci95']
        lines.append(f'|{name}|{c["delta"]*100:+.2f}|[{lo*100:+.2f},{hi*100:+.2f}]|{c["simultaneous_lower95"]*100:+.2f}|{c["gained_state_streams"]}/{c["lost_state_streams"]}|{c["strong_signal"]}|')
    lines+=['','动作与计算记录：','',
        '|组|首动作归一化幅度|首动作追踪余弦|后段决策数|网络调用|环境转移|几何查询|候选回退|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for u,g in diag.items():
        lines.append(f'|{u}|{g["first_magnitudes_mean"]:.3f}|{g["first_alignment_mean"]:.3f}|{g["calls"]}|{g["policy_evaluations"]}|{g["step_calls"]}|{g["geometry_queries"]:.0f}|{g["proposal_fallbacks"]:.0f}|')
    lines+=['','源流S/P/PR来自原记录，其他流为新续跑；两流不是独立物理样本。历史开发状态不作独立效果确认。',
        'F-S同时改变均匀槽位与满幅均值方向，不单独证明幅度因果贡献；D-F、G-D、P-G用于逐层识别候选包的附加作用。',
        '新旧路径从相同快照独立响应，未共用未来对手动作。失败计80步；后段成本减首捕时间。',
        '本轮不替换部署推荐old0:novalue，不改变原开发候选PR，不自动进入新场景实验。']
    plots(args,result,diag)
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (args.output/'CONCLUSION.md').write_text('\n'.join(lines[:13])+'\n')
    lock_json(args.output/'results.json',dict(reason='smoke_complete' if args.smoke else 'mechanism_experiment_complete',
        scientific_suffix_requests=len(rows),actual_new_suffixes=sum(not r['reused_historical'] for r in rows),
        historical_reused=sum(r['reused_historical'] for r in rows),physical_scenes=len(cases),
        training_episodes=0,recommended='old0:novalue',original_development_candidate='PR',
        strong_signals=[n for n in PRIMARY if result['comparisons'][n]['strong_signal']]))
    return result

def plots(args,result,diag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=args.output/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].bar(UIDS,[result['summary'][u]['full_success_rate']*100 for u in UIDS],color=['#777777','#5799c0','#399c88','#78ad50','#d39839','#bc7143'])
    axes[0].set(ylim=(0,100),ylabel='Legal double capture (%)',title='Two streams averaged within physical scene')
    axes[1].bar(UIDS,[result['summary'][u]['mean_post_cost'] for u in UIDS]);axes[1].set(ylabel='Remaining-step cost',title='All 99 takeover states, including failures')
    for i,name in enumerate(PRIMARY):
        c=result['comparisons'][name];lo,hi=np.array(c['ci95'])*100;v=c['delta']*100
        axes[2].errorbar(v,len(PRIMARY)-1-i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=3)
    axes[2].set(yticks=range(len(PRIMARY)),yticklabels=list(reversed(PRIMARY)),xlabel='Paired difference (pp)',title='Ordinary 95% bootstrap intervals');axes[2].axvline(0,color='gray',ls='--')
    fig.tight_layout();fig.savefig(out/'mechanism_ablation.png',dpi=190);plt.close(fig)

def replay(args,items):
    item=next(i for i in items if i['uid']=='G' and i['task']['stream']==1)
    row,d=read_episode(args,item);again=evaluate_one(item['task'])
    if old.previous.science(row)!=old.previous.science(again) or row['trajectory_sha256']!=again['trajectory_sha256']:
        raise ValueError('Suffix replay differs')
    for side in ('own_rng','opponent_rng'):
        if not old.s.equal_tree(d['final'][side],again['_record']['final'][side]):raise ValueError('Replay random stream differs')
    lock_json(args.output/'replay.json',dict(passed=True,uid='G',stream=1,both_rng_exact=True,engineering_extra_suffixes=1))

def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',nargs='?',default='pipeline',choices=['pipeline','prepare','pilot','report','watch','replay'])
    parser.add_argument('--workers',type=int,default=80);parser.add_argument('--output',type=Path)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--once',action='store_true')
    args=parser.parse_args(argv);args.output=(args.output or (SMOKE_OUTPUT if args.smoke else DEFAULT_OUTPUT)).resolve()
    if args.workers<1:parser.error('workers must be positive')
    return args

def main(argv=None):
    args=parse_args(argv)
    if args.stage=='watch':return old.previous.watch(args)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(pid=os.getpid(),phase='prepare',workers=args.workers,started=time.time())
        mutex=threading.Lock();stop=threading.Event()
        def status(phase,**fields):
            with mutex:
                state.update(phase=phase,updated_unix=time.time(),**fields)
                children=[p.pid for p in mp.active_children() if p.is_alive()]
                state.update(actual_workers=len(children),worker_pids=children,
                    committed_cache=len(list((args.output/'cache_checksums').glob('*.lock.json'))))
                o.write_json(args.output/'status.json',state)
        def heartbeat():
            while not stop.wait(5):status(state['phase'])
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:
            cases=prepare(args);status('prepared',eligible_scenes=len(cases))
            if args.stage=='prepare':return
            items=requests(args,cases)
            if args.stage in ('pipeline','pilot'):rows=run(args,items,status)
            else:rows=[read_cached(args,i) for i in items]
            if args.stage in ('pipeline','pilot','report'):
                status('report');report(args,cases,items,rows)
            if args.stage in ('pipeline','replay'):status('replay');replay(args,items)
            status('complete',reason='smoke_complete' if args.smoke else 'mechanism_experiment_complete',
                recommended='old0:novalue',elapsed_seconds=time.time()-state['started'])
        except BaseException as error:
            status('failed',reason=f'{type(error).__name__}: {error}');raise
        finally:stop.set();thread.join(timeout=1)

if __name__=='__main__':main()
