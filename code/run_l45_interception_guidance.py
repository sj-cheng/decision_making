"""Five-group development experiment for post-capture interception guidance."""
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
import shutil
import threading
import time
import numpy as np
import torch
import run_l45_search_factorial as previous
from solvers.interception_puct import InterceptionPUCT, cpp

torch.set_num_threads(1)
o,s=previous.o,previous.s
ROOT,CODE=o.ROOT,o.CODE
SOURCE=previous.DEFAULT_OUTPUT
DEFAULT_OUTPUT=ROOT/'current/l45_interception_guidance_20261024'
SMOKE_OUTPUT=ROOT/'current/l45_interception_guidance_smoke_20261024'
UIDS=('B','S','P','R','PR')
lock_json,locked_read=previous.lock_json,previous.locked_read

def config(args):
    return dict(schema=1,smoke=args.smoke,training_episodes=0,source_episode=2048,
        budget=3 if args.smoke else 1000,opponent_budget=3 if args.smoke else 3000,depth=100,
        pursuer_betas=[.5,0.],evader_betas=[.5,.1],uids=list(UIDS),
        activate='one_real_active_evader_and_one_real_active_pursuer_at_search_root',
        graph_offset_radius=.05,heading_separation_degrees=30,
        forecast_steps=[1,2,4,8,12,16,24],root_mixture=[.5,.25,.25],noise_fraction=.1,
        rollout_guided_probability=.8,rollout_prefix_steps=8,
        splits={'pilot_four':['four',1 if args.smoke else 100],
                'smoke_none':['none',1] if args.smoke else ['none',0],
                'timing_four':['four',0 if args.smoke else 30],
                'warmup_four':['four',0 if args.smoke else 1]},
        scenario_namespace=5502026102401 if args.smoke else 5502026102402,
        bootstrap_seed=54020261024,bootstrap_draws=10000,
        minimum_net_gained=5,maximum_extra_own_illegal=2,
        request_caps={'pilot':10 if args.smoke else 500,'timing':90,'warmup':3},
        recommendation='old0:novalue',development_only=True)

def settings(args,role=1):
    return dict(search_budget=config(args)['budget'],search_only_budget=config(args)['opponent_budget'],
        search_depth=100,beta_policy=.5,beta_value=0. if role else .1)

def registry(args):
    directory=args.output/'models/O'
    return {'O':dict(models=str(directory),model_hash=o.model_fingerprint(directory))}

def fingerprints():
    paths=[Path(__file__),CODE/'cpp/solvers/interception_guidance.hpp',CODE/'build_interception_search.py',
           CODE/'solvers/interception_puct.py',CODE/'L45_INTERCEPTION_GUIDANCE.md',
           CODE/'cpp/build_interception/build_manifest.json']
    binary=Path(cpp.__file__)
    return dict(files={str(p):o.sha(p) for p in paths+[binary]},parent=previous.fingerprints())

def prepare(args):
    if (args.output/'experiment.lock.json').exists():
        record=locked_read(args.output/'experiment.json')
        if record['config']!=config(args) or record['implementation']!=fingerprints() or record['models']!=registry(args):
            raise ValueError('Experiment settings or implementation changed')
        for path,sha in record['sources'].items():
            if o.sha(path)!=sha:raise ValueError('Frozen source changed: '+path)
        return locked_read(args.output/'scenarios.json')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.output==SOURCE or SOURCE in args.output.parents:raise ValueError('Use isolated output')
    origin=SOURCE/'models/O';dest=args.output/'models/O';dest.mkdir(parents=True,exist_ok=True)
    for name in o.MODEL_NAMES:shutil.copyfile(origin/name,dest/name)
    history=s.history_files(args);used=s.history_scenes(history);old_n=len(used)
    banks={}
    for phase,(name,(layout,n)) in enumerate(config(args)['splits'].items(),3000):
        cases=[];index=0
        while len(cases)<n:
            spec=o.make_scenario(o.scenario_seed(phase,config(args)['scenario_namespace'],index),layout);index+=1
            physical=s.previous.physical_hash(spec)
            if physical not in used:used.add(physical);cases.append(spec)
        banks[name]=cases
    lock_json(args.output/'scenarios.json',banks)
    sources={str(origin/name):o.sha(origin/name) for name in o.MODEL_NAMES}
    sources[str(args.output/'scenarios.json')]=o.sha(args.output/'scenarios.json')
    lock_json(args.output/'experiment.json',dict(config=config(args),implementation=fingerprints(),
        models=registry(args),sources=sources,history={str(p):o.sha(p) for p in history}))
    lock_json(args.output/'scenario_independence.json',dict(historical_scenes=old_n,
        new_scenes=len(used)-old_n,overlap=0,all_stages_frozen=True))
    shutil.copyfile(CODE/'L45_INTERCEPTION_GUIDANCE.md',args.output/'PLAN.md')
    return banks

def effective(uid,reg):
    return dict(model='O',weights=reg['O']['model_hash'],
        algorithm='legacy' if uid=='B' else 'sync',proposal=uid in ('P','PR'),rollout=uid in ('R','PR'),
        activation='real_post_first_capture',evader='complete_O_legacy')

def requests(args,uids,scenes,timing='parallel'):
    reg=registry(args);items=[]
    for uid in uids:
        for spec in scenes:
            behavior=effective(uid,reg)
            identity=dict(protocol=o.sha(args.output/'experiment.json'),policy_id=o.json_hash(behavior),
                episode_hash=spec['episode_hash'],role=1,opponent='Search Only',timing_context=timing,
                own_settings=settings(args),opponent_settings=settings(args,0),mode='search')
            key=o.json_hash(identity)
            task=dict(**identity,key=key,uid=uid,spec=spec,registry=reg,effective=behavior)
            items.append(dict(key=key,uid=uid,task=task))
    return items

def make_own(p,reg,uid,requested,seed):
    paths,value_path=previous.research.h.oracle_paths(str(ROOT),iteration=45,model_dir=reg['O']['models'])
    with contextlib.redirect_stdout(io.StringIO()):
        policies,value=previous.research.h.get_oracles(p,value_oracle_name='deterministic',value_oracle_path=value_path,
            policy_oracle_name='gaussian',policy_oracle_paths=paths)
    raw=InterceptionPUCT(policies,value,requested['search_budget'],requested['beta_policy'],
        requested['beta_value'],uid,requested['search_depth'])
    raw.solver_wrapper.set_seed(int(seed))
    return o.RoleMaskedSolver(raw,o.TEAMS[1])

def make_pair(task):
    p,state=o.restore_scenario(task['spec']);p.clear_closed_loop_cache()
    own=make_own(p,task['registry'],task['uid'],task['own_settings'],task['spec']['search_seeds'][1])
    other=previous.base.private_solver(p,task['registry']['O'],task['opponent'],0,
        task['opponent_settings'],task['spec']['search_seeds'][0])
    return p,state,own,other

def simulate(p,state,own,other,stats=None):
    stats=previous.initial_stats() if stats is None else copy.deepcopy(stats)
    initial=previous.snapshot(p,state,stats,own,other)
    team=o.TeamSolver(other,own);trace=[];roots=[];decisions=[];first=None;deployment_ms=0.
    while not p.is_terminal(state):
        phase=int(stats['capture_count']>0);cost_before=dict(own.solver.cost)
        action=team.policy(p,state)
        latency=[team.last_pursuer_latency_ms,team.last_evader_latency_ms,team.last_joint_latency_ms]
        decisions.append(latency)
        started=time.perf_counter();nxt=previous.advance(p,state,action,stats)
        deployment_ms+=latency[2]+(time.perf_counter()-started)*1000
        result=own.solver.last_result
        if not np.isfinite(action).all():raise ValueError('Nonfinite action')
        trace.append(dict(state=state.copy(),action=action.copy(),next_state=nxt.copy()))
        roots.append(dict(phase=phase,table=np.asarray(result.root_audit).copy(),
            guidance=dict(getattr(result,'guidance_stats',{})),
            routes=np.asarray(getattr(result,'guidance_routes',np.zeros((0,12)))).copy(),
            paths=np.asarray(getattr(result,'guidance_paths',np.zeros((0,4)))).copy(),
            sources=list(getattr(result,'root_sources',[])),
            visits=int(getattr(result,'num_visits',np.asarray(result.root_audit)[:,1].sum())),
            cost={k:own.solver.cost[k]-cost_before[k] for k in cost_before}))
        state=nxt
        if first is None and phase==0 and stats['capture_count']:
            first=previous.snapshot(p,state,stats,own,other)
            first.update(next_decision=len(trace),branch_available=bool(stats['capture_count']==1 and not p.is_terminal(state)))
        if stats['steps']>int(p.tf/p.dt):raise RuntimeError('Exceeded horizon')
    return previous.metrics(p,state,stats),dict(initial=initial,first_capture=first,trajectory=trace,roots=roots,
        final=previous.snapshot(p,state,stats,own,other)),decisions,deployment_ms

def evaluate_one(task):
    previous.seed_task(task);p,state,own,other=make_pair(task);started=time.perf_counter()
    row,detail,decisions,deployment_ms=simulate(p,state,own,other)
    row.update({k:task[k] for k in ('key','policy_id','role','opponent','timing_context','mode')})
    row.update(uid=task['uid'],episode_hash=task['spec']['episode_hash'],layout=task['spec']['layout'],
        search_seeds=task['spec']['search_seeds'],actual_own=dict(task['own_settings'],effective=task['effective']),
        actual_opponent=other.actual_settings,own_cost=own.solver.cost,own_mcts_calls=own.solver.calls,
        episode_wall_ms=(time.perf_counter()-started)*1000,deployment_episode_ms=deployment_ms,
        trajectory_sha256=previous.trajectory_digest(detail['trajectory']),decision_latency_ms=decisions)
    row['_record']=dict(key=task['key'],scenario=task['spec'],effective=task['effective'],role=1,opponent=task['opponent'],**detail)
    return row

commit_row=previous.commit_row
read_cached=previous.read_cached
read_episode=previous.read_episode

def run_evaluation(args,name,items,pool,status):
    if len(items)>config(args)['request_caps'][name]:raise ValueError('Stage budget exceeded')
    lock_json(args.output/name/'manifest.json',[{k:v for k,v in i.items() if k!='task'} for i in items])
    missing=[]
    for item in items:
        if (args.output/'cache_checksums'/(item['key']+'.lock.json')).exists():read_cached(args,item)
        else:missing.append(item['task'])
    done=len(items)-len(missing);start=done;began=time.perf_counter()
    status(name,completed=done,total=len(items),actual_execution_requests=len(missing),reused=done)
    iterator=pool.imap_unordered(evaluate_one,missing,chunksize=1) if pool else map(evaluate_one,missing)
    for row in iterator:
        commit_row(args,row);done+=1
        if done%5==0 or done==len(items):status(name,completed=done,total=len(items),
            throughput_per_second=(done-start)/max(.001,time.perf_counter()-began))
    return [read_cached(args,i) for i in items]

def paired_rows(rows):
    parts={u:{r['episode_hash']:r for r in rows if r['uid']==u} for u in UIDS}
    keys=sorted(parts['S'])
    if not keys or any(len(parts[u])!=sum(r['uid']==u for r in rows) or set(parts[u])!=set(keys) for u in UIDS):
        raise ValueError('Missing or duplicate scene pair')
    for key in keys:
        if any(parts[u][key]['search_seeds']!=parts['S'][key]['search_seeds'] for u in UIDS):raise ValueError('Unpaired search seeds')
    return parts,keys

def selection(rows,smoke=False):
    parts,keys=paired_rows(rows)
    summary={u:dict(games=len(keys),successes=sum(r['legal_double_capture'] for r in parts[u].values()),
        own_illegal=sum(r['pursuer_illegal'] for r in parts[u].values()),
        zero_capture=sum(r['capture_count']==0 for r in parts[u].values()),
        one_capture_timeout=sum(r['one_capture_timeout'] for r in parts[u].values()),
        mean_completion_cost=float(np.mean([r['completion_cost'] for r in parts[u].values()]))) for u in UIDS}
    comparisons={}
    for u in ('P','R','PR'):
        comparisons[u]={}
        for ref in ('B','S'):
            a,b=summary[u],summary[ref]
            comparisons[u][ref]=dict(net_gained=a['successes']-b['successes'],
                extra_own_illegal=a['own_illegal']-b['own_illegal'],
                gained=sum(parts[u][k]['legal_double_capture']>parts[ref][k]['legal_double_capture'] for k in keys),
                lost=sum(parts[u][k]['legal_double_capture']<parts[ref][k]['legal_double_capture'] for k in keys))
    eligible=[u for u in ('P','R','PR') if not smoke and all(v['net_gained']>=5 and v['extra_own_illegal']<=2 for v in comparisons[u].values())]
    eligible.sort(key=lambda u:(-summary[u]['successes'],summary[u]['own_illegal'],summary[u]['mean_completion_cost'],('P','R','PR').index(u)))
    return dict(summary=summary,comparisons=comparisons,passed=bool(eligible),
        selected=eligible[0] if eligible else None,development_only=True,recommended='old0:novalue')

def statistics(rows):
    parts,keys=paired_rows(rows);rng=np.random.default_rng(54020261024)
    indices=rng.integers(0,len(keys),size=(10000,len(keys)))
    vectors={u:np.array([parts[u][k]['legal_double_capture'] for k in keys],float) for u in UIDS}
    diffs={u+'-S':vectors[u]-vectors['S'] for u in ('P','R','PR')}
    diffs['interaction']=vectors['PR']-vectors['P']-vectors['R']+vectors['S']
    for u in ('P','R','PR'):diffs[u+'-B']=vectors[u]-vectors['B']
    return dict(seed=54020261024,draws=10000,exploratory=True,comparisons={
        name:dict(delta=float(x.mean()),ci95=np.quantile(x[indices].mean(axis=1),[.025,.975]).tolist()) for name,x in diffs.items()})

def prefix_check(args,items):
    records={u:{} for u in UIDS}
    for item in items:
        _,d=read_episode(args,item);records[item['uid']][item['task']['spec']['episode_hash']]=d
    n=0
    for key,baseline in records['S'].items():
        first=baseline['first_capture'];count=first['next_decision'] if first else len(baseline['trajectory'])
        for u in ('P','R','PR'):
            d=records[u][key]
            if not s.equal_tree(baseline['trajectory'][:count],d['trajectory'][:count]):raise ValueError('Pre-capture trajectory differs: '+u)
            if not s.equal_tree(first,d['first_capture']):raise ValueError('First capture snapshot/RNG differs: '+u)
            if any(r['guidance'].get('active',0) for r in d['roots'][:count]):raise ValueError('Premature guidance')
        n+=1
    return dict(passed=True,scenes=n,first_capture_state_and_rng_exact=True)

def timing(args,banks,result,status):
    if not result['passed'] or args.smoke:raise ValueError('Timing requires a passing development candidate')
    uids=('B','S',result['selected'])
    run_evaluation(args,'warmup',requests(args,uids,banks['warmup_four'],'warmup'),None,status)
    items=[]
    for i,spec in enumerate(banks['timing_four']):
        order=uids[i%3:]+uids[:i%3]
        items+=requests(args,order,[spec],'single_process')
    rows=run_evaluation(args,'timing',items,None,status);out={}
    for u in uids:
        part=[r for r in rows if r['uid']==u];values=[v[0] for r in part for v in r['decision_latency_ms']]
        post=[]
        for item in items:
            if item['uid']!=u:continue
            row,d=read_episode(args,item)
            post.extend(v[0] for v,root in zip(row['decision_latency_ms'],d['roots']) if root['phase'])
        out[u]=dict(games=len(part),mean_step_ms=float(np.mean(values)),p95_step_ms=float(np.quantile(values,.95)),
            mean_post_capture_ms=float(np.mean(post)) if post else None,p95_post_capture_ms=float(np.quantile(post,.95)) if post else None,
            mean_joint_episode_ms=float(np.mean([r['deployment_episode_ms'] for r in part])),
            mean_pursuit_episode_ms=float(np.mean([sum(v[0] for v in r['decision_latency_ms']) for r in part])))
    lock_json(args.output/'timing_result.json',out);return out

def diagnostic_data(args,items):
    result={u:dict(phase_cost={str(i):dict(calls=0,step_calls=0,policy_evaluations=0,value_evaluations=0) for i in (0,1)},
        guidance_totals={},post_capture_calls=0,available_calls=0,max_guided_prefix=0,selected_sources={},
        conditional_second_capture={},speed_groups={}) for u in UIDS}
    records=[]
    for item in items:
        row,d=read_episode(args,item);uid=item['uid'];out=result[uid]
        for root in d['roots']:
            phase=out['phase_cost'][str(root['phase'])];phase['calls']+=1
            for k,v in root['cost'].items():phase[k]+=v
            g=root['guidance']
            if root['phase']:
                out['post_capture_calls']+=1;out['available_calls']+=int(g.get('routes',0)>0)
                source=str(int(g.get('selected_source',-1)))
                out['selected_sources'][source]=out['selected_sources'].get(source,0)+1
            out['max_guided_prefix']=max(out['max_guided_prefix'],g.get('max_guided_prefix',0))
            for k,v in g.items():out['guidance_totals'][k]=out['guidance_totals'].get(k,0)+v
        ratio=d['scenario']['pursuer_speed']/d['scenario']['evader_speed']
        records.append(dict(uid=uid,scene=row['episode_hash'],speed_ratio=ratio,
            success=row['legal_double_capture'],first_capture_time=row['first_capture_time'],
            second_capture_time=row['capture_time']-row['first_capture_time'] if row['legal_double_capture'] else None,
            completion_cost=row['completion_cost'],illegal=row['pursuer_illegal'],
            trajectory=row['episode_record']['path']))
    for u in UIDS:
        part=[r for r in records if r['uid']==u]
        for name,test in (('<1.2',lambda x:x<1.2),('>=1.2',lambda x:x>=1.2)):
            group=[r for r in part if test(r['speed_ratio'])]
            result[u]['speed_groups'][name]=dict(n=len(group),successes=sum(r['success'] for r in group))
        times=[r['second_capture_time'] for r in part if r['second_capture_time'] is not None]
        result[u]['conditional_second_capture']=dict(n=len(times),mean=float(np.mean(times)) if times else None)
    return result,records

def plots(args,items,rows,result,diagnostics,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination=args.output/'figures';destination.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(UIDS,[result['summary'][u]['successes']/result['summary'][u]['games'] for u in UIDS])
    axes[0].set(ylabel='Legal double-capture rate',ylim=(0,1),title='100 paired physical scenes')
    axes[1].bar(UIDS,[result['summary'][u]['mean_completion_cost'] for u in UIDS]);axes[1].set(ylabel='Completion cost (failure = 80)',title='Completion cost')
    fig.tight_layout();fig.savefig(destination/'ablation.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for u in UIDS:
        part=sorted([r for r in records if r['uid']==u],key=lambda r:r['speed_ratio'])
        axes[0].scatter([r['speed_ratio'] for r in part],[r['completion_cost'] for r in part],s=12,alpha=.45,label=u)
    axes[0].set(xlabel='Pursuer / evader action-limit ratio',ylabel='Completion cost');axes[0].legend()
    x=np.arange(2)
    for i,u in enumerate(UIDS):
        groups=[diagnostics[u]['speed_groups'][g] for g in ('<1.2','>=1.2')]
        axes[1].bar(x+(i-2)*.15,[g['successes']/g['n'] if g['n'] else 0 for g in groups],width=.15,label=u)
    axes[1].set(xticks=x,xticklabels=['<1.2','>=1.2'],ylim=(0,1),ylabel='Legal double-capture rate');axes[1].legend()
    fig.tight_layout();fig.savefig(destination/'speed_ratio.png',dpi=180);plt.close(fig)
    parts,keys=paired_rows(rows);chosen=result['selected'] or max(('P','R','PR'),key=lambda u:result['summary'][u]['successes'])
    cases={name:[k for k in keys if condition(parts[chosen][k]['legal_double_capture'],parts['S'][k]['legal_double_capture'])]
           for name,condition in [('gained',lambda a,b:a>b),('lost',lambda a,b:a<b),('both_failed',lambda a,b:a==b==0)]}
    index={}
    for category,scenes in cases.items():
        if not scenes:continue
        scene=scenes[0];fig,axes=plt.subplots(1,2,figsize=(11,5))
        for ax,u in zip(axes,('S',chosen)):
            item=next(i for i in items if i['uid']==u and i['task']['spec']['episode_hash']==scene)
            row,d=read_episode(args,item);p,_=o.restore_scenario(d['scenario'])
            for box in p.obstacles:
                ax.add_patch(plt.Rectangle((box[0,0],box[1,0]),box[0,1]-box[0,0],box[1,1]-box[1,0],color='gray',alpha=.5))
            states=np.array([t['state'].reshape(-1) for t in d['trajectory']]+[d['final']['state'].reshape(-1)])
            for r in range(4):ax.plot(states[:,2*r],states[:,2*r+1],label=f'robot {r}')
            first=d['first_capture']
            if first and first['next_decision']<len(d['roots']):
                root=d['roots'][first['next_decision']];table=root['routes'];paths=root['paths']
                if paths.size:
                    for route in np.unique(paths[:,0]):
                        for kind in (0,1):
                            points=paths[(paths[:,0]==route)&(paths[:,1]==kind)]
                            ax.plot(points[:,2],points[:,3],'k--' if kind==0 else 'r:',alpha=.5)
                if table.size:
                    for route in np.unique(table[:,0]):
                        group=table[table[:,0]==route];ax.scatter(group[:,2],group[:,3],s=8,color='black')
                        selected=group[group[:,8]>.5];ax.scatter(selected[:,4],selected[:,5],marker='*',s=120,color='red')
            ax.set(title=f'{u}, success={row["legal_double_capture"]}',xlim=p.state_lims[0],ylim=p.state_lims[1],aspect='equal');ax.legend(fontsize=7)
        fig.suptitle(category+' / '+scene[:12]);fig.tight_layout();fig.savefig(destination/(category+'.png'),dpi=180);plt.close(fig)
        index[category]=dict(scene=scene,candidate=chosen)
    o.write_json(args.output/'representative_trajectories.json',index)

def report(args,banks,result):
    items=requests(args,UIDS,banks['pilot_four']+banks['smoke_none'])
    rows=[read_cached(args,i) for i in items]
    prefix=prefix_check(args,items);lock_json(args.output/'prefix_check.json',prefix)
    diagnostics,records=diagnostic_data(args,items)
    o.write_json(args.output/'diagnostics.json',diagnostics);o.write_json(args.output/'trajectory_index.json',records)
    if args.smoke:
        (args.output/'REPORT.md').write_text('隔离 smoke：10局低预算，前缀/首捕状态与随机流一致。无效果结论。\n')
        return
    stats=statistics(rows);o.write_json(args.output/'statistics.json',stats)
    parts,keys=paired_rows(rows)
    o.write_json(args.output/'paired_results.json',[dict(scene=k,models={u:{name:parts[u][k][name] for name in
        ('legal_double_capture','pursuer_illegal','capture_count','first_capture_time','capture_time','completion_cost','one_capture_timeout')} for u in UIDS}) for k in keys])
    plots(args,items,rows,result,diagnostics,records)
    lines=['首捕后的障碍感知拦截引导：五组开发实验','',
        '固定O/PPO第2048局权重；无新增训练；追捕1000模拟，对手Search Only 3000模拟。',
        'P仅改首捕后的根候选；R仅改rollout前八步。当前部署推荐保持old0:novalue。','',
        '|配置|合法双捕|自身非法|零捕|首捕后超时|完成成本|','|---|---:|---:|---:|---:|---:|']
    for u in UIDS:
        r=result['summary'][u]
        lines.append(f'|{u}|{r["successes"]}/{r["games"]}|{r["own_illegal"]}|{r["zero_capture"]}|{r["one_capture_timeout"]}|{r["mean_completion_cost"]:.2f}|')
    lines+=['',f'开发门槛：{"通过" if result["passed"] else "未通过"}；下一阶段候选：{result["selected"]}。','',
        '逐场景挽回、丢失：']
    for u,comparisons in result['comparisons'].items():
        for ref,v in comparisons.items():lines.append(f'{u}−{ref}：挽回{v["gained"]}、丢失{v["lost"]}，净增{v["net_gained"]}，自身非法增加{v["extra_own_illegal"]}场。')
    lines+=['','物理场景配对bootstrap，普通95%区间（开发集探索性结果）：']
    for name,v in stats['comparisons'].items():lines.append(f'{name}: {100*v["delta"]:+.2f}点，区间[{100*v["ci95"][0]:+.2f},{100*v["ci95"][1]:+.2f}]点。')
    lines+=['','同首捕局面核验通过：S/P/R/PR 的首捕前轨迹、首捕状态、双方随机流完全一致。',
        '引导统计与速度分层见diagnostics.json；全部轨迹见trajectory_index.json；根候选来源、路线与拦截点见episodes/*.pt。',
        '几何表的静态建表、每次调用规划与搜索时间分别记录；并行耗时不能解释为单进程部署耗时。']
    timing_path=args.output/'timing_result.json'
    if timing_path.exists():lines+=['','单进程计时：',str(locked_read(timing_path))]
    else:lines+=['','未执行单进程计时：开发门槛未通过。']
    if result['passed']:
        lines+=['','下一步：锁定所选组成部分，登记新的独立场景验证，扩展对手、地图、速度和等耗时对照。']
    else:
        lines+=['','下一步：结束本版本，不扩样本或调参数。根据引导可用率、挽回/丢失和失败轨迹分析路线假设与到达时间估计。']
    lines+=['','论文定位：拦截机会表示及其在神经搜索中的应用；本轮为一个固定检查点的开发结果。PPO是权重来源，创新与有效性由机制和消融结果支持。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (args.output/'CONCLUSION.md').write_text('\n'.join(lines[:15])+f'\n\n开发门槛通过={result["passed"]}，候选={result["selected"]}。\n')

def replay(args,banks):
    item=requests(args,['PR'],banks['pilot_four'][:1])[0]
    row,d=read_episode(args,item);again=evaluate_one(item['task'])
    if previous.science(row)!=previous.science(again) or row['trajectory_sha256']!=again['trajectory_sha256']:
        raise ValueError('Replay differs')
    for side in ('own_rng','opponent_rng'):
        if not s.equal_tree(d['final'][side],again['_record']['final'][side]):raise ValueError('Final RNG differs')
    lock_json(args.output/'replay.json',dict(passed=True,full_episode=True,both_rng_exact=True))

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='pipeline',choices=['pipeline','prepare','pilot','timing','report','watch','replay'])
    p.add_argument('--workers',type=int,default=80);p.add_argument('--output',type=Path)
    p.add_argument('--smoke',action='store_true');p.add_argument('--once',action='store_true')
    a=p.parse_args(argv);a.output=(a.output or (SMOKE_OUTPUT if a.smoke else DEFAULT_OUTPUT)).resolve()
    if a.workers<1:p.error('workers must be positive')
    return a

def main(argv=None):
    args=parse_args(argv)
    if args.stage=='watch':return previous.watch(args)
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
                    worker_threads={str(pid):len(list(Path('/proc',str(pid),'task').glob('*'))) for pid in children},
                    committed_cache=len(list((args.output/'cache_checksums').glob('*.lock.json'))))
                o.write_json(args.output/'status.json',state)
                if children:
                    o.write_json(args.output/'worker_snapshot.json',dict(pids=children,workers=len(children),
                        threads=state['worker_threads'],observed_unix=time.time()))
        def heartbeat():
            while not stop.wait(5):status(state['phase'])
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:
            banks=prepare(args);status('prepared')
            if args.stage=='prepare':return
            if args.stage in ('pipeline','pilot'):
                items=requests(args,UIDS,banks['pilot_four']+banks['smoke_none'])
                with mp.get_context('spawn').Pool(args.workers) as pool:
                    o.write_json(args.output/'worker_snapshot.json',dict(pids=[p.pid for p in pool._pool],workers=args.workers,threads_per_worker=1))
                    rows=run_evaluation(args,'pilot',items,pool,status)
                result=selection(rows,args.smoke);lock_json(args.output/'selection.json',result)
                lock_json(args.output/'prefix_check.json',prefix_check(args,items))
            else:result=locked_read(args.output/'selection.json')
            if args.stage in ('pipeline','timing') and result['passed'] and not args.smoke:timing(args,banks,result,status)
            if args.stage in ('pipeline','report','pilot'):report(args,banks,result)
            if args.stage in ('pipeline','replay'):replay(args,banks)
            lock_json(args.output/'results.json',dict(reason='smoke_complete' if args.smoke else 'development_passed' if result['passed'] else 'development_no_gain',
                selected=result['selected'],recommended='old0:novalue',training_episodes=0))
            status('complete',reason='smoke_complete' if args.smoke else 'development_passed' if result['passed'] else 'development_no_gain',
                selected=result['selected'],recommended='old0:novalue',elapsed_seconds=time.time()-state['started'])
        except BaseException as e:status('failed',reason=f'{type(e).__name__}: {e}');raise
        finally:stop.set();thread.join(timeout=1)

if __name__=='__main__':main()
