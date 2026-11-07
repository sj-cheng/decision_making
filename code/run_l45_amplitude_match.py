"""M versus historical F/S; at most 198 new closed-loop M suffixes."""
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
import random
import shutil
import threading
import time
import types
import numpy as np
import torch
import run_l45_candidate_ablation as prior
from solvers.amplitude_match import AmplitudeMatchPUCT,cpp

torch.set_num_threads(1)
o,old=prior.o,prior.old
ROOT,CODE=o.ROOT,o.CODE
SOURCE=prior.DEFAULT_OUTPUT
DEFAULT_OUTPUT=ROOT/'current/l45_amplitude_match_20261024'
UIDS=('S','M','F')
lock_json,locked_read=prior.lock_json,prior.locked_read
commit_row,read_cached,read_episode=prior.commit_row,prior.read_cached,prior.read_episode
CandidateAblationPUCT=AmplitudeMatchPUCT
stream_seed=prior.stream_seed

def clone(method):
    return types.FunctionType(method.__code__,globals(),method.__name__,method.__defaults__,method.__closure__)

# Reuse frozen restore, physical simulation, cache and telemetry without mutations.
make_own=clone(prior.make_own)
make_pair=clone(prior.make_pair)
restore=clone(prior.restore)
decorate=clone(prior.decorate)
evaluate_one=clone(prior.evaluate_one)
diagnostics=clone(prior.diagnostics)

def config():
    return dict(schema=1,source=str(SOURCE),training_episodes=0,new_method='M',uids=list(UIDS),
        states=99,streams=[0,1],new_suffix_cap=198,historical_F=198,historical_S=198,
        search_budget=1000,opponent_budget=3000,depth=100,pursuer_betas=[.5,0.],
        only_change='M base=physical mean; F base=mean/max(abs(mean)/axis_upper)',
        mixture=[.5,.25,.25],noise_fraction=.1,stream_management='exact_source_restore',
        primary='M-F',secondary='M-S',bootstrap_seed=57020261024,bootstrap_draws=20000,
        cluster_unit='physical_scene; average_both_streams_before_bootstrap',recommended='old0:novalue')

def fingerprints():
    files=[Path(__file__),CODE/'build_amplitude_match.py',CODE/'solvers/amplitude_match.py',
        CODE/'L45_AMPLITUDE_MATCH.md',CODE/'cpp/build_amplitude_match/build_manifest.json',Path(cpp.__file__)]
    return dict(files={str(p):o.sha(p) for p in files},parent=prior.fingerprints())

def registry(args):
    directory=args.output/'models/O'
    return {'O':dict(models=str(directory),model_hash=o.model_fingerprint(directory))}

def source_data(restore_all=False):
    args=prior.parse_args([]);cases=prior.prepare(args)
    items=[i for i in prior.requests(args,cases) if i['uid'] in ('S','F')]
    refs={};sources={str(SOURCE/n):o.sha(SOURCE/n) for n in ('experiment.json','cases.json','results.json','statistics.json','integrity.json')}
    for item in items:
        row,d=prior.read_episode(args,item);task=item['task'];snap=o.load_torch(task['snapshot_path'])
        if row['actual_own']['search_budget']!=1000 or row['actual_own']['search_depth']!=100 or \
                row['actual_own']['beta_policy']!=.5 or row['actual_own']['beta_value']!=0. or \
                row['actual_opponent']!=dict(method='Search Only',budget=3000,depth=100,beta_policy=0.,beta_value=0.):
            raise ValueError('Incompatible historical search settings')
        for name in ('state','stats','time','remaining_time'):
            if not old.s.equal_tree(d['initial'][name],snap[name]):raise ValueError('Source snapshot or remaining time differs')
        if restore_all:
            p,state,own,other,_=prior.restore(task)
            if not old.s.equal_tree(state,d['initial']['state']) or \
                not old.s.equal_tree(own.solver.get_rng_state(p),d['initial']['own_rng']) or \
                not old.s.equal_tree(other.solver.get_rng_state(p),d['initial']['opponent_rng']) or \
                not old.s.equal_tree(o.rng_state(),d['initial']['python_rng']):
                raise ValueError('Historical random state is not reproducible')
        metric=SOURCE/'cache'/(item['key']+'.json')
        ref=dict(key=item['key'],metric=str(metric),record=row['episode_record']['path'])
        refs[(task['scene_id'],task['stream'],item['uid'])]=dict(item=item,reference=ref)
        sources[str(metric)]=o.sha(metric);sources[ref['record']]=o.sha(ref['record'])
    if len(cases)!=99 or len(refs)!=396:raise ValueError('Expected all 99 states, both streams and both historical controls')
    return cases,refs,sources

def prepare(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if out==SOURCE or SOURCE in out.parents:raise ValueError('Use an independent output directory')
    if (out/'experiment.lock.json').exists():
        record=locked_read(out/'experiment.json')
        if record['config']!=config() or record['implementation']!=fingerprints() or record['models']!=registry(args):
            raise ValueError('Frozen implementation or model configuration changed')
        for path,sha in record['sources'].items():
            if o.sha(path)!=sha:raise ValueError('Frozen source changed: '+path)
        return locked_read(out/'cases.json')
    source_cases,refs,sources=source_data(restore_all=True)
    dest=out/'models/O';dest.mkdir(parents=True,exist_ok=True)
    for name in o.MODEL_NAMES:shutil.copyfile(SOURCE/'models/O'/name,dest/name)
    if registry(args)['O']['model_hash']!=prior.registry(prior.parse_args([]))['O']['model_hash']:
        raise ValueError('Model copy differs')
    cases=[]
    for original in source_cases:
        case=copy.deepcopy(original);path=out/'snapshots'/(case['scene_id']+'.pt');path.parent.mkdir(exist_ok=True)
        if not path.exists():shutil.copyfile(case['snapshot_path'],path)
        if o.sha(path)!=case['snapshot_sha256']:raise ValueError('Snapshot copy differs')
        case.update(snapshot_path=str(path),historical={})
        for stream in (0,1):
            for uid in ('S','F'):
                ref=refs[(case['scene_id'],stream,uid)]
                case['historical'][f'{uid}:{stream}']=dict(reference=ref['reference'],spec=ref['item']['task']['spec'])
        sources[str(path)]=o.sha(path);cases.append(case)
    lock_json(out/'cases.json',cases);sources[str(out/'cases.json')]=o.sha(out/'cases.json')
    lock_json(out/'source_compatibility.json',dict(passed=True,source=str(SOURCE),states=99,streams=2,
        historical_F_verified=198,historical_S_verified=198,all_396_snapshot_and_rng_restores_exact=True,
        weights_exact=True,remaining_time_preserved=True))
    lock_json(out/'experiment.json',dict(config=config(),implementation=fingerprints(),models=registry(args),sources=sources))
    shutil.copyfile(CODE/'L45_AMPLITUDE_MATCH.md',out/'PLAN.md')
    runtime=out/'runtime';runtime.mkdir(exist_ok=True)
    shutil.copyfile(Path(cpp.__file__),runtime/Path(cpp.__file__).name)
    shutil.copyfile(CODE/'cpp/build_amplitude_match/build_manifest.json',runtime/'build_manifest.json')
    shutil.copytree(CODE/'cpp/build_amplitude_match/source',runtime/'source',dirs_exist_ok=True)
    shutil.copyfile(CODE/'solvers/amplitude_match.py',runtime/'amplitude_match.py')
    return cases

def requests(args,cases):
    reg=registry(args);items=[]
    for case in cases:
        for stream in (0,1):
            for uid in UIDS:
                ref=case['historical'][f'{"F" if uid=="M" else uid}:{stream}']
                behavior=prior.effective('F' if uid=='M' else uid,reg);behavior['root_proposal']=uid
                requested=dict(search_budget=1000,search_only_budget=3000,search_depth=100,beta_policy=.5,beta_value=0.)
                identity=dict(protocol=o.sha(args.output/'experiment.json'),policy_id=o.json_hash(behavior),
                    episode_hash=ref['spec']['episode_hash'],scene_id=case['scene_id'],stream=stream,
                    snapshot_sha256=case['snapshot_sha256'],role=1,opponent='Search Only',timing_context='parallel',
                    own_settings=requested,opponent_settings=requested,mode='search')
                key=o.json_hash(identity)
                task=dict(**identity,key=key,uid=uid,spec=ref['spec'],registry=reg,effective=behavior,
                    snapshot_path=case['snapshot_path'],historical=None if uid=='M' else ref['reference'])
                items.append(dict(key=key,uid=uid,task=task))
    return items

def reuse_one(task):
    ref=task['historical'];row=o.read_json(ref['metric']);record=o.load_torch(ref['record'])
    metadata={'key','scenario','effective','role','opponent','scene_id','stream'}
    detail={k:v for k,v in record.items() if k not in metadata};detail['matched_source']=ref
    science={k:row[k] for k in old.previous.SCIENCE}
    return decorate(task,science,detail,row['decision_latency_ms'],row['deployment_episode_ms'],0.)

def checks(args,cases,status):
    if (args.output/'checks.lock.json').exists():return locked_read(args.output/'checks.json')
    items=requests(args,cases);source_cases,refs,_=source_data()
    fitems=[i for i in items if i['uid']=='F'];rng_checked=0
    for n,item in enumerate(fitems):
        task=item['task'];ref=refs[(task['scene_id'],task['stream'],'F')]
        row,d=prior.read_episode(prior.parse_args([]),ref['item'])
        p,state,own,_,_=restore(task);actual=own.policy(p,state);root=own.solver.last_result
        if not np.array_equal(root.root_audit,d['roots'][0]['table']) or \
            not np.array_equal(actual[4:],d['trajectory'][0]['action'][4:]):
            raise ValueError('Private F no longer reproduces its cached first root')
        if n<8:
            p2,s2,frozen,_,_=prior.restore(ref['item']['task']);frozen.policy(p2,s2)
            if own.solver.get_rng_state(p)!=frozen.solver.get_rng_state(p2):raise ValueError('F RNG changed in private build')
            rng_checked+=1
        if n%20==0:status('check',completed=n+1,total=198)
    task=next(i['task'] for i in items if i['uid']=='M')
    p,state,own,_,_=restore(task)
    robots=[r for r in range(4) if p.is_active(state,r)];evader,pursuer=robots
    state[p.state_idxs[evader],0]=[-5.,0.];state[p.state_idxs[pursuer],0]=[5.,0.]
    mean=np.array([.2,.4],np.float32);same_noise=0
    for seed in range(20):
        f=cpp.matched_proposal_debug(own.solver._problem_wrapper_for(p),state,1,mean,seed)
        m=cpp.matched_proposal_debug(own.solver._problem_wrapper_for(p),state,4,mean,seed)
        if not f['ok'] or not m['ok'] or f['rng']!=m['rng']:raise ValueError('Matched candidate random draws differ')
        np.testing.assert_allclose(np.asarray(m['base']).reshape(-1),mean,atol=1e-7,rtol=0)
        upper=p.action_lims[p.action_idxs[pursuer],1]
        np.testing.assert_allclose(np.asarray(f['base']).reshape(-1),mean/np.max(abs(mean)/upper),atol=1e-6,rtol=0)
        fa,fb=np.asarray(f['action']).reshape(-1),np.asarray(f['base']).reshape(-1)
        ma,mb=np.asarray(m['action']).reshape(-1),np.asarray(m['base']).reshape(-1)
        for axis in range(2):
            if abs(fa[axis])<upper[axis]-1e-5 and abs(ma[axis])<upper[axis]-1e-5:
                np.testing.assert_allclose(fa[axis]-fb[axis],ma[axis]-mb[axis],atol=1e-6,rtol=0);same_noise+=1
    if same_noise<20:raise ValueError('Insufficient unclipped perturbations checked')
    # One root only: no extra M closed-loop suffix is executed by engineering checks.
    state[p.time_idx,0]=30.;before=own.solver.get_rng_state(p)
    action=own.policy(p,state);result=own.solver.last_result
    if result.num_visits!=1000 or result.root_audit[:,1].sum()!=1000:raise ValueError('M budget changed')
    if result.guidance_stats.get('guided_steps',0)!=0 or result.guidance_stats.get('control_proposals',0)<1:
        raise ValueError('M root/rollout configuration changed')
    for robot in range(4):
        if robot!=pursuer and np.any(action[p.action_idxs[robot]]):raise ValueError('Inactive or opponent action not masked')
    after=own.solver.get_rng_state(p);own.solver.set_rng_state(p,before);again=own.policy(p,state)
    if not np.array_equal(action,again) or after!=own.solver.get_rng_state(p):raise ValueError('M root resume differs')
    return lock_json(args.output/'checks.json',dict(passed=True,private_F_cached_roots_exact=198,
        private_F_rng_pairs_exact=rng_checked,matched_proposal_seed_pairs=20,matched_unclipped_noise_axes=same_noise,
        M_original_mean_exact=True,M_root_budget=1000,M_root_resume_exact=True,
        extra_M_closed_loop_suffixes=0,engineering_M_root_queries=2))

def run(args,items,status):
    if len(items)!=594 or sum(i['uid']=='M' for i in items)!=198:raise ValueError('Scientific budget differs')
    lock_json(args.output/'pilot/manifest.json',[dict(key=i['key'],uid=i['uid'],scene_id=i['task']['scene_id'],stream=i['task']['stream']) for i in items])
    for item in items:
        if (args.output/'cache_checksums'/(item['key']+'.lock.json')).exists():read_cached(args,item)
        elif item['task']['historical']:commit_row(args,reuse_one(item['task']))
    pending=[i['task'] for i in items if not (args.output/'cache_checksums'/(i['key']+'.lock.json')).exists()]
    if any(t['uid']!='M' for t in pending) or len(pending)>198:raise ValueError('Only M may execute new suffixes')
    done=len(items)-len(pending);initial=done;started=time.perf_counter()
    status('pilot',completed=done,total=594,M_completed=198-len(pending),M_total=198,historical_reused=396)
    if pending:
        with mp.get_context('spawn').Pool(min(args.workers,len(pending))) as pool:
            o.write_json(args.output/'worker_snapshot.json',dict(workers=len(pool._pool),pids=[p.pid for p in pool._pool],compute_threads_per_worker=1))
            for row in pool.imap_unordered(evaluate_one,pending,chunksize=1):
                commit_row(args,row);done+=1
                status('pilot',completed=done,total=594,M_completed=done-396,M_total=198,
                    throughput_per_second=(done-initial)/max(.001,time.perf_counter()-started))
    return [read_cached(args,i) for i in items]

def analyze(rows):
    index={};scenes=sorted({r['scene_id'] for r in rows})
    for row in rows:
        key=(row['uid'],row['scene_id'],row['stream'])
        if key in index:raise ValueError('Duplicate pairing')
        index[key]=row
    if len(scenes)!=99 or len(index)!=594:raise ValueError('Missing physical scene/stream pairing')
    for scene in scenes:
        for stream in (0,1):
            seed=index[('F',scene,stream)]['search_seeds']
            if any(index[(u,scene,stream)]['search_seeds']!=seed for u in UIDS):raise ValueError('M/F/S seeds differ')
    rng=np.random.default_rng(57020261024);draws=rng.integers(0,99,size=(20000,99))
    vectors={u:np.array([[index[(u,s,t)]['legal_double_capture'] for t in (0,1)] for s in scenes],float) for u in UIDS}
    diffs={name:(vectors[name[0]]-vectors[name[-1]]).mean(1) for name in ('M-F','M-S','F-S')}
    means=np.array([diffs[n].mean() for n in ('M-F','M-S')])
    samples=np.stack([diffs[n][draws].mean(1) for n in ('M-F','M-S')],1)
    critical=float(np.quantile(np.max(np.abs(samples-means),axis=1),.95))
    lower_critical=float(np.quantile(np.max(samples-means,axis=1),.95))
    comparisons={}
    for name,x in diffs.items():
        a,b=name[0],name[-1];raw=vectors[a]-vectors[b];delta=float(x.mean())
        cost=np.array([np.mean([index[(a,s,t)]['completion_cost']-index[(b,s,t)]['completion_cost'] for t in (0,1)]) for s in scenes])
        g=dict(delta=delta,ci95=np.quantile(x[draws].mean(1),[.025,.975]).tolist(),
            by_stream_delta=raw.mean(0).tolist(),gained_state_streams=int((raw>0).sum()),lost_state_streams=int((raw<0).sum()),
            improved_physical_states=int((x>0).sum()),worsened_physical_states=int((x<0).sum()),tied_physical_states=int((x==0).sum()),
            mean_post_cost_delta=float(cost.mean()),post_cost_ci95=np.quantile(cost[draws].mean(1),[.025,.975]).tolist())
        if name in ('M-F','M-S'):g.update(simultaneous_ci95=[delta-critical,delta+critical],simultaneous_lower95=delta-lower_critical)
        for field in ('own_mcts_calls','step_calls','policy_evaluations'):
            values=np.array([np.mean([(index[(a,s,t)][field] if field=='own_mcts_calls' else index[(a,s,t)]['own_cost'][field])-
                (index[(b,s,t)][field] if field=='own_mcts_calls' else index[(b,s,t)]['own_cost'][field]) for t in (0,1)]) for s in scenes])
            g[field+'_delta_per_scene']=float(values.mean());g[field+'_ci95']=np.quantile(values[draws].mean(1),[.025,.975]).tolist()
        comparisons[name]=g
    summary={}
    for u in UIDS:
        group=[index[(u,s,t)] for s in scenes for t in (0,1)]
        summary[u]=dict(suffixes=198,successes=int(vectors[u].sum()),successes_by_stream=vectors[u].sum(0).astype(int).tolist(),
            conditional_success_rate=float(vectors[u].mean()),full100_reference_rate=float(vectors[u].sum()/200),
            mean_post_cost=float(np.mean([r['completion_cost']-r['first_capture_time'] for r in group])),
            successful_second_capture_mean=float(np.mean([r['capture_time']-r['first_capture_time'] for r in group if r['legal_double_capture']])),
            own_illegal=sum(r['pursuer_illegal'] for r in group),opponent_illegal=sum(r['evader_illegal'] for r in group),
            decisions=sum(r['own_mcts_calls'] for r in group),
            own_cost={k:sum(r['own_cost'][k] for r in group) for k in ('step_calls','policy_evaluations','value_evaluations')})
    paired=[]
    for scene in scenes:
        streams=[]
        for t in (0,1):
            streams.append(dict(stream=t,**{u:dict(success=index[(u,scene,t)]['legal_double_capture'],
                post_cost=index[(u,scene,t)]['completion_cost']-index[(u,scene,t)]['first_capture_time'],key=index[(u,scene,t)]['key']) for u in UIDS}))
        paired.append(dict(scene_id=scene,streams=streams,mean_success={u:float(np.mean([r[u]['success'] for r in streams])) for u in UIDS}))
    return dict(summary=summary,comparisons=comparisons,physical_states=99,streams_per_state=2,
        bootstrap_seed=57020261024,draws=20000,ordinary_ci='paired_physical_state_bootstrap',
        simultaneous_comparisons=['M-F','M-S'],exploratory_historical_states=True),paired

def report(args,items,rows):
    result,pairs=analyze(rows);diag,records=diagnostics(args,items)
    lock_json(args.output/'statistics.json',result);lock_json(args.output/'paired_scene_results.json',pairs)
    lock_json(args.output/'diagnostics.json',diag);lock_json(args.output/'trajectory_index.json',records)
    mf,ms=result['comparisons']['M-F'],result['comparisons']['M-S']
    amplitude=mf['ci95'][1]<0;direction=ms['ci95'][0]>0
    interpretation=dict(amplification_benefit_clear=amplitude,M_better_than_S_clear=direction,
        recommendation='retain_full_amplitude_F' if amplitude else 'no_clear_amplitude_advantage; simplify_only_after_independent_validation',
        deployment='old0:novalue',extra_experiments_started=False)
    lock_json(args.output/'interpretation.json',interpretation)
    lines=['M/F匹配幅度对照','',
        f'实际来源：{SOURCE}。兼容性检查通过；仅新增198条M后缀，F/S各复用198条。',
        'M与F仅附加根候选基础均值是否放大不同；候选比例、方向、扰动、合法性、模型、搜索预算及随机流恢复相同。','',
        '|组|流0成功/99|流1成功/99|接管后成功率|全100参考口径|自身非法/198|后段成本 步|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for u,g in result['summary'].items():
        a,b=g['successes_by_stream'];lines.append(f'|{u}|{a}|{b}|{100*g["conditional_success_rate"]:.2f}%|{100*g["full100_reference_rate"]:.2f}%|{g["own_illegal"]}|{g["mean_post_cost"]:.2f}|')
    lines+=['','统计单位为99个物理状态，先在状态内平均两流；20000次配对bootstrap，种子57020261024。','',
        '|比较|成功率差 pp|普通95%区间 pp|同时95%区间 pp|挽回/丢失 状态×流|改善/退化 物理状态|后段成本差 步 [95%区间]|',
        '|---|---:|---|---|---:|---:|---|']
    for name,g in result['comparisons'].items():
        lo,hi=np.array(g['ci95'])*100;sim=g.get('simultaneous_ci95');st=f'[{sim[0]*100:+.2f},{sim[1]*100:+.2f}]' if sim else '辅助比较'
        cl,ch=g['post_cost_ci95']
        lines.append(f'|{name}|{g["delta"]*100:+.2f}|[{lo:+.2f},{hi:+.2f}]|{st}|{g["gained_state_streams"]}/{g["lost_state_streams"]}|{g["improved_physical_states"]}/{g["worsened_physical_states"]}|{g["mean_post_cost_delta"]:+.2f} [{cl:+.2f},{ch:+.2f}]|')
    lines+=['','计算开销：全部为首捕后追捕方调用；对局长度不同会影响总量。','',
        '|组|决策数|环境转移|网络调用|几何查询|平均转移/决策|首动作归一化幅度|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for u,g in result['summary'].items():
        c=g['own_cost'];d=diag[u]
        lines.append(f'|{u}|{g["decisions"]}|{c["step_calls"]}|{c["policy_evaluations"]}|{d["geometry_queries"]:.0f}|{c["step_calls"]/g["decisions"]:.1f}|{d["first_magnitudes_mean"]:.3f}|')
    lines+=['','并发耗时保存在逐后缀记录中；F/S来自上一轮，资源争用不同，不以跨轮并发延迟判断部署速度。未追加单进程计时对局。','',
        f'动作幅度放大有明确成功率收益：{amplitude}。不放大均值的M明显优于S：{direction}。',
        'M-F为匹配幅度干预；M-S仍同时改变神经均值候选的比例及其扰动分布，支持候选包作用，不等同于纯方向的独立因果效应。',
        '若M-S区间跨零，不解释为等效或彻底无效；本轮历史状态结果不替代新场景确认。',
        '原零捕场景只在全100参考口径计入失败，未虚构额外后缀。当前部署推荐保持old0:novalue。',
        '所有新增M闭环后缀共198条；工程检查仅根搜索，不额外生成M闭环smoke或重放。未启动其他实验。']
    plots(args,result)
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    lock_json(args.output/'results.json',dict(reason='complete',new_M_suffixes=198,historical_F=198,historical_S=198,
        physical_states=99,training_episodes=0,recommended='old0:novalue',**interpretation))
    return result

def plots(args,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=args.output/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    axes[0].bar(UIDS,[result['summary'][u]['conditional_success_rate']*100 for u in UIDS]);axes[0].set(ylim=(0,100),ylabel='Second capture success (%)',title='99 paired states, two search streams')
    axes[1].bar(UIDS,[result['summary'][u]['mean_post_cost'] for u in UIDS]);axes[1].set(ylabel='Remaining-step cost',title='Failures use remaining time')
    for y,name in enumerate(('M-F','M-S')):
        g=result['comparisons'][name];v=g['delta']*100;lo,hi=np.array(g['ci95'])*100
        axes[2].errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4)
    axes[2].axvline(0,color='gray',ls='--');axes[2].set(yticks=[0,1],yticklabels=['M-F','M-S'],xlabel='Paired difference (pp)',title='Ordinary 95% confidence intervals')
    fig.tight_layout();fig.savefig(out/'MF_matched.png',dpi=180);plt.close(fig)

def integrity(args,items):
    count=0;checks=0;M=0
    for item in items:
        row,d=read_episode(args,item)
        if item['uid']=='M':M+=1
        for root in d['roots']:
            if root['visits']!=1000 or root['table'][:,1].sum()!=1000 or root['cost']['value_evaluations']!=0:
                raise ValueError('Saved search budget or value configuration changed')
            if root['guidance'].get('guided_steps',0)!=0:raise ValueError('Unexpected guided rollout')
            count+=1
        if item['task']['historical']:
            source=item['task']['historical'];original=o.read_json(source['metric']);saved=o.load_torch(source['record'])
            if old.previous.science(row)!=old.previous.science(original) or row['trajectory_sha256']!=original['trajectory_sha256'] or \
                not old.s.equal_tree(d['final'],saved['final']):raise ValueError('Historical result was altered')
            checks+=1
    if M!=198 or checks!=396:raise ValueError('Final execution count differs')
    return lock_json(args.output/'integrity.json',dict(passed=True,new_M_suffixes=M,historical_exact=checks,
        verified_records=len(items),root_searches=count,every_budget_1000=True,models_unchanged=True,
        additional_M_closed_loop_checks=0,recommended='old0:novalue'))

def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',nargs='?',default='pipeline',choices=['pipeline','prepare','check','pilot','report','watch'])
    parser.add_argument('--workers',type=int,default=80);parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--once',action='store_true');args=parser.parse_args(argv);args.output=args.output.resolve()
    if args.workers<1:parser.error('workers must be positive')
    return args

def main(argv=None):
    args=parse_args(argv)
    if args.stage=='watch':return old.previous.watch(args)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(pid=os.getpid(),phase='compatibility',workers=args.workers,started=time.time());mutex=threading.Lock();stop=threading.Event()
        def status(phase,**fields):
            with mutex:
                state.update(phase=phase,updated_unix=time.time(),**fields)
                state.update(actual_workers=len([p for p in mp.active_children() if p.is_alive()]),
                    committed_cache=len(list((args.output/'cache_checksums').glob('*.lock.json'))))
                o.write_json(args.output/'status.json',state)
        def heartbeat():
            while not stop.wait(5):status(state['phase'])
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:
            cases=prepare(args);status('compatible',physical_states=len(cases))
            if args.stage=='prepare':return
            status('check');checks(args,cases,status)
            if args.stage=='check':return
            items=requests(args,cases)
            rows=run(args,items,status) if args.stage in ('pipeline','pilot') else [read_cached(args,i) for i in items]
            status('report');integrity(args,items);report(args,items,rows)
            status('complete',new_M_suffixes=198,historical_reused=396,recommended='old0:novalue',elapsed_seconds=time.time()-state['started'])
        except BaseException as error:
            status('stopped',reason=f'{type(error).__name__}: {error}');raise
        finally:stop.set();thread.join(timeout=1)

if __name__=='__main__':main()
