"""Map-paired E49 statistics, diagnostics and standalone scientific plots."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import numpy as np

ARMS = ('IG','AG','RouteAPF','MAPPO_RouteAPF','GEOM_SEARCH','SINGLE_PRIOR',
        'A1','A2','A3','A4','A5','PPO_C')
CONTRASTS = (('A3','A2'),('A3','A5'),('A2','A4'),('A3','A1'),
             ('A2','GEOM_SEARCH'),('SINGLE_PRIOR','GEOM_SEARCH'),
             ('MAPPO_RouteAPF','RouteAPF'),('PPO_C','A4'))


def map_matrix(rows, metric='double_capture'):
    cells, metadata = defaultdict(list), {}
    for row in rows:
        t = row['task']
        cells[t['group'], t['arm']].append(float(row['summary'][metric]))
        metadata[t['group']] = (str(t['obstacle_tier']), t['suite'], t['family'])
    names = sorted(metadata)
    values = np.array([[np.mean(cells[name, arm]) if cells[name,arm] else np.nan
                        for arm in ARMS] for name in names])
    return names, values, metadata


def weights_for(names, metadata):
    groups = defaultdict(list)
    for i, name in enumerate(names):
        tier, suite, _ = metadata[name]
        groups[tier, suite].append(i)
    weights = np.zeros(len(names))
    for ids in groups.values():
        weights[ids] = 1/(len(groups)*len(ids))
    return weights


def statistics(rows, repetitions=10000):
    names, values, metadata = map_matrix(rows)
    if not np.isfinite(values).all():
        raise RuntimeError('Missing methods in paired E49 report')
    weights = weights_for(names, metadata)
    means = weights @ values
    strata = defaultdict(list)
    for i,name in enumerate(names):
        tier, suite, family = metadata[name]
        strata[tier,suite,family].append(i)
    rng = np.random.default_rng(14920261108)
    bootstrap = np.zeros((repetitions, len(ARMS)))
    for ids in strata.values():
        ix = rng.choice(ids, (repetitions,len(ids)), replace=True)
        bootstrap += values[ix].sum(axis=1)*weights[ids[0]]
    per_method = {}
    costs = map_matrix(rows, 'completion_cost')[1]
    utils = map_matrix(rows, 'utility')[1]
    first = map_matrix(rows, 'at_least_one')[1]
    for j,arm in enumerate(ARMS):
        own = [r for r in rows if r['task']['arm']==arm]
        latencies = [v for r in own for v in r['summary']['latencies_ms']]
        p = float(means[j]);f=float(weights@first[:,j])
        per_method[arm] = dict(double_capture=p, at_least_one=f,
            post_first_completion=p/f if f else None,
            ci95=np.quantile(bootstrap[:,j],[.025,.975]).tolist(),
            completion_cost=float(weights@costs[:,j]), utility=float(weights@utils[:,j]),
            zero_capture=1-f, first_only=f-p,
            timing_ms={key:float(np.percentile(latencies,q)) for key,q in [('p50',50),('p95',95),('p99',99),('maximum',100)]},
            online_overrun_fraction=sum(r['summary']['online_overruns'] for r in own)/len(latencies),
            episodes=len(own), maps=len(names))
    contrasts = []
    alpha = .05/len(CONTRASTS)
    for a,b in CONTRASTS:
        ia,ib=ARMS.index(a),ARMS.index(b)
        contrast=bootstrap[:,ia]-bootstrap[:,ib]
        contrasts.append(dict(a=a,b=b,difference=float(means[ia]-means[ib]),
            simultaneous_ci95=np.quantile(contrast,[alpha/2,1-alpha/2]).tolist(),
            interval='Map-stratified paired percentile bootstrap; Bonferroni eight predeclared contrasts'))
    tier_results={}
    suite_results={}
    for tier in sorted({m[0] for m in metadata.values()}):
        subset=[r for r in rows if str(r['task']['obstacle_tier'])==tier]
        ns,v,md=map_matrix(subset)
        tier_results[tier]=dict(zip(ARMS,map(float,weights_for(ns,md)@v)))
    for suite in sorted({m[1] for m in metadata.values()}):
        subset=[r for r in rows if r['task']['suite']==suite]
        ns,v,md=map_matrix(subset)
        suite_results[suite]=dict(zip(ARMS,map(float,weights_for(ns,md)@v)))
    return dict(methods=per_method,contrasts=contrasts,tiers=tier_results,suites=suite_results,
                maps=len(names),episodes=len(rows),bootstrap_repetitions=repetitions,
                weighting='models/worlds/streams within map; ID/OOD each50% within tier; tiers equal')


def progress_curve(rows, horizon):
    cells=defaultdict(list);metadata={}
    for row in rows:
        t=row['task'];s=row['summary'];time=np.arange(horizon+1)
        f=s['first_capture_time'];d=s['final_capture_time']
        cells[t['group'],t['arm']].append(np.stack((np.zeros_like(time) if f is None else time>=f,
                                                  np.zeros_like(time) if d is None else time>=d)).astype(float))
        metadata[t['group']]=(str(t['obstacle_tier']),t['suite'],t['family'])
    names=sorted(metadata);weights=weights_for(names,metadata)
    return {arm:np.einsum('i,ijk->jk',weights,
                         np.stack([np.mean(cells[name,arm],axis=0) for name in names])) for arm in ARMS}


def _distribution(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not values:return dict(count=0,mean=None,p50=None,p95=None,p99=None,maximum=None)
    return dict(count=len(values),mean=float(np.mean(values)),
                **{key:float(np.percentile(values,q)) for key,q in
                   [('p50',50),('p95',95),('p99',99),('maximum',100)]})


def paired_table(rows, repetitions=10000):
    """One subgroup, retaining map pairing and the registered weighting rule."""
    names,values,metadata=map_matrix(rows)
    if not np.isfinite(values).all():raise RuntimeError('Incomplete paired subgroup')
    weights=weights_for(names,metadata);means=weights@values
    strata=defaultdict(list)
    for i,name in enumerate(names):strata[metadata[name]].append(i)
    rng=np.random.default_rng(14920261108)
    boot=np.zeros((repetitions,len(ARMS)))
    for ids in strata.values():
        indices=rng.choice(ids,(repetitions,len(ids)),replace=True)
        boot+=values[indices].sum(axis=1)*weights[ids[0]]
    first=map_matrix(rows,'at_least_one')[1]
    costs=map_matrix(rows,'completion_cost')[1]
    utility=map_matrix(rows,'utility')[1]
    methods={}
    for j,arm in enumerate(ARMS):
        own=[r for r in rows if r['task']['arm']==arm]
        methods[arm]=dict(double_capture=float(means[j]),ci95=np.quantile(boot[:,j],[.025,.975]).tolist(),
            at_least_one=float(weights@first[:,j]),completion_cost=float(weights@costs[:,j]),
            utility=float(weights@utility[:,j]),raw_episodes=len(own),
            raw_double_capture=sum(bool(r['summary']['double_capture']) for r in own),
            raw_at_least_one=sum(bool(r['summary']['at_least_one']) for r in own))
    alpha=.05/len(CONTRASTS)
    contrasts=[]
    for a,b in CONTRASTS:
        ia,ib=ARMS.index(a),ARMS.index(b)
        contrasts.append(dict(a=a,b=b,difference=float(means[ia]-means[ib]),
            simultaneous_ci95=np.quantile(boot[:,ia]-boot[:,ib],[alpha/2,1-alpha/2]).tolist()))
    return dict(maps=len(names),methods=methods,contrasts=contrasts,
        uncertainty='Map-stratified paired bootstrap; eight contrasts Bonferroni-corrected within this table; exploratory tables are not jointly corrected across strata')


def paired_breakdowns(rows,repetitions=10000):
    tiers={};families={};tier_families={}
    for tier in sorted({str(r['task']['obstacle_tier']) for r in rows}):
        subset=[r for r in rows if str(r['task']['obstacle_tier'])==tier]
        tiers[tier]=paired_table(subset,repetitions)
        for family in sorted({r['task']['family'] for r in subset}):
            tier_families[f'{tier}/{family}']=paired_table([r for r in subset if r['task']['family']==family],repetitions)
    for family in sorted({r['task']['family'] for r in rows}):
        families[family]=paired_table([r for r in rows if r['task']['family']==family],repetitions)
    return dict(tiers=tiers,families=families,tier_families=tier_families)


def method_diagnostics(rows,load_traces=False):
    """Raw event/exposure counts alongside map-weighted diagnostic rates.

    Raw counts are descriptive and never replace the registered map-weighted
    estimator. Hidden runs ending in capture/end remain censored, rather than
    being counted as failed rediscoveries or discarded from their duration.
    """
    results={}
    for arm in ARMS:
        own=[r for r in rows if r['task']['arm']==arm]
        if not own:continue
        maps=defaultdict(list);metadata={};rediscovered=[];censored=[];first_times=[];remaining=[]
        final_times=[];post_times=[];events=[];categories=Counter();simulations=[];stats_covered=0
        decisions=zero=stops=overruns=late=first_only_late=0
        hidden=visible=post_visible=post_active=0;direct_decisions=search_decisions=0
        exact_visibility=Counter();latencies=[];cpu=[];construction=[];history=[]
        for row in own:
            task=row['task'];s=row['summary'];h=task['horizon']
            metadata[task['group']]=(str(task['obstacle_tier']),task['suite'],task['family'])
            maps[task['group']].append(s)
            n=int(s.get('decisions',len(s.get('latencies_ms',[]))));decisions+=n
            zero+=int(s.get('zero_simulation_decisions',0));stops+=int(s.get('stop_decisions',0));overruns+=int(s.get('online_overruns',0))
            hidden+=int(s.get('hidden_target_decisions',0));visible+=int(s.get('visible_target_decisions',0))
            post_visible+=int(s.get('post_capture_visible_decisions',0));post_active+=int(s.get('post_capture_active_decisions',0))
            categories[s.get('failure_category','unreported')]+=1
            rediscovered.extend(s.get('rediscoveries',[]));censored.extend(s.get('censored_occlusions',[]));events.extend(s.get('capture_events',[]))
            first=s.get('first_capture_time')
            if first is not None:
                first_times.append(first);remaining.append(h-first)
                if first>=.75*h:
                    late+=1;first_only_late+=int(not s['double_capture'])
            if s.get('final_capture_time') is not None:final_times.append(s['final_capture_time'])
            if s.get('post_capture_seconds') is not None:post_times.append(s['post_capture_seconds'])
            latencies.extend(s.get('latencies_ms',[]));cpu.extend(s.get('cpu_ms',[]))
            construction.append(s.get('construction_ms'));history.extend(s.get('history_ms',[]))
            trace=row.get('trace')
            if trace is None and load_traces and row.get('episode'):
                from run_e49_benchmark import load_episode
                trace=load_episode(row['episode'])['trace']
            for frame in trace or []:
                stats=frame.get('stats',{});obs=frame.get('observation',{})
                if 'simulations' in stats:
                    simulations.append(stats['simulations']);stats_covered+=1
                    if arm in ('GEOM_SEARCH','SINGLE_PRIOR','A1','A2','A3'):search_decisions+=1
                    else:direct_decisions+=1
                for j,active in enumerate(obs.get('active_e',[])):
                    if active:exact_visibility['visible' if any(q is not None for q in obs['reports'][j]) else 'hidden']+=1
        names=sorted(maps);weights=weights_for(names,metadata)
        def weighted_mean(key):
            return float(sum(w*np.mean([float(s.get(key,0)) for s in maps[name]]) for w,name in zip(weights,names)))
        def weighted_ratio(numerator,denominator):
            den=weighted_mean(denominator)
            return weighted_mean(numerator)/den if den else None
        weighted_hidden=weighted_mean('hidden_target_decisions');weighted_visible=weighted_mean('visible_target_decisions')
        nfirst=sum(bool(r['summary']['at_least_one']) for r in own)
        ndouble=sum(bool(r['summary']['double_capture']) for r in own)
        results[arm]=dict(raw=dict(episodes=len(own),maps=len(names),decisions=decisions,
            double_capture=ndouble,first_capture=nfirst,first_only=nfirst-ndouble,no_capture=len(own)-nfirst,
            late_first_capture=late,late_first_only=first_only_late,late_definition='first_capture_time >= .75H',
            visibility_target_decisions=dict(hidden=hidden,visible=visible,post_first_visible=post_visible,post_first_active=post_active),
            rediscovered_runs=len(rediscovered),censored_runs=len(censored),censor_reasons=dict(Counter(r.get('reason','unreported') for r in censored)),
            capture_events=len(events),captures_by_pursuer=dict(Counter(str(e['pursuer']) for e in events)),
            failure_categories=dict(categories),zero_simulation_decisions=zero,stop_decisions=stops,overrun_decisions=overruns),
            weighted=dict(post_first_completion=weighted_ratio('double_capture','at_least_one'),
                visible_fraction=weighted_visible/(weighted_visible+weighted_hidden) if weighted_visible+weighted_hidden else None,
                post_first_visible_fraction=weighted_ratio('post_capture_visible_decisions','post_capture_active_decisions')),
            timing=dict(action_ms=_distribution(latencies),cpu_ms=_distribution(cpu),construction_ms=_distribution(construction),history_update_ms=_distribution(history)),
            computation=dict(completed_simulations=_distribution(simulations),total_completed_simulations=int(sum(simulations)),
                trace_decisions_with_simulation_count=stats_covered,search_decisions=search_decisions,direct_decisions=direct_decisions,
                simulation_evidence_complete=stats_covered==decisions,
                zero_simulation_fraction=zero/decisions if decisions else None,stop_fraction=stops/decisions if decisions else None,
                overrun_fraction=overruns/decisions if decisions else None,
                simulated_transitions=sum(int(r['summary'].get('simulated_transitions',0)) for r in own)),
            exposure=dict(first_capture_time=_distribution(first_times),first_capture_remaining_seconds=_distribution(remaining),
                total_post_first_opportunity_seconds=float(sum(remaining)),final_capture_time=_distribution(final_times),
                successful_post_first_seconds=_distribution(post_times),rediscovered_run_steps=_distribution([r['steps'] for r in rediscovered]),
                censored_run_steps=_distribution([r['steps'] for r in censored]),trace_visibility_counts=dict(exact_visibility)),
            interpretation='Post-first completion is conditional on achieving first capture; late first captures have less remaining opportunity. Rediscovery durations exclude separately reported censored runs; these are not unconditional recovery-time estimates. Direct controls intentionally report zero simulations.')
    return results


def plot_trajectories(out, rows):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from run_e49_benchmark import load_episode,read
    figures=Path(out)/'figures';figures.mkdir(exist_ok=True)
    # Common, outcome-independent tasks: first ID and first OOD map, AB stream0.
    selected=[]
    for suite in ('ID','OOD'):
        candidates=[r for r in rows if r['task']['suite']==suite and r['task']['world']=='AB' and r['task']['stream']==0]
        if not candidates:continue
        group=min(r['task']['group'] for r in candidates)
        selected.append([min((r for r in candidates if r['task']['group']==group and r['task']['arm']==arm),key=lambda r:r['task']['train_seed']) for arm in ARMS])
    for suite_rows in selected:
        case=read(suite_rows[0]['task']['case']);fig,axes=plt.subplots(3,4,figsize=(16,12))
        for ax,row in zip(axes.ravel(),suite_rows):
            data=load_episode(row['episode']);trace=data['trace']
            states=np.array([trace[0]['state']]+[f['next_state'] for f in trace])
            for b in case['map']:
                ax.add_patch(Rectangle((b[0][0],b[1][0]),b[0][1]-b[0][0],b[1][1]-b[1][0],color='.65'))
            for i in range(4):
                ax.plot(states[:,2*i],states[:,2*i+1],color=('tab:red' if i<2 else 'tab:blue'),linestyle=('-' if i%2==0 else '--'),alpha=.8)
                ax.scatter(states[0,2*i],states[0,2*i+1],color=('tab:red' if i<2 else 'tab:blue'),s=15)
            ax.set(xlim=(-10,10),ylim=(-10,10),aspect='equal',title=f"{row['task']['arm']}: {row['summary']['captures']}/2, {row['summary']['decisions']} steps")
        fig.suptitle('Common task: '+suite_rows[0]['task']['group']);fig.tight_layout()
        fig.savefig(figures/f"paired_{suite_rows[0]['task']['suite']}.pdf");plt.close(fig)


def report(out, rows):
    from run_e49_benchmark import write
    out=Path(out)
    if len(rows)!=7392 or not all(r['valid'] for r in rows):
        raise RuntimeError('Formal report requires all7392 valid episodes')
    horizons={r['task']['horizon'] for r in rows}
    if len(horizons)!=1:raise RuntimeError('Mixed formal horizons')
    horizon=horizons.pop();stats=statistics(rows)
    write(out/'final/results.json',stats)
    diagnostic=method_diagnostics(rows,load_traces=True)
    breakdowns=paired_breakdowns(rows)
    for arm,data in diagnostic.items():
        comparisons=[]
        for contrast in stats['contrasts']:
            if arm not in (contrast['a'],contrast['b']):continue
            sign=1 if arm==contrast['a'] else -1
            interval=sorted(sign*np.asarray(contrast['simultaneous_ci95']))
            comparisons.append(dict(reference=contrast['b'] if sign==1 else contrast['a'],
                double_capture_difference=sign*contrast['difference'],simultaneous_ci95=list(map(float,interval)),
                evidence='advantage_supported' if interval[0]>0 else 'limitation_supported' if interval[1]<0 else 'inconclusive'))
        data['predeclared_paired_evidence']=comparisons
    write(out/'final/diagnostic.json',diagnostic)
    write(out/'final/paired_breakdowns.json',breakdowns)
    for arm,data in diagnostic.items():write(out/'final/diagnostics'/arm/'diagnostic.json',data)
    with (out/'final/tier_family_paired.csv').open('w') as handle:
        writer=csv.writer(handle)
        writer.writerow(['tier_family','method','maps','episodes','raw_double','raw_first','double_capture','ci_low','ci_high','at_least_one','completion_cost','utility'])
        for cell,table in breakdowns['tier_families'].items():
            for arm,s in table['methods'].items():
                writer.writerow([cell,arm,table['maps'],s['raw_episodes'],s['raw_double_capture'],s['raw_at_least_one'],s['double_capture'],*s['ci95'],s['at_least_one'],s['completion_cost'],s['utility']])
    with (out/'final/methods.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['method','double_capture','ci_low','ci_high','at_least_one','completion_cost','utility','latency_p95_ms'])
        for arm in ARMS:
            s=stats['methods'][arm];w.writerow([arm,s['double_capture'],*s['ci95'],s['at_least_one'],s['completion_cost'],s['utility'],s['timing_ms']['p95']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=out/'figures';figures.mkdir(exist_ok=True)
    curves=progress_curve(rows,horizon)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for arm,data in curves.items():
        for ax,y in zip(axes,data):ax.plot(np.arange(horizon+1),y,label=arm)
    for ax,title in zip(axes,['At least one capture','Double capture']):
        ax.set(xlabel='Seconds',ylabel='Fraction of all games',ylim=(0,1.02),title=title);ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(figures/'capture_progress.pdf');fig.savefig(figures/'capture_progress.png',dpi=180);plt.close(fig)
    plot_trajectories(out,rows)
    lines=['# E49：持续追击双捕获 Benchmark 结果','',f'正式测试7,392局全部有效，84张地图，统一时域H={horizon}。','',
           '总体对三档障碍等权，各档内ID/OOD各50%；模型种子、隐藏世界与随机流先在地图内平均。', '',
           '| 方法 | 双捕率（95%区间） | 至少一捕 | 完成成本 | P95动作耗时ms |',
           '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        s=stats['methods'][arm];lo,hi=s['ci95'];lines.append(f"| {arm} | {s['double_capture']:.1%} [{lo:.1%},{hi:.1%}] | {s['at_least_one']:.1%} | {s['completion_cost']:.2f} | {s['timing_ms']['p95']:.2f} |")
    lines+=['','## 预设配对比较','', '| 比较 | 双捕率差 | 同时95%区间 |','|---|---:|---:|']
    for c in stats['contrasts']:
        lo,hi=c['simultaneous_ci95'];lines.append(f"| {c['a']}−{c['b']} | {c['difference']:+.1%} | [{lo:+.1%},{hi:+.1%}] |")
    lines+=['','区间采用地图分层配对bootstrap10,000次，并对8个预设比较作Bonferroni校正。','', '## 逐方法表现与失败分解','']
    for arm in ARMS:
        s=stats['methods'][arm];post=s['post_first_completion']
        d=diagnostic[arm];raw=d['raw'];exposure=d['exposure'];computing=d['computation'];visibility=raw['visibility_target_decisions']
        remaining=exposure['first_capture_remaining_seconds']['p50'];remaining_text='无首捕样本' if remaining is None else f'{remaining:.2f}秒'
        sims=computing['completed_simulations'];sim_text='缺少逐步记录' if sims['p50'] is None else f"P50/P95={sims['p50']:.0f}/{sims['p95']:.0f}"
        paired=[]
        for comparison in d['predeclared_paired_evidence']:
            lo,hi=comparison['simultaneous_ci95'];label=dict(advantage_supported='区间支持本方法较高',limitation_supported='区间支持本方法较低',inconclusive='区间跨零，方向未确定')[comparison['evidence']]
            paired.append(f"相对{comparison['reference']}双捕率差{comparison['double_capture_difference']:+.1%}，同时区间[{lo:+.1%},{hi:+.1%}]，{label}")
        paired_text='；'.join(paired)+'。' if paired else '本方法没有预设成对假设，不依据事后挑选比较宣称稳定优势。'
        lines+=[f"### {arm}",'',f"本批零捕比例{s['zero_capture']:.1%}，仅首捕未双捕比例{s['first_only']:.1%}；"+(f"已首捕后的完成比例{post:.1%}。" if post is not None else '本批未首捕，无法估计首捕后完成率。'),
                paired_text,
                f"原始计数：{raw['episodes']}局中双捕{raw['double_capture']}、仅首捕{raw['first_only']}、零捕{raw['no_capture']}；这些计数不替代上表的地图加权估计。首捕时剩余时间中位数{remaining_text}；末四分之一时域才首捕{raw['late_first_capture']}局，其中{raw['late_first_only']}局未完成第二捕。首捕后完成率存在首捕选择与剩余机会差异，不能单独视为协同能力排名。",
                '三档双捕率：'+ '，'.join(f"{tier}障碍 {v[arm]:.1%}" for tier,v in stats['tiers'].items())+'。',
                'ID/OOD双捕率：'+ '，'.join(f"{suite} {v[arm]:.1%}" for suite,v in stats['suites'].items())+'。',
                f"观测暴露量：可见目标决策{visibility['visible']}、隐藏目标决策{visibility['hidden']}；首捕后{visibility['post_first_active']}次活动目标决策中可见{visibility['post_first_visible']}次。完成再发现的遮挡段{raw['rediscovered_runs']}段，另有{raw['censored_runs']}段被捕获或终局截尾；不能只用已再发现片段推断无条件恢复速度。",
                '终局类型原始计数：'+ '，'.join(f'{name}={count}' for name,count in sorted(raw['failure_categories'].items()))+'。这些类别区分未首捕、首捕后仍无正观测以及已见但未完成追捕，不能单凭类别指定失败因果。',
                f"计算代价P50/P95为{s['timing_ms']['p50']:.2f}/{s['timing_ms']['p95']:.2f}ms；超过200ms比例{s['online_overrun_fraction']:.1%}。共{raw['decisions']}次决策，停车{raw['stop_decisions']}次、零完整模拟{raw['zero_simulation_decisions']}次；完整模拟数{sim_text}，总计{computing['total_completed_simulations']}次。直控的零模拟是方法定义，搜索的零模拟才反映未完成规划。",
                '这些证据描述本批的任务完成、遮挡暴露和计算代价；方法优势或不足的比较仅在预设配对区间支持的范围内解释，不把障碍数量或可见暴露差异当成单独因果效应。','']
    lines+=['## 分档与布局族配对明细','',
            'paired_breakdowns.json提供三档、七族及“档位×布局族”的地图配对估计、原始计数和区间；tier_family_paired.csv便于逐格审阅。每个档位×布局族仅4张独立地图，多个世界、随机流与模型种子不会增加独立地图数。分格比较仅作探索，8项比较的校正只在各表内进行，不涵盖全部分格。','']
    lines+=['## 解释范围','', '时域只由开发集预设饱和规则选择。正式测试不反向训练或改时域。每个障碍档位×布局族只有4张地图，细分排名存在不确定性。累计曲线是冻结时域策略的真实进度，不能冒充改变截止时间后的独立闭环结果。', '', '完整主表见results.json；diagnostic.json及diagnostics/各方法/diagnostic.json记录暴露分母、截尾段、首捕剩余机会、实际模拟数和耗时分布。capture_progress.pdf为捕获进度曲线；paired_ID.pdf与paired_OOD.pdf使用按任务编号固定选取的共同场景。']
    (out/'final/REPORT.md').write_text('\n'.join(lines)+'\n')
    return stats
