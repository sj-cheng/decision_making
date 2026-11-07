"""Review the serial module/cost study without additional searches or episodes."""
from pathlib import Path
import numpy as np
import run_l45_search_contribution as m

def main():
    args=m.parse_args([]);banks=m.prepare(args);tasks=m.prefix_tasks(args,banks)
    items=m.requests(args,tasks,'fixed')+m.requests(args,tasks,'timed')
    result=m.locked_read(args.output/'statistics.json');diag=m.locked_read(args.output/'diagnostics.json')
    rows=[m.read_cached(args,i) for i in items]
    prefixes=[m.read_prefix(args,t) for t in tasks]
    prefix_summary={}
    for p in prefixes:
        key=p['scenario']['layout']+'|'+p['opponent'];g=prefix_summary.setdefault(key,dict(games=0,eligible=0,zero_capture=0,direct_double=0,illegal=0))
        g['games']+=1;g['eligible']+=int(p['snapshot']['branch_available'])
        g['zero_capture']+=int(p['metrics']['capture_count']==0);g['direct_double']+=int(p['metrics']['capture_count']==2)
        g['illegal']+=p['metrics']['illegal_terminal']
    # Validate all scientific input identities, output artifacts and neural/cost accounting.
    forward_checks=0;cap_hits=0;new_suffixes=0;paired=[]
    for item in items:
        row,d=m.read_episode(args,item);task=item['task'];pre=m.o.load_torch(task['prefix_path'])
        snap=pre['snapshot']
        for field in ('state','stats','time','remaining_time','own_rng','opponent_rng','python_rng'):
            if not m.old.s.equal_tree(d['initial'][field],snap[field]):raise ValueError('Initial snapshot differs: '+field)
        if row['branch_available'] and not row['reused_geometry']:new_suffixes+=1
        for r in d['roots']:
            g=r['guidance'];calls=g.get('policy_pursuer',0)+g.get('policy_evader',0)
            if calls!=r['cost']['policy_evaluations']:raise ValueError('Team forward-call counts do not sum')
            if item['uid']=='G-no-neural' and g.get('policy_pursuer',0):raise ValueError('Neural ablation violated')
            if item['uid']=='Geometry-only' and (r['visits'] or calls or r['cost']['step_calls']):raise ValueError('Geometry baseline ran a tree/network')
            cap_hits+=int(g.get('simulation_cap_hit',0));forward_checks+=1
        paired.append(dict(uid=row['uid'],phase=row['phase'],scene=row['episode_hash'],opponent=row['opponent'],
            success=row['legal_double_capture'],completion_cost=row['completion_cost'],branch=row['branch_available'],
            first_capture_time=row['first_capture_time'],suffix_compute_ms=sum(z[0] for z in row['decision_latency_ms']),
            record=row['episode_record']['path']))
    if cap_hits:raise ValueError('Time-budget cap reached: fairness review required before effect conclusion')
    independent=m.locked_read(args.output/'scenario_independence.json');used=set()
    for name,sha in independent['history_files'].items():
        if m.o.sha(name)!=sha:raise ValueError('Historical scene file changed: '+name)
        for spec in m.prior.extract_specs(m.o.read_json(name)):used.add(m.prior.physical_hash(spec))
    fresh=[s for bank in banks.values() for s in bank]
    if set(m.prior.physical_hash(s) for s in fresh)&used:raise ValueError('Scene leakage')
    m.o.write_json(args.output/'paired_scene_results.json',paired)
    m.o.write_json(args.output/'prefix_summary.json',prefix_summary)
    conditional={}
    for phase in ('fixed','timed'):
        conditional[phase]={}
        for uid in m.UIDS:
            selected=[r for r in rows if r['phase']==phase and r['uid']==uid and r['branch_available']]
            conditional[phase][uid]=dict(
                states=len(selected),successes=sum(r['legal_double_capture'] for r in selected),
                mean_remaining_steps=float(np.mean([80-r['shared_prefix_steps'] for r in selected])),
                mean_completion_cost=float(np.mean([r['completion_cost']-r['shared_prefix_steps'] for r in selected])),
                mean_pursuit_ms=float(np.mean([sum(z[0] for z in r['decision_latency_ms']) for r in selected])))
    m.o.write_json(args.output/'conditional_suffix_results.json',conditional)
    m.o.write_json(args.output/'final_review.json',dict(passed=True,new_physical_scenes=len(fresh),prefix_tasks=len(tasks),
        new_closed_loop_suffixes=new_suffixes,all_saved_artifacts_checked=True,team_call_count_checks=forward_checks,
        timed_cap_hits=cap_hits,shared_input_snapshots_and_rng_exact=True,history_overlap=0,tests=5,
        isolated_smoke=str(m.SMOKE_OUTPUT),current_deployment='old0:novalue',no_new_training=True,
        no_additional_scientific_experiments=True))
    all_data=result['sections']['all'];root=diag['equal_input_roots'];rootdiff=diag['equal_input_G_minus_F']
    decisions={}
    for phase in ('fixed','timed'):
        c=all_data[phase]['mean']['comparisons']
        ng=c['G-no-neural minus G'];geo=c['Geometry-only minus G'];geon=c['Geometry-only minus G-no-neural']
        decisions[phase]=dict(neural_pursuer_gain_supported=ng['ci95'][1]<0,
            no_neural_delta=ng['delta'],no_neural_ci95=ng['ci95'],
            geometry_noninferior_to_G=geo['lower95']>-.03,geometry_G_ci95=geo['ci95'],
            search_better_than_geometry_supported=geo['ci95'][1]<0,
            no_neural_search_better_than_geometry_supported=geon['ci95'][1]<0)
    fixedgf=all_data['fixed']['mean']['comparisons']['G minus F']
    decisions['timing']=dict(equal_input_G_faster=rootdiff['ci95'][1]<0 if rootdiff else None,
        equal_input_G_F=rootdiff,fixed_suffix_G_less_time=fixedgf['suffix_ms_ci95'][1]<0,
        fixed_suffix_G_F_ms=fixedgf['suffix_ms_delta'],fixed_suffix_G_F_ms_ci95=fixedgf['suffix_ms_ci95'])
    m.o.write_json(args.output/'interpretation.json',decisions)
    conclusions=[]
    if rootdiff:
        lo,hi=rootdiff['ci95'];direction='更快' if hi<0 else '更慢' if lo>0 else '没有明确快慢差异'
        conclusions.append(f'G是否比F省时：相同状态、1000模拟的根调用显示G{direction}；G-F为{rootdiff["delta_ms"]:+.3f}ms，95%区间[{lo:+.3f},{hi:+.3f}]。')
        fraction=-rootdiff['delta_ms']/root['F']['mean_ms']*100
        conclusions.append(f'单次根调用的均值省时仅约{fraction:.2f}%；热根P95为G {root["G"]["p95_ms"]:.2f}ms、F {root["F"]["p95_ms"]:.2f}ms，不能概括为G显著改善尾延迟。')
    for phase,label in (('fixed','1000模拟'),('timed','50ms预算')):
        cc=all_data[phase]['mean']['comparisons'];ng=cc['G-no-neural minus G'];gg=cc['Geometry-only minus G'];gn=cc['Geometry-only minus G-no-neural']
        lo,hi=np.array(ng['ci95'])*100
        evidence='支持追捕神经候选有额外成功率贡献' if hi<0 else '未检出删去追捕神经候选造成的成功率下降' if lo<=0<=hi else '删去追捕神经候选反而改善成功率'
        conclusions.append(f'{label}神经消融：G-no-neural-G={ng["delta"]*100:+.2f}pp，95%区间[{lo:+.2f},{hi:+.2f}]；{evidence}。')
        lo,hi=np.array(gg['ci95'])*100
        search='搜索G相对直接几何有明确成功率收益' if hi<0 else '直接几何相对搜索G有明确成功率收益' if lo>0 else '当前样本未检出搜索G相对直接几何的成功率收益'
        conclusions.append(f'{label}MCTS对照：Geometry-only-G={gg["delta"]*100:+.2f}pp，95%区间[{lo:+.2f},{hi:+.2f}]；{search}。')
        if gn['ci95'][1]<0:
            conclusions.append(f'{label}下，保留G但移除追捕网络的搜索也显著优于直接几何，表明这部分收益不能只用追捕神经候选解释。')
        if ng['ci95']==[0.,0.] or gg['ci95']==[0.,0.]:
            conclusions.append(f'{label}存在全平分导致的退化bootstrap区间；零宽区间不能作为精确总体等效的证据。')
        summary=all_data[phase]['mean']['summary'];fraction=1-summary['G']['mean_suffix_pursuit_ms']/summary['F']['mean_suffix_pursuit_ms']
        conclusions.append(f'{label}G相对F的整段追捕计算减少{100*fraction:.1f}%，同时完成成本下降{-cc["G minus F"]["completion_cost_delta"]:.2f}步；主要优势是更早收尾，而非大幅降低每次搜索开销。')
    fixed_summary=all_data['fixed']['mean']['summary'];ratio=fixed_summary['G']['mean_suffix_pursuit_ms']/fixed_summary['Geometry-only']['mean_suffix_pursuit_ms']
    conclusions.append(f'直接几何的固定阶段每次后段追捕计算为{fixed_summary["Geometry-only"]["mean_suffix_pursuit_ms"]:.2f}ms，G为{fixed_summary["G"]["mean_suffix_pursuit_ms"]:.2f}ms，约相差{ratio:.1f}倍。这不包含共同首捕前缀及真实对手计算，不是整个系统的倍数加速。')
    if all(decisions[p]['search_better_than_geometry_supported'] for p in ('fixed','timed')):
        conclusions.append('下一步优先保留搜索决策流程，并据神经消融的幅度选择G或G-no-neural作为后续候选；直接几何保留低成本基线。')
    elif all(not decisions[p]['search_better_than_geometry_supported'] for p in ('fixed','timed')):
        conclusions.append('下一步优先检验“首捕前神经搜索＋首捕后直接几何”的简化候选，保留G/F为对照，在更广独立场景与更困难布局、能力条件下定位几何失效情况。当前一追一逃后段没有显示保留1000模拟的收益，不再优先优化这部分搜索开销。首捕前贡献尚未消融，继续保留。这里只提出下一阶段，不自动执行。')
    else:
        conclusions.append('两种预算下搜索的证据不同，下一步应按实际延迟约束选择保留的决策流程，先解释预算敏感性，再考虑部署简化。')
    conclusions.append('本轮不替换部署版本，当前推荐仍为old0:novalue；未追加训练、参数扫描或其他科学实验。')
    lines=['G实际计算效率与追捕神经候选/MCTS贡献','',
        f'正式完成{len(fresh)}个全新物理场景、三个原对手、{len(tasks)}条共同S首捕前缀。两个阶段均为固定CPU单进程。',
        f'实际新增闭环后缀{new_suffixes}条，Geometry-only的时间阶段复用固定阶段；无首捕/直接双捕原样计入整局参考口径。',
        '本轮只隔离首捕后一追一逃：Geometry-only后段无MCTS/网络，共同首捕前缀仍使用神经MCTS。',
        'G-no-neural移除追捕方所有树深的神经候选，保留模拟逃逸方网络；其结论不是“全部神经网络可删除”。',
        '模型、奖励、对手、几何引导器均冻结；30个样本用于较大模块差异和串行计时，小差异区间可能不足定论。','',
        '**成功率与实际运行成本。** 成功率按物理场景配对，三个对手共同保留；20000次地图组内bootstrap，种子59020261024。','',
        '|阶段|组|成功/90请求|双捕率 [95%区间]|完成成本 步|后段单步mean/P95 ms|每请求后段追捕计算 ms|',
        '|---|---|---:|---|---:|---|---:|']
    for phase in ('fixed','timed'):
        for uid,g in all_data[phase]['mean']['summary'].items():
            lo,hi=np.array(g['ci95'])*100
            latency='—' if g['mean_decision_ms'] is None else f'{g["mean_decision_ms"]:.3f}/{g["p95_decision_ms"]:.3f}'
            lines.append(f'|{phase}|{uid}|{g["successes"]}/{g["games"]}|{100*g["success_rate"]:.2f}% [{lo:.2f},{hi:.2f}]|{g["mean_completion_cost"]:.2f}|{latency}|{g["mean_suffix_pursuit_ms"]:.2f}|')
    lines+=['','固定阶段为四搜索方案每次1000模拟；时间阶段为50ms实际追捕决策预算。Geometry-only不被迫等待至50ms。网络载入、真实对手计算和轨迹落盘不计入追捕决策预算。',
        'Geometry-only全部成功、部分配对结果全平分，bootstrap区间因此退化；表中的[100%,100%]或[0,0]不代表已精确确定总体成功率或模块等效。','',
        '**实际接管的条件结果。** 仅包含存在可续跑首捕状态的请求；条件样本也按物理初态相关，不作为新增独立样本。后段失败成本为原剩余步数。','',
        '|阶段|组|第二捕成功/接管数|后段完成成本 步|平均剩余步数|每次接管追捕计算 ms|',
        '|---|---|---:|---:|---:|---:|']
    for phase in ('fixed','timed'):
        for uid,g in conditional[phase].items():
            lines.append(f'|{phase}|{uid}|{g["successes"]}/{g["states"]}|{g["mean_completion_cost"]:.2f}|{g["mean_remaining_steps"]:.2f}|{g["mean_pursuit_ms"]:.2f}|')
    lines+=['',
        '**配对差异。** 负的no-neural-G代表删去追捕神经候选后下降；负的Geometry-G代表搜索更好。','',
        '|阶段|比较|成功率差 pp|普通95%区间 pp|同时95%区间 pp|挽回/丢失|完成成本差 步|后段计算时间差 ms [95%区间]|',
        '|---|---|---:|---|---|---:|---:|---|']
    for phase in ('fixed','timed'):
        for name,c in all_data[phase]['mean']['comparisons'].items():
            lo,hi=np.array(c['ci95'])*100;sim=c.get('simultaneous_ci95');st=f'[{sim[0]*100:+.2f},{sim[1]*100:+.2f}]' if sim else '—'
            cl,ch=c['suffix_ms_ci95']
            lines.append(f'|{phase}|{name}|{100*c["delta"]:+.2f}|[{lo:+.2f},{hi:+.2f}]|{st}|{c["gained"]}/{c["lost"]}|{c["completion_cost_delta"]:+.2f}|{c["suffix_ms_delta"]:+.2f} [{cl:+.2f},{ch:+.2f}]|')
    lines+=['','同时区间共同覆盖两阶段的no-neural-G、Geometry-no-neural、Geometry-G六项成功率比较。完整各项完成成本差的区间见statistics.json。','',
        '**同状态、同1000模拟的实际根延迟。** 每个首捕状态预热后各方法重复三次，恢复相同搜索随机状态，方法顺序交错；以物理初态成组统计。','',
        '|组|热根mean ms|热根P95 ms|计时次数|',
        '|---|---:|---:|---:|']
    for uid,g in root.items():
        lines.append(f'|{uid}|{g["mean_ms"]:.3f}|{g["p95_ms"]:.3f}|{g["measurements"]}|')
    if rootdiff:
        lo,hi=rootdiff['ci95'];lines.append(f'\nG-F同输入根延迟差{rootdiff["delta_ms"]:+.3f}ms，95%区间[{lo:+.3f},{hi:+.3f}]。')
    lines+=['','同输入根延迟回答单次搜索的实现开销；完整后缀总耗时还受到动作质量与决策数影响。两者分别报告，不能互相替代。','',
        '根计时先对每个对手状态的三次重复取均值，再在物理场景内汇总；根延迟差区间采用物理场景配对bootstrap。成功率和闭环成本区间另按地图组分层。','',
        '**时间预算实际执行。** 每次模拟完成后检查截止，因此最后一次原子模拟、Python返回或操作系统调度可能造成超时。','',
        '|组|实际mean/P95/max ms|模拟数mean/P95|超时>1ms的决策数|最大超时 ms|触顶次数|',
        '|---|---|---|---:|---:|---:|']
    for uid,g in diag['by_phase']['timed'].items():
        if uid=='Geometry-only':continue
        lines.append(f'|{uid}|{g["latencies_mean"]:.3f}/{g["latencies_p95"]:.3f}/{g["latencies_max"]:.3f}|{g["simulations_mean"]:.1f}/{g["simulations_p95"]:.1f}|{g["deadline_overshoots_gt1ms"]}|{g["deadline_excess_ms_max"]:.3f}|{g["cap_hits"]}|')
    lines+=['','**计算调用量。** 环境转移为求解器模拟转移，不含真实环境推进；Geometry-only仍实际执行动作。','',
        '|阶段|组|后段决策数|模拟环境转移|追捕网络调用|模拟逃逸网络调用|几何查询|',
        '|---|---|---:|---:|---:|---:|---:|']
    for phase in ('fixed','timed'):
        for uid,g in all_data[phase]['mean']['summary'].items():
            d=diag['by_phase'][phase][uid]
            lines.append(f'|{phase}|{uid}|{g["decisions"]}|{g["own_cost"]["step_calls"]}|{d["policy_pursuer"]:.0f}|{d["policy_evader"]:.0f}|{d["geometry_queries"]:.0f}|')
    lines+=['','几何版本完整保留原G引导器的地图/路线准备，F也保留其历史实现中的共同预处理。本轮未顺带删除辅助计算，以下时间反映实际被测实现。','',
        '|阶段|组|几何建表总 ms|几何计划总 ms|每决策建表+计划 ms|',
        '|---|---|---:|---:|---:|']
    for phase in ('fixed','timed'):
        for uid,d in diag['by_phase'][phase].items():
            n=all_data[phase]['mean']['summary'][uid]['decisions']
            lines.append(f'|{phase}|{uid}|{d["graph_build_ms"]:.2f}|{d["plan_ms"]:.2f}|{(d["graph_build_ms"]+d["plan_ms"])/max(n,1):.4f}|')
    lines+=['','**分地图与对手成功率。** 地图家族沿用上一轮，物理初态和动作限幅为全新，不称本轮新增地图家族。','',
        '|地图|阶段|对手|S|F|G|G-no-neural|Geometry-only|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for scope in ('original_four','unseen_four','none','four'):
        for phase in ('fixed','timed'):
            for opponent in ('mean',)+m.prior.OPPONENTS:
                g=result['sections'][scope][phase][opponent]['summary']
                cells=[f'{g[u]["successes"]}/{g[u]["games"]}' for u in m.UIDS]
                lines.append('|'+scope+'|'+phase+'|'+opponent+'|'+'|'.join(cells)+'|')
    lines+=['','**失败分解。** 首捕前失败无法由任何本轮后段方法挽回。','',
        '|阶段|组|零捕|首捕后超时|自身非法|对手非法|',
        '|---|---|---:|---:|---:|---:|']
    for phase in ('fixed','timed'):
        for uid,g in all_data[phase]['mean']['summary'].items():
            lines.append(f'|{phase}|{uid}|{g["zero_capture"]}|{g["one_capture_timeout"]}|{g["own_illegal"]}|{g["opponent_illegal"]}|')
    lines+=['','**四个问题的结论。**','']+conclusions+['','**判断范围与下一步。**',
        '结合历史结果：M/F幅度匹配中，不放大的M比F低44.95pp、比S低21.21pp；随后150新场景验证中，G达到F的效果并降低调用量。本轮进一步显示，相同G几何规则直接控制也能完成全部90次后段接管。现有证据更支持有效追击方向与动作能力利用，尚未证明这部分收益需要追捕神经候选或树搜索评价。',
        '历史依据见[幅度匹配报告](../l45_amplitude_match_20261024/REPORT.md)与[150场景独立验证](../l45_direction_validation_20261024/REPORT.md)。不同轮次的绝对成功率不直接相减，本轮S已达95.56%，存在明显天花板。',
        '对神经网络的结论只针对当前追捕候选，搜索内部对手网络和首捕前S前缀都没有消融。',
        '直接几何对照与no-neural搜索的差异包含候选采样、模拟评价及动作选择；不单独归于某一树组件。',
        '若区间跨零，不能据此断言模块无用；同样不能以一点成功率优势直接替换部署。30个物理状态的区间需结合任务覆盖解读。',
        '本轮没有训练、修改old0:novalue推荐或启动后续实验。具体结果判断见interpretation.json。','',
        '![实际成本与贡献](figures/contributions_review.png)']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (args.output/'CONCLUSION.md').write_text('G效率与神经/MCTS贡献结论\n\n'+'\n\n'.join(conclusions)+'\n\n完整分项与配对统计见[REPORT.md](REPORT.md)。\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['S','F','G','G-no-NN','Geometry']
    x=np.arange(len(m.UIDS));fig,axes=plt.subplots(1,3,figsize=(14,4.3))
    for j,phase in enumerate(('fixed','timed')):
        ss=all_data[phase]['mean']['summary'];y=np.array([ss[u]['success_rate']*100 for u in m.UIDS]);ci=np.array([ss[u]['ci95'] for u in m.UIDS]).T*100
        axes[0].errorbar(x+(j-.5)*.16,y,yerr=np.maximum(0,np.array([y-ci[0],ci[1]-y])),fmt='o',capsize=3,label=phase)
        axes[2].bar(x+(j-.5)*.36,[conditional[phase][u]['mean_completion_cost'] for u in m.UIDS],.36,label=phase)
    axes[0].set(xticks=x,xticklabels=labels,ylim=(88,102),ylabel='Legal double capture (%)',title='30 physical scenes, paired 95% CI')
    axes[0].legend();axes[0].grid(axis='y',alpha=.2)
    axes[1].semilogy(x,[root[u]['mean_ms'] for u in m.UIDS],'o',label='mean')
    axes[1].semilogy(x,[root[u]['p95_ms'] for u in m.UIDS],'x',label='P95')
    axes[1].set(xticks=x,xticklabels=labels,ylabel='Same-input root latency (ms, log)',title='1000 simulations / direct geometry');axes[1].legend();axes[1].grid(axis='y',alpha=.2)
    axes[2].set(xticks=x,xticklabels=labels,ylabel='Post-first-capture steps (failure capped)',title='Shared first-capture states');axes[2].legend()
    fig.text(.5,.015,'Geometry timed result is reused; all-success / all-tie bootstrap intervals are degenerate.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(args.output/'figures/contributions_review.png',dpi=180);plt.close(fig)
    m.o.write_json(args.output/'analysis_provenance.json',dict(script=str(Path(__file__).resolve()),sha256=m.o.sha(__file__),
        inputs={n:m.o.sha(args.output/n) for n in ('statistics.json','diagnostics.json','integrity.json','experiment.json')},
        extra_searches=0,extra_episodes=0))

if __name__=='__main__':main()
