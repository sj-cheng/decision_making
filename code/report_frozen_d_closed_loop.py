"""Render frozen closed-loop results and non-causal trajectory diagnostics."""
from pathlib import Path
import json,collections
import numpy as np
from run_frozen_d_closed_loop import ROOT,OUT,OLD,METHODS,save,sha,statistics,inputs


def number(value,digits=2):return 'N/A' if value is None else f'{value:.{digits}f}'


def interval(item,scale=1):
    return f'{scale*item["mean"]:+.2f} [{scale*item["ci95"][0]:+.2f}, {scale*item["ci95"][1]:+.2f}]'


def figures(stats,audits,episodes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    target=OUT/'figures';target.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    colors=('#4c78a8','#f58518','#54a24b');x=np.arange(3)
    confirm=stats['confirmation']
    axes[0].bar(x,[confirm['summary'][m]['capture_rate']*100 for m in METHODS],color=colors)
    axes[0].set(xticks=x,xticklabels=METHODS,ylim=(0,105),ylabel='Legal capture (%)',title='Unseen confirmation: 24 history blocks')
    for i,m in enumerate(METHODS):
        v=confirm['summary'][m];axes[0].text(i,v['capture_rate']*100+1,f'{v["capture"]}/{v["games"]}',ha='center',fontsize=9)
    for j,stage in enumerate(('development','confirmation')):
        axes[1].bar(x+(j-.5)*.32,[stats[stage]['summary'][m]['completion_cost'] for m in METHODS],.32,label=stage)
    axes[1].set(xticks=x,xticklabels=METHODS,ylabel='Completion cost (failure = remaining steps)',title='Time to capture, failures retained')
    axes[1].legend(fontsize=8)
    for j,(kind,label) in enumerate((('latency','Controller return'),('cycle','Joint cycle'))):
        axes[2].bar(x+(j-.5)*.32,[confirm['summary'][m][kind]['p95_ms'] for m in METHODS],.32,label=label)
    axes[2].axhline(200,color='red',ls='--',label='200 ms own budget')
    axes[2].axhline(1000,color='black',ls=':',label='1 s simulated period')
    axes[2].set(xticks=x,xticklabels=METHODS,ylabel='P95 latency (ms)',title='Full decision return; setup reported separately')
    axes[2].legend(fontsize=7)
    for ext in ('png','pdf'):fig.savefig(target/f'confirmation_summary.{ext}',dpi=170)
    plt.close(fig)
    # Select largest observed excess cost. Selection is explicitly descriptive.
    rows={r['key']:r for r in episodes['confirmation']}
    diagnostic=audits['confirmation']['descriptive_comparisons']
    selected=[]
    for comparison in ('T-G','T+Geo-T'):
        selected.append((comparison,diagnostic[comparison]['slowest'][0]))
    selected.append(('T-G improvement',diagnostic['T-G']['fastest'][0]))
    fig,axes=plt.subplots(3,3,figsize=(12,11),constrained_layout=True)
    for line,(comparison,pair) in enumerate(selected):
        h,w=pair['group'],pair['world']
        family=pair['family']
        for j,method in enumerate(METHODS):
            row=next(r for r in episodes['confirmation'] if r['history_id']==h and r['world']==w and r['method']==method)
            payload=json.loads(Path(row['episode']).read_text());packet,_=inputs(payload['task']);trace=payload['trace']
            at=np.array([r['state'][4:6] for r in trace]+[payload['final_state'][4:6]])
            other=np.array([r['state'][:2] for r in trace]+[payload['final_state'][:2]])
            ax=axes[line,j]
            for b in packet['map']:ax.add_patch(Rectangle((b[0][0],b[1][0]),b[0][1]-b[0][0],b[1][1]-b[1][0],facecolor='.6'))
            ax.plot(at[:,0],at[:,1],color=colors[j],lw=1.5,label='Pursuer')
            ax.plot(other[:,0],other[:,1],color='.25',lw=1,ls='--',label='Actual adversary')
            ax.scatter(*at[0],s=25,color=colors[j],marker='o');ax.scatter(*at[-1],s=35,color=colors[j],marker='x')
            first=row['first_rediscovery']
            if first is not None:ax.scatter(*at[first],s=65,marker='*',color='red',label='First reacquisition')
            ax.set(xlim=(-10,10),ylim=(-10,10),aspect='equal',title=f'{method}: {row["reason"]}, {row["executed_steps"]} steps\n{family}, {w}; first seen={first}')
            ax.grid(alpha=.15)
            if j==0:ax.legend(fontsize=7,loc='lower left')
    fig.suptitle('Largest adverse pairs and a search-improvement example; descriptive selection only',fontsize=11)
    for ext in ('png','pdf'):fig.savefig(target/f'diagnostic_trajectories.{ext}',dpi=170)
    plt.close(fig)


def main():
    assert (OUT/'decision.json').exists() and (OUT/'postrun_integrity.json').exists()
    stats={stage:json.loads((OUT/stage/'statistics.json').read_text()) for stage in ('development','confirmation')}
    audit={stage:json.loads((OUT/stage/'postrun_audit.json').read_text()) for stage in stats}
    episodes={stage:json.loads((OUT/stage/'results.json').read_text()) for stage in stats}
    decision=json.loads((OUT/'decision.json').read_text());preflight=json.loads((OUT/'preflight.json').read_text())
    confirm=stats['confirmation']
    lines=['# 冻结D的G、T、T+Geo真实闭环捕获对照','',
        '2026-10-26，Asia/Shanghai。采用原环境逐步闭环与真实搜索对手，不是固定历史估计覆盖实验，也不是实机实验。','',
        f'**预定确认结论：T相对G{"确认" if decision["reliable_T_over_G"] else "未确认"}可靠额外捕获收益；T+Geo相对T{"确认" if decision["reliable_TGeo_over_T"] else "未确认"}可靠额外捕获收益。**',
        'D及其权重输出保持冻结，保留此前50ms失败记录。本轮不训练神经网络、不制作教师、不修改部署。','',
        '## 主要结果','',
        '开发12组公共历史×A/B×3方法=72局；独立确认24组全新公共历史×A/B×3方法=144局。开发与确认分开统计，未依据开发输赢调参。',
        '以下区间以公共历史为聚类单位，同时保留A/B世界及三方法；不是把48个初态当作48个独立场景。','',
        '|阶段|方法|合法捕获|目标重发现|失败惩罚完成成本|成功局步数|首次重发现步数均值|',
        '|---|---|---:|---:|---:|---:|---:|']
    for stage in ('development','confirmation'):
        for method in METHODS:
            s=stats[stage]['summary'][method]
            lines.append(f'|{stage}|{method}|{s["capture"]}/{s["games"]}|{s["rediscovered"]}/{s["games"]}|{s["completion_cost"]:.2f}|{number(s["success_steps"])}|{number(s["first_rediscovery_mean"])}|')
    lines+=['','失败惩罚完成成本：成功用接管后捕获步数；失败用接管时剩余步数。重新发现包括合法捕获末状态的可见性评分，不将评分回流控制器。','',
        '|确认方法|合法捕获率 [逐历史bootstrap 95% CI]|重发现率 [95% CI]|完成成本 [95% CI] 步|',
        '|---|---:|---:|---:|']
    for method,metrics in audit['confirmation']['marginal_history_bootstrap_intervals'].items():
        def rate(v):return f'{100*v["mean"]:.2f}% [{100*v["ci95"][0]:.2f}, {100*v["ci95"][1]:.2f}]'
        cost=metrics['completion_cost']
        lines.append(f'|{method}|{rate(metrics["capture"])}|{rate(metrics["rediscovered"])}|{cost["mean"]:.2f} [{cost["ci95"][0]:.2f}, {cost["ci95"][1]:.2f}]|')
    lines+=['','单方法区间是辅助边际区间；主结论使用同历史配对差及两比较的同时区间。','',
        '|独立确认主比较|捕获率差 pp：同时95% CI|成本差 步：配对95% CI|预定额外收益门槛|',
        '|---|---:|---:|---|']
    for key in ('T-G','T+Geo-T'):
        c=confirm['comparisons'][key];v=c['capture'];ci=v['simultaneous_ci95']
        lines.append(f'|{key}|{100*v["mean"]:+.2f} [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]|{interval(c["completion_cost"])}|{"通过" if c["reliable_extra_capture_value"] else "未通过"}|')
    lines+=['','门槛为捕获差同时95%下界>0、成本差95%上界≤0，且无新增工程失败机制。配对bootstrap10000次、种子81120261026，两项主捕获比较共用max-absolute-centered半径。',
        '效率/首次重发现步数是次终点，需与捕获主终点区别；未通过主门槛不能宣称捕获贡献已成立。','',
        '|独立确认辅助比较|捕获率差 pp：普通95% CI|重发现率差 pp：普通95% CI|成本差 步：普通95% CI|',
        '|---|---:|---:|---:|']
    for key,c in confirm['comparisons'].items():
        lines.append(f'|{key}|{interval(c["capture"],100)}|{interval(c["rediscovered"],100)}|{interval(c["completion_cost"])}|')
    if any(c['capture']['ci95']==[0.,0.] for c in confirm['comparisons'].values()):
        lines+=['','**退化区间的限制：**若某配对捕获比较在全部公共历史上均相同，经验bootstrap只能给出[0,0]。这说明样本未观察到不一致，不能证明总体等效、零风险或零潜在差异。',
            f'24个历史块零不一致时，不一致事件率的一侧97.5%零事件上界约{100*audit["confirmation"]["zero_discordance_one_sided_97_5_bound"]:.2f}%。这是事件率诊断，不能当作捕获率差的等效区间。']
    lines+=['','![确认结果与计算开销](figures/confirmation_summary.png)','',
        '## 为什么采用200ms及公平性边界','',
        '仓库本研究环境高层dt=1s、tf=80s；MINCO的0.1s低层piece_dt不能当作本轮高层周期。本轮use_minco_dynamics=False，仍是轴限速的直接位置增量动力学。',
        '在开发效果运行前登记200ms完整追捕决策预算，占仿真高层周期20%。这是研究配置，不是实机周期保证。每次更新、按需建图、候选、树准备、搜索、最终检查和函数返回均计入。',
        'T/T+Geo共同最多1000模拟、深度8、预留2ms；G保持低成本，不人为等到200ms。**这是同一预算上限加同一模拟上限，不是每步强行耗满相同CPU时间**。',
        '超过200ms仍保留原结果并报告，不删局、不换估计器、不移动建图计费。此前50ms失败报告保持原样：[旧审计](../l45_weighted_search_interface_v2_20261026/REPORT.md)。',
        '初次从保存历史接管的地图/加载、history[:-1]初始化和首次决策单列；重新可见后再遮挡的重建包含在当步实际开销中。',
        '仿真采用锁步：计算期间不额外推进机器人或年龄化观测，超时是计算指标，不自动变成延迟动力学。捕获结论因此只适用于该锁步协议；不能据此宣称实机带延迟闭环也成立。',
        '双方从相同决策前状态分别选动作，再用原Python Example8联合推进；真实对手为原RootPUCT，3000模拟、深度100、不含网络、拥有全状态信息。每组对手响应自己形成的当前状态，不能复用其他组未来动作。',
        '对手原始动作不合法时使用同一预登记停止盾并记录，未为某方法换对手。该对手策略不同于D的运动先验；“共用模拟模型”不等于“模拟模型准确描述真实对手”。','',
        '## 闭环接口和独立复核','',
        f'工程预检9个6步前缀通过：三拓扑×三方法，共54个决策；固定实际模拟数重放9/9逐位一致。预检自身P95={preflight["latency"]["p95_ms"]:.2f}ms，联合周期P95={preflight["cycle"]["p95_ms"]:.2f}ms。',
        '三方法公共历史输入一致时D位置、权重与估计随机状态一致。每个成组初态的首个真实对手原始动作也一致，验证其不依赖追捕者本步已选动作。',
        '控制器只收Observation，隐藏时target=None；真值只用于传感器边界、环境、真实对手及事后评分，未用于填粒子、定候选或选动作。',
        '运行结束后逐局检查轨迹散列、成组初态与种子、独立几何观测、双方扫掠动作、一步联合运动、合法捕获、64权重归一化。按所有保存的公开观测重新运行冻结D，逐帧粒子及权重精确相同。',
        '完整复核数据：[开发审计](development/postrun_audit.json)、[确认审计](confirmation/postrun_audit.json)、[源码与部署完整性](postrun_integrity.json)。','',
        '|阶段|模拟联合转移独立核验数量|独立复核记录错误数|真实对手停止盾触发|',
        '|---|---:|---:|---:|']
    for stage in ('development','confirmation'):
        s=stats[stage]['summary']
        lines.append(f'|{stage}|{sum(s[m]["simulated_transitions"] for m in METHODS):,}|{len(audit[stage]["errors"])}|{sum(s[m]["shield_count"] for m in METHODS)}|')
    lines+=['','## 困难遮挡、墙角和多通路','',
        '|确认拓扑|方法|合法捕获|重发现|成本 步|首次重发现 步|',
        '|---|---|---:|---:|---:|---:|']
    for family,part in confirm['by_family'].items():
        for method,s in part.items():
            lines.append(f'|{family}|{method}|{s["capture"]}/{s["games"]}|{s["rediscovered"]}/{s["games"]}|{s["completion_cost"]:.2f}|{number(s["first_rediscovery_mean"])}|')
    long=confirm['by_family']['long_wall'];multi=confirm['by_family']['multi_route'];corner=confirm['by_family']['wall_corner']
    lines+=['',f'分层现象：长墙G {long["G"]["capture"]}/16，T和T+Geo各{long["T"]["capture"]}/16，存在搜索挽救G未捕获初态的正例；多通路G {multi["G"]["capture"]}/16，T与T+Geo各{multi["T"]["capture"]}/16，是当前搜索的主要失效区。',
        f'墙角三组均{corner["G"]["capture"]}/16捕获，但G平均{corner["G"]["completion_cost"]:.2f}步，T {corner["T"]["completion_cost"]:.2f}步，T+Geo {corner["T+Geo"]["completion_cost"]:.2f}步；捕获持平没有转化为执行效率优势。',
        '子拓扑样本每方法16局、8个公共历史；上述分层均值为描述性分析，不是额外独立确认，未对它们追调算法。','',
        '## 完整计算时延','',
        '|确认方法|自身返回均值/P95/最大 ms|联合周期P95/最大 ms|追捕方冷启动含首决策P95 ms|>50ms / >200ms|联合周期>1s|',
        '|---|---:|---:|---:|---:|---:|']
    for method,s in confirm['summary'].items():
        a=audit['confirmation']['summary'][method]
        lines.append(f'|{method}|{s["latency"]["mean_ms"]:.2f}/{s["latency"]["p95_ms"]:.2f}/{s["latency"]["maximum_ms"]:.2f}|{s["cycle"]["p95_ms"]:.2f}/{s["cycle"]["maximum_ms"]:.2f}|{s["cold_start_first_decision"]["p95_ms"]:.2f}|{a["over_50ms"]}/{a["over_200ms"]}（{a["decisions"]}次）|{a["over_high_level_period"]}|')
    lines+=['','联合周期包含局部传感、控制器、真实对手搜索、双方动作合法性和环境推进；独立轨迹审计、文件落盘不混入控制器耗时，也不将其称为实机端到端延迟。',
        '各方法串行执行、数值线程1、按历史/世界轮换顺序。没有将G/T/T+Geo同时运行来争抢同一核；未固定CPU亲和性，不作硬实时保证。','',
        '|确认方法|静态准备P95 ms|历史初始化P95 ms|可见决策P95 ms|普通隐藏P95 ms|遮挡重建帧P95 ms|隐藏帧平均模拟数|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method,s in audit['confirmation']['summary'].items():
        region=s['regions']
        lines.append(f'|{method}|{s["initialize_static"]["p95_ms"]:.2f}|{s["initialize_history"]["p95_ms"]:.2f}|{number(region.get("visible",{}).get("own_ms",{}).get("p95_ms"))}|{number(region.get("ordinary_hidden",{}).get("own_ms",{}).get("p95_ms"))}|{number(region.get("rebuild",{}).get("own_ms",{}).get("p95_ms"))}|{s["hidden_simulations_mean"]:.1f}|')
    lines+=['','时延分量、重建次数、超过预算的每帧和对手耗时均保存在postrun_audit.json与逐局轨迹。P95不能当作最大耗时保证；冷启动与正常单步不能互换。','',
        '本批闭环中没有实际发生“重新可见后再遮挡”的建图帧，故表中该列为N/A；首段隐藏的图构造已在history[:-1]初始化内计费。**不能将这里普通决策的低时延解释为旧重建瓶颈已修复**，旧审计的重建失败记录仍有效。',
        '真实对手对象/原生包装器的初始化单列在initialization/opponent_setup_ms；对手逐步搜索包含在联合周期，不混入追捕方200ms预算。','',
        '## 失败对局与原因','',
        '以下列出全部未合法捕获对局，开发与确认不合并计算置信区间。没有失败不等于证明总体成功率100%。','',
        '|阶段|方法/世界|公共历史|终止原因|执行步数|首次重发现|同初态G捕获/成本|',
        '|---|---|---|---|---:|---:|---|']
    failure_count=0
    for stage in ('development','confirmation'):
        index={(r['history_id'],r['world'],r['method']):r for r in episodes[stage]}
        for r in episodes[stage]:
            if r['capture']:continue
            failure_count+=1;g=index[(r['history_id'],r['world'],'G')]
            link=str(Path(r['episode']).relative_to(OUT))
            lines.append(f'|{stage}|{r["method"]}/{r["world"]}|[{r["history_id"]}]({link})|{r["reason"]}|{r["executed_steps"]}|{r["first_rediscovery"]}|{g["capture"]}/{g["completion_cost"]}|')
    if failure_count==0:lines.append('|全部|—|—|未观察到未捕获局|—|—|—|')
    lines+=['','|失败行为分类（描述性）|开发|独立确认|','|---|---:|---:|']
    categories=('engineering_failure','not_rediscovered','rediscovered_with_less_than_8_steps_left','rediscovered_but_not_captured_by_horizon')
    for category in categories:
        counts=[sum(r['failure_type']==category for r in audit[stage]['episode_diagnostics']) for stage in ('development','confirmation')]
        lines.append(f'|{category}|{counts[0]}|{counts[1]}|')
    lines+=['','为区别“无法重新发现”和“重新发现太晚”，下列次终点按公共历史配对（只在双方都重新发现的历史上计算，存在选择限制，不能替代主终点）。','',
        '|确认比较|首次重发现步数差 [95% CI]|重发现后完成步数差 [95% CI]|成本更慢/更快/相同的成组初态数|',
        '|---|---:|---:|---:|']
    for key,c in audit['confirmation']['descriptive_comparisons'].items():
        lines.append(f'|{key}|{interval(c["first_rediscovery_delta"]) if c["first_rediscovery_delta"] else "N/A"}|{interval(c["post_discovery_steps_delta"]) if c["post_discovery_steps_delta"] else "N/A"}|{c["slower"]}/{c["faster"]}/{c["ties"]}|')
    lines+=['','|确认方法|隐藏决策数|隐藏停止动作|所有根累计回报为0的隐藏决策|隐藏粒子1m覆盖（逐历史均值）|',
        '|---|---:|---:|---:|---:|']
    for method,s in audit['confirmation']['summary'].items():
        lines.append(f'|{method}|{s["hidden_decisions"]}|{s["hidden_stop_steps"]}|{s["hidden_zero_root_return_decisions"]}|{100*s["hidden_coverage_history_equal"]:.2f}%|')
    lines+=['','覆盖率仅由评分器事后读取真值，控制器从未获得该量；三方法闭环观察不同，覆盖率差不能直接归因为估计器版本差（D相同）。',
        'T与T+Geo所有隐藏决策都完成1000次模拟，未触及200ms时限；所有根回报为0的隐藏决策为0。因此本批退化不能归为预算被截断、零次搜索或整棵树没有任何回报。',
        '确认中未捕获共26局：17局未重新发现，8局发现时剩余不足8步，1局已发现但未能在时限前捕获；没有控制器异常、非法动作或模拟非法截止。未重新发现与发现太晚是可直接复核的主要行为失败。',
        '几何根候选的收益需要T+Geo−T回答；不能因其某局比G快、或相比T少失败，就跳过成组确认区间。',
        '本轮树只看8步、以重发现/捕获为模拟回报，且没有未来观测条件分支；D的随机运动先验和真实全状态搜索对手也不一致。这些是已登记的机制限制，可能造成短期可见性回报与实际赶路/截获不一致。',
        '轨迹能直接支持“绕路、反复换向、重发现偏晚”等行为诊断；没有单独更换深度/奖励/模型的消融，不能把某一个机制写成已证实的唯一原因，也不能据此直接叠加价值网络。',
        'G也有明确失败：确认多通路`multi_route_H6_h8.468628_v1.3_3506197457`的A世界，G停止42步、同位置最多重复43次，全局未重新发现。按保存的公开样本/权重重算最早三个停止帧（第36–38步），停止在共同几何评分中是唯一最优，分数领先约0.09–0.13。',
        '这项[事后评分复核](geometry_failure_probe.json)表明该例停滞来自当前期望路程/静态可视评分选择，不是动作非法回退或随机平局；但没有修改G，也不把一个例子外推成所有G失败的唯一原因。',
        '![描述性选择的慢局轨迹](figures/diagnostic_trajectories.png)',
        '图前两行按实际完成成本差最大的两个负例选取，第三行保留T挽救G未捕获初态的正例；均只用于描述。每行三个子图的真实对手分别响应各方法，虚线不应相互交换作为共同未来。','',
        '## 结论、限制与研究决定','',
        f'- T−G：预定额外捕获收益门槛{"通过" if decision["reliable_T_over_G"] else "未通过"}；捕获差、失败惩罚步数差均见主表，不用单局例子代替结论。',
        f'- T+Geo−T：预定额外捕获收益门槛{"通过" if decision["reliable_TGeo_over_T"] else "未通过"}；候选的独立贡献必须按这一比较判断。',
        '- 本轮结论限于冻结D、当前强几何G、8步开环UCT、3000模拟全状态对手、三类难拓扑实例与200ms上限，不是所有MCTS均无效的证明。',
        '- 独立确认是未见高度/平移/旋转/公开历史实例，并非从未见过的拓扑类别；每世界只有一个预登记随机流，尚无跨对手或实机泛化结论。',
        '- 200ms放宽使闭环效果可以被评价，不会抹去此前50ms失败；若本轮有超200ms或超1s，照实保留。',
        '- 未确认搜索和几何候选额外收益前，继续保留G为强研究基线，不制作教师、不训练NTE/其他网络、不改部署，不自动继续叠加模块。','',
        '## 数据、源码与复现','',
        '- [结果前登记方案](../../code/L45_CLOSED_LOOP_D_20261026.md)、[冻结运行配置和输入散列](experiment.json)、[新确认银行](bank_frozen.json)、[独立几何验收](bank_acceptance.json)。',
        '- [工程预检](preflight.json)、[开发72局结果](development/results.json)、[开发聚类统计](development/statistics.json)、[确认144局结果](confirmation/results.json)、[确认聚类统计](confirmation/statistics.json)、[最终研究决定](decision.json)。',
        '- 所有逐步状态、公开观测、动作、64位置/权重、根统计、搜索随机数、模拟轨迹散列与时延在每局episode文件；模型/部署源码未变，源文件与结果散列另见DELIVERY_SHA256.json。',
        '- [执行器](../../code/run_frozen_d_closed_loop.py)、[结束后独立复核](../../code/analyze_frozen_d_closed_loop.py)、[报告生成器](../../code/report_frozen_d_closed_loop.py)。',
        '- 四项闭环测试通过：独立接触几何、原环境联合动作、成组统计与成本门槛、预算放宽不修改旧失败或D。源码和配置在效果结果生成前冻结，未做确认集调参。','',
        '```bash',
        'PY=/home/sjc/miniconda3/envs/dm_env_py311/bin/python',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" -m unittest discover -s code/tests -p test_frozen_d_closed_loop.py -v',
        '# 以下仅复核已完成结果，不重新生成确认集、不重复计时对局',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" code/analyze_frozen_d_closed_loop.py',
        '"$PY" code/report_frozen_d_closed_loop.py',
        '```','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    figures(stats,audit,episodes)


if __name__=='__main__':main()
