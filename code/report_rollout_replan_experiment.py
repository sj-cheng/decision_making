"""Render the single rollout-mechanism experiment without tuning actors."""
from pathlib import Path
import json,collections,argparse
import numpy as np
from run_rollout_replan_experiment import ROOT,OUT,METHODS,MAIN,save


def ci(v,scale=1):return f'{scale*v["mean"]:+.2f} [{scale*v["ci95"][0]:+.2f}, {scale*v["ci95"][1]:+.2f}]'
def num(v):return 'N/A' if v is None else f'{v:.2f}'


def main():
    stats={s:json.loads((OUT/s/'statistics.json').read_text()) for s in ('development','confirmation')}
    audits={s:json.loads((OUT/s/'postrun_audit.json').read_text()) for s in stats}
    results={s:json.loads((OUT/s/'results.json').read_text()) for s in stats}
    decision=json.loads((OUT/'decision.json').read_text());pre=json.loads((OUT/'preflight.json').read_text())
    confirm=stats['confirmation']
    lines=['# 单机制真实闭环：rollout按自身位置重算几何方向','',
        '2026-10-26，Asia/Shanghai。D及原G/T/T+Geo冻结；新增T-R与T+Geo-R，仅改变rollout中自身几何动作的重算。','',
        f'**最终研究决定：{"停止这一修改方向，不继续叠加模块" if decision["stop_modification_direction"] else "至少一项修正通过预定收益门槛，仍不自动训练网络或修改部署"}。**',
        '本轮没有神经训练、教师生成、价值网络或部署变更。以下结论区分工程验收、相对原版改善及相对强G的额外价值。','',
        '## 只改了什么','',
        '原版在一次树扩展后的rollout中重复根G位移；修正版每步用模拟追捕者当前位姿，调用原G的带权几何候选函数取得合法动作。',
        '输入仅为根64位置/权重/几何区域、公开地图和模拟自身位置；不接收模拟目标位置或模式，更不接收真实隐藏状态。',
        '信念仍是当步根快照，未添加未来观测滤波、观测树分支、叶价值或新奖励；这只是自身几何重算，不是完整belief-tree修复。',
        '根候选、深层9动作、UCT、8步规划、首次重发现+1/捕获+1、0.95折扣、D的目标模式转移及真实对手均保留。',
        '缓存仅在一个真实决策中按精确浮点自身位置复用原G输出，不跨观测、不量化坐标。缓存与直算固定模拟数结果逐位相同。','',
        '## 先完成工程验收','',
        '- 关闭重算并恢复原合法停止行为时，新内核与原内核固定32模拟的动作、RNG、根访问/回报和模拟轨迹精确一致。',
        '- 缓存/直算、固定种子重放、改变模拟隐藏目标但保持同位姿的几何动作隔离、部分模拟预算中断不回传均通过4项单元测试。',
        f'- 12公共历史×五方法×两次计时，共{pre["total_root_decisions"]}次根决策；独立检查{pre["independent_transitions"]:,}条模拟转移，工程门槛通过后才运行开发完整对局。',
        '- 每组修正版另做固定32模拟重放、缓存/直算固定8模拟核对；固定数测试关闭时限中断，不能把它们称为200ms效果实验。','',
        '|方法|工程决策P95 ms|工程初始化+首决策P95 ms|最少完成模拟数|',
        '|---|---:|---:|---:|']
    for m,s in pre['timing'].items():lines.append(f'|{m}|{s["decision"]["p95_ms"]:.2f}|{s["startup"]["p95_ms"]:.2f}|{s["min_simulations"]}|')
    lines+=['','工程第一版测试桩曾漏掉原版的合法停止回退；只修正了测试桩，未改算法。冻结前草稿配置保留在before_test_stub_correction_experiment.json。','',
        '## 真实闭环结果','',
        '开发12组已见公共历史×A/B×五方法=120局；确认24组全新公共历史×A/B×五方法=240局。连续高度/平移/旋转/停顿随机实例与全部旧公共历史不重叠，仍属于三类已知拓扑。',
        '算法和确认银行在开发完整结果前冻结，未依据开发输赢调参。真实对手为无网络RootPUCT，3000模拟、深度100，全状态信息；分别响应各方法当前状态，不复用未来动作。',
        '双方从共同决策前状态独立选动作，再用原Python Example8直接运动环境联合推进；dt=1s、范围6m、轴限速与80步时限保持。','',
        '|阶段|方法|合法捕获|重发现|失败惩罚完成成本|失败惩罚重发现时间|成功局步数|',
        '|---|---|---:|---:|---:|---:|---:|']
    for stage in stats:
        for m,s in stats[stage]['summary'].items():
            lines.append(f'|{stage}|{m}|{s["capture"]}/{s["games"]}|{s["rediscovered"]}/{s["games"]}|{s["completion_cost"]:.2f}|{s["rediscovery_cost"]:.2f}|{num(s["success_steps"])}|')
    lines+=['','完成成本：成功用接管后捕获步数，未捕获用接管时剩余步数。重发现时间：发现用首次发现步数，未发现用原剩余步数。失败不能从均值删除。',
        '重发现包含合法捕获末状态的独立可见性评分，不回流控制器；条件成功局步数与条件重发现时间不能代替失败惩罚指标。','',
        '|确认主比较|捕获差pp 同时95% CI|完成成本差步 普通95% CI|重发现成本差步 普通95% CI|可靠额外捕获收益|',
        '|---|---:|---:|---:|---|']
    for a,b in MAIN:
        c=confirm['comparisons'][a+'__'+b];p=c['capture'];v=p['simultaneous_ci95']
        lines.append(f'|{a} − {b}|{100*p["mean"]:+.2f} [{100*v[0]:+.2f}, {100*v[1]:+.2f}]|{ci(c["completion_cost"])}|{ci(c["rediscovery_cost"])}|{"通过" if c["reliable_extra_capture_value"] else "未通过"}|')
    lines+=['','bootstrap按24公共历史成组，组内A/B和五方法共同保留，10000次，种子83120261026。四项主捕获比较使用共同max-absolute-centered同时95%区间。',
        '可靠捕获收益需捕获差同时95%下界>0、成本差普通95%上界≤0且无新工程失败；只改善成本或重发现时间，不称捕获收益成立。',
        '继续该修改方向还要求同时胜过对应原版与强G；区间跨0不是等效或无差异的证明。','',
        '## 多通路与其他难场景','',
        '|确认拓扑|方法|合法捕获|重发现|完成成本|条件首次重发现步数|',
        '|---|---|---:|---:|---:|---:|']
    for f,part in confirm['by_family'].items():
        for m,s in part.items():lines.append(f'|{f}|{m}|{s["capture"]}/{s["games"]}|{s["rediscovered"]}/{s["games"]}|{s["completion_cost"]:.2f}|{num(s["first_rediscovery_mean"])}|')
    lines+=['','每拓扑8个公共历史、每方法16局；此分层是描述性结果，不能挑出一类正例替代总体独立确认。','',
        '## 200ms内真实计算代价','',
        '五方法共同扣除首次地图/加载和history[:-1]初始化时间，再把首次200ms预算的剩余时间交给act；后续每步完整200ms。原算法源码未改，共同预算适配层单独记录。',
        'P95/最大值使用函数外实际返回计时；包含估计、按需建图、候选、缓存、树准备、搜索和最终合法性检查。每个进程先在串行锁内JIT预热，预热另存pipeline_warmup.json。',
        '预算不足的部分模拟单列，不回传；已执行合法转移保留审计。其root采样计入attempted，未计入completed。固定种子精确重放指固定完成模拟数、无中断的检查，不声称壁钟中断次数天然逐位可复现。','',
        '|确认方法|决策P95/最大ms|首次含初始化P95/最大ms|决策>200ms|首次总计>200ms|联合周期>1s|隐藏帧平均完成模拟|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m,s in audits['confirmation']['summary'].items():
        lines.append(f'|{m}|{s["own_timing"]["p95_ms"]:.2f}/{s["own_timing"]["maximum_ms"]:.2f}|{s["first_charged_timing"]["p95_ms"]:.2f}/{s["first_charged_timing"]["maximum_ms"]:.2f}|{s["over_200ms"]}|{s["first_charged_over_200ms"]}|{s["over_cycle_period"]}|{s["hidden_simulations_mean"]:.1f}|')
    lines+=['','相同200ms是相同上限，不要求G等待或强迫原版耗满。重算几何增加了每模拟成本，修正版模拟数会减少；不能将此时间预算对照当作同模拟数消融。',
        '原50ms失败不覆盖。计算期间环境锁步，不模拟观测年龄或机器人继续运动；联合周期含真实对手和环境，但不含记录落盘/独立审计，不是实机延迟保证。','',
        '|确认方法|隐藏帧数|未遍历全部合法根候选帧|仅单次根访问帧|隐藏停止动作|仅单次访问且停止|预算中断调用数|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m,s in audits['confirmation']['summary'].items():
        lines.append(f'|{m}|{s["hidden_decisions"]}|{s["hidden_incomplete_root_pools"] if m!="G" else "N/A"}|{s["hidden_only_single_visit_decisions"]}|{s["hidden_stops"]}|{s["single_visit_and_stop"]}|{s["budget_abort_calls"]}|')
    lines+=['','“未遍历”按合法候选计数，排除原T罗盘中不合法的方向；末次中断创建但访问为0的根节点不算已经评估。',
        '原根顺序、最大访问次数选动作及平局规则保持不变。模拟少时的候选访问不足/平局是需要报告的计算后果，本轮没有另改规则以补偿。','',
        '|修正版|缓存命中/未命中|几何查询总秒|几何查询占决策总时长|',
        '|---|---:|---:|---:|']
    for m in ('T-R','T+Geo-R'):
        s=audits['confirmation']['summary'][m]
        lines.append(f'|{m}|{s["cache_hits"]}/{s["cache_misses"]}|{s["geometry_seconds"]:.2f}|{100*s["query_share_of_decision_time"]:.1f}%|')
    tr=audits['confirmation']['summary']['T-R'];geo=audits['confirmation']['summary']['T+Geo-R']
    lines+=['',f'本确认中T-R全部{tr["hidden_decisions"]}个隐藏决策均完成1000次模拟，无预算中断；T+Geo-R平均{geo["hidden_simulations_mean"]:.1f}次，最少{geo["hidden_simulations_min"]}次。两种修正版未遍历完合法根候选的帧均为0，也无仅单次访问选动作帧。',
        '因此不能用初版执行剖析中的“根候选来不及评估”解释本确认结论；尤其T-R已与原T匹配1000次模拟，仍未确认收益。T+Geo-R的模拟次数减少则仍是实践代价之一，不与策略效应强行分离。']
    lines+=['','## 失败与信息隔离复核','',
        '完整失败清单在failure_cases.json，逐局轨迹保留真实状态、公开观测、64带权样本、双方动作、根统计、完成与中断模拟数及时延。真值仅供环境、真实对手和事后评分。',
        '运行结束后按所有公开观测精确重放冻结D，并核对双方独立几何、传感器、联合推进、合法捕获、权重、原始对手首动作一致以及预算中断统计。','',
        '|阶段|独立核验模拟转移|复核错误数|',
        '|---|---:|---:|']
    for stage,a in audits.items():lines.append(f'|{stage}|{a["simulated_transitions"]:,}|{len(a["errors"])}|')
    lines+=['','|失败行为类别|开发|确认|','|---|---:|---:|']
    for category in ('engineering_failure','not_rediscovered','late_rediscovery','rediscovered_not_captured'):
        counts=[sum(e['failure_class']==category for e in audits[s]['episode_failure_diagnostics']) for s in stats]
        lines.append(f'|{category}|{counts[0]}|{counts[1]}|')
    failures=[r for stage in results for r in results[stage] if not r['capture']]
    save(OUT/'failure_cases.json',failures)
    index={(r['history_id'],r['world'],r['method']):r for r in results['confirmation']}
    failure_diagnostics={r['key']:r for r in audits['confirmation']['episode_failure_diagnostics']}
    lines+=['','|确认修正版失败例（每法取与G成本差最大的三例）|世界|首次重发现|修正版/G成本|失败类型|',
        '|---|---|---:|---:|---|']
    for method in ('T-R','T+Geo-R'):
        selected=[r for r in results['confirmation'] if r['method']==method and not r['capture'] and index[(r['history_id'],r['world'],'G')]['capture']]
        selected.sort(key=lambda r:r['completion_cost']-index[(r['history_id'],r['world'],'G')]['completion_cost'],reverse=True)
        for r in selected[:3]:
            g=index[(r['history_id'],r['world'],'G')];link=str(Path(r['episode']).relative_to(OUT))
            lines.append(f'|[{r["key"]}]({link})|{r["world"]}|{r["first_rediscovery"]}|{r["completion_cost"]}/{g["completion_cost"]}|{failure_diagnostics[r["key"]]["failure_class"]}|')
    lines+=['','这些例子按成本差作描述性选择，不代替总体成组统计，也不用于再调整版本。','',
        '|确认与对应原版的逐初态变化|原版失败被挽救|原版成功变失败|',
        '|---|---:|---:|']
    for revised,original in (('T-R','T'),('T+Geo-R','T+Geo')):
        own=[r for r in results['confirmation'] if r['method']==revised]
        gained=sum(r['capture'] and not index[(r['history_id'],r['world'],original)]['capture'] for r in own)
        lost=sum(not r['capture'] and index[(r['history_id'],r['world'],original)]['capture'] for r in own)
        lines.append(f'|{revised} − {original}|{gained}|{lost}|')
    lines+=['','不把模型中更合理的几何延续当作真实捕获收益。若真实闭环没有可靠改善，必须同时看搜索次数减少、根候选访问、目标模型失配与无未来观测分支等限制；本轮没有单独修改它们，不能唯一归因。','',
        '## 研究决定','',
        f'- T-R：相对原版并相对强G均通过门槛：{decision["reliable_T_revision"]}。',
        f'- T+Geo-R：相对原版并相对强G均通过门槛：{decision["reliable_TGeo_revision"]}。',
        f'- 停止此修改方向：{decision["stop_modification_direction"]}。不继续自动增加深度、奖励、价值网络、PPO或其他模块。',
        '- D与原始G/T/T+Geo及历史数据保持冻结，强G基线保留；无神经网络训练、无教师生成、无部署修改。','',
        '## 数据与源码','',
        '- [结果前方案](../../code/L45_ROLLOUT_REPLAN_20261026.md)、[冻结源码和配置](experiment.json)、[新确认银行](bank_frozen.json)、[几何验收](bank_acceptance.json)。',
        '- [工程门槛](preflight.json)、[开发120局](development/results.json)、[开发统计](development/statistics.json)、[确认240局](confirmation/results.json)、[确认统计](confirmation/statistics.json)。',
        '- [开发逐帧复核](development/postrun_audit.json)、[确认逐帧复核](confirmation/postrun_audit.json)、[最终决定](decision.json)、[失败清单](failure_cases.json)。',
        '- [独立研究solver](../../code/solvers/rollout_replan.py)、[共用实验执行器](../../code/run_rollout_replan_experiment.py)。','']
    if (OUT/'execution_revision.json').exists():
        lines+=['- [本轮精确几何执行实现](../../code/solvers/rollout_geometry_exact.py)、[本轮实际启动器](../../code/launch_rollout_replan_exact.py)。','']
    else:
        lines+=['- [初版预热启动器](../../code/launch_rollout_replan.py)。','']
    if (OUT/'execution_revision.json').exists():
        insertion=['## 精确执行版本与初版记录','',
            '初版逐次直算几何在开发完成62/120局后暴露根访问不足；已完成数据全部保留，开发未完成、确认未运行，不对该部分样本作确认推断。',
            '本执行版本仅预计算根样本/公开顶点到固定可视图顶点的路程尾代价；当前位置的可见首段、合法性、候选顺序/去重及评分累加仍逐次计算。',
            '198个地图/位置检查的候选数组及动作逐字节相同，固定64完成模拟的动作、RNG、根访问/回报及轨迹精确一致；这是一项执行复用，不是另一种控制策略。',
            '同一策略在真实200ms中可完成更多模拟，因此计算速度会影响实践结果。本目录重做了全部工程、120局开发和240局确认，没有把初版部分数据混入统计。',
            '新确认银行在初版前已经冻结，执行复用前没有运行任何确认对局，故沿用该银行而不重新选例。',
            '见[精确执行登记](../../code/L45_ROLLOUT_REPLAN_EXACT_EXECUTION_20261026.md)、[执行源码散列](execution_revision.json)、[初版记录](../l45_rollout_replan_20261026/REPORT.md)。','']
        offset=lines.index('## 先完成工程验收')
        lines=lines[:offset]+insertion+lines[offset:]
        lines=[line.replace('冻结前草稿配置保留在before_test_stub_correction_experiment.json。','冻结前草稿配置保留在初版目录的before_test_stub_correction_experiment.json。') for line in lines]
    lines+=['![确认捕获、成本与计算工作量](figures/confirmation.png)','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True);x=np.arange(5)
    colors=['#4c78a8','#f58518','#e45756','#72b7b2','#54a24b']
    axes[0].bar(x,[confirm['summary'][m]['capture_rate']*100 for m in METHODS],color=colors)
    axes[0].set(xticks=x,xticklabels=METHODS,ylim=(0,105),ylabel='Legal capture (%)',title='Unseen confirmation: 24 history groups')
    axes[1].bar(x,[confirm['summary'][m]['completion_cost'] for m in METHODS],color=colors)
    axes[1].set(xticks=x,xticklabels=METHODS,ylabel='Failure-penalized completion steps',title='Failures remain in completion cost')
    axes[2].bar(x,[audits['confirmation']['summary'][m]['hidden_simulations_mean'] for m in METHODS],color=colors)
    axes[2].set(xticks=x,xticklabels=METHODS,ylabel='Completed simulations / hidden decision',title='Same 200 ms cap, measured work differs')
    (OUT/'figures').mkdir(exist_ok=True)
    fig.savefig(OUT/'figures/confirmation.png',dpi=170);fig.savefig(OUT/'figures/confirmation.pdf');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output.resolve();main()
