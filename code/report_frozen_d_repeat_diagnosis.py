"""Analyze exact repeats and disclose limits of the offline 2x2 diagnostic."""
from pathlib import Path
import json,collections,hashlib
import numpy as np
from repeat_frozen_d_closed_loop import ROOT,SOURCE,OUT
from run_frozen_d_closed_loop import save,sha

CONTINUATIONS=('repeated_root_G_displacement','replan_G_using_fixed_root_belief')


def paired(values):
    values=np.asarray(values,float)
    draws=np.random.default_rng(82120261026).integers(0,len(values),(10000,len(values)))
    return dict(mean=float(values.mean()),ci95=np.quantile(values[draws].mean(1),[.025,.975]).tolist())


def analyze(data):
    rows=data['rows'];groups=sorted({r['group'] for r in rows});index={}
    for r in rows:
        for c in r['cases']:index[(r['group'],r['root_label'],c['continuation'],c['horizon'],c['method'])]=c
    comparisons={};summary={}
    for continuation in CONTINUATIONS:
        for horizon in (8,24):
            key=continuation+'_'+str(horizon)
            summary[key]={m:{field:float(np.mean([c[field] for r in rows for c in r['cases'] if
                c['continuation']==continuation and c['horizon']==horizon and c['method']==m])) for field in
                ('model_rediscovery','model_capture','model_return','static_geometric_progress','illegal_displacement_fallbacks','stop_steps')}
                for m in ('G','T','T+Geo')}
            comparisons[key]={}
            for a,b in (('T','G'),('T+Geo','G'),('T+Geo','T')):
                result={}
                for field in ('model_return','model_capture','model_rediscovery','static_geometric_progress'):
                    values=[np.mean([index[(g,label,continuation,horizon,a)][field]-index[(g,label,continuation,horizon,b)][field]
                        for label in ('handoff','hidden_midpoint')]) for g in groups]
                    result[field]=paired(values)
                comparisons[key][a+'-'+b]=result
    continuation_effects={m:{field:paired([np.mean([
        index[(g,label,CONTINUATIONS[1],8,m)][field]-index[(g,label,CONTINUATIONS[0],8,m)][field]
        for label in ('handoff','hidden_midpoint')]) for g in groups])
        for field in ('model_return','model_capture','static_geometric_progress')}
        for m in ('G','T','T+Geo')}
    changes={}
    for a,b in (('T','G'),('T+Geo','G')):
        signs={}
        for continuation in CONTINUATIONS:
            for horizon in (8,24):
                signs[(continuation,horizon)]=np.array([next(x['mean'] for x in r['paired_model_differences'] if
                    x['continuation']==continuation and x['horizon']==horizon and x['comparison']==a+'-'+b) for r in rows])
        changes[a+'-'+b]=dict(
            feedback_rank_sign_changes_8=int(np.sum(signs[(CONTINUATIONS[0],8)]*signs[(CONTINUATIONS[1],8)]<0)),
            longer_horizon_sign_changes_constant=int(np.sum(signs[(CONTINUATIONS[0],8)]*signs[(CONTINUATIONS[0],24)]<0)),
            feedback_effect_8=paired([np.mean([(signs[(CONTINUATIONS[1],8)]-signs[(CONTINUATIONS[0],8)])[i]
                for i,r in enumerate(rows) if r['group']==g]) for g in groups]),
            horizon_effect_constant=paired([np.mean([(signs[(CONTINUATIONS[0],24)]-signs[(CONTINUATIONS[0],8)])[i]
                for i,r in enumerate(rows) if r['group']==g]) for g in groups]))
    return dict(groups=len(groups),roots=len(rows),summary=summary,paired_comparisons=comparisons,
        G_missing_from_T=sum(not r['G_in_T_candidates'] for r in rows),G_missing_from_TGeo=sum(not r['G_in_TGeo_candidates'] for r in rows),
        ranking_changes=changes,bootstrap_seed=82120261026,bootstrap_draws=10000,
        continuation_effects_8=continuation_effects,
        root_sampling_uncertainty_separately_recorded=True,
        inference_scope='12 known history blocks; both roots retained; hypothetical model scores, not new capture outcomes')


def interval(value):return f'{value["mean"]:+.4f} [{value["ci95"][0]:+.4f}, {value["ci95"][1]:+.4f}]'


def main():
    repeat=json.loads((OUT/'repeat_statistics.json').read_text())
    exact=json.loads((OUT/'repeat_comparisons.json').read_text())
    data=json.loads((OUT/'root_diagnostics.json').read_text())
    diagnosis=analyze(data);save(OUT/'root_diagnostic_statistics.json',diagnosis)
    lines=['# 再次实验：为什么当前T/T+Geo不如G','',
        '2026-10-26。D保持冻结，未训练网络、未改部署；原闭环结果完整保留。','',
        '## 先确认是否复现','',
        f'重新执行原24公共历史×A/B×三方法144局，真实对手每步重新搜索。逐帧科学结果精确一致{sum(r["exact"] for r in exact["rows"])}/144，耗时另测。',
        '这不是144个新增独立样本，仍是同24组历史；不合并重复局数缩小区间。','',
        '|方法|重复合法捕获|重复重发现|失败惩罚完成成本|决策P95 ms|>200ms|',
        '|---|---:|---:|---:|---:|---:|']
    for m,s in repeat['summary'].items():
        lines.append(f'|{m}|{s["capture"]}/{s["games"]}|{s["rediscovered"]}/{s["games"]}|{s["completion_cost"]:.2f}|{s["latency"]["p95_ms"]:.2f}|{s["over_200ms"]}|')
    lines+=['','[原样复现逐局对比](repeat_comparisons.json)、[重复统计](repeat_statistics.json)、[144局结果与新轨迹入口](repeat_results.json)。',
        '若某局科学字段有差异，在上述JSON中保留首个差异位置；不能靠覆盖旧结果或调种子消除差异。','',
        '## 这次最有支持的解释','',
        '**当前rollout用一段固定根位移代替了G后续逐步绕障的能力，在本批模型对照中低估了G动作的后续价值。** 这是本轮共同根诊断支持的具体机制，不是原样多跑后猜测出的结论。',
        '在相同1024条目标路径、相同根信念和奖励下：固定延续时T−G模型回报差约+0.067；每步重算几何动作后降至约+0.001，区间跨0。T+Geo−G也从约+0.052变为约−0.013。',
        '保留固定延续，仅把8步延长到24步，24个根的回报差没有符号变化；因此本轮证据不支持“只加深搜索就能解决”。',
        '这仍没有证明全部真实捕获差都由该项造成：受控诊断简化了树前缀、使用静态根信念，尚无修改版的真实闭环单机制确认。','',
        '## 代码中可以直接确认的原因线索','',
        '1. **当前T的rollout没有像真实G那样持续重新计算绕障动作。** `one_simulation`在本次树扩展后，后续每步复制根节点计算的`base`位移；不合法时停止。真实G下一决策会重新读取D并按当前位置选几何方向。一次位移不能代表沿墙绕角的多步闭环策略。',
        '2. **树只规划8步。** 根奖励是首次重发现+1、捕获+1，折扣0.95；没有用几何剩余代价评价叶节点。G则每步直接评价带权几何剩余路程和静态可视质量。两者评价目标和时间范围并不等价。',
        f'3. **检查了候选缺失解释，但本批未获支持。** 24个共同隐藏根中，T不含G精确动作的根数为{diagnosis["G_missing_from_T"]}，T+Geo为{diagnosis["G_missing_from_TGeo"]}。本批根都允许选择G动作，不能把差距解释为“搜索没有G这个选项”；其他时刻的动作空间限制仍需另查。',
        '4. **没有未来观测分支。** 这轮T是8步开环UCT探针，不是完整观测条件belief-tree或完整NTE。模拟目标样本不作为追捕动作输入，但树和rollout也没有在未来观测后执行与实际控制器相同的重规划。',
        '5. **预测模型与真实对手不同。** D是地图意图/模式持久化先验，真实对手是3000次全状态搜索。接口核一致不代表模型能预测这种对手。当前没有更换真实对手的因果消融，不能称它是唯一已证实原因。','',
        '上述机制在原代码中就是如此，本轮没有把它们包装为新修复，也没有改写原实验口径。','',
        '## 公共根上的2×2受控诊断','',
        '预登记取12个已见公共历史的A世界，每个取G接管根和隐藏中位根，共24根；不按输赢选根。三方法在同一公共根、相同D和种子上生成动作。',
        '每根1024条目标路径都按64样本权重、条件模式及冻结D转移核采样，三个根动作共用完全相同的目标路径。不得将原真实隐藏状态传入候选或动作选择。',
        '因素一：固定8步/24步评估。因素二：后续重复根G位移/按自身当前位置重新计算G几何动作。后者仍使用根64信念快照，不吸收未来观测。',
        '**这是对固定根动作的共同延续策略评估，不是原UCT完整树值的精确分解，也不是新控制器真实捕获率消融。** 搜索的树前缀被固定根动作加延续策略替代；该简化及静态信念的偏差必须保留。',
        '自身路径在看到采样目标路径前生成，只用公开根信息和自身位置。非法位移统一转停止并计数；奖励保持原首次可见/捕获与0.95折扣。','',
        '|后续策略|视野|根动作|模型重发现率|模型捕获率|模型回报|静态几何进展|',
        '|---|---:|---|---:|---:|---:|---:|']
    for key,methods in diagnosis['summary'].items():
        continuation,horizon=key.rsplit('_',1)
        for method,s in methods.items():
            lines.append(f'|{continuation}|{horizon}|{method}|{100*s["model_rediscovery"]:.2f}%|{100*s["model_capture"]:.2f}%|{s["model_return"]:.4f}|{s["static_geometric_progress"]:.3f}|')
    lines+=['','“模型捕获率”是先验采样世界中的分数，不是对真实搜索对手的捕获率。静态几何进展也是对根信念的评分，不等于实际目标距离。','',
        '|根动作|改为逐步几何延续的8步模型回报增量 [95% CI]|8步静态几何进展增量 [95% CI]|固定延续24步非法提议转停止均数|重算延续24步非法提议转停止均数|',
        '|---|---:|---:|---:|---:|']
    for method,effect in diagnosis['continuation_effects_8'].items():
        fixed=diagnosis['summary'][CONTINUATIONS[0]+'_24'][method]
        feedback=diagnosis['summary'][CONTINUATIONS[1]+'_24'][method]
        lines.append(f'|{method}|{interval(effect["model_return"])}|{interval(effect["static_geometric_progress"])}|{fixed["illegal_displacement_fallbacks"]:.2f}|{feedback["illegal_displacement_fallbacks"]:.2f}|')
    lines+=['','这里是24步整条公共延续路径中“提议不合法、因此合法停止”的次数，不是执行非法转移，也不是原UCT全部rollout的故障次数。固定位移在墙前反复停止，而重算几何动作可以转向，是此模型对照中直接可观察的差异。','',
        '|后续策略/视野|模型回报T−G [逐历史95% CI]|模型回报T+Geo−G [95% CI]|',
        '|---|---:|---:|']
    for key,comparisons in diagnosis['paired_comparisons'].items():
        lines.append(f'|{key}|{interval(comparisons["T-G"]["model_return"])}|{interval(comparisons["T+Geo-G"]["model_return"])}|')
    lines+=['','12个公共历史成组bootstrap10000次，种子82120261026，组内两个根一起保留。区间是已见历史上的机制诊断，未做多重比较调整，不用来认定新算法通过验收。',
        '逐根配对Monte Carlo标准误也在root_diagnostics.json保存，1024条模型路径不能冒充1024个真实独立场景。','',
        '|比较|改延续策略时排名差变号根数（8步）|改8→24步时排名差变号根数（原延续）|延续策略对回报差的影响 [95% CI]|视野对回报差的影响 [95% CI]|',
        '|---|---:|---:|---:|---:|']
    for key,s in diagnosis['ranking_changes'].items():
        lines.append(f'|{key}|{s["feedback_rank_sign_changes_8"]}/24|{s["longer_horizon_sign_changes_constant"]}/24|{interval(s["feedback_effect_8"])}|{interval(s["horizon_effect_constant"])}|')
    lines+=['','排名变号只是描述性统计，接近0的变化可能有Monte Carlo误差，不能仅凭变号数量宣称因果效果。',
        '改变公共几何延续策略或评估视野后，模型内动作相对评分如何变化，应以上述配对差和完整根记录判断。即使模拟评分改善，也不能推断真实闭环必然改善。','',
        '![共同根模型诊断](figures/root_diagnostic.png)','',
        '## 能回答与仍不能回答的部分','',
        '- 原样复现检查相同种子是否可重现、有无运行或保存差异；它不能排除原种子恰好不利，也不是跨随机流鲁棒性验证。相同结果不是新增统计显著性。',
        '- 原闭环数据显示主要损失发生在首次重发现之前；T比G的成组条件重发现步数差约+10步，而非可见后几何追踪明显变差。失败包含未重发现和重发现过晚。',
        '- 相同D与合法性审计排除了“不同估计器”“非法rollout提前截止”作为本批主要解释；全部隐藏搜索完成1000次，也不能归为200ms被截断。',
        '- 代码与共同根诊断检验了粗候选、固定方向rollout、短视野及目标函数差异，但没有把这些机制分别改成真实闭环新版本并在全新场景确认。因此不能声称已唯一证明其中某一项解释了全部捕获差。',
        '- 最准确的表述是：当前简化搜索的规划近似与强G的实际闭环策略存在实质差异，尚未证明搜索价值；该结果不是MCTS或NTE一般无效的证明。',
        '- 不因本轮结果直接添加网络、价值网络或PPO。后续如做真实闭环单机制消融，应另登记版本，D继续冻结，保留同预算强G，并使用未消耗的新确认集。','',
        '## 数据与复核','',
        '- [结果前登记](../../code/L45_SEARCH_DIAGNOSIS_REPEAT_20261026.md)、[源码与原数据散列](experiment.json)、[复现完整性](repeat_integrity.json)。',
        '- [24根请求](root_diagnostic_requests.json)、[逐根2×2原始结果](root_diagnostics.json)、[成组诊断统计](root_diagnostic_statistics.json)。',
        '- [原闭环报告](../l45_closed_loop_D_20261026/REPORT.md)、[原源码](../../code/solvers/frozen_d_search_interface.py)、[实际重复执行器](../../code/repeat_frozen_d_closed_loop.py)、[诊断执行器](../../code/diagnose_frozen_d_root_rollout.py)。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    x=np.arange(4);keys=list(diagnosis['summary'])
    for j,method in enumerate(('G','T','T+Geo')):
        axes[0].bar(x+(j-1)*.25,[diagnosis['summary'][k][method]['model_return'] for k in keys],.25,label=method+' root action')
    axes[0].set(xticks=x,xticklabels=['Fixed/8','Fixed/24','Replan/8','Replan/24'],ylabel='Hypothetical discounted event return',title='Same weighted model paths, different continuation')
    axes[0].legend(fontsize=8)
    for j,method in enumerate(('G','T','T+Geo')):
        axes[1].bar(x+(j-1)*.25,[diagnosis['summary'][k][method]['static_geometric_progress'] for k in keys],.25,label=method)
    axes[1].set(xticks=x,xticklabels=['Fixed/8','Fixed/24','Replan/8','Replan/24'],ylabel='Static-belief geometric progress (steps)',title='Geometry score differs from capture/visibility reward')
    axes[1].axhline(0,color='.4',lw=.7)
    (OUT/'figures').mkdir(exist_ok=True)
    fig.savefig(OUT/'figures/root_diagnostic.png',dpi=170);fig.savefig(OUT/'figures/root_diagnostic.pdf');plt.close(fig)
    save(OUT/'decision.json',dict(repeat_exact=exact['all_exact'],new_independent_confirmation=False,
        real_closed_loop_ablation_run=False,neural_training=False,deployment_changed=False,
        conclusion='Repeated simplified search; diagnostic model assumptions tested, no adoption or unique causal attribution'))
    save(OUT/'status.json',dict(stage='complete',repeated_games=144,diagnostic_roots=24,pid=None,training=False,deployment_changed=False))


if __name__=='__main__':main()
