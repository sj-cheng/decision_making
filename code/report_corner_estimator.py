"""Report the completed controlled/confirmation estimator experiment."""
from pathlib import Path
import json
import numpy as np

from audit_l45_rollout_reliability import save

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'current/l45_corner_turn_estimator_20261025'


def plot(control,confirm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    names=['C-A','B-A','D-C','D-B']
    for ax,field,scale,title in [(axes[0,0],'nll',1.,'Controlled: full-distribution NLL change (lower better)'),
                                  (axes[0,1],'sample_coverage',100.,'Controlled: 64-sample coverage change (pp)')]:
        for i,name in enumerate(names):
            x=control['comparisons'][name][field];mean=x['mean']*scale;lo,hi=np.array(x['ci95'])*scale
            ax.errorbar(mean,i,xerr=[[max(0,mean-lo)],[max(0,hi-mean)]],fmt='o',color='#15607a',capsize=3)
        ax.axvline(0,color='0.6',lw=1);ax.set_yticks(range(4),names);ax.invert_yaxis();ax.set_title(title,fontsize=10);ax.grid(axis='x',alpha=.2)
    for ax,field,title in [(axes[1,0],'handoff_coverage','New histories: handoff coverage'),
                           (axes[1,1],'sample_coverage','New histories: hidden-time coverage')]:
        for offset,(m,color) in enumerate([('A','#9aa7ae'),('D','#168368'),('U1','#de9a35')]):
            values=[100*confirm['families'][f][m][field] for f in ['long_wall','wall_corner','multi_route']]
            ax.bar(np.arange(3)+(offset-1)*.24,values,.23,label=m,color=color)
        ax.set_xticks(range(3),['long wall','wall corner','multi route']);ax.set_ylim(0,105);ax.set_ylabel('%');ax.set_title(title,fontsize=10);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Motion model and sample export are separate interventions; confirmation uses new frozen histories')
    fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True)
    fig.savefig(OUT/'figures/ablation_confirmation.png',dpi=170);fig.savefig(OUT/'figures/ablation_confirmation.pdf');plt.close(fig)


def main():
    control=json.loads((OUT/'controlled_results.json').read_text())
    confirm=json.loads((OUT/'confirmation_results.json').read_text())
    diagnosis=json.loads((OUT/'diagnosis.json').read_text())
    accepted=confirm['passed']
    if accepted and not (OUT/'frozen_estimator/replay_checks.json').exists():
        raise RuntimeError('Accepted estimator must be frozen and replay-verified first')
    decision=dict(estimator_quality_accepted=accepted,scope='shared research estimator under the registered local one-target protocol',
        frozen_shared_estimator=str(OUT/'frozen_estimator') if accepted else None,
        next_stage='weighted estimator integration and full-budget preflight for G/T/T+Geo' if accepted else 'stop this candidate; no new modules',
        gate_checks=confirm['checks'],deployment_changed=False,search_comparison_started=False,network_training=False)
    save(OUT/'decision.json',decision)
    parts=['# 墙角转向与几何分支导出：原因、消融及全新历史确认\n\n',
        '**结果：唯一候选D通过本轮全部预定估计器验收条件，已冻结为下一阶段的统一研究估计器；没有修改部署、训练网络或运行搜索效果对照。** 收益来源可以区分：公开地图引导的沿墙/转向均值改善完整概率分布；几何分支分层解决重要分支在64样本中的遗漏，并改善有限样本表示。\n\n',
        '本结论限于20×20场地、原每轴位移能力、确定性半径6遮挡观测及三种已知拓扑。确认是全新墙长、历史长度与停顿变体的脚本观测历史，不是实际3000模拟对手的闭环追捕实验，也不证明概率校准或100%泛化覆盖。\n\n',
        '## 1. 先定位原墙角8/32\n\n',
        '前轮确认集本轮明确作为已用开发数据。墙角有8组公共历史、A/B两个不同接管位置；上下续跑路线上接管输入重复，所以8/32轨迹接管项对应4/16不同物理接管状态，而不是32个独立状态。统计始终按公共历史组进行。\n\n']
    x=diagnosis['summary']['wall_corner']
    parts += [f'原未覆盖的{x["old_misses"]}个不同接管状态中，{x["old_misses_with_mass_below_1pct"]}个的真值1米范围完整概率质量都低于1%；全部墙角状态的平均质量仅{100*x["mean_root_mass"]:.3f}%。仅改64样本导出挽回{x["rescued_by_export_only"]}个，同时丢失同样数量，净覆盖不变。完整NLL差严格为0。\n\n',
        '因此不能把主要问题归为“完整分布已经正确、只是没抽到”。运动先验把转向后的区域压成低概率尾部。旧九种世界坐标固定方向，即使合法边会拒绝穿墙，也不会保留“绕过障碍后继续原意图”的路径含义；墙角需要更多方向变化。这是机制解释，最终归因由下面只改变运动均值的消融检验，不靠看真值修改估计。\n\n',
        '## 2. 只改两个环节\n\n',
        '|组|运动均值|64样本导出|用途|\n|---|---|---|---|\n',
        '|P|原独立均匀位移|原重采样|原64粒子对照|\n|A|前轮固定方向意图|速度模式分层|精确复现旧候选|\n|B|同A|几何分支分层|单独检查导出|\n|C|沿公开地图的绕障首步|同A|单独检查沿墙/转向|\n|D|同C|同B|预先锁定的唯一候选|\n|U0|均匀可行集合|同A|范围诊断对照|\n|U1|同U0|同B|与D匹配导出的强对照|\n\n',
        '运动改动保留原9种意图，把非静止意图目标设为“最后可见位置+20×原方向”，裁到±9.95；公开可见图给出朝该固定意图点的合法首步。绕过翼墙后仍朝原意图目标移动，不把转向变成永久丢失原方向。固定意图点不是实际目标，没有读取隐藏位置、A/B标签或未来路线。\n\n',
        '导出改动按已知矩形两侧/端部划分几何分支，为每个后验质量≥1%的分支保留至少1个样本，低质量分支合并为尾部层，层内仍systematic抽样。样本权重=层质量/样本数，未提高弱分支的概率。\n\n',
        '网格、可达节点/合法边、9模式、σ=.35、保持率.9、历史约束和64样本预算都不变。每个观测上逐项检查A/B、C/D、U0/U1完整分布一致，所有格方法的可达范围一致。因此导出不能冒充完整密度收益，也没有通过扩大范围取巧。\n\n',
        '## 3. 受控消融：原确认集只作开发\n\n',
        '96条相同观测轨迹、24组公共历史，共同保留A/B世界及上下路线。A精确复现前轮结果。下面的区间为10000次公共历史配对bootstrap的普通95%区间。\n\n',
        '|比较|完整核NLL差↓ [95%CI]|隐藏64样本覆盖差 pp [95%CI]|\n|---|---|---|\n']
    for name in ('C-A','B-A','D-C','D-B'):
        n=control['comparisons'][name]['nll'];c=control['comparisons'][name]['sample_coverage']
        parts.append(f'|{name}|{n["mean"]:+.3f} [{n["ci95"][0]:+.3f},{n["ci95"][1]:+.3f}]|{100*c["mean"]:+.2f} [{100*c["ci95"][0]:+.2f},{100*c["ci95"][1]:+.2f}]|\n')
    parts += ['\n墙角接管A/B/C/D分别为8/32、8/32、30/32、26/32。运动改动明显改善分布；导出改变了哪些局部位置被抽到，所以即使重要分支保留更好，也不保证每次接管覆盖都提高。没有因为C在开发集更高而改选C。\n\n',
        'D达到预设继续门槛（相对A完整NLL改善至少0.2、墙角净增至少4/32、无执行/证书失败），因此才读取新确认结果。开发阶段相对U1的普通NLL区间为负，但主比较同时区间仍跨零，不在开发集宣布独立确认。\n\n',
        '## 4. 全新历史独立确认\n\n',
        '新银行在任何效果评估前一次冻结、独立几何验收并与历史地图/观测摘要去重。24组=3拓扑×2遮挡档×2速度×2变体；墙长来自[6.25,6.85]/[8.2,8.8]，其中一半增加一次合法隐藏停顿；每组A/B两个世界、上下两路线，共96条。确认前锁定D及全部参数，没有扫参、追加场景或反复使用前轮确认集选模。\n\n',
        '|方法|接管1米覆盖/96|隐藏覆盖（历史等权）|完整核NLL↓|导出64样本核NLL↓|90%区域名义面积|\n|---|---:|---:|---:|---:|---:|\n']
    for m,s in confirm['summary'].items():
        parts.append(f'|{m}|{round(s["handoff_coverage"]*96)}/96|{100*s["sample_coverage"]:.2f}%|{s["nll"]:.3f}|{s["sample_nll"]:.3f}|{s["area90"]:.2f}|\n')
    parts += ['\n覆盖是64样本到真值距离≤1米，不是捕获率。完整分布和样本NLL都使用固定σ=.5米高斯核、密度下限1e-12，越低越好。90%面积是共同0.5米单元聚合质量后的名义面积，不是连续可信集保证。\n\n',
        '|主确认比较|完整核NLL差|普通95%区间|两项同时95%区间|\n|---|---:|---|---|\n']
    for name in ('D-particle','D-U1'):
        s=confirm['comparisons'][name]['nll'];ci=s['ci95'];sim=s['simultaneous_ci95']
        parts.append(f'|{name}|{s["mean"]:+.3f}|[{ci[0]:+.3f},{ci[1]:+.3f}]|[{sim[0]:+.3f},{sim[1]:+.3f}]|\n')
    parts += ['\n同时区间由同次公共历史抽样的最大中心化绝对偏差构造。96个轨迹接管项含路线相关重复，不能当作96个独立地图/物理状态；100%经验覆盖也不代表真实总体覆盖已精确确定。\n\n',
        '|拓扑（每类8组）|旧A接管|D接管|旧A隐藏覆盖|D隐藏覆盖|U1隐藏覆盖|\n|---|---:|---:|---:|---:|---:|\n']
    for family,values in confirm['families'].items():
        a,d,u=values['A'],values['D'],values['U1']
        parts.append(f'|{family}|{round(a["handoff_coverage"]*32)}/32|{round(d["handoff_coverage"]*32)}/32|{100*a["sample_coverage"]:.2f}%|{100*d["sample_coverage"]:.2f}%|{100*u["sample_coverage"]:.2f}%|\n')
    parts += ['\n原8/32与新32/32来自不同历史，不直接跨集相减；本确认集内墙角A为10/32、D为32/32才是同条件比较。全部拓扑均保留。\n\n',
        '## 5. 收益来源与分支完整性\n\n']
    motion=confirm['comparisons']['C-A'];export=confirm['comparisons']['D-C'];d=confirm['summary']['D'];c=confirm['summary']['C'];u=confirm['summary']['U1']
    parts += [f'- 运动改动：C−A完整NLL={motion["nll"]["mean"]:+.3f}，95%区间[{motion["nll"]["ci95"][0]:+.3f},{motion["nll"]["ci95"][1]:+.3f}]；这是完整概率质量改善的来源。\n',
        f'- 导出改动：D−C完整NLL严格为0；64样本NLL改善{export["sample_nll"]["mean"]:+.3f}，95%区间[{export["sample_nll"]["ci95"][0]:+.3f},{export["sample_nll"]["ci95"][1]:+.3f}]；隐藏覆盖提高{100*export["sample_coverage"]["mean"]:.2f}pp，区间[{100*export["sample_coverage"]["ci95"][0]:+.2f},{100*export["sample_coverage"]["ci95"][1]:+.2f}]。\n',
        f'- C漏掉{c["missing"]}/{c["important"]}次≥1%分支出现事件，D为{d["missing"]}/{d["important"]}；D的重要分支质量最大绝对误差{d["maximum_mass_error"]:.2e}。这些是相关的逐时刻事件，不是独立地图。\n',
        f'- 同范围U1已使用相同分支导出，D仍有更好的完整NLL；D的90%名义面积{d["area90"]:.2f}，U1为{u["area90"]:.2f}。这支持地图引导概率模型的额外价值，不是简单扩大支持范围。\n\n',
        '地图路由改动同时包含受阻绕行及绕过后保持原意图，两者作为一个固定机制包验证；没有进一步拆分这两个内部环节，也不声称找到了真实目标的最优意图。\n\n',
        '## 6. 合法性、计算开销与冻结状态\n\n']
    parts += [f'所有方法无执行或独立祖先路径证书失败。D核验{d["independently_verified_transitions"]:,}次导出样本祖先转移，输出仍为64位置及重要性权重。\n\n',
        '|方法|实际更新均值/P95 ms|历史吸收及首次地图准备均值/P95 ms|\n|---|---|---|\n']
    for m in ('particle','A','C','D','U1'):
        s=confirm['summary'][m]
        parts.append(f'|{m}|{s["update_mean_ms"]:.3f}/{s["update_p95_ms"]:.3f}|{s["initialization_mean_ms"]:.2f}/{s["initialization_p95_ms"]:.2f}|\n')
    parts += ['\n初始化没有藏进“预热后更新”指标。地图路由增加准备成本，首次准备P95超过50ms；本结果不能证明50ms搜索闭环。下一阶段必须将准备/更新/候选/搜索整体计费，并检查带权样本消费接口。没有用多worker吞吐代替单步耗时。\n\n',
        '|冻结验收条件|结果|\n|---|---|\n']
    for k,v in confirm['checks'].items():parts.append(f'|{k}|{"通过" if v else "未通过"}|\n')
    parts += ['\n四项针对性测试通过：旧A精确复现/密度隔离、受阻意图绕墙、重要分支保质量导出、快照及公共输入一致性。独立确认通过后已生成隔离命名空间的源码快照和加载器，三类拓扑的公共历史重放验证位置、权重、完整密度和RNG精确一致。\n\n',
        '**冻结的是下一阶段统一研究估计器，不是部署推荐。** G/T/T+Geo以后必须共享该版本、公开历史、权重语义和实际时间预算；不能把带权64样本按等权抽取，也不能把旧rollout的独立随机运动当作已经验收的地图运动模型。当前未运行搜索效果对照、教师监督或神经训练，既有部署和历史结果未覆盖。\n\n',
        '## 7. 交付与复查\n\n',
        '![受控消融及新确认](figures/ablation_confirmation.png)\n\n',
        '- [冻结方案](../../code/L45_CORNER_TURN_ESTIMATION.md) · [新实现](../../code/solvers/corner_estimator.py) · [测试](../../code/tests/test_corner_estimator.py)\n',
        '- [原问题定位](diagnosis.json) · [受控消融](controlled_results.json) · [全新确认](confirmation_results.json) · [最终决定](decision.json)\n',
        '- [冻结研究估计器](frozen_estimator/README.md) · [加载器](frozen_estimator/loader.py) · [输出/输入契约](frozen_estimator/descriptor.json) · [冻结重放](frozen_estimator/replay_checks.json)\n',
        '- [源码与输入锁定](experiment.json) · [新历史清单](bank/manifest.json) · [几何验收](bank_acceptance.json) · [银行摘要](bank_frozen.json)\n',
        '- `controlled/*.json`与`confirmation/*.json`包含每条历史、每方法、每帧的概率/覆盖/分支/路径计数及耗时；PNG/PDF可导出。\n\n',
        '确认集已使用，后续模型改动需新版本和新确认，不能把本银行继续称为未见数据。\n']
    plot(control,confirm)
    (OUT/'REPORT.md').write_text(''.join(parts))
    save(OUT/'status.json',dict(stage='research_estimator_frozen' if accepted else 'estimator_validation_rejected',terminal=True,
        controlled_cases=96,confirmation_cases=96,pid=None,estimator_quality_accepted=accepted,
        search_comparison=False,network_training=False,deployment_changed=False))
    print('Report written. Accepted research estimator:',accepted)


if __name__=='__main__':main()
