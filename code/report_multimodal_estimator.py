"""Render the frozen estimator diagnosis/validation without rerunning it."""
from pathlib import Path
import json

from audit_l45_rollout_reliability import save

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'current/l45_multimodal_estimator_20261025'


def main():
    dev=json.loads((OUT/'development_results.json').read_text())
    confirm=json.loads((OUT/'confirmation_results.json').read_text())
    diagnosis=json.loads((OUT/'diagnosis_summary.json').read_text())
    regions=json.loads((OUT/'region_audit.json').read_text())['summary']
    families=json.loads((OUT/'family_results.json').read_text())['results']
    timing=json.loads((OUT/'timing_and_pooling.json').read_text())
    labels={'particle':'原64粒子','systematic':'仅系统重采样','diffuse':'确定性扩散格','persistent':'持续运动多模态加权格','feasible':'均匀可行集合（诊断）'}
    p=confirm['summary']['persistent'];f=confirm['summary']['feasible']
    decision=dict(estimator_accepted=confirm['passed'],frozen_shared_deployment_estimator=None,
        next_stage='public_motion_prior_and_geometric_sample_representation_redesign',
        search_comparison_started=False,network_training_started=False,deployment_changed=False,
        gate_checks=confirm['checks'],reason='Candidate improves over old particles but fails fixed sample coverage and extra-density-value gates against matched feasible support; do not adopt or tune on confirmation histories.')
    save(OUT/'decision.json',decision)
    parts=['# 正常遮挡期间的多模态目标估计：定位、开发与未见历史确认\n\n',
        '**结论：本轮完成问题定位与估计器验证，但新候选未通过验收；不替换部署、不进入G/T/T+Geo、不训练网络。** 原问题主要涉及运动先验与有限样本支持，随机重采样加重退化；没有证据表明合法真值被历史约束错误排除。新候选保留多分支并改善旧粒子，却尚不能提供足够可靠、比均匀可行集合更有信息的位置分布。\n\n',
        '本轮所有估计接口只读公开地图、已知运动/感知参数和历史传感器观测。真值、A/B身份、未来路线及完整环境状态只进入数据构造/独立评分，不进入估计器。没有调用MCTS决策、对手搜索、教师标签、优化器或神经训练。\n\n',
        '## 1. 数据、冻结与样本单位\n\n',
        '- 开发：24组已查看公共历史、原G的96条保存轨迹，共3292个已记录观测帧；原记录中的1次执行失败保留，未补造其失败后的观测。\n',
        '- 确认：24组全新公共历史、A/B两个世界、上下两条合法路线，共96条脚本观测轨迹。墙长、观测历史长度与原集不同，且与可恢复历史公开地图/历史摘要无重叠。全部先独立几何验收，再冻结。\n',
        '- 固定格距0.5米、运动核σ=0.35、模式保持率0.9、64导出样本；确认前锁定代码/参数，未依据确认结果改参数或追加场景。确认数据仍属于三类既定拓扑和脚本运动，不代表新拓扑、真实3000次搜索对手或闭环捕获泛化。\n',
        '- 表中隐藏覆盖等为先在每条历史内平均、再按公共历史组等权；10000次配对bootstrap保留组内A/B和两路线。帧不是独立样本，接管计数中的96也不是96张独立地图。\n\n',
        '## 2. 为什么正常粒子漏掉目标\n\n',
        '|因素|独立证据|能够支持的判断|\n|---|---|---|\n']
    d=diagnosis['per_method']
    parts += [f'|运动先验|真实合法隐藏段的平均轴速度约{diagnosis["axis_speed_mean"]:.3f}，连续方向余弦约{diagnosis["directional_cosine_mean"]:.3f}；位移约为无约束iid均匀模型尺度的{diagnosis["iid_displacement_z_range"][0]:.2f}—{diagnosis["iid_displacement_z_range"][1]:.2f}倍|独立零均值位移使持续快速运动落在低概率尾部；尺度比不是正态显著性或唯一因果证明|\n',
        f'|有限样本与退化|原接管状态唯一祖先中位数{d["particle"]["unique_ancestors_median"]:.0f}/64，唯一粒子{d["particle"]["unique_particles_median"]:.0f}/64；末次预测仍有约{d["particle"]["survivors_median"]:.1f}/64合法幸存|支持可在不耗尽时偏离真值，故仅修复恢复函数不会改善正常覆盖|\n',
        f'|随机重采样|systematic将唯一祖先中位数提高到{d["systematic"]["unique_ancestors_median"]:.0f}、唯一粒子到{d["systematic"]["unique_particles_median"]:.0f}，但接管仍0/96覆盖|重采样加重退化，但仅换重采样不足以解决问题|\n',
        f'|历史约束|48条真实隐藏前缀共{diagnosis["independent_true_transitions"]}次转移独立核验无违反；相同公开历史可行支持的直径下界为{diagnosis["public_ambiguity_diameter_range"][0]:.2f}—{diagnosis["public_ambiguity_diameter_range"][1]:.2f}米|未见误用历史排除真值；空观测允许多个相距很远的合法解释，不能唯一定位目标|\n\n',
        f'在相同开发接管状态，确定性扩散格给真值1米内的平均质量仅{100*d["diffuse"]["handoff_mass1m"]:.4f}%；持续运动混合提高到{100*d["persistent"]["handoff_mass1m"]:.2f}%。这区分“可行点存在”与“运动先验给它足够质量”。扩散格与原连续粒子在离散化/边界条件上不完全相同，不能把两者差值全部分配给有限粒子这一项。\n\n',
        '“0/96”只指接管时。旧粒子在开发隐藏帧仍有455/2661的池化覆盖17.10%，按公共历史等权为13.28%；它不是整个过程始终零覆盖。两种口径都保留，不混用。\n\n',
        '## 3. 实现了什么、如何防止靠扩大范围取巧\n\n',
        '新隔离模块`solvers/multimodal_estimator.py`维护位置×九种固定运动模式的加权分布，逐步联合推进并应用每一次空观测；不进行会永久删除稀有模式的全局随机重采样。可见测量重新出现后只使用该位置。\n\n',
        '所有格方法使用同一套合法节点、精确检查过的直线边、全部历史约束和数值保护余量；完整可行范围一致。持续模型保留模式质量，导出仍为64样本，并以模式质量/样本数赋权，弱模式没有被强行升成等概率。完整分布、64样本和名义90%高密度区域分别评分。\n\n',
        '代价也不同：格方法维护辅助稠密权重与邻接结构，64只是导出数量，不能声称与64粒子等内存/算力。每个典型图约1764节点，转移表上界约6.3MiB量级，另有权重和路径父指针；首次建图/历史吸收与逐帧开销分开报告。尚未接入实际搜索或部署。\n\n',
        '## 4. 全新历史确认结果\n\n',
        '|方法|接管1米覆盖/96|隐藏帧覆盖（历史等权）|真值1米内平均质量|核NLL↓|90%区域名义面积↓|90%区域经验覆盖|逐帧P95 ms|\n|---|---:|---:|---:|---:|---:|---:|---:|\n']
    for m,x in confirm['summary'].items():
        parts.append(f'|{labels[m]}|{round(x["handoff_coverage"]*96)}/96|{100*x["sample_coverage"]:.2f}%|{100*x["mass_1m"]:.2f}%|{x["nll"]:.3f}|{x["area90"]:.2f}|{100*x["coverage90"]:.2f}%|{x["update_p95_ms"]:.3f}|\n')
    parts += ['\n核NLL用相同σ=0.5米的高斯核在真值处评分，密度下限1e-12，越低越好；不是训练loss，也不证明概率已校准。名义面积按共同0.5米单元聚合质量后取90%集合，单位为名义平方米；锚定点/边界不构成连续面积或连续覆盖保证。\n\n',
        '|新候选减对照|隐藏覆盖差 pp [95%CI]|核NLL差 [95%CI]|\n|---|---|---|\n']
    for m in ('particle','systematic','diffuse','feasible'):
        cov=confirm['comparisons']['persistent-'+m]['sample_coverage'];nll=confirm['comparisons']['persistent-'+m]['nll']
        parts.append(f'|{labels[m]}|{100*cov["mean"]:+.2f} [{100*cov["ci95"][0]:+.2f},{100*cov["ci95"][1]:+.2f}]|{nll["mean"]:+.3f} [{nll["ci95"][0]:+.3f},{nll["ci95"][1]:+.3f}]|\n')
    parts += ['\n新候选相对旧粒子与确定性扩散有明确改善，但相对均匀可行集合的核NLL区间跨零。虽然90%区域更集中，经验覆盖也下降，不能把较小面积本身当作更好的估计。\n\n',
        '三个预设拓扑分别如下；每类8组公共历史、32条轨迹。这是完整分项披露，不据此重选模型。\n\n',
        '|拓扑|新候选接管覆盖|新候选隐藏覆盖|新候选核NLL|均匀集合核NLL|\n|---|---:|---:|---:|---:|\n']
    for family,values in families.items():
        x=values['persistent'];u=values['feasible']
        parts.append(f'|{family}|{round(x["handoff_coverage"]*32)}/32|{100*x["sample_coverage"]:.2f}%|{x["nll"]:.3f}|{u["nll"]:.3f}|\n')
    parts += ['\n墙角是突出不足：接管只覆盖8/32、隐藏覆盖约31%。这与固定方向保持模型没有表达绕过翼墙后的转向/通路意图相符，但没有额外因果对照证明这一项独自造成全部误差。长墙和多通路的较好结果不能掩盖墙角失败；不事后删掉这个拓扑。\n\n',
        '## 5. 多通路、合法性与计算代价\n\n',
        '|集合|完整可行区域模式|导出保留的≥1%质量区域|最大未导出区域总质量|\n|---|---|---:|---:|\n']
    for split,label in [('development','开发'),('confirmation','确认')]:
        x=regions[split]
        parts.append(f'|{label}|每组{x["min_regions"]}—{x["max_regions"]}个，完整加权支持与可行集合相同={x["same_full_region_support"]}|{x["important_regions_retained"]}/{x["important_regions_total"]}|{100*x["maximum_missing_mass"]:.2f}%|\n')
    t=timing['confirmation']['persistent']
    parts += ['\n区域由公开矩形两侧/端部的联合半空间定义，不使用真实路线身份。完整格保留了这些合法分支，但64样本的速度模式分层不能保证代表全部重要几何分支；这也是本候选仍不足的具体证据。\n\n',
        f'确认集每种方法独立核验约{p["independently_verified_transitions"]:,}个导出样本的祖先转移，所有方法无证书或执行失败。新模型正常更新均值{p["update_mean_ms"]:.3f}ms、P95={p["update_p95_ms"]:.3f}ms；此前公共历史吸收/首次建图均值{t["initial_history_mean_ms"]:.2f}ms、P95={t["initial_history_p95_ms"]:.2f}ms，另行计费。固定轨迹回放中的计时不能保证MCTS闭环50ms截止。\n\n',
        '## 6. 验收决定\n\n',
        '|预先冻结条件|结果|\n|---|---|\n']
    for k,v in confirm['checks'].items():parts.append(f'|{k}|{"通过" if v else "未通过"}|\n')
    parts += ['\n接管覆盖72.92%低于75%，隐藏覆盖58.65%低于70%，且没有确认相对均匀可行集合的额外核密度价值。因此**未通过**，不因改善幅度较大或只差2个接管样本而降低标准。确认数据不再是未见数据，不能用它继续调参数后重复宣称独立确认。\n\n',
        '当前可以排除“只换重采样就能解决”与“只在完全耗尽时恢复就足够”两条简单解释。仍未解决的是运动模式概率/转向的预测、几何分支在64样本中的代表性，以及无正观测时本来就存在的不可辨识性。下一版本如继续，应先登记从可见运动与公开几何构造运动假设、按几何分支保质量导出的单项改进，并使用新的确认历史；不能继续加PPO/价值头或把本轮改善算成搜索贡献。本轮不启动这些下一项实验。\n\n',
        '## 7. 图、数据与复现\n\n',
        '![开发固定接管状态](figures/estimator_handoff_comparison.png)\n\n',
        '每类图例按原开发清单首次出现顺序选择，未按效果挑图；红星只用于事后评分。蓝色为加权支持、绿色为64导出样本。\n\n',
        '- [冻结方案](../../code/L45_MULTIMODAL_ESTIMATION.md) · [实现](../../code/solvers/multimodal_estimator.py) · [测试](../../code/tests/test_multimodal_estimator.py)\n',
        '- [开发统计](development_results.json) · [确认统计](confirmation_results.json) · [验收决定](decision.json)\n',
        '- [原因诊断](diagnosis_summary.json) · [逐前缀诊断](diagnosis_prefix_rows.json) · [真实运动独立审计](diagnosis_true_motion_audit.json)\n',
        '- [区域审计](region_audit.json) · [预设拓扑分项](family_results.json) · [初始化/池化指标](timing_and_pooling.json) · [冻结参数/源码](experiment.json)\n',
        '- [新历史清单](bank/manifest.json) · [独立几何验收](bank_acceptance.json) · [确认集摘要](bank_frozen.json)\n',
        '- `development/*.json`、`confirmation/*.json`含每条历史的逐帧指标与失败记录；PNG与PDF可直接导出。\n\n',
        '复查已完成结果使用本报告/JSON；运行入口拒绝覆盖完成的阶段。主科学统计按24组公共历史配对，而不是按独立帧计算。\n']
    (OUT/'REPORT.md').write_text(''.join(parts))
    save(OUT/'status.json',dict(stage='estimator_validation_rejected',terminal=True,development_cases=96,confirmation_cases=96,
        pid=None,estimator_accepted=False,network_training=False,search_comparison=False,deployment_changed=False))
    print('Estimator diagnosis and validation report written; candidate rejected by frozen gates.')


if __name__=='__main__':main()
