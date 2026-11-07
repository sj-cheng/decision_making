"""Render completed interface audit; does not run or tune a controller."""
from pathlib import Path
import json
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT/'current/l45_weighted_search_interface_20261026'
OUT=ROOT/'current/l45_weighted_search_interface_v2_20261026'


def main():
    d=json.loads((OUT/'results.json').read_text())
    old=json.loads((OLD/'results.json').read_text())
    external=json.loads((OUT/'external_timing.json').read_text())
    rows=[r for p in (OUT/'groups').glob('*.json') for r in json.loads(p.read_text())['decisions']]
    bad=[sum(r['independent_errors'][i] for r in rows) for i in range(4)]
    oldrows=[r for p in (OLD/'groups').glob('*.json') for r in json.loads(p.read_text())['decisions']]
    oldbad=[sum(r['independent_errors'][i] for r in oldrows) for i in range(4)]
    probabilities=d['probabilities']
    methods=('G','T','T+Geo')
    lines=['# 冻结D的带权搜索接口与50ms预算审计','',
        '2026-10-26，Asia/Shanghai。**样本内带权接口、运动核、合法性与重放检查通过；完整50ms预算失败，因此未开展G/T/T+Geo效果对照或独立效果确认。**',
        'D继续作为已认可的统一研究估计器，源文件、参数、64样本导出及概率约束完全不变。本轮不训练网络、不修改部署。','',
        '## 结论与阶段边界','',
        f'- 使用已经看过的24组公共历史，轮换三方法，共{d["decision_rows"]}次观测驱动决策；没有将A/B隐藏世界复制为独立样本。',
        f'- v2独立检查{d["simulated_transitions"]:,}条模拟联合转移，四类错误（自身运动/目标运动/传感器/捕获）为{bad}；检查{sum(p["graph_edges"] for p in probabilities):,}条目标图边，非法为0。',
        '- 每方法首次遮挡24/24次超过50ms；即使完全不运行树搜索，G也超时。瓶颈首先在冻结D按需构图，而非树模拟次数。',
        '- 目标重新可见会重置图，随后再遮挡仍发生重建；本轮公共历史往返复现了该开销，不能只称为启动一次的成本。',
        '- 本轮是工程门槛检查，没有真实闭环对手/捕获效果比较。因此尚不能判断T−G或T+Geo−T的额外收益，更不能给出有效教师或NTE训练结论。','',
        '## 预登记与输入隔离','',
        '研究方案在结果生成前登记于[L45_WEIGHTED_SEARCH_INTERFACE.md](../../code/L45_WEIGHTED_SEARCH_INTERFACE.md)，源文件和公共输入散列见[experiment.json](experiment.json)。',
        '采用前阶段24组确认公共历史作为已见工程开发输入；只读取controller_inputs，未读取environment中的隐藏状态或后续路线。独立效果确认集尚未生成或消耗。',
        '控制器构造函数仅接受公开地图、限速、方法、种子和预算，act仅接受Observation与可选固定模拟数；不存在环境对象或隐藏状态入口。',
        '三方法的估计随机流与搜索流独立。同一观测历史下，64位置、重要性权重和估计器RNG逐位一致。',
        '同输入的两个外部世界标签构造以及禁止hidden_state参数的检查验证接口隔离；外部标签没有传入控制器。这不是用真实隐藏世界回测收益。','',
        '## 权重与运动核','',
        '1. 根位置严格按64样本的输出权重抽取；重复位置保留其合计概率。每个位置的潜在模式从该点D后验的条件分布抽取。三方法获得相同辅助模式信息。',
        '2. 先以0.9保持模式、以0.1均匀重置到9个模式，再按冻结图中该模式的归一化合法边概率运动。没有沿用旧R2的均匀位移或逃离模拟追捕者的模型。',
        '3. 分别对完整后验和64样本诱导先验，解析推进后施加相同负观测约束，与冻结D propagate逐点核对。',
        f'4. 最大解析误差{max(p["filter_max_error"] for p in probabilities):.3g}。24组每组200000次根/模式/下一步抽样，均通过预登记的逐格6倍二项标准差+2/N检查。',
        f'5. 根样本的非均匀权重与等权分布TV为{min(p["root_weights_uniform_tv"] for p in probabilities):.4f}–{max(p["root_weights_uniform_tv"] for p in probabilities):.4f}；另用单样本权重0.9的合成例验证不会被转换为等权。',
        f'6. 64位置诱导的一步预测与完整网格预测TV为{min(p["exported_full_prediction_tv"] for p in probabilities):.3f}–{max(p["exported_full_prediction_tv"] for p in probabilities):.3f}。这是已冻结有限表示的离散概率近似差异，不能混同运动核不一致，也不能宣称64点等于完整后验。没有据此调整D。','',
        '传感器采用公开地图、6m半径和闭矩形遮挡。24个接管位置上，精确传感器与D保守负观测保护的分歧概率为0；**这不是全状态一致性证明**。',
        '单元测试中的[精确墙角接触例](boundary_fixture_gap.json)仍揭示D的数值保护与精确传感器的已知差异（一步概率约0.00847）。该约1e-6边界保护是冻结D的一部分，未删除。',
        '未来正式研究必须显式约定这类边界近似；不能将本轮样本内检查描述成全局观测模型完全一致。','',
        '## 工程探针的搜索语义','',
        '搜索探针为带权根采样的开环UCT，预定8步规划深度、最多1000次模拟。T与T+Geo共用树选择、rollout、奖励、目标运动核与传感器，仅根候选不同。',
        'T根候选为9个罗盘/停止动作，T+Geo增加公开顶点与带权区域代表点的几何首段动作；深层统一使用9个基础动作。共同G基线检查这些几何方向的期望路程与静态可视概率。',
        '候选按真实权重计算，不把弱分支均匀加权。G的静态可视评分不是预期信息增益。',
        '树节点只按动作历史索引；动作选择和rollout动作不读取采样的隐藏目标。模拟器可用采样世界计算观测、捕获与奖励。规划叶节点单独记录，非法动作不当作正常终局回传。',
        '开环树没有未来观测分支，不在模拟中重新运行完整D滤波；这是一项明确的规划近似，尚不能声称已验证完整belief-tree。它只用于本阶段接口/预算，不用其模拟捕获次数宣称实际捕获收益。',
        f'9次（3种拓扑×3方法）固定实际模拟数重放均逐位一致，包括动作、根采样次数、根访问/回报、搜索RNG、估计状态与模拟轨迹散列。','',
        '## 初版失效与最小接口修复','',
        f'初版同24组审计检测到{oldbad[0]}条自身模拟接触线段、{oldbad[2]}次传感器分歧，目标运动非法为{oldbad[1]}。见[初版结果](../l45_weighted_search_interface_20261026/results.json)、[精确重放现场](../l45_weighted_search_interface_20261026/numerical_diagnosis.json)。',
        '典型原因为数学上接触矩形墙角时，slab交集边界浮点值相差一个ULP，旧clear误判畅通。独立检查用不同的方向/边相交算法识别。',
        'v2只在研究适配器中对自身候选与模拟动作使用1e-6障碍膨胀，并在传感器slab交集加入1e-12参数容差；目标图、D概率、D更新和部署几何函数没有改动。',
        '初版结果、源文件快照和输入散列保留，没有覆盖失败数据。v2重新运行全部工程检查后零非法；不能把这一修复称为估计覆盖或搜索收益提升。','',
        '## 实际总耗时','',
        '同一进程串行执行，数值线程数1，三方法按组轮换。首次JIT/缓存加载单列[warmup.json](warmup.json)，下表为预热后真实调用。',
        '时间从act入口到最终动作合法性验证，包含估计、按需建图、权重视图、几何候选、树数组与模拟；搜索预留2ms，已超时则不启动树，仍记录超时。',
        '静态地图/加载在setup单列，历史仅吸收history[:-1]一次，当前观测在首决策处理。历史初始化+首决策、加静态准备的完整启动都报告。','',
        '|方法|普通隐藏帧 P95 ms（每方法224帧）|首次遮挡 P95 ms（24帧）|历史+首决策 P95 ms|完整启动 P95 ms|首次遮挡>50ms|',
        '|---|---:|---:|---:|---:|---:|']
    for m in methods:
        s=d['summary'][m]
        lines.append(f'|{m}|{s["ordinary_hidden"]["p95_ms"]:.2f}|{s["occlusion_build"]["p95_ms"]:.2f}|{s["history_plus_first_decision"]["p95_ms"]:.2f}|{s["complete_startup"]["p95_ms"]:.2f}|{s["occlusion_build"]["over_50ms"]}/24|')
    lines+=['','“普通隐藏帧”每方法224个，来自同24条相关历史；表内224不是224个独立场景。',
        '普通隐藏帧三方法均无>50ms；首次遮挡三方法24/24超时。不能删除这24帧后宣称全流程满足50ms。','',
        '|方法/帧类型|估计P95 ms（其中建图）|候选P95 ms|搜索P95 ms（含树准备）|最终检查P95 ms|',
        '|---|---:|---:|---:|---:|']
    for m in methods:
        for kind in ('ordinary_hidden','occlusion_build'):
            c=d['summary'][m]['components'][kind]
            lines.append(f'|{m}/{kind}|{c["update_ms"]["p95_ms"]:.3f}（{c["build_ms"]["p95_ms"]:.3f}）|{c["candidate_ms"]["p95_ms"]:.3f}|{c["search_including_prepare_ms"]["p95_ms"]:.3f}|{c["final_check_ms"]["p95_ms"]:.3f}|')
    lines+=['','各分量P95不能相加当作总P95，完整总时延以第一张表的入口/出口计时为准。',
        f'T普通隐藏帧累计{d["summary"]["T"]["simulations"]["total"]:,}次模拟，T+Geo为{d["summary"]["T+Geo"]["simulations"]["total"]:,}次；初次遮挡两者都因估计超时而零模拟。',
        '最多1000次模拟是本探针的上限；并未强制搜索耗满50ms，也未把不同实际模拟数作效果比较。',
        '同一公开历史合法往返后重新变为不可见，六次构图（每拓扑两次）的具体时间见results.json/repeated_occlusions，证明该成本在重发现后可重复发生。',
        '图构造依赖最后一次可见位置；不能简单把它归入只发生一次、与在线决策无关的静态地图成本。',
        '![完整时延与50ms门槛](figures/latency_gate.png)','',
        '### 补核：从外部计时到函数实际返回','',
        '上表内部计时在动作最终检查完成后取时间戳，尚不含其后的诊断字典组装与返回。为核算这部分，另按[补核方案](external_timing_protocol.json)重放同24组、共960次调用，直接在act外部计时至返回。',
        '此补核使用相同已见工程历史，发生在首批结果已知之后，不是新的独立效果确认；原批和补核均保留。','',
        '|方法|完整返回：普通隐藏 P95 ms|完整返回：首次遮挡 P95 ms|首次遮挡超时|',
        '|---|---:|---:|---:|']
    for m in methods:
        s=external['summary'][m]
        lines.append(f'|{m}|{s["ordinary_hidden"]["p95_ms"]:.2f}|{s["occlusion_build"]["p95_ms"]:.2f}|{s["occlusion_build"]["over_50ms"]}/24|')
    lines+=[f'','诊断组装/返回增量的同调用配对P95为'
        f'{external["return_overhead"]["p95_ms"]:.4f}ms、最大{external["return_overhead"]["maximum_ms"]:.4f}ms，见[external_timing.json](external_timing.json)。',
        '两个批次的构图耗时存在本机运行波动，两批完整预算均失败；两批P95之差不能解释为诊断组装开销或修复收益。未固定CPU亲和性，不作硬实时保证。','',
        '## 验收与下一步','',
        '|检查|结果|','|---|---|']
    for k,v in d['interface_checks'].items():lines.append(f'|{k}（本轮样本内）|{"通过" if v else "失败"}|')
    for k,v in d['budget_checks'].items():lines.append(f'|{k}完整预算|{"通过" if v else "失败"}|')
    lines+=['','**本阶段在预算门槛停止。**没有运行真实对手闭环，因此无本轮重发现率、合法捕获率或完成步数的效果结果，也无新场景泛化结论。',
        'D不再优化且继续冻结。若继续推进，只应先研究保持D输出/RNG逐位不变的接口执行安排及初始化预算协议，并公开计入总计算开销；不能偷偷放宽50ms或更换估计器。',
        '预算通过后才能冻结新的受控与独立效果确认场景、同一真实对手、搜索语义和比较门槛，再比较T−G与T+Geo−T。保留强G，未确认额外收益前不制作教师或训练网络。','',
        '## 数据与复核','',
        '- [预登记与源文件/公共输入散列](experiment.json)、[逐组决策记录](groups/)、[全量审计汇总](results.json)、[阶段决定](decision.json)。',
        '- 初版[源文件快照](../l45_weighted_search_interface_20261026/source_snapshot.json)与失败结果保留；v2代码：[适配器](../../code/solvers/frozen_d_search_interface.py)、[审计执行器](../../code/audit_frozen_d_search_interface.py)。',
        '- [冻结D](../l45_corner_turn_estimator_20261025/frozen_estimator/README.md)及部署相关5个文件SHA均与本轮开始一致。',
        '- [960个最终输出动作的独立几何检查](final_action_checks.json)零错误；[源文件完整性复核](source_integrity_checks.json)通过。',
        '- 测试包含权重0.9反例、同历史估计与固定模拟数重放、解析转移与负观测、拒绝非法权重/隐藏参数、精确墙角回归；5项通过。',
        '- 未请求Git提交，未修改既有部署工作区改动。','',
        '```bash',
        'PY=/home/sjc/miniconda3/envs/dm_env_py311/bin/python',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" -m unittest discover -s code/tests -p test_frozen_d_search_interface.py -v',
        '# 原结果拒绝覆盖；复现审计须指定另一个输出目录，并单独串行执行',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" code/audit_frozen_d_search_interface.py --output /tmp/l45_weighted_interface_replay',
        '```','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    (OLD/'REPORT.md').write_text('# 冻结D搜索接口：初版失败记录\n\n2026-10-26。此版本已冻结保留，未进行实际搜索效果比较。\n\n'
        f'24组历史、960次决策；独立检查自身运动{oldbad[0]}次、目标运动{oldbad[1]}次、传感器{oldbad[2]}次、捕获{oldbad[3]}次异常。'
        '精确重放定位为墙角接触的浮点判定。首次遮挡的三方法P95约81ms，均未通过50ms预算。\n\n'
        '原始结果见[results.json](results.json)、现场见[numerical_diagnosis.json](numerical_diagnosis.json)、'
        '源文件散列及副本见[source_snapshot.json](source_snapshot.json)。\n\n'
        '随后仅修复研究适配层的几何判定；D保持不变。完整重跑结果见[v2报告](../l45_weighted_search_interface_v2_20261026/REPORT.md)。\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    x=np.arange(3);width=.25
    for j,(kind,label,color) in enumerate((('ordinary_hidden','Ordinary hidden','#4c78a8'),
            ('occlusion_build','First occlusion','#f58518'),('complete_startup','Startup incl. first decision','#b279a2'))):
        values=[d['summary'][m][kind]['p95_ms'] for m in methods]
        axes[0].bar(x+(j-1)*width,values,width,label=label,color=color)
    axes[0].axhline(50,color='red',ls='--',label='50 ms budget')
    axes[0].set(xticks=x,xticklabels=methods,ylabel='Measured P95 latency (ms)',title='Full timing gate: fails at occlusion')
    axes[0].legend(fontsize=8)
    for j,(field,label,color) in enumerate((('build_ms','Graph construction','#f58518'),('candidate_ms','Candidates','#4c78a8'))):
        vals=[d['summary'][m]['components']['occlusion_build'][field]['p95_ms'] for m in methods]
        axes[1].bar(x+(j-.5)*.3,vals,.3,label=label,color=color)
    axes[1].axhline(50,color='red',ls='--')
    axes[1].set(xticks=x,xticklabels=methods,ylabel='Component P95 (ms)',title='Construction alone exceeds budget')
    axes[1].legend(fontsize=8)
    (OUT/'figures').mkdir(exist_ok=True)
    fig.savefig(OUT/'figures/latency_gate.png',dpi=170)
    fig.savefig(OUT/'figures/latency_gate.pdf')
    plt.close(fig)


if __name__=='__main__':main()
