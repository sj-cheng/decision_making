"""Add engineering evidence and immutable-delivery links to the study report."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def complete(output):
    output=Path(output).resolve()
    p=output/'REPORT.md';text=p.read_text()
    marker='\n## 工程证据与计算解释\n'
    if marker in text:text=text.split(marker)[0]
    preflight=json.loads((output/'preflight.json').read_text())
    lines=[marker.strip(),'','12项模型概率、信息隔离、固定数重放、在线D隔离、预算中断以及精确执行复用测试通过。',
        '27组候选数组/动作和OL/FB各固定16次模拟的慢/快执行逐位一致。尾路程复用只预计算公开地图代价，图重锚后重新准备；准备时间计入决策。',
        '工程计时15次根决策，三拓扑×五方法；同一公共历史下五法D逐位相同，固定8次模拟重放一致，独立几何/传感器/捕获核验无差异。','',
        '|拓扑|方法|初始化+首决策ms|完成模拟|已访问/合法根候选|重放|',
        '|---|---|---:|---:|---:|---|']
    for r in preflight['records']:
        lines.append(f"|{r['family']}|{r['method']}|{r['initial_total_ms']:.2f}|{r['simulations']}|{r['visited_roots']}/{r['root_candidates']}|{r['replay_exact']}|")
    if (output/'resume_record.json').exists():
        resume=json.loads((output/'resume_record.json').read_text())
        lines += ['','客户端中断时进程结束；已提交的108局完整缓存逐文件保留，未提交的下一局按原配置重做。续跑独立预热不计入决策，没有更改方法、参数或使用确认结果。详见[续跑记录](resume_record.json)。']
    for record_path in sorted(output.glob('confirmation_resume_record*.json')):
        resume=json.loads(record_path.read_text())
        lines += ['',f"确认运行也曾因客户端中断结束：该次已完成{resume['committed_confirmation_games']}局的缓存与轨迹逐文件冻结，未提交局按相同配置重跑。重启预热不计入决策；未根据中途表现调整方案。见[确认续跑记录]({record_path.name})。"]
    for stage in ('development','confirmation'):
        result=output/stage/'results.json'
        if not result.exists():continue
        rows=json.loads(result.read_text());audit_path=output/stage/'postrun_audit_v2.json'
        if not audit_path.exists():audit_path=output/stage/'postrun_audit.json'
        audit=json.loads(audit_path.read_text())
        stats=collections.defaultdict(lambda:collections.Counter())
        for row in rows:
            payload=json.loads(Path(row['episode']).read_text())
            for frame in payload['trace']:
                if frame['observation']['target'] is not None:continue
                key=(row['opponent'],row['method']);s=frame['statistics'];c=stats[key]
                c['hidden']+=1;c['simulations']+=s['simulations'];c['aborted']+=s.get('budget_aborted_simulations',0)
                c['conditionings']+=s.get('exact_conditionings',0)
                c['positive_nodes']+=s.get('positive_observation_nodes',0)
                c['negative_nodes']+=s.get('negative_observation_nodes',0)
                c['unvisited']+=int(row['method']!='G' and sum(v>0 for v in s['root_visits'])<s['root_candidates'])
                c['zero_simulations']+=int(row['method']!='G' and s['simulations']==0)
                c['single_visit']+=int(row['method']!='G' and bool(s['root_visits']) and max(s['root_visits'])==1)
                c['over_budget']+=int(frame['own_ms']>row['budget_ms'])
                for timing in ('update_ms','candidate_ms','search_including_prepare_ms','final_check_ms'):
                    c[timing]+=s[timing]
                c['actual_hidden_ms']+=frame['own_ms']
        lines += ['',f"### {stage}逐帧独立复核",'',
            f"审计通过：{audit['passed']}；审计错误{len(audit['errors'])}。模拟转移累计{sum(r['simulated_transitions'] for r in rows):,}条，保存时逐条核验动作/传感器/捕获。事后按完整公共观测精确重放D、核对配对初态/逃跑方首动作，并独立重放matched目标全部转移。",'',
            '|目标|方法|隐藏决策|平均完成模拟|根未遍历完决策|零完成模拟|仅单次根访问|部分模拟中断|超过本法预算|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for (opponent,method),c in sorted(stats.items()):
            lines.append(f"|{opponent}|{method}|{c['hidden']}|{c['simulations']/max(1,c['hidden']):.1f}|{c['unvisited']}|{c['zero_simulations']}|{c['single_visit']}|{c['aborted']}|{c['over_budget']}|")
        lines += ['','|目标|方法|初始化均值ms|隐藏更新均值ms|候选均值ms|搜索含准备均值ms|最终检查均值ms|隐藏实际返回均值ms|',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for (opponent,method),c in sorted(stats.items()):
            selected=[r for r in rows if r['opponent']==opponent and r['method']==method]
            init=np.mean([r['initialization']['total_ms'] for r in selected])
            means=[c[k]/max(1,c['hidden']) for k in ('update_ms','candidate_ms','search_including_prepare_ms','final_check_ms','actual_hidden_ms')]
            lines.append(f"|{opponent}|{method}|{init:.2f}|"+'|'.join(f'{v:.2f}' for v in means)+'|')
        lines += ['','初始化包括构造与接管前公开历史回放，按每局平均，已从首次决策预算扣除。动态分项按隐藏决策平均；根准备已含在搜索，建图已含在更新，不重复相加。外层实际返回还包括控制器内部计时之外的调用开销；P95和最大值见前表。']
        lines += ['','|目标|方法|完整联合周期P95/最大ms|联合周期>1s|对手合法停止盾触发|',
            '|---|---|---:|---:|---:|']
        for opponent,method in sorted(stats):
            selected=[r for r in rows if r['opponent']==opponent and r['method']==method]
            cycles=[v for r in selected for v in r['cycle_ms']]
            lines.append(f"|{opponent}|{method}|{np.percentile(cycles,95):.2f}/{max(cycles):.2f}|{sum(v>1000. for v in cycles)}|{sum(r['shield_count'] for r in selected)}|")
        lines += ['','联合周期含双方决策和环境推进，不含落盘/事后审计。环境在计算期间锁步；高预算超过1秒物理周期，不能作为实时部署证据。登记期限用于主动停止搜索，实际超时帧按表保留，不能把2000ms登记期限当作硬实时保证。']
        lines += ['','根访问不足和低模拟数是有限预算限制，不能解释成所有观测反馈搜索无效。即使高预算根候选遍历完整，也不代表8步树已经充分收敛或达到最优。',
            'FB的信念是整个格点×模式的概率数组；其根先验仍由64个带权导出样本诱导，不是完整D后验。几何动作也由64个带权导出样本计算。树内没有用单个采样隐藏位置替代信念，也没有把模式暴露给追捕动作。',
            '开环和反馈的几何导出来自各自预测/条件信念，随机数调用数量可能不同；相同种子和初态用于配对，不声称分叉后的整段模拟随机轨迹完全相同。','']
        fig=output/'figures'/f'{stage}_costs.png'
        if fig.exists():lines.append(f'![成本与成组区间](figures/{stage}_costs.png)')
    if (output/'corrective_audit_registration.json').exists():
        lines += ['','### 独立审计器更正与保留记录','',
            '首版审计硬编码13维状态，且把限幅加法后的实际物理坐标要求为理想格点逐位相等，产生112条报错。完整公开状态模板与独立物理限幅递推已验证：这是审计器错误，未更改控制器、D或环境。',
            'v2从实际公开环境读取状态维数/活动索引，并逐位检查raw及physical_next=physical_current+clip(raw,-1,1)。保持D逐位重放、原动作/传感器/捕获检查，不增加坐标atol。5项有针对性审计回归测试通过。',
            '原112条报错及首版停止决定原样保留；v2源码/测试/登记散列在首个确认结果前固定。见[更正原因](../../code/L45_SEARCH_VALUE_AUDIT_CORRECTION_20261026.md)、[登记与散列](corrective_audit_registration.json)、[原审计](development/postrun_audit.json)、[修正审计](development/postrun_audit_v2.json)、[审计回归测试](audit_unit_tests.json)。']
    lines += ['','### 如何解释这一轮','',
        '对局沿用Example8单个活跃追捕者/单个活跃逃跑者接管设置：逃跑方轴向动作上限1.0，追捕方1.1或1.3；局部可见半径6，地图遮挡；物理步长1、原总时限80。真实对手为原RootPUCT，3000次模拟、深度100、无网络、可读全状态；追捕器只收局部公开观测。',
        'OL/FB是独立研究对照，不替换原始T/T+Geo或部署控制器，也不等同于完整NTE复现。两者唯一区别是模拟观测是否条件化规划信念、分裂树节点并影响后续动作；有限耗时下完成的模拟数会随反馈计算开销改变。',
        '两者都规划8个物理步，首次重发现奖励1、捕获奖励1、折扣0.95；固定根图目标模式0.9保持、0.1均匀重置。根候选使用强G全部合法几何候选，深层使用合法九方向。奖励、深度、目标核在本轮没有调整。',
        '本轮只检验登记的固定根运动模型、8步规划和候选机制。目标转移与真实RootPUCT策略的差异未通过改变对手模型来修复；奖励、规划深度和网络也未同时改变。',
        'matched的转移核匹配仅限首次重发现前。在线D负观测数值保护、64导出近似，以及首次正观测后的重锚仍存在，不能称其完整后验与生成世界完全匹配。',
        '开发若触发预定停止，只能说该版本没有达到继续投入的筛选门槛；不能把开发置信区间称独立确认，也不能证明NTE/MCTS或未来神经扩展普遍无效。',
        '原50ms失败、上一轮rollout修正失败和强G结果保留；本轮没有训练、教师生成或部署修改。','',
        '### 复现与原始记录','',
        '- [结果前方案](../../code/L45_SEARCH_VALUE_VALIDATION_20261026.md)、[冻结源码/参数](experiment.json)、[新确认银行散列](bank_frozen.json)。',
        '- [工程验收](preflight.json)、[精确执行复用前后](engineering_profile.json)、[12项测试记录](unit_tests.json)。',
        '- [运行日志](pipeline.log)、[开发继续门槛](development_gate.json)、[原执行决定](decision.json)。',
        '- [独立solver](../../code/solvers/search_value_belief.py)、[概率模型](../../code/search_value_model.py)、[执行器](../../code/run_search_value_validation.py)。',
        '- [交付核验](delivery_checks.json)、[完整文件散列](DELIVERY_SHA256.json)。','']
    if (output/'final_decision.json').exists():lines += ['- [最终权威执行决定](final_decision.json)。','']
    for stage in ('development','confirmation'):
        if (output/stage/'failure_diagnosis.md').exists():
            lines += [f'- [{stage}失败与挽救轨迹的事后诊断]({stage}/failure_diagnosis.md)：固定规则保留正反例，不用于调整确认方案。','']
    p.write_text(text.rstrip()+'\n\n'+'\n'.join(lines))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'current/l45_search_value_20261026')
    complete(ap.parse_args().output)
