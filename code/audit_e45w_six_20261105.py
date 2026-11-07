"""Read-only synthesis of W's six registered experiments, after model freeze."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
W=ROOT/'current/e45w_joint_ppo_and_planning_20261103'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pair_summary(final,cfg):
    summaries=final['summaries'];result=[]
    for opponent_arm in ('A','B'):
        for suite in ('all','benchmark','generalization'):
            groups=sorted(r['id'] for r in cfg['test_entries'] if suite=='all' or r['suite']==suite)
            values=[]
            for group in groups:
                values.append(np.mean([summaries[f'C_{seed}_selected']['test']['stochastic_primary']['group_utility'][group]
                    -summaries[f'A_{seed}' if opponent_arm=='A' else f'B_{seed}_selected']['test']['stochastic_primary']['group_utility'][group]
                    for seed in cfg['seeds']]))
            values=np.asarray(values);rng=np.random.default_rng(14530261103)
            boot=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
            result.append(dict(contrast=f'C_selected-{opponent_arm}',suite=suite,groups=len(groups),
                utility_difference=float(values.mean()),descriptive95=np.percentile(boot,[2.5,97.5]).tolist(),
                scope='Fixed three training seeds, both opponents combined, uncorrected map-group exploratory interval'))
    return result


def main(wait):
    if not read(W/'test_scores_unlock_repair1.json')['unlocked']:raise RuntimeError('Do not inspect sealed test efficacy')
    if wait:
        while True:
            status=read(W/'final_evaluation_repair1/status.json')
            if status['status']=='complete':break
            if status['status']=='failed':raise RuntimeError('Final evaluation failed; see preserved failure')
            time.sleep(20)
    cfg=read(W/'experiment.json');registry=read(W/'repair_coordinator/registry.json')
    inputs=[W/'experiment.json',W/'test_scores_unlock_repair1.json',W/'repair_coordinator/registry.json',Path(__file__)]
    planning={mode:read(W/directory/'summary.json') for mode,directory in
        [('wall','planning_wall_repair1'),('sim','planning_sim_v2')]}
    inputs.extend([W/'planning_wall_repair1/summary.json',W/'planning_sim_v2/summary.json'])
    validation=[]
    for name,row in registry.items():
        d=Path(row['directory']);status=read(d/'status.json');selection=read(d/'selection.json')
        assert status['status']=='complete',name
        records=[json.loads(line) for line in (d/'validation.jsonl').read_text().splitlines()]
        initial=next(r for r in records if r['team_transitions']==0)
        selected=next(r for r in records if r['team_transitions']==selection['team_transitions'])
        validation.append(dict(name=name,initial=initial['stochastic'],selected=selected['stochastic'],
            selected_step=selection['team_transitions'],latest=records[-1]['stochastic'],
            latest_step=records[-1]['team_transitions'],repair_budget=status.get('repair_budget'),
            effective_optimizer_steps=status['counts']['optimizer_steps'],source_sampling_spent=status.get('source_counts_at_cap')))
        inputs.extend([d/'status.json',d/'selection.json',d/'validation.jsonl'])
    status=read(W/'final_evaluation_repair1/status.json');final=None;comparisons=[]
    if status['status']=='complete':
        final=read(W/'final_evaluation_repair1/results.json');comparisons=pair_summary(final,cfg)
        inputs.extend([W/'final_evaluation_repair1/results.json',W/'final_evaluation_repair1/paired_group_statistics.json'])
    report=dict(timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        nominal_experiment_count=6,learners_in_experiment_6=6,training_budget_finished=True,
        final_evaluation_complete=final is not None,final_evaluation_progress=status['counts'],
        B29_true_cap_endpoint_missing=True,B29_reference_endpoint=1843200,B29_added_training_steps=0,
        validation=validation,planning=planning,final_evaluation=final,final_selected_comparisons=comparisons,
        input_sha256={str(p):sha(p) for p in inputs},
        classification='Centralized joint team PPO with public observations; C supervised initialization then pure PPO; not author RMAPPO or decentralized MARL.')
    target=W/'six_experiment_audit_20261105.json';target.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    lines=['# W六项实验：目的、结果与结论审查','',f'生成时间：{report["timestamp"]}。',
        '',f'六路预算均已结束；统一最终评测完成：{report["final_evaluation_complete"]}。',
        'B29在预算跑满后的验证阶段失败，真实最终Actor未保存；仅用1843200步可恢复检查点收尾，没有新增训练。不能声称完整原终点已找回。',
        '', '|实验|问题|结果与边界|','|---|---|---|',
        '|1 墙钟曲线|Single PRIOR相对GEOM/U是否有同耗时优势？|2520有效局。Single−GEOM的12项效用同时区间全部含零；200ms部分条件显著优于U。|',
        '|2 冻结A直控|监督资产本身多强，能否提供与C同接口起点？|原1944验证/测试任务已采集；新语义A在统一评测中重评。C−A检验RL增益。|',
        '|3 等模拟次数|相同16/64/128完整模拟时，学习提议是否改善搜索？|2520有效局。仅ID、Root、16模拟相对GEOM的效用同时区间明确为正。不能转述为普遍同耗时优势。|',
        '|4 成本|完整控制返回花多少时间？|复用实验1；50ms经常超时。200ms主要p95约196–197ms；不构成硬实时保证。离线教师/训练总成本不在这张在线账单内。|',
        '|5 工程验收|实现概率、物理、退休与保存恢复语义是否可靠？|初始25项与修复28项检查通过；本次预算收尾3项通过。支持受检实现，不是算法最优性证明。|',
        '|6 B/C学习|随机初始化能否学习，RL能否超过同监督起点？|三种子B、三种子C；下表分开列验证与最终测试。|',
        '', '## 验证集（每项96局）','', '|模型|初始双捕|验证选中双捕|选中步数|有效终点双捕|','|---|---:|---:|---:|---:|']
    for row in validation:lines.append(f'|{row["name"]}|{row["initial"]["double_captures"]}|{row["selected"]["double_captures"]}|{row["selected_step"]}|{row["latest"]["double_captures"]}|')
    if final is not None:
        lines+=['','## 统一最终测试（随机执行为主）','','|模型|双捕/336|效用|','|---|---:|---:|']
        for model in final['models']:
            row=final['summaries'][model['name']]['test']['stochastic_primary']
            lines.append(f'|{model["name"]}|{row["double_captures"]}/{row["episodes"]}|{row["utility"]:.5f}|')
        lines+=['','|选中模型对比|地图范围|效用差|地图95%描述区间|','|---|---|---:|---|']
        for row in comparisons:lines.append(f'|{row["contrast"]}|{row["suite"]}|{row["utility_difference"]:.5f}|{row["descriptive95"]}|')
        lines+=['','区间固定这三个训练种子，仅反映地图重采样；没有量化训练种子总体不确定性，也未作全家族同时校正。B29 terminal指可恢复端点。']
    else:lines+=['','最终评测进行中，以上学习结论仅为验证证据，不能替代最终测试；完整结果生成后重新运行本脚本。']
    lines+=['','## 对原研究方向的影响','',
        'C−A才是RL阶段的收益；C−B主要反映监督初始化在当前配方和预算下的价值。C的监督数据/教师成本必须披露。',
        '得到高分学习对照并不能自动证明学习规划方法优于强几何。当前同耗时Single−GEOM优势仍未建立，低模拟数ID证据应按其限定范围报告。',
        '等模拟旧G与墙钟r1是不同完整运行；成功正常路径语义相同，但不能拼成同版本同轨迹的单因素对照。',
        'A的policy_inference_ms只计网络推理，不能与规划完整act时延直接比较；嵌套分项不能相加为总时间。',
        'A与C执行接口一致；Single PRIOR树内合法提议与A的一次提议安全映射存在差异，Single PRIOR−A不单独隔离树搜索机制。',
        '集中式团队PPO应准确标注；AG/IG/RouteAPF等强参考不因不在本轮六项中而从论文消失。',
        'X/Y/Z创新候选另属开发诊断，未升级为新主方法，不与W冻结评测混合或当作独立强算法数量。']
    (ROOT/'六项实验目的与结果审查_20261105.md').write_text('\n'.join(lines)+'\n')
    print('Audit written; final complete:',final is not None)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait-final',action='store_true')
    main(parser.parse_args().wait_final)
