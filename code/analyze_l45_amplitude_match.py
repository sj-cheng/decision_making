"""Explain committed M/F results and verify matching; no new search or episodes."""
import numpy as np
import run_l45_amplitude_match as m

def main():
    args=m.parse_args([]);cases=m.prepare(args);items=m.requests(args,cases)
    result=m.locked_read(args.output/'statistics.json');diag=m.locked_read(args.output/'diagnostics.json')
    index={}
    for item in items:
        row,record=m.read_episode(args,item);index[(item['uid'],row['scene_id'],row['stream'])]=(row,record)
    matched=0
    for scene in sorted({k[1] for k in index}):
        for stream in (0,1):
            reference=index[('F',scene,stream)][1]['initial']
            for uid in ('M','S'):
                row,d=index[(uid,scene,stream)]
                for key in ('state','stats','time','remaining_time','own_rng','opponent_rng','python_rng'):
                    if not m.old.s.equal_tree(reference[key],d['initial'][key]):
                        raise ValueError(f'Initial matching changed: {uid} {scene} {stream} {key}')
            matched+=1
    if sum(not row['reused_historical'] for row,_ in index.values())!=198:
        raise ValueError('New execution ledger differs')
    mf=result['comparisons']['M-F'];ms=result['comparisons']['M-S'];g=result['summary']
    audit=dict(passed=True,matched_initial_state_rng_triplets=matched,all_F_M_S_initial_fields_exact=True,
        verified_artifacts=len(index),new_closed_loop_M_suffixes=198,extra_searches=0,extra_environment_episodes=0)
    m.o.write_json(args.output/'matching_audit.json',audit)
    m.o.write_json(args.output/'analysis_provenance.json',dict(script=str(__file__),script_sha256=m.o.sha(__file__),
        inputs={name:m.o.sha(args.output/name) for name in ('statistics.json','diagnostics.json','integrity.json','experiment.json')},
        additional_search_calls=0,additional_environment_episodes=0))
    lines=['M/F匹配幅度对照结果','',
        '**结论：满幅放大带来明确收益；不放大均值的M明显差于S。下一阶段保留F的幅度归一化，不将M作为简化替代。**','',
        f'实际历史来源：{m.SOURCE}。99个相同首捕状态、每状态两个原配对搜索流；只新增198条M后缀，F/S各复用198条，不训练。','',
        '|组|流0成功/99|流1成功/99|总成功/198|接管后成功率|全100参考口径|后段成本 步|自身非法|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for u in ('S','M','F'):
        x=g[u];a,b=x['successes_by_stream']
        lines.append(f'|{u}|{a}|{b}|{x["successes"]}|{100*x["conditional_success_rate"]:.2f}%|{100*x["full100_reference_rate"]:.2f}%|{x["mean_post_cost"]:.2f}|{x["own_illegal"]}|')
    lines+=['','全100参考口径保留原来的一个零捕场景；其未产生新后缀。主统计以99个物理状态为单位，先平均两流，再进行20000次配对bootstrap，种子57020261024。','',
        '|比较|接管后成功率差 pp|普通95%区间 pp|同时95%区间 pp|挽回/丢失 状态×流|平均后段成本差 步 [95%区间]|',
        '|---|---:|---|---|---:|---|']
    for name in ('M-F','M-S','F-S'):
        x=result['comparisons'][name];lo,hi=np.array(x['ci95'])*100;sim=x.get('simultaneous_ci95')
        simtext=f'[{sim[0]*100:+.2f},{sim[1]*100:+.2f}]' if sim else '辅助比较'
        cl,ch=x['post_cost_ci95']
        lines.append(f'|{name}|{100*x["delta"]:+.2f}|[{lo:+.2f},{hi:+.2f}]|{simtext}|{x["gained_state_streams"]}/{x["lost_state_streams"]}|{x["mean_post_cost_delta"]:+.2f} [{cl:+.2f},{ch:+.2f}]|')
    lines+=['','同时区间共同覆盖M-F、M-S两项比较；两项均完全低于零。198条后缀不当作198个独立物理样本。逐场景两流结果及差异见paired_scene_results.json。','',
        f'相对F，M没有挽回，丢失{mf["lost_state_streams"]}个状态×流；平均到物理状态后，{mf["worsened_physical_states"]}个退化、{mf["improved_physical_states"]}个改善。',
        f'相对S，M挽回{ms["gained_state_streams"]}、丢失{ms["lost_state_streams"]}个状态×流；{ms["worsened_physical_states"]}个物理状态退化、{ms["improved_physical_states"]}个改善。','',
        '这一结果否定了“增加更多原幅度神经均值方向候选就能解释F收益”的解释。M与F保持同一基础方向、绝对扰动和候选比例，去掉均值放大后，成功率和收尾成本同时明显恶化。',
        'S仍包含50%原神经候选；M是在此基础上用原幅度均值候选替换25%均匀候选。因此M-S说明这一不放大的替换方案有害，不说明神经方向信息本身毫无价值。',
        '结合上一轮D/G与F接近、P没有超过G，当前证据支持在有效方向上充分利用动作能力：网络均值方向可用于生成有效大位移动作，预测拦截并非获得当前收益的必要条件。','',
        '**计算开销与动作记录。**','',
        '|组|后段决策数|环境转移|网络调用|几何查询|平均环境转移/决策|首动作归一化幅度|附加候选归一化幅度|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for u in ('S','M','F'):
        x=g[u];d=diag[u];cost=x['own_cost'];added=d['candidate_magnitudes_mean']
        addedtext=f'{added:.3f}' if added is not None else '无'
        lines.append(f'|{u}|{x["decisions"]}|{cost["step_calls"]}|{cost["policy_evaluations"]}|{d["geometry_queries"]:.0f}|{cost["step_calls"]/x["decisions"]:.1f}|{d["first_magnitudes_mean"]:.3f}|{addedtext}|')
    env_reduction=100*(1-g['F']['own_cost']['step_calls']/g['M']['own_cost']['step_calls'])
    net_reduction=100*(1-g['F']['own_cost']['policy_evaluations']/g['M']['own_cost']['policy_evaluations'])
    lines+=['',f'F相对M的后段总环境转移少{env_reduction:.1f}%，网络调用少{net_reduction:.1f}%；主要因为决策次数更少、收尾更早。',
        'M/F保留相同几何预计算；没有同时优化或删除辅助代码。F每次决策的环境转移并不更少，不能将总调用下降写成单次搜索更便宜。',
        '并发耗时保存在逐后缀缓存中，跨轮资源争用不同，本轮不据其声称串行部署加速；未额外执行计时对局。','',
        '**运行前与完成后核验。**','',
        '历史396条F/S记录的权重、配置、快照、剩余时间、双方C++及Python/NumPy/Torch状态精确恢复。',
        '独立扩展精确复现198个历史F首根动作与根统计；8组F搜索后随机状态对照一致；20组种子的M/F共同扰动检查通过。',
        '4项针对性测试通过。完整594条记录的缓存和轨迹均已校验，16917次已保存追捕搜索均为1000模拟，无价值调用或引导rollout。',
        '完成后198组M/F/S初始状态与全部随机状态再次精确匹配；只新增198条M闭环后缀。工程检查仅根搜索，没有另跑M闭环smoke或重放。',
        '新入口缺失导入的首次启动已在任何科学后缀执行前归档，见../l45_amplitude_match_startup_archive_20261024；修正后重新核验通过。','',
        '**下一步建议。** 保留满幅F作为后续独立验证对象，不采用原幅度M。候选生成可在更简单的F/D/G之间研究，当前不需要保留预测拦截作为必要组件。',
        '本轮为固定O检查点、99个历史首捕状态上的匹配干预；新场景泛化尚未检验。当前部署推荐保持old0:novalue，没有顺带启动其他实验。','',
        '![M/F匹配对照](figures/MF_matched.png)']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (args.output/'CONCLUSION.md').write_text('\n'.join(lines[:12])+'\n\n'+
        'M-F=-44.95pp，95%区间[-52.53,-37.37]；M-S=-21.21pp，95%区间[-28.79,-13.64]。保留F幅度放大，不采用M；未启动后续实验。\n')

if __name__=='__main__':main()
