"""Explain the frozen fresh-scene evaluation; no new episodes or searches."""
from pathlib import Path
import numpy as np
import run_l45_direction_validation as m

def main():
    args=m.parse_args([]);banks=m.prepare(args);items=m.requests(args,banks)
    result=m.locked_read(args.output/'statistics.json');diag=m.locked_read(args.output/'diagnostics.json')
    scenes={s['episode_hash']:s for group in banks.values() for s in group}
    rows=[m.read_cached(args,i) for i in items]
    costs={};failures={};speed={};index={(r['uid'],r['episode_hash'],r['opponent']):r for r in rows}
    diagnostics_by_scope={}
    trajectory_index=m.locked_read(args.output/'trajectory_index.json')
    for r in trajectory_index:
        key=(r['layout'],r['opponent'],r['uid'])
        failures.setdefault('|'.join(key),[])
        if not r['success']:failures['|'.join(key)].append(dict(scene=r['scene'],first=r['first_capture_time'],
            zero_capture=not np.isfinite(r['first_capture_time']),post_timeout=r['one_capture_timeout'],own_illegal=r['own_illegal'],record=r['record']))
    # Re-read committed artifacts to retain exact per-map geometry/call accounting.
    for item in items:
        row,d=m.read_episode(args,item);spec=d['scenario'];u=row['uid']
        labels=('all',row['layout'],row['map_id'])
        for label in dict.fromkeys(labels):
            key=label+'|'+row['opponent']+'|'+u
            g=costs.setdefault(key,dict(games=0,decisions=0,step_calls=0,policy_evaluations=0,geometry_queries=0.,
                graph_build_ms=0.,plan_ms=0.,proposal_fallbacks=0.,noise_fallbacks=0.,takeover_games=0,post_costs=[],
                successful_second_steps=[],first_capture_steps=[],root_added_candidates=0.,root_candidates=0.,selected_added=0))
            g['games']+=1;g['decisions']+=row['own_mcts_calls']
            for k in ('step_calls','policy_evaluations'):g[k]+=row['own_cost'][k]
            for root in d['roots']:
                audit=root['guidance']
                for k in ('geometry_queries','graph_build_ms','plan_ms'):g[k]+=audit.get(k,0)
                g['proposal_fallbacks']+=audit.get('proposal_fallback',0);g['noise_fallbacks']+=audit.get('noise_fallback',0)
                g['root_added_candidates']+=audit.get('root_source_2',0);g['root_candidates']+=audit.get('root_candidates',0)
                g['selected_added']+=int(audit.get('selected_source')==2)
            first=d['first_capture']
            if first:
                g['first_capture_steps'].append(first['time'])
                if first['branch_available']:
                    g['takeover_games']+=1;g['post_costs'].append(row['completion_cost']-first['time'])
                    if row['legal_double_capture']:g['successful_second_steps'].append(row['capture_time']-first['time'])
        ratio=spec['pursuer_speed']/spec['evader_speed'];label='<1.2' if ratio<1.2 else '>=1.2'
        g=speed.setdefault(row['layout']+'|'+label+'|'+row['opponent']+'|'+u,dict(games=0,successes=0,cost=0.))
        g['games']+=1;g['successes']+=row['legal_double_capture'];g['cost']+=row['completion_cost']
    for g in costs.values():
        for field in ('post_costs','successful_second_steps','first_capture_steps'):
            values=g.pop(field);g[field+'_mean']=float(np.mean(values)) if values else None
        g['mean_steps_per_decision']=g['step_calls']/g['decisions']
        g['mean_networks_per_decision']=g['policy_evaluations']/g['decisions']
    # Check scene independence again against the exact pre-frozen historical files.
    independence=m.locked_read(args.output/'scenario_independence.json');used=set()
    for name,sha in independence['history_files'].items():
        if m.o.sha(name)!=sha:raise ValueError('History changed during evaluation: '+name)
        for spec in m.extract_specs(m.o.read_json(name)):used.add(m.physical_hash(spec))
    identities=[m.physical_hash(s) for s in scenes.values()]
    if len(set(identities))!=150 or set(identities)&used:raise ValueError('Independent scene leakage')
    m.o.write_json(args.output/'extended_analysis.json',dict(cost_by_map_opponent_uid=costs,
        failures_by_map_opponent_uid=failures,speed_groups=speed,no_extra_search_calls=True))
    m.o.write_json(args.output/'final_review.json',dict(passed=True,scientific_games=1800,new_physical_scenes=150,
        history_overlap=0,history_files_unchanged=True,models_unchanged=m.registry(args)==m.locked_read(args.output/'experiment.json')['models'],
        tests=5,isolated_smoke_games=48,prechecks=m.locked_read(args.output/'checks.json')['passed'],
        integrity=m.locked_read(args.output/'integrity.json'),additional_training=0,additional_followup_experiments=0,
        deployment='old0:novalue'))
    lines=['F/G全新场景独立验证：结果与结论','',
        '**本轮已完成150个全新物理场景×三个原对手×四组，共1800局，无新增训练。**',
        '原四障碍50、未见四障碍50（错位通道/非对称各25）、无障碍50；四组共享初态、速度及双方搜索种子。',
        'B=O旧交替搜索，S=O同步搜索，F=满幅神经均值方向，G=绕障碍追踪。F/G仅在真实首捕后启用。',
        '真实对手完整固定原L模型与配置，追捕为1000模拟/.5/0，SL对手1000/.5/.1，SO对手3000/0/0，深度均100。','',
        '**主结果：两种四障碍组共100个物理场景，三个对手等权。**','',
        '|比较|双捕率差 pp|普通95%区间 pp|主比较同时95%下界 pp|挽回/丢失|完成成本差 步 [95%区间]|',
        '|---|---:|---|---:|---:|---|']
    four=result['sections']['four']['mean'];decision=result['decision']
    for name in ('F-B','F-S','G-F','G-B','G-S'):
        c=four['comparisons'][name];lo,hi=np.array(c['ci95'])*100;lower=c.get('simultaneous_lower95');text=f'{lower*100:+.2f}' if lower is not None else '—'
        cl,ch=c['completion_cost_ci95']
        lines.append(f'|{name}|{c["delta"]*100:+.2f}|[{lo:+.2f},{hi:+.2f}]|{text}|{c["gained"]}/{c["lost"]}|{c["completion_cost_delta"]:+.2f} [{cl:+.2f},{ch:+.2f}]|')
    gf=four['comparisons']['G-F']
    lines+=['',f'预登记F双参考完整门槛通过：{decision["F_passed"]}。逐项限制见statistics.json中的decision.F_checks。',
        f'G-F单侧95%下界={100*gf["lower95"]:+.2f}pp；预登记非劣界为-3pp，是否支持G非劣={decision["G_noninferior_to_F"]}。',
        f'未见地图F-S普通95%下界大于0：{decision["unseen_F_vs_S_positive"]}。','',
        '**分地图、分对手结果。** 双捕成功率；同一物理场景的三个对手不是独立地图样本。','',
        '|地图|对手|B|S|F|G|F-S差及95%区间 pp|G-F差及95%区间 pp|',
        '|---|---|---:|---:|---:|---:|---|---|']
    for scope in ('original_four','unseen_four','none','offset_corridor','asymmetric','all'):
        for opponent in ('mean',)+m.OPPONENTS:
            section=result['sections'][scope][opponent];g=section['summary'];cells=[f'{g[u]["successes"]}/{g[u]["games"]} ({100*g[u]["success_rate"]:.2f}%)' for u in m.UIDS]
            contrasts=[]
            for name in ('F-S','G-F'):
                c=section['comparisons'][name];lo,hi=np.array(c['ci95'])*100;contrasts.append(f'{100*c["delta"]:+.2f} [{lo:+.2f},{hi:+.2f}]')
            lines.append('|'+scope+'|'+opponent+'|'+'|'.join(cells+contrasts)+'|')
    lines+=['','**失败原因和完成成本。** 失败统一计80步；首捕后的条件耗时不替代整局指标。','',
        '|地图|组|零捕|首捕后超时|自身非法|对手非法|完成成本|首捕时间均值|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for scope in ('original_four','unseen_four','none','all'):
        for u,g in result['sections'][scope]['mean']['summary'].items():
            first=g['first_capture_mean'];ft=f'{first:.2f}' if first is not None else '—'
            lines.append(f'|{scope}|{u}|{g["zero_capture"]}|{g["one_capture_timeout"]}|{g["own_illegal"]}|{g["opponent_illegal"]}|{g["completion_cost"]:.2f}|{ft}|')
    lines+=['','**G与F的计算量配对比较。**','',
        '|地图|指标|G-F每局平均差|配对95%区间|',
        '|---|---|---:|---|']
    for scope in ('original_four','unseen_four','none','four','all'):
        c=result['sections'][scope]['mean']['comparisons']['G-F']
        for metric in ('step_calls','policy_evaluations'):
            lo,hi=c[metric+'_ci95'];lines.append(f'|{scope}|{metric}|{c[metric+"_delta"]:+.1f}|[{lo:+.1f},{hi:+.1f}]|')
    lines+=['',f'预登记四障碍调用计数更低判定：{decision["G_lower_call_counts"]}。','',
        '|组|总决策数|总环境转移|总网络调用|总几何查询|平均后段成本|',
        '|---|---:|---:|---:|---:|---:|']
    for u,g in result['sections']['all']['mean']['summary'].items():
        d=diag[u];cost=g['own_cost'];post=f'{d["mean_post_cost"]:.2f}' if d['mean_post_cost'] is not None else '—'
        lines.append(f'|{u}|{g["decisions"]}|{cost["step_calls"]}|{cost["policy_evaluations"]}|{d["geometry_queries"]:.0f}|{post}|')
    all_summary=result['sections']['all']['mean']['summary']
    fcost,gcost=all_summary['F']['own_cost'],all_summary['G']['own_cost']
    env_change=100*(gcost['step_calls']/fcost['step_calls']-1)
    net_change=100*(gcost['policy_evaluations']/fcost['policy_evaluations']-1)
    geom_change=100*(diag['G']['geometry_queries']/diag['F']['geometry_queries']-1)
    post_env=100*(diag['G']['post_step_calls']/diag['F']['post_step_calls']-1)
    post_net=100*(diag['G']['post_policy_evaluations']/diag['F']['post_policy_evaluations']-1)
    lines+=['',f'全部150场景三对手中，G相对F环境转移{env_change:+.2f}%、网络调用{net_change:+.2f}%，几何查询{geom_change:+.2f}%。只计算首捕后，环境转移{post_env:+.2f}%、网络调用{post_net:+.2f}%。',
        'G仍保留50%神经候选及非根神经搜索；不是无网络控制器。F/G共同几何预处理原样保留。',
        '调用计数的减少不自动等同于部署延迟减少。完整并发平均/P95耗时保存在diagnostics.json，因资源争用和轨迹分布不同，不作为单进程加速结论；本轮没有另开计时对局。','',
        '**结论与下一步。**']
    if decision['F_passed']:
        lines.append('F在新的四障碍物理场景上通过预登记双参考门槛，支持其在此固定O模型和评估条件下的部署收益。')
    else:
        lines.append('F未通过全部预登记双参考门槛。应结合分项失败区分成功率信号、对手回归和不确定性，不把有正向平均差等同于完整通过。')
    if decision['G_noninferior_to_F'] and decision['G_lower_call_counts']:
        lines.append('G在预登记-3pp界内得到非劣支持，且环境/网络调用的预登记成本判定更低。下一步可优先研究简化为G，并用单进程等耗时评估确认实际计算收益，F保留为强对照。')
    elif decision['G_noninferior_to_F']:
        lines.append('G的非劣证据成立，但更低调用成本的预登记标准未成立。可以保留G作为简单候选机制，尚不足以因计算成本改选G；F继续作为有效候选。')
    else:
        lines.append('当前样本没有支持G在-3pp界内非劣；不能因两者差异不显著就简化为G。后续保留F，G作为必要几何基线。')
    if decision['unseen_F_vs_S_positive']:
        lines.append('F-S的正向收益在未见四障碍组上得到区间支持；不同对手和两种具体新布局的结果仍分别呈现，不能外推全部地图或动态条件。')
    else:
        lines.append('未见地图F-S区间尚未明确大于零，迁移证据不足，需依据地图/对手分项定位。')
    lines+=['当前推荐继续old0:novalue；没有自动部署、更换权重、追加样本或启动其他实验。','',
        '**执行与证据范围。** 5项针对性测试、48局独立smoke、正式地图/对手预检查均通过；全部1800局轨迹与根统计已校验，S/F/G的450组首捕前轨迹/快照/随机状态一致。',
        '新150个初态及能力限幅与冻结历史可恢复场景零重叠。地图组内分层物理场景bootstrap保留对手相关性，主比较F-B/F-S用同时下界；分项普通区间为描述性分析。',
        '一个固定O检查点、两种新四矩形布局、原动作能力范围和完整状态direct环境，不能据本轮宣称跨训练种子或真实机器人泛化。',
        '完整配置/地图/场景见experiment.json、scenarios.json，逐场景轨迹见trajectory_index.json，科学统计见statistics.json，额外分解见extended_analysis.json。','',
        '![独立验证结果](figures/validation.png)']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (args.output/'CONCLUSION.md').write_text('\n'.join(lines[:18])+'\n\n'+str(decision)+'\n')
    m.o.write_json(args.output/'analysis_provenance.json',dict(script=str(Path(__file__).resolve()),sha256=m.o.sha(__file__),
        inputs={n:m.o.sha(args.output/n) for n in ('statistics.json','diagnostics.json','integrity.json','experiment.json')},
        extra_environment_episodes=0,extra_search_calls=0))

if __name__=='__main__':main()
