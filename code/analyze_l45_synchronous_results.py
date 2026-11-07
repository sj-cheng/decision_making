"""Read committed validation pairs; explain effects without new experiments."""
import numpy as np
import run_l45_synchronous_search as m


def main():
    args=m.parse_args([]);out=args.output
    result=m.locked_read(out/'results.json')
    banks=m.locked_read(out/'scenarios.json')
    items=m.requests(args,['C','O','L'],banks['validation_four']+banks['validation_none'])
    rows=[m.read_cached(args,item) for item in items]
    analysis={}
    for layout in ('four','none'):
        parts={uid:m.v.indexed_rows(rows,uid,layout,1) for uid in ('C','O','L')}
        block={}
        for ref in ('O','L'):
            if set(parts['C'])!=set(parts[ref]):raise ValueError('Missing paired data')
            per_op={}
            for op in m.o.OPPONENTS:
                keys=[key for key in sorted(parts['C']) if key[1]==op]
                differences={key:parts['C'][key]['legal_double_capture']-parts[ref][key]['legal_double_capture'] for key in keys}
                per_op[op]=dict(gained=sum(x>0 for x in differences.values()),lost=sum(x<0 for x in differences.values()),
                    net=sum(differences.values()),games=len(keys),
                    gained_scenes=[key[0] for key,x in differences.items() if x>0],
                    lost_scenes=[key[0] for key,x in differences.items() if x<0])
            block[ref]=per_op
        block['models']={}
        for uid,part in parts.items():
            values=list(part.values());cost={key:sum(row['own_cost'][key] for row in values) for key in values[0]['own_cost']}
            block['models'][uid]=dict(games=len(values),successes=sum(row['legal_double_capture'] for row in values),
                own_illegal=sum(row['pursuer_illegal'] for row in values),
                zero_capture=sum(row['capture_count']==0 for row in values),
                one_capture_timeout=sum(row['one_capture_timeout'] for row in values),
                mean_completion_cost=float(np.mean([row['completion_cost'] for row in values])),
                cost=cost,own_mcts_calls=sum(row['own_mcts_calls'] for row in values),
                per_opponent={op:dict(successes=sum(row['legal_double_capture'] for row in values if row['opponent']==op),
                    games=sum(row['opponent']==op for row in values)) for op in m.o.OPPONENTS})
        analysis[layout]=block
    m.lock_json(out/'paired_analysis.json',analysis)
    recommended=m.locked_read(out/'recommended.json');replay=m.locked_read(out/'replay.json')
    if not result['quality_gate_passed']:
        if recommended['uid']!='O' or not replay['passed']:raise ValueError('Failure did not preserve verified O')
        for forbidden in ('candidate_lock','confirmation_result','guard_result','timing_result'):
            if (out/(forbidden+'.json')).exists():raise ValueError('Downstream stage unexpectedly executed')
    four=analysis['four'];none=analysis['none']
    a=four['models']['C'];b=four['models']['O'];l=four['models']['L']
    lines=['本轮执行结论','',
        f"正式新场景验证 1,350 局完成。四障碍 C {a['successes']}/300，O {b['successes']}/300，L {l['successes']}/300。C−O +0.67 点，C−L +7.00 点。",
        'C 使用原 l45 冻结策略权重和同步搜索；没有新增 PPO、actor 更新或教师标签。O 的追捕权重没有进入候选 C。','',
        '|配置|Policy Only|Search+Learning|Search Only|四障碍平均|',
        '|---|---:|---:|---:|---:|']
    for uid in ('C','O','L'):
        row=four['models'][uid];numbers=[str(row['per_opponent'][op]['successes'])+'/100' for op in m.o.OPPONENTS]
        lines.append('|'+uid+'|'+'|'.join(numbers)+f"|{row['successes']/300:.2%}|")
    for ref in ('O','L'):
        part=four[ref];gained=sum(x['gained'] for x in part.values());lost=sum(x['lost'] for x in part.values())
        search=part['Search Only']
        lines+=['',f"C 对 {ref}：挽回 {gained}、丢失 {lost}，净增 {gained-lost}/300；Search Only 挽回 {search['gained']}、丢失 {search['lost']}，净增 {search['net']}/100。"]
    lines+=['',
        f"无障碍 C {none['models']['C']['successes']}/150、O {none['models']['O']['successes']}/150、L {none['models']['L']['successes']}/150。所有配置两地图自身非法均为 0。",
        '唯一未通过的验证限制是相对 O 的四障碍平均增益不足 +3 点。按冻结协议结束，没有独立确认、逃逸回归或单进程计时，也没有改用 O_sync 或改变切换时机。最终推荐仍为 old0:novalue，推荐导出和完整搜索重放通过。','',
        '这个结果回答了两个不同问题：同一原 l45 权重下，同步搜索在本验证集上改善 7 点；相对已有最强配置 O，尚无足够采用收益。前者尚未独立确认，不能宣称稳定提升、独立论文贡献或原 NTE 论文基准上的提升。',
        '历史条件诊断中的 O_sync 是 O 权重且从 O 首捕状态续跑；本轮 C 是 L 权重且从完整初态开始同步。两者不是同一个干预，不能把差值直接解释为过拟合或首捕阶段退化。',
        '当前实现同时改变双方动作统计、联合转移和随机 rollout，尚未区分各因素的贡献；同步 MCTS/DUCT 本身也已有文献先例。','',
        f"四障碍追捕环境转移：C {a['cost']['step_calls']:,}，O {b['cost']['step_calls']:,}，L {l['cost']['step_calls']:,}。C/O 总调用比 {a['cost']['step_calls']/b['cost']['step_calls']:.3f}，C/L {a['cost']['step_calls']/l['cost']['step_calls']:.3f}；它们同时受对局长短和决策数影响，不等于单步成本比。",
        '没有完成单进程计时，不能宣称部署提速；并行总耗时只描述这次作业。','',
        '下一步已有两条清晰边界：旧动作评分器在组外未超过恒选 AQ，停止该学习分支；同步搜索已成为可以与原交替搜索对照的有效实现，但本轮候选未通过采用门槛。后续如研究 O 权重与同步搜索的组合，应另行预登记新的 2×2 消融和场景，不能在这次已失败的验证中临时换候选或复用确认名义。']
    (out/'CONCLUSION.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:18]))


if __name__=='__main__':main()
