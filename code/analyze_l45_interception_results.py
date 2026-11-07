"""Explain the completed interception pilot without additional rollouts."""
import json
from pathlib import Path
import numpy as np
import run_l45_interception_guidance as m

def analyze(args):
    banks=m.prepare(args)
    if args.smoke:raise ValueError('No effectiveness analysis for smoke')
    selection=m.locked_read(args.output/'selection.json')
    items=m.requests(args,m.UIDS,banks['pilot_four'])
    records={u:{} for u in m.UIDS};rows={u:[] for u in m.UIDS}
    for item in items:
        row,detail=m.read_episode(args,item)
        records[item['uid']][row['episode_hash']]=detail;rows[item['uid']].append(row)
    output={}
    for u in m.UIDS:
        branch=[];successful=[];post_cost=[];fail_dist=[];last_dist=[];first_times=[];latencies=[]
        root_stats=[];post=[];positive=0;available=0;first_magnitude=[];first_alignment=[]
        for row in rows[u]:
            detail=records[u][row['episode_hash']];first=detail['first_capture']
            if first is None or not first['branch_available']:continue
            branch.append(row);first_times.append(first['time'])
            post_cost.append(row['completion_cost']-first['time'])
            if row['legal_double_capture']:successful.append(row['capture_time']-first['time'])
            start=first['next_decision']
            flags=first['state'].reshape(-1)[9:13]
            active=np.flatnonzero(flags>.5);assert len(active)==2
            distances=[]
            for step in detail['trajectory'][start:]:
                state=step['next_state'].reshape(-1)
                distances.append(float(np.linalg.norm(state[2*active[0]:2*active[0]+2]-state[2*active[1]:2*active[1]+2])))
            if not row['legal_double_capture'] and distances:fail_dist.append(min(distances));last_dist.append(distances[-1])
            post.extend(detail['roots'][start:]);latencies.extend(v[0] for v in row['decision_latency_ms'][start:])
            first_root=detail['roots'][start]
            step=detail['trajectory'][start];state=step['state'].reshape(-1)
            robot=int(active[1]);evader=int(active[0])
            action=step['action'].reshape(-1)[2*robot:2*robot+2]
            upper=np.asarray(detail['scenario']['action_lims'])[2*robot:2*robot+2,1]
            first_magnitude.append(float(np.max(np.abs(action)/upper)))
            delta=state[2*evader:2*evader+2]-state[2*robot:2*robot+2]
            first_alignment.append(float(action@delta/max(1e-12,np.linalg.norm(action)*np.linalg.norm(delta))))
            root_stats.append(first_root['guidance'])
            table=first_root['routes']
            available+=int(table.size>0)
            positive+=int(table.size>0 and np.any((table[:,8]>.5)&(table[:,7]>=0)))
        sums={}
        for root in post:
            for k,v in root['guidance'].items():sums[k]=sums.get(k,0)+v
        def mean(x):return float(np.mean(x)) if x else None
        output[u]=dict(takeover_scenes=len(branch),post_capture_successes=len(successful),
            post_capture_success_rate=len(successful)/len(branch) if branch else None,
            successful_second_capture_mean_steps=mean(successful),mean_first_capture_step=mean(first_times),
            mean_post_capture_cost_failures_use_remaining_time=mean(post_cost),
            post_capture_timeouts=sum(r['one_capture_timeout'] for r in branch),
            timeouts_with_first_by_40=sum(r['one_capture_timeout'] and r['first_capture_time']<=40 for r in branch),
            failed_minimum_distance_mean=mean(fail_dist),failed_final_distance_mean=mean(last_dist),
            first_decision_route_available=available,first_decision_positive_time_margin=positive,
            post_capture_calls=len(post),post_capture_mean_parallel_latency_ms=mean(latencies),
            env_steps_per_post_capture_call=sum(z['cost']['step_calls'] for z in post)/len(post) if post else None,
            policy_evaluations_per_post_capture_call=sum(z['cost']['policy_evaluations'] for z in post)/len(post) if post else None,
            geometry_queries_per_post_capture_call=sums.get('geometry_queries',0)/len(post) if post else None,
            graph_build_ms_per_post_capture_call=sums.get('graph_build_ms',0)/len(post) if post else None,
            plan_ms_per_post_capture_call=sums.get('plan_ms',0)/len(post) if post else None,
            guided_rollout_fraction=sums.get('guided_rollouts',0)/sums['rollouts'] if sums.get('rollouts') else None,
            guided_steps=sums.get('guided_steps',0),max_guided_prefix=max([z['guidance'].get('max_guided_prefix',0) for z in post],default=0),
            guided_root_candidate_fraction=sums.get('root_source_2',0)/sums['root_candidates'] if sums.get('root_candidates') else None,
            selected_guided_candidates=sum(z['guidance'].get('selected_source')==2 for z in post),
            selected_guided_fraction=sum(z['guidance'].get('selected_source')==2 for z in post)/len(post) if post else None,
            first_action_mean_normalized_magnitude=mean(first_magnitude),
            first_action_mean_direct_pursuit_cosine=mean(first_alignment),
            proposal_fallbacks=sums.get('proposal_fallback',0),noise_fallbacks=sums.get('noise_fallback',0),
            rollout_fallbacks=sums.get('rollout_fallback',0),rollout_goals_reached=sums.get('rollout_goal_reached',0))
    # S and P have the same initial takeover state. Use P's deterministic
    # hypotheses against S's actual future, without selecting successful cases.
    forecast={}
    for scene,baseline in records['S'].items():
        first=baseline['first_capture']
        if first is None or not first['branch_available']:continue
        start=first['next_decision'];guide=records['P'][scene]['roots'][start]['routes']
        if not guide.size:continue
        evader=int(np.flatnonzero(first['state'].reshape(-1)[9:11]>.5)[0])
        for t in (1,2,4,8,12,16,24):
            if start+t>len(baseline['trajectory']):continue
            forecasts=guide[guide[:,1]==t,2:4]
            if not len(forecasts):continue
            real=baseline['trajectory'][start+t-1]['next_state'].reshape(-1)[2*evader:2*evader+2]
            errors=np.linalg.norm(forecasts-real,axis=1)
            forecast.setdefault(t,[]).append((float(errors.min()),float(errors.mean())))
    coverage={str(t):dict(n_states=len(v),mean_best_of_three_error=float(np.mean(np.array(v)[:,0])),
        median_best_of_three_error=float(np.median(np.array(v)[:,0])),mean_route_error=float(np.mean(np.array(v)[:,1]))) for t,v in forecast.items()}
    scenes=sorted(records['P'])
    row_index={u:{r['episode_hash']:r for r in rows[u]} for u in m.UIDS}
    difference=np.array([row_index['PR'][k]['legal_double_capture']-row_index['P'][k]['legal_double_capture'] for k in scenes],float)
    rng=np.random.default_rng(54020261024)
    samples=difference[rng.integers(0,len(scenes),(10000,len(scenes)))].mean(1)
    pr_vs_p=dict(delta=float(difference.mean()),gained=int((difference>0).sum()),lost=int((difference<0).sum()),
        ci95=np.quantile(samples,[.025,.975]).tolist(),exploratory=True)
    m.o.write_json(args.output/'phase_analysis.json',dict(groups=output,forecast_coverage_against_S=coverage,PR_minus_P=pr_vs_p,
        forecast_note='Best-of-up-to-three is an optimistic descriptive coverage measure, not a deployable selector.',
        parallel_latency_note='Parallel contention; use timing_result.json for deployment latency.'))
    lines=['','首捕后的同状态结果：','',
        'S/P/R/PR共享相同首捕状态；B使用自身形成的首捕状态，其条件统计仅作描述。',
        '',
        '|配置|实际接管局数|接管后成功|成功收尾步数均值|超时且首捕≤40步|后段成本（失败计剩余时间）|',
        '|---|---:|---:|---:|---:|---:|']
    for u in m.UIDS:
        d=output[u];time=d['successful_second_capture_mean_steps'];cost=d['mean_post_capture_cost_failures_use_remaining_time']
        time_text=f'{time:.2f}' if time is not None else '—';cost_text=f'{cost:.2f}' if cost is not None else '—'
        lines.append(f'|{u}|{d["takeover_scenes"]}|{d["post_capture_successes"]}|{time_text}|' +
            f'{d["timeouts_with_first_by_40"]}|{cost_text}|')
    lines+=['','P/R/PR引导实际使用情况：']
    for u in ('P','R','PR'):
        d=output[u]
        lines.append(f'{u}：首个接管决策路线可用{d["first_decision_route_available"]}/{d["takeover_scenes"]}；'
            f'存在非负时间差的拦截方案{d["first_decision_positive_time_margin"]}局；'
            f'后段实际选中几何候选{d["selected_guided_candidates"]}/{d["post_capture_calls"]}次；'
            f'引导模拟前缀最长{d["max_guided_prefix"]:.0f}步。')
    lines+=['','路线假设对S真实后续的覆盖（全部可计算场景；三条中最小误差是乐观诊断指标）：']
    for t in (4,8,16):
        if str(t) in coverage:
            c=coverage[str(t)];lines.append(f'第{t}步，n={c["n_states"]}，最佳路线位置误差均值{c["mean_best_of_three_error"]:.2f}，所有路线误差均值{c["mean_route_error"]:.2f}。')
    lines+=['','本轮解释与下一步：']
    for u in ('P','R','PR'):
        comp=selection['comparisons'][u]['S'];lines.append(f'{u}对S净增{comp["net_gained"]}场，挽回{comp["gained"]}、丢失{comp["lost"]}。')
    lines.append(f'PR对P仅净增{pr_vs_p["delta"]*100:.0f}点，挽回{pr_vs_p["gained"]}、丢失{pr_vs_p["lost"]}；'
        f'探索性95%区间[{pr_vs_p["ci95"][0]*100:+.0f},{pr_vs_p["ci95"][1]*100:+.0f}]点。新增引导模拟相对P的收益尚不清晰。')
    lines.append(f'相同首捕局面的第一步，S动作分量归一化幅度均值{output["S"]["first_action_mean_normalized_magnitude"]:.3f}，'
        f'P为{output["P"]["first_action_mean_normalized_magnitude"]:.3f}；'
        f'与直接追向对手方向的余弦均值分别为{output["S"]["first_action_mean_direct_pursuit_cosine"]:.3f}、'
        f'{output["P"]["first_action_mean_direct_pursuit_cosine"]:.3f}。')
    lines.append('下一轮优先补充满速直接追踪、无预测的绕障碍追踪候选，以及满速神经候选对照，检验预测拦截相对这些简单机制的额外收益。')
    if selection['passed']:
        lines.append(f'开发门槛通过，下一阶段锁定{selection["selected"]}；下一轮用独立场景验证并补充三个对手与等耗时比较。当前推荐仍为old0:novalue。')
    else:
        best=max(('P','R','PR'),key=lambda u:selection['summary'][u]['successes'])
        lines.append(f'本版本未通过开发门槛，停止继续投入该参数配置。最高成功数来自{best}；'
            '结合路线覆盖误差及挽回/丢失分析，区分候选几何信息不足与模拟路线失配。覆盖误差本身不证明某一项就是失败原因。')
    (args.output/'PHASE_ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    report=args.output/'REPORT.md'
    if report.exists():
        text=report.read_text();marker='\n首捕后的同状态结果：'
        if marker in text:text=text.split(marker)[0]
        timing_path=args.output/'timing_result.json'
        if timing_path.exists():
            timing=m.locked_read(timing_path)
            table=['单进程计时：30个新场景，三组交错执行；另3局预热不计入统计。','',
                '|配置|单步均值ms|单步P95 ms|首捕后均值ms|首捕后P95 ms|整局追捕计算s|整局双方＋推进s|',
                '|---|---:|---:|---:|---:|---:|---:|']
            for u,v in timing.items():
                table.append(f'|{u}|{v["mean_step_ms"]:.2f}|{v["p95_step_ms"]:.2f}|{v["mean_post_capture_ms"]:.2f}|'
                    f'{v["p95_post_capture_ms"]:.2f}|{v["mean_pursuit_episode_ms"]/1000:.3f}|{v["mean_joint_episode_ms"]/1000:.3f}|')
            chosen=timing[selection['selected']]
            table+=['']
            for reference in ('B','S'):
                base=timing[reference]
                step=(chosen['mean_step_ms']/base['mean_step_ms']-1)*100
                episode=(chosen['mean_pursuit_episode_ms']/base['mean_pursuit_episode_ms']-1)*100
                table.append(f'{selection["selected"]}相对{reference}：单步均值{step:+.1f}%，整局追捕计算时间{episode:+.1f}%。')
            text=text.replace('单进程计时：\n'+str(timing),'\n'.join(table))
        report.write_text(text+'\n'.join(lines)+'\n')
    print(json.dumps(dict(selection=selection,phase_analysis=output,coverage=coverage),ensure_ascii=False,indent=2))

if __name__=='__main__':analyze(m.parse_args())
