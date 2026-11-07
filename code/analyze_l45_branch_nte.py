"""Post-process committed branch-NTE trajectories; no new decisions or games."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import run_l45_branch_nte as run
from verify_l45_occlusion_scenarios import visible as independent_visible, blocked


def analyze(output):
    run.check_frozen(output)
    status=json.loads((output/'status.json').read_text())
    if not status.get('terminal'):raise RuntimeError('Wait for the registered pipeline to stop before final analysis')
    results=json.loads((output/'teacher_check/results.json').read_text())
    tasks=json.loads((output/'teacher_check/requests.json').read_text())
    if len(results)!=192 or len(tasks)!=192:raise RuntimeError('Incomplete paired prerequisite')
    frames=0;bad_uniform=0;illegal=0;visibility_mismatches=0;motion_mismatches=0;covered=0;root_decisions=0
    records=[];chosen=[]
    for row in results:
        task=next(t for t in tasks if t['key']==row['key']);run.cached(task,output)
        saved=run.load(row['episode_path']);packet,_=run.inputs(task['entry']);boxes=packet['map'];speed=packet['motion']['pursuer_axis_limit']
        zero=0;travel=0.;sizes=[];initial_covered=None;reacquisition=None;latencies=[];root_zero_prior=[]
        for k,frame in enumerate(saved['trajectory']):
            frames+=1;state,nxt,action=frame['state'],frame['next_state'],frame['action'];o=frame['observation']
            expected=independent_visible(state[4:6,0].tolist(),state[:2,0].tolist(),boxes)
            visibility_mismatches+=int(expected!=(o.target is not None))
            for indices,limit in ((slice(0,2),1.),(slice(4,6),speed)):
                raw=action[indices,0];start=state[indices,0];end=nxt[indices,0]
                motion_mismatches+=int(np.max(np.abs(end-start-raw))>4e-6 or np.max(np.abs(raw))>limit+1e-10)
                illegal+=int(blocked(start.tolist(),end.tolist(),boxes))
            zero+=int(np.linalg.norm(action[4:6,0])<1e-10);travel+=float(np.linalg.norm(action[4:6,0]));sizes.append(frame['own']['support_cells'])
            covered+=int(frame['support_covered']);latencies.append(frame['own']['latency_ms'])
            roots=frame['own']['root']
            if roots:
                root_decisions+=1
                bad_uniform+=int(np.ptp([r['prior'] for r in roots])>1e-12)
                root_zero_prior.extend(r['prior'] for r in roots if np.linalg.norm(r['action'])<1e-10)
            # Include terminal post-transition sensing, absent from pre-action
            # logs if capture ends the episode immediately after rediscovery.
            if reacquisition is None:
                if expected:reacquisition=k
                elif independent_visible(nxt[4:6,0].tolist(),nxt[:2,0].tolist(),boxes):reacquisition=k+1
        values=dict(key=row['key'],history_id=row['history_id'],family=row['family'],world=row['world'],stream=row['stream'],method=row['method'],
            capture=row['capture'],completion_cost=row['completion_cost'],failure=row['failure'],steps=len(saved['trajectory']),
            zero_steps=zero,zero_fraction=zero/max(1,len(saved['trajectory'])),travel=travel,
            support_initial=sizes[0] if sizes else None,support_final=sizes[-1] if sizes else None,
            reappearance_time=reacquisition,reappearance_cost=reacquisition if reacquisition is not None else 80-row['handoff_step'],
            mean_ms=float(np.mean(latencies)) if latencies else None,mean_zero_prior=float(np.mean(root_zero_prior)) if root_zero_prior else None)
        records.append(values)
    if bad_uniform or illegal or visibility_mismatches or motion_mismatches:
        raise RuntimeError(f'Independent audit issue: prior={bad_uniform}, collision={illegal}, sensor={visibility_mismatches}, motion={motion_mismatches}')
    with (output/'paired_trajectory_metrics.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    summaries={}
    for m in ('T-first','E-first'):
        rs=[r for r in records if r['method']==m]
        summaries[m]=dict(requests=len(rs),captures=sum(r['capture'] for r in rs),completion_cost=float(np.mean([r['completion_cost'] for r in rs])),
            zero_action_fraction=float(sum(r['zero_steps'] for r in rs)/sum(r['steps'] for r in rs)),
            mean_travel=float(np.mean([r['travel'] for r in rs])),mean_support_initial=float(np.mean([r['support_initial'] for r in rs])),
            mean_support_final=float(np.mean([r['support_final'] for r in rs])),reacquired=sum(r['reappearance_time'] is not None for r in rs),
            mean_reappearance_cost=float(np.mean([r['reappearance_cost'] for r in rs])))
    budget=json.loads((output/'budget.json').read_text())
    diagnostics=dict(passed=True,requests=192,frames=frames,all_search_roots_uniform_after_deduplication=True,
        searched_roots=root_decisions,independent_sensor_mismatches=visibility_mismatches,independent_motion_mismatches=motion_mismatches,
        independent_swept_collisions=illegal,true_cell_coverage=covered/max(1,frames),summaries=summaries,budget=budget,
        no_new_environment_transitions=True,metric_correction='reappearance includes sensor result after final transition; raw records preserved',
        geometry_counter='original path/legality API calls only; grid predicates accounted separately by support-update time and active-cell count')
    run.put(output/'trajectory_diagnostics.json',diagnostics)
    text='# 配对轨迹与实现审计\n\n'
    text+=f'只读取192条已提交续跑，共{frames}次转移，没有新运行搜索、对局或训练。校正版本所有{root_decisions}个搜索根节点的合并候选先验均匀，独立传感器、动作尺度及线段碰撞检查通过。\n\n'
    text+='|方法|捕获/96|完成成本|重新发现/96|零动作比例|平均路程|初始/结束支持格数|\n|---|---:|---:|---:|---:|---:|---|\n'
    for m,s in summaries.items():text+=f'|{m}|{s["captures"]}/96|{s["completion_cost"]:.3f}|{s["reacquired"]}/96|{s["zero_action_fraction"]:.2%}|{s["mean_travel"]:.2f}|{s["mean_support_initial"]:.1f}/{s["mean_support_final"]:.1f}|\n'
    text+='\n初批存在“合并前槽位均匀”实现错误，已完整隔离于原目录；本报告只统计修正后的独立记录。修正恢复了预登记的均匀先验，没有改候选数、网格、奖励、场景或随机种子。\n'
    gate=json.loads((output/'teacher_check/gate.json').read_text())
    if not gate['passed']:
        text+='\n教师第一步动作筛选没有通过，因此没有训练M/H。这个结果只能评价当前共同支持、候选、搜索和统一续跑组合下的一步动作干预；尚不能判断分支条件神经网络是否有用，也不能断言全信息专家持续控制无效。\n'
    if sum(s['captures'] for s in summaries.values())==0:
        text+='\n两组均未捕获，捕获差的配对bootstrap区间退化为[0,0]。这是所有观测结果相同造成的经验区间，不能解释为证明两方法真实效果相等。全失败的完成成本均由剩余时间定义，成本差同样无法区分动作质量。\n'
    text+='\n支持格包含真值并不意味着估计概率准确。支持面积增长、零动作和徘徊可以作为退化线索，但这些轨迹没有单独隔离均匀支持采样、候选设计、模拟对手与回报各自的责任。历史高捕获率来自不同估计和搜索实现，不能作为同条件因果对照。\n'
    text+='\n耗时来自固定模拟预算的并行续跑，不能视为单进程50ms性能。本阶段也没有执行H或G的实际部署比较。几何查询计数仅对应原路径/合法性API，不包括网格内部全部几何谓词；网格额外成本由更新耗时与活跃格数说明。\n'
    text+='\n重新发现指标从保存的前后状态重建，包含捕获终止那一步的传感器结果；原始记录未更改。逐局数据见paired_trajectory_metrics.csv，完整核验见trajectory_diagnostics.json。\n'
    (output/'ANALYSIS.md').write_text(text)
    plot(output,results)
    report=output/'REPORT.md';base=report.read_text()
    marker='\n## 轨迹与实现核验补充\n'
    if marker not in base:report.write_text(base+marker+'\n详见[ANALYSIS.md](ANALYSIS.md)、逐场景指标paired_trajectory_metrics.csv和trajectory_diagnostics.json。初批错误已经隔离，本目录是修正先验后的结果。\n')
    manifest=output/'DELIVERY_SHA256.json'
    value=json.loads(manifest.read_text())
    value['files']={str(p.relative_to(output)):run.sha(p) for p in output.rglob('*') if p.is_file() and p.name not in ('run.lock','DELIVERY_SHA256.json') and '.tmp' not in p.name}
    value['analysis_source']={str(Path(__file__).resolve()):run.sha(__file__)};run.put(manifest,value)
    return diagnostics


def plot(output,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.backends.backend_pdf import PdfPages
    # A complete trajectory atlas, chosen by fixed history ID, never filtered
    # for success. A/B and both streams are retained in the index CSV.
    histories=sorted({r['history_id'] for r in results})
    with PdfPages(output/'figures/teacher_trajectory_atlas.pdf') as pdf:
        for history in histories:
            subset=[r for r in results if r['history_id']==history]
            fig,axes=plt.subplots(2,4,figsize=(15,8))
            for ax,row in zip(axes.ravel(),sorted(subset,key=lambda r:(r['world'],r['stream'],r['method']))):
                saved=run.load(row['episode_path']);trace=saved['trajectory'];entry=saved['task']['entry'];packet,_=run.inputs(entry)
                for b in packet['map']:ax.add_patch(Rectangle((b[0][0],b[1][0]),b[0][1]-b[0][0],b[1][1]-b[1][0],color='0.4'))
                positions=np.array([f['state'][:,0] for f in trace]+[saved['final']['state'][:,0]])
                ax.plot(positions[:,4],positions[:,5],'b-',lw=1,label='pursuer');ax.plot(positions[:,0],positions[:,1],'r-',lw=1,label='evader (scoring only)')
                ax.scatter(positions[0,4],positions[0,5],c='blue',s=12);ax.scatter(positions[0,0],positions[0,1],c='red',s=12)
                ax.set(xlim=(-10,10),ylim=(-10,10),aspect='equal',title=f'{row["world"]} flow{row["stream"]} {row["method"]}; cap={row["capture"]}')
            fig.suptitle(history,fontsize=9);fig.tight_layout();pdf.savefig(fig);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=run.ROOT/'current/l45_branch_nte_corrected_20261025');args=parser.parse_args()
    print(json.dumps(analyze(args.output.resolve()),ensure_ascii=False,indent=2))
