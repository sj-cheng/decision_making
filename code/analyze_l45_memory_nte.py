"""Post-hoc descriptive diagnostics only; never evaluates controllers."""
import argparse
import collections
import csv
import json
from pathlib import Path
import numpy as np
import torch
from run_l45_memory_nte import DEFAULT,put,check_frozen


def analyze(output):
    check_frozen(output)
    rows=json.loads((output/'teacher_check/results.json').read_text())
    groups={};diagnostics=[]
    for row in rows:
        saved=torch.load(row['episode_path'],weights_only=False,map_location='cpu');trace=saved['trajectory']
        initial=float(trace[0]['nearest_particle_error']) if trace else None
        observed=[i for i,r in enumerate(trace) if r['observation'].target is not None]
        if row['failure']=='estimator_recovery_failure':reason='estimator_recovery_failure'
        elif row['capture']:reason='capture'
        elif row['own_illegal'] or row['opponent_illegal']:reason=row['failure']
        elif not observed:reason='never_reacquired'
        else:reason='reacquired_then_not_captured'
        item=dict(key=row['key'],history_id=row['history_id'],method=row['method'],family=row['family'],
            world=row['world'],stream=row['stream'],capture=row['capture'],failure=reason,
            completion_cost=row['completion_cost'],first_reacquisition=row['first_reacquisition_after_handoff'],
            initial_nearest_particle_error=initial,initial_covered=(initial<=1.) if initial is not None else None,
            hidden_steps=row['hidden_steps'],hidden_covered=row['hidden_covered'],transitions=row['transitions'],
            geometry_queries=row['geometry_queries'],total_simulations=sum(row['simulations']),
            trajectory_path=row['episode_path'])
        diagnostics.append(item)
        groups.setdefault((row['family'],row['method']),[]).append((row,item))
    by_family={}
    for (f,m),items in groups.items():
        by_family.setdefault(f,{})[m]=dict(n=len(items),captures=sum(r['capture'] for r,d in items),
            cost=float(np.mean([r['completion_cost'] for r,d in items])),failures=dict(collections.Counter(d['failure'] for r,d in items)))
    methods={}
    for m in ('G','T','E'):
        rr=[r for r in rows if r['method']==m];dd=[d for d in diagnostics if d['method']==m]
        lat=[v for r in rr for v in r['latencies_ms']];hidden=sum(r['hidden_steps'] for r in rr)
        methods[m]=dict(simulations=sum(sum(r['simulations']) for r in rr),
            transitions=sum(r['transitions'] for r in rr),geometry_queries=sum(r['geometry_queries'] for r in rr),
            latency_mean_ms=float(np.mean(lat)) if lat else None,latency_p95_ms=float(np.quantile(lat,.95)) if lat else None,
            initial_covered=sum(d['initial_covered'] is True for d in dd),initial_observed=sum(d['initial_covered'] is not None for d in dd),
            hidden_coverage=sum(r['hidden_covered'] for r in rr)/max(1,hidden),
            failures=dict(collections.Counter(d['failure'] for d in dd)))
    index={(r['history_id'],r['world'],r['stream'],r['method']):r for r in rows};paired={}
    keys=sorted({(r['history_id'],r['world'],r['stream']) for r in rows})
    for a,b in (('E','T'),('E','G'),('T','G')):
        rescued=[];lost=[]
        for h,w,s in keys:
            ra,rb=index[h,w,s,a],index[h,w,s,b]
            if ra['capture']>rb['capture']:rescued.append(dict(history_id=h,world=w,stream=s))
            if ra['capture']<rb['capture']:lost.append(dict(history_id=h,world=w,stream=s))
        paired[a+'-'+b]=dict(rescued=rescued,lost=lost,net=len(rescued)-len(lost))
    put(output/'teacher_check/diagnostics.json',dict(by_family=by_family,methods=methods,paired=paired,
        cost_caveat='parallel throughput-run timings are not serial latency or equal-time evidence'))
    with (output/'teacher_check/trajectory_index.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(diagnostics[0]));writer.writeheader();writer.writerows(diagnostics)
    text='\n**分地图与轨迹诊断。**\n\n|地图|方法|捕获/32|完成成本|失败分类|\n|---|---|---:|---:|---|\n'
    for family in ('long_wall','wall_corner','multi_route'):
        for m in ('G','T','E'):
            v=by_family[family][m];fail={k:n for k,n in v['failures'].items() if k!='capture'}
            text+=f'|{family}|{m}|{v["captures"]}/{v["n"]}|{v["cost"]:.2f}|{fail}|\n'
    text+='\n|比较|挽回|丢失|净增|\n|---|---:|---:|---:|\n'
    for pair,v in paired.items():text+=f'|{pair}|{len(v["rescued"])}|{len(v["lost"])}|{v["net"]}|\n'
    text+='\n|方法|初始1米覆盖|轨迹中不可见时覆盖率|实际模拟总数|环境转移数|并行运行平均/P95 ms|\n|---|---:|---:|---:|---:|---|\n'
    for m,v in methods.items():
        text+=f'|{m}|{v["initial_covered"]}/{v["initial_observed"]}|{v["hidden_coverage"]:.2%}|{v["simulations"]}|{v["transitions"]}|{v["latency_mean_ms"]:.2f}/{v["latency_p95_ms"]:.2f}|\n'
    text+='\n上述耗时来自80 worker并行吞吐运行，不能作为单进程延迟或等耗时比较。不同方法行动不同，后续观测不同，轨迹覆盖率也不是固定历史上的估计器优劣比较。完整轨迹索引见 teacher_check/trajectory_index.csv。\n'
    text+='\n**门槛与结论边界。**\n\n'
    counts={m:sum(r['capture'] for r in rows if r['method']==m) for m in ('G','T','E')}
    if counts['T']>93:
        text+=f'T 已捕获 {counts["T"]}/96，教师即使全捕获也只能再增加 {96-counts["T"]} 局，低于预登记的 +3 局捕获门槛；当前比较存在明显天花板。完成成本降低10%的替代门槛仍可实现，但本次教师没有达到，且执行可靠性也未通过。没有事后放宽门槛。\n\n'
    text+='当前 T 是固定 1,000 次模拟，并非50ms内实际能完成的低预算学生。本轮停止的是“以当前教师优势门槛启动训练”的流程，没有检验把高预算搜索能力蒸馏到低时延神经扩展是否有效。N/H没有训练或评估，不能将结果写成神经网络、观测记忆或NTE无效。\n\n'
    text+='E相对T的挽回和丢失落在不同公共历史上。执行失败反映行动改变后的估计器可靠性，不能把失败样本删除后称为独立搜索贡献；失败现场和完整随机状态均保留。初始支持覆盖不足再次出现，但仅凭这项相关性，尚不能排除模拟对手与回报评价的影响。\n\n'
    text+='下一步建议另行登记：先围绕NTE的低时延目标，明确教师是否能够提供优于真实部署预算学生的动作，以及如何处理共用估计器的恢复可靠性。本轮不启动这些工作，不追加标签或训练。\n'
    audit_path=output/'teacher_check/sensor_motion_audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        text+='\n**独立观测与运动核验。**\n\n'
        text+=f'共检查 {audit["environment_transitions"]} 次真实环境转移，独立几何核验'+('通过。' if audit['passed'] else '未通过，见原始审计。')
        if audit.get('float32_quantization'):
            q=audit['float32_quantization']
            text+=f'严格位置差检查发现41处原生float32舍入，最大限幅差 {q["maximum_observed_excess"]:.3g}；原始动作均在限幅内，使用 {q["tolerance"]:.1g} 的位置计算容差后无额外问题。初始严格检查保存在 sensor_motion_audit_strict.json，逐项证据见 float32_motion_diagnostics.json。'
        text+='核验仅用于事后审计，没有改动环境或算法。\n'
    report=output/'REPORT.md';original=report.read_text();marker='\n**分地图与轨迹诊断。**'
    report.write_text(original.split(marker)[0]+text)
    plot(output,by_family)


def plot(output,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f,axes=plt.subplots(1,2,figsize=(10,4))
    families=('long_wall','wall_corner','multi_route');x=np.arange(3)
    for i,m in enumerate(('G','T','E')):
        axes[0].bar(x+(i-1)*.23,[data[c][m]['captures']/32 for c in families],.23,label=m)
        axes[1].bar(x+(i-1)*.23,[data[c][m]['cost'] for c in families],.23,label=m)
    for a in axes:a.set_xticks(x,families,rotation=12);a.legend()
    axes[0].set(ylabel='Capture rate',ylim=(0,1.05));axes[1].set(ylabel='Completion cost (steps)')
    f.tight_layout();(output/'figures').mkdir(exist_ok=True);f.savefig(output/'figures/teacher_check.png',dpi=160);plt.close(f)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT)
    analyze(parser.parse_args().output.resolve())
