"""Read-only scientific-artifact aggregation for E46; never selects new models."""
import argparse
import collections
import json
import sqlite3
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    db=sqlite3.connect(f'file:{out/"jobs.sqlite"}?mode=ro',uri=True)
    progress=json.loads((out/'progress.json').read_text())
    groups=collections.defaultdict(list);failures=[]
    for stage,status,payload in db.execute('select stage,status,result from jobs where result is not null'):
        row=json.loads(payload)
        if status=='failed':failures.append(dict(stage=stage,error=row.get('error')))
        elif status=='done':groups[stage].append(row)
    tables={}
    for stage,rows in groups.items():
        if not rows or 'summary' not in rows[0]['result']:continue
        cells=collections.defaultdict(lambda:collections.defaultdict(list))
        for row in rows:
            task=row['task'];s=row['result']['summary']
            key=(s['method'],task.get('suite',''),task.get('mode','sim'),task.get('control_value',task.get('fixed')),task['opponent'])
            cells[key][task['group']].append(s)
        table=[]
        for key,maps in sorted(cells.items()):
            units=[v for values in maps.values() for v in values]
            table.append(dict(method=key[0],suite=key[1],mode=key[2],budget=key[3],opponent=key[4],maps=len(maps),
                utility=float(np.mean([np.mean([s['utility'] for s in v]) for v in maps.values()])),
                double=float(np.mean([np.mean([s['double_capture'] for s in v]) for v in maps.values()])),
                completion_cost=float(np.mean([np.mean([s['completion_cost'] for s in v]) for v in maps.values()])),
                decision_cpu_seconds=sum(sum(s.get('cpu_ms',[])) for s in units)/1000.,
                wall_decision_ms=sum(sum(s.get('latencies_ms',[])) for s in units),episodes=len(units)))
        tables[stage]=table
    teacher=collections.defaultdict(list)
    for row in groups.get('E2_teacher',[]):
        for q in row['result']['queries']:
            teacher[(q['teacher'],q['budget'])].append(q)
    teacher_table=[dict(teacher=k[0],simulations=k[1],queries=len(v),
        reference_return=float(np.mean([np.mean(q['reference_values']) for q in v])),
        query_ms=float(np.mean([q['stats']['total_ms'] for q in v]))) for k,v in sorted(teacher.items())]
    candidate=collections.defaultdict(list)
    for row in groups.get('E3_candidates',[]):
        for method,counts in row['result']['metrics'].items():
            for n,m in counts.items():candidate[(method,n)].append(m)
    candidate_table=[dict(method=k[0],candidate_prefix=int(k[1]),states=len(v),
        prefix_reference_return=float(np.mean([x['original_prefix_best_report'] for x in v])),
        ideal_reference_return=float(np.mean([x['ideal_top1_report'] for x in v]))) for k,v in sorted(candidate.items())]
    timing=collections.defaultdict(list)
    for row in groups.get('E4_profile',[]):
        for q in row['result']['rows']:
            if 'budget' in q:timing[(q['arm'],q['budget'])].append(q)
    timing_table=[dict(method=k[0],budget=k[1],queries=len(v),
        zero_simulation_rate=float(np.mean([q['stats']['simulations']==0 for q in v])),
        p95_ms=float(np.percentile([q['stats']['total_ms'] for q in v],95)),
        completed_simulations=float(np.mean([q['stats']['simulations'] for q in v]))) for k,v in sorted(timing.items())]
    model_rows=[]
    for p in (out/'training').glob('*/result.json'):
        r=json.loads(p.read_text());model_rows.append(dict(name=r['job']['name'],physical_gpu=r['physical_gpu'],
            validation_loss=r['best_validation_loss'],training_loop_seconds=r['seconds'],parameters=r['parameters']))
    result=dict(phase=progress['phase'],smoke=json.loads((out/'experiment.json').read_text())['smoke'],
        closed_loop_tables=tables,teacher_quality=teacher_table,candidate_diagnostic=candidate_table,
        timing_diagnostic=timing_table,models=model_rows,failures=failures,
        training_failures=progress.get('training_failures',[]),
        job_wall_seconds_by_stage={k:sum(r.get('wall_seconds',0) for r in v) for k,v in groups.items()},
        cost_scope='job wall sums are not pure CPU hours; model loop timing excludes process startup and dataset loading',
        inference_scope='descriptive development diagnostics; final independent comparison CIs in final/results.json')
    (out/'eight_experiment_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    lines=['# E46八组实验汇总','',f'状态：{progress["phase"]}；完整结果以final/COMPLETE.json为准。',
        f'已保存训练结果：{len(model_rows)}；失败任务：{len(failures)}。','',
        '|实验|已完成记录|结果文件|','|---|---:|---|',
        f'|1 混合提议|{len(groups.get("E1_sim",[]))+len(groups.get("E1_wall",[]))}|eight_experiment_summary.json / closed_loop_tables|',
        f'|2 教师质量|{len(groups.get("E2_teacher",[]))}|teacher_quality|',
        f'|3 候选诊断|{len(groups.get("E3_candidates",[]))}|candidate_diagnostic|',
        f'|4 耗时拆解|{len(groups.get("E4_profile",[]))}|timing_diagnostic|',
        '|5/6 教师学习与排序|见模型清单|models与training/各作业|',
        '|7 学生状态纠错|见progress.json|correction_*与r1/r2模型|',
        '|8 封存确认|见final目录|final/REPORT.md与全家族同时区间|','',
        '开发结果仅用于已登记的方案选择，不把单个预算、种子或高分检查点挑作最终结论。',
        '模型内参考回报不等于真实任务最优值；只在同完整时间获益时讨论实际计算收益。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if progress['phase']!='finished' or failures or progress.get('training_failures'):raise SystemExit(2)


if __name__=='__main__':main()
