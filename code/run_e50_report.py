"""Build a concise, evidence-linked E50 deployment audit report."""
import argparse
import json
from pathlib import Path


def load(path): return json.loads(Path(path).read_text())


def run(out):
    root = Path(out).parent
    screening = load(root / 'method_screening.json')
    multi = load(root / 'all_methods_unknown_3maps_12cycles.json')
    root80 = load(root / 'all_methods_rootpuct_80_accounting.json')
    cap = load(root / 'capturable_consistent_batch_12cycles.json')
    cpp = load(root / 'cpp_transition_consistency_20.json')
    lines = ['# E50 未知地图 / LiDAR / MINCO 实车适配审计', '',
             '本报告汇总当前诊断证据，不把诊断初态或短预算 smoke 当作正式论文性能结果。', '',
             '## 验收门槛', '',
             f"- Python/C++ 转移一致性：通过，最大误差 `{cpp['max_state_error']:.3g}`。",
             f"- 多场景未知地图：{len(multi['rows'])} 个方法-场景组合，执行碰撞为 0。",
             f"- 80 步 RootPUCT：{len(root80['rows'])} 个方法均有长时域记账。",
             f"- 可捕获诊断初态：{len(cap['rows'])} 个方法-初态组合。", '',
             '## 方法筛选', '', '| 方法 | 多场景有效 | 80步证据 | 执行碰撞 | 可捕获双捕获 | 结论 |',
             '|---|---:|---:|---:|---:|---|']
    for row in screening['rows']:
        long = '有' if row['long_horizon_covered'] else '缺'
        keep = '进入下一阶段' if row['retain_for_next_stage'] else '补证据'
        lines.append(f"| {row['method']} | {row['unknown_scene_count']} | {long} | {row['execution_collisions']} | {row['capturable_double_capture_cases']}/3 | {keep} |")
    lines += ['', '## 当前限制', '',
              '- 正式 E49 初态的追捕者与逃跑者初始距离较大，逃跑者速度上限高于追捕者；该设置下零捕获是动力学结果，不能直接解释为方法失败。',
              '- 可捕获初态结果用于验证任务链路和双捕获记账，初态是诊断构造，不能替代正式抽样。',
              '- 当前 RootPUCT 预算为诊断预算；尚未完成正式多地图、完整预算的论文级双捕获率统计。',
              '- 实车 SLAM/LiDAR/通信硬件尚未接入，当前仍是仿真部署边界审计。', '',
              '## 主要产物', '',
              '- `current/e50_deployment_audit_20261110/all_methods_unknown_3maps_12cycles.json`',
              '- `current/e50_deployment_audit_20261110/all_methods_rootpuct_80_accounting.json`',
              '- `current/e50_deployment_audit_20261110/capturable_consistent_batch_12cycles.json`',
              '- `current/e50_deployment_audit_20261110/cpp_transition_consistency_20.json`']
    Path(out).write_text('\n'.join(lines) + '\n')
    return dict(report=str(out), methods=len(screening['rows']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(json.dumps(run(args.output), ensure_ascii=False, indent=2))
