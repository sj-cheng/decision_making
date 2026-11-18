"""Machine-readable requirement audit for the current E50 plan state."""
import argparse
import json
from pathlib import Path


def run(output, root='current/e50_deployment_audit_20261117'):
    root = Path(root)
    legacy = Path('current/e50_deployment_audit_20261110')
    def exists(name): return (root / name).exists() or (legacy / name).exists()
    checks = [
        ('known_map_regression', exists('rolling_long_wall_0001.json'), 'execution smoke exists'),
        ('lidar_ray_legality', exists('lidar_safety_sweep.json'), 'ray/safety sweep exists'),
        ('minco_legality', exists('compat_long_wall_0001.json'), 'MINCO compatibility audit exists'),
        ('python_transition_consistency', exists('transition_consistency_100.json'), '100-case Python parity exists'),
        ('cpp_transition_consistency', exists('cpp_transition_consistency_20.json'), '20-case native parity exists'),
        ('unknown_map_multi_scene', exists('all_methods_unknown_3maps_12cycles.json'), '11 methods x 3 maps exists'),
        ('distributed_degradation', exists('unknown_map_delay_drop_pose_batch.json'), 'delay/drop/pose audit exists'),
        ('method_screening', exists('method_screening.json'), 'screening ledger exists'),
        ('runtime_input_boundary', exists('runtime_adapter_smoke.json'), 'runtime boundary smoke exists'),
        ('free_ray_frontier', exists('FREE_RAY_SPARSE_160.md') and exists('FREE_RAY_CORNER_160.md'), 'LiDAR free-ray frontier evidence exists'),
        ('leaf_value_training', exists('LEAF_VALUE_TRAINING.md'), 'trained value checkpoint and holdout evidence exists'),
        ('leaf_value_ablation', exists('LEAF_VALUE_ABLATION_160.md'), 'trained vs untrained multi-initial ablation exists'),
        ('trajectory_visualization', exists('wall_leafvalue.gif') and exists('corner_leafvalue.gif') and exists('sparse_leafvalue.gif'), 'latest executed-prefix GIFs exist'),
        ('ros_message_adapter_replay', exists('ros_shaped_adapter_smoke.json'), 'ROS-shaped pose/scan/belief replay and degraded gate exists'),
        ('ros2_bridge_skeleton', exists('ROS2_BRIDGE_HANDOFF.md'), 'portable ROS2 bridge package and handoff exists'),
        ('ros2_static_preflight', exists('ros2_static_preflight.json'), 'ROS2 package/topic static preflight exists'),
        ('hardware_ros_integration', False, 'requires real ROS/SLAM/LiDAR/communication hardware'),
        ('formal_e49_double_capture_statistics', exists('FORMAL_RANDOM_30_160.md'), 'history-synchronised 30-case frozen-budget statistic exists'),
    ]
    rows = [dict(requirement=name, status='passed' if passed else 'missing_or_pending', evidence=note)
            for name, passed, note in checks]
    result = dict(stage='e50_completion_audit', root=str(root), rows=rows,
                  passed=sum(r['status'] == 'passed' for r in rows), total=len(rows),
                  complete=False)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(json.dumps(run(args.output), ensure_ascii=False, indent=2))
