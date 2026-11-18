"""Static preflight for the ROS2 bridge when ROS is unavailable locally."""
import argparse, json
from pathlib import Path

def run(output):
    root = Path(__file__).parent / 'ros2' / 'e50_runtime_bridge'
    node = (root / 'e50_runtime_bridge' / 'node.py').read_text()
    package = (root / 'package.xml').read_text()
    setup = (root / 'setup.py').read_text()
    checks = {
        'package_xml': (root / 'package.xml').exists() and '<name>e50_runtime_bridge</name>' in package,
        'setup_entrypoint': 'e50_runtime_bridge = e50_runtime_bridge.node:main' in setup,
        'pose_topic': "'/e50/poses'" in node,
        'lidar_topics': "'/e50/lidar0'" in node and "'/e50/lidar1'" in node,
        'belief_topic': "'/e50/belief'" in node,
        'requested_commands': "'/e50/requested_cmd0'" in node and "'/e50/requested_cmd1'" in node,
        'gated_outputs': "'/e50/cmd0'" in node and "'/e50/cmd1'" in node,
        'degraded_gate': 'gate_command' in node,
    }
    result = dict(stage='ros2_bridge_static_preflight', ros_installed=False,
                  checks=checks, passed=sum(checks.values()), total=len(checks),
                  next_command='colcon build --packages-select e50_runtime_bridge')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output', required=True); a=p.parse_args()
    print(json.dumps(run(a.output), ensure_ascii=False, indent=2))
