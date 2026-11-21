"""Split E50 rolling outcomes into method/observation/trajectory/communication causes."""
import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
import repo_paths
repo_paths.install()
import argparse
import json
from pathlib import Path


def account(source, output):
    data = json.loads(Path(source).read_text())
    rows = []
    for row in data.get('rows', []):
        cycles = row.get('cycles', [])
        captures = sum(len(c.get('capture_events', [])) for c in cycles)
        rows.append(dict(
            method=row.get('method'), valid=bool(row.get('valid')), error=row.get('error'),
            decisions=len(cycles), captures=captures,
            uncaptured_active=max(0, len(cycles) - captures),
            execution_collisions=sum(bool(c.get('collision')) for c in cycles),
            trajectory_infeasible=sum(not bool(c.get('feasible', False)) for c in cycles),
            communication_stop=sum(c.get('communication_event') == 'stop_degraded' for c in cycles),
            packet_drops=sum(c.get('communication_event') == 'packet_dropped' for c in cycles),
            local_projections=sum(c.get('safety_override') == 'local_action_projection' for c in cycles),
            future_collision_stops=sum(c.get('safety_override') == 'stop_future_collision' for c in cycles),
            rootpuct_search_failures=max((c.get('rootpuct_search_failures', 0) for c in cycles), default=0),
            min_target_distance=min((c.get('min_target_distance', float('inf')) for c in cycles), default=None),
            planning_ms_mean=(sum(c.get('planning_ms', 0.) for c in cycles) / len(cycles)
                              if cycles else None),
        ))
    result = dict(stage='e50_method_accounting', source=str(source), rows=rows)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(account(args.source, args.output), ensure_ascii=False, indent=2))
