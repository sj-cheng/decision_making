"""Pair frozen E49 formal baseline metrics with E50 deployment audit metrics."""
import argparse
import csv
import json
from pathlib import Path


def run(e49_csv, e50_json, output):
    baseline = {}
    with Path(e49_csv).open() as handle:
        for row in csv.DictReader(handle):
            name = row.get('method') or row.get('arm') or row.get('name')
            if name: baseline[name] = row
    audit = json.loads(Path(e50_json).read_text())
    rows = []
    for row in audit['rows']:
        name = row['method']; b = baseline.get(name, {})
        rows.append(dict(method=name,
                         frozen_e49_double_capture_rate=float(b['double_capture']) if b.get('double_capture') else None,
                         frozen_e49_at_least_one_rate=float(b['at_least_one']) if b.get('at_least_one') else None,
                         unknown_map_scene_count=row.get('unknown_scene_count'),
                         unknown_map_execution_collisions=row.get('execution_collisions'),
                         unknown_map_capturable_double_cases=row.get('capturable_double_capture_cases'),
                         deployment_long_horizon_covered=row.get('long_horizon_covered'),
                         deployment_screening=row.get('retain_for_next_stage')))
    result = dict(stage='e50_frozen_baseline_vs_deployment', e49_source=str(e49_csv),
                  e50_source=str(e50_json), rows=rows,
                  note='Frozen E49 capture rates are direct/known-map formal results; E50 rows are deployment diagnostics and are not pooled.')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n'); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--e49-csv', required=True)
    parser.add_argument('--e50-json', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(json.dumps(run(args.e49_csv, args.e50_json, args.output), ensure_ascii=False, indent=2))
