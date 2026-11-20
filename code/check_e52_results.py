"""Check trace integrity and report performance misses before formal scoring."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('summary', type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    rows = data['rows']
    stage = data['manifest']['stage']
    expected = len(data['manifest']['jobs'])
    if not rows or len(rows) != expected:
        raise SystemExit(f'Incomplete {stage} results: {len(rows)}/{expected}')

    defects = []
    outcomes = []
    for row in rows:
        for key in ('rootpuct_failures', 'terminal_opponent_skips', 'lifecycle_errors'):
            if row[key]:
                defects.append(dict(run=row['run'], reason=key, count=row[key]))
        for key in ('collision_cycles', 'minco_infeasible'):
            if row[key]:
                finding = dict(run=row['run'], reason=key, count=row[key])
                outcomes.append(finding)
                if stage == 'teacher':
                    defects.append(finding)
        if row['mode'] == 'both' and row['decision_p95_ms'] > 200.:
            outcomes.append(dict(run=row['run'], reason='latency', p95_ms=row['decision_p95_ms']))

    print(json.dumps(dict(stage=stage, integrity_errors=defects,
                         performance_misses=outcomes), indent=2))
    if defects:
        raise SystemExit(f'Investigate {len(defects)} {stage} data issues before continuing')
    if outcomes:
        print('Preserving observed failures in all paired evaluation results.')
    else:
        print(f'{stage} checks passed.')


if __name__ == '__main__':
    main()
