import json
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

P=Path('/data2/sjc/decision_making/current/e45x_first_capture_diagnostic_20261104')


def main():
    print('E45-X | first-capture event diagnostic | all new CUDA work on GPU 3')
    print(datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    f=P/'status.json'
    if not f.exists():print('Preparing / engineering acceptance');return
    s=json.loads(f.read_text());print('State:',s.get('state'),'stage:',s.get('stage'),s.get('substage',''))
    completed=s.get('completed',0);planned=s.get('planned');elapsed=s.get('elapsed_seconds',0)
    if planned:
        print(f'Jobs: {completed}/{planned}; CPU workers: {s.get("workers")}; stage elapsed {elapsed/60:.1f} min')
        if completed>0:print(f'Early stage-only linear remaining {(planned-completed)*elapsed/completed/3600:.2f} h')
    roots=P/'root_manifest.json'
    if roots.exists():
        r=json.loads(roots.read_text());print('Public roots:',len(r['roots']),'collection episodes:',len(r['collection']))
    heads=P/'leaf_training_audit.json'
    if heads.exists():
        h=json.loads(heads.read_text());print('Training-only frozen-embedding readouts:',h['training_examples'],'examples; residual',h['residual_examples'])
    if s.get('error'):print('ERROR:',s['error'])
    print('Actor optimizer updates: 0; W final test is sealed and not used here.')
    print('This measures one-action ranking and continuation value; full new-policy closed-loop gains are not implied.')


if __name__=='__main__':main()
