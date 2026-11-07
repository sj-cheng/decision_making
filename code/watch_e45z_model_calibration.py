import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

P=Path('/data2/sjc/decision_making/current/e45z_model_calibration_20261104')


def main():
    print('E45-Z | one-step opponent model calibration | GPU 3')
    print(datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    f=P/'status.json'
    if not f.exists():print('Preparing / engineering acceptance');return
    s=json.loads(f.read_text());print('State:',s.get('state'),'phase:',s.get('phase'))
    if 'planned' in s:
        print('Public roots:',s['completed'],'/',s['planned'],'CPU workers:',s['workers'])
        elapsed=s.get('elapsed_seconds',0.);done=s['completed']
        if done:print('Elapsed / rough remaining seconds:',round(elapsed),round(elapsed/done*(s['planned']-done)))
    if s.get('error'):print('ERROR:',s['error'])
    if (P/'completion_audit.json').exists():
        a=json.loads((P/'completion_audit.json').read_text())
        print('RootPUCT calls:',a['Root_search_calls'],'physical audits:',a['physical_one_step_audits'])
    print('No policy training; prior-sampled one-step diagnostic, not closed-loop task efficacy.')


if __name__=='__main__':main()
