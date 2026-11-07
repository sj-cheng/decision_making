import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

P=Path('/data2/sjc/decision_making/current/e45y_compute_allocation_20261104')


def main():
    print('E45-Y | decision-sensitive compute allocation feasibility | GPU 3')
    print(datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    f=P/'status.json'
    if not f.exists():
        found=list(P.glob('calibration_*/status.json'))
        f=max(found,key=lambda x:x.stat().st_mtime) if found else None
    if f is None or not f.exists():print('Preparing / engineering acceptance');return
    s=json.loads(f.read_text());print('State:',s.get('state'),'stage:',s.get('stage'))
    if 'planned' in s:print('Jobs:',s['completed'],'/',s['planned'],'CPU workers:',s['workers'])
    if s.get('error'):print('ERROR:',s['error'])
    print('First-capture branch X complete; no independent event gain established.')
    print('Current scope: state-batch prefix replay, not a live full-episode scheduler.')


if __name__=='__main__':main()
