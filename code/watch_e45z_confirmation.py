import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

P=Path('/data2/sjc/decision_making/current/e45z_independent_confirmation_20261104')


def main():
    print('E45-Z independent development replication | GPU3 | no training')
    print(datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    f=P/'status.json'
    if not f.exists():print('Preparing / engineering acceptance');return
    s=json.loads(f.read_text());print('State:',s.get('state'),'phase:',s.get('phase'))
    if 'planned' in s:
        print('Jobs:',s['completed'],'/',s['planned'],'workers:',s['workers'])
        done=s['completed'];elapsed=s.get('elapsed_seconds',0)
        if done:print('Phase elapsed / rough remaining minutes:',round(elapsed/60,1),round(elapsed/done*(s['planned']-done)/60,1))
    if s.get('error'):print('ERROR:',s['error'])
    if (P/'completion_audit.json').exists():
        audit=json.loads((P/'completion_audit.json').read_text())
        print('Maps / public roots / reference episodes:',audit['maps'],audit['public_roots'],audit['unique_reference_episodes'])
        print('Reference physical steps:',audit['reference_physical_steps'])
        print('Resource gate for one cheap predictor pilot:',s.get('advance_to_predictor_pilot'))
    print('Six maps new to X/Y/Z, not sealed tests; expensive selector, one-action intervention.')


if __name__=='__main__':main()
