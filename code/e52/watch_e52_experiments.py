"""Persistent read-only tmux watchdog for E52 stages, worker progress and results."""
import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
import repo_paths
repo_paths.install()
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo


def alive(pid,needle=None):
    try:
        path=Path('/proc')/str(int(pid))
        command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        state=(path/'stat').read_text().split(') ',1)[1][0]
        return state!='Z' and bool(command) and (needle is None or needle in command)
    except (OSError,ValueError,TypeError):return False


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return None


def inspect(root,stale_seconds=180,previous=None,now=None):
    now=time.time() if now is None else now;previous=previous or {}
    status=read(root/'status.json') or {};stages=status.get('stages',{})
    alerts=[];summary={};workers={}
    pipeline=stages.get('pipeline',{}).get('state')
    supervisor_alive=alive(status.get('supervisor_pid'),'launch_e52_parallel.py')
    if pipeline=='failed':alerts.append('CRITICAL: pipeline failed; inspect supervisor.log / failure_analysis.json')
    elif status and not supervisor_alive and pipeline!='complete':alerts.append('CRITICAL: supervisor exited before completion')
    for name,stage in stages.items():
        if name != 'pipeline' and stage.get('state')=='failed':alerts.append(f'ERROR: stage {name} failed (exit {stage.get("exit_code")})')
        if stage.get('state')=='running' and not alive(stage.get('pid'),'python'):
            alerts.append(f'ERROR: stage {name} says running but PID is missing or reused')
    for path in root.glob('*/*.progress.json'):
        data=read(path)
        if not data:continue
        group=path.parent.name;key=str(path)
        item=summary.setdefault(group,dict(started=0,completed=0,live_workers=0,min_cycle=None,max_cycle=0))
        item['started']+=1;cycle=data.get('completed_cycles',0)
        item['min_cycle']=cycle if item['min_cycle'] is None else min(item['min_cycle'],cycle)
        item['max_cycle']=max(item['max_cycle'],cycle)
        if data.get('state')=='complete':item['completed']+=1;continue
        live=alive(data.get('pid'),'run_e52_experiments.py');item['live_workers']+=int(live)
        old=previous.get(key,{})
        advanced=old.get('cycle')!=cycle or old.get('pid')!=data.get('pid')
        last_advance=now if advanced else old.get('last_advance',now)
        workers[key]=dict(cycle=cycle,pid=data.get('pid'),last_advance=last_advance)
        if not live:alerts.append(f'ERROR: {group}/{path.stem}: unfinished worker exited')
        elif now-last_advance>stale_seconds:alerts.append(f'WARNING: {group}/{path.stem}: no cycle progress for {int(now-last_advance)}s')
    for path in root.glob('*/summary.json'):
        data=read(path)
        if not data:continue
        rows=data.get('rows',[]);bad=[r for r in rows if any(r.get(k,0) for k in ('collision_cycles','minco_infeasible','rootpuct_failures','terminal_opponent_skips','lifecycle_errors'))]
        if bad:alerts.append(f'DATA: {path.parent.name}: {len(bad)}/{len(rows)} episodes need analysis')
        gates=data.get('gates',{})
        if gates.get('formal_complete'):
            failed=[key for key in ('capture_improvement','execution_safe','lifecycle','latency') if not gates.get(key)]
            if failed: alerts.append('RESULT: formal acceptance gates not met: '+','.join(failed))
    training=stages.get('training',{})
    if training.get('state')=='running' and alive(training.get('pid'),'run_e52_train_nar_nae.py'):
        log=Path(training['log'])
        if log.exists() and now-log.stat().st_mtime>1200:alerts.append('WARNING: training log has not advanced for 20 minutes')
    return dict(checked_at=datetime.fromtimestamp(now,ZoneInfo('Asia/Shanghai')).isoformat(),
        watchdog_pid=os.getpid(),supervisor_alive=supervisor_alive,pipeline_state=pipeline or 'running',
        stages={name:s.get('state') for name,s in stages.items()},groups=summary,
        alerts=sorted(set(alerts)),workers=workers)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--interval',type=float,default=30.)
    p.add_argument('--stale-seconds',type=float,default=180.);p.add_argument('--once',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve();root.mkdir(parents=True,exist_ok=True)
    import fcntl
    handle=(root/'watchdog.lock').open('w');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous={};last_signature=None
    while not (root/'watchdog.stop').exists():
        try:
            result=inspect(root,a.stale_seconds,previous);previous=result.pop('workers')
            result['interval_seconds']=a.interval
            temp=root/'watchdog_status.tmp';temp.write_text(json.dumps(result,indent=2,ensure_ascii=False));temp.replace(root/'watchdog_status.json')
            signature=hashlib.sha256(json.dumps(result['alerts']).encode()).hexdigest()
            if signature!=last_signature:
                with (root/'watchdog_alerts.jsonl').open('a') as stream:stream.write(json.dumps(dict(at=result['checked_at'],alerts=result['alerts']),ensure_ascii=False)+'\n')
                last_signature=signature
            print(json.dumps(result,ensure_ascii=False),flush=True)
        except Exception as exc:print(json.dumps(dict(watchdog_error=repr(exc))),flush=True)
        if a.once:break
        time.sleep(max(1.,a.interval))


if __name__=='__main__':main()
