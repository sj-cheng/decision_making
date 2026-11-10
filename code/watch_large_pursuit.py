"""Read-only status: verifies coordinator liveness instead of trusting stale progress."""
import json
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

root=Path(__file__).resolve().parents[1]/'current/large_pursuit_v2_20261108'
def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}
def alive(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text();return stat.rsplit(')',1)[1].split()[0]!='Z'
    except (OSError,IndexError):return False
status=read(root/'progress.json');coordinator=read(root/'full_coordinator.json');verification=read(root/'verification_coordinator.json')
verifying=verification.get('status')=='running' and alive(verification.get('pid',0))
pid=verification.get('pid',0) if verifying else coordinator.get('pid',0)
running=alive(pid) and not (root/'audit_hold.json').exists()
print(datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S %Z'))
print('调度：'+('修复验证中，实验未启动' if verifying else ('审核阻断，正式实验未启动' if (root/'audit_hold.json').exists() else ('运行中' if running else ('已完成' if status.get('stage')=='complete' else '已停止，需查看错误日志')))))
print('阶段：',status.get('stage','未知'),'调度PID：',pid)
for phase in ('pilot','formal','independent'):
    folders=list((root/phase).glob('r*'))
    if not folders:continue
    complete=sum((p/'result.json').exists() for p in folders)
    print(f'{phase}: {complete}/{len(folders)} 个已有任务完成')
for item in status.get('active',[]) if isinstance(status.get('active'),list) else []:
    if not isinstance(item,dict):continue
    progress=read(Path(item['folder'])/'progress.json')
    job=read(Path(item['folder'])/'job.json')
    print(Path(item['folder']).name,'存活='+str(alive(item['pid'])),str(progress.get('steps',0))+'/'+str(job.get('team_steps','?')),'GPU'+str(item.get('gpu','?')))
if not running and (root/'full_failure.json').exists():
    failure=read(root/'full_failure.json');print(failure.get('traceback','').strip().splitlines()[-1])
print('日志：',root/'full_console.log')
