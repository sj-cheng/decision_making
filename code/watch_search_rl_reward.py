"""Read-only progress for the isolated second-capture reward experiment."""
import argparse
import json
import os
from pathlib import Path
import time


def read(path,default):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default


def running_pids(script):
    result=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            if path.stat().st_uid!=os.getuid():continue
            args=(path/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if args and Path(args[0]).name.startswith('python') and any(Path(arg).name==script for arg in args[1:] if arg):
                result.append(int(path.name))
        except (OSError,ValueError):continue
    return sorted(result)


def render(root):
    experiment=read(root/'experiment.json',{});config=experiment.get('config',{})
    state=read(root/'status.json',{});selected=read(root/'selection.json',{}).get('selected')
    extension=read(root/'completion_extension/status.json',{})
    names={'legacy':'原奖励','completion_bonus_v1':'合法双捕 +1','capture_floor_v1':'捕获奖励不衰减'}
    phase=state.get('phase','等待启动')
    main_pids=running_pids('run_search_rl_reward_experiment.py')
    bonus_pids=running_pids('extend_completion_reward.py')
    live='已完成' if phase=='complete' else f'运行中 PID {main_pids[0]}' if main_pids else '已停止：以下为旧进度'
    lines=[f'第二次捕获奖励实验  {time.strftime("%H:%M:%S")}',f'主任务：{live}；阶段：{phase}',
           f'候选：{names.get(selected,selected) if selected else "尚未选择"}；逃逸方冻结；搜索预算1000/深度8','']
    for seed in config.get('seeds',[]):
        for arm in names:
            owner=root/'completion_extension' if extension and arm=='completion_bonus_v1' else root
            metrics=owner/f'seed{seed}'/arm/'metrics.jsonl';count=0
            try:
                for line in metrics.read_text().splitlines():count=max(count,json.loads(line)['episodes'])
            except (OSError,ValueError):pass
            target=config.get('long_episodes',0) if selected and (arm in ('legacy',selected) or owner!=root) else config.get('short_episodes',0)
            ratio=min(count/max(target,1),1.);fill=int(ratio*18)
            lines.append(f'seed{seed} {names[arm]:<10} [{"█"*fill}{"·"*(18-fill)}] {count}/{target}')
    lines.append('')
    for folder,label in [('short_validation','短训验证'),('long_confirmation','较长训练确认'),('search_confirmation','完整搜索评估')]:
        manifest=read(root/folder/'manifest.json',[])
        done=sum((root/folder/(r['key']+'.json')).exists() for r in manifest)
        lines.append(f'{label}：{done}/{len(manifest) if manifest else "--"}')
    if extension:
        bonus_live='已完成' if extension.get('phase')=='complete' else f'运行中 PID {bonus_pids[0]}' if bonus_pids else '已停止（旧进度）'
        lines.append('双捕奖励续测：'+bonus_live)
        manifest=read(root/'completion_extension/search_confirmation/manifest.json',[])
        done=sum((root/'completion_extension/search_confirmation'/(r['key']+'.json')).exists() for r in manifest)
        lines.append(f'双捕奖励搜索：{done}/{len(manifest) if manifest else "--"}')
    lines+=['','进度表示完成工作量，胜率以独立评估为准。','Ctrl+C 只关闭看板。']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'current/search_rl_reward_20261022')
    p.add_argument('--once',action='store_true');p.add_argument('--interval',type=float,default=3.)
    args=p.parse_args()
    try:
        while True:
            print(('' if args.once else '\033[2J\033[H')+render(args.output),flush=True)
            if args.once:return
            time.sleep(max(.2,args.interval))
    except KeyboardInterrupt:pass


if __name__=='__main__':main()
