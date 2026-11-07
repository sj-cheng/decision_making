"""Read-only, stdlib-only dashboard for original-l45 PPO continuation."""
import argparse
import json
import os
from pathlib import Path
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo


CODE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = CODE.parent / 'current/l45_ppo_control_20261022'
RUNNER = CODE / 'train_l45_ppo.py'
PHASES = {
    'prepared': '场景与配置已冻结',
    'correctness_checks': '正确性检查',
    'checks_passed': '正确性检查已通过',
    'profiling_serial_search': '单进程搜索测速',
    'profile_complete': '搜索测速已完成',
    'training': 'PPO 训练',
    'evaluation': '评测',
    'stopped_at_checkpoint': '已保存检查点并停止',
    'training_complete': '训练已完成，完整确认尚未完成',
    'summarizing': '汇总报告',
    'complete': '全部完成',
    'interrupted': '收到中断',
    'failed': '执行失败',
}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def runner_output(argv, cwd):
    """Return an output path only for this runner, never shell/spawn workers."""
    if not argv or not re.fullmatch(r'python(?:\d+(?:\.\d+)*)?', Path(argv[0]).name):
        return None
    index = 1
    while index < len(argv) and argv[index].startswith('-'):
        if argv[index] in ('-c', '-m'):
            return None
        index += 2 if argv[index] in ('-X', '-W') else 1
    if index >= len(argv):
        return None
    script = Path(argv[index])
    if not script.is_absolute():
        script = cwd / script
    if script.resolve() != RUNNER:
        return None
    output = DEFAULT_OUTPUT
    options = argv[index + 1:]
    for position, option in enumerate(options):
        if option == '--output':
            if position + 1 >= len(options):
                return None
            output = Path(options[position + 1])
        elif option.startswith('--output='):
            output = Path(option.split('=', 1)[1])
    return (output if output.is_absolute() else cwd / output).resolve()


def running_pids(output, proc_root=Path('/proc')):
    """Check real, non-zombie python drivers and their exact output directory."""
    output = Path(output).resolve()
    found = []
    try:
        entries = list(Path(proc_root).iterdir())
    except OSError:
        return found
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            state = (entry / 'stat').read_text().rsplit(')', 1)[1].split()[0]
            if state in ('Z', 'X'):
                continue
            argv = (entry / 'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
            cwd = (entry / 'cwd').resolve(strict=True)
            if runner_output(argv, cwd) == output:
                found.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return sorted(found)


def duration(seconds):
    minutes = max(0, round(seconds / 60))
    if minutes == 0:
        return '不足 1 分钟'
    hours, minutes = divmod(minutes, 60)
    return f'{hours} 小时 {minutes} 分钟' if hours else f'{minutes} 分钟'


def configured_workers(pid):
    try:
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
        for index, option in enumerate(argv):
            if option == '--workers':
                return int(argv[index+1])
            if option.startswith('--workers='):
                return int(option.split('=', 1)[1])
        return 32
    except (OSError, ValueError, IndexError):
        return None


def render(output):
    output = Path(output).resolve()
    experiment = read_json(output / 'experiment.json')
    config = experiment.get('config', {})
    state = read_json(output / 'status.json')
    phase = state.get('phase')
    pids = running_pids(output)
    report_ready = (output / 'REPORT.md').is_file() and (output / 'results.json').is_file()
    if phase == 'complete' and report_ready:
        live = '已完成' + ('（进程收尾中）' if pids else '')
    elif pids:
        live = '运行中，PID ' + ', '.join(str(pid) for pid in pids)
    elif phase == 'failed':
        live = '失败：没有活跃 runner，请查看 run.log'
    elif phase in ('training_complete', 'stopped_at_checkpoint'):
        live = '已停止，检查点已保存'
    elif phase in ('prepared', 'checks_passed', 'profile_complete'):
        live = '当前阶段已完成，尚未启动后续训练'
    elif phase or any(output.glob('seed*/progress.json')):
        live = '已中断/停止：以下为保存进度，没有活跃 runner'
    else:
        live = '未启动：源码或测试可能尚未完成，尚无运行状态'
    clock = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%m-%d %H:%M:%S')
    lines = [f'原始 l45 直接追加 PPO  {clock}（北京时间）',
             f'状态：{live}', f'阶段：{PHASES.get(phase, "等待启动")}',
             f'输出：{output}']
    if pids:
        workers = configured_workers(pids[0])
        if workers is not None:
            lines.append(f'并发任务进程：{workers}（单进程决策计时另行执行）')
    if config.get('smoke'):
        lines.append('功能 smoke：缩小配置，仅验证流程，不是正式训练结果。')
    lines += [f'原 270 维策略；只训练追捕；旧搜索价值网络冻结；'
              f'搜索 {config.get("search_budget", 1000)} 次 / 深度 {config.get("search_depth", 100)}', '']
    seeds = config.get('seeds', [0, 1, 2])
    total = config.get('episodes', 2048)
    batch = config.get('batch', 64)
    remaining = 0
    rates = []
    for seed in seeds:
        progress = read_json(output / f'seed{seed}/progress.json')
        selection = read_json(output / f'seed{seed}/selection.json')
        completed = progress.get('episode', 0)
        target = progress.get('total', total)
        remaining += max(0, target - completed)
        fill = int(18 * min(1., completed / max(1, target)))
        lines.append(f'seed {seed}  [{"█" * fill}{"·" * (18 - fill)}] '
                     f'{completed}/{target} 局；最佳第 {selection.get("best_episode", progress.get("best_episode", 0))} 局')
        sample = progress.get('sampling_seconds', 0.)
        update = progress.get('update_seconds', 0.)
        if sample + update > 0:
            rates.append((sample + update) / max(1, batch))
            lines.append(f'  最近一批 {batch} 局：采样 {sample:.1f}s，更新 {update:.1f}s，'
                         f'{batch / (sample + update):.3f} 局/s')
    if rates and remaining:
        estimate = remaining * sum(rates) / len(rates)
        label = '估计' if pids else '按恢复后相同吞吐估计'
        lines.append(f'剩余训练采样与更新{label}：{duration(estimate)}。')
        lines.append('此 ETA 不含后续验证、完整搜索确认、回归和汇总，不是全部完成时间。')
    elif remaining:
        lines.append('首批尚未完成，暂无训练吞吐和 ETA。')
    else:
        lines.append('所有采样训练已完成；评测是否完成请看阶段和确认进度。')
    lines.append('')
    if phase == 'evaluation':
        label = str(state.get('evaluation', '未知评测'))
        label = ''.join(char for char in label if char.isprintable())
        lines.append(f'当前评测 {label}：{state.get("completed", 0)}/{state.get("total", "--")}')
    elif phase == 'profiling_serial_search':
        lines.append(f'单进程搜索测速：{state.get("completed", 0)}/{state.get("total", 12)} 次决策')
    for folder, label in (('baseline_validation', '原 l45 验证'),
                          ('confirmation', '四障碍独立确认'), ('regression', '无障碍回归'),
                          ('guard_four', '四障碍逃逸回归'), ('guard_none', '无障碍逃逸回归'),
                          ('serial_latency', '单进程决策耗时')):
        manifest = read_json(output / folder / 'manifest.json', [])
        completed = sum((output / folder / f'{row["key"]}.json').is_file()
                        for row in manifest if isinstance(row, dict) and 'key' in row)
        lines.append(f'{label}：{completed}/{len(manifest) if manifest else "待开始"}')
    checks = read_json(output / 'correctness.json')
    lines += ['', '正确性检查：' + ('通过' if checks.get('passed') else '尚无通过记录'),
              f'报告：{output / "REPORT.md"}' + ('' if report_ready else '（尚未生成）'),
              '进度不代表胜率。Ctrl+C 只关闭看板，不会停止训练。']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=3.)
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error('--interval must be positive')
    try:
        while True:
            print(('' if args.once else '\033[2J\033[H') + render(args.output), flush=True)
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
