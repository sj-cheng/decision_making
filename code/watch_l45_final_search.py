"""Read-only, stdlib-only dashboard for independent final-checkpoint search tests."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo


CODE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = CODE.parent / 'current/l45_ppo_final_search_20261023'
RUNNER = CODE / 'confirm_l45_final_search.py'
PHASES = {
    'checking': '核对模型、配置与独立场景',
    'prepared': '场景与配置已冻结',
    'evaluation': '独立搜索评测',
    'summarizing': '汇总确认与回归结果',
    'diagnostics': '配对失败诊断',
    'complete': '报告核验',
    'failed': '执行失败',
    'interrupted': '收到中断，保留已完成评测',
}
SPLITS = (
    ('confirmation', '四障碍独立确认'),
    ('regression', '无障碍独立回归'),
    ('diagnostics', '配对分歧诊断'),
)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def runner_output(argv, cwd):
    """Resolve output only for a direct Python invocation of this exact runner."""
    if not argv or not re.fullmatch(r'python(?:\d+(?:\.\d+)*)?', Path(argv[0]).name):
        return None
    index = 1
    while index < len(argv) and argv[index].startswith('-'):
        if argv[index] in ('-c', '-m'):
            return None
        if argv[index] == '--':
            index += 1
            break
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
    """Do not trust saved PIDs or include shell wrappers, spawn workers or zombies."""
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


def configured_workers(pid, proc_root=Path('/proc')):
    try:
        argv = (Path(proc_root) / str(pid) / 'cmdline').read_bytes().decode(
            errors='replace').rstrip('\0').split('\0')
        for index, option in enumerate(argv):
            if option == '--workers':
                return int(argv[index + 1])
            if option.startswith('--workers='):
                return int(option.split('=', 1)[1])
        return 80
    except (OSError, ValueError, IndexError):
        return None


def cached_progress(output, split):
    """Count only expected task result files, never cached status counters."""
    folder = Path(output) / split
    rows = read_json(folder / 'manifest.json', [])
    if isinstance(rows, dict):
        rows = rows.get('tasks', [])
    if not isinstance(rows, list):
        rows = []
    keys = {str(row['key']) for row in rows
            if isinstance(row, dict) and 'key' in row}
    return sum((folder / f'{key}.json').is_file() for key in keys), len(keys)


def duration(seconds):
    minutes = max(0, round(seconds / 60))
    if not minutes:
        return '不足 1 分钟'
    hours, minutes = divmod(minutes, 60)
    return f'{hours} 小时 {minutes} 分钟' if hours else f'{minutes} 分钟'


def split_eta(state, completed, total):
    throughput = state.get('throughput', {})
    if not isinstance(throughput, dict):
        throughput = {}
    rate = throughput.get('rate', state.get('rate'))
    elapsed = throughput.get('elapsed_seconds', state.get('elapsed_seconds'))
    try:
        rate = float(rate)
        elapsed = float(elapsed)
    except (TypeError, ValueError):
        return None
    if rate > 0 and elapsed > 0 and total > completed:
        return duration((total - completed) / rate)
    return None


def render(output):
    output = Path(output).resolve()
    state = read_json(output / 'status.json')
    if not isinstance(state, dict):
        state = {}
    phase = state.get('phase')
    pids = running_pids(output)
    report_ready = (output / 'REPORT.md').is_file() and (output / 'results.json').is_file()
    finished = phase == 'complete' and report_ready
    if finished:
        live = '已完成' + ('（进程收尾中）' if pids else '')
    elif pids:
        live = '运行中，PID ' + ', '.join(str(pid) for pid in pids)
    elif phase == 'failed':
        live = '失败：没有活跃 runner，请查看 run.log'
    elif state or any((output / split / 'manifest.json').is_file() for split, _ in SPLITS):
        live = '已中断/停止：以下为保存进度，没有活跃 runner'
    else:
        live = '未启动：尚无运行状态'
    now = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%m-%d %H:%M:%S')
    lines = [f'l45 PPO 最终检查点独立评测  {now}（北京时间）',
             '本实验不训练；原正式实验、模型和搜索参数保留。',
             f'状态：{live}',
             f'阶段：{"全部完成" if finished else PHASES.get(phase, "等待启动")}',
             f'输出：{output}',
             '模型：原 l45 ＋ 3 个验证最佳检查点 ＋ 3 个最终检查点',
             '原 MCTS：1,000 次模拟 / 深度 100；Search Only 对手：3,000 次 / 深度 100。']
    if pids:
        worker_values = [configured_workers(pid) for pid in pids]
        descriptions = [f'PID {pid}: {workers}' for pid, workers in zip(pids, worker_values)
                        if workers is not None]
        if descriptions:
            lines.append('实际评测并行数：' + '；'.join(descriptions))
        if state.get('pid') is not None and state.get('pid') not in pids:
            lines.append('保存状态的 PID 与当前进程不同；以 /proc 实际进程为准。')
    updated = state.get('updated_unix')
    if isinstance(updated, (int, float)):
        try:
            written = datetime.fromtimestamp(updated, ZoneInfo('Asia/Shanghai')).strftime('%m-%d %H:%M:%S')
            lines.append(f'最近状态写入：{written}（北京时间；不是存活证明）')
        except (ValueError, OverflowError, OSError):
            pass
    lines.append('')
    progress = {}
    for split, label in SPLITS:
        completed, total = cached_progress(output, split)
        progress[split] = completed, total
        summary = '；统计文件已生成' if (output / split / 'summary.json').is_file() else ''
        lines.append(f'{label}缓存实际文件：{completed}/{total if total else "待准备"}{summary}')
    evaluation = state.get('evaluation')
    if evaluation in progress and phase in ('evaluation', 'diagnostics'):
        completed, total = progress[evaluation]
        saved_done = state.get('completed', 0)
        saved_total = state.get('total', '--')
        lines.append(f'当前阶段 {evaluation} 保存计数：{saved_done}/{saved_total}（以上方实际缓存为准）')
        estimate = split_eta(state, completed, total)
        if estimate and pids:
            lines.append(f'按实际吞吐估计当前 {evaluation} 剩余：{estimate}。')
            lines.append('此 ETA 仅含当前 split，不含后续评测、诊断与汇总。')
        else:
            lines.append('暂无可用的当前 split 吞吐 ETA；不推测全部完成时间。')
    lines += ['', f'报告：{output / "REPORT.md"}' + ('' if report_ready else '（尚未齐备）'),
              '进度不代表胜率。Ctrl+C 只关闭看板，不会停止评测。']
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
