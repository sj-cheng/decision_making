"""Read-only, dependency-free terminal dashboard for a running search/RL job.

Does not import the trainer, load checkpoints, change its fingerprint or send
signals. Completed episode counts come from durable metrics, never an estimated
within-batch percentage. Ctrl+C closes only this viewer.
"""
import argparse
import functools
import json
import os
from pathlib import Path
import shutil
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=64)
def _read_json(path, modified, size):
    return json.loads(Path(path).read_text())


def read_json(path, default=None):
    try:
        stat = path.stat()
        return _read_json(str(path), stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError):
        return default


def read_metrics(path):
    try:
        # The final line can be in flight while the trainer appends it.
        lines = path.read_text().splitlines(keepends=True)
    except OSError:
        return []
    rows = []
    for line in lines:
        if not line.endswith('\n'):
            continue
        try:
            row = json.loads(line)
            if 'episodes' in row and row.get('role') in (0, 1):
                rows.append(row)
        except ValueError:
            continue
    return rows


def process_activity(output, previous):
    """Find the exact output's trainer, and sample worker CPU counters."""
    matches = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = [part.decode(errors='replace') for part in (proc/'cmdline').read_bytes().split(b'\0') if part]
            names = ('train_search_rl.py','train_search_rl_lookahead.py')
            if not any(Path(arg).name in names for arg in args) or '--output' not in args:
                continue
            target = Path(args[args.index('--output')+1])
            if not target.is_absolute():
                target = (proc/'cwd').resolve()/target
            if target.resolve() == output:
                matches.append(proc)
        except (OSError, ValueError, IndexError):
            continue
    if not matches:
        return dict(running=False, pids=[], workers=0, busy=None, cpu=None), {}
    ticks = os.sysconf('SC_CLK_TCK')
    now = time.monotonic()
    samples = {}
    worker_ids = set()
    for proc in matches:
        try:
            children = (proc/'task'/proc.name/'children').read_text().split()
            for pid in children:
                child = Path('/proc')/pid
                if b'spawn_main' in (child/'cmdline').read_bytes():
                    worker_ids.add(pid)
        except OSError:
            continue
    busy = 0
    total_cpu = 0.
    measured = 0
    for pid in worker_ids:
        try:
            fields = (Path('/proc')/pid/'stat').read_text().rsplit(')', 1)[1].split()
            cpu = (int(fields[11])+int(fields[12]))/ticks
            samples[pid] = (now, cpu)
            if pid in previous:
                dt = now-previous[pid][0]
                use = max(0., cpu-previous[pid][1])/max(dt, .001)
                busy += use > .05
                total_cpu += use*100.
                measured += 1
        except (OSError, ValueError, IndexError):
            continue
    return dict(running=True, pids=[int(p.name) for p in matches], workers=len(worker_ids),
                busy=busy if measured else None, cpu=total_cpu if measured else None), samples


def validation(path, role):
    data = read_json(path)
    if not data:
        return None
    metric = 'legal_survival' if role == 0 else 'double_capture'
    episodes = data.get('episodes', [])
    values = [row[metric] for row in episodes if metric in row]
    if not values:
        return None
    return dict(rate=sum(values)/len(values), n=len(values), path=str(path), score=data.get('score', 0.))


def snapshot(output, activity, now=None):
    now = time.time() if now is None else now
    config = read_json(output/'invocation.json', {})
    fork = read_json(output/'fork.json', {})
    source = Path(fork['source']).parent if fork.get('source') else output
    baseline_source = source
    visited = set()
    while not (baseline_source/'validation_role0_ep0.json').exists() and baseline_source not in visited:
        visited.add(baseline_source)
        ancestor = read_json(baseline_source/'fork.json', {})
        if not ancestor.get('source'):
            break
        baseline_source = Path(ancestor['source']).parent
    records = read_metrics(output/'train_metrics.jsonl')
    counters = list(fork.get('source_counters', [0, 0]))
    cost = [[], []]
    for row in records:
        role = row['role']
        count = row['episodes'][role]-counters[role]
        if count > 0 and row.get('seconds', 0) > 0:
            cost[role].append(row['seconds']/count)
        counters = row['episodes']
    goal = int(config.get('episodes', 5000))
    trained = (output/'ppo_final.pt').exists() or min(counters) >= goal
    if trained:
        counters = [goal, goal]
    eta = sum(max(0, goal-counters[r])*sum(cost[r][-8:])/len(cost[r][-8:]) for r in range(2)) if all(cost) else None
    recent_validation = []
    for role in range(2):
        candidates = list(output.glob(f'validation_role{role}_ep*.json'))
        if not candidates:
            candidates = list(source.glob(f'validation_role{role}_ep*.json'))
        candidates.sort(key=lambda p: int(p.stem.rsplit('ep', 1)[1]))
        last_path = candidates[-1] if candidates else None
        latest = validation(last_path, role) if last_path else None
        baseline = validation(baseline_source/f'validation_role{role}_ep0.json', role)
        best_candidates = [validation(p, role) for p in candidates]
        best_candidates += [validation(output/f'fork_previous_best_role{role}.json', role), baseline]
        best = max((v for v in best_candidates if v), key=lambda v: v['score'], default=None)
        recent_validation.append(dict(baseline=baseline, latest=latest, best=best,
                                      at=int(last_path.stem.rsplit('ep', 1)[1]) if last_path else None))
    total_eval = int(config.get('pilot_episodes', 20))*3*2*4
    completed_eval = len(list((output/'pilot/chunks').glob('*.json')))
    gate = read_json(output/'pilot_gate.json')
    if gate is not None:
        phase = 'complete'
    elif trained:
        phase = 'evaluation' if (output/'export/policy.pt').exists() else 'export'
    else:
        phase = 'ppo'
    metrics_path = output/'train_metrics.jsonl'
    age = max(0, now-metrics_path.stat().st_mtime) if metrics_path.exists() else None
    next_role = 0 if counters[0] <= counters[1] else 1
    batch = min(int(config.get('episodes_per_update', 16)), max(0, goal-counters[next_role]))
    return dict(config=config, counters=counters, goal=goal, eta=eta, records=records,
                validation=recent_validation, phase=phase, age=age, next_role=next_role,
                batch=batch, evaluation=[completed_eval, total_eval], gate=gate, activity=activity)


def duration(seconds):
    if seconds is None:
        return '等待样本'
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h}小时{m:02d}分' if h else f'{m}分{s:02d}秒'


def bar(done, goal, width=32):
    fraction = min(1., max(0., done/max(goal, 1)))
    full = int(width*fraction)
    return '['+'█'*full+'░'*(width-full)+f'] {100*fraction:5.1f}%  {done:,}/{goal:,}'


def percentage(value):
    return '   -- ' if value is None else f'{value["rate"]*100:5.1f}%'


def spark(values):
    if not values:
        return '等待数据'
    values = values[-24:]
    low, high = min(values), max(values)
    return ''.join('▁▂▃▄▅▆▇█'[min(7, int((v-low)/max(high-low, 1e-9)*7))] for v in values)


def fit(line, width):
    result = ''; used = 0
    for char in line:
        size = 0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in ('W', 'F') else 1
        if used+size > width-1:
            return result+'…'
        result += char; used += size
    return result


def render(data, frame=0, width=110, height=40):
    activity = data['activity']
    running = activity['running']
    spinner = '◐◓◑◒'[frame % 4]
    state = f'{spinner} 正在运行' if running else '○ 未发现训练进程'
    if data['phase'] == 'complete':
        state = '✓ 试验已完成'
    pid = ','.join(map(str, activity['pids'])) or '--'
    cpu = '--' if activity['cpu'] is None else f'{activity["cpu"]:.0f}%'
    busy = '--' if activity['busy'] is None else str(activity['busy'])
    config = data['config']
    phase = data['phase']
    phases = {'ppo':'PPO 强化学习', 'export':'模型导出', 'evaluation':'独立搜索配对评估', 'complete':'完成'}
    if config.get('rollout_backend') == 'paired_role_lookahead_v1':
        phases['ppo'] = 'PPO·追逃并采'
    total, goal = sum(data['counters']), data['goal']*2
    lines = [
        f' SEARCH + LEARNING + PPO  实时训练看板                  {time.strftime("%H:%M:%S")}',
        '─'*min(100, width),
        f' {state}    PID {pid}    阶段：{phases[phase]}',
        f' 工作进程：{activity["workers"]}/{config.get("workers", "--")}   最近活跃：{busy}   合计CPU：{cpu}（100%=1核）',
        ' 流程：蒸馏 ✓  →  PPO '+('进行中' if phase == 'ppo' else '✓')+'  →  模型导出  →  独立评估',
        '',
        ' 总体训练  '+bar(total, goal),
        ' 逃逸训练  '+bar(data['counters'][0], data['goal']),
        ' 追捕训练  '+bar(data['counters'][1], data['goal']),
        f' 剩余 PPO 约 {duration(data["eta"])}（近期批次外推；后续搜索评估另计）',
    ]
    if phase == 'ppo':
        role = ('逃逸', '追捕')[data['next_role']]
        lines += [f' 当前待完成批次：{role} {data["batch"]} 局   距上次批次记录：{duration(data["age"])}',
                  ' 局数在整批结束后刷新；批内逐局完成数未记录。旋转图标/CPU活动每2秒刷新。']
    else:
        lines += [' 配对评估  '+bar(*data['evaluation']),
                  ' 评估需运行搜索对局，耗时单独计算；PPO 100%不代表整个试验结束。']
    lines += ['', ' 学得怎么样：固定蒸馏对手，纯策略验证（成功率越高越好）',
              '                  蒸馏基线      最近验证      已保留最佳      相对基线']
    for role, val in enumerate(data['validation']):
        delta = '--'
        if val['latest'] and val['baseline']:
            delta = f'{(val["latest"]["rate"]-val["baseline"]["rate"])*100:+.1f}个百分点'
        label = '逃逸合法生存率' if role == 0 else '追捕双捕获率  '
        lines.append(f' {label}    {percentage(val["baseline"])}        {percentage(val["latest"])}        {percentage(val["best"])}        {delta}')
    counts = [f'{("逃逸", "追捕")[r]}第{v["at"]}局/{v["latest"]["n"]}个验证回合' for r, v in enumerate(data['validation']) if v['latest']]
    lines += [' 最近验证：'+'；'.join(counts),
              ' 每约500局更新验证成绩；小样本会波动，最终以独立搜索配对评估判定。', '',
              ' 最近训练奖励走势（各自缩放；不等于成功率）']
    for role in range(2):
        values = [r['reward'] for r in data['records'] if r['role'] == role and 'reward' in r]
        tail = f'{values[-1]:+.4f}' if values else '--'
        lines.append(f' {("逃逸", "追捕")[role]}  {spark(values)}   最近每步团队奖励 {tail}')
    if len(activity['pids']) > 1:
        lines.append(' ! 发现多个训练主进程使用同一输出目录，请检查是否重复启动。')
    if not running and phase != 'complete':
        lines.append(' ! 当前进程视图未找到训练主进程；本页保留最后记录，请检查 train.log。')
    if data['gate'] is not None:
        lines.append(' 试验继续条件：'+('通过' if data['gate'].get('continue_formal') else '未通过，请先分析结果'))
    lines += ['', ' 只读监控，不影响训练。Ctrl+C关闭看板；tmux Ctrl+b 再按0返回原窗口。']
    if height < 29 or width < 95:
        small = max(10, min(24, width-42))
        lines = [f' PPO 训练实时看板                         {time.strftime("%H:%M:%S")}',
                 f' {state}  PID {pid}  | {phases[phase]}',
                 f' workers {activity["workers"]}/{config.get("workers", "--")}  活跃 {busy}  CPU {cpu}（100%=1核）',
                 '', ' 总体 '+bar(total, goal, small),
                 ' 逃逸 '+bar(data['counters'][0], data['goal'], small),
                 ' 追捕 '+bar(data['counters'][1], data['goal'], small),
                 f' 剩余PPO约 {duration(data["eta"])}；搜索评估另计']
        if phase == 'ppo':
            lines += [f' 待记入批次：{("逃逸", "追捕")[data["next_role"]]} {data["batch"]}局；日志距今 {duration(data["age"])}',
                      ' 整批结束刷新局数；批内完成数未记录。CPU/状态每2秒刷新。']
        else:
            lines += [' 评估 '+bar(*data['evaluation'], small), ' PPO完成后还需导出和搜索配对评估。']
        lines += ['', ' 纯策略验证：成功率越高越好（每约500局更新）',
                  '                    基线     最近     保留最佳']
        for role, val in enumerate(data['validation']):
            label = '逃逸合法生存率' if role == 0 else '追捕双捕获率  '
            lines.append(f' {label}    {percentage(val["baseline"])}    {percentage(val["latest"])}    {percentage(val["best"])}')
        at = '/'.join(str(v['at'] or '--') for v in data['validation'])
        counts = [v['latest']['n'] for v in data['validation'] if v['latest']]
        lines += [f' 最近验证局数(逃逸/追捕)：{at}；样本数 '+ '/'.join(map(str, counts)),
                  ' 小样本会波动，最终以独立搜索评估判断收益。']
        if not running and phase != 'complete':
            lines.append(' ! 未找到训练主进程，请检查 train.log。')
        if len(activity['pids']) > 1:
            lines.append(' ! 同一输出目录存在多个训练主进程。')
        if data['gate'] is not None:
            lines.append(' 试验继续条件：'+('通过' if data['gate'].get('continue_formal') else '未通过'))
        if height >= 22:
            lines.append(' 本页只读。Ctrl+C关闭；Ctrl+b 再按0返回原窗口。')
    return '\n'.join(fit(line, width) for line in lines[:max(1,height-1)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'current/search_rl_v2_parallel/pilot_seed0')
    parser.add_argument('--interval', type=float, default=2.)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--json', action='store_true', help='Print one machine-readable snapshot')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    output = args.output.resolve()
    previous = {}; frame = 0
    interactive = sys.stdout.isatty() and not (args.once or args.json)
    if interactive:
        sys.stdout.write('\033[?1049h\033[?25l')
    try:
        while True:
            activity, previous = process_activity(output, previous)
            data = snapshot(output, activity)
            if args.json:
                print(json.dumps(data, ensure_ascii=False)); break
            if interactive:
                sys.stdout.write('\033[H\033[J')
            size = shutil.get_terminal_size((110, 40))
            print(render(data, frame, size.columns, size.lines), flush=True)
            if args.once:
                break
            frame += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if interactive:
            sys.stdout.write('\033[?25h\033[?1049l'); sys.stdout.flush()


if __name__ == '__main__':
    main()
