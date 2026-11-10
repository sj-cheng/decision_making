#!/usr/bin/env python3
"""Pause extra worker processes when whole-server used memory exceeds 220 GiB.

The formal pursuit sources are frozen, so this does not change how many jobs
the parent submits. It sends SIGSTOP to the newest worker processes, which
stops them from allocating more, and SIGCONT once used memory is back under
200 GiB. Used memory matches the 已用 column of ``free``.
"""
import argparse
import os
import signal
import time

from memory_guard import RELEASE_GIB, used_gib


def command_of(pid):
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as handle:
            raw = handle.read()
    except OSError:
        return ''
    return raw.replace(b'\0', b' ').decode('utf-8', 'replace')


def ppid_of(pid):
    try:
        stat = open(f'/proc/{pid}/stat').read()
    except OSError:
        return None
    return int(stat[stat.rfind(')') + 2:].split()[1])


def alive(pid):
    return os.path.exists(f'/proc/{pid}')


def stopped(pid):
    try:
        stat = open(f'/proc/{pid}/stat').read()
    except OSError:
        return False
    return stat[stat.rfind(')') + 2:].split()[0] == 'T'


def workers():
    """Spawned children of this user's pursuit and E49 parents."""
    own = os.getuid()
    parents = set()
    for name in os.listdir('/proc'):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            if os.stat(f'/proc/{pid}').st_uid != own:
                continue
        except OSError:
            continue
        command = command_of(pid)
        if 'spawn_main' in command or 'resource_tracker' in command:
            continue
        if 'run_large_pursuit_parallel.py' in command or 'run_e49_benchmark.py' in command:
            parents.add(pid)
    found = []
    for name in os.listdir('/proc'):
        if not name.isdigit():
            continue
        pid = int(name)
        parent = ppid_of(pid)
        if parent not in parents:
            continue
        command = command_of(pid)
        if 'spawn_main' not in command:
            continue
        found.append(pid)
    return sorted(found)


def signal_pid(pid, sig):
    try:
        os.kill(pid, sig)
    except OSError:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit-gib', type=float, default=220)
    parser.add_argument('--release-gib', type=float, default=RELEASE_GIB)
    parser.add_argument('--floor', type=int, default=8)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--confirmations', type=int, default=2)
    args = parser.parse_args()
    if args.limit_gib <= args.release_gib or args.floor < 1 or args.interval <= 0:
        parser.error('limit must exceed release, and floor and interval must be positive')
    paused = set()
    over = 0
    print(
        f'watching used memory; pause newest workers above {args.limit_gib:.0f} GiB, resume below {args.release_gib:.0f} GiB',
        flush=True,
    )
    while True:
        paused = {pid for pid in paused if alive(pid)}
        running = [pid for pid in workers() if pid not in paused and not stopped(pid)]
        used = used_gib()
        if used >= args.limit_gib:
            over += 1
        else:
            over = 0
        if over >= args.confirmations and len(running) > args.floor:
            step = max(1, len(running) // 8)
            victims = running[-step:]
            for pid in victims:
                if signal_pid(pid, signal.SIGSTOP):
                    paused.add(pid)
            print(
                f'memory guard: used {used:.1f} GiB, paused {len(victims)} workers, still running {len(running) - len(victims)}',
                flush=True,
            )
        elif used <= args.release_gib and paused:
            step = max(1, (len(running) + len(paused)) // 16)
            resume = sorted(paused)[:step]
            for pid in resume:
                if signal_pid(pid, signal.SIGCONT):
                    paused.discard(pid)
            print(
                f'memory guard: used {used:.1f} GiB, resumed {len(resume)} workers, paused left {len(paused)}',
                flush=True,
            )
        else:
            print(
                f'used {used:.1f} GiB, running workers {len(running)}, paused {len(paused)}',
                flush=True,
            )
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
