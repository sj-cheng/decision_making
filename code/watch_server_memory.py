#!/usr/bin/env python3
"""Stop this user's benchmark when whole-server used memory exceeds a limit.

"Used" is the same number as the 已用 column of ``free``: total minus free
minus buffers and cache. File cache does not count. The watcher sends SIGINT
to the benchmark parent so finished episodes stay on disk, then escalates if
that process is still alive.
"""

import argparse
import os
import signal
import time


def memory_kib():
    info = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            key, rest = line.split(":", 1)
            info[key] = int(rest.split()[0])
    used = (
        info["MemTotal"]
        - info["MemFree"]
        - info.get("Buffers", 0)
        - info.get("Cached", 0)
        - info.get("SReclaimable", 0)
    )
    return used, info["MemTotal"], info.get("MemAvailable", 0)


def command_of(pid):
    try:
        with open("/proc/{}/cmdline".format(pid), "rb") as handle:
            raw = handle.read()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace")


def benchmark_parents(match):
    own_uid = os.getuid()
    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == os.getpid():
            continue
        try:
            status = os.stat("/proc/{}".format(pid))
        except OSError:
            continue
        if status.st_uid != own_uid:
            continue
        command = command_of(pid)
        if match not in command or "spawn_main" in command or "watch_server_memory.py" in command:
            continue
        found.append(pid)
    return found


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def descendants(pid):
    children = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        child = int(name)
        try:
            with open("/proc/{}/stat".format(child)) as handle:
                stat = handle.read()
        except OSError:
            continue
        parts = stat.rsplit(")", 1)
        if len(parts) != 2:
            continue
        fields = parts[1].split()
        if len(fields) < 2:
            continue
        if int(fields[1]) == pid:
            children.append(child)
    output = []
    for child in children:
        output.append(child)
        output.extend(descendants(child))
    return output


def signal_pids(pids, sig):
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def stop_benchmark(pid, grace_s, term_s):
    print("sending SIGINT to benchmark parent {}".format(pid), flush=True)
    signal_pids([pid], signal.SIGINT)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and alive(pid):
        time.sleep(1)
    if not alive(pid):
        print("benchmark parent {} exited after SIGINT".format(pid), flush=True)
        return
    tree = [pid] + descendants(pid)
    print("SIGINT did not stop {}; sending SIGTERM to {} processes".format(pid, len(tree)), flush=True)
    signal_pids(tree, signal.SIGTERM)
    deadline = time.monotonic() + term_s
    while time.monotonic() < deadline and any(alive(item) for item in tree):
        time.sleep(1)
    tree = [item for item in ([pid] + descendants(pid)) if alive(item)]
    if tree:
        print("sending SIGKILL to {} remaining processes".format(len(tree)), flush=True)
        signal_pids(tree, signal.SIGKILL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-gib", type=float, default=200.0)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--confirmations", type=int, default=2,
                        help="consecutive readings above the limit before stopping")
    parser.add_argument("--grace-seconds", type=float, default=60.0,
                        help="how long to wait after SIGINT for the benchmark to save and exit")
    parser.add_argument("--match", default="benchmark_baselines.py")
    args = parser.parse_args()
    if args.limit_gib <= 0 or args.interval <= 0 or args.confirmations < 1:
        parser.error("limit, interval, and confirmations must be positive")

    limit_kib = args.limit_gib * 1024 * 1024
    streak = 0
    print(
        "watching whole-server used memory; stop {} above {:.0f} GiB".format(
            args.match, args.limit_gib
        ),
        flush=True,
    )
    while True:
        used_kib, total_kib, available_kib = memory_kib()
        used_gib = used_kib / (1024 * 1024)
        over = used_kib >= limit_kib
        streak = streak + 1 if over else 0
        parents = benchmark_parents(args.match)
        print(
            "used {:.1f} GiB / {:.1f} GiB, available {:.1f} GiB, benchmark={}".format(
                used_gib,
                total_kib / (1024 * 1024),
                available_kib / (1024 * 1024),
                parents or "not running",
            ),
            flush=True,
        )
        if streak >= args.confirmations and parents:
            print(
                "used memory {:.1f} GiB has stayed at or above {:.0f} GiB".format(
                    used_gib, args.limit_gib
                ),
                flush=True,
            )
            for pid in parents:
                stop_benchmark(pid, args.grace_seconds, 15.0)
            streak = 0
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
