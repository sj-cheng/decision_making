"""Use released experiment cores to finish counterfactuals and their online tests."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import torch
import advanced_network_runs as runs

ROOT = runs.OUT
CASES = sorted(runs.TRAIN + runs.VALID, key=lambda c:(c['case'][-3:],c['case']), reverse=True)


def claim(task):
    arm,case=task
    parent=ROOT/'pairs' if arm=='collect' else ROOT/'eval'/arm
    folder=parent/case['case']
    result=folder/'result.json'
    if result.exists():return
    locks=ROOT/'backfill_locks'
    locks.mkdir(exist_ok=True)
    with (locks/f"{arm}_{case['case']}").open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        if result.exists():return
        if arm!='collect':
            runs.evaluate(task)
            return
        # The original pool works from the opposite end. Leave all of its
        # started cases there, including partly completed reference replays.
        if folder.exists():return
        with TemporaryDirectory(dir=ROOT) as tmp:
            runs.OUT=Path(tmp)
            try:runs.collect(case)
            finally:runs.OUT=ROOT
            src=Path(tmp)/'pairs'/case['case']
            folder.mkdir(exist_ok=True)
            for name in ('reference','cut','alt','result'):
                f=src/(name+'.json')
                if f.exists():
                    try:os.link(f,folder/f.name)
                    except FileExistsError:pass


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-pid',type=int)
    args=parser.parse_args()
    while args.wait_pid and Path(f'/proc/{args.wait_pid}').exists():time.sleep(10)
    torch.set_num_threads(1)
    with ProcessPoolExecutor(len(os.sched_getaffinity(0))) as pool:
        list(pool.map(claim,[('collect',c) for c in CASES]))
        while not (ROOT/'value_ready.json').exists() or not (ROOT/'stop_ready.json').exists():time.sleep(15)
        list(pool.map(claim,[(arm,c) for c in reversed(runs.DEV) for arm in ('learned_stop','value')]))
