"""Split capacity training by architecture after the in-flight XL seed completes."""
import argparse
import json
import os
import signal
import time
from pathlib import Path

import torch
from network_geometry_experiment import data
from network_scale_trials import OUT, SEEDS, batch, fit, model_folder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lane', choices=['xl', 'xxl', 'eval'])
    parser.add_argument('--handover', type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads((OUT/'manifest.json').read_text())
    if args.handover:
        pid = args.handover
        cmd = Path(f'/proc/{pid}/cmdline')
        identity = cmd.read_bytes()
        assert b'network_scale_trials.py\0larger_models\0' in identity
        while not (model_folder('expanded','xl',52)/'done.json').exists() and cmd.exists():
            time.sleep(.5)
        if cmd.exists() and cmd.read_bytes() == identity:
            os.kill(pid, signal.SIGTERM)
    if args.lane == 'eval':
        # The original evaluation pool consumes the same locked jobs from the front.
        for seed in reversed(SEEDS):
            for cap in ('large','xxl','xl'):
                model = model_folder('expanded',cap,seed)
                while not (model/'done.json').exists():
                    time.sleep(10)
                batch([(model.name,64,c) for c in reversed(manifest['dev'])])
    else:
        splits = [manifest['small']+manifest['extra'],manifest['valid']]
        datasets = []
        for split in splits:
            x,y,mask=data(split)
            datasets.append((x,y,torch.zeros_like(y),mask))
        for seed in (SEEDS if args.lane=='xl' else reversed(SEEDS)):
            print(f'{args.lane} seed {seed}',flush=True)
            fit('expanded',args.lane,seed,manifest,datasets)
