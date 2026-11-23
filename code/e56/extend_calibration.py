"""Run the second paired calibration seed alongside the original E56 screening."""
import argparse
import fcntl
import json
from pathlib import Path

from start_e56 import SPEEDS, batch, digest, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('screen', type=Path)
    parser.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.screen / 'manifest.json').read_text())
    code = Path(__file__).resolve().parents[1]
    for name, expected in manifest['sources'].items():
        assert any(digest(p) == expected for p in code.rglob(name)), name
    dest = args.screen / 'calibration_second_seed'
    dest.mkdir(exist_ok=True)
    with (dest / 'batch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cases = []
        for original in manifest['cases']:
            assert digest(Path(original['path'])) == original['sha']
            cases.append(dict(original, seed={17:43, 43:17}[original['seed']]))
        arms = [(speed, 'none', 256) for speed in SPEEDS]
        save(dest / 'design.json', dict(cases=cases, arms=arms,
             purpose='paired replication; primary calibration remains the original 16 cases'))
        rows = batch(cases, arms, dest, args.workers)
        save(dest / 'results.json', rows)


if __name__ == '__main__':
    main()
