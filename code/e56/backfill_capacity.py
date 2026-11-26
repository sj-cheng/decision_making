"""Use spare CPU cycles for unlocked capacity trials."""
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from network_scale_trials import OUT, SEEDS, steal

if __name__ == '__main__':
    cases=json.loads((OUT/'manifest.json').read_text())['dev']
    tasks=[]
    for seed in (73,97,52):
        for cap in ('xxl','xl','large'):
            method=f'expanded_{cap}_{seed}'
            for case in cases:
                if not (OUT/'episodes'/method/'b64'/case['case']/'result.json').exists():
                    tasks.append((method,64,case))
    with ProcessPoolExecutor(8,mp_context=get_context('spawn')) as pool:
        for _ in pool.map(steal,tasks):pass
