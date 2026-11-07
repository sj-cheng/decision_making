"""Run the frozen P training functions with an audited execution-only repair."""
import os
from pathlib import Path
import subprocess
import sys
import time
import run_e45p_hasac_bc_anchor as p
from learning.e45p_execution_precision import pool_for


def run_children(root,cfg,args,phase,started):
    original_start=p.j.read(root/'precision_amendment.json')['original_launch_unix']
    combos=[('bc',s) for s in cfg['seeds']] if phase=='bc' else [(a,s) for a in cfg['arms'] for s in cfg['seeds']]
    cores=p.j.choose_cpus(len(combos)*(args.workers+1));assert len(cores)==len(combos)*(args.workers+1)
    children=[];allocation=[]
    try:
        for i,(arm,seed) in enumerate(combos):
            folder=root/'bc'/f'HASAC_{seed}' if phase=='bc' else root/'runs'/f'{arm}_{seed}'
            folder.mkdir(parents=True,exist_ok=True)
            if (folder/'decision.json').exists():continue
            use=cores[i*(args.workers+1):(i+1)*(args.workers+1)]
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'bc' if phase=='bc' else 'rl',
                '--output',str(root),'--seed',str(seed),'--arm',arm,'--device',args.device,
                '--cpus',','.join(map(str,use)),'--parent',str(os.getpid())]
            with (folder/'run.log').open('a') as log:child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            children.append(child);allocation.append(dict(arm=arm,seed=seed,pid=child.pid,cores=use,device=args.device))
        p.j.save(root/f'execution_{phase}.json',dict(allocation=allocation,workers_per_model=args.workers,precision_amendment=True))
        while any(child.poll() is None for child in children):
            p.report(root)
            if any(child.poll() not in (None,0) for child in children):raise RuntimeError('Amended P child failed; inspect retained logs')
            if time.time()-original_start>cfg['maximum_wall_seconds']:raise RuntimeError('Original registered wall cap reached')
            time.sleep(5)
        assert all(child.returncode==0 for child in children)
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()


if __name__=='__main__':
    # The original experiment and source remain immutable. Amendment hashes
    # are additional provenance, verified in every new parent/child process.
    args=sys.argv
    root=Path(args[args.index('--output')+1]) if '--output' in args else p.OUT
    amendment=p.j.read(root/'precision_amendment.json')
    for name,digest in amendment['source_hashes'].items():assert p.j.sha(name)==digest
    assert p.j.sha(root/'experiment.json')==amendment['experiment_sha256']
    p.o.pool_for=pool_for;p.run_children=run_children
    p.main()
