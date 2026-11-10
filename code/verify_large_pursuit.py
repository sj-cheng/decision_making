"""Closed-loop engineering acceptance using scale-specific functional-test models."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
from pathlib import Path
from run_large_pursuit import read,write,resources,cpu_batch,DEFAULT
from large_pursuit.methods import METHODS,METHOD_BRANCH


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT);args=parser.parse_args();out=args.output.resolve()
    manifest=read(out/'verification/bank/manifest.json');layout=resources();jobs=[]
    for n in (5,16):
        entry=next(e for e in manifest['entries'] if e['n']==n and e['split']=='train')
        checkpoint=out/'verification'/f'cuda_{n}_pursuer'/'uninterrupted'/'best.pt'
        if not checkpoint.exists():raise RuntimeError('Scale-specific functional-test model missing')
        for method in METHODS:
            jobs.append(dict(kind='evaluate',case=str(out/'verification/bank'/entry['path']),initial=0,method=method,seed=991,
                checkpoint=str(checkpoint) if method in METHOD_BRANCH else None,opponent='D_kernel',budget='engineering_sim2',simulations=2,budget_ms=None,
                environment_seed=91291,folder=str(out/'verification/closed_loop'/f'{n}_{method}')))
    cpus=layout['bulk'];rows=cpu_batch(out,jobs,cpus,min(16,len(cpus)),'repair_closed_loop_all_methods')
    write(out/'closed_loop_acceptance.json',dict(passed=len(rows)==2*len(METHODS),episodes=len(rows),scales=[5,16],methods=METHODS,
        sealed_used=False,formal_comparison=False,scale_specific_models=True,
        scope='Functional models only; discarded before fresh pilots and formal training. No scientific performance claim.',
        rows=[dict(n=r['n'],method=r['method'],decisions=r['decisions'],trajectory=r['trajectory']) for r in rows]))

if __name__=='__main__':main()
