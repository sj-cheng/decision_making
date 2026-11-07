"""Durable experiment driver: checks -> profile -> pilot -> gated formal run.

Each subprocess streams into run.log. Failures stop the pipeline and leave
atomic checkpoints; rerun the identical command to resume. Formal training
starts only when the pilot's search-based continuation gate passes.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl')
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(CODE.parent/'current/search_rl_v1'))
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--pilot-only',action='store_true',help='Stop after the pilot even if the gate passes')
    args=parser.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if args.workers<=0:parser.error('workers must be positive')
    lock=(out/'run.lock').open('a')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('This output already has an active experiment driver')
    log=(out/'run.log').open('a',buffering=1)
    def status(phase,**extra):
        record=dict(phase=phase,updated_unix=time.time(),**extra)
        tmp=out/'status.json.tmp';tmp.write_text(json.dumps(record,indent=2));os.replace(tmp,out/'status.json')
        print(json.dumps(record),flush=True);log.write(json.dumps(record)+'\n')
    def run(arguments,phase):
        arguments=list(map(str,arguments))
        status(phase,command=arguments)
        result=subprocess.run([sys.executable,*map(str,arguments)],cwd=CODE,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status('failed',failed_phase=phase,exit_code=result.returncode)
            raise SystemExit(result.returncode)
    worker_args=['--workers',str(args.workers)]
    try:
        run(['-m','unittest','discover','-s','tests','-p','test_*.py','-v'],'correctness_tests')
        pilot=out/'pilot_seed0'
        from train_search_rl import implementation_hash
        profile_path=pilot/'profile.json'
        if not profile_path.exists() or json.loads(profile_path.read_text()).get('implementation_hash')!=implementation_hash():
            run(['train_search_rl.py','--stage','profile','--output',pilot,*worker_args],'profile_100_episodes_100_searches')
        if not (pilot/'pilot_gate.json').exists():
            run(['train_search_rl.py','--stage','pipeline','--output',pilot,*worker_args],'distill_and_ppo_pilot')
        gate=json.loads((pilot/'pilot_gate.json').read_text())
        if not gate['continue_formal'] or args.pilot_only:
            status('pilot_complete',continue_formal=gate['continue_formal'],formal_started=False)
            return
        evaluation_dirs=[]
        for seed in range(3):
            directory=out/f'formal_seed{seed}'
            common=['--output',directory,'--seed',str(seed),'--episodes','50000',*worker_args]
            if not (directory/'distilled.pt').exists():
                run(['train_search_rl.py','--stage','distill',*common],f'formal_seed{seed}_distill')
            if not (directory/'ppo_final.pt').exists():
                run(['train_search_rl.py','--stage','ppo',*common],f'formal_seed{seed}_ppo')
            run(['train_search_rl.py','--stage','export',*common],f'formal_seed{seed}_export')
            for budget in (200,500,1000):
                evaluation=directory/f'evaluation_B{budget}';evaluation_dirs.append(str(evaluation))
                methods=['distilled','ppo','search_distilled','search_ppo'] if budget==1000 else ['search_distilled','search_ppo']
                run(['benchmark_local_obstacles.py','--checkpoint',directory/'export/policy.pt',
                     '--reference',directory/'distilled.pt','--output',evaluation,'--budget',str(budget),
                     '--methods',*methods,*worker_args],f'formal_seed{seed}_evaluate_B{budget}')
            run(['benchmark_local_obstacles.py','--checkpoint',directory/'export/policy.pt','--reference',directory/'distilled.pt',
                 '--output',directory/'empty_map','--layouts','none','--episodes','100',*worker_args],f'formal_seed{seed}_empty_map')
            # Fair scratch control: same fixed IL opponent and no IL anchor,
            # actor weights are random. Training histories use the same seeds.
            scratch=out/f'scratch_seed{seed}'
            if not (scratch/'ppo_final.pt').exists():
                run(['train_search_rl.py','--stage','ppo','--scratch','--anchor','0','--input',directory/'distilled.pt',
                     '--output',scratch,'--seed',str(seed),'--episodes','50000',*worker_args],f'scratch_seed{seed}_ppo')
            run(['benchmark_local_obstacles.py','--checkpoint',scratch/'ppo_final.pt','--reference',directory/'distilled.pt',
                 '--output',scratch/'evaluation','--methods','scratch',*worker_args],f'scratch_seed{seed}_evaluate')
            ablation=out/f'no_belief_seed{seed}'
            ablation_common=['--output',ablation,'--seed',str(seed),'--without-belief','--episodes','50000',*worker_args]
            if not (ablation/'distilled.pt').exists():
                run(['train_search_rl.py','--stage','distill',*ablation_common],f'no_belief_seed{seed}_distill')
            if not (ablation/'ppo_final.pt').exists():
                run(['train_search_rl.py','--stage','ppo',*ablation_common],f'no_belief_seed{seed}_ppo')
            run(['benchmark_local_obstacles.py','--checkpoint',ablation/'ppo_final.pt','--reference',directory/'distilled.pt',
                 '--output',ablation/'evaluation','--methods','no_belief',*worker_args],f'no_belief_seed{seed}_evaluate')
            if seed==0:
                from learning.local_rl import PolicyBundle
                legacy=PolicyBundle();legacy.save(out/'legacy.pt')
                run(['benchmark_local_obstacles.py','--checkpoint',out/'legacy.pt','--reference',directory/'distilled.pt',
                     '--output',out/'legacy_evaluation','--methods','legacy','oracle_search','mappo_v4',*worker_args],'global_and_legacy_references')
        # Separate budget strata: never mix B=200 and B=1000 into one estimate.
        for budget in (200,500,1000):
            selected=[path for path in evaluation_dirs if path.endswith(f'B{budget}')]
            run(['benchmark_local_obstacles.py','--aggregate',*selected,'--output',out/f'aggregate_B{budget}'],f'aggregate_B{budget}')
        status('formal_complete',training_seeds=3)
    finally:
        log.close()


if __name__=='__main__':main()
