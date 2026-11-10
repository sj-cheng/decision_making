"""Exact execution equivalence on engineering fixtures, never sealed maps."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import concurrent.futures
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--cpus',default='40,41,42,43');args=parser.parse_args()
    cpus=list(map(int,args.cpus.split(',')));os.sched_setaffinity(0,{cpus[0]})
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    from large_pursuit import training as canonical
    from large_pursuit.core import PublicMap,Config,World,from_initial,Belief
    from large_pursuit.provenance import sources
    from pursuit_parallel import runtime
    root=args.root.resolve();out=root/'execution/equivalence';out.mkdir(parents=True,exist_ok=True)
    before=runtime.runtime_sources();algorithm=sources();checks=[]
    context=multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=context,
            initializer=runtime.worker_init,initargs=(cpus,)) as executor:
        parallel_teacher=runtime.make_teacher(executor,4)
        for n in (5,16):
            case=json.loads((root/'verification/bank'/f'{n}_train.json').read_text())
            public=PublicMap(Config.scale(n),case['boxes']);state=from_initial(case,0)
            belief=Belief(public,19);obs=World(public.cfg,public.boxes).observe(state);belief.update(obs)
            pool=canonical.candidate_pool(public,state.p,state.e,state.alive,np.random.default_rng(7))
            rng=np.random.default_rng(113);before_rng=copy.deepcopy(rng.bit_generator.state)
            before_belief=pickle.dumps(canonical.belief_state(belief));start=time.perf_counter()
            expected=runtime.SERIAL_TEACHER(public,obs,belief,pool,rng,2,2)
            serial_seconds=time.perf_counter()-start;after_rng=copy.deepcopy(rng.bit_generator.state)
            rng.bit_generator.state=before_rng;start=time.perf_counter()
            actual=parallel_teacher(public,obs,belief,pool,rng,2,2);parallel_seconds=time.perf_counter()-start
            np.testing.assert_array_equal(expected,actual)
            assert pickle.dumps(rng.bit_generator.state)==pickle.dumps(after_rng)
            assert pickle.dumps(canonical.belief_state(belief))==before_belief
            checks.append(dict(check='teacher',n=n,exact_scores=True,exact_parent_rng=True,unmodified_posterior=True,
                               serial_seconds=serial_seconds,parallel_seconds=parallel_seconds,workers=4))
            print('PASS teacher exact scores/RNG/posterior n='+str(n),flush=True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA validation equivalence also required')
    checkpoint=root/'verification/cuda_5_pursuer/uninterrupted/best.pt'
    for device in ('cpu','cuda:0'):
        for role in ('pursuer','evader'):
            model=canonical.load_model(checkpoint,device)
            folder=out/(device.replace(':','_')+'_'+role);folder.mkdir(exist_ok=True)
            job=dict(bank=str(root/'verification/bank'),n=5,branch='joint',seed=619,role=role,
                     folder=str(folder),validation_split='validation',validation_maps=1,verification=True,
                     validation_simulations=2,search_depth=2,gpu=3)
            job['opponents']=([dict(method='D_kernel'),dict(method='learned_evader',checkpoint=str(checkpoint),seed=619,round=0)]
                              if role=='pursuer' else [dict(method='AG'),dict(method='NET',checkpoint=str(checkpoint))])
            cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
            original_transition=World.transition;digests=[]
            def traced(self,state,pa,ea):
                nxt=original_transition(self,state,pa,ea)
                if sys._getframe(1).f_code is runtime.SERIAL_VALIDATION.__code__:
                    if state.t==0:digests.append(hashlib.sha256())
                    runtime.trace_update(digests[-1],state,pa,ea,nxt)
                return nxt
            World.transition=traced;started=time.perf_counter()
            try:expected=runtime.SERIAL_VALIDATION(model,job,device)
            finally:World.transition=original_transition
            serial_seconds=time.perf_counter()-started
            expected_cpu_rng=torch.get_rng_state();expected_cuda_rng=torch.cuda.get_rng_state()
            torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng);started=time.perf_counter()
            actual=runtime.parallel_validation(model,job,device,cpus,2,root)
            parallel_seconds=time.perf_counter()-started
            assert actual==expected,(actual,expected)
            assert torch.equal(expected_cpu_rng,torch.get_rng_state()),'Parent CPU RNG differs'
            assert torch.equal(expected_cuda_rng,torch.cuda.get_rng_state()),'Parent CUDA RNG differs'
            logs=sorted((folder/'execution').glob('validation_*.json'))
            rows=json.loads(logs[-1].read_text())['rows']
            assert [d.hexdigest() for d in digests]==[row[2]['trajectory_sha256'] for row in rows],'Trajectory differs'
            checks.append(dict(check='validation',device=device,role=role,exact_result=True,exact_trajectories=True,
                               exact_parent_cpu_cuda_rng=True,serial_seconds=serial_seconds,parallel_seconds=parallel_seconds,workers=2))
            print('PASS validation exact trajectories/results/CPU+CUDA RNG '+device+' '+role,flush=True)
    assert runtime.runtime_sources()==before and sources()==algorithm,'Source changed during equivalence proof'
    canonical.write_json(root/'execution/parallel_equivalence.json',dict(status='passed',algorithm_sources=algorithm,
        runtime_sources=before,checks=checks,engineering_only=True,sealed_used=False,
        scope='Exact per-candidate scores and RNG at both scales; full two-initial validation trajectories and parent CPU/CUDA RNG for both roles/devices',
        checkpoint_compatibility='Scientific sources, task identities and optimizer/environment checkpoint representation unchanged',
        timing_claim=False))
    print('ALL PARALLEL EQUIVALENCE CHECKS PASSED',flush=True)


if __name__=='__main__':main()
