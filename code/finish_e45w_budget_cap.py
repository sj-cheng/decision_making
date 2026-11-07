"""Close a validation-only failure at W's exhausted cap, without training.

B29's true last actor was never serialized. Use the last recoverable full
checkpoint as an explicitly degraded endpoint, charge ALL spent steps, and
revalidate all retained candidates before unsealing the registered test.
Original sources, failures, checkpoints and optimizer tensors stay untouched.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import argparse
import copy
import fcntl
import multiprocessing as mp
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import torch
from stable_baselines3.common.save_util import load_from_zip_file
import run_e45w_joint_training as train
from learning.e45w_support_recovery import install,REVISION

ROOT=train.ROOT;OUT=ROOT/'current/e45w_joint_ppo_and_planning_20261103'
STATE=OUT/'repair_coordinator';DEST=OUT/'repaired_training/B_29_finalize1'
BLOCK=train.FORMAL_N_ENVS*train.FORMAL_N_STEPS;CAP=BLOCK*train.FORMAL_ROLLOUTS
ACTORS={}


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def finalization_budget(status,base,failure):
    if status['status']!='failed' or status['counts']['team_transitions']!=CAP:
        raise RuntimeError('Only a failed, fully spent cap may be finalized')
    if status['counts']['rollouts_collected']!=train.FORMAL_ROLLOUTS:
        raise RuntimeError('Sampling completion not proven')
    if 'terminal = audit.evaluate(model.num_timesteps)' not in failure['traceback']:
        raise RuntimeError('Failure is not certified terminal validation')
    if not 0<=base<=CAP or base%BLOCK:raise RuntimeError('Invalid recoverable checkpoint')
    return dict(cap=CAP,checkpoint_step=base,source_reported_step=CAP,
        charged_attempt_steps=CAP,charged_lost_steps=CAP-base,target_logical_steps=base,
        maximum_lifetime_team_steps=CAP,new_training_team_steps=0,new_optimizer_steps=0,
        source_sampling_completed=True,failure_phase='terminal validation before final checkpoint save',
        true_cap_actor_available=False,recovered_latest_checkpoint_not_true_final_actor=True,
        scope='No fictitious crash collection tick during validation; all actual spent sampling charged')


def init_worker(paths,evidence,queue):
    os.sched_setaffinity(0,{queue.get()});torch.set_num_threads(1)
    install(evidence);train.init_evaluation_worker(paths)


def evaluate(job):
    step,path,spec,index=job
    if step not in ACTORS:ACTORS[step]=torch.load(path,map_location='cpu',weights_only=False)
    return step,index,train.evaluate_frozen_actor(ACTORS[step],spec)


def finalize(source,manifest,workers):
    source_status=train.read(source/'status.json');failure=train.read(source/'failure.json')
    checkpoints=sorted((source/'checkpoints').glob('*/model.zip'));base=int(checkpoints[-1].parent.name)
    budget=finalization_budget(source_status,base,failure)
    DEST.mkdir(parents=True,exist_ok=False)
    files=[source/'status.json',source/'failure.json',source/'validation.jsonl']
    for checkpoint in checkpoints:files.extend([checkpoint,checkpoint.parent/'actor.pt',checkpoint.parent/'rng_audit.pt'])
    inputs={str(p):train.sha(p) for p in files}
    train.write(DEST/'finalization_protocol.json',dict(started=now(),repair_budget=budget,
        input_sha256=inputs,read_test_scores=False,policy_training=False,
        endpoint_caveat='True cap actor unavailable; endpoint is latest recoverable 1843200-step checkpoint.'))
    for checkpoint in checkpoints:
        _,parameters,_=load_from_zip_file(checkpoint,device='cpu')
        archived=torch.load(checkpoint.parent/'actor.pt',map_location='cpu',weights_only=False)
        actor={k[len('actor_network.'):]:v for k,v in parameters['policy'].items() if k.startswith('actor_network.')}
        assert train.tensor_hash(actor)==train.tensor_hash(archived),checkpoint
    specs=train.evaluation_specs(manifest['validation_entries'],2,False)+train.evaluation_specs(manifest['validation_entries'],1,True)
    jobs=[(int(p.parent.name),str(p.parent/'actor.pt'),s,i) for p in checkpoints for i,s in enumerate(specs)]
    ctx=mp.get_context('spawn');queue=ctx.Queue();workers=min(workers,48)
    for cpu in range(workers):queue.put(cpu)
    rows={};start=time.time();pool=ProcessPoolExecutor(max_workers=workers,mp_context=ctx,
        initializer=init_worker,initargs=([r['path'] for r in manifest['validation_entries']],str(DEST/'belief_failures'),queue))
    failed=True
    try:
        active=[pool.submit(evaluate,j) for j in jobs];done=0
        for future in as_completed(active):
            step,index,row=future.result();rows.setdefault(step,{})[index]=row;done+=1
            if done%8==0 or done==len(jobs):
                train.write(DEST/'progress.json',dict(phase='validation-only finalization',completed=done,
                    planned=len(jobs),workers=workers,elapsed_seconds=time.time()-start,training_team_steps_added=0))
        failed=False
    finally:
        processes=list((pool._processes or {}).values());pool.shutdown(wait=not failed,cancel_futures=failed)
        if failed:
            for process in processes:
                if process.is_alive():process.terminate()
            for process in processes:process.join(timeout=2)
        queue.close()
    best=None;latest=None
    for checkpoint in checkpoints:
        step=int(checkpoint.parent.name);values=[rows[step][i] for i in range(len(specs))]
        stochastic=[r for r in values if not r['deterministic']];det=[r for r in values if r['deterministic']]
        primary=train.score(stochastic);diagnostic=train.score(det)
        record=dict(team_transitions=step,stochastic=primary,deterministic_diagnostic=diagnostic,
            primary_mode='stochastic',revalidated_saved_candidate=True,optimizer_updates_added=0)
        with (DEST/'validation.jsonl').open('a') as f:
            import json
            f.write(json.dumps(record)+'\n')
        train.write(DEST/'validation_episodes'/f'{step:08d}.json',dict(stochastic=stochastic,deterministic=det))
        if best is None or primary['utility']>best['utility']:
            best=dict(team_transitions=step,utility=primary['utility'],double_capture_rate=primary['double_capture_rate'],
                mode='stochastic',checkpoint=str(checkpoint))
            shutil.copy2(checkpoint.parent/'actor.pt',DEST/'selected_actor.pt')
        latest=record
    train.write(DEST/'selection.json',best)
    terminal=checkpoints[-1];shutil.copy2(terminal.parent/'actor.pt',DEST/'terminal_actor.pt')
    shutil.copy2(checkpoints[0].parent/'actor.pt',DEST/'initial_actor.pt')
    shutil.copy2(source/'initialization_audit.json',DEST/'initialization_audit.json')
    train.write(DEST/'terminal_checkpoint.json',dict(model=str(terminal),team_transitions=base,
        recovered_latest_checkpoint_not_true_final_actor=True,true_cap_actor_available=False))
    saved=torch.load(terminal.parent/'rng_audit.pt',map_location='cpu',weights_only=False)
    counts=copy.deepcopy(saved['counts']);counts['validation_episodes']+=len(jobs)
    counts['validation_environment_steps']+=sum(r['decisions'] for group in rows.values() for r in group.values())
    assert counts['team_transitions']==base
    train.write(DEST/'status.json',dict(status='complete',pid=os.getpid(),arm='B',seed=29,device='cpu',counts=counts,
        SB3_reported_epoch_updates=saved['SB3_reported_epoch_updates'],latest_validation=latest,best_validation=best,
        repair_budget=budget,repair_revision=REVISION,training_budget_audit_passed=True,
        policy_optimizer_updates_added=0,additional_training_team_steps=0,endpoint_recovered_not_true_final=True,
        source_counts_at_cap=source_status['counts'],resume_source=str(source),test_episodes_evaluated=0))
    for path,h in inputs.items():assert train.sha(path)==h,path
    train.write(DEST/'completion_audit.json',dict(complete=True,input_hashes_unchanged=True,validation_episodes=len(jobs),
        validation_physical_steps=sum(r['decisions'] for group in rows.values() for r in group.values()),
        optimizer_updates_added=0,training_team_steps_added=0,true_cap_actor_available=False,
        elapsed_seconds=time.time()-start))


def main(args):
    lock=(STATE/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    archive=STATE/'cap_failure_before_finalization_20261105';archive.mkdir(exist_ok=False)
    for name in ('registry.json','progress.json'):shutil.copy2(STATE/name,archive/name)
    sources={str(p):train.sha(p) for p in [Path(__file__),ROOT/'code/tests/test_e45w_budget_cap.py',
        ROOT/'code/E45W_BUDGET_CAP_FINALIZATION_20261105.md']}
    train.write(archive/'new_source_manifest.json',dict(files=sources,no_scientific_recipe_change=True))
    manifest=train.read(train.MANIFEST);train.validate_manifest(manifest)
    def integrity():
        for path,h in sources.items():assert train.sha(path)==h,path
        for file in (OUT/'component_source_manifest.json',STATE/'source_manifest.json'):
            for path,h in train.read(file)['files'].items():assert train.sha(path)==h,path
    integrity()
    registry=train.read(STATE/'registry.json');old=train.read(STATE/'progress.json')
    for name,row in registry.items():
        if name!='B_29':assert train.read(Path(row['directory'])/'status.json')['status']=='complete',name
    train.write(STATE/'progress.json',old|dict(state='budget_cap_validation_finalization',timestamp=now(),finalization_directory=str(DEST)))
    finalize(OUT/'training/B_29',manifest,args.workers)
    registry['B_29']=dict(directory=str(DEST),attempt=2,status=train.read(DEST/'status.json'),
        source=str(OUT/'training/B_29'),kind='evaluation-only finalization; no extra training',
        prior_failed_attempt_preserved=str(archive),pid=os.getpid(),exit_code=0)
    for row in registry.values():row['status']=train.read(Path(row['directory'])/'status.json')
    train.write(STATE/'registry.json',registry);integrity()
    assert all(r['status']['status']=='complete' and r['status']['training_budget_audit_passed'] for r in registry.values())
    train.write(OUT/'test_scores_unlock_repair1.json',dict(unlocked=True,all_six_complete=True,
        registry_sha256=train.sha(STATE/'registry.json'),revision=REVISION,timestamp=now(),
        B29_true_cap_actor_available=False,B29_latest_recoverable_step=1843200,B29_new_training_steps=0))
    log=(OUT/'logs/final_evaluation_repair1.log').open('a')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='3'
    command=[str(OUT/'venv/bin/python'),'-u','code/run_e45w_repaired_evaluation.py','--workers','40']
    process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
    while True:
        integrity();status_path=OUT/'final_evaluation_repair1/status.json'
        status=train.read(status_path) if status_path.exists() else None
        data=old|dict(state='final_evaluation_running',registry=registry,timestamp=now(),
            final_pid=process.pid,final_status=status,test_scores_unlocked=True,
            finalization_directory=str(DEST),cap_endpoint_caveat='B29 uses latest recoverable checkpoint, not missing true endpoint')
        if process.poll() is not None:
            data.update(state='finished' if process.returncode==0 else 'final_evaluation_failed',final_exit_code=process.returncode)
            train.write(STATE/'progress.json',data)
            if process.returncode:raise RuntimeError('Final evaluation failed; inspect preserved log')
            break
        train.write(STATE/'progress.json',data);time.sleep(10)
    log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=48)
    main(parser.parse_args())
