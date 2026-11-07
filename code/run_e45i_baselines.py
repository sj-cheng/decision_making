"""E45-I: author RMAPPO / public APF pilot and a common nine-family benchmark."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('TZ', 'Asia/Shanghai')
import argparse
import collections
import concurrent.futures as futures
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
from learning.e45i_official_mappo import (policy, APFController, RecurrentController,
    DEFAULT_APF, finish_episodes, UPSTREAM)
from learning.e45g_ppo import install_belief, safe_action, execution_legal_action
from solvers.e45b_joint_nte import coverage_potential
from e45g_training_acceleration import public_geometry_cache
from e45c_capture_audit import audit_simulated
from e45f_scenarios import ID_FAMILIES, OOD_FAMILIES, new_id_case, shifted_case, geometry_key
from run_frozen_d_closed_loop import save, sha
from run_e44_parallel import choose_cpus
import run_e45b_role_free as base
import run_e45f_generalization_benchmark as reference

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'current/e45i_official_baselines_20261031'
CROOT = ROOT/'current/e45c_budget_scaling_20261028'
BASE_SEED = 946120261031
SEEDS = (17, 29, 43)
RL = ('RMAPPO', 'MAPPO_APF')
STATIC = ('SINGLE_PRIOR', 'APF', 'AG', 'IG', 'SINGLE_DIRECT', 'U', 'GEOM_SEARCH')
ARMS = ('SINGLE_PRIOR', 'RMAPPO', 'MAPPO_APF', 'APF', 'AG', 'IG', 'SINGLE_DIRECT', 'U', 'GEOM_SEARCH')
SOURCES = ('run_e45i_baselines.py', 'learning/e45i_official_mappo.py',
           'tests/test_e45i_baselines.py', 'E45I_BASELINES_PROTOCOL_20261031.md')
CAPTURE = None
ROLLOUT_ACTOR = None
ACTORS = {}


def read(path):
    return json.loads(Path(path).read_text())


def emit(folder, stage, **values):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    row = dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'), pid=os.getpid(), stage=stage, **values)
    save(folder/'status.json', row)
    with (folder/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)


def coefficients():
    path = OUT/'apf_selected.json'
    return read(path)['coefficients'] if path.exists() else DEFAULT_APF.tolist()


def configure(output):
    global OUT
    OUT = Path(output); base.OUT = OUT; base.SMOKE = False; base.make_controller = factory
    base.audit_simulated = audit_simulated; base.legal_action = execution_legal_action
    reference.OUT = OUT
    install_belief(OUT/'belief_failures')


def factory(case, task):
    global CAPTURE
    arm = task['arm']; gains = task.get('apf_coefficients', coefficients())
    if arm == 'APF':
        own = APFController(case['map'], case['speed'], task['seed'], gains)
    elif arm in RL:
        if task['stage'] == 'training':
            actor = ROLLOUT_ACTOR
        else:
            key = (arm, task['checkpoint_sha256'])
            if key not in ACTORS:
                model, _ = policy(arm)
                model.actor.load_state_dict(torch.load(task['checkpoint'], map_location='cpu', weights_only=False)['actor'])
                ACTORS[key] = model.actor
            actor = ACTORS[key]
        own = RecurrentController(case['map'],case['speed'],task['seed'],actor,arm,
                                  deterministic=task['stage']!='training',coefficients=gains)
        if task['stage'] == 'training':
            CAPTURE = own
    else:
        own = reference.factory(case,task)
    original = own.act
    def act(obs, fixed_simulations=None):
        result = original(obs,fixed_simulations)
        result, changed = safe_action(np.asarray(obs['positions']),result/own.public.speed,
            obs['active_p'],own.public.speed,own.public.boxes)
        own.last_stats['outer_safety_shortened'] = changed
        return result
    own.act = act
    return own


def verify():
    config = read(OUT/'experiment.json')
    for section in ('source_hashes','protected_hashes','bank_hashes','upstream_hashes'):
        for path,digest in config[section].items():
            if sha(path) != digest:
                raise RuntimeError('Registered artifact changed: '+path)
    if (OUT/'apf_freeze.json').exists():
        if sha(OUT/'apf_selected.json')!=read(OUT/'apf_freeze.json')['sha256']:
            raise RuntimeError('Selected fixed APF parameters changed')
    return config


def prepare(steps=100_000, smoke=False):
    if (OUT/'experiment.json').exists():
        return verify()
    OUT.mkdir(parents=True,exist_ok=True)
    source = read(CROOT/'bank/manifest.json')['entries']
    train = [e for e in source if e['split']=='train']
    validation = [e for e in source if e['split']=='validation']
    known = set()
    for directory in ROOT.glob('current/e45*/bank'):
        for path in directory.glob('*/*.json'):
            packet = read(path)
            if 'map' in packet:
                known.add(geometry_key(packet['map']))
    test = []; rejected = []
    # A bounded PILOT: six ID and eight OOD public histories, not 952 maps.
    for family in ID_FAMILIES+OOD_FAMILIES:
        builder = new_id_case if family in ID_FAMILIES else shifted_case
        accepted = 0
        for index in range(1200):
            try:
                case = builder(family,index,'test',BASE_SEED+int(smoke)*1_000_000)
                digest = geometry_key(case['map'])
                if digest in known:
                    raise ValueError('Historical geometry duplicate')
            except (AssertionError,ValueError) as exc:
                rejected.append(dict(family=family,index=index,error=str(exc)));continue
            known.add(digest)
            path = OUT/'bank/test'/(case['id']+'.json');save(path,case)
            test.append(dict(id=case['id'],path=str(path),family=family,
                split='test',suite='benchmark' if family in ID_FAMILIES else 'generalization',
                pattern=case['pattern'],density=case['density']))
            accepted += 1
            if accepted == (1 if smoke else 2):
                break
        else:
            raise RuntimeError('Could not generate fresh public geometry: '+family)
    manifest = dict(train=train,validation=validation,test=test,rejections=rejected)
    save(OUT/'bank/manifest.json',manifest)
    protected = {str(ROOT/'code'/name):sha(ROOT/'code'/name) for name in (
        'run_e45b_role_free.py','run_e45f_generalization_benchmark.py','learning/e45g_ppo.py',
        'solvers/e45_team_adapter.py','solvers/e45b_joint_nte.py','e45f_scenarios.py',
        'learning/e45b_joint_network.py','solvers/e45f_baselines.py',
        'solvers/e45f_support_refinement.py','solvers/e45f_geometry_capacity.py',
        'e45g_training_acceleration.py')}
    for seed in SEEDS:
        path = CROOT/f'learning/x20/round_12/models/single_{seed}.npz'
        protected[str(path)] = sha(path)
    upstream = {str(p):sha(p) for p in (UPSTREAM/'onpolicy').rglob('*.py')}
    upstream[str(UPSTREAM/'LICENSE')] = sha(UPSTREAM/'LICENSE')
    bank_hashes = {e['path']:sha(e['path']) for e in train+validation+test}
    bank_hashes[str(OUT/'bank/manifest.json')] = sha(OUT/'bank/manifest.json')
    config = dict(experiment='E45-I',scope='bounded baseline pilot, not convergence/SOTA confirmation',
        smoke=smoke,seeds=list(SEEDS),arms=list(ARMS),training_steps_per_run=steps,
        training_runs=6,training_maps=len(train),validation_maps=len(validation),
        test_groups=len(test),formal_tasks=len(test)*4*17,worlds=['AB','BA'],opponents=list(base.OPPONENTS),
        batch_episodes=24,validation_at=[0,25_000,50_000,100_000],
        opponent_curriculum='first third fixed_D; remaining balanced fixed_D/rootpuct3000',
        gamma=1.,gae_lambda=.95,potential_eta=.05,epochs=5,minibatches=2,chunk_length=10,
        learning_rate=3e-4,entropy_start=.01,entropy_final=.001,critic_privileged=True,
        deployment='deterministic mean, public-only actor, same legal physical action mapping',
        APF='public belief assignment + attraction/obstacle/interagent/tangential field; own adaptation',
        MAPPO_APF='same APF executor, four bounded gain multipliers; no AG fallback',
        upstream_url='https://github.com/marlbenchmark/on-policy',
        upstream_commit=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip(),
        dependencies=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,absl_py='2.5.1'),
        open_repository_review=dict(url='https://github.com/thu-uav/Multi-UAV-pursuit-evasion',
            commit='c1041e822e3e0fafbaad687de4e5093db0742e0b',
            scope='README/env/source inspected; no reusable APF/DACOOP Python module found at this commit; no UAV installation'),
        source_hashes={str(ROOT/'code'/p):sha(ROOT/'code'/p) for p in SOURCES},
        protected_hashes=protected,bank_hashes=bank_hashes,upstream_hashes=upstream,
        independent_units='public history groups; worlds and network seeds are within-group repeats',
        inference='pilot descriptive estimates; no claim of converged strong MARL')
    save(OUT/'experiment.json',config)
    for name in SOURCES:
        dest = OUT/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((ROOT/'code'/name).read_bytes())
    (OUT/'PROTOCOL.md').write_bytes((ROOT/'code/E45I_BASELINES_PROTOCOL_20261031.md').read_bytes())
    emit(OUT,'registered',training_runs=6,steps_per_run=steps,test_groups=len(test),formal_tasks=config['formal_tasks'])
    return config


def request(entry, world, opponent, arm, seed=0, stage='test', checkpoint=None, episode_seed=None, gains=None):
    environment_seed = int(np.random.SeedSequence([BASE_SEED,int(sha(entry['path'])[:8],16),int(world=='BA')]).generate_state(1)[0])
    if episode_seed is not None:
        environment_seed = int(episode_seed)
    checkpoint = str(checkpoint) if checkpoint else None
    digest = sha(checkpoint) if checkpoint else None
    gains = coefficients() if gains is None else np.asarray(gains).tolist()
    identity = hashlib.sha256(json.dumps(dict(checkpoint=digest,coefficients=gains),sort_keys=True).encode()).hexdigest()[:16]
    method = arm+str(seed) if arm in RL+('SINGLE_PRIOR','SINGLE_DIRECT') else arm
    return dict(key=f'{entry["id"]}_{world}_{opponent}_{method}_{stage}_{environment_seed}_{identity}',
        case=entry['path'],group=entry['id'],family=entry['family'],suite=entry.get('suite',stage),
        split=entry['split'],world=world,opponent=opponent,arm=arm,method=method,train_seed=seed,
        seed=environment_seed,stage=stage,round=0,model_round=12,budget_ms=200.,
        checkpoint=checkpoint,checkpoint_sha256=digest,apf_coefficients=gains)


def worker_init(output, queue, parent):
    import ctypes,signal
    ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    if os.getppid()!=parent:
        raise RuntimeError('Coordinator exited')
    os.sched_setaffinity(0,{queue.get()});torch.set_num_threads(1)
    configure(output);base.PARENT=parent
    entry=read(OUT/'bank/manifest.json')['validation'][0]
    # JIT and shared safety warmup are outside measured benchmark returns.
    base.episode(request(entry,'AB','fixed_D_kernel','APF',stage='warmup')|dict(limit=2))
    base.episode(request(entry,'AB','fixed_D_kernel','U',stage='warmup')|dict(limit=1,fixed=2))
    coverage_potential(np.array([[0.,5.],[0.,6.]]),np.ones(2,dtype=bool),np.ones(2,dtype=bool),
        np.zeros((2,64,2)),np.full((2,64),1/64),1.1,np.zeros((0,2)),np.zeros((0,0)),np.zeros((0,2,2)))


def pool_for(cpus):
    context=mp.get_context('spawn');queue=context.Queue()
    for cpu in cpus:queue.put(cpu)
    pool=futures.ProcessPoolExecutor(max_workers=len(cpus),mp_context=context,
        initializer=worker_init,initargs=(str(OUT),queue,os.getpid()))
    return pool,queue


def evaluate_job(task):
    # Any exception stops this stage; no invalid episode is silently scored.
    row=base.episode_job(task)
    if row.get('checkpoint_sha256')!=task.get('checkpoint_sha256'):
        raise RuntimeError('Cached checkpoint identity mismatch')
    return row


def evaluate(tasks, cpus, label):
    pool,queue=pool_for(cpus);rows=[]
    try:
        for row in pool.map(evaluate_job,tasks):
            rows.append(row)
            if len(rows)%16==0 or len(rows)==len(tasks):
                emit(OUT,label,completed=len(rows),planned=len(tasks))
    finally:
        pool.shutdown(wait=True,cancel_futures=True);queue.close()
    return rows


def rollout_job(job):
    global ROLLOUT_ACTOR
    model,_=policy(job['kind']);model.actor.load_state_dict(job['actor']);model.actor.eval()
    ROLLOUT_ACTOR=model.actor;torch.manual_seed(job['request']['seed'])
    with public_geometry_cache():
        payload,_=base.episode(job['request'])
    own=CAPTURE;samples=own.samples
    episode={key:np.stack([s[key] for s in samples]) for key in ('obs','actor_hidden','raw','logp','active')}
    trace=payload['trace'];episode['privileged']=np.asarray([r['state'][:4] for r in trace],np.float32)/10
    # Public potential is computed during action from the exact belief sample,
    # never reconstructed from hidden positions or final belief state.
    phi=np.array([s['phi'] for s in samples])
    reward=np.array([r['reward'] for r in trace],np.float32)
    episode['reward']=(reward+.05*(np.r_[phi[1:],0.]-phi)).astype(np.float32)
    episode['summary']=payload['summary']|dict(raw_utility=float(reward.sum()))
    return episode


def checkpoint(folder, model, trainer, kind, seed, steps, update, rng, best, validations, latest=False):
    folder=Path(folder)
    path=folder/'latest.pt' if latest else folder/'checkpoints'/f'steps_{steps}.pt'
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(actor={k:v.cpu() for k,v in model.actor.state_dict().items()},
        critic={k:v.cpu() for k,v in model.critic.state_dict().items()},
        actor_optimizer=model.actor_optimizer.state_dict(),critic_optimizer=model.critic_optimizer.state_dict(),
        kind=kind,seed=seed,steps=steps,update=update,numpy_rng=rng.bit_generator.state,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(model.device) if model.device.type=='cuda' else None,
        best=best,validations=validations,experiment_sha256=sha(OUT/'experiment.json'))
    temp=path.with_suffix('.tmp.pt');torch.save(payload,temp);temp.replace(path)
    return path


def validate(pool,kind,seed,path,steps):
    entries=read(OUT/'bank/manifest.json')['validation']
    tasks=[request(e,w,o,kind,seed,'validation',path) for e in entries for w in ('AB','BA') for o in base.OPPONENTS]
    rows=list(pool.map(evaluate_job,tasks))
    return dict(steps=steps,checkpoint=str(path),checkpoint_sha256=sha(path),
        utility=float(np.mean([r['utility'] for r in rows])),double_capture=float(np.mean([r['double_capture'] for r in rows])),
        episodes=len(rows),public_replay_exact=all(r['public_replay_exact'] for r in rows))


def train(args):
    config=verify();folder=OUT/'learning'/f'{args.algorithm}_{args.seed}';folder.mkdir(parents=True,exist_ok=True)
    cpus=[int(c) for c in args.cpus.split(',')];os.sched_setaffinity(0,{cpus[-1]});torch.set_num_threads(1)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(torch.device(args.device))
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    model,trainer=policy(args.algorithm,args.device);rng=np.random.default_rng(args.seed+BASE_SEED)
    steps=0;update=0;best=-float('inf');validations=[]
    resume=folder/'resume.json'
    if resume.exists():
        saved=torch.load(read(resume)['checkpoint'],map_location=model.device,weights_only=False)
        if saved['experiment_sha256']!=sha(OUT/'experiment.json'):raise RuntimeError('Resume protocol mismatch')
        model.actor.load_state_dict(saved['actor']);model.critic.load_state_dict(saved['critic'])
        model.actor_optimizer.load_state_dict(saved['actor_optimizer']);model.critic_optimizer.load_state_dict(saved['critic_optimizer'])
        steps=saved['steps'];update=saved['update'];best=saved['best'];validations=saved['validations']
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'].cpu())
        if saved['cuda_rng'] is not None:torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),model.device)
    pool,queue=pool_for(cpus[:-1]);entries=read(OUT/'bank/manifest.json')['train'];began=time.time()
    try:
        if not validations:
            path=checkpoint(folder,model,trainer,args.algorithm,args.seed,steps,update,rng,best,validations)
            row=validate(pool,args.algorithm,args.seed,path,steps);validations.append(row);best=row['utility']
            save(folder/'selected.json',row)
            latest=checkpoint(folder,model,trainer,args.algorithm,args.seed,steps,update,rng,best,validations,latest=True)
            save(folder/'resume.json',dict(checkpoint=str(latest),steps=steps,update=update))
            save(folder/'validation.json',validations)
        while steps<config['training_steps_per_run']:
            stamp=time.time();trainer.prep_rollout()
            weights={k:v.detach().cpu() for k,v in model.actor.state_dict().items()}
            jobs=[]
            for _ in range(config['batch_episodes']):
                e=entries[int(rng.integers(len(entries)))];world='AB' if rng.random()<.5 else 'BA'
                opponent='fixed_D_kernel' if steps/config['training_steps_per_run']<1/3 or rng.random()<.5 else 'rootpuct'
                req=request(e,world,opponent,args.algorithm,args.seed,'training',episode_seed=rng.integers(2**31))
                jobs.append(dict(kind=args.algorithm,actor=weights,request=req))
            episodes=list(pool.map(rollout_job,jobs));rollout_seconds=time.time()-stamp
            buffer=finish_episodes(episodes,model)
            fraction=min(1.,steps/config['training_steps_per_run']);trainer.entropy_coef=.01*(1-fraction)+.001*fraction
            trainer.prep_training();info=trainer.train(buffer)
            info={key:float(value.detach().cpu()) if torch.is_tensor(value) else float(value) for key,value in info.items()}
            if not all(np.isfinite(v) for v in info.values()):raise RuntimeError('Non-finite author PPO update')
            steps+=sum(len(e['obs']) for e in episodes);update+=1
            record=dict(steps=steps,update=update,device=args.device,episodes=len(episodes),
                training_raw_utility=float(np.mean([e['summary']['utility'] for e in episodes])),
                training_double_capture=float(np.mean([e['summary']['double_capture'] for e in episodes])),
                by_opponent={o:dict(episodes=sum(e['summary']['opponent']==o for e in episodes),
                    double_capture=float(np.mean([e['summary']['double_capture'] for e in episodes if e['summary']['opponent']==o])))
                    for o in base.OPPONENTS if any(e['summary']['opponent']==o for e in episodes)},
                rollout_seconds=rollout_seconds,total_seconds=time.time()-stamp,**info)
            save(folder/'updates'/f'{update:05d}.json',record)
            previous=validations[-1]['steps']
            due=any(previous<p<=steps for p in config['validation_at']) or steps>=config['training_steps_per_run']
            path=checkpoint(folder,model,trainer,args.algorithm,args.seed,steps,update,rng,best,validations)
            if due:
                row=validate(pool,args.algorithm,args.seed,path,steps);validations.append(row)
                if row['utility']>best:
                    best=row['utility'];save(folder/'selected.json',row)
                save(folder/'validation.json',validations)
                # Resume includes validation/selection results, not a stale pre-validation copy.
                emit(folder,'validation',**row)
            latest=checkpoint(folder,model,trainer,args.algorithm,args.seed,steps,update,rng,best,validations,latest=True)
            save(folder/'resume.json',dict(checkpoint=str(latest),steps=steps,update=update))
            emit(folder,'training',**record)
        save(folder/'decision.json',dict(status='complete',steps=steps,updates=update,
            selected=read(folder/'selected.json'),convergence_claim=False,wall_seconds=time.time()-began))
        emit(folder,'complete',steps=steps,updates=update)
    finally:
        pool.shutdown(wait=True,cancel_futures=True);queue.close()


def static_tasks(entries,stage):
    return [request(e,w,o,arm,seed,stage,
        CROOT/f'learning/x20/round_12/models/single_{seed}.npz' if arm in ('SINGLE_PRIOR','SINGLE_DIRECT') else None)
        for e in entries for w in ('AB','BA') for o in base.OPPONENTS
        for arm in STATIC for seed in (SEEDS if arm in ('SINGLE_PRIOR','SINGLE_DIRECT') else (0,))]


def summarize(rows, label):
    import csv
    grouped=collections.defaultdict(list)
    for row in rows:grouped[row['suite'],row['opponent'],row['arm']].append(row)
    records=[]
    for (suite,opponent,arm),rr in sorted(grouped.items()):
        lat=[v for r in rr for v in r['latencies_ms']]
        records.append(dict(suite=suite,opponent=opponent,arm=arm,episodes=len(rr),
            independent_groups=len({r['group'] for r in rr}),double_capture=sum(r['double_capture'] for r in rr),
            double_capture_rate=float(np.mean([r['double_capture'] for r in rr])),
            completion_cost=float(np.mean([r['completion_cost'] for r in rr])),
            utility=float(np.mean([r['utility'] for r in rr])),p95_ms=float(np.percentile(lat,95)),
            overruns=sum(v>200 for v in lat),decisions=len(lat),public_replay_exact=all(r['public_replay_exact'] for r in rr)))
    path=OUT/(label+'.csv')
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    save(OUT/(label+'.json'),dict(rows=records,total_episodes=len(rows),
        all_public_replay_exact=all(r['public_replay_exact'] for r in rows),
        independent_groups=len({r['group'] for r in rows}),scope='pilot descriptive benchmark; limited training budget'))
    lines=['# E45-I '+label,'','Pilot results; not converged strong-MARL confirmation.','',
           '|suite|opponent|arm|capture|rate|cost|P95 ms|','|---|---|---|---:|---:|---:|---:|']
    for r in records:
        lines.append(f'|{r["suite"]}|{r["opponent"]}|{r["arm"]}|{r["double_capture"]}/{r["episodes"]}|{r["double_capture_rate"]:.2%}|{r["completion_cost"]:.2f}|{r["p95_ms"]:.2f}|')
    (OUT/(label+'.md')).write_text('\n'.join(lines)+'\n')
    return records


def tune_apf(cpus):
    path=OUT/'apf_selected.json'
    if path.exists():
        if not (OUT/'apf_freeze.json').exists():
            save(OUT/'apf_freeze.json',dict(sha256=sha(path),selection='validation only'))
        return read(path)
    entries=read(OUT/'bank/manifest.json')['validation']
    candidates=[DEFAULT_APF.tolist(),[1.,.3,.15,.3],[1.,1.5,.2,1.2]]
    scores=[]
    for i,gains in enumerate(candidates):
        tasks=[request(e,w,o,'APF',stage='apf_validation',gains=gains)
            for e in entries for w in ('AB','BA') for o in base.OPPONENTS]
        rows=evaluate(tasks,cpus,f'apf_tune_{i}')
        scores.append(dict(coefficients=gains,utility=float(np.mean([r['utility'] for r in rows])),
            double_capture=float(np.mean([r['double_capture'] for r in rows])),episodes=len(rows)))
    selected=max(scores,key=lambda r:r['utility'])
    save(path,selected|dict(candidates=scores,selection='validation raw utility, earliest tie; test unopened'))
    save(OUT/'apf_freeze.json',dict(sha256=sha(path),selection='validation only'))
    return selected


def pipeline(args):
    config=prepare(args.steps);cpus=choose_cpus(50)
    if len(cpus)<50:raise RuntimeError('Pilot schedule requires 50 distinct physical CPUs')
    configure(OUT)
    tune_apf(cpus[:8]);verify()
    static=evaluate(static_tasks(read(OUT/'bank/manifest.json')['test'],'static_test'),cpus[:8],'static_test')
    summarize(static,'static_benchmark');emit(OUT,'static_complete',episodes=len(static))
    # Two trainers per GPU, each with six sampler cores and one parent core.
    children=[];schedule=[]
    for index,(kind,seed) in enumerate((k,s) for k in RL for s in SEEDS):
        run_cpus=cpus[8+7*index:8+7*(index+1)]
        device=('cuda:2','cuda:3')[index%2]
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'train','--output',str(OUT),
            '--algorithm',kind,'--seed',str(seed),'--device',device,'--cpus',','.join(map(str,run_cpus))]
        folder=OUT/'learning'/f'{kind}_{seed}';folder.mkdir(parents=True,exist_ok=True)
        stream=(folder/'run.log').open('a')
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT);stream.close()
        children.append(child);schedule.append(dict(kind=kind,seed=seed,pid=child.pid,device=device,cpus=run_cpus))
    save(OUT/'execution.json',dict(training=schedule,cpus=cpus,workers=36,parent_cores=6,
        gpu_note='uses GPUs 2/3; no existing jobs terminated',static_workers=8))
    emit(OUT,'training_started',runs=schedule,steps_per_run=config['training_steps_per_run'])
    while any(c.poll() is None for c in children):
        failed=[c for c in children if c.poll() not in (None,0)]
        if failed:
            # Stop this experiment's siblings only, retaining checkpoints.
            for child in children:
                if child.poll() is None:child.terminate()
            raise RuntimeError('Baseline trainer failed; inspect per-run logs')
        time.sleep(10)
    if any(c.returncode for c in children):raise RuntimeError('Trainer exited unsuccessfully')
    verify();tasks=[]
    for e in read(OUT/'bank/manifest.json')['test']:
        for w in ('AB','BA'):
            for o in base.OPPONENTS:
                for kind in RL:
                    for seed in SEEDS:
                        selected=read(OUT/'learning'/f'{kind}_{seed}'/'selected.json')
                        if sha(selected['checkpoint'])!=selected['checkpoint_sha256']:
                            raise RuntimeError('Selected model changed')
                        tasks.append(request(e,w,o,kind,seed,'test',selected['checkpoint']))
    rows=evaluate(tasks,cpus[:8],'learned_test')
    summarize(static+rows,'nine_family_benchmark');verify()
    save(OUT/'decision.json',dict(status='complete',episodes=len(static)+len(rows),
        expected=config['formal_tasks'],families=len(ARMS),all_public_replay_exact=True,
        convergence_claim=False,scope=config['scope']))
    emit(OUT,'complete',episodes=len(static)+len(rows),families=len(ARMS))


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','smoke','pipeline','train','static'])
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--steps',type=int,default=100_000)
    parser.add_argument('--algorithm',choices=RL,default='RMAPPO');parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--device',default='cuda:2');parser.add_argument('--cpus',default='')
    args=parser.parse_args();OUT=args.output.resolve();torch.set_num_threads(1)
    if args.command=='prepare':prepare(args.steps)
    elif args.command=='pipeline':pipeline(args)
    elif args.command=='train':configure(OUT);train(args)
    elif args.command=='static':
        verify();configure(OUT);cpus=choose_cpus(8);tune_apf(cpus)
        rows=evaluate(static_tasks(read(OUT/'bank/manifest.json')['test'],'static_test'),cpus,'static_test')
        summarize(rows,'static_benchmark')
    elif args.command=='smoke':
        prepare(args.steps,True);configure(OUT);cpus=choose_cpus(2)
        entry=read(OUT/'bank/manifest.json')['validation'][0]
        model,_=policy('RMAPPO');path=OUT/'random_actor.pt';torch.save(dict(actor=model.actor.state_dict()),path)
        model2,_=policy('MAPPO_APF');path2=OUT/'random_apf_actor.pt';torch.save(dict(actor=model2.actor.state_dict()),path2)
        tasks=[request(entry,'AB',o,arm,17,'engineering',p)|dict(limit=3,fixed=3)
               for o in base.OPPONENTS for arm,p in [('APF',None),('RMAPPO',path),('MAPPO_APF',path2)]]
        rows=evaluate(tasks,cpus,'engineering');save(OUT/'engineering.json',dict(episodes=len(rows),
            public_replay_exact=all(r['public_replay_exact'] for r in rows),performance_evidence=False))
        # Complete real episodes and the actual upstream optimizer on a GPU,
        # including critic replay and complete-episode recurrent minibatches.
        updates=[]
        torch.cuda.set_device(torch.device(args.device))
        for kind in RL:
            model,trainer=policy(kind,args.device)
            before={k:v.detach().cpu().clone() for k,v in model.actor.state_dict().items()}
            episode=rollout_job(dict(kind=kind,actor=before,
                request=request(entry,'AB','fixed_D_kernel',kind,17,'training',episode_seed=17)))
            buffer=finish_episodes([episode],model);info=trainer.train(buffer)
            changed=any(not torch.equal(before[k],v.detach().cpu()) for k,v in model.actor.state_dict().items())
            values={k:float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k,v in info.items()}
            if not changed or not all(np.isfinite(v) for v in values.values()):
                raise RuntimeError('Real-episode author GPU update failed')
            updates.append(dict(kind=kind,steps=len(episode['obs']),actor_changed=changed,
                public_replay_exact=episode['summary']['public_replay_exact'],device=args.device,**values))
        save(OUT/'engineering_updates.json',dict(updates=updates,performance_evidence=False))
        emit(OUT,'engineering_complete',prefix_episodes=len(rows),real_gpu_updates=len(updates))


if __name__=='__main__':main()
