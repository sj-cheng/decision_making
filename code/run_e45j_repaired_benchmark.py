"""E45-J repaired-baseline training with gates and fresh paired benchmarks."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('TZ','Asia/Shanghai')
import argparse
import collections
import copy
import csv
import ctypes
import fcntl
import json
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
import run_e45i_baselines as common
from learning.e45i_official_mappo import policy as old_policy,RecurrentController as OldController,APFController
from learning.e45j_adapter_repairs import RouteAPFController
from learning.e45j_repaired_mappo import policy,Controller,finish_episodes,training_rewards
from learning.e45g_ppo import safe_action
from e45g_training_acceleration import public_geometry_cache
from solvers.e45_team_adapter import public_observation
from corner_estimator_scenarios import original_model
from run_frozen_d_closed_loop import save,sha
from run_e44_parallel import choose_cpus

ROOT=common.ROOT
OUT=ROOT/'current/e45j_repaired_benchmark_20261031'
IROOT=ROOT/'current/e45i_official_baselines_20261031'
BASE_SEED=946220261031
SEEDS=(17,29,43)
MAIN=('SINGLE_PRIOR','RMAPPO','MAPPO_APF','APF','AG','IG','SINGLE_DIRECT','U','GEOM_SEARCH')
AUX=('RMAPPO_I','MAPPO_APF_I','APF_I','INIT_RMAPPO','INIT_MAPPO_APF')
LABELS=dict(RMAPPO='RMAPPO-Clearance',MAPPO_APF='MAPPO+RouteAPF',APF='RouteAPF',
    RMAPPO_I='I: RMAPPO',MAPPO_APF_I='I: MAPPO+APF',APF_I='I: fixed APF',
    INIT_RMAPPO='Initial RMAPPO-Clearance',INIT_MAPPO_APF='Initial MAPPO+RouteAPF')
RL=('RMAPPO','MAPPO_APF')
LEARNED=RL+('SINGLE_PRIOR','SINGLE_DIRECT','RMAPPO_I','MAPPO_APF_I','INIT_RMAPPO','INIT_MAPPO_APF')
PHASES=(dict(name='single_static_visible',minimum=10_000,maximum=40_000,gate=.75),
    dict(name='two_static_visible',minimum=10_000,maximum=60_000,gate=.50),
    dict(name='two_moving_visible',minimum=20_000,maximum=80_000,gate=.25),
    dict(name='hidden_fixed',minimum=80_000,maximum=80_000,gate=None),
    dict(name='hidden_mixed',minimum=0,maximum=500_000,gate=None))
SOURCES=('run_e45j_repaired_benchmark.py','learning/e45j_repaired_mappo.py',
    'learning/e45j_adapter_repairs.py','tests/test_e45j_repaired_benchmark.py',
    'tests/test_e45j_adapter_repairs.py','E45J_REPAIRED_BENCHMARK_PROTOCOL_20261031.md')
_BASE_CONFIGURE=common.configure
_KERNEL=common.base.FixedKernelTargets
ACTORS={};ROLLOUT_ACTOR=None;CAPTURE=None


def read(path):return json.loads(Path(path).read_text())
def emit(stage,**kwargs):common.emit(OUT,stage,**kwargs)


class StaticTargets:
    def __init__(self,case,seed):
        # Environment-only inactive coordinates; active targets are publicly
        # measured at this curriculum start, never exposed through the actor.
        self.at=np.asarray(case['worlds'][0]['initial'][:4],float).reshape(2,2)
    def positions(self):return self.at.copy()
    def actions(self,active):return np.zeros((2,2))


class VisibleKernelTargets(_KERNEL):
    """Same frozen D transition, initialized at the public visible anchor.

    The original hidden-start constructor needs an already hidden posterior.
    Only the visible-start curriculum uses this uniform nine-mode prior.
    """
    def __init__(self,case,seed):
        import l45_corner_shared_20261025.corner_estimator as frozen
        from solvers.frozen_d_search_interface import PublicMap,uniform
        boxes=np.asarray(case['map'],float);public=PublicMap(boxes,1.)
        self.rng=[np.array([(seed+94501+104729*j)%4294967296],np.int64) for j in range(2)]
        self.graphs=[];self.ids=[];self.modes=[]
        for j in range(2):
            measured=[v for v in case['history'][0]['reports'][j] if v is not None]
            anchor=np.asarray(measured[0],float)
            graph=list(frozen.make_graph(anchor,boxes));points,links,lengths=graph[:3]
            graph[3]=frozen.map_probabilities(points,links,lengths,anchor,boxes,public.vertices,public.dist)[0]
            self.graphs.append(tuple(graph));self.ids.append(graph[5]);self.modes.append(min(8,int(9*uniform(self.rng[j]))))


def kernel(case,seed):
    if case.get('stationary_curriculum',False):return StaticTargets(case,seed)
    if case.get('curriculum_variant')=='moving':return VisibleKernelTargets(case,seed)
    return _KERNEL(case,seed)


def configure(output):
    global OUT
    OUT=Path(output).resolve();common.OUT=OUT;common.BASE_SEED=BASE_SEED
    _BASE_CONFIGURE(OUT)
    common.base.make_controller=factory;common.base.FixedKernelTargets=kernel


def gains():
    path=OUT/'apf_selected.json'
    return read(path)['coefficients'] if path.exists() else [1.,1.5,.2,1.2]


def request(entry,world,opponent,arm,seed=0,stage='test',checkpoint=None,episode_seed=None,coefficients=None):
    old_gains=read(IROOT/'apf_selected.json')['coefficients']
    coeff=old_gains if arm in ('APF_I','MAPPO_APF_I','RMAPPO_I') else gains()
    if coefficients is not None:coeff=coefficients
    task=common.request(entry,world,opponent,arm,seed,stage,checkpoint,episode_seed,gains=coeff)
    if arm in LEARNED:
        task['method']=arm+str(seed);task['key']+='_seed_'+str(seed)
    return task


def factory(case,task):
    global CAPTURE
    arm=task['arm'];coeff=task['apf_coefficients']
    if arm=='APF':own=RouteAPFController(case['map'],case['speed'],task['seed'],coeff)
    elif arm=='APF_I':own=APFController(case['map'],case['speed'],task['seed'],coeff)
    elif arm in RL+('INIT_RMAPPO','INIT_MAPPO_APF','RMAPPO_I','MAPPO_APF_I'):
        kind='MAPPO_APF' if 'MAPPO_APF' in arm else 'RMAPPO'
        old=arm.endswith('_I')
        if task['stage']=='training':actor=ROLLOUT_ACTOR
        else:
            key=(arm,task['checkpoint_sha256'])
            if key not in ACTORS:
                model,_=(old_policy if old else policy)(kind)
                saved=torch.load(task['checkpoint'],map_location='cpu',weights_only=False)
                model.actor.load_state_dict(saved['actor']);model.actor.eval();ACTORS[key]=model.actor
            actor=ACTORS[key]
        cls=OldController if old else Controller
        own=cls(case['map'],case['speed'],task['seed'],actor,kind,
            deterministic=task['stage']!='training',coefficients=coeff)
        if task['stage']=='training':CAPTURE=own
    else:own=common.reference.factory(case,task)
    original=own.act
    def act(obs,fixed_simulations=None):
        action=original(obs,fixed_simulations)
        result,changed=safe_action(np.asarray(obs['positions']),action/own.public.speed,
            obs['active_p'],own.public.speed,own.public.boxes)
        own.last_stats['outer_safety_shortened']=changed
        return result
    own.act=act
    return own


def verify():
    cfg=read(OUT/'experiment.json')
    for section in ('source_hashes','protected_hashes','bank_hashes','upstream_hashes'):
        for path,digest in cfg[section].items():
            if sha(path)!=digest:raise RuntimeError('Frozen J artifact changed: '+path)
    if (OUT/'apf_freeze.json').exists() and sha(OUT/'apf_selected.json')!=read(OUT/'apf_freeze.json')['sha256']:
        raise RuntimeError('Frozen RouteAPF coefficients changed')
    return cfg


def curriculum_case(case,variant):
    result=copy.deepcopy(case);result['id']+='_'+variant;result['history']=result['history'][:1]
    p=original_model(result['map'],result['speed'])
    for world in result['worlds']:
        state=np.asarray(world['scripted_states'][0],float).reshape(-1,1).copy()
        if variant.startswith('single'):
            pair=int(variant[-1]);state[p.active_idxs,0]=0
            state[p.active_idxs[pair],0]=1;state[p.active_idxs[2+pair],0]=1
        state[p.base_state_dim:]=0;p._update_observation_memory(state,state,0.)
        assert p.is_valid(state) and not p.get_capture_pairs(state)
        world['initial']=state[:,0].tolist();world['scripted_states']=[world['initial']]
    s=np.asarray(result['worlds'][0]['initial']);ap=s[p.active_idxs[2:]]>.5;ae=s[p.active_idxs[:2]]>.5
    obs=public_observation(0,s[4:8].reshape(2,2),s[:4].reshape(2,2),ap,ae,np.asarray(result['map']))
    for j in np.flatnonzero(ae):assert any(q is not None for q in obs['reports'][j])
    result['history']=[obs];result['stationary_curriculum']=variant!='moving'
    result['curriculum_variant']=variant
    return result


def prepare(smoke=False,steps=500_000):
    if (OUT/'experiment.json').exists():return verify()
    OUT.mkdir(parents=True,exist_ok=True)
    source=read(IROOT/'experiment.json')
    for section in ('source_hashes','protected_hashes','bank_hashes','upstream_hashes'):
        for path,digest in source[section].items():
            if sha(path)!=digest:raise RuntimeError('I source/bank changed before J: '+path)
    manifest=read(common.CROOT/'bank/manifest.json');base_entries=manifest['entries']
    train=[];validation=[]
    for e in base_entries:
        if e['split'] not in ('train','validation'):continue
        item=e.copy();item['curriculum']={}
        if smoke and any(x['family']==e['family'] for x in (train if e['split']=='train' else validation)):continue
        case=read(e['path'])
        for variant in ('single0','single1','static','moving'):
            warm=curriculum_case(case,variant)
            path=OUT/'bank/curriculum'/e['split']/(warm['id']+'.json');save(path,warm)
            item['curriculum'][variant]=str(path)
        (train if e['split']=='train' else validation).append(item)
    known=set()
    for directory in ROOT.glob('current/e45*/bank'):
        if directory==OUT/'bank':continue
        for path in directory.glob('*/*.json'):
            packet=read(path)
            if 'map' in packet:known.add(common.geometry_key(packet['map']))
    test=[];rejected=[]
    for family in common.ID_FAMILIES+common.OOD_FAMILIES:
        builder=common.new_id_case if family in common.ID_FAMILIES else common.shifted_case
        seen=set();accepted=0
        for index in range(1200):
            try:
                case=builder(family,index,'test',BASE_SEED+int(smoke)*1_000_000)
                cell=(case['pattern'],case['speed'])
                if cell in seen and not smoke:continue
                key=common.geometry_key(case['map'])
                if key in known:raise ValueError('Previously used geometry')
            except (ValueError,AssertionError) as exc:
                rejected.append(dict(family=family,index=index,error=repr(exc)));continue
            known.add(key);seen.add(cell);accepted+=1
            path=OUT/'bank/test'/(case['id']+'.json');save(path,case)
            test.append(dict(id=case['id'],path=str(path),family=family,split='test',pattern=case['pattern'],
                density=case['density'],suite='benchmark' if family in common.ID_FAMILIES else 'generalization'))
            if accepted==(1 if smoke else 6):break
        else:raise RuntimeError('Fresh benchmark construction exhausted: '+family)
    save(OUT/'bank/manifest.json',dict(train=train,validation=validation,test=test,rejections=rejected))
    protected={str(IROOT/'experiment.json'):sha(IROOT/'experiment.json')}
    protected.update(source['source_hashes']);protected.update(source['protected_hashes'])
    old_models={}
    for kind in RL:
        old_models[kind]={}
        for seed in SEEDS:
            chosen=read(IROOT/'learning'/f'{kind}_{seed}'/'selected.json')
            assert sha(chosen['checkpoint'])==chosen['checkpoint_sha256']
            protected[chosen['checkpoint']]=chosen['checkpoint_sha256'];old_models[kind][str(seed)]=chosen['checkpoint']
    protected[str(IROOT/'apf_selected.json')]=sha(IROOT/'apf_selected.json')
    bank={str(p):sha(p) for p in (OUT/'bank').rglob('*.json')}
    for e in train+validation:bank[e['path']]=sha(e['path'])
    cfg=dict(experiment='E45-J',scope='repaired task adapters and gated curriculum, bounded budget',smoke=smoke,
        seeds=list(SEEDS),main_arms=list(MAIN),auxiliary_arms=list(AUX),method_labels=LABELS,
        training_steps_per_run=steps,training_runs=6,batch_episodes=24,
        phases=PHASES,gate_check_every=10_000,main_validation_every=50_000,
        worlds=['AB','BA'],opponents=list(common.base.OPPONENTS),test_groups=len(test),
        main_tasks=len(test)*4*17,auxiliary_tasks=len(test)*4*13,formal_tasks=len(test)*4*30,
        online_budget_ms=200,old_models=old_models,epochs=5,minibatches=2,chunk_length=10,
        upstream_commit=source['upstream_commit'],upstream_url=source['upstream_url'],
        learning_rate=3e-4,final_learning_rate=3e-5,entropy_start=.005,entropy_final=.0005,
        blocked_intent_cost=.05,distance_potential=.05,capture_progress_potential=.25,
        reward='raw team utility + terminal-zero public potential difference - .05 active shortened-intent fraction',
        objective_disclosure='blocked-intent feedback is an additional training cost, not claimed policy-invariant shaping',
        information='same public map/team/D stream; canonical particles, public moments/ego offsets/clearance rays; privileged critic only',
        gate_failure='stop this run with explicit failed learning gate; still evaluate its validation-selected policy',
        test_selection='fresh geometry, one per pattern x speed per family, no controller outcome selection',
        independent_units='18 ID /24 OOD histories; worlds and network seeds are within-history repeats',
        source_hashes={str(ROOT/'code'/name):sha(ROOT/'code'/name) for name in SOURCES},
        protected_hashes=protected,bank_hashes=bank,upstream_hashes=source['upstream_hashes'])
    save(OUT/'experiment.json',cfg)
    for name in SOURCES:
        dest=OUT/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'code'/name,dest)
    shutil.copy2(ROOT/'code/E45J_REPAIRED_BENCHMARK_PROTOCOL_20261031.md',OUT/'PROTOCOL.md')
    emit('registered',test_groups=len(test),formal_tasks=cfg['formal_tasks'],steps_per_run=steps)
    return cfg


def worker_init(output,queue,parent):
    ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    if os.getppid()!=parent:raise RuntimeError('Parent exited')
    os.sched_setaffinity(0,{queue.get()});torch.set_num_threads(1);configure(output);common.base.PARENT=parent
    entry=read(OUT/'bank/manifest.json')['validation'][0];case=read(entry['path'])
    own=Controller(case['map'],case['speed'],1,policy('RMAPPO')[0].actor,'RMAPPO')
    for obs in case['history'][:-1]:own.observe(obs)
    own.act(case['history'][-1])
    common.base.episode(request(entry,'AB','fixed_D_kernel','APF',stage='warmup')|dict(limit=2))
    common.base.episode(request(entry,'AB','fixed_D_kernel','U',stage='warmup')|dict(limit=1,fixed=2))


common.worker_init=worker_init


def pool_for(cpus):return common.pool_for(cpus)


def rollout(job):
    global ROLLOUT_ACTOR
    model,_=policy(job['kind']);model.actor.load_state_dict(job['actor']);model.actor.eval();ROLLOUT_ACTOR=model.actor
    torch.manual_seed(job['request']['seed'])
    with public_geometry_cache():payload,_=common.base.episode(job['request'])
    own=CAPTURE;trace=payload['trace'];samples=own.samples
    assert len(trace)==len(samples)
    ep={key:np.stack([r[key] for r in samples]) for key in ('obs','critic_public','actor_hidden','raw','logp','active')}
    ep['privileged']=np.asarray([r['state'][:4] for r in trace],np.float32)/10
    phi=np.array([r['phi'] for r in samples]);blocked=np.array([r['blocked_fraction'] for r in samples])
    raw_reward=np.array([r['reward'] for r in trace])
    ep['reward']=training_rewards(raw_reward,phi,blocked)
    ep['summary']=payload['summary']|dict(blocked_fraction=float(blocked.mean()),
        first_capture=payload['summary']['first_capture'],initial_active_targets=int(sum(trace[0]['observation']['active_e'])))
    return ep


def warm_entry(entry,phase,rng=None,index=0):
    if phase=='single_static_visible':variant='single'+str(int(rng.integers(2)) if rng is not None else index%2)
    else:variant='static' if phase=='two_static_visible' else 'moving'
    path=entry['curriculum'][variant]
    return entry|dict(id=entry['id']+'_'+variant,path=path,suite='curriculum')


def save_checkpoint(folder,model,kind,seed,steps,update,rng,state,latest=False):
    path=folder/'latest.pt' if latest else folder/'checkpoints'/f'steps_{steps}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    obj=dict(actor={k:v.detach().cpu() for k,v in model.actor.state_dict().items()},
        critic={k:v.detach().cpu() for k,v in model.critic.state_dict().items()},
        actor_optimizer=model.actor_optimizer.state_dict(),critic_optimizer=model.critic_optimizer.state_dict(),
        kind=kind,seed=seed,steps=steps,update=update,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(model.device) if model.device.type=='cuda' else None,
        curriculum=state,experiment_sha256=sha(OUT/'experiment.json'))
    temp=path.with_suffix('.tmp.pt');torch.save(obj,temp);temp.replace(path);return path


def validate(pool,kind,seed,path,steps,phase=None):
    entries=read(OUT/'bank/manifest.json')['validation']
    if phase is not None:
        tasks=[request(warm_entry(e,phase,index=i),'AB','fixed_D_kernel',kind,seed,'gate_'+phase,path) for i,e in enumerate(entries)]
    else:
        tasks=[request(e,w,o,kind,seed,'validation',path) for e in entries for w in ('AB','BA') for o in common.base.OPPONENTS]
    rows=list(pool.map(common.evaluate_job,tasks))
    return dict(steps=steps,checkpoint=str(path),checkpoint_sha256=sha(path),phase=phase,
        utility=float(np.mean([r['utility'] for r in rows])),double_capture=float(np.mean([r['double_capture'] for r in rows])),
        episodes=len(rows),public_replay_exact=all(r['public_replay_exact'] for r in rows))


def train(args):
    cfg=verify();folder=OUT/'learning'/f'{args.algorithm}_{args.seed}';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpus=list(map(int,args.cpus.split(',')));os.sched_setaffinity(0,{cpus[-1]});torch.set_num_threads(1)
    if args.device.startswith('cuda'):torch.cuda.set_device(torch.device(args.device))
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    model,trainer=policy(args.algorithm,args.device);rng=np.random.default_rng(BASE_SEED+args.seed)
    steps=0;update=0;state=dict(phase=0,phase_start=0,next_gate=10_000,next_validation=50_000,
        best=None,validations=[],gates=[],gate_failed=False)
    if (folder/'resume.json').exists():
        saved=torch.load(read(folder/'resume.json')['checkpoint'],map_location=model.device,weights_only=False)
        assert saved['experiment_sha256']==sha(OUT/'experiment.json')
        model.actor.load_state_dict(saved['actor']);model.critic.load_state_dict(saved['critic'])
        model.actor_optimizer.load_state_dict(saved['actor_optimizer']);model.critic_optimizer.load_state_dict(saved['critic_optimizer'])
        steps=saved['steps'];update=saved['update'];state=saved['curriculum'];rng.bit_generator.state=saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if saved['cuda_rng'] is not None:torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),model.device)
    entries=read(OUT/'bank/manifest.json')['train'];pool,queue=pool_for(cpus[:-1]);started=time.time()
    try:
        if not state['validations']:
            path=save_checkpoint(folder,model,args.algorithm,args.seed,steps,update,rng,state)
            row=validate(pool,args.algorithm,args.seed,path,steps)
            state['validations'].append(row);state['best']=row['utility'];save(folder/'selected.json',row)
            save(folder/'initial.json',row)
        while steps<cfg['training_steps_per_run'] and not state['gate_failed']:
            begin=time.time();phase=PHASES[state['phase']];trainer.prep_rollout()
            weights={k:v.detach().cpu() for k,v in model.actor.state_dict().items()};jobs=[]
            for _ in range(cfg['batch_episodes']):
                e=entries[int(rng.integers(len(entries)))];world='AB' if rng.random()<.5 else 'BA'
                if state['phase']<3:e=warm_entry(e,phase['name'],rng)
                opponent='rootpuct' if state['phase']==4 and rng.random()<.5 else 'fixed_D_kernel'
                req=request(e,world,opponent,args.algorithm,args.seed,'training',episode_seed=rng.integers(2**31))
                jobs.append(dict(kind=args.algorithm,actor=weights,request=req))
            episodes=list(pool.map(rollout,jobs));rollout_seconds=time.time()-begin
            data=finish_episodes(episodes,model);fraction=min(1.,steps/cfg['training_steps_per_run'])
            lr=3e-4*(1-fraction)+3e-5*fraction
            for optimizer in (model.actor_optimizer,model.critic_optimizer):
                for group in optimizer.param_groups:group['lr']=lr
            trainer.entropy_coef=.005*(1-fraction)+.0005*fraction
            trainer.prep_training();info=trainer.train(data)
            info={k:float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k,v in info.items()}
            if not all(np.isfinite(v) for v in info.values()):raise RuntimeError('Nonfinite repaired PPO update')
            steps+=sum(len(e['obs']) for e in episodes);update+=1
            metadata=[{k:e['summary'][k] for k in ('key','case','world','opponent','seed','decisions','utility','double_capture','captures','first_capture','blocked_fraction','initial_active_targets','public_replay_exact')} for e in episodes]
            record=dict(update=update,steps=steps,phase=phase['name'],phase_steps=steps-state['phase_start'],
                training_capture=float(np.mean([e['summary']['double_capture'] for e in episodes])),
                raw_utility=float(np.mean([e['summary']['utility'] for e in episodes])),
                blocked_fraction=float(np.mean([e['summary']['blocked_fraction'] for e in episodes])),
                rollout_seconds=rollout_seconds,total_seconds=time.time()-begin,learning_rate=lr,episodes=metadata,**info)
            save(folder/'updates'/f'{update:05d}.json',record)
            gate_due=state['phase']<3 and record['phase_steps']>=state['next_gate']
            transition=state['phase']==3 and record['phase_steps']>=phase['maximum']
            main_due=steps>=state['next_validation'] or transition or gate_due or steps>=cfg['training_steps_per_run']
            if main_due:
                path=save_checkpoint(folder,model,args.algorithm,args.seed,steps,update,rng,state)
                if gate_due:
                    gate=validate(pool,args.algorithm,args.seed,path,steps,phase['name'])
                    passed=record['phase_steps']>=phase['minimum'] and gate['double_capture']>=phase['gate']
                    gate.update(passed=passed,threshold=phase['gate']);state['gates'].append(gate)
                    if passed:transition=True
                    elif record['phase_steps']>=phase['maximum']:state['gate_failed']=True
                    state['next_gate']+=10_000;save(folder/'gates.json',state['gates'])
                    common.emit(folder,'gate',**gate)
                row=validate(pool,args.algorithm,args.seed,path,steps)
                state['validations'].append(row)
                if row['utility']>state['best']:state['best']=row['utility'];save(folder/'selected.json',row)
                state['next_validation']=max(state['next_validation'],(steps//50_000+1)*50_000)
                save(folder/'validation.json',state['validations'])
                common.emit(folder,'validation',**row)
            if transition:
                state['phase']+=1;state['phase_start']=steps;state['next_gate']=10_000
                common.emit(folder,'curriculum_advanced',steps=steps,phase=PHASES[state['phase']]['name'])
            latest=save_checkpoint(folder,model,args.algorithm,args.seed,steps,update,rng,state,True)
            save(folder/'resume.json',dict(checkpoint=str(latest),steps=steps))
            common.emit(folder,'training',**{k:v for k,v in record.items() if k!='episodes'})
        decision=dict(status='complete',steps=steps,updates=update,learning_gate_failed=state['gate_failed'],
            budget_completed=steps>=cfg['training_steps_per_run'],selected=read(folder/'selected.json'),
            convergence_claim=False,wall_seconds=time.time()-started)
        save(folder/'decision.json',decision);common.emit(folder,'complete',**{k:v for k,v in decision.items() if k!='status'})
    finally:pool.shutdown(wait=True,cancel_futures=True);queue.close();lock.close()


def summarize(rows,label):
    records=common.summarize(rows,label)
    for r in records:r['label']=LABELS.get(r['arm'],r['arm'])
    save(OUT/(label+'.json'),dict(rows=records,total_episodes=len(rows),
        all_public_replay_exact=all(r['public_replay_exact'] for r in rows),
        independent_groups=len({r['group'] for r in rows}),
        scope='E45-J repaired adapters; bounded gated curriculum; not a convergence claim'))
    with (OUT/(label+'.csv')).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['# E45-J '+label,'','|suite|opponent|method|captures|rate|cost|P95 ms|','|---|---|---|---:|---:|---:|---:|']
    for r in records:lines.append(f'|{r["suite"]}|{r["opponent"]}|{r["label"]}|{r["double_capture"]}/{r["episodes"]}|{r["double_capture_rate"]:.2%}|{r["completion_cost"]:.2f}|{r["p95_ms"]:.2f}|')
    (OUT/(label+'.md')).write_text('\n'.join(lines)+'\n')
    return records


def frozen_checkpoint(arm,seed):
    if arm in ('SINGLE_PRIOR','SINGLE_DIRECT'):return common.CROOT/f'learning/x20/round_12/models/single_{seed}.npz'
    if arm in ('RMAPPO_I','MAPPO_APF_I'):
        return Path(read(OUT/'experiment.json')['old_models']['RMAPPO' if arm=='RMAPPO_I' else 'MAPPO_APF'][str(seed)])
    if arm in ('INIT_RMAPPO','INIT_MAPPO_APF'):
        return Path(read(OUT/'learning'/f'{"RMAPPO" if arm=="INIT_RMAPPO" else "MAPPO_APF"}_{seed}'/'initial.json')['checkpoint'])
    if arm in RL:return Path(read(OUT/'learning'/f'{arm}_{seed}'/'selected.json')['checkpoint'])
    return None


def tasks_for(arms,stage='test'):
    result=[]
    for e in read(OUT/'bank/manifest.json')['test']:
        for w in ('AB','BA'):
            for o in common.base.OPPONENTS:
                for arm in arms:
                    for seed in (SEEDS if arm in LEARNED else (0,)):
                        result.append(request(e,w,o,arm,seed,stage,frozen_checkpoint(arm,seed)))
    return result


def tune(cpus):
    if (OUT/'apf_selected.json').exists():return
    entries=read(OUT/'bank/manifest.json')['validation'];scores=[]
    for coeff in ([1.,.8,.2,.6],[1.,1.5,.2,1.2]):
        tasks=[request(e,w,o,'APF',stage='route_tuning',coefficients=coeff) for e in entries for w in ('AB','BA') for o in common.base.OPPONENTS]
        rows=common.evaluate(tasks,cpus,'route_tuning')
        scores.append(dict(coefficients=coeff,utility=float(np.mean([r['utility'] for r in rows])),double_capture=float(np.mean([r['double_capture'] for r in rows]))))
    selected=max(scores,key=lambda r:r['utility']);save(OUT/'apf_selected.json',selected|dict(candidates=scores,selection='validation utility only; earliest tie'))
    save(OUT/'apf_freeze.json',dict(sha256=sha(OUT/'apf_selected.json')))


def benchmark_statistics(rows):
    contrasts=[('SINGLE_PRIOR',a) for a in MAIN if a!='SINGLE_PRIOR']
    repairs=[('RMAPPO','RMAPPO_I'),('MAPPO_APF','MAPPO_APF_I'),('APF','APF_I'),
        ('RMAPPO','INIT_RMAPPO'),('MAPPO_APF','INIT_MAPPO_APF'),('MAPPO_APF','APF')]
    result={};rng=np.random.default_rng(BASE_SEED)
    for suite in ('benchmark','generalization'):
        rr=[r for r in rows if r['suite']==suite];groups=sorted({r['group'] for r in rr});families={r['group']:r['family'] for r in rr}
        for name,pairs in [('main',contrasts),('repairs',repairs)]:
            cells=[(a,b,o) for a,b in pairs for o in common.base.OPPONENTS]
            def value(group,arm,opp):
                items=[r['utility'] for r in rr if r['group']==group and r['arm']==arm and r['opponent']==opp]
                if not items:raise RuntimeError('Missing paired contrast cell')
                return float(np.mean(items))
            data=np.array([[value(g,a,o)-value(g,b,o) for a,b,o in cells] for g in groups]);mean=data.mean(0)
            samples=np.zeros((10_000,len(cells)))
            for family in sorted(set(families.values())):
                ids=np.array([i for i,g in enumerate(groups) if families[g]==family])
                draws=rng.choice(ids,size=(10_000,len(ids)),replace=True)
                samples+=data[draws].sum(1)/len(groups)
            scale=samples.std(0);valid=scale>1e-12
            critical=float(np.quantile(np.max(np.abs((samples[:,valid]-mean[valid])/scale[valid]),axis=1),.95)) if valid.any() else 0.
            result[suite+'_'+name]=dict(groups=len(groups),comparisons=[dict(first=a,second=b,opponent=o,
                utility_difference=float(mean[i]),simultaneous95=[float(mean[i]-critical*scale[i]),float(mean[i]+critical*scale[i])] if valid[i] else None,
                degenerate_bootstrap=not bool(valid[i])) for i,(a,b,o) in enumerate(cells)])
    save(OUT/'paired_statistics.json',result)


def pipeline(args):
    cfg=prepare(False,args.steps);configure(OUT);lock=(OUT/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpus=choose_cpus(54)
    if len(cpus)<54:raise RuntimeError('Need54 distinct physical cores for registered schedule')
    tune(cpus[:12]);verify()
    # Immediate repaired fixed-controller comparison, before any new RL test.
    fixed=common.evaluate(tasks_for(('APF','APF_I')),cpus[:12],'fixed_repair_test')
    summarize(fixed,'fixed_repair_benchmark');emit('fixed_repair_complete',episodes=len(fixed))
    children=[];schedule=[]
    try:
        for index,(kind,seed) in enumerate((k,s) for k in RL for s in SEEDS):
            folder=OUT/'learning'/f'{kind}_{seed}';folder.mkdir(parents=True,exist_ok=True)
            allocated=cpus[index*7:(index+1)*7];device=('cuda:2','cuda:3')[index%2]
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'train','--output',str(OUT),'--algorithm',kind,
                '--seed',str(seed),'--device',device,'--cpus',','.join(map(str,allocated)),'--parent',str(os.getpid())]
            stream=(folder/'run.log').open('a');child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT);stream.close()
            children.append(child);schedule.append(dict(kind=kind,seed=seed,pid=child.pid,cpus=allocated,device=device))
        save(OUT/'execution.json',dict(training=schedule,sampler_cores=36,parent_cores=6,evaluation_workers=12,
            phases_separate=True,other_jobs_terminated=False))
        emit('training_started',runs=schedule,steps_per_run=cfg['training_steps_per_run'])
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('J trainer failed; inspect per-run logs')
            time.sleep(10)
        if any(c.returncode!=0 for c in children):raise RuntimeError('Unsuccessful trainer exit')
        verify();remaining=tuple(a for a in MAIN+AUX if a not in ('APF','APF_I'))
        learned=common.evaluate(tasks_for(remaining),cpus[:12],'remaining_benchmark');all_rows=fixed+learned
        summarize([r for r in all_rows if r['arm'] in MAIN],'nine_family_benchmark')
        summarize(all_rows,'all_controls_benchmark');benchmark_statistics(all_rows);verify()
        decisions=[read(OUT/'learning'/f'{k}_{s}'/'decision.json') for k in RL for s in SEEDS]
        if len(all_rows)!=cfg['formal_tasks']:raise RuntimeError('Incomplete J benchmark')
        save(OUT/'decision.json',dict(status='complete',episodes=len(all_rows),expected=cfg['formal_tasks'],
            main_families=9,total_families=14,gate_failed_runs=sum(d['learning_gate_failed'] for d in decisions),
            training_steps=sum(d['steps'] for d in decisions),convergence_claim=False))
        emit('complete',episodes=len(all_rows),training_steps=sum(d['steps'] for d in decisions))
    except BaseException:
        for child in children:
            if child.poll() is None:child.terminate()
        raise
    finally:lock.close()


def engineering(args):
    prepare(True,1000);configure(OUT);cpus=choose_cpus(2)
    manifest=read(OUT/'bank/manifest.json');e=manifest['validation'][0];tasks=[]
    for kind in RL:
        model,_=policy(kind);path=OUT/(kind+'_random.pt');torch.save(dict(actor=model.actor.state_dict()),path)
        for opponent in common.base.OPPONENTS:
            tasks.append(request(e,'AB',opponent,kind,17,'engineering',path)|dict(limit=3,fixed=3))
    tasks.extend(request(warm_entry(e,p['name'],index=0),'AB','fixed_D_kernel','APF',stage='engineering')|dict(limit=3) for p in PHASES[:3])
    rows=common.evaluate(tasks,cpus,'engineering')
    torch.cuda.set_device(torch.device(args.device));updates=[]
    for kind in RL:
        model,trainer=policy(kind,args.device);before={k:v.detach().cpu().clone() for k,v in model.actor.state_dict().items()}
        episodes=[]
        for i in range(2):
            req=request(warm_entry(e,'single_static_visible',index=i),'AB','fixed_D_kernel',kind,17,'training',episode_seed=17+i)
            episodes.append(rollout(dict(kind=kind,actor=before,request=req)))
        buffer=finish_episodes(episodes,model)
        trainer.train(buffer);changed=any(not torch.equal(v.detach().cpu(),before[k]) for k,v in model.actor.state_dict().items())
        assert changed;updates.append(dict(kind=kind,steps=sum(len(ep['obs']) for ep in episodes),actor_changed=changed,
            public_replay_exact=all(ep['summary']['public_replay_exact'] for ep in episodes)))
    args.cpus=','.join(map(str,choose_cpus(3)));args.seed=17
    for kind in RL:args.algorithm=kind;train(args)
    save(OUT/'engineering.json',dict(episodes=len(rows),all_public_replay_exact=all(r['public_replay_exact'] for r in rows),
        gpu_updates=updates,full_training_loops=[read(OUT/'learning'/f'{k}_17'/'decision.json') for k in RL],
        performance_evidence=False));emit('engineering_complete',prefix_episodes=len(rows),gpu_updates=2,training_loops=2)


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','engineering','pipeline','train'])
    parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--steps',type=int,default=500_000)
    parser.add_argument('--algorithm',choices=RL,default='RMAPPO');parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--device',default='cuda:2');parser.add_argument('--cpus',default='');parser.add_argument('--parent',type=int)
    args=parser.parse_args();OUT=args.output.resolve();configure(OUT);torch.set_num_threads(1)
    if args.parent:
        ctypes.CDLL(None).prctl(1,signal.SIGTERM)
        if os.getppid()!=args.parent:raise RuntimeError('Parent coordinator exited')
    try:
        if args.command=='prepare':prepare(False,args.steps)
        elif args.command=='engineering':engineering(args)
        elif args.command=='train':train(args)
        else:pipeline(args)
    except BaseException as exc:
        common.emit(OUT/'learning'/f'{args.algorithm}_{args.seed}' if args.command=='train' else OUT,
            'failed',error=repr(exc));raise


if __name__=='__main__':main()
