"""Resumable E46 eight-experiment coordinator; no historical file mutations."""
from __future__ import annotations

import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_k] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ['TZ'] = 'Asia/Shanghai'

import argparse
import collections
import concurrent.futures as futures
import copy
import ctypes
import fcntl
import gzip
import hashlib
import json
import multiprocessing as mp
import pickle
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from learning.e45b_joint_network import Network, KEYS, log_density, augment
from learning.e46_geometry_teacher import (OLD_MODELS, MixedController, PoolController,
    RankNetwork, RankRuntime, make_controller, prepare_public, root_labels, reference_values,
    draw_geometry, draw_random)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'current/e46_geometry_teacher_20261105'
SEEDS = (17,29,43)
OPPONENTS = ('fixed_D_kernel','rootpuct')
BASE = 1460520261105
WORK_OUT = None
DISPATCH_CURSOR = collections.defaultdict(int)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
    os.replace(tmp,p)


def save_npz(p,arrays):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with open(tmp,'wb') as f:np.savez_compressed(f,**arrays)
    os.replace(tmp,p)


def save_pickle(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    with gzip.open(tmp,'wb',compresslevel=3) as f:pickle.dump(x,f,protocol=5)
    os.replace(tmp,p)


def load_pickle(p):
    with gzip.open(p,'rb') as f:return pickle.load(f)


def stamp():return time.strftime('%Y-%m-%d %H:%M:%S %Z')


def event(out,kind,**kw):
    row=dict(time=stamp(),event=kind,**kw)
    with open(Path(out)/'events.jsonl','a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)


def configure(out):
    global WORK_OUT
    WORK_OUT=Path(out)
    import run_e45b_role_free as base
    from learning.e45g_ppo import install_belief
    import learning.e45w_support_recovery as recovery
    import solvers.e45_team_adapter as adapter
    import solvers.e45b_joint_nte as joint
    from e45c_capture_audit import audit_simulated
    install_belief(WORK_OUT/'belief_failures')
    recovery.EVIDENCE=WORK_OUT/'belief_failures'
    adapter.TeamBelief=joint.TeamBelief=base.TeamBelief=recovery.TeamBelief
    # E46 does not use the Gym/PPO environment; avoid importing that optional
    # dependency solely to install the same W-r1 public-history belief class.
    base.OUT=WORK_OUT;base.SMOKE=False;base.make_controller=factory
    base.audit_simulated=audit_simulated


def factory(case,spec):
    own=make_controller(case,spec)
    if spec.get('profile'):
        timings=collections.defaultdict(list)
        act_started=[None];first_completed=[None]
        original_act=own.act
        for name in ('observe','features','potential','prepare_models','parameters','propose','trajectory'):
            fn=getattr(own,name)
            def wrapped(*args,_fn=fn,_name=name,**kwargs):
                started=time.perf_counter()
                try:
                    result=_fn(*args,**kwargs)
                    if _name=='trajectory' and first_completed[0] is None and act_started[0] is not None:
                        first_completed[0]=1000*(time.perf_counter()-act_started[0])
                    return result
                finally:timings[_name].append(1000*(time.perf_counter()-started))
            setattr(own,name,wrapped)
        def act(*args,**kwargs):
            timings.clear();started=time.perf_counter()
            act_started[0]=started;first_completed[0]=None
            result=original_act(*args,**kwargs)
            own.last_stats['component_totals_ms']={k:sum(v) for k,v in timings.items()}
            own.last_stats['component_calls']={k:len(v) for k,v in timings.items()}
            own.last_stats['first_trajectory_ms']=timings.get('trajectory',[None])[0]
            own.last_stats['first_completed_simulation_after_ms']=first_completed[0]
            own.last_stats['component_scope']='nested, not additive; observe/features outside budget guards retained'
            own.last_stats['wrapper_total_ms']=1000*(time.perf_counter()-started)
            return result
        own.act=act
    return own


def worker_init(out,queue):
    ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    os.sched_setaffinity(0,{queue.get()})
    os.environ['CUDA_VISIBLE_DEVICES']=''
    torch.set_num_threads(1)
    configure(out)


def prepare(out,smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    path=out/'experiment.json'
    if path.exists():
        cfg=read(path)
        assert cfg['smoke']==smoke
        verify(out,cfg)
        return cfg
    from e45f_scenarios import ID_FAMILIES,OOD_FAMILIES,new_id_case,shifted_case,geometry_key
    known=set();old_files=0
    for folder in (ROOT/'current').glob('*/bank'):
        if folder.parent==out:continue
        for p in folder.rglob('*.json'):
            try:
                item=read(p)
                if isinstance(item,dict) and 'map' in item:
                    known.add(geometry_key(item['map']));old_files+=1
            except (ValueError,KeyError,TypeError):continue
    write(out/'preparation_progress.json',dict(stage='deduplication',old_maps=old_files,canonical_maps=len(known),time=stamp()))
    quantities=dict(train=(1 if smoke else 80),validation=(1 if smoke else 8),
                    development=(1 if smoke else 8),sealed=(1 if smoke else 12))
    entries=[];attempts=[]
    for split in ('train','validation','development','sealed'):
        families=list(ID_FAMILIES)+(list(OOD_FAMILIES) if split in ('development','sealed') else [])
        for family in families:
            count=quantities[split] if family in ID_FAMILIES else (1 if smoke else (6 if split=='development' else 9))
            accepted=0
            namespace=BASE+dict(train=10,validation=20,development=30,sealed=40)[split]
            for index in range(10000):
                if accepted>=count:break
                try:
                    generator=new_id_case if family in ID_FAMILIES else shifted_case
                    generated_split=split if split in ('train','validation') else 'test'
                    case=generator(family,index,generated_split,namespace)
                    key=geometry_key(case['map'])
                    if key in known:raise ValueError('Duplicate canonical geometry')
                except (AssertionError,ValueError) as exc:
                    attempts.append(dict(split=split,family=family,index=index,accepted=False,reason=str(exc)))
                    continue
                known.add(key)
                case.update(id=f'e46_{split}_{family}_{accepted:03d}',split=split,
                            suite='ID' if family in ID_FAMILIES else 'OOD')
                target=out/'bank'/split/(case['id']+'.json');write(target,case)
                entries.append(dict(id=case['id'],split=split,suite=case['suite'],family=family,
                    path=str(target),sha256=sha(target),geometry_sha256=key,pattern=case['pattern'],speed=case['speed']))
                attempts.append(dict(split=split,family=family,index=index,accepted=True,id=case['id']))
                accepted+=1
                write(out/'preparation_progress.json',dict(stage='map_generation',split=split,family=family,
                    accepted=accepted,total=count,all_maps=len(entries),time=stamp()))
            if accepted!=count:raise RuntimeError(f'Cannot construct {split}/{family}: {accepted}/{count}')
            event(out,'bank_family',split=split,family=family,count=accepted)
    model_files={str(p):sha(p) for p in OLD_MODELS.glob('*.npz')}
    assert all(str(OLD_MODELS/f'{kind}_{seed}.npz') in model_files for kind in ('single','joint') for seed in SEEDS)
    # Runtime dependencies, rather than only newly authored files, are frozen.
    source_files={str(ROOT/'code'/p):sha(ROOT/'code'/p) for p in (
        'run_e46_geometry_teacher.py','learning/e46_geometry_teacher.py','E46_GEOMETRY_TEACHER_PROTOCOL_20261105.md',
        'solvers/e45b_joint_nte.py','solvers/e45f_baselines.py','solvers/e45_team_adapter.py',
        'learning/e45b_joint_network.py','learning/e45w_support_recovery.py','learning/e45g_ppo.py',
        'solvers/e45f_support_refinement.py','solvers/e45f_geometry_capacity.py',
        'run_e45b_role_free.py','run_e45w_planning_evaluation.py','e45f_scenarios.py','e45b_scenarios.py',
        'run_e45_team_pilot.py','e45c_capture_audit.py','e45b_sensor_audit.py','tests/test_e46_geometry_teacher.py')}
    for p in source_files:
        target=out/'source_snapshot'/Path(p).relative_to(ROOT/'code')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(p).read_bytes())
    cfg=dict(experiment='E46',created=stamp(),smoke=smoke,entries=entries,attempts=attempts,
        model_hashes=model_files,source_hashes=source_files,seeds=list(SEEDS),base_seed=BASE,
        collection_simulations=4 if smoke else 64,teacher_simulations=8 if smoke else 256,
        teacher_probe_budgets=[2,4] if smoke else [64,256,1024],
        probe_states=2 if smoke else 1024,timing_states=1 if smoke else 256,
        timing_repeats=1 if smoke else 5,reference_worlds=2 if smoke else 32,
        states_per_train_map=2 if smoke else 100,epochs=2 if smoke else 30,
        correction_epochs=1 if smoke else 10,correction_rounds=1 if smoke else 2,
        correction_maps=1 if smoke else 24,correction_states_per_episode=2 if smoke else 16,
        value_training=False,selection='minimum validation action loss, earliest tie',
        final_selection='two new configurations by equal-map equal-seed development utility at 100/200ms; no sealed feedback',
        train_configurations=dict(L1=['single','OLD_JOINT'],L2=['single','GEOM'],
                                  L3=['joint','OLD_JOINT'],L4=['joint','GEOM'],L5=['rank','POOL_UNIFORM']),
        physical_gpu_preference=[3,2,1],forbidden_gpu=[0],minimum_reserved_logical_cpus=20,
        protocol_sha256=source_files[str(ROOT/'code/E46_GEOMETRY_TEACHER_PROTOCOL_20261105.md')])
    write(out/'experiment.json',cfg)
    write(out/'bank/manifest.json',dict(entries=entries))
    write(out/'preparation_progress.json',dict(stage='complete',maps=len(entries),time=stamp()))
    return cfg


def verify(out,cfg):
    for key in ('source_hashes','model_hashes'):
        for p,h in cfg[key].items():
            if sha(p)!=h:raise RuntimeError(f'Frozen source/model changed: {p}')
    for entry in cfg['entries']:
        if sha(entry['path'])!=entry['sha256']:raise RuntimeError('Frozen map changed: '+entry['path'])


def environment_seed(entry,world,repeat=0):
    return int(np.random.SeedSequence([BASE,int(entry['sha256'][:8],16),world=='BA',repeat]).generate_state(1)[0])


def episode_task(entry,spec,stage,mode='sim',value=64,world='AB',opponent='rootpuct',repeat=0):
    seed=environment_seed(entry,world,repeat)
    return dict(kind='episode',entry=entry,group=entry['id'],stage=stage,case=entry['path'],
        world=world,opponent=opponent,seed=seed,repeat=repeat,method=spec['name'],arm=spec['arm'],
        train_seed=spec.get('train_seed',17),alpha=spec.get('alpha',1),checkpoint=spec.get('checkpoint'),
        profile=spec.get('profile',False),model_round=12,round=0,split=entry['split'],family=entry['family'],
        suite=entry['suite'],mode=mode,budget_ms=value if mode=='wall' else 200,
        fixed=value if mode=='sim' else None,control_value=value,limit=spec.get('limit'))


def base_specs():
    return [dict(name='GEOM',arm='GEOM')]+[dict(name=f'PRIOR{s}',arm='PRIOR',train_seed=s) for s in SEEDS]


def mix_specs():
    return base_specs()+[dict(name=f'MIX{alpha}_{s}',arm='MIX',alpha=alpha,train_seed=s)
                        for alpha in (.25,.5,.75) for s in SEEDS]


def run_episode(task,out):
    import run_e45b_role_free as base
    row=dict(task);row['key']=digest(task)[:24]
    if row.get('limit') is None:row.pop('limit',None)
    if row.get('fixed') is None:row.pop('fixed',None)
    payload,_=base.episode(row)
    target=out/'episodes'/task['stage']/(digest(task)+'.pkl.gz')
    save_pickle(target,payload)
    summary=payload['summary']
    return dict(summary=summary,episode=str(target),episode_sha256=sha(target))


def packet_batch(task):
    packets=load_pickle(task['packets'])
    return packets if 'index' not in task else [packets[task['index']]]


def label_job(task,out):
    items=[];details=[]
    for packet in packet_batch(task):
        spec=dict(arm=task['teacher'],seed=packet['seed'],train_seed=17,
                  force_root=task['teacher']=='POOL_UNIFORM')
        started=time.perf_counter()
        own,_,cost=prepare_public(packet,spec,task['simulations'])
        actions,weights=root_labels(own)
        items.append(dict(**own.features(own.root),label_actions=actions,label_weights=weights))
        details.append(dict(id=packet['id'],completed=own.last_stats['simulations'],
            transitions=own.last_stats['simulated_transitions'],query_ms=1000*(time.perf_counter()-started),**cost))
    arrays={k:np.stack([x[k] for x in items]).astype(np.float32) for k in items[0]}
    target=out/'labels'/task['teacher']/task['split']/(task['group']+'.npz')
    save_npz(target,arrays)
    return dict(path=str(target),sha256=sha(target),states=len(items),queries=details,
                feature_sha256=digest({k:hashlib.sha256(arrays[k].tobytes()).hexdigest() for k in KEYS}))


def probe_job(task,out,cfg):
    packet=packet_batch(task)[0]
    results=[];actions=[]
    for arm in ('OLD_JOINT','GEOM','PRIOR'):
        for budget in cfg['teacher_probe_budgets']:
            for repeat in range(1 if cfg['smoke'] else 3):
                seed=int(np.random.SeedSequence([packet['seed'],repeat,1462]).generate_state(1)[0])
                own,a,cost=prepare_public(packet,dict(arm=arm,seed=packet['seed'],search_seed=seed,train_seed=17),budget)
                la,lw=root_labels(own)
                results.append(dict(teacher=arm,budget=budget,repeat=repeat,action=a.tolist(),
                    labels=la.tolist(),weights=lw.tolist(),stats=own.last_stats,**cost))
                actions.append(a)
    values=reference_values(packet,np.asarray(actions),cfg['reference_worlds'],packet['seed']^1462)
    for row,v in zip(results,values):row['reference_values']=v.tolist()
    return dict(packet_id=packet['id'],group=packet['group'],phase=packet['phase'],queries=results,
                reference_scope='independent posterior worlds, public Single17 continuation, 8 steps')


def candidates_job(task,out,cfg):
    packet=packet_batch(task)[0];actions=[]
    for arm in ('PRIOR','GEOM'):
        own,_,_=prepare_public(packet,dict(arm=arm,seed=packet['seed'],train_seed=17),0)
        own.deadline=float('inf')
        actions.extend(own.propose(own.root) for _ in range(16))
    actions=np.asarray(actions)
    select=reference_values(packet,actions,cfg['reference_worlds'],packet['seed']^14631)
    report=reference_values(packet,actions,cfg['reference_worlds'],packet['seed']^14632)
    sets=dict(PRIOR16=list(range(16)),GEOM16=list(range(16,32)),MIX16=list(range(8))+list(range(16,24)),UNION32=list(range(32)))
    metrics={}
    for name,ids in sets.items():
        order=sorted(ids,key=lambda i:-float(select[i].mean()))
        metrics[name]={str(n):dict(original_prefix_best_report=float(report[max(ids[:n],key=lambda i:select[i].mean())].mean()),
            ideal_top1_report=float(report[order[0]].mean()))
            for n in (1,4,8,16)}
    return dict(packet_id=packet['id'],group=packet['group'],phase=packet['phase'],
        actions=actions.tolist(),selection_returns=select.tolist(),report_returns=report.tolist(),metrics=metrics,
        scope='model-local potential-guided reference, no deployment oracle claim')


def profile_job(task,out,cfg):
    packet=packet_batch(task)[0];rows=[]
    for arm in ('PRIOR','GEOM','MIX'):
        for budget in ([50] if cfg['smoke'] else [50,100,150,200]):
            for repeat in range(cfg['timing_repeats']):
                spec=dict(arm=arm,seed=packet['seed'],train_seed=17,alpha=.5,budget_ms=budget,profile=True)
                started=time.perf_counter()
                own=factory(dict(map=packet['map'],speed=packet['speed']),spec)
                construction=1000*(time.perf_counter()-started)
                started=time.perf_counter()
                for obs in packet['records'][:-1]:own.observe(obs)
                history=1000*(time.perf_counter()-started)
                a=own.act(packet['records'][-1])
                rows.append(dict(arm=arm,budget=budget,repeat=repeat,history_ms=history,
                    construction_ms=construction,action=a.tolist(),stats=own.last_stats))
        spec=dict(arm=arm,seed=packet['seed'],train_seed=17,alpha=.5,profile=True)
        own=factory(dict(map=packet['map'],speed=packet['speed']),spec)
        for obs in packet['records'][:-1]:own.observe(obs)
        own.act(packet['records'][-1],fixed_simulations=2 if cfg['smoke'] else 16)
        rows.append(dict(arm=arm,fixed_simulations=2 if cfg['smoke'] else 16,stats=own.last_stats))
    return dict(packet_id=packet['id'],group=packet['group'],phase=packet['phase'],rows=rows)


def calibration_job(task,out,cfg):
    rows=[]
    for entry in [e for e in cfg['entries'] if e['split']=='validation'][:3]:
        case=read(entry['path'])
        packet=dict(map=case['map'],speed=case['speed'],records=case['history'],seed=1460601)
        # Warmup is separate from repeated fixed-work measurements.
        prepare_public(packet,dict(arm='GEOM',seed=packet['seed']),2)
        values=[]
        for _ in range(3):
            own,_,_=prepare_public(packet,dict(arm='GEOM',seed=packet['seed']),2 if cfg['smoke'] else 16)
            values.append(own.last_stats['total_ms'])
        rows.append(dict(group=entry['id'],median_ms=float(np.median(values)),samples_ms=values))
    return dict(rows=rows,scope='fixed work; no efficacy used for concurrency selection')


def run_job(task,out):
    out=Path(out);cfg=read(out/'experiment.json');started=time.perf_counter()
    identity=digest(task);target=out/'results'/task['stage']/(identity+'.json')
    if target.exists():return read(target)
    try:
        kind=task['kind']
        if kind=='episode':result=run_episode(task,out)
        elif kind=='label':result=label_job(task,out)
        elif kind=='teacher_probe':result=probe_job(task,out,cfg)
        elif kind=='candidates':result=candidates_job(task,out,cfg)
        elif kind=='profile':result=profile_job(task,out,cfg)
        elif kind=='calibration':result=calibration_job(task,out,cfg)
        else:raise ValueError(kind)
        row=dict(valid=True,task=task,result=result,wall_seconds=time.perf_counter()-started,
                 finished=stamp(),pid=os.getpid(),cpu=list(os.sched_getaffinity(0)))
        write(target,row);return row
    except Exception as exc:
        failure=dict(valid=False,task=task,error=repr(exc),traceback=traceback.format_exc(),
                     time=stamp(),pid=os.getpid(),cpu=list(os.sched_getaffinity(0)))
        write(out/'failures'/task['stage']/(identity+'.json'),failure)
        return failure


def connect(out):
    db=sqlite3.connect(Path(out)/'jobs.sqlite',timeout=60)
    db.execute('pragma journal_mode=WAL');db.execute('pragma synchronous=FULL')
    db.execute('create table if not exists jobs(id text primary key, stage text, kind text, grp text, pool text, payload text, status text, result text, created real, updated real)')
    db.execute('create index if not exists job_status on jobs(pool,status,stage)')
    db.commit();return db


def enqueue(db,task,pool='bulk'):
    identity=digest(task);now=time.time()
    db.execute('insert or ignore into jobs values(?,?,?,?,?,?,?,?,?,?)',(identity,task['stage'],task['kind'],task.get('group',''),
        pool,json.dumps(task,sort_keys=True),'pending',None,now,now))


def completed(db,stage,group=None):
    q='select result from jobs where stage=? and status="done"';args=[stage]
    if group is not None:q+=' and grp=?';args.append(group)
    return [json.loads(r[0]) for r in db.execute(q,args)]


def pending_stage(db,stage):
    return db.execute('select count(*) from jobs where stage=? and status!="done"',(stage,)).fetchone()[0]


def initialize_jobs(db,cfg):
    for entry in cfg['entries']:
        split=entry['split']
        if split in ('train','validation','development'):
            specs=base_specs() if split!='development' else [dict(name='GEOM',arm='GEOM'),dict(name='PRIOR17',arm='PRIOR',train_seed=17)]
            for spec in specs:
                for world in ('AB','BA'):
                    for opp in OPPONENTS:
                        t=episode_task(entry,spec,'gather_'+split,value=cfg['collection_simulations'],world=world,opponent=opp)
                        if cfg['smoke']:t['limit']=3
                        enqueue(db,t)
        if split=='development':
            for spec in (mix_specs() if not cfg['smoke'] else mix_specs()[:2]+[dict(name='MIX0.5_17',arm='MIX',alpha=.5,train_seed=17)]):
                for mode,values in (('sim',[2] if cfg['smoke'] else [16,64]),('wall',[100] if cfg['smoke'] else [100,200])):
                    for value in values:
                        for world in (('AB',) if cfg['smoke'] else ('AB','BA')):
                            for opp in (('fixed_D_kernel',) if cfg['smoke'] else OPPONENTS):
                                t=episode_task(entry,spec,'E1_'+mode,mode,value,world,opp)
                                if cfg['smoke']:t['limit']=2
                                enqueue(db,t,'timing' if mode=='wall' else 'bulk')
            for arm in ('POOL_UNIFORM','POOL_GEOMETRIC'):
                spec=dict(name=arm,arm=arm)
                for mode,values in (('sim',[2] if cfg['smoke'] else [16,64]),('wall',[100] if cfg['smoke'] else [100,200])):
                    for value in values:
                        for world in (('AB',) if cfg['smoke'] else ('AB','BA')):
                            for opp in (('fixed_D_kernel',) if cfg['smoke'] else OPPONENTS):
                                t=episode_task(entry,spec,'E6_controls_'+mode,mode,value,world,opp)
                                if cfg['smoke']:t['limit']=2
                                enqueue(db,t,'timing' if mode=='wall' else 'bulk')
    db.commit()


def phase_of(obs):
    if sum(obs['active_p'])==1:return 'post_capture'
    if any(q is not None for r in obs['reports'] for q in r):return 'visible'
    return 'hidden'


def assemble_packets(out,entry,rows,cap):
    packets=[]
    for row in rows:
        result=row['result'];payload=load_pickle(result['episode']);records=read(entry['path'])['history'][:-1].copy()
        previous_visible=False
        for index,frame in enumerate(payload['trace']):
            obs=frame['observation'];records.append(obs)
            phase=phase_of(obs)
            if phase=='hidden' and previous_visible:phase='just_lost'
            previous_visible=any(q is not None for r in obs['reports'] for q in r)
            packets.append(dict(id=digest([payload['summary']['key'],index])[:24],group=entry['id'],
                map=read(entry['path'])['map'],speed=entry['speed'],records=copy.deepcopy(records),
                seed=payload['summary']['seed'],phase=phase,source=payload['summary']['method']))
    by_phase=collections.defaultdict(list)
    # Deterministic outcome-blind sampling. No success/failure filtering.
    for p in sorted(packets,key=lambda x:digest(x['id'])):by_phase[p['phase']].append(p)
    chosen=[]
    while len(chosen)<min(cap,len(packets)):
        progressed=False
        for phase in ('visible','just_lost','hidden','post_capture'):
            if by_phase[phase] and len(chosen)<cap:
                chosen.append(by_phase[phase].pop());progressed=True
        if not progressed:break
    target=out/'packets'/entry['split']/(entry['id']+'.pkl.gz')
    save_pickle(target,chosen)
    write(target.with_suffix('.json'),dict(path=str(target),sha256=sha(target),states=len(chosen),
        phases=dict(collections.Counter(p['phase'] for p in chosen)),source_episodes=len(rows),group=entry['id']))
    return target


def advance_data(db,out,cfg):
    for entry in cfg['entries']:
        split=entry['split']
        if split not in ('train','validation','development'):continue
        target=out/'packets'/split/(entry['id']+'.pkl.gz')
        if not target.exists():
            rows=completed(db,'gather_'+split,entry['id'])
            needed=8 if split=='development' else 16
            if len(rows)!=needed:continue
            target=assemble_packets(out,entry,rows,cfg['states_per_train_map'] if split!='development' else 32)
        if split in ('train','validation'):
            for teacher in ('OLD_JOINT','GEOM','POOL_UNIFORM'):
                enqueue(db,dict(kind='label',stage='labels_'+teacher+'_'+split,group=entry['id'],
                    packets=str(target),teacher=teacher,split=split,simulations=cfg['teacher_simulations']))
    # Probes are chosen from all development maps after all public packet files exist.
    dev=[e for e in cfg['entries'] if e['split']=='development']
    if all((out/'packets/development'/(e['id']+'.pkl.gz')).exists() for e in dev):
        refs=[]
        for e in dev:
            p=out/'packets/development'/(e['id']+'.pkl.gz')
            refs.extend((str(p),i) for i in range(len(load_pickle(p))))
        # Round-robin state indices across maps, rather than selecting easy outcomes.
        refs=sorted(refs,key=lambda v:(v[1],v[0]))[:cfg['probe_states']]
        for i,(p,index) in enumerate(refs):
            common=dict(group=Path(p).name,packets=p,index=index)
            enqueue(db,dict(kind='teacher_probe',stage='E2_teacher',**common))
            enqueue(db,dict(kind='candidates',stage='E3_candidates',**common))
            if i<cfg['timing_states']:enqueue(db,dict(kind='profile',stage='E4_profile',**common),'timing')
        write(out/'probe_manifest.json',dict(states=len(refs),timing_states=min(len(refs),cfg['timing_states']),refs=refs))
    db.commit()


def consolidate(out,db,cfg,teacher,split):
    target=out/'datasets'/teacher/(split+'.npz')
    if target.exists():return target
    entries=[e for e in cfg['entries'] if e['split']==split]
    rows=completed(db,'labels_'+teacher+'_'+split)
    if len(rows)!=len(entries):return None
    rows=sorted(rows,key=lambda r:r['task']['group']);arrays=[]
    if teacher!='OLD_JOINT':
        reference={r['task']['group']:r['result']['feature_sha256'] for r in completed(db,'labels_OLD_JOINT_'+split)}
        for r in rows:
            if r['task']['group'] in reference and r['result']['feature_sha256']!=reference[r['task']['group']]:
                raise RuntimeError('Teachers received different public features: '+r['task']['group'])
    for row in rows:
        info=row['result']
        assert sha(info['path'])==info['sha256']
        with np.load(info['path'],allow_pickle=False) as d:arrays.append({k:d[k] for k in d.files})
    save_npz(target,{k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]})
    write(target.with_suffix('.json'),dict(states=sum(len(a['public']) for a in arrays),
        sha256=sha(target),source_jobs=[digest(r['task']) for r in rows]))
    return target


def data_tensors(path,device):
    with np.load(path,allow_pickle=False) as d:return {k:torch.from_numpy(d[k]).float().to(device) for k in d.files}


def student_loss(model,batch,rank=False):
    if rank:
        logits=model(batch,batch['label_actions'])
        logits=logits.masked_fill(batch['label_weights']<=0,-1e9)
        return -(torch.log_softmax(logits,-1)*batch['label_weights']).sum(-1).mean()
    params,_=model(batch)
    return -(log_density(params,batch['label_actions'],batch['agents'][:,:,4],model.kind)*batch['label_weights']).sum(-1).mean()


def train_model(out,job,physical_gpu):
    ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    out=Path(out);cfg=read(out/'experiment.json');torch.set_num_threads(1)
    os.sched_setaffinity(0,{int(job['cpu'])})
    torch.manual_seed(job['seed']);torch.cuda.manual_seed_all(job['seed'])
    torch.use_deterministic_algorithms(True)
    device='cuda:0';torch.cuda.set_device(0)
    rank=job['network']=='rank'
    model=(RankNetwork() if rank else Network(job['network'])).to(device)
    folder=out/'training'/job['name'];folder.mkdir(parents=True,exist_ok=True)
    resume=folder/'resume.pt';selected=folder/'selected.npz';selected_pt=folder/'selected.pt'
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=1e-5)
    rng=np.random.default_rng(job['seed']);begin=0;best=float('inf');logs=[]
    if job.get('initial_pt'):model.load_state_dict(torch.load(job['initial_pt'],map_location=device,weights_only=True))
    if resume.exists():
        saved=torch.load(resume,map_location=device,weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['rng'];begin=saved['epoch'];best=saved['best'];logs=saved['logs']
    train=data_tensors(job['train_path'],device);val=data_tensors(job['validation_path'],device)
    extra=data_tensors(job['extra_path'],device) if job.get('extra_path') else None
    started=time.perf_counter()
    for epoch in range(begin,job['epochs']):
        model.train();total=0.;n=0
        order=rng.permutation(len(train['public']))
        for ids in np.array_split(order,max(1,int(np.ceil(len(order)/128)))):
            ids=torch.as_tensor(ids,device=device);batch={k:v[ids] for k,v in train.items()}
            if extra is not None:
                count=max(1,len(ids)//2)
                ex=torch.as_tensor(rng.integers(len(extra['public']),size=count),device=device)
                batch={k:torch.cat((v[:len(ids)-count],extra[k][ex]),0) for k,v in batch.items()}
            batch=augment(batch,bool(rng.integers(2)),bool(rng.integers(2)))
            optimizer.zero_grad();loss=student_loss(model,batch,rank)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite E46 loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
            total+=float(loss.detach())*len(ids);n+=len(ids)
        model.eval()
        with torch.no_grad():
            vl=[]
            for i in range(0,len(val['public']),256):
                b={k:v[i:i+256] for k,v in val.items()}
                vl.append((float(student_loss(model,b,rank)),len(b['public'])))
            value=sum(a*b for a,b in vl)/sum(b for _,b in vl)
        logs.append(dict(epoch=epoch+1,train_loss=total/n,validation_loss=value))
        if value<best:
            best=value;model.export(selected)
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},selected_pt)
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,
                   epoch=epoch+1,best=best,logs=logs)
        torch.save(state,resume.with_suffix('.tmp'));os.replace(resume.with_suffix('.tmp'),resume)
        write(folder/'progress.json',dict(job=job,epoch=epoch+1,epochs=job['epochs'],logs=logs,
            physical_gpu=physical_gpu,cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),time=stamp()))
    result=dict(job=job,path=str(selected),pt=str(selected_pt),sha256=sha(selected),best_validation_loss=best,
                logs=logs,seconds=time.perf_counter()-started,physical_gpu=physical_gpu,
                value_loss_used=False,parameters=sum(p.numel() for p in model.parameters()))
    write(folder/'result.json',result)
    return result


def training_jobs(out,db,cfg):
    ready=[]
    for label,(network,teacher) in cfg['train_configurations'].items():
        train=consolidate(out,db,cfg,teacher,'train');val=consolidate(out,db,cfg,teacher,'validation')
        if not train or not val:continue
        for seed in SEEDS:
            name=f'{label}_{seed}_base'
            ready.append(dict(name=name,configuration=label,network=network,teacher=teacher,seed=seed,
                phase='base',epochs=cfg['epochs'],train_path=str(train),validation_path=str(val)))
    return ready


def model_spec(result):
    job=result['job'];rank=job['network']=='rank'
    return dict(name=job['name'],arm='POOL_LEARNED' if rank else 'STUDENT',
                train_seed=job['seed'],checkpoint=result['path'])


def enqueue_model_evaluation(db,cfg,result,stage='development_models'):
    spec=model_spec(result)
    for e in cfg['entries']:
        if e['split']!='development':continue
        for mode,values in (('sim',[2] if cfg['smoke'] else [16,64]),('wall',[100] if cfg['smoke'] else [100,200])):
            for value in values:
                for world in (('AB',) if cfg['smoke'] else ('AB','BA')):
                    for opp in (('fixed_D_kernel',) if cfg['smoke'] else OPPONENTS):
                        t=episode_task(e,spec,stage+'_'+mode,mode,value,world,opp)
                        if cfg['smoke']:t['limit']=2
                        enqueue(db,t,'timing' if mode=='wall' else 'bulk')


def resource_layout():
    rows=subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET'],text=True)
    cores=collections.defaultdict(list)
    available=os.sched_getaffinity(0)
    for line in rows.splitlines():
        if line.startswith('#'):continue
        cpu,core,socket=map(int,line.split(',')[:3])
        if cpu in available:cores[(socket,core)].append(cpu)
    physical=[sorted(v) for k,v in sorted(cores.items())]
    if len(physical)<24:raise RuntimeError('Insufficient physical cores for registered E46 resource layout')
    reserved=[x for core in physical[:10] for x in core]
    if len(reserved)<20:raise RuntimeError('Cannot reserve 20 logical CPUs with this topology')
    train=[x for core in physical[10:13] for x in core]
    timing=[core[0] for core in physical[-8:]]
    timing_siblings=[x for core in physical[-8:] for x in core[1:]]
    bulk=[x for core in physical[13:-8] for x in core]
    return dict(reserved=reserved,bulk=bulk,training=train,timing=timing,
                timing_siblings_idle=timing_siblings,logical_total=len(available),physical_total=len(physical))


def choose_gpu():
    values=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    free={int(a):int(b) for line in values.splitlines() for a,b in [line.split(',')]}
    return next((gpu for gpu in (3,2,1) if free.get(gpu,0)>=3072),None)


def gpu_schedule(out,jobs,active,layout,max_jobs=3):
    for name,(proc,handle,job) in list(active.items()):
        code=proc.poll()
        if code is not None:
            handle.close();active.pop(name)
            if code!=0:write(out/'training'/name/'failure.json',dict(job=job,exit_code=code,time=stamp()))
    for job in jobs:
        folder=out/'training'/job['name']
        if len(active)>=max_jobs:break
        if job['name'] in active or (folder/'result.json').exists() or (folder/'failure.json').exists():continue
        gpu=choose_gpu()
        if gpu is None:break
        folder.mkdir(parents=True,exist_ok=True)
        used={v[2]['cpu'] for v in active.values()}
        cpu=next(c for c in layout['training'] if c not in used)
        job=dict(job,cpu=cpu);write(folder/'job.json',job)
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        handle=open(folder/'console.log','a')
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--output',str(out),
            '--train-job',str(folder/'job.json'),'--physical-gpu',str(gpu)],env=env,stdout=handle,stderr=subprocess.STDOUT)
        active[job['name']]=(proc,handle,job)
        event(out,'training_started',name=job['name'],physical_gpu=gpu,pid=proc.pid,cpu=cpu)


def correction_extra(out,db,cfg,job,round_):
    stage=f'correction_{job["configuration"]}_{job["seed"]}_r{round_}'
    target=out/'datasets'/stage/'extra.npz'
    if target.exists():return target
    train_entries=[e for e in cfg['entries'] if e['split']=='train']
    # Equal family representation and predefined selection, independent of success.
    family_groups=collections.defaultdict(list)
    for e in train_entries:family_groups[e['family']].append(e)
    selected=[]
    while len(selected)<min(cfg['correction_maps'],len(train_entries)):
        for family in sorted(family_groups):
            if family_groups[family] and len(selected)<cfg['correction_maps']:
                selected.append(family_groups[family].pop(0))
    parent_name=f'{job["configuration"]}_{job["seed"]}_'+('base' if round_==1 else f'r{round_-1}_aggregate')
    parent_path=out/'training'/parent_name/'result.json'
    if not parent_path.exists():return None
    parent=read(parent_path);spec=model_spec(parent)
    for entry in selected:
        for world in ('AB','BA'):
            for opp in OPPONENTS:
                task=episode_task(entry,spec,stage,value=cfg['collection_simulations'],world=world,opponent=opp,repeat=round_)
                if cfg['smoke']:task['limit']=3
                enqueue(db,task)
    label_stage=stage+'_labels'
    for entry in selected:
        packet_path=out/'packets'/stage/(entry['id']+'.pkl.gz')
        if not packet_path.exists():
            rows=completed(db,stage,entry['id'])
            if len(rows)!=4:continue
            # Public records are derived from actual student visits; namespace
            # the packet shard so base training packets can never be overwritten.
            assemble_packets(out,dict(entry,split=stage),rows,4*cfg['correction_states_per_episode'])
        enqueue(db,dict(kind='label',stage=label_stage,group=entry['id'],packets=str(packet_path),
            teacher=job['teacher'],split=stage,simulations=cfg['teacher_simulations']))
    rows=completed(db,label_stage)
    if len(rows)!=len(selected):return None
    arrays=[]
    for row in sorted(rows,key=lambda r:r['task']['group']):
        info=row['result'];assert sha(info['path'])==info['sha256']
        with np.load(info['path'],allow_pickle=False) as d:arrays.append({k:d[k] for k in d.files})
    if round_>1:
        previous=out/'datasets'/f'correction_{job["configuration"]}_{job["seed"]}_r{round_-1}'/'extra.npz'
        with np.load(previous,allow_pickle=False) as d:arrays.append({k:d[k] for k in d.files})
    save_npz(target,{k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]})
    write(target.with_suffix('.json'),dict(states=sum(len(a['public']) for a in arrays),sha256=sha(target),
        teacher=job['teacher'],source_stage=stage,includes_previous_rounds=round_>1))
    return target


def arm_key(task):
    name=task['method']
    if task['arm']=='GEOM':return 'GEOM'
    if task['arm']=='PRIOR':return 'PRIOR'
    if task['arm']=='MIX':return f'MIX{task["alpha"]}'
    if task['arm'].startswith('POOL_') and not task.get('checkpoint'):return task['arm']
    parts=name.split('_')
    return parts[0]+'_'+'_'.join(parts[2:])


def development_summary(db,out):
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    details=[]
    for stage in ('E1_wall','E6_controls_wall','development_models_wall'):
        for row in completed(db,stage):
            task=row['task'];s=row['result']['summary']
            if task['control_value'] not in (100,200):continue
            key=arm_key(task)
            groups[key][task['group']].append(s['utility'])
            details.append(dict(arm=key,group=task['group'],budget=task['control_value'],
                train_seed=task['train_seed'],world=task['world'],opponent=task['opponent'],
                utility=s['utility'],double=s['double_capture'],cost=s['completion_cost']))
    scores={arm:float(np.mean([np.mean(v) for v in maps.values()])) for arm,maps in groups.items()}
    write(out/'development_summary.json',dict(scores=scores,rows=details,selection='equal map, equal seed, budgets100/200'))
    return scores


def final_specs(out,cfg,scores):
    alpha=max((.25,.5,.75),key=lambda a:(scores.get(f'MIX{a}',-1e9),-abs(a-.5)))
    specs=base_specs()+[dict(name=f'MIX{alpha}_{s}',arm='MIX',alpha=alpha,train_seed=s) for s in SEEDS]
    new_keys=[k for k in scores if k.startswith(('L1_','L2_','L3_','L4_','L5_'))]
    # Prefer base (simpler) in exact ties; selection never inspects sealed output.
    chosen=sorted(new_keys,key=lambda k:(-scores[k],0 if k.endswith('base') else 1,k))[:2]
    for key in chosen:
        label,phase=key.split('_',1)
        for seed in SEEDS:
            result=read(out/'training'/f'{label}_{seed}_{phase}'/'result.json')
            specs.append(model_spec(result))
    return specs,dict(mixed_alpha=alpha,new_configurations=chosen,development_scores=scores,
                      final_map_count=sum(e['split']=='sealed' for e in cfg['entries']))


def report_final(db,out,cfg):
    rows=[]
    for stage in ('E8_final_sim','E8_final_wall'):
        for r in completed(db,stage):
            t=r['task'];s=r['result']['summary']
            rows.append(dict(arm=arm_key(t),group=t['group'],suite=t['suite'],mode=t['mode'],
                budget=t['control_value'],seed=t['train_seed'],world=t['world'],opponent=t['opponent'],
                utility=s['utility'],double=float(s['double_capture']),cost=s['completion_cost'],
                p95=float(np.percentile(s['latencies_ms'],95)),overruns=s['online_overruns'],
                decisions=s['decisions'],transitions=s['simulated_transitions']))
    aggregate=collections.defaultdict(list)
    for r in rows:aggregate[(r['arm'],r['suite'],r['mode'],r['budget'],r['opponent'])].append(r)
    table=[]
    for key,rr in sorted(aggregate.items()):
        bymap=collections.defaultdict(list)
        for r in rr:bymap[r['group']].append(r)
        table.append(dict(arm=key[0],suite=key[1],mode=key[2],budget=key[3],opponent=key[4],
            utility=float(np.mean([np.mean([r['utility'] for r in v]) for v in bymap.values()])),
            double=float(np.mean([np.mean([r['double'] for r in v]) for v in bymap.values()])),
            cost=float(np.mean([np.mean([r['cost'] for r in v]) for v in bymap.values()])),episodes=len(rr),maps=len(bymap)))
    selection=read(out/'final/selection.json');comparisons=[];vectors=[]
    for new in selection['new_configurations']:
        for reference in ('GEOM','PRIOR',f'MIX{selection["mixed_alpha"]}'):
            for budget in (100,200):
                for suite in ('ID','OOD'):
                    rr=[r for r in rows if r['mode']=='wall' and r['budget']==budget and r['suite']==suite]
                    groups=sorted({r['group'] for r in rr});delta=[]
                    for group in groups:
                        a=[r['utility'] for r in rr if r['group']==group and r['arm']==new]
                        b=[r['utility'] for r in rr if r['group']==group and r['arm']==reference]
                        if a and b:delta.append(float(np.mean(a)-np.mean(b)))
                    if delta:
                        comparisons.append(dict(method=new,reference=reference,budget=budget,suite=suite,
                            maps=len(delta),mean=float(np.mean(delta))))
                        vectors.append(np.asarray(delta))
    if vectors:
        rng=np.random.default_rng(BASE+818);boot=[]
        # Identical resampling indices within each suite across all comparisons.
        counts={s:len(vectors[i]) for i,s in enumerate(c['suite'] for c in comparisons)}
        indices={s:rng.integers(n,size=(10000,n)) for s,n in counts.items()}
        for c,d in zip(comparisons,vectors):boot.append(d[indices[c['suite']]].mean(1)-d.mean())
        radius=float(np.quantile(np.max(np.abs(np.stack(boot,axis=1)),axis=1),.95))
        for c in comparisons:c['simultaneous95']=[c['mean']-radius,c['mean']+radius]
    write(out/'final/results.json',dict(rows=rows,table=table,comparisons=comparisons,
        scope='independent frozen-map confirmation; current three training seeds fixed, no seed-population CI'))
    lines=['# E46最终确认结果','','|方法|范围|预算定义|预算|对手|双捕|效用|完成成本|',
           '|---|---|---|---:|---|---:|---:|---:|']
    for r in table:lines.append(f'|{r["arm"]}|{r["suite"]}|{r["mode"]}|{r["budget"]}|{r["opponent"]}|{100*r["double"]:.2f}%|{r["utility"]:.4f}|{r["cost"]:.2f}|')
    lines+=['','主要效用比较采用按地图成组的全家族同时区间；详细时延和全部逐局结果见results.json。',
            '失败/非法任务不静默删除；若存在未解决失败，本报告不能标记完整确认。']
    (out/'final/REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'final/COMPLETE.json',dict(finished=stamp(),episodes=len(rows),selection=selection,
        source_verification='run-level frozen source and model hashes',result_sha256=sha(out/'final/results.json')))


def advance_later_stages(db,out,cfg,jobs,active_gpu,layout):
    base_jobs=list(jobs)
    for base_job in base_jobs:
        for round_ in range(1,cfg['correction_rounds']+1):
            extra=correction_extra(out,db,cfg,base_job,round_)
            if extra is None:break
            parent_name=f'{base_job["configuration"]}_{base_job["seed"]}_'+('base' if round_==1 else f'r{round_-1}_aggregate')
            parent=read(out/'training'/parent_name/'result.json')
            for route in ('repeat','aggregate'):
                name=f'{base_job["configuration"]}_{base_job["seed"]}_r{round_}_{route}'
                jobs.append(dict(base_job,name=name,phase=f'r{round_}_{route}',epochs=cfg['correction_epochs'],
                    initial_pt=parent['pt'],extra_path=str(extra) if route=='aggregate' else None))
    # Every newly completed checkpoint receives the same development comparisons.
    for job in jobs:
        p=out/'training'/job['name']/'result.json'
        if p.exists():enqueue_model_evaluation(db,cfg,read(p))
    expected=15*(1+2*cfg['correction_rounds'])
    ready=len(jobs)==expected and all((out/'training'/j['name']/'result.json').exists() for j in jobs)
    if not ready:return
    if db.execute('select count(*) from jobs where stage not like "E8_%" and status!="done"').fetchone()[0]:return
    if not (out/'final/selection.json').exists():
        scores=development_summary(db,out)
        specs,selection=final_specs(out,cfg,scores)
        selection['models']=specs;selection['frozen_at']=stamp()
        write(out/'final/selection.json',selection)
        calibration=read(out/'timing_calibration.json')
        calibration.update(passed=False,enqueued=False,round=calibration['round']+1,
                           workers=1 if cfg['smoke'] else len(layout['timing']))
        write(out/'timing_calibration.json',calibration)
        event(out,'final_selection_frozen',selection=selection)
    selection=read(out/'final/selection.json')
    entries=[e for e in cfg['entries'] if e['split']=='sealed']
    specs=selection['models']
    if cfg['smoke']:
        entries=entries[:1];specs=specs[:2]+specs[-1:]
    for e in entries:
        for spec in specs:
            for mode,values in (('sim',[2] if cfg['smoke'] else [16,64,128]),('wall',[100] if cfg['smoke'] else [50,100,200])):
                for value in values:
                    for world in (('AB',) if cfg['smoke'] else ('AB','BA')):
                        for opp in (('fixed_D_kernel',) if cfg['smoke'] else OPPONENTS):
                            t=episode_task(e,spec,'E8_final_'+mode,mode,value,world,opp)
                            if cfg['smoke']:t['limit']=2
                            enqueue(db,t,'timing' if mode=='wall' else 'bulk')
    db.commit()
    if pending_stage(db,'E8_final_sim')==0 and pending_stage(db,'E8_final_wall')==0:
        verify(out,cfg);report_final(db,out,cfg)


def dispatch(db,pool,active,capacity,out,pool_name,only_stage=None):
    stages=[r[0] for r in db.execute('select distinct stage from jobs where pool=? and status="pending" order by stage',(pool_name,))]
    if only_stage is not None:stages=[s for s in stages if s==only_stage]
    while len(active)<capacity and stages:
        stage=stages[DISPATCH_CURSOR[pool_name]%len(stages)]
        DISPATCH_CURSOR[pool_name]+=1
        row=db.execute('select id,payload from jobs where pool=? and status="pending" and stage=? order by created,id limit 1',(pool_name,stage)).fetchone()
        if not row:
            stages.remove(stage);continue
        identity,payload=row;task=json.loads(payload)
        db.execute('update jobs set status="running",updated=? where id=?',(time.time(),identity))
        future=pool.submit(run_job,task,str(out));active[future]=identity
    db.commit()


def collect_finished(db,active):
    for future,identity in list(active.items()):
        if not future.done():continue
        try:row=future.result()
        except Exception as exc:row=dict(valid=False,error=repr(exc),traceback=traceback.format_exc())
        db.execute('update jobs set status=?,result=?,updated=? where id=?',
            ('done' if row.get('valid') else 'failed',json.dumps(row),time.time(),identity))
        active.pop(future)
    db.commit()


def progress(out,db,active_gpu,phase='running'):
    stages={}
    for stage,status,n in db.execute('select stage,status,count(*) from jobs group by stage,status'):
        stages.setdefault(stage,{})[status]=n
    models={p.parent.name:read(p) for p in (out/'training').glob('*/progress.json')}
    report=dict(experiment='E46',phase=phase,time=stamp(),pid=os.getpid(),stages=stages,
        training={name:dict(epoch=v['epoch'],epochs=v['epochs'],physical_gpu=v['physical_gpu']) for name,v in models.items()},
        active_gpu_jobs=list(active_gpu),failures=db.execute('select count(*) from jobs where status="failed"').fetchone()[0])
    report['training_failures']=[str(p) for p in (out/'training').glob('*/failure.json')]
    if (out/'timing_calibration.json').exists():report['timing_calibration']=read(out/'timing_calibration.json')
    write(out/'progress.json',report)
    text=['# E46实时进展','',f'更新时间：{report["time"]}；状态：{phase}。','',
          '|实验/阶段|完成|运行|待运行|失败|','|---|---:|---:|---:|---:|']
    for name,v in sorted(stages.items()):text.append(f'|{name}|{v.get("done",0)}|{v.get("running",0)}|{v.get("pending",0)}|{v.get("failed",0)}|')
    text+=['','GPU作业：'+', '.join(active_gpu),'','固定模拟任务高并发；墙钟任务使用独立CPU池。结果尚未完成时不作科学结论。']
    (out/'LIVE.md').write_text('\n'.join(text)+'\n')
    return report


def timing_gate(db,out,cfg,capacity,bulk_active):
    """Under-load calibration; fall back to an isolated timing window if needed."""
    path=out/'timing_calibration.json';state=read(path)
    if state.get('passed'):return state
    stage=f'E0_calibration_{state["round"]}'
    if not state.get('enqueued'):
        if state.get('require_bulk_idle') and bulk_active:return state
        count=state['workers']
        for repeat in range(count):
            enqueue(db,dict(kind='calibration',stage=stage,group=str(repeat),repeat=repeat))
            # Calibration tasks belong to timing, not the bulk pool.
            db.execute('update jobs set pool="timing" where stage=?',(stage,))
        state['enqueued']=True;state['stage']=stage;write(path,state);db.commit()
    rows=completed(db,stage)
    if len(rows)!=state['workers']:return state
    ref=read(out/'timing_serial_reference.json')
    baseline={r['group']:r['median_ms'] for r in ref['rows']}
    ratios=[r['median_ms']/baseline[r['group']] for v in rows for r in v['result']['rows']]
    ratio=float(np.median(ratios));state.setdefault('attempts',[]).append(dict(round=state['round'],workers=state['workers'],
        median_fixed_work_slowdown=ratio,bulk_idle=state.get('require_bulk_idle',False)))
    if ratio<=1.35:
        state['passed']=True;state['slowdown_ratio']=ratio
        state['bulk_paused_for_wall']=state.get('require_bulk_idle',False)
    else:
        if state['workers']>1:state['workers']=max(1,state['workers']//2)
        elif not state.get('require_bulk_idle'):state['workers']=capacity;state['require_bulk_idle']=True
        else:raise RuntimeError('Independent timing calibration failed even at one worker; preserve outputs, no timing claim')
        state['round']+=1;state['enqueued']=False
    write(path,state);return state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--train-job',type=Path)
    parser.add_argument('--physical-gpu',type=int)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.train_job:
        if args.physical_gpu not in (3,2,1):raise RuntimeError('GPU0 is forbidden')
        train_model(out,read(args.train_job),args.physical_gpu);return
    lock=open(out/'coordinator.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=prepare(out,args.smoke)
    if args.prepare_only:return
    layout=resource_layout();write(out/'resources.json',dict(**layout,gpu_preference=[3,2,1],time=stamp()))
    os.sched_setaffinity(0,set(layout['training']))
    configure(out);db=connect(out)
    db.execute('update jobs set status="pending" where status="running"');db.commit()
    initialize_jobs(db,cfg)
    if not (out/'timing_serial_reference.json').exists():
        before=os.sched_getaffinity(0);os.sched_setaffinity(0,{layout['timing'][0]})
        try:write(out/'timing_serial_reference.json',calibration_job({},out,cfg))
        finally:os.sched_setaffinity(0,before)
    if not (out/'timing_calibration.json').exists():
        write(out/'timing_calibration.json',dict(round=0,workers=1 if args.smoke else len(layout['timing']),
            passed=False,enqueued=False,attempts=[]))
    context=mp.get_context('spawn');bq=context.Queue();tq=context.Queue()
    bulk_cpus=layout['bulk'][::2][:6] if args.smoke else layout['bulk']
    time_cpus=layout['timing'][:1] if args.smoke else layout['timing']
    for c in bulk_cpus:bq.put(c)
    for c in time_cpus:tq.put(c)
    bulk=futures.ProcessPoolExecutor(len(bulk_cpus),mp_context=context,initializer=worker_init,initargs=(str(out),bq))
    timing=futures.ProcessPoolExecutor(len(time_cpus),mp_context=context,initializer=worker_init,initargs=(str(out),tq))
    ba={};ta={};ga={};last=0.;finished=False
    event(out,'coordinator_started',bulk_workers=len(bulk_cpus),timing_workers=len(time_cpus),layout=layout)
    try:
        while True:
            collect_finished(db,ba);collect_finished(db,ta)
            if time.time()-last>=(1 if cfg['smoke'] else 15):
                advance_data(db,out,cfg)
                jobs=training_jobs(out,db,cfg)
                # Later dependencies are handled in the same persistent coordinator.
                advance_later_stages(db,out,cfg,jobs,ga,layout)
                gpu_schedule(out,jobs,ga,layout,3)
                db.commit();progress(out,db,ga);last=time.time()
            timing_state=timing_gate(db,out,cfg,len(time_cpus),bool(ba))
            pause_bulk=timing_state.get('require_bulk_idle',False)
            timing_waiting=db.execute('select count(*) from jobs where pool="timing" and status in ("pending","running")').fetchone()[0]
            if not pause_bulk or (timing_state.get('passed') and not timing_waiting):
                dispatch(db,bulk,ba,len(bulk_cpus),out,'bulk')
            if timing_state.get('passed'):
                if not pause_bulk or not ba:dispatch(db,timing,ta,timing_state['workers'],out,'timing')
            else:
                dispatch(db,timing,ta,timing_state['workers'],out,'timing',only_stage=timing_state.get('stage'))
            if not ba and not ta and not ga and db.execute('select count(*) from jobs where status="pending"').fetchone()[0]==0:
                if (out/'final/COMPLETE.json').exists():finished=True;break
                if db.execute('select count(*) from jobs where status="failed"').fetchone()[0] or list((out/'training').glob('*/failure.json')):
                    event(out,'dependent_stages_blocked_by_failures');break
            time.sleep(1)
    finally:
        progress(out,db,ga,'finished' if finished else 'stopped_incomplete')
        bulk.shutdown(wait=True);timing.shutdown(wait=True)
        for proc,handle,job in ga.values():
            proc.wait();handle.close()


if __name__=='__main__':main()
