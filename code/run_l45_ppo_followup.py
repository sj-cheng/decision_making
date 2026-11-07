"""Frozen l45 PPO follow-up: audit, search screening, conditional reward trial."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import contextlib
import copy
import fcntl
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import threading
import time
import numpy as np
import torch
import train_l45_ppo as o
import confirm_l45_final_search as previous

ROOT = o.ROOT
DEFAULT_OUTPUT = ROOT/'current/l45_ppo_followup_20261023'
SOURCE = previous.DEFAULT_SOURCE
FINAL_SOURCE = previous.DEFAULT_OUTPUT
SEEDS = (0,1,2)
CONFIGS = dict(base=(.5,.1),low=(.25,.1),high=(.75,.1),novalue=(.5,0.))
SPLITS = dict(screen_four=('four',40),screen_none=('none',20),
    search_four=('four',200),search_none=('none',100),
    validation_four=('four',60),validation_none=('none',30),
    reward_four=('four',200),reward_none=('none',100),
    guard_four=('four',100),guard_none=('none',100),latency_four=('four',30))
_ID_CACHE = {}


def config(args):
    return dict(smoke=args.smoke,seeds=list(SEEDS),configs=CONFIGS,
        episodes=8 if args.smoke else 512,batch=4 if args.smoke else 64,
        eval_every=4 if args.smoke else 256,ppo_epochs=4,minibatch=256,
        lr=3e-5,critic_lr=3e-4,clip=.1,entropy=.001,target_kl=.02,
        gamma=1.,gae_lambda=.95,gradient_clip=.5,completion_bonus=dict(A=0.,B=1.),
        search_budget=3 if args.smoke else 1000,search_only_budget=3 if args.smoke else 3000,
        depth=100,opponents=o.OPPONENTS,training_mix={'Policy Only':.5,'Search + Learning':.5},
        splits={k:[layout,1 if args.smoke else n] for k,(layout,n) in SPLITS.items()},
        screen_gain=.03,confirmation_gain=.05,illegal_margin=.02,regression_margin=.03,
        no_promotion=True,continue_formal=False)


def settings(args,name='base'):
    c=config(args); bp,bv=CONFIGS[name]
    return dict(search_budget=c['search_budget'],search_only_budget=c['search_only_budget'],
                search_depth=100,beta_policy=bp,beta_value=bv)


def make_solver(p,models,method,role,requested,seed):
    """The new adapter passes requested betas; the historical helper stays frozen."""
    if requested['search_depth']!=100:
        raise ValueError('Search depth must remain 100')
    budget=requested['search_only_budget'] if method=='Search Only' else requested['search_budget']
    with contextlib.redirect_stdout(io.StringIO()):
        solver=o.build_solver(p,method,str(ROOT),budget,model_dir=str(models),model_iteration=45,
                              beta_policy=requested['beta_policy'],beta_value=requested['beta_value'])
    if method=='Policy Only':
        actual=dict(method=method)
    else:
        ss=solver.solver_settings
        expected=(0.,0.) if method=='Search Only' else (requested['beta_policy'],requested['beta_value'])
        if ss.number_simulations!=budget or ss.search_depth!=100 or not np.allclose(
                [ss.beta_policy,ss.beta_value],expected,atol=1e-7,rtol=0):
            raise ValueError('C++ did not receive the requested search parameters')
        actual=dict(method=method,budget=int(ss.number_simulations),depth=int(ss.search_depth),
                    beta_policy=float(ss.beta_policy),beta_value=float(ss.beta_value))
        solver.solver_wrapper.set_seed(int(seed))
    masked=o.RoleMaskedSolver(solver,o.TEAMS[role]); masked.actual_settings=actual
    return masked


def policy_id(directory):
    """Tensor content identity permits selected/final aliases to share evaluations."""
    directory=Path(directory); stamp=tuple((p.stat().st_mtime_ns,p.stat().st_size) for p in
                                          (directory/n for n in o.MODEL_NAMES))
    key=(str(directory),stamp)
    if key not in _ID_CACHE:
        digest=hashlib.sha256()
        for name in o.MODEL_NAMES:
            for field,tensor in sorted(o.load_torch(directory/name).items()):
                data=tensor.detach().cpu().contiguous()
                digest.update(f'{name}:{field}:{data.dtype}:{tuple(data.shape)}'.encode())
                digest.update(data.numpy().tobytes())
        _ID_CACHE[key]=digest.hexdigest()
    return _ID_CACHE[key]


def model(label,directory,seed=None,cfg='base'):
    directory=Path(directory)
    if (directory/'export.json').exists() and o.read_json(directory/'export.json')['models']!=o.model_fingerprint(directory):
        raise ValueError('Exported model fingerprint changed')
    return dict(label=label,models=str(directory),seed=seed,config=cfg,
                model_hash=o.model_fingerprint(directory),policy_id=policy_id(directory))


def fixed_models(args,cfg='base'):
    return [model('baseline',args.output/'models/baseline',cfg=cfg)]+[
        model(f'old{seed}',args.output/f'models/seed{seed}_final',seed,cfg) for seed in SEEDS]


def source_record():
    record=previous.source_contract(SOURCE)
    previous.verify(previous.parse_args([]))
    return dict(parent=record,final_signature=o.sha(FINAL_SOURCE/'signature.json'),
                final_results=o.sha(FINAL_SOURCE/'results.json'))


def record(args):
    return json.loads(json.dumps(dict(config=config(args),source=source_record(),
        implementation=dict(runner=o.sha(__file__),previous=o.sha(previous.__file__),
                            original=o.implementation_files()))))


def history_scenes():
    used=previous.used_physical_scenes(SOURCE)
    used.update(previous.physical_hash(s) for cases in o.read_json(FINAL_SOURCE/'scenarios.json').values() for s in cases)
    return used


def verify(args):
    out=args.output
    current=record(args)
    if o.read_json(out/'experiment.json')!=current:
        raise ValueError('Source/configuration/implementation changed; refuse mixed resume')
    sig=o.read_json(out/'signature.json')
    if sig['scenarios']!=o.sha(out/'scenarios.json') or sig['initial']!=o.read_json(out/'initial_manifest.json'):
        raise ValueError('Scenario/initial-state signature changed')
    for seed,digest in sig['initial'].items():
        if o.sha(out/f'initial/seed{seed}.pt')!=digest:
            raise ValueError('Shared initial critic/actors changed')
    registry=current['source']['parent']['registry']
    for item in registry:
        if item['label']=='baseline' or item['label'].endswith('_final'):
            if o.model_fingerprint(out/'models'/item['label'])!=item['model_hash']:
                raise ValueError('Frozen model changed')


def prepare(args):
    out=args.output
    protected=(SOURCE.resolve(),FINAL_SOURCE.resolve(),o.SOURCE_MODELS.resolve())
    if any(out==p or out.is_relative_to(p) or p.is_relative_to(out) for p in protected):
        raise ValueError('Output must be separate from all historical sources')
    out.mkdir(parents=True,exist_ok=True)
    current=record(args)
    if (out/'experiment.json').exists() and o.read_json(out/'experiment.json')!=current:
        raise ValueError('Experiment fingerprint/configuration changed')
    if not (out/'experiment.json').exists():
        for item in current['source']['parent']['registry']:
            if item['label']=='baseline' or item['label'].endswith('_final'):
                target=out/'models'/item['label']; target.mkdir(parents=True,exist_ok=True)
                for name in o.MODEL_NAMES: shutil.copyfile(Path(item['models'])/name,target/name)
        o.write_json(out/'experiment.json',current)
    if not (out/'scenarios.json').exists():
        used=history_scenes(); old_count=len(used); banks={}
        definitions=[(name,layout,n,650+i,0) for i,(name,(layout,n)) in enumerate(SPLITS.items())]
        definitions += [(f'train{seed}','four',config(args)['episodes'],680,seed) for seed in SEEDS]
        for name,layout,count,phase,seed in definitions:
            count=(1 if args.smoke else count) if not name.startswith('train') else count
            bank=[]; index=0
            while len(bank)<count:
                spec=o.make_scenario(o.scenario_seed(phase,20261023+seed,index),layout); index+=1
                identity=previous.physical_hash(spec)
                if identity in used: continue
                used.add(identity); bank.append(spec)
            banks[name]=bank
        o.write_json(out/'scenarios.json',banks)
        o.write_json(out/'scenario_independence.json',dict(previous=old_count,new=len(used)-old_count,
            overlap=0,paired_AB_training_is_intentional=True))
    if not (out/'initial_manifest.json').exists():
        hashes={}
        for seed in SEEDS:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            actors=o.load_actors(o.original_problem(),directory=out/'models/baseline')
            critic=o.TrainingCritic()
            path=out/f'initial/seed{seed}.pt'
            o.save_torch(path,dict(actors=o.actor_states(actors),critic=critic.state_dict(),rng=o.rng_state()))
            hashes[str(seed)]=o.sha(path)
        o.write_json(out/'initial_manifest.json',hashes)
    signature=dict(scenarios=o.sha(out/'scenarios.json'),initial=o.read_json(out/'initial_manifest.json'))
    if (out/'signature.json').exists() and o.read_json(out/'signature.json')!=signature:
        raise ValueError('Frozen scenarios or initial states changed')
    banks=o.read_json(out/'scenarios.json')
    identities=[previous.physical_hash(s) for bank in banks.values() for s in bank]
    if len(set(identities))!=len(identities) or history_scenes().intersection(identities):
        raise ValueError('Physical scenario leakage')
    for name,(_,n) in SPLITS.items():
        if len(banks[name])!=(1 if args.smoke else n): raise ValueError('Split count changed')
    for seed in SEEDS:
        if len(banks[f'train{seed}'])!=config(args)['episodes']:raise ValueError('Training count changed')
    o.write_json(out/'signature.json',signature)
    verify(args)
    return banks


def requests(args,models,scenes,modes=('search',),roles=(1,),opponents=o.OPPONENTS,timing='parallel'):
    result=[]
    for m in models:
        for spec in scenes:
            for mode in modes:
                for role in roles:
                    for opponent in opponents:
                        own=settings(args,m['config'])
                        if mode=='policy': own=settings(args,'base')
                        identity=dict(policy_id=m['policy_id'],episode_hash=spec['episode_hash'],
                            own_settings=own,opponent_settings=settings(args,'base'),mode=mode,role=role,
                            opponent=opponent,timing_context=timing)
                        key=o.json_hash(identity)
                        task=dict(**identity,key=key,models=m['models'],model_hash=m['model_hash'],
                                  baseline=str(args.output/'models/baseline'),spec=spec)
                        result.append(dict(key=key,label=m['label'],training_seed=m['seed'],
                            config=m['config'],model_hash=m['model_hash'],task=task))
    return result


def evaluate_one(task):
    torch.set_num_threads(1)
    if o.model_fingerprint(task['models'])!=task['model_hash']:
        raise ValueError('Evaluation weights changed after task creation')
    p,state=o.restore_scenario(task['spec']); role=task['role']
    own=make_solver(p,task['models'],'Policy Only' if task['mode']=='policy' else 'Search + Learning',
                    role,task['own_settings'],task['spec']['search_seeds'][role])
    other=make_solver(p,task['baseline'],task['opponent'],1-role,task['opponent_settings'],
                      task['spec']['search_seeds'][1-role])
    team=o.TeamSolver(own if role==0 else other,own if role==1 else other)
    metric=o.run_episode(p,team,state)
    illegal=metric['illegal_terminal']
    metric.update(legal_double_capture=int(metric['double_capture'] and not illegal),
        legal_survival=int(metric['capture_count']<2 and metric['survival_time']>=p.tf and not illegal),
        one_capture_timeout=int(metric['capture_count']==1 and metric['survival_time']>=p.tf and not illegal),
        pursuer_illegal=int(metric['pursuer_obstacle_collision'] or metric['pursuer_out_of_bounds']),
        evader_illegal=int(metric['evader_obstacle_collision'] or metric['evader_out_of_bounds']))
    metric['completion_cost']=metric['capture_time'] if metric['legal_double_capture'] else float(p.tf)
    metric.update(**{k:task[k] for k in ('key','policy_id','mode','role','opponent','timing_context')},
        episode_hash=task['spec']['episode_hash'],layout=task['spec']['layout'],
        search_seeds=task['spec']['search_seeds'],actual_own=own.actual_settings,actual_opponent=other.actual_settings)
    return metric


def run_evaluation(args,name,items,pool,status):
    folder=args.output/name; folder.mkdir(parents=True,exist_ok=True)
    manifest=[{k:v for k,v in item.items() if k!='task'} for item in items]
    if len({(m['label'],m['config'],m['key']) for m in manifest})!=len(manifest):
        raise ValueError('Duplicate evaluation requests')
    if (folder/'manifest.json').exists() and o.read_json(folder/'manifest.json')!=manifest:
        raise ValueError('Evaluation manifest changed')
    o.write_json(folder/'manifest.json',manifest)
    cache=args.output/'cache'; cache.mkdir(exist_ok=True)
    unique={item['key']:item['task'] for item in items}
    missing=[task for key,task in unique.items() if not (cache/(key+'.json')).exists()]
    counts={}
    for item in items: counts[item['key']]=counts.get(item['key'],0)+1
    done=len(items)-sum(counts[t['key']] for t in missing)
    status('evaluation',evaluation=name,completed=done,total=len(items),unique_tasks=len(unique))
    iterator=pool.imap_unordered(evaluate_one,missing,chunksize=1) if pool else map(evaluate_one,missing)
    for i,row in enumerate(iterator,1):
        row['content_sha256']=o.json_hash(row)
        o.write_json(cache/(row['key']+'.json'),row); done+=counts[row['key']]
        if i%10==0 or i==len(missing):
            status('evaluation',evaluation=name,completed=done,total=len(items),unique_tasks=len(unique))
    rows=[]
    for item in items:
        row=o.read_json(cache/(item['key']+'.json')); task=item['task']
        digest=row.pop('content_sha256',None)
        if digest!=o.json_hash(row):raise ValueError('Evaluation cache content changed')
        if any(row[k]!=task[k] for k in ('key','policy_id','mode','role','opponent')) or \
                row['episode_hash']!=task['spec']['episode_hash'] or row['search_seeds']!=task['spec']['search_seeds']:
            raise ValueError('Evaluation cache mismatch')
        row.update({k:v for k,v in item.items() if k not in ('task','key')})
        rows.append(row)
    groups={}
    for row in rows:
        group=tuple(row[k] for k in ('label','config','mode','role','layout','opponent'))
        groups.setdefault(group,[]).append(row)
    o.write_json(folder/'summary.json',[dict(zip(('label','config','mode','role','layout','opponent'),g),
                                             **o.summarize(rs)) for g,rs in groups.items()])
    return rows


def compare(rows,labels,references,cfg='base',ref_cfg=None,layout='four',mode='search',role=1):
    ref_cfg=cfg if ref_cfg is None else ref_cfg
    def index(label,configuration):
        result={}
        for r in rows:
            if (r['label'],r['config'],r['mode'],r['role'],r['layout'])!=(label,configuration,mode,role,layout):continue
            key=(r['layout'],r['episode_hash'],r['opponent'])
            if key in result:raise ValueError('Duplicate paired row')
            result[key]=r
        if not result:raise ValueError(f'Missing paired rows: {label}/{configuration}/{layout}')
        return result
    candidates=[index(label,cfg) for label in labels]
    refs=[index(label,ref_cfg) for label in references]
    keys=sorted(refs[0])
    if any(set(r)!=set(keys) for r in candidates+refs):raise ValueError('Unpaired cases')
    metric='legal_double_capture' if role==1 else 'legal_survival'
    side='pursuer_illegal' if role==1 else 'evader_illegal'
    for c,b in zip(candidates,refs):
        if any(c[k]['search_seeds']!=b[k]['search_seeds'] for k in keys):raise ValueError('Unpaired search seeds')
    differences=[[c[k][metric]-b[k][metric] for k in keys] for c,b in zip(candidates,refs)]
    result=previous.paired_statistics(keys,differences)
    result.update(own_illegal_delta=float(np.mean([[c[k][side]-b[k][side] for k in keys] for c,b in zip(candidates,refs)])),
                  metric=metric,layout=layout,mode=mode,role=role)
    return result


def improvement(item,minimum,confidence=False):
    return (item['delta']>=minimum-1e-12 and min(item['per_seed_delta'])>=-1e-12
            and item['own_illegal_delta']<=.02+1e-12 and (not confidence or item['ci95'][0]>0))


def regression(item):
    return min(item['per_seed_delta'])>=-.03-1e-12 and item['own_illegal_delta']<=.02+1e-12


def lock_json(path,value):
    if path.exists():locked_read(path)
    if path.exists() and o.read_json(path)!=json.loads(json.dumps(value)):
        raise ValueError('Locked selection changed')
    o.write_json(path,value)
    o.write_json(path.with_suffix('.lock.json'),dict(sha256=o.json_hash(value)))
    return value


def locked_read(path):
    value=o.read_json(path)
    if o.read_json(path.with_suffix('.lock.json'))['sha256']!=o.json_hash(value):
        raise ValueError('Locked selection digest changed')
    return value


def stage_report(args,name,title,text,details):
    target=args.output/'phase_reports'/(name+'.md')
    target.parent.mkdir(parents=True,exist_ok=True)
    relative=os.path.relpath(details,target.parent)
    target.write_text(f'# {title}\n\n{text}\n\n[阶段数据]({relative})\n',encoding='utf-8')
    if args.output==DEFAULT_OUTPUT.resolve() and not args.smoke:
        main=ROOT/'current/search_rl_diagnosis_20261022/REPORT.md'
        marker=f'<!-- l45-followup-{name} -->'
        if marker not in main.read_text():
            link=os.path.relpath(target,main.parent)
            with main.open('a') as stream:
                stream.write(f'\n{marker}\n\n### l45 后续闭环：{title}\n\n{text}\n\n[阶段报告]({link})\n')


def select_search(rows):
    labels=[f'old{seed}' for seed in SEEDS]; refs=['baseline']*3; decisions=[]
    for order,cfg in enumerate(CONFIGS):
        same=compare(rows,labels,refs,cfg=cfg)
        anchor=compare(rows,labels,refs,cfg=cfg,ref_cfg='base')
        same_none=compare(rows,labels,refs,cfg=cfg,layout='none')
        anchor_none=compare(rows,labels,refs,cfg=cfg,ref_cfg='base',layout='none')
        own=[r for r in rows if r['label'] in labels and r['config']==cfg and r['layout']=='four']
        illegal=float(np.mean([r['pursuer_illegal'] for r in own]))
        cost=float(np.mean([r['completion_cost'] for r in own]))
        decisions.append(dict(config=cfg,same=same,anchor=anchor,same_none=same_none,anchor_none=anchor_none,
            eligible=bool(improvement(same,.03) and improvement(anchor,.03) and
                          regression(same_none) and regression(anchor_none)),
            rank=[same['delta'],-illegal,-cost,-order]))
    eligible=[d for d in decisions if d['eligible']]
    chosen=max(eligible,key=lambda d:d['rank'])['config'] if eligible else None
    return dict(selected=chosen,decisions=decisions,selection_split='screen_only')


def search_screen(args,banks,pool,status):
    verify(args)
    items=[item for cfg in CONFIGS for item in requests(args,fixed_models(args,cfg),
                                              banks['screen_four']+banks['screen_none'])]
    expected=48*(len(banks['screen_four'])+len(banks['screen_none']))
    if len(items)!=expected:raise AssertionError('Search matrix incomplete')
    rows=run_evaluation(args,'search_screen',items,pool,status)
    status('selecting',evaluation='search_screen')
    result=lock_json(args.output/'search_selection.json',select_search(rows))
    text=(f"锁定搜索配置 {result['selected']}，将使用独立场景确认。" if result['selected'] is not None
          else '没有搜索配置同时通过配对筛选门槛，进入 A/B 奖励对照。')
    stage_report(args,'search_screen','搜索筛选完成',text,args.output/'search_selection.json')
    return result


def confirmation_result(rows,labels,cfg='base',also_anchor=False):
    refs=['baseline']*3
    same=compare(rows,labels,refs,cfg=cfg)
    same_none=compare(rows,labels,refs,cfg=cfg,layout='none')
    anchor=compare(rows,labels,refs,cfg=cfg,ref_cfg='base') if also_anchor else same
    anchor_none=compare(rows,labels,refs,cfg=cfg,ref_cfg='base',layout='none') if also_anchor else same_none
    return dict(same=same,anchor=anchor,same_none=same_none,anchor_none=anchor_none,
        pursuit_pass=bool(improvement(same,.05,True) and improvement(anchor,.05,True)),
        regression_pass=bool(regression(same_none) and regression(anchor_none)),
        policy_four=compare(rows,labels,refs,cfg=cfg,mode='policy'),
        policy_none=compare(rows,labels,refs,cfg=cfg,mode='policy',layout='none'))


def search_confirm(args,banks,pool,status):
    selection=locked_read(args.output/'search_selection.json'); cfg=selection['selected']
    if cfg is None:raise ValueError('No locked search candidate')
    models=fixed_models(args,cfg)
    items=requests(args,models,banks['search_four']+banks['search_none'],modes=('search','policy'))
    if cfg!='base':
        items+=requests(args,[model('baseline',args.output/'models/baseline')],banks['search_four']+banks['search_none'])
    rows=run_evaluation(args,'search_confirmation',items,pool,status)
    result=confirmation_result(rows,[f'old{seed}' for seed in SEEDS],cfg,True)
    o.write_json(args.output/'search_confirmation/result.json',result)
    stage_report(args,'search_confirmation','搜索独立确认完成',
        f"搜索提升门槛通过：{result['pursuit_pass']}；无障碍回归通过：{result['regression_pass']}。",args.output/'search_confirmation/result.json')
    return result,models,[f'old{seed}' for seed in SEEDS],cfg


def shape_rewards(episode,bonus):
    """Two captures terminate the original episode; bonus belongs to that last transition."""
    for row in episode['rows']:
        row['original_reward']=float(row.get('original_reward',row['reward']))
        row['completion_bonus']=0.
        row['reward']=row['original_reward']
    if episode['stats']['legal_double_capture']:
        if not episode['rows'] or not episode['rows'][-1]['done']:
            raise ValueError('Legal double capture must terminate the original episode')
        episode['rows'][-1]['completion_bonus']=float(bonus)
        episode['rows'][-1]['reward']+=float(bonus)
    episode['stats']['environment_return']=sum(r['original_reward'] for r in episode['rows'])
    episode['stats']['training_return']=sum(r['reward'] for r in episode['rows'])
    return episode


def reward_rollout(task):
    return shape_rewards(o.rollout(task),task['completion_bonus'])


def checkpoint_score(rows):
    four=[r for r in rows if r['layout']=='four' and r['mode']=='search']
    return tuple(o.selection_score(four))


def checkpoint_eligible(rows,baseline_rows):
    own=o.summarize([r for r in rows if r['layout']=='none' and r['mode']=='search'])
    base=o.summarize([r for r in baseline_rows if r['layout']=='none' and r['mode']=='search'])
    return own['success_rate']>=base['success_rate']-.03-1e-12


def commit_checkpoint(directory,payload,selection):
    seq=payload.get('commit',0)+1
    payload['commit']=seq; payload['selection']=copy.deepcopy(selection)
    target=directory/f'checkpoints/commit{seq:04d}_ep{payload["episode"]:06d}.pt'
    o.save_torch(target,payload)
    o.write_json(directory/'latest.json',dict(path=str(target),sha256=o.sha(target),
                 episode=payload['episode'],pending_validation=payload['pending_validation'],commit=seq))
    return payload


def load_checkpoint(args,directory):
    pointer=o.read_json(directory/'latest.json')
    path=Path(pointer['path'])
    if not path.is_relative_to(directory/'checkpoints') or o.sha(path)!=pointer['sha256']:
        raise ValueError('Checkpoint pointer/digest mismatch')
    payload=o.load_torch(path)
    if payload['signature']!=o.json_hash(o.read_json(args.output/'signature.json')) or \
        payload['experiment_sha']!=o.sha(args.output/'experiment.json') or payload['episode']!=pointer['episode']:
        raise ValueError('Training resume fingerprint/counter mismatch')
    return payload


def export_checkpoint(args,directory,episode,states):
    destination=directory/f'models_ep{episode}'
    if not destination.exists():
        o.export_models(states,destination,args.output/'models/baseline')
    for robot,name in enumerate(o.MODEL_NAMES[:4]):
        exported=o.load_torch(destination/name)
        if set(exported)!=set(states[robot]) or any(not torch.equal(v,states[robot][k]) for k,v in exported.items()):
            raise ValueError('Checkpoint export changed')
    baseline=o.model_fingerprint(args.output/'models/baseline')
    exported=o.model_fingerprint(destination)
    if any(exported[n]!=baseline[n] for n in (*o.MODEL_NAMES[:2],o.MODEL_NAMES[-1])):
        raise ValueError('Frozen evader/value changed in training export')
    return destination


def train_one(args,banks,arm,seed,baseline_rows,pool,status,remaining):
    verify(args); c=config(args)
    directory=args.output/f'reward/{arm}/seed{seed}'; directory.mkdir(parents=True,exist_ok=True)
    initial=o.load_torch(args.output/f'initial/seed{seed}.pt')
    p=o.original_problem()
    actors=o.load_actors(p,states=initial['actors']); critic=o.TrainingCritic()
    critic.load_state_dict(initial['critic'])
    for robot in o.TEAMS[0]:
        for parameter in actors[robot].parameters():parameter.requires_grad_(False)
    actor_opt=torch.optim.Adam([v for robot in o.TEAMS[1] for v in actors[robot].parameters()],lr=c['lr'])
    critic_opt=torch.optim.Adam(critic.parameters(),lr=c['critic_lr'])
    if (directory/'latest.json').exists():
        saved=load_checkpoint(args,directory)
        if saved['arm']!=arm or saved['seed']!=seed or saved['initial_sha']!=o.sha(args.output/f'initial/seed{seed}.pt'):
            raise ValueError('Training arm/seed/shared initialization changed')
        for actor,state in zip(actors,saved['actors']):actor.load_state_dict(state,strict=True)
        critic.load_state_dict(saved['critic'])
        actor_opt.load_state_dict(saved['actor_optimizer']); critic_opt.load_state_dict(saved['critic_optimizer'])
        o.restore_rng(saved['rng']); selection=saved['selection']
    else:
        o.restore_rng(initial['rng'])
        selection=dict(best_episode=0,models=str(args.output/'models/baseline'),score=list(checkpoint_score(baseline_rows)))
        saved=o.checkpoint_payload(actors,critic,actor_opt,critic_opt,0,0,selection['score'],False,
                                  o.json_hash(o.read_json(args.output/'signature.json')))
        saved.update(arm=arm,seed=seed,initial_sha=o.sha(args.output/f'initial/seed{seed}.pt'),
                     experiment_sha=o.sha(args.output/'experiment.json'))
        saved=commit_checkpoint(directory,saved,selection)
    while saved['episode']<c['episodes'] or saved['pending_validation']:
        verify(args)
        completed=saved['episode']
        if saved['pending_validation']:
            path=export_checkpoint(args,directory,completed,o.actor_states(actors))
            descriptor=model(f'{arm}{seed}',path,seed)
            rows=run_evaluation(args,f'reward/{arm}/seed{seed}/validation_ep{completed}',
                requests(args,[descriptor],banks['validation_four']+banks['validation_none']),pool,status)
            score=checkpoint_score(rows)
            if checkpoint_eligible(rows,baseline_rows) and score>tuple(selection['score']):
                selection=dict(best_episode=completed,models=str(path),score=list(score))
            saved.update(pending_validation=False,best_episode=selection['best_episode'],best_score=selection['score'],rng=o.rng_state())
            saved=commit_checkpoint(directory,saved,selection)
            o.write_json(directory/'selection.json',dict(**selection,current_episode=completed,selection_split='validation_only'))
        if completed==c['episodes']:break
        if remaining[0]==0:
            status('stopped_at_checkpoint',arm=arm,seed=seed,episode=completed)
            return False
        snapshot=directory/f'sampling/ep{completed:06d}.pt'
        o.save_torch(snapshot,dict(actors=o.actor_states(actors),episode=completed))
        tasks=[dict(spec=banks[f'train{seed}'][index],snapshot=str(snapshot),snapshot_episode=completed,
            baseline=str(args.output/'models/baseline'),sampling_seed=o.scenario_seed(690,seed,index),
            opponent=o.OPPONENTS[index%2],settings=settings(args),completion_bonus=c['completion_bonus'][arm])
            for index in range(completed,completed+c['batch'])]
        status('reward_training',arm=arm,seed=seed,episode=completed,total=c['episodes'])
        started=time.monotonic()
        episodes=list(pool.imap(reward_rollout,tasks,chunksize=1)) if pool else list(map(reward_rollout,tasks))
        sampling_seconds=time.monotonic()-started
        if any(e['snapshot_episode']!=completed for e in episodes):raise ValueError('Stale on-policy reward batch')
        metric=o.update_ppo(actors,critic,actor_opt,critic_opt,episodes,c)
        metric.update(arm=arm,seed=seed,episode=completed+c['batch'],sampling_seconds=sampling_seconds,
            mean_training_return=float(np.mean([e['stats']['training_return'] for e in episodes])),
            double_capture_episodes=sum(e['stats']['legal_double_capture'] for e in episodes))
        completed+=c['batch']
        o.write_json(directory/f'updates/{completed:06d}.json',metric)
        o.write_json(directory/f'rollouts/{completed:06d}.json',[dict(spec=e['spec'],opponent=e['opponent'],stats=e['stats']) for e in episodes])
        pending=completed%c['eval_every']==0 or completed==c['episodes']
        updated=o.checkpoint_payload(actors,critic,actor_opt,critic_opt,completed,selection['best_episode'],
                                    selection['score'],pending,saved['signature'],metric)
        updated.update(arm=arm,seed=seed,initial_sha=saved['initial_sha'],experiment_sha=saved['experiment_sha'],commit=saved['commit'])
        saved=commit_checkpoint(directory,updated,selection)
        o.write_json(directory/'progress.json',dict(arm=arm,seed=seed,episode=completed,total=c['episodes'],
                     best_episode=selection['best_episode']))
        if remaining[0]>0:remaining[0]-=1
    export_checkpoint(args,directory,c['episodes'],saved['actors'])
    o.write_json(directory/'selection.json',dict(**selection,current_episode=c['episodes'],selection_split='validation_only'))
    stage_report(args,f'reward_{arm}_seed{seed}',f'{arm} 组 seed{seed} 训练完成',
        f"已训练 {c['episodes']} 局，搜索验证选用第 {selection['best_episode']} 局；训练完成不代表已确认提升。",directory/'selection.json')
    return True


def reward_models(args,kind='selected'):
    result=[model('baseline',args.output/'models/baseline')]
    for arm in ('A','B'):
        for seed in SEEDS:
            directory=args.output/f'reward/{arm}/seed{seed}'
            if kind=='selected':
                selected=o.read_json(directory/'selection.json'); path=Path(selected['models']); label=f'{arm}{seed}'
                committed=load_checkpoint(args,directory)
                if any(selected[k]!=v for k,v in committed['selection'].items()) or selected['current_episode']!=committed['episode']:
                    raise ValueError('Training selection differs from committed checkpoint')
            else:
                path=directory/f'models_ep{config(args)["episodes"]}'; label=f'{arm}{seed}_final'
            result.append(model(label,path,seed))
    return result


def select_reward(args,banks,pool,status):
    models=reward_models(args)
    rows=run_evaluation(args,'reward_selection_validation',
        requests(args,models,banks['validation_four']+banks['validation_none']),pool,status)
    choices=[]
    for order,arm in enumerate(('A','B')):
        labels=[f'{arm}{seed}' for seed in SEEDS]
        four=compare(rows,labels,['baseline']*3)
        none=compare(rows,labels,['baseline']*3,layout='none')
        choices.append(dict(arm=arm,four=four,none=none,
                            eligible=bool(improvement(four,.03) and regression(none)),
                            rank=[four['delta'],-order]))
    eligible=[item for item in choices if item['eligible']]
    chosen=max(eligible,key=lambda item:item['rank'])['arm'] if eligible else None
    frozen=[dict(arm=arm,seed=seed,**o.read_json(args.output/f'reward/{arm}/seed{seed}/selection.json'))
            for arm in ('A','B') for seed in SEEDS]
    result=lock_json(args.output/'reward_selection.json',dict(selected=chosen,choices=choices,
                     checkpoints=frozen,models=models,selection_split='validation_only'))
    stage_report(args,'reward_selection','奖励对照验证完成',
        f'锁定组别：{chosen}。未入选时结束本轮，不自动延长训练。',args.output/'reward_selection.json')
    return result


def reward_train(args,banks,pool,status):
    baseline=model('baseline',args.output/'models/baseline')
    baseline_rows=run_evaluation(args,'reward_baseline_validation',
        requests(args,[baseline],banks['validation_four']+banks['validation_none']),pool,status)
    remaining=[args.max_updates if args.max_updates else -1]
    for arm in ('A','B'):
        for seed in SEEDS:
            if not train_one(args,banks,arm,seed,baseline_rows,pool,status,remaining):return None
    return select_reward(args,banks,pool,status)


def reward_confirm(args,banks,pool,status):
    locked=locked_read(args.output/'reward_selection.json'); arm=locked['selected']
    if arm is None:raise ValueError('No locked reward candidate')
    models=reward_models(args)
    if models!=locked['models']:raise ValueError('Reward model selection changed after locking')
    items=requests(args,models,banks['reward_four']+banks['reward_none'],modes=('search','policy'))
    final=reward_models(args,'final')[1:]
    items+=requests(args,final,banks['reward_four']+banks['reward_none'],modes=('policy',))
    rows=run_evaluation(args,'reward_confirmation',items,pool,status)
    result=confirmation_result(rows,[f'{arm}{seed}' for seed in SEEDS])
    result['B_vs_A_search']=compare(rows,[f'B{seed}' for seed in SEEDS],[f'A{seed}' for seed in SEEDS])
    result['completion_bonus_supported']=bool(result['B_vs_A_search']['delta']>0 and result['B_vs_A_search']['ci95'][0]>0)
    o.write_json(args.output/'reward_confirmation/result.json',result)
    stage_report(args,'reward_confirmation','奖励模型独立确认完成',
        f"搜索提升门槛通过：{result['pursuit_pass']}；无障碍回归通过：{result['regression_pass']}。",args.output/'reward_confirmation/result.json')
    candidate=[m for m in models if m['label']=='baseline' or m['label'].startswith(arm)]
    return result,candidate,[f'{arm}{seed}' for seed in SEEDS],'base'


def audit(args,banks):
    verify(args)
    baseline=args.output/'models/baseline'
    spec=copy.deepcopy(banks['screen_four'][0])
    p,state=o.restore_scenario(spec); state[p.time_idx,0]=p.tf-p.dt
    spec['initial_state']=state.tolist()
    spec['episode_hash']=o.json_hash({k:v for k,v in spec.items() if k!='episode_hash'})
    descriptor=model('baseline',baseline)
    task=requests(args,[descriptor],[spec],opponents=('Search Only',))[0]['task']
    current=evaluate_one(task)
    legacy=o.evaluation_tasks(baseline,'baseline',None,baseline,[spec],settings(args),
        modes=('search',),opponents=('Search Only',))[0]
    old=o.evaluate_one(legacy)
    fields={k:old[k] for k in old if k in current and 'latency' not in k and
            k not in ('key','label','training_seed','model_hash','timing_context')}
    if o.json_hash(fields)!=o.json_hash({k:current[k] for k in fields}):
        raise ValueError('Original-configuration parity audit failed')
    effective={}
    for cfg in CONFIGS:
        solver=make_solver(p,baseline,'Search + Learning',1,settings(args,cfg),7)
        effective[cfg]=solver.actual_settings
    neutral=settings(args); neutral['beta_policy']=0.
    first=make_solver(p,baseline,'Search + Learning',1,neutral,91).policy(p,state)
    second=make_solver(p,args.output/'models/seed0_final','Search + Learning',1,neutral,91).policy(p,state)
    if not np.array_equal(first,second):raise ValueError('Zero-policy-prior negative control failed')
    result=dict(original_configuration_exact_match=True,zero_policy_prior_exact_match=True,
                effective_parameters=effective,opponent_settings=settings(args),
                source_verified=True,no_original_code_modified=True)
    o.write_json(args.output/'audit.json',result)
    return result


def guards_and_latency(args,banks,pool,status,models,labels,cfg,origin,pursuit):
    locked=dict(origin=origin,models=models,labels=labels,config=cfg,pursuit=pursuit)
    lock_json(args.output/'deployment_candidate.json',locked)
    items=requests(args,models,banks['guard_four']+banks['guard_none'],roles=(0,))
    if cfg!='base':
        items+=requests(args,[model('baseline',args.output/'models/baseline')],
                        banks['guard_four']+banks['guard_none'],roles=(0,))
    rows=run_evaluation(args,'evader_regression',items,pool,status)
    result={}
    for layout in ('four','none'):
        result[layout]=compare(rows,labels,['baseline']*3,cfg=cfg,layout=layout,role=0)
        result[layout+'_anchor']=compare(rows,labels,['baseline']*3,cfg=cfg,ref_cfg='base',layout=layout,role=0)
    passed=all(regression(r) for r in result.values())
    o.write_json(args.output/'evader_regression/result.json',dict(passed=passed,comparisons=result))
    stage_report(args,'evader_regression','逃逸搜索回归完成',f'全部种子的逃逸回归通过：{passed}。',args.output/'evader_regression/result.json')
    if not passed:return finish(args,'evader_regression_failed',origin,pursuit,False)
    timing_models=models[:]
    if cfg!='base':timing_models.append(model('baseline',args.output/'models/baseline'))
    run_evaluation(args,'serial_latency',
        requests(args,timing_models,banks['latency_four'],modes=('search','policy'),
                 opponents=('Policy Only',),timing='single_process'),None,status)
    return finish(args,'quality_confirmed',origin,pursuit,True)


def finish(args,reason,origin=None,pursuit=None,passed=False):
    verify(args)
    if passed and (not (args.output/'evader_regression/result.json').exists() or
                   not o.read_json(args.output/'evader_regression/result.json')['passed'] or
                   not (args.output/'serial_latency/summary.json').exists()):
        raise ValueError('Full gate requires evader regression and serial timing')
    result=dict(reason=reason,origin=origin,smoke=args.smoke,
        quality_gate_passed=bool(passed and not args.smoke),pursuit=pursuit,
        original_sources_verified=True,continue_formal=False,model_promoted=False)
    for name in ('search_selection','reward_selection'):
        if (args.output/(name+'.json')).exists():result[name]=locked_read(args.output/(name+'.json'))
    for folder in ('search_confirmation','reward_confirmation','evader_regression'):
        if (args.output/folder/'result.json').exists():result[folder]=o.read_json(args.output/folder/'result.json')
    o.write_json(args.output/'results.json',result)
    lines=['# l45 PPO 搜索诊断与奖励对照', '',
        f"功能 smoke：{args.smoke}；结束原因：{reason}；完整搜索门槛通过：{result['quality_gate_passed']}。",
        '原 l45、原环境与原 C++ 搜索均保留；没有启动 5 万局，也没有替换模型。',
        '所有区间以同一个原始 l45 为条件，三个 PPO 种子不代表三个独立预训练模型。','',
        '## 搜索筛选','',
        '| 配置 | 相对同配置 l45 | 相对原配置 l45 | 入选资格 |',
        '|---|---:|---:|---|']
    selection=result.get('search_selection')
    if selection:
        for d in selection['decisions']:
            lines.append(f"| {d['config']} | {100*d['same']['delta']:+.2f} 个百分点 | {100*d['anchor']['delta']:+.2f} 个百分点 | {d['eligible']} |")
        lines.append(f"\n锁定配置：{selection['selected']}。筛选不属于正式提升结论。")
    else:lines.append('本次未执行搜索筛选。')
    if 'reward_selection' in result:
        sel=result['reward_selection']
        lines+=['','## 奖励对照','',f"验证集锁定组别：{sel['selected']}。A 为原奖励，B 为合法双捕终止时增加一次 1.0。"]
        for item in sel['choices']:
            lines.append(f"- {item['arm']}：验证搜索双捕率平均变化 {100*item['four']['delta']:+.2f} 个百分点；入选资格 {item['eligible']}。")
        for item in sel['checkpoints']:
            lines.append(f"- {item['arm']} / seed{item['seed']}：选用第 {item['best_episode']} 局；最终训练到 {item['current_episode']} 局。")
    if pursuit:
        lines+=['','## 独立确认','']
        for name,title in (('same','搜索相对同配置 l45'),('anchor','搜索相对原配置 l45'),
                           ('policy_four','四障碍纯策略'),('policy_none','无障碍纯策略')):
            item=pursuit[name]; ci=item['ci95']
            lines.append(f"- {title}：{100*item['delta']:+.2f} 个百分点，95% 区间 [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]；各种子变化 {item['per_seed_delta']}。")
        if 'B_vs_A_search' in pursuit:
            item=pursuit['B_vs_A_search']
            lines.append(f"- B 相对同种子 A 搜索：{100*item['delta']:+.2f} 个百分点，95% 区间 {item['ci95']}；额外完成奖励的收益得到支持：{pursuit['completion_bonus_supported']}。")
    lines+=['','## 边界与后续','',
        '- 主指标是合法双捕；首捕后超时、双方非法、完成用时及原基准指标见各阶段 summary.json。',
        '- 纯策略和搜索分别报告；纯策略退化时，不宣称纯策略可以替代原模型。',
        '- 并行耗时只表示实验吞吐；仅 serial_latency 为单进程计时，本轮不自动减少搜索预算。',
        '- 代码或指纹检查失败必须修复后重新核验；质量证据不足时保留失败结论，不追加训练量。',
        '- 历史模型和结果不覆盖；下一步扩大训练或修改局部观测须另行制定方案。','']
    (args.output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    stage_report(args,'complete','本轮完成',
        f"结束原因：{reason}；完整搜索质量门槛通过：{result['quality_gate_passed']}。原 l45 保留，不启动 5 万局。",args.output/'REPORT.md')
    return result


def pipeline(args,banks,pool,status):
    selected=search_screen(args,banks,pool,status)
    if selected['selected'] is not None:
        pursuit,models,labels,cfg=search_confirm(args,banks,pool,status)
        if not pursuit['regression_pass']:
            return finish(args,'search_no_obstacle_regression_failed','search',pursuit)
        if pursuit['pursuit_pass']:
            return guards_and_latency(args,banks,pool,status,models,labels,cfg,'search',pursuit)
    selected=reward_train(args,banks,pool,status)
    if selected is None:return None
    if selected['selected'] is None:return finish(args,'no_reward_validation_candidate','reward')
    pursuit,models,labels,cfg=reward_confirm(args,banks,pool,status)
    if not pursuit['regression_pass']:return finish(args,'reward_no_obstacle_regression_failed','reward',pursuit)
    if not pursuit['pursuit_pass']:return finish(args,'reward_search_gain_not_confirmed','reward',pursuit)
    return guards_and_latency(args,banks,pool,status,models,labels,cfg,'reward',pursuit)


def live_pids(output):
    found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            if entry.stat().st_uid!=os.getuid():continue
            state=(entry/'stat').read_text().rsplit(')',1)[1].split()[0]
            if state in ('Z','X'):continue
            argv=(entry/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if not Path(argv[0]).name.startswith('python'):continue
            cwd=(entry/'cwd').resolve()
            scripts=[i for i,arg in enumerate(argv) if arg.endswith('run_l45_ppo_followup.py') and
                     (Path(arg) if Path(arg).is_absolute() else cwd/arg).resolve()==Path(__file__).resolve()]
            if not scripts or '-c' in argv[:scripts[0]] or '-m' in argv[:scripts[0]]:continue
            if '--stage=watch' in argv or ('--stage' in argv and argv[argv.index('--stage')+1]=='watch'):continue
            target=DEFAULT_OUTPUT
            if '--output' in argv:target=Path(argv[argv.index('--output')+1])
            for arg in argv:
                if arg.startswith('--output='):target=Path(arg.split('=',1)[1])
            if (target if target.is_absolute() else cwd/target).resolve()==output.resolve():found.append(int(entry.name))
        except (OSError,ValueError,IndexError):pass
    return sorted(found)


def dashboard(args):
    out=args.output; state=o.read_json(out/'status.json') if (out/'status.json').exists() else {}
    pids=live_pids(out); done=state.get('phase')=='complete' and (out/'results.json').exists() and (out/'REPORT.md').exists()
    lines=['l45 PPO 后续闭环（不自动启动 5 万局）',
           f"状态：{'已完成' if done else ('运行中 PID '+str(pids) if pids else '未运行／保存进度')}",
           f"阶段：{state.get('phase','未启动')}；任务：{state.get('evaluation','--')}",
           f"配置 worker：{state.get('workers','--')}；输出：{out}"]
    if pids:
        try:children=Path(f'/proc/{pids[0]}/task/{pids[0]}/children').read_text().split()
        except OSError:children=[]
        actual=0
        for pid in children:
            try:actual+=b'spawn_main' in Path(f'/proc/{pid}/cmdline').read_bytes()
            except OSError:pass
        lines.append(f'实际 spawn worker：{actual}')
    for path in sorted(out.glob('**/manifest.json')):
        if path.name!='manifest.json':continue
        try:
            manifest=o.read_json(path)
            if not isinstance(manifest,list):continue
            actual=sum((out/'cache'/(m['key']+'.json')).exists() for m in manifest)
            lines.append(f'{path.parent.relative_to(out)}：{actual}/{len(manifest)}')
        except (ValueError,KeyError):pass
    for arm in ('A','B'):
        for seed in SEEDS:
            path=out/f'reward/{arm}/seed{seed}/progress.json'
            if path.exists():
                value=o.read_json(path); lines.append(f"{arm} seed{seed}：{value['episode']}/{value['total']}，验证选优 ep{value['best_episode']}")
    rate=state.get('rate',0)
    if pids and rate>0 and state.get('total',0)>state.get('completed',0):
        minutes=(state['total']-state['completed'])/rate/60
        lines.append(f'当前阶段吞吐 {rate:.2f} 局/秒，预计剩余 {minutes:.1f} 分钟（不含后续阶段）')
    lines.append('进度不代表胜率；关闭看板不会停止任务。')
    return '\n'.join(lines)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--workers',type=int,default=80)
    p.add_argument('--stage',choices=('pipeline','prepare','audit','search-screen','search-confirm',
        'reward-train','reward-confirm','guard','report','watch'),default='pipeline')
    p.add_argument('--max-updates',type=int,default=0,help='Checkpoint stop for controlled resume; zero means no early stop')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--once',action='store_true',help='Print dashboard once')
    args=p.parse_args(argv); args.output=args.output.resolve()
    if args.workers<1 or args.max_updates<0:p.error('Invalid workers/update limit')
    if args.smoke and args.output==DEFAULT_OUTPUT.resolve():p.error('Smoke requires isolated --output')
    for source in (SOURCE,FINAL_SOURCE,o.SOURCE_MODELS):
        source=source.resolve()
        if args.output==source or args.output.is_relative_to(source) or source.is_relative_to(args.output):
            p.error('Output overlaps a protected historical source')
    return args


def main(argv=None):
    args=parse_args(argv); torch.set_num_threads(1)
    if args.stage=='watch':
        try:
            while True:
                print(('' if args.once else '\033[2J\033[H')+dashboard(args),flush=True)
                if args.once:return
                time.sleep(3)
        except KeyboardInterrupt:return
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'run.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as error:raise RuntimeError('Another runner owns this output') from error
    clocks={}; latest_status={}; status_lock=threading.RLock(); heartbeat_stop=threading.Event()
    def status(phase,**fields):
        if phase=='evaluation':
            name=fields['evaluation']; now=time.monotonic()
            start,initial=clocks.setdefault(name,(now,fields['completed']))
            fields.update(elapsed_seconds=now-start,rate=(fields['completed']-initial)/max(now-start,1e-9))
        payload=dict(phase=phase,pid=os.getpid(),workers=args.workers,updated_unix=time.time(),**fields)
        with status_lock:
            latest_status.clear(); latest_status.update(payload)
            o.write_json(args.output/'status.json',payload)
        print(json.dumps(payload,ensure_ascii=False),flush=True)
    def heartbeat():
        while not heartbeat_stop.wait(5):
            with status_lock:
                if latest_status:
                    latest_status['heartbeat_unix']=time.time()
                    o.write_json(args.output/'status.json',latest_status)
    heartbeat_thread=threading.Thread(target=heartbeat,daemon=True)
    heartbeat_thread.start()
    try:
        status('checking'); banks=prepare(args)
        if args.stage=='prepare':status('prepared');return
        status('audit'); audit(args,banks)
        if args.stage=='audit':status('prepared');return
        if args.stage=='report':
            old=o.read_json(args.output/'results.json')
            result=finish(args,old['reason'],old['origin'],old['pursuit'],old['quality_gate_passed'])
            status('complete',reason=result['reason'],quality_gate_passed=result['quality_gate_passed'],continue_formal=False)
            return
        with mp.get_context('spawn').Pool(args.workers) as pool:
            result=None
            if args.stage=='pipeline':result=pipeline(args,banks,pool,status)
            elif args.stage=='search-screen':search_screen(args,banks,pool,status)
            elif args.stage=='search-confirm':
                pursuit,models,labels,cfg=search_confirm(args,banks,pool,status)
                if pursuit['pursuit_pass'] and pursuit['regression_pass']:
                    lock_json(args.output/'deployment_candidate.json',dict(origin='search',models=models,labels=labels,config=cfg,pursuit=pursuit))
            elif args.stage=='reward-train':reward_train(args,banks,pool,status)
            elif args.stage=='reward-confirm':
                pursuit,models,labels,cfg=reward_confirm(args,banks,pool,status)
                if pursuit['pursuit_pass'] and pursuit['regression_pass']:
                    lock_json(args.output/'deployment_candidate.json',dict(origin='reward',models=models,labels=labels,config=cfg,pursuit=pursuit))
            elif args.stage=='guard':
                value=locked_read(args.output/'deployment_candidate.json')
                result=guards_and_latency(args,banks,pool,status,value['models'],value['labels'],value['config'],
                                           value['origin'],value['pursuit'])
        if result is not None:status('complete',reason=result['reason'],quality_gate_passed=result['quality_gate_passed'],continue_formal=False)
        elif o.read_json(args.output/'status.json')['phase']!='stopped_at_checkpoint':status('prepared')
    except KeyboardInterrupt:
        status('interrupted');raise
    except Exception as error:
        status('failed',error=repr(error));raise
    finally:
        heartbeat_stop.set(); heartbeat_thread.join(timeout=1); lock.close()


if __name__=='__main__':
    main()
