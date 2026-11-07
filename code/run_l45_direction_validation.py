"""Frozen B/S/F/G validation on 150 new physical scenes and three opponents."""
import os
for _var in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_var]='1'
import argparse
import contextlib
import copy
import fcntl
import io
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import threading
import time
import numpy as np
import torch
import run_l45_candidate_ablation as parent
from solvers.candidate_ablation import CandidateAblationPUCT

torch.set_num_threads(1)
old,o=parent.old,parent.o
ROOT,CODE=o.ROOT,o.CODE
SOURCE=parent.DEFAULT_OUTPUT
DEFAULT_OUTPUT=ROOT/'current/l45_direction_validation_20261024'
SMOKE_OUTPUT=ROOT/'current/l45_direction_validation_smoke_20261024'
UIDS=('B','S','F','G')
OPPONENTS=tuple(o.OPPONENTS)
lock_json,locked_read=parent.lock_json,parent.locked_read
commit_row,read_cached,read_episode=parent.commit_row,parent.read_cached,parent.read_episode
ORIGINAL_MAP=[box.tolist() for box in o.original_problem('four').obstacles]
MAPS=dict(original_four=ORIGINAL_MAP,
    offset_corridor=[[[-4.5,-2.5],[-5.,-3.]], [[-4.5,-2.5],[1.,3.]],
                     [[2.5,4.5],[-3.,-1.]], [[2.5,4.5],[3.,5.]]],
    asymmetric=[[[-5.,-3.],[-4.,-2.]], [[-1.,1.],[-.5,1.5]],
                [[3.,5.],[-3.,-1.]], [[-3.,-1.],[3.,5.]]],none=[])
GROUPS=dict(original_four='original_four',offset_corridor='unseen_four',asymmetric='unseen_four',none='none')

def config(args):
    return dict(schema=1,smoke=args.smoke,training_episodes=0,uids=list(UIDS),opponents=list(OPPONENTS),
        source_models='O=PPO_seed0_episode2048; L=frozen_original_opponents',
        budget=3 if args.smoke else 1000,opponent_search_budget=3 if args.smoke else 1000,
        search_only_budget=3 if args.smoke else 3000,depth=100,pursuer_betas=[.5,0.],opponent_betas=[.5,.1],
        scenes_per_map={m:1 if args.smoke else (25 if GROUPS[m]=='unseen_four' else 50) for m in MAPS},
        maps=MAPS,map_groups=GROUPS,namespace=5802026102401 if args.smoke else 5802026102402,
        games=48 if args.smoke else 1800,root_mixture_FG=[.5,.25,.25],noise_fraction=.1,
        activation='next_real_decision_after_first_capture; one_on_one_at_root',
        comparisons=['F-B','F-S','G-F','G-B','G-S'],primary=['F-B','F-S'],primary_maps=['original_four','unseen_four'],
        bootstrap_seed=58020261024,bootstrap_draws=20000,minimum_gain=.03,regression_margin=.03,illegal_margin=.02,
        G_noninferiority_margin=.03,adoption=False,recommended='old0:novalue',no_automatic_expansion=True)

def settings(args,role=1):
    c=config(args)
    return dict(search_budget=c['budget'] if role else c['opponent_search_budget'],
        search_only_budget=c['search_only_budget'],search_depth=100,beta_policy=.5,beta_value=0. if role else .1)

def fingerprints():
    files=[Path(__file__),CODE/'L45_DIRECTION_VALIDATION.md',CODE/'tests/test_l45_direction_validation.py']
    return dict(files={str(p):o.sha(p) for p in files},frozen_parent=parent.fingerprints())

def registry(args):
    return {u:dict(models=str(args.output/'models'/u),model_hash=o.model_fingerprint(args.output/'models'/u)) for u in ('O','L')}

def problem(map_id):
    p=o.original_problem('none');p.obstacles=[np.array(box,float) for box in MAPS[map_id]]
    p.evader_spawn_near_obstacles=False;p.use_minco_dynamics=False;p.clear_closed_loop_cache()
    return p

def make_scenario(seed,map_id):
    previous=np.random.get_state()
    try:
        np.random.seed(seed);p=problem(map_id);state=p.initialize()
        spec=dict(seed=int(seed),layout=GROUPS[map_id],map_id=map_id,obstacles=MAPS[map_id],
            initial_state=state.tolist(),action_lims=p.action_lims.tolist(),control_lims=p.control_lims.tolist(),
            evader_speed=float(p.current_evader_speed_lim),pursuer_speed=float(p.current_pursuer_speed_lim),
            search_seeds=[o.scenario_seed(19,seed,r) for r in range(2)])
        spec['episode_hash']=o.json_hash(spec)
        return spec
    finally:np.random.set_state(previous)

def restore_scenario(spec):
    if o.json_hash({k:v for k,v in spec.items() if k!='episode_hash'})!=spec['episode_hash']:
        raise ValueError('Scenario identity changed')
    if spec['obstacles']!=MAPS[spec['map_id']] or spec['layout']!=GROUPS[spec['map_id']]:
        raise ValueError('Scenario map differs from frozen definition')
    p=problem(spec['map_id']);p.action_lims=np.array(spec['action_lims'],float);p.control_lims=np.array(spec['control_lims'],float)
    p.current_evader_speed_lim=spec['evader_speed'];p.current_pursuer_speed_lim=spec['pursuer_speed']
    state=np.array(spec['initial_state'],float)
    if state.shape!=(53,1) or p.is_terminal(state):raise ValueError('Invalid initial state')
    return p,state.copy()

def physical_hash(spec):
    # Ignore map and memory: reject shared physical positions/capabilities even on a different map.
    state=np.asarray(spec['initial_state'],float).reshape(-1)
    return o.json_hash(dict(position=state[:8].tolist(),action_lims=spec['action_lims']))

def extract_specs(value):
    if isinstance(value,dict):
        if 'initial_state' in value and 'action_lims' in value:
            yield value;return
        for x in value.values():yield from extract_specs(x)
    elif isinstance(value,list):
        for x in value:yield from extract_specs(x)

def history(args):
    paths=set(old.s.history_files(args))
    for folder in (ROOT/'current').glob('*'):
        if not folder.is_dir() or folder.resolve()==args.output:continue
        for pattern in ('**/scenarios.json','**/cases.json','**/all_cases.json','**/diagnostic_bank.json','**/bank.json'):
            paths.update(folder.glob(pattern))
    paths={p for p in paths if p.is_file() and not p.is_relative_to(args.output)}
    used=set();map_signatures=set();count=0
    for path in sorted(paths):
        data=o.read_json(path)
        for spec in extract_specs(data):
            used.add(physical_hash(spec));count+=1
            engineering_map_probe=any('l45_direction_validation_smoke' in part for part in path.parts)
            if 'obstacles' in spec and not engineering_map_probe:map_signatures.add(o.json_hash(spec['obstacles']))
            elif spec.get('layout') in ('four','none','two'):
                if spec['layout']=='four':map_signatures.add(o.json_hash(ORIGINAL_MAP))
                elif spec['layout']=='none':map_signatures.add(o.json_hash([]))
    return sorted(paths),used,map_signatures,count

def prepare(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if out==SOURCE or SOURCE in out.parents:raise ValueError('Use independent output')
    if (out/'experiment.lock.json').exists():
        record=locked_read(out/'experiment.json')
        if record['config']!=config(args) or record['implementation']!=fingerprints() or record['models']!=registry(args):
            raise ValueError('Frozen implementation/configuration changed')
        for p,sha in record['sources'].items():
            if o.sha(p)!=sha:raise ValueError('Frozen source changed: '+p)
        return locked_read(out/'scenarios.json')
    parent.prepare(parent.parse_args([]))
    origins=dict(O=SOURCE/'models/O',L=old.SOURCE/'models/L')
    sources={}
    for u,path in origins.items():
        dest=out/'models'/u;dest.mkdir(parents=True,exist_ok=True)
        for name in o.MODEL_NAMES:
            shutil.copyfile(path/name,dest/name);sources[str(path/name)]=o.sha(path/name)
    reg=registry(args)
    for name in (o.MODEL_NAMES[0],o.MODEL_NAMES[1],o.MODEL_NAMES[-1]):
        if reg['O']['model_hash'][name]!=reg['L']['model_hash'][name]:raise ValueError('Frozen evader actors/value differ')
    paths,used,old_maps,occurrences=history(args);before=len(used)
    for map_id in ('offset_corridor','asymmetric'):
        if o.json_hash(MAPS[map_id]) in old_maps:raise ValueError('Requested unseen map already used')
    banks={};namespace=config(args)['namespace']
    for phase,(map_id,n) in enumerate(config(args)['scenes_per_map'].items(),6000):
        bank=[];i=0
        while len(bank)<n:
            spec=make_scenario(o.scenario_seed(phase,namespace,i),map_id);i+=1
            key=physical_hash(spec)
            if key not in used:used.add(key);bank.append(spec)
        banks[map_id]=bank
    lock_json(out/'scenarios.json',banks);sources[str(out/'scenarios.json')]=o.sha(out/'scenarios.json')
    lock_json(out/'scenario_independence.json',dict(passed=True,previous_physical_states=before,
        previous_spec_occurrences=occurrences,new_physical_states=sum(map(len,banks.values())),physical_overlap=0,
        ignore_map_for_physical_dedup=True,unseen_maps_checked=True,history_files={str(p):o.sha(p) for p in paths},
        scope='all recoverable scenario/case/bank JSON plus l45 rollout specs; archives and unused banks included'))
    sources[str(SOURCE/'experiment.json')]=o.sha(SOURCE/'experiment.json')
    sources[str(ROOT/'current/l45_amplitude_match_20261024/results.json')]=o.sha(ROOT/'current/l45_amplitude_match_20261024/results.json')
    lock_json(out/'experiment.json',dict(config=config(args),implementation=fingerprints(),models=reg,sources=sources))
    shutil.copyfile(CODE/'L45_DIRECTION_VALIDATION.md',out/'PLAN.md')
    runtime=out/'runtime';runtime.mkdir(exist_ok=True)
    binary=Path(parent.cpp.__file__);shutil.copyfile(binary,runtime/binary.name)
    shutil.copytree(CODE/'cpp/build_candidate_ablation/source',runtime/'candidate_source',dirs_exist_ok=True)
    shutil.copyfile(CODE/'cpp/build_candidate_ablation/build_manifest.json',runtime/'candidate_build_manifest.json')
    shutil.copyfile(Path(old.cpp.__file__),runtime/Path(old.cpp.__file__).name)
    return banks

def effective(uid,reg):
    return dict(model='O',weights=reg['O']['model_hash'],algorithm='legacy' if uid=='B' else 'sync',
        root_proposal=uid if uid in ('F','G') else 'original',activation='real_post_capture',rollout='original',
        opponents='original_L_and_original_configuration')

def make_own(p,reg,uid,requested,seed):
    if uid=='B':return old.make_own(p,reg,'B',requested,seed)
    return parent.make_own(p,reg,uid,requested,seed)

def make_pair(task):
    p,state=restore_scenario(task['spec']);p.clear_closed_loop_cache()
    own=make_own(p,task['registry'],task['uid'],task['own_settings'],task['spec']['search_seeds'][1])
    other=old.previous.base.private_solver(p,task['registry']['L'],task['opponent'],0,
        task['opponent_settings'],task['spec']['search_seeds'][0])
    return p,state,own,other

def requests(args,banks):
    reg=registry(args);items=[]
    for scenes in banks.values():
        for spec in scenes:
            for opponent in OPPONENTS:
                for uid in UIDS:
                    behavior=effective(uid,reg)
                    identity=dict(protocol=o.sha(args.output/'experiment.json'),policy_id=o.json_hash(behavior),
                        episode_hash=spec['episode_hash'],role=1,opponent=opponent,timing_context='parallel',
                        own_settings=settings(args),opponent_settings=settings(args,0),mode='search')
                    key=o.json_hash(identity)
                    task=dict(**identity,key=key,uid=uid,spec=spec,registry=reg,effective=behavior)
                    items.append(dict(key=key,uid=uid,task=task))
    return items

def evaluate_one(task):
    old.previous.seed_task(task);p,state,own,other=make_pair(task);began=time.perf_counter()
    row,d,decisions,elapsed=old.simulate(p,state,own,other)
    final=d['final']['state'];violations={}
    for team,robots in (('pursuer',p.pursuers),('evader',p.evaders)):
        violations[team+'_obstacle_collision']=int(any(p.check_obstacle_collision(final,r) for r in robots))
        violations[team+'_out_of_bounds']=int(any(np.any(final[p.state_idxs[r],0]<p.state_lims[p.state_idxs[r],0]) or
            np.any(final[p.state_idxs[r],0]>p.state_lims[p.state_idxs[r],1]) for r in robots))
    row.update({k:task[k] for k in ('key','policy_id','role','opponent','timing_context','mode')})
    row.update(uid=task['uid'],episode_hash=task['spec']['episode_hash'],layout=task['spec']['layout'],map_id=task['spec']['map_id'],
        search_seeds=task['spec']['search_seeds'],actual_own=dict(task['own_settings'],effective=task['effective']),
        actual_opponent=other.actual_settings,own_cost=own.solver.cost,own_mcts_calls=own.solver.calls,
        episode_wall_ms=(time.perf_counter()-began)*1000,deployment_episode_ms=elapsed,decision_latency_ms=decisions,
        trajectory_sha256=old.previous.trajectory_digest(d['trajectory']),**violations)
    row['_record']=dict(key=task['key'],scenario=task['spec'],effective=task['effective'],role=1,opponent=task['opponent'],**d)
    return row

def check(args,banks,status):
    if (args.output/'checks.lock.json').exists():return locked_read(args.output/'checks.json')
    probes=[];reg=registry(args);rng_roundtrips=0
    for map_id,bank in banks.items():
        spec=bank[0];p,state=restore_scenario(spec)
        memory=state.copy();p._update_observation_memory(memory,memory,0.)
        if not np.array_equal(memory,state):raise ValueError('Initial memory does not match actual map')
        for opponent in OPPONENTS:
            task=next(i['task'] for i in requests(args,{map_id:[spec]}) if i['uid']=='S' and i['task']['opponent']==opponent)
            p,state,own,other=make_pair(task)
            beta=[0.,0.] if opponent=='Search Only' else [.5,.1]
            if opponent!='Policy Only':
                actual=other.actual_settings
                wanted=settings(args,0)['search_only_budget' if opponent=='Search Only' else 'search_budget']
                if actual['budget']!=wanted or actual['depth']!=100 or [actual['beta_policy'],actual['beta_value']]!=beta:
                    raise ValueError('Opponent did not receive frozen settings')
            for raw in (own.solver,other.solver) if opponent!='Policy Only' else (own.solver,):
                wrapper=raw._problem_wrapper_for(p)
                for robot in range(4):
                    np.testing.assert_allclose(np.array(wrapper.policy_encoding(state,robot)).reshape(-1),p.policy_encoding(state,robot).reshape(-1),atol=3e-5,rtol=0)
            initial=own.solver.get_rng_state(p);own.solver.set_rng_state(p,initial)
            if initial!=own.solver.get_rng_state(p):raise ValueError('RNG roundtrip changed')
            rng_roundtrips+=1
            reference=own.policy(p,state);refroot=own.solver.last_result;ref_rng=own.solver.get_rng_state(p)
            for uid in ('F','G'):
                q,s,otherown,other2=make_pair(dict(task,uid=uid,effective=effective(uid,reg)))
                actual=otherown.policy(q,s)
                if not np.array_equal(actual,reference) or not np.array_equal(otherown.solver.last_result.root_audit,refroot.root_audit) or otherown.solver.get_rng_state(q)!=ref_rng:
                    raise ValueError('F/G changed pre-capture search')
            # Original versus frozen private B action and joint Python/C++ step.
            before=state.copy();before[[p.active_idxs[0],p.active_idxs[2]],0]=0.;before[p.time_idx,0]=77.
            for uid in ('B','S','F','G'):
                trial=make_own(p,reg,uid,settings(args),spec['search_seeds'][1]);wr=trial.solver._problem_wrapper_for(p)
                if uid=='B':reference_solver=old.make_own(p,reg,'B',settings(args),spec['search_seeds'][1])
                else:reference_solver=parent.make_own(p,reg,uid,settings(args),spec['search_seeds'][1])
                a=trial.policy(p,before);b=reference_solver.policy(p,before)
                if not np.array_equal(a,b) or trial.solver.get_rng_state(p)!=reference_solver.solver.get_rng_state(p):raise ValueError('Frozen method changed')
                # Check the exact environment propagation used by all four groups.
                cppstate=np.asarray(wr.step(before,a,p.dt)) if hasattr(wr,'step') else None
                if cppstate is not None:np.testing.assert_allclose(cppstate.reshape(-1),p.step(before,a,p.dt).reshape(-1),atol=3e-5,rtol=0)
            probes.append(dict(map_id=map_id,opponent=opponent,passed=True));status('check',completed=len(probes),total=12)
    return lock_json(args.output/'checks.json',dict(passed=True,probes=probes,precapture_FG_exact=True,
        RNG_roundtrip_checks=rng_roundtrips,explicit_map_restore=True,frozen_weights=True,
        no_new_training=True))

def run(args,items,status):
    if len(items)!=config(args)['games']:raise ValueError('Unexpected evaluation budget')
    lock_json(args.output/'pilot/manifest.json',[dict(key=i['key'],uid=i['uid'],scene=i['task']['spec']['episode_hash'],opponent=i['task']['opponent']) for i in items])
    missing=[]
    for item in items:
        if (args.output/'cache_checksums'/(item['key']+'.lock.json')).exists():read_cached(args,item)
        else:missing.append(item['task'])
    done=len(items)-len(missing);initial=done;started=time.perf_counter()
    status('pilot',completed=done,total=len(items),resumed=done)
    if missing:
        with mp.get_context('spawn').Pool(min(args.workers,len(missing))) as pool:
            o.write_json(args.output/'worker_snapshot.json',dict(workers=len(pool._pool),pids=[p.pid for p in pool._pool],compute_threads_per_worker=1))
            for row in pool.imap_unordered(evaluate_one,missing,chunksize=1):
                commit_row(args,row);done+=1
                if done%5==0 or done==len(items):status('pilot',completed=done,total=len(items),throughput_per_second=(done-initial)/max(.001,time.perf_counter()-started))
    return [read_cached(args,i) for i in items]

def pair(rows):
    index={};scene_meta={}
    for r in rows:
        key=(r['uid'],r['episode_hash'],r['opponent'])
        if key in index:raise ValueError('Duplicate scene/model/opponent')
        index[key]=r;scene_meta[r['episode_hash']]=(r['layout'],r['map_id'])
    scenes=sorted(scene_meta)
    if not scenes or len(index)!=len(scenes)*12:raise ValueError('Incomplete scene/model/opponent pairing')
    for s in scenes:
        for opponent in OPPONENTS:
            if any((u,s,opponent) not in index for u in UIDS):raise ValueError('Missing opponent/model pairing')
            seeds=[index[(u,s,opponent)]['search_seeds'] for u in UIDS]
            if any(x!=seeds[0] for x in seeds):raise ValueError('Unpaired solver seeds')
    return index,scenes,scene_meta

def comparison(index,scenes,opponents,draws,a,b):
    raw=np.array([[index[(a,s,opp)]['legal_double_capture']-index[(b,s,opp)]['legal_double_capture'] for opp in opponents] for s in scenes],float)
    x=raw.mean(1);mean=float(x.mean());sample=x[draws].mean(1)
    cost=np.array([np.mean([index[(a,s,opp)]['completion_cost']-index[(b,s,opp)]['completion_cost'] for opp in opponents]) for s in scenes])
    result=dict(delta=mean,ci95=np.quantile(sample,[.025,.975]).tolist(),lower95=float(np.quantile(sample,.05)),
        gained=int((raw>0).sum()),lost=int((raw<0).sum()),scenes=len(scenes),opponents=len(opponents),
        completion_cost_delta=float(cost.mean()),completion_cost_ci95=np.quantile(cost[draws].mean(1),[.025,.975]).tolist())
    for metric in ('step_calls','policy_evaluations'):
        c=np.array([np.mean([index[(a,s,opp)]['own_cost'][metric]-index[(b,s,opp)]['own_cost'][metric] for opp in opponents]) for s in scenes])
        result[metric+'_delta']=float(c.mean());result[metric+'_ci95']=np.quantile(c[draws].mean(1),[.025,.975]).tolist()
    for side in ('pursuer_illegal','evader_illegal'):
        result[side+'_delta']=float(np.mean([index[(a,s,opp)][side]-index[(b,s,opp)][side] for s in scenes for opp in opponents]))
    return result,sample

def analyze(rows,smoke=False):
    index,scenes,meta=pair(rows);rng=np.random.default_rng(58020261024);sections={}
    scopes=dict(all=scenes,four=[s for s in scenes if meta[s][0]!='none'])
    scopes.update({group:[s for s in scenes if meta[s][0]==group] for group in ('original_four','unseen_four','none')})
    scopes.update({m:[s for s in scenes if meta[s][1]==m] for m in ('offset_corridor','asymmetric')})
    primary_draws=None;primary_means=None
    for scope,selected in scopes.items():
        if not selected:continue
        strata={group:[i for i,s in enumerate(selected) if meta[s][0]==group] for group in ('original_four','unseen_four','none')}
        draws=np.concatenate([rng.choice(ii,size=(20000,len(ii)),replace=True) for ii in strata.values() if ii],axis=1)
        groups={}
        for opp_label,opponents in [('mean',OPPONENTS)]+[(x,(x,)) for x in OPPONENTS]:
            summary={};comparisons={};samples=[];means=[]
            for u in UIDS:
                part=[index[(u,s,op)] for s in selected for op in opponents]
                rates=np.array([np.mean([index[(u,s,op)]['legal_double_capture'] for op in opponents]) for s in selected])
                summary[u]=dict(games=len(part),physical_scenes=len(selected),successes=sum(r['legal_double_capture'] for r in part),
                    success_rate=float(rates.mean()),ci95=np.quantile(rates[draws].mean(1),[.025,.975]).tolist(),
                    own_illegal=sum(r['pursuer_illegal'] for r in part),opponent_illegal=sum(r['evader_illegal'] for r in part),
                    zero_capture=sum(r['capture_count']==0 for r in part),one_capture_timeout=sum(r['one_capture_timeout'] for r in part),
                    completion_cost=float(np.mean([r['completion_cost'] for r in part])),
                    first_capture_mean=float(np.mean([r['first_capture_time'] for r in part if np.isfinite(r['first_capture_time'])])) if any(np.isfinite(r['first_capture_time']) for r in part) else None,
                    own_cost={k:sum(r['own_cost'][k] for r in part) for k in ('step_calls','policy_evaluations','value_evaluations')},
                    decisions=sum(r['own_mcts_calls'] for r in part),
                    obstacle_collisions=sum(r.get('pursuer_obstacle_collision',0) for r in part),out_of_bounds=sum(r.get('pursuer_out_of_bounds',0) for r in part))
            for name in ('F-B','F-S','G-F','G-B','G-S'):
                a,b=name.split('-');c,sample=comparison(index,selected,opponents,draws,a,b);comparisons[name]=c
                if name in ('F-B','F-S'):samples.append(sample);means.append(c['delta'])
            if scope=='four' and opp_label=='mean':
                primary_draws=np.stack(samples,1);primary_means=np.array(means)
            groups[opp_label]=dict(summary=summary,comparisons=comparisons)
        sections[scope]=groups
    if primary_draws is None:raise ValueError('No four-obstacle scenes')
    critical=float(np.quantile((primary_draws-primary_means).max(1),.95));checks={}
    for ref in ('B','S'):
        c=sections['four']['mean']['comparisons']['F-'+ref];c['simultaneous_lower95']=c['delta']-critical
        checks[ref]=dict(minimum_gain=c['delta']>=.03-1e-12,simultaneous_positive=c['simultaneous_lower95']>0,
            SO_improved=sections['four']['Search Only']['comparisons']['F-'+ref]['delta']>0,
            Policy_Only_regression=sections['four']['Policy Only']['comparisons']['F-'+ref]['delta']>=-.03-1e-12,
            SL_regression=sections['four']['Search + Learning']['comparisons']['F-'+ref]['delta']>=-.03-1e-12,
            none_regression=sections['none']['mean']['comparisons']['F-'+ref]['delta']>=-.03-1e-12,
            illegal_per_map=all(sections[g]['mean']['comparisons']['F-'+ref]['pursuer_illegal_delta']<=.02+1e-12 for g in ('original_four','unseen_four','none')))
    gf=sections['four']['mean']['comparisons']['G-F']
    cheaper=all(gf[k+'_delta']<=0 for k in ('step_calls','policy_evaluations')) and any(gf[k+'_ci95'][1]<0 for k in ('step_calls','policy_evaluations'))
    decision=dict(F_passed=not smoke and all(all(v.values()) for v in checks.values()),F_checks=checks,
        G_noninferior_to_F=not smoke and gf['lower95']>-.03,G_noninferiority_margin=.03,G_lower_call_counts=bool(cheaper) and not smoke,
        unseen_F_vs_S_positive=not smoke and sections['unseen_four']['mean']['comparisons']['F-S']['ci95'][0]>0,
        recommended='old0:novalue',no_adoption=True)
    return dict(sections=sections,decision=decision,bootstrap_seed=58020261024,draws=20000,
        physical_scenes=len(scenes),paired_games=len(rows),stratified_by='original_four/unseen_four/none',smoke=smoke)

def inspect_records(args,items):
    prefixes={};roots=0;budget=config(args)['budget'];records=[]
    total={u:dict(geometry_queries=0.,control_proposals=0.,proposal_fallbacks=0.,noise_fallbacks=0.,selected_added=0,
        post_calls=0,post_step_calls=0,post_policy_evaluations=0,post_costs=[],success_second_times=[],
        graph_build_ms=0.,plan_ms=0.,latencies=[],post_latencies=[]) for u in UIDS}
    for item in items:
        row,d=read_episode(args,item);u=item['uid'];spec=item['task']['spec'];key=(row['episode_hash'],row['opponent'])
        if not np.array_equal(d['initial']['state'],np.array(spec['initial_state'])):raise ValueError('Actual initial state changed')
        prefixes.setdefault(key,{})[u]=d
        g=total[u];first=d['first_capture']
        if first and first['branch_available']:
            g['post_costs'].append(row['completion_cost']-first['time'])
            if row['legal_double_capture']:g['success_second_times'].append(row['capture_time']-first['time'])
        for r,latency in zip(d['roots'],row['decision_latency_ms']):
            roots+=1
            if r['visits']!=budget or r['table'][:,1].sum()!=budget:raise ValueError('Saved simulation budget differs')
            if r['cost']['value_evaluations'] or r['guidance'].get('guided_steps',0):raise ValueError('Unexpected value or guided rollout')
            g['latencies'].append(latency[0])
            if r['phase']:
                g['post_calls']+=1;g['post_step_calls']+=r['cost']['step_calls'];g['post_policy_evaluations']+=r['cost']['policy_evaluations'];g['post_latencies'].append(latency[0])
            elif r['guidance'].get('active',0):raise ValueError('Guidance activated before first capture')
            for field in ('geometry_queries','control_proposals','graph_build_ms','plan_ms'):g[field]+=r['guidance'].get(field,0)
            g['proposal_fallbacks']+=r['guidance'].get('proposal_fallback',0);g['noise_fallbacks']+=r['guidance'].get('noise_fallback',0)
            g['selected_added']+=int(r['guidance'].get('selected_source')==2)
        records.append(dict(uid=u,scene=row['episode_hash'],map_id=row['map_id'],layout=row['layout'],opponent=row['opponent'],
            speed_ratio=spec['pursuer_speed']/spec['evader_speed'],success=row['legal_double_capture'],completion_cost=row['completion_cost'],
            first_capture_time=row['first_capture_time'],one_capture_timeout=row['one_capture_timeout'],own_illegal=row['pursuer_illegal'],
            record=row['episode_record']['path']))
    for key,group in prefixes.items():
        s=group['S'];n=s['first_capture']['next_decision'] if s['first_capture'] else len(s['trajectory'])
        for u in ('F','G'):
            if not old.s.equal_tree(s['trajectory'][:n],group[u]['trajectory'][:n]) or not old.s.equal_tree(s['first_capture'],group[u]['first_capture']):
                raise ValueError('S/F/G actual first capture prefix or RNG differs')
    for g in total.values():
        g['takeover_games']=len(g['post_costs']);g['mean_post_cost']=float(np.mean(g.pop('post_costs'))) if g['post_costs'] else None
        g['successful_second_capture_mean']=float(np.mean(g.pop('success_second_times'))) if g['success_second_times'] else None
        for field in ('latencies','post_latencies'):
            values=g.pop(field);g[field+'_mean']=float(np.mean(values)) if values else None;g[field+'_p95']=float(np.quantile(values,.95)) if values else None
    integrity=dict(passed=True,games=len(items),root_searches=roots,simulation_budget=budget,first_capture_prefixes_exact=len(prefixes),
        no_value_or_guided_rollout=True,weights_frozen=True,recommended='old0:novalue')
    return total,records,integrity

def report(args,items,rows):
    result=analyze(rows,args.smoke);diagnostics,records,integrity=inspect_records(args,items)
    lock_json(args.output/'statistics.json',result);lock_json(args.output/'diagnostics.json',diagnostics)
    lock_json(args.output/'trajectory_index.json',records);lock_json(args.output/'integrity.json',integrity)
    lines=['F/G全新场景独立验证' if not args.smoke else '隔离smoke，仅工程检查','',
        f'冻结B/S/F/G，{result["physical_scenes"]}个全新物理场景，三个原对手，共{len(rows)}局。无训练，当前推荐保持old0:novalue。','',
        'B=O旧搜索，S=O同步搜索，F=满幅神经均值方向，G=绕障碍追踪；F/G仅在真实首捕后启用。',
        '三个真实对手固定原L模型；SL为1000模拟/.5/.1，SO为3000模拟/0/0。全部追捕1000模拟/.5/0。smoke预算均为3。','',
        '|地图组|对手|B合法双捕|S合法双捕|F合法双捕|G合法双捕|',
        '|---|---|---:|---:|---:|---:|']
    for scope in ('original_four','unseen_four','none','four','all'):
        for opponent in ('mean',)+OPPONENTS:
            g=result['sections'][scope][opponent]['summary'];cells=[f'{g[u]["successes"]}/{g[u]["games"]}={100*g[u]["success_rate"]:.2f}%' for u in UIDS]
            lines.append('|'+scope+'|'+opponent+'|'+'|'.join(cells)+'|')
    lines+=['','配对区间按物理场景在地图组内分层bootstrap，共同保留四组及三个对手；20000次，seed58020261024。','',
        '|地图组|比较|差值 pp|普通95%区间 pp|挽回/丢失|完成成本差 步|',
        '|---|---|---:|---|---:|---:|']
    for scope in ('original_four','unseen_four','none','four','all','offset_corridor','asymmetric'):
        for name,g in result['sections'][scope]['mean']['comparisons'].items():
            lo,hi=np.array(g['ci95'])*100;lines.append(f'|{scope}|{name}|{g["delta"]*100:+.2f}|[{lo:+.2f},{hi:+.2f}]|{g["gained"]}/{g["lost"]}|{g["completion_cost_delta"]:+.2f}|')
    decision=result['decision'];lines+=['','预登记判断：',str(decision),'',
        '失败与成本（所有地图、对手）：','',
        '|组|零捕|首捕后超时|自身非法|对手非法|完成成本|决策数|环境转移|网络调用|几何查询|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for u in UIDS:
        g=result['sections']['all']['mean']['summary'][u];d=diagnostics[u];c=g['own_cost']
        lines.append(f'|{u}|{g["zero_capture"]}|{g["one_capture_timeout"]}|{g["own_illegal"]}|{g["opponent_illegal"]}|{g["completion_cost"]:.2f}|{g["decisions"]}|{c["step_calls"]}|{c["policy_evaluations"]}|{d["geometry_queries"]:.0f}|')
    lines+=['','完整每地图每对手的成功率区间、配对成本和调用差区间见statistics.json；逐场景完整轨迹见trajectory_index.json。',
        'G仍是神经MCTS，F/G都保留冻结的共同几何预处理。调用量可配对比较，并发延迟不能证明独立部署加速。',
        'G非劣需四障碍G-F单侧95%下界>-3pp；差异不显著不等同于效果相当。',
        '本轮不替换部署推荐、不自动追加样本或启动其他实验。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    lock_json(args.output/'results.json',dict(reason='smoke_complete' if args.smoke else 'validation_complete',games=len(rows),
        physical_scenes=result['physical_scenes'],training_episodes=0,recommended='old0:novalue',decision=decision))
    plots(args,result)
    return result

def plots(args,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=args.output/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4));groups=('original_four','unseen_four','none');x=np.arange(3)
    for i,u in enumerate(UIDS):
        y=[result['sections'][s]['mean']['summary'][u]['success_rate']*100 for s in groups]
        axes[0].bar(x+(i-1.5)*.18,y,.18,label=u)
    axes[0].set(xticks=x,xticklabels=['Original four','Unseen four','No obstacle'],ylim=(0,105),ylabel='Legal double capture (%)');axes[0].legend()
    for i,name in enumerate(('F-B','F-S','G-F')):
        y=[];lo=[];hi=[]
        for scope in groups:
            c=result['sections'][scope]['mean']['comparisons'][name];v=c['delta']*100;y.append(v);lo.append(v-100*c['ci95'][0]);hi.append(100*c['ci95'][1]-v)
        axes[1].errorbar(x+(i-1)*.1,y,yerr=[lo,hi],fmt='o',capsize=3,label=name)
    axes[1].axhline(0,color='gray',ls='--');axes[1].set(xticks=x,xticklabels=['Original','Unseen','None'],ylabel='Paired change (pp)');axes[1].legend()
    axes[2].bar(UIDS,[result['sections']['all']['mean']['summary'][u]['completion_cost'] for u in UIDS]);axes[2].set(ylabel='Completion cost (failure=80)',title='All physical scenes / opponents')
    fig.tight_layout();fig.savefig(out/'validation.png',dpi=190);plt.close(fig)

def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',nargs='?',default='pipeline',choices=['pipeline','prepare','check','pilot','report','watch'])
    parser.add_argument('--workers',type=int,default=80);parser.add_argument('--output',type=Path)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--once',action='store_true')
    args=parser.parse_args(argv);args.output=(args.output or (SMOKE_OUTPUT if args.smoke else DEFAULT_OUTPUT)).resolve()
    if args.workers<1:parser.error('workers must be positive')
    return args

def main(argv=None):
    args=parse_args(argv)
    if args.stage=='watch':return old.previous.watch(args)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(pid=os.getpid(),phase='prepare',workers=args.workers,started=time.time());mutex=threading.Lock();stop=threading.Event()
        def status(phase,**fields):
            with mutex:
                state.update(phase=phase,updated_unix=time.time(),**fields)
                children=[p.pid for p in mp.active_children() if p.is_alive()]
                state.update(actual_workers=len(children),worker_pids=children,committed_cache=len(list((args.output/'cache_checksums').glob('*.lock.json'))))
                o.write_json(args.output/'status.json',state)
        def heartbeat():
            while not stop.wait(5):status(state['phase'])
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:
            banks=prepare(args);status('prepared',physical_scenes=sum(map(len,banks.values())))
            if args.stage=='prepare':return
            check(args,banks,status)
            if args.stage=='check':return
            items=requests(args,banks)
            rows=run(args,items,status) if args.stage in ('pipeline','pilot') else [read_cached(args,i) for i in items]
            status('report');report(args,items,rows)
            status('complete',reason='smoke_complete' if args.smoke else 'validation_complete',recommended='old0:novalue',elapsed_seconds=time.time()-state['started'])
        except BaseException as e:status('stopped',reason=f'{type(e).__name__}: {e}');raise
        finally:stop.set();thread.join(timeout=1)

if __name__=='__main__':main()
