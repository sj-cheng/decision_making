"""Fresh-state, serial fixed/time-budget study of geometry, neural proposals and search."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import contextlib
import copy
import fcntl
import io
import multiprocessing as mp
from pathlib import Path
import platform
import shutil
import threading
import time
import types
import numpy as np
import torch
import run_l45_direction_validation as prior
from solvers.search_contribution import ContributionPUCT,cpp

torch.set_num_threads(1)
o,old=prior.o,prior.old
ROOT,CODE=o.ROOT,o.CODE
SOURCE=prior.DEFAULT_OUTPUT
DEFAULT_OUTPUT=ROOT/'current/l45_search_contribution_20261024'
SMOKE_OUTPUT=ROOT/'current/l45_search_contribution_smoke_20261024'
UIDS=('S','F','G','G-no-neural','Geometry-only')
SEARCH_UIDS=UIDS[:-1]
VARIANT={'S':'S','F':'F','G':'G','G-no-neural':'N','Geometry-only':'Geometry'}
COMPARISONS=('G-F','G-no-neural-G','Geometry-only-G-no-neural','Geometry-only-G','G-S')
PAIRS=(('G','F'),('G-no-neural','G'),('Geometry-only','G-no-neural'),('Geometry-only','G'),('G','S'))
lock_json,locked_read=prior.lock_json,prior.locked_read
commit_row,read_cached,read_episode=prior.commit_row,prior.read_cached,prior.read_episode

def config(args):
    return dict(schema=1,smoke=args.smoke,training_episodes=0,uids=list(UIDS),source=str(SOURCE),
        scenarios_per_map={m:1 if args.smoke else (5 if prior.GROUPS[m]=='unseen_four' else 10) for m in prior.MAPS},
        namespace=5902026102401 if args.smoke else 5902026102402,
        fixed_simulations=3 if args.smoke else 1000,deadline_ms=2. if args.smoke else 50.,
        timed_simulation_cap=100000,opponent_search_budget=3 if args.smoke else 1000,
        opponent_SO_budget=3 if args.smoke else 3000,depth=100,primary_scope='shared_S_real_first_capture_suffix',
        neural_ablation='replace_all_pursuer_neural_proposals_by_uniform; retain_simulated_evader_network',
        geometry_only='unperturbed_G_base; no_suffix_search_or_network; shared_S_prefix',
        root_timing_repeats=3,bootstrap_seed=59020261024,bootstrap_draws=20000,
        comparisons=list(COMPARISONS),noninferiority_margin=.03,
        prefix_cap=12 if args.smoke else 90,new_suffix_cap=108 if args.smoke else 810,
        geometry_timed_reuse=True,serial_fixed=True,serial_timed=True,adoption=False,recommended='old0:novalue')

def fingerprints():
    files=[Path(__file__),CODE/'build_search_contribution.py',CODE/'solvers/search_contribution.py',
        CODE/'L45_SEARCH_CONTRIBUTION.md',CODE/'tests/test_l45_search_contribution.py',
        CODE/'cpp/build_search_contribution/build_manifest.json',Path(cpp.__file__)]
    return dict(files={str(p):o.sha(p) for p in files},parent=prior.fingerprints())

def registry(args):
    return {u:dict(models=str(args.output/'models'/u),model_hash=o.model_fingerprint(args.output/'models'/u)) for u in ('O','L')}

def settings(args,uid='S',phase='fixed'):
    c=config(args)
    return dict(search_budget=0 if uid=='Geometry-only' else c['timed_simulation_cap'] if phase=='timed' else c['fixed_simulations'],
        search_only_budget=c['opponent_SO_budget'],search_depth=100,beta_policy=.5,beta_value=0.,
        deadline_ms=c['deadline_ms'] if phase=='timed' and uid!='Geometry-only' else 0.)

def opponent_settings(args):
    c=config(args)
    return dict(search_budget=c['opponent_search_budget'],search_only_budget=c['opponent_SO_budget'],search_depth=100,beta_policy=.5,beta_value=.1)

def effective(uid,reg,phase):
    return dict(uid=uid,weights=reg['O']['model_hash'],phase=phase,scope='post_capture_only',
        pursuer_neural=uid not in ('G-no-neural','Geometry-only'),simulated_evader_neural=uid!='Geometry-only',
        tree_search=uid!='Geometry-only',shared_prefix='O_synchronous',opponents='original_L')

def make_own(p,reg,uid,requested,seed):
    policies,value=[None]*4,None
    if uid!='Geometry-only':
        paths,vpath=old.previous.research.h.oracle_paths(str(ROOT),iteration=45,model_dir=reg['O']['models'])
        with contextlib.redirect_stdout(io.StringIO()):
            policies,value=old.previous.research.h.get_oracles(p,value_oracle_name='deterministic',value_oracle_path=vpath,
                policy_oracle_name='gaussian',policy_oracle_paths=paths)
    raw=ContributionPUCT(policies,value,requested['search_budget'],VARIANT[uid],requested['deadline_ms'])
    raw.solver_wrapper.set_seed(int(seed))
    return o.RoleMaskedSolver(raw,o.TEAMS[1])

def prepare(args):
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    if out==SOURCE or SOURCE in out.parents:raise ValueError('Use independent output')
    if (out/'experiment.lock.json').exists():
        record=locked_read(out/'experiment.json')
        if record['config']!=config(args) or record['implementation']!=fingerprints() or record['models']!=registry(args):
            raise ValueError('Frozen protocol/model/implementation changed')
        for p,sha in record['sources'].items():
            if o.sha(p)!=sha:raise ValueError('Frozen source changed: '+p)
        return locked_read(out/'scenarios.json')
    prior.prepare(prior.parse_args([]))
    sources={}
    for u in ('O','L'):
        dest=out/'models'/u;dest.mkdir(parents=True,exist_ok=True)
        for name in o.MODEL_NAMES:
            origin=SOURCE/'models'/u/name;shutil.copyfile(origin,dest/name);sources[str(origin)]=o.sha(origin)
    paths,used,_,count=prior.history(args);before=len(used);banks={}
    for phase,(map_id,n) in enumerate(config(args)['scenarios_per_map'].items(),7000):
        cases=[];i=0
        while len(cases)<n:
            spec=prior.make_scenario(o.scenario_seed(phase,config(args)['namespace'],i),map_id);i+=1
            key=prior.physical_hash(spec)
            if key not in used:used.add(key);cases.append(spec)
        banks[map_id]=cases
    warm=prior.make_scenario(o.scenario_seed(7100,config(args)['namespace'],0),'none')
    p,state=prior.restore_scenario(warm);state[[p.active_idxs[0],p.active_idxs[2]],0]=0.
    state[p.state_idxs[1],0]=[-5.,0.];state[p.state_idxs[3],0]=[5.,0.];state[p.time_idx,0]=30.
    for idx in p.memory_idxs:state[idx,0]=0.
    p._update_observation_memory(state,state,0.)
    warm['initial_state']=state.tolist();warm['episode_hash']=o.json_hash({k:v for k,v in warm.items() if k!='episode_hash'})
    if prior.physical_hash(warm) in used:raise ValueError('Engineering state overlaps history')
    lock_json(out/'scenarios.json',banks);lock_json(out/'engineering_scenario.json',warm)
    for n in ('scenarios.json','engineering_scenario.json'):sources[str(out/n)]=o.sha(out/n)
    for n in ('experiment.json','results.json','statistics.json'):sources[str(SOURCE/n)]=o.sha(SOURCE/n)
    lock_json(out/'scenario_independence.json',dict(passed=True,previous_states=before,historical_occurrences=count,
        new_states=sum(map(len,banks.values())),overlap=0,engineering_state_separate=True,
        history_files={str(p):o.sha(p) for p in paths}))
    lock_json(out/'experiment.json',dict(config=config(args),implementation=fingerprints(),models=registry(args),sources=sources,
        platform=dict(system=platform.platform(),python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__),
        cost_estimate=dict(max_prefixes=config(args)['prefix_cap'],max_new_suffixes=config(args)['new_suffix_cap'],
            estimated_serial_minutes=[15,30],basis='fixed1000_and_timed50ms; shared_post_capture_states; opponent_search_separately_recorded')))
    shutil.copyfile(CODE/'L45_SEARCH_CONTRIBUTION.md',out/'PLAN.md')
    runtime=out/'runtime';runtime.mkdir(exist_ok=True)
    shutil.copyfile(Path(cpp.__file__),runtime/Path(cpp.__file__).name)
    shutil.copyfile(CODE/'cpp/build_search_contribution/build_manifest.json',runtime/'build_manifest.json')
    shutil.copytree(CODE/'cpp/build_search_contribution/source',runtime/'source',dirs_exist_ok=True)
    shutil.copyfile(CODE/'solvers/search_contribution.py',runtime/'search_contribution.py')
    return banks

def prefix_tasks(args,banks):
    reg=registry(args);tasks=[]
    for bank in banks.values():
        for spec in bank:
            for opponent in prior.OPPONENTS:
                identity=dict(protocol=o.sha(args.output/'experiment.json'),episode_hash=spec['episode_hash'],opponent=opponent)
                key=o.json_hash(identity)
                tasks.append(dict(**identity,key=key,spec=spec,registry=reg,own_settings=settings(args),opponent_settings=opponent_settings(args)))
    return tasks

def prefix_one(task):
    old.previous.seed_task(task);p,state=prior.restore_scenario(task['spec']);stats=old.previous.initial_stats()
    own=prior.make_own(p,task['registry'],'S',task['own_settings'],task['spec']['search_seeds'][1])
    other=old.previous.base.private_solver(p,task['registry']['L'],task['opponent'],0,task['opponent_settings'],task['spec']['search_seeds'][0])
    team=o.TeamSolver(other,own);initial=old.previous.snapshot(p,state,stats,own,other);trace=[];latencies=[]
    while not p.is_terminal(state) and stats['capture_count']==0:
        action=team.policy(p,state);nxt=old.previous.advance(p,state,action,stats)
        trace.append(dict(state=state.copy(),action=action.copy(),next_state=nxt.copy()));state=nxt
        latencies.append([team.last_pursuer_latency_ms,team.last_evader_latency_ms])
    snap=old.previous.snapshot(p,state,stats,own,other)
    snap['branch_available']=bool(stats['capture_count']==1 and not p.is_terminal(state));snap['next_decision']=len(trace)
    return dict(key=task['key'],scenario=task['spec'],opponent=task['opponent'],initial=initial,snapshot=snap,
        trajectory=trace,latencies=latencies,metrics=old.previous.metrics(p,state,stats),actual_opponent=other.actual_settings)

def prefix_path(args,key):return args.output/'prefixes'/(key+'.pt')

def read_prefix(args,task):
    commit=locked_read(args.output/'prefix_commits'/(task['key']+'.json'));path=prefix_path(args,task['key'])
    if o.sha(path)!=commit['sha256']:raise ValueError('Prefix record changed')
    record=o.load_torch(path)
    if record['key']!=task['key'] or record['scenario']!=task['spec'] or record['opponent']!=task['opponent']:
        raise ValueError('Prefix identity differs')
    return record

def run_prefixes(args,tasks,status):
    if len(tasks)>config(args)['prefix_cap']:raise ValueError('Prefix budget exceeded')
    lock_json(args.output/'prefix_manifest.json',[dict(key=t['key'],scene=t['spec']['episode_hash'],opponent=t['opponent']) for t in tasks])
    missing=[]
    for task in tasks:
        if (args.output/'prefix_commits'/(task['key']+'.lock.json')).exists():read_prefix(args,task)
        else:missing.append(task)
    done=len(tasks)-len(missing);status('prefix',completed=done,total=len(tasks))
    if missing:
        with mp.get_context('spawn').Pool(min(args.workers,len(missing))) as pool:
            for record in pool.imap_unordered(prefix_one,missing,chunksize=1):
                path=prefix_path(args,record['key']);o.save_torch(path,record)
                lock_json(args.output/'prefix_commits'/(record['key']+'.json'),dict(sha256=o.sha(path)))
                done+=1;status('prefix',completed=done,total=len(tasks))

def requests(args,tasks,phase):
    reg=registry(args);items=[]
    for t in tasks:
        for uid in UIDS:
            requested=settings(args,uid,phase);behavior=effective(uid,reg,phase)
            identity=dict(protocol=o.sha(args.output/'experiment.json'),policy_id=o.json_hash(behavior),
                episode_hash=t['spec']['episode_hash'],role=1,opponent=t['opponent'],timing_context='single_process_'+phase,
                own_settings=requested,opponent_settings=opponent_settings(args),mode='search',phase=phase,prefix_key=t['key'])
            key=o.json_hash(identity)
            task=dict(**identity,key=key,uid=uid,spec=t['spec'],registry=reg,effective=behavior,
                prefix_path=str(prefix_path(args,t['key'])),prefix_sha=o.sha(prefix_path(args,t['key'])))
            items.append(dict(key=key,uid=uid,task=task))
    return items

def restore(task):
    if o.sha(task['prefix_path'])!=task['prefix_sha']:raise ValueError('Source prefix changed')
    pre=o.load_torch(task['prefix_path']);snap=pre['snapshot'];p,_=prior.restore_scenario(task['spec'])
    own=make_own(p,task['registry'],task['uid'],task['own_settings'],task['spec']['search_seeds'][1])
    other=old.previous.base.private_solver(p,task['registry']['L'],task['opponent'],0,task['opponent_settings'],task['spec']['search_seeds'][0])
    own.solver.set_rng_state(p,snap['own_rng'])
    if snap['opponent_rng'] is not None:other.solver.set_rng_state(p,snap['opponent_rng'])
    o.restore_rng(snap['python_rng']);p.clear_closed_loop_cache()
    return p,snap['state'].copy(),own,other,copy.deepcopy(snap['stats']),pre

def evaluate_one(task):
    p,state,own,other,stats,pre=restore(task);started=time.perf_counter();snap=pre['snapshot']
    if snap['branch_available']:
        row,detail,decisions,elapsed=old.simulate(p,state,own,other,stats)
    else:
        row=pre['metrics'].copy();now=old.previous.snapshot(p,state,stats,own,other)
        detail=dict(initial=now,first_capture=None,trajectory=[],roots=[],final=now);decisions=[];elapsed=0.
    row.update({k:task[k] for k in ('key','policy_id','role','opponent','timing_context','mode','phase','prefix_key')})
    row.update(uid=task['uid'],episode_hash=task['spec']['episode_hash'],layout=task['spec']['layout'],map_id=task['spec']['map_id'],
        search_seeds=task['spec']['search_seeds'],actual_own=dict(task['own_settings'],effective=task['effective']),
        actual_opponent=other.actual_settings,own_cost=own.solver.cost,own_mcts_calls=own.solver.calls,
        episode_wall_ms=(time.perf_counter()-started)*1000,deployment_episode_ms=elapsed,
        trajectory_sha256=old.previous.trajectory_digest(detail['trajectory']),decision_latency_ms=decisions,
        branch_available=snap['branch_available'],shared_prefix_steps=pre['snapshot']['stats']['steps'],reused_geometry=False)
    row['_record']=dict(key=task['key'],scenario=task['spec'],effective=task['effective'],role=1,opponent=task['opponent'],
        prefix_key=task['prefix_key'],phase=task['phase'],**detail)
    return row

def reuse_geometry(args,item,fixed_items):
    task=item['task'];source=next(i for i in fixed_items if i['uid']=='Geometry-only' and i['task']['prefix_key']==task['prefix_key'])
    row,d=read_episode(args,source);row=copy.deepcopy(row)
    for k in ('episode_record','content_sha256'):row.pop(k,None)
    for k in ('key','policy_id','timing_context','phase'):row[k]=task[k]
    row.update(reused_geometry=True,actual_own=dict(task['own_settings'],effective=task['effective']))
    detail={k:v for k,v in d.items() if k not in ('key','scenario','effective','role','opponent','prefix_key','phase')}
    row['_record']=dict(key=task['key'],scenario=task['spec'],effective=task['effective'],role=1,opponent=task['opponent'],
        prefix_key=task['prefix_key'],phase=task['phase'],reused_from=source['key'],**detail)
    return row

def cpu_snapshot():
    return dict(loadavg=Path('/proc/loadavg').read_text().strip(),affinity=sorted(os.sched_getaffinity(0)),
        observed_unix=time.time(),pid=os.getpid())

def choose_cpu(args):
    path=args.output/'serial_cpu.json'
    if path.exists():record=locked_read(path)
    else:
        def sample():
            values={}
            for line in Path('/proc/stat').read_text().splitlines():
                name,*rest=line.split()
                if name.startswith('cpu') and name[3:].isdigit():
                    x=list(map(int,rest));values[int(name[3:])]=(sum(x[:8]),x[3]+x[4])
            return values
        first=sample();time.sleep(.25);last=sample();allowed=sorted(os.sched_getaffinity(0))
        usage={c:1-(last[c][1]-first[c][1])/max(1,last[c][0]-first[c][0]) for c in allowed}
        cpu=min(allowed,key=lambda c:(usage[c],-c))
        record=lock_json(path,dict(cpu=cpu,allowed=allowed,observed_cpu_usage=usage,selection='lowest_sampled_usage; deterministic_high_id_tie'))
    if record['cpu'] not in os.sched_getaffinity(0):raise ValueError('Frozen serial CPU is unavailable')
    os.sched_setaffinity(0,{record['cpu']});o.write_json(args.output/'serial_environment_start.json',cpu_snapshot())
    return record['cpu']

def check(args,status):
    if (args.output/'checks.lock.json').exists():return locked_read(args.output/'checks.json')
    spec=locked_read(args.output/'engineering_scenario.json');p,state=prior.restore_scenario(spec);reg=registry(args);probes=[]
    for uid in ('S','F','G'):
        a=make_own(p,reg,uid,settings(args),spec['search_seeds'][1]);b=prior.make_own(p,reg,uid,settings(args),spec['search_seeds'][1])
        initial=a.solver.get_rng_state(p);actual=a.policy(p,state);expected=b.policy(p,state)
        if not np.array_equal(actual,expected) or not np.array_equal(a.solver.last_result.root_audit,b.solver.last_result.root_audit) or a.solver.get_rng_state(p)!=b.solver.get_rng_state(p):
            raise ValueError('Fixed solver differs from frozen implementation: '+uid)
        a.solver.set_rng_state(p,initial);again=a.policy(p,state)
        if not np.array_equal(actual,again):raise ValueError('Fixed root resume changed')
        probes.append(dict(uid=uid,action_root_rng_exact=True))
    nn_probe=settings(args,'G-no-neural');nn_probe['search_budget']=max(256,nn_probe['search_budget'])
    n=make_own(p,reg,'G-no-neural',nn_probe,spec['search_seeds'][1])
    n.policy(p,state);audit=n.solver.last_result.guidance_stats
    if audit.get('policy_pursuer')!=0 or audit.get('discarded_neural_noise_pairs',0)<1:raise ValueError('Pursuer neural calls were not isolated')
    if audit.get('policy_evader',0)<1:raise ValueError('Simulated opponent network was unexpectedly removed')
    geo=make_own(p,reg,'Geometry-only',settings(args,'Geometry-only'),spec['search_seeds'][1]);rng=geo.solver.get_rng_state(p)
    action=geo.policy(p,state);result=geo.solver.last_result
    if result.num_visits!=0 or result.policy_evaluations!=0 or result.step_calls!=0 or geo.solver.get_rng_state(p)!=rng:
        raise ValueError('Direct geometry ran search, a network or random sampling')
    probe=cpp.candidate_debug(geo.solver._problem_wrapper_for(p),state,3,np.zeros(2,np.float32))
    robot=next(r for r in p.pursuers if p.is_active(state,r))
    np.testing.assert_array_equal(action[p.action_idxs[robot],0],np.asarray(probe['action']).reshape(-1))
    timed=make_own(p,reg,'G',settings(args,'G','timed'),spec['search_seeds'][1]);t=time.perf_counter();timed.policy(p,state);ms=(time.perf_counter()-t)*1000
    if timed.solver.last_result.num_visits<1 or timed.solver.last_result.guidance_stats['simulation_cap_hit']:
        raise ValueError('Time budget did not stop real search before simulation cap')
    return lock_json(args.output/'checks.json',dict(passed=True,fixed_parity=probes,neural_pursuer_calls=0,
        simulated_evader_network_retained=True,Geometry_no_simulations_no_network_no_rng=True,G_base_action_exact=True,
        time_budget_ms=config(args)['deadline_ms'],time_probe_actual_ms=ms,time_probe_simulations=timed.solver.last_result.num_visits,
        extra_closed_loop_scientific_games=0))

def warmup(args):
    if (args.output/'warmup.lock.json').exists():return
    spec=locked_read(args.output/'engineering_scenario.json');p,state=prior.restore_scenario(spec);records=[]
    for phase in ('fixed','timed'):
        for uid in UIDS:
            raw=make_own(p,registry(args),uid,settings(args,uid,phase),spec['search_seeds'][1])
            raw.solver._problem_wrapper_for(p);t=time.perf_counter();raw.policy(p,state)
            records.append(dict(uid=uid,phase=phase,ms=(time.perf_counter()-t)*1000,simulations=int(raw.solver.last_result.num_visits)))
    lock_json(args.output/'warmup.json',dict(records=records,independent_engineering_state=True,extra_closed_loop_games=0))

def run_stage(args,items,phase,status,fixed_items=None):
    if phase not in ('fixed','timed') or len(items)>config(args)['prefix_cap']*len(UIDS):
        raise ValueError('Frozen stage budget exceeded')
    lock_json(args.output/phase/'manifest.json',[dict(key=i['key'],uid=i['uid'],prefix=i['task']['prefix_key']) for i in items])
    order=[];prefixes=list(dict.fromkeys(i['task']['prefix_key'] for i in items))
    by_key={(i['task']['prefix_key'],i['uid']):i for i in items}
    for j,key in enumerate(prefixes):
        uids=UIDS[j%len(UIDS):]+UIDS[:j%len(UIDS)]
        order.extend(by_key[(key,u)] for u in uids)
    done=0;started=time.perf_counter()
    for item in order:
        if (args.output/'cache_checksums'/(item['key']+'.lock.json')).exists():read_cached(args,item)
        elif phase=='timed' and item['uid']=='Geometry-only':commit_row(args,reuse_geometry(args,item,fixed_items))
        else:commit_row(args,evaluate_one(item['task']))
        done+=1
        status(phase,completed=done,total=len(items),current_method=item['uid'],
            stage_elapsed_seconds=time.perf_counter()-started,serial_cpu=list(os.sched_getaffinity(0)),serial_pid=os.getpid())

def root_timing(args,tasks,status):
    done=0
    for task in tasks:
        path=args.output/'root_timing'/(task['key']+'.json')
        if path.with_suffix('.lock.json').exists():locked_read(path);done+=1;continue
        pre=read_prefix(args,task);snap=pre['snapshot']
        if not snap['branch_available']:
            lock_json(path,dict(key=task['key'],eligible=False,records=[]));done+=1;continue
        p,_=prior.restore_scenario(task['spec']);state=snap['state'];solvers={}
        for uid in UIDS:
            own=make_own(p,task['registry'],uid,settings(args,uid),task['spec']['search_seeds'][1])
            own.solver.set_rng_state(p,snap['own_rng']);t=time.perf_counter();own.policy(p,state)
            solvers[uid]=(own,(time.perf_counter()-t)*1000)
        records=[]
        for repeat in range(config(args)['root_timing_repeats']):
            offset=(done+repeat)%len(UIDS);order=UIDS[offset:]+UIDS[:offset]
            for uid in order:
                own,cold=solvers[uid];own.solver.set_rng_state(p,snap['own_rng'])
                t=time.perf_counter();own.policy(p,state);elapsed=(time.perf_counter()-t)*1000;r=own.solver.last_result
                records.append(dict(uid=uid,repeat=repeat,ms=elapsed,cold_ms=cold,simulations=int(r.num_visits),
                    policy_pursuer=r.guidance_stats.get('policy_pursuer',0),policy_evader=r.guidance_stats.get('policy_evader',0),
                    geometry_queries=r.guidance_stats.get('geometry_queries',0),step_calls=int(r.step_calls)))
        lock_json(path,dict(key=task['key'],scene=task['spec']['episode_hash'],opponent=task['opponent'],eligible=True,records=records))
        done+=1;status('root_timing',completed=done,total=len(tasks),serial_cpu=list(os.sched_getaffinity(0)))

def pair_rows(rows):
    index={};meta={}
    for row in rows:
        key=(row['phase'],row['uid'],row['episode_hash'],row['opponent'])
        if key in index:raise ValueError('Duplicate phase/scene/method/opponent')
        index[key]=row;meta[row['episode_hash']]=row['layout']
    scenes=sorted(meta)
    if len(index)!=len(scenes)*len(UIDS)*3*2:raise ValueError('Missing paired scientific row')
    for phase in ('fixed','timed'):
        for scene in scenes:
            for opponent in prior.OPPONENTS:
                keys=[(phase,u,scene,opponent) for u in UIDS]
                if any(k not in index for k in keys):raise ValueError('Incomplete opponent pairing')
                seeds=[index[k]['search_seeds'] for k in keys]
                if any(s!=seeds[0] for s in seeds):raise ValueError('Unpaired search seeds')
    return index,scenes,meta

def analyze(rows):
    index,scenes,meta=pair_rows(rows);rng=np.random.default_rng(59020261024);sections={};primary_samples=[];primary_means=[];primary_targets=[]
    scopes=dict(all=scenes,four=[s for s in scenes if meta[s]!='none'])
    scopes.update({g:[s for s in scenes if meta[s]==g] for g in ('original_four','unseen_four','none')})
    for scope,selected in scopes.items():
        if not selected:continue
        strata=[[i for i,s in enumerate(selected) if meta[s]==g] for g in ('original_four','unseen_four','none')]
        draws=np.concatenate([rng.choice(ii,(20000,len(ii))) for ii in strata if ii],axis=1)
        sections[scope]={}
        for phase in ('fixed','timed'):
            groups={}
            for opp_label,opponents in [('mean',prior.OPPONENTS)]+[(opp,(opp,)) for opp in prior.OPPONENTS]:
                summary={};comparisons={}
                for uid in UIDS:
                    group=[index[(phase,uid,s,opp)] for s in selected for opp in opponents]
                    y=np.array([np.mean([index[(phase,uid,s,opp)]['legal_double_capture'] for opp in opponents]) for s in selected])
                    latency=[x[0] for row in group for x in row['decision_latency_ms']]
                    total=np.array([np.mean([sum(z[0] for z in index[(phase,uid,s,opp)]['decision_latency_ms']) for opp in opponents]) for s in selected])
                    summary[uid]=dict(games=len(group),successes=sum(r['legal_double_capture'] for r in group),success_rate=float(y.mean()),
                        ci95=np.quantile(y[draws].mean(1),[.025,.975]).tolist(),
                        own_illegal=sum(r['pursuer_illegal'] for r in group),opponent_illegal=sum(r['evader_illegal'] for r in group),
                        zero_capture=sum(r['capture_count']==0 for r in group),one_capture_timeout=sum(r['one_capture_timeout'] for r in group),
                        mean_completion_cost=float(np.mean([r['completion_cost'] for r in group])),
                        mean_suffix_pursuit_ms=float(total.mean()),suffix_pursuit_ms_ci95=np.quantile(total[draws].mean(1),[.025,.975]).tolist(),
                        mean_decision_ms=float(np.mean(latency)) if latency else None,p95_decision_ms=float(np.quantile(latency,.95)) if latency else None,
                        maximum_decision_ms=float(max(latency)) if latency else None,decisions=len(latency),
                        own_cost={k:sum(r['own_cost'][k] for r in group) for k in ('step_calls','policy_evaluations','value_evaluations')})
                for a,b in PAIRS:
                    name=a+' minus '+b
                    raw=np.array([[index[(phase,a,s,opp)]['legal_double_capture']-index[(phase,b,s,opp)]['legal_double_capture'] for opp in opponents] for s in selected],float)
                    x=raw.mean(1);sample=x[draws].mean(1)
                    cost=np.array([np.mean([index[(phase,a,s,opp)]['completion_cost']-index[(phase,b,s,opp)]['completion_cost'] for opp in opponents]) for s in selected])
                    elapsed=np.array([np.mean([sum(z[0] for z in index[(phase,a,s,opp)]['decision_latency_ms'])-sum(z[0] for z in index[(phase,b,s,opp)]['decision_latency_ms']) for opp in opponents]) for s in selected])
                    comparisons[name]=dict(delta=float(x.mean()),ci95=np.quantile(sample,[.025,.975]).tolist(),lower95=float(np.quantile(sample,.05)),
                        gained=int((raw>0).sum()),lost=int((raw<0).sum()),completion_cost_delta=float(cost.mean()),completion_cost_ci95=np.quantile(cost[draws].mean(1),[.025,.975]).tolist(),
                        suffix_ms_delta=float(elapsed.mean()),suffix_ms_ci95=np.quantile(elapsed[draws].mean(1),[.025,.975]).tolist())
                    if scope=='all' and opp_label=='mean' and (a,b) in PAIRS[1:4]:
                        primary_samples.append(sample);primary_means.append(float(x.mean()));primary_targets.append(comparisons[name])
                groups[opp_label]=dict(summary=summary,comparisons=comparisons)
            sections[scope][phase]=groups
    samples=np.stack(primary_samples,1);center=np.array(primary_means);critical=float(np.quantile(np.abs(samples-center).max(1),.95))
    for record in primary_targets:record['simultaneous_ci95']=[record['delta']-critical,record['delta']+critical]
    return dict(sections=sections,physical_scenes=len(scenes),bootstrap_seed=59020261024,draws=20000,
        cluster_unit='physical_scene; same opponents and repeated measurements together',same_prefix_scope=True)

def diagnostics(args,items,tasks):
    total={phase:{uid:dict(policy_pursuer=0.,policy_evader=0.,geometry_queries=0.,simulations=[],latencies=[],
        deadline_excess_ms=[],cap_hits=0,graph_build_ms=0.,plan_ms=0.,geometry_decisions=0,eligible=0) for uid in UIDS} for phase in ('fixed','timed')}
    index=[];initials={};simchecks=0;new_suffixes=0
    for item in items:
        row,d=read_episode(args,item);phase=item['task']['phase'];uid=item['uid'];key=item['task']['prefix_key'];g=total[phase][uid]
        initials.setdefault(key,[]).append(d['initial']);new_suffixes+=int(bool(row['branch_available']) and not row['reused_geometry'])
        g['eligible']+=int(row['branch_available'])
        for root,latency in zip(d['roots'],row['decision_latency_ms']):
            g['simulations'].append(root['visits']);g['latencies'].append(latency[0]);audit=root['guidance']
            for field in ('policy_pursuer','policy_evader','geometry_queries','graph_build_ms','plan_ms'):g[field]+=audit.get(field,0)
            g['cap_hits']+=int(audit.get('simulation_cap_hit',0));g['geometry_decisions']+=int(audit.get('direct_geometry',0))
            if uid=='Geometry-only':
                if root['visits']!=0 or any(root['cost'].values()):raise ValueError('Geometry-only used search/network')
            elif phase=='fixed':
                if root['visits']!=config(args)['fixed_simulations']:raise ValueError('Fixed budget differs')
            elif root['visits']<1:raise ValueError('Time-limited search completed no simulation')
            if uid=='G-no-neural' and audit.get('policy_pursuer',0)!=0:raise ValueError('No-neural queried pursuer actor')
            if root['cost']['value_evaluations'] or audit.get('guided_steps',0):raise ValueError('Unexpected value/guided rollout')
            if uid in SEARCH_UIDS and phase=='timed':g['deadline_excess_ms'].append(max(0.,latency[0]-config(args)['deadline_ms']))
            simchecks+=1
        index.append(dict(key=row['key'],phase=phase,uid=uid,scene=row['episode_hash'],opponent=row['opponent'],
            success=row['legal_double_capture'],record=row['episode_record']['path'],branch=row['branch_available'],reused=row['reused_geometry']))
    for key,snaps in initials.items():
        for snap in snaps[1:]:
            for field in ('state','stats','time','remaining_time','own_rng','opponent_rng','python_rng'):
                if not old.s.equal_tree(snaps[0][field],snap[field]):raise ValueError('Shared initial snapshot/RNG differs')
    if new_suffixes>config(args)['new_suffix_cap']:raise ValueError('Scientific suffix ledger exceeded')
    for phase in total.values():
        for g in phase.values():
            for name in ('simulations','latencies','deadline_excess_ms'):
                values=g.pop(name);g[name+'_mean']=float(np.mean(values)) if values else None;g[name+'_p95']=float(np.quantile(values,.95)) if values else None
                g[name+'_max']=float(max(values)) if values else None
                if name=='deadline_excess_ms':g['deadline_overshoots']=sum(v>0 for v in values);g['deadline_overshoots_gt1ms']=sum(v>1 for v in values)
    # Physical-state clustering for repeated equal-input root timing.
    bench=[locked_read(args.output/'root_timing'/(task['key']+'.json')) for task in tasks]
    values={uid:{} for uid in UIDS};raw={uid:[] for uid in UIDS}
    for entry in bench:
        if not entry['eligible']:continue
        for uid in UIDS:
            part=[r['ms'] for r in entry['records'] if r['uid']==uid];values[uid].setdefault(entry['scene'],[]).append(float(np.mean(part)));raw[uid].extend(part)
    scenes=sorted(values['F']);rng=np.random.default_rng(59020261024)
    bench_summary={uid:dict(mean_ms=float(np.mean([np.mean(v) for v in values[uid].values()])) if values[uid] else None,
        p95_ms=float(np.quantile(raw[uid],.95)) if raw[uid] else None,measurements=len(raw[uid])) for uid in UIDS}
    if scenes:
        x=np.array([np.mean(values['G'][s])-np.mean(values['F'][s]) for s in scenes]);draws=rng.integers(0,len(scenes),(20000,len(scenes)))
        delta=dict(delta_ms=float(x.mean()),ci95=np.quantile(x[draws].mean(1),[.025,.975]).tolist(),physical_scenes=len(scenes))
    else:delta=None
    integrity=dict(passed=True,verified_root_decisions=simchecks,paired_prefix_groups=len(initials),new_closed_loop_suffixes=new_suffixes,
        common_prefixes=len(tasks),models_frozen=True,Geometry_zero_search_network=True,no_pursuer_neural_calls_in_N=True)
    return dict(by_phase=total,equal_input_roots=bench_summary,equal_input_G_minus_F=delta),index,integrity

def report(args,tasks,items,rows):
    result=analyze(rows);diag,index,integrity=diagnostics(args,items,tasks)
    lock_json(args.output/'statistics.json',result);lock_json(args.output/'diagnostics.json',diag)
    lock_json(args.output/'trajectory_index.json',index);lock_json(args.output/'integrity.json',integrity)
    lines=['G效率与神经候选/搜索贡献实验' if not args.smoke else '隔离工程smoke，不产生效果结论','',
        f'{result["physical_scenes"]}个全新物理场景，三个原对手，共同S首捕前缀；fixed{config(args)["fixed_simulations"]}模拟与timed{config(args)["deadline_ms"]}ms均为单进程。',
        'G-no-neural移除所有追捕神经候选、保留模拟逃逸网络；Geometry-only仅在后段无网络无MCTS，首捕前缀仍为S。','',
        '|阶段|组|成功/请求|合法双捕率|完成成本|单步mean ms|P95 ms|每局后段追捕耗时 ms|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for phase in ('fixed','timed'):
        for uid,g in result['sections']['all'][phase]['mean']['summary'].items():
            mean=g['mean_decision_ms'];p95=g['p95_decision_ms'];mt=f'{mean:.3f}' if mean is not None else '—';pt=f'{p95:.3f}' if p95 is not None else '—'
            lines.append(f'|{phase}|{uid}|{g["successes"]}/{g["games"]}|{100*g["success_rate"]:.2f}%|{g["mean_completion_cost"]:.2f}|{mt}|{pt}|{g["mean_suffix_pursuit_ms"]:.2f}|')
    lines+=['','按物理场景配对bootstrap20000次，保留对手与阶段关联。','',
        '|阶段|比较|差 pp|95%区间 pp|挽回/丢失|完成成本差|后段追捕耗时差 ms [95%区间]|',
        '|---|---|---:|---|---:|---:|---|']
    for phase in ('fixed','timed'):
        for name,c in result['sections']['all'][phase]['mean']['comparisons'].items():
            lo,hi=np.array(c['ci95'])*100;cl,ch=c['suffix_ms_ci95']
            lines.append(f'|{phase}|{name}|{100*c["delta"]:+.2f}|[{lo:+.2f},{hi:+.2f}]|{c["gained"]}/{c["lost"]}|{c["completion_cost_delta"]:+.2f}|{c["suffix_ms_delta"]:+.2f} [{cl:+.2f},{ch:+.2f}]|')
    lines+=['','同输入、同1000模拟、三次热根计时：',str(diag['equal_input_roots']),
        'G-F同状态实际根延迟差：'+str(diag['equal_input_G_minus_F']),
        '', '完整计数及截止超时见diagnostics.json；阶段/地图/对手分项见statistics.json。',
        '时间预算原子模拟完成后停止，实际返回延迟可能超过50ms；不以配置值替代实测。Geometry-only不人为等待。',
        'Geometry-only与no-neural对比是整个搜索决策流程的附加价值，不分解为单个树内组件。',
        '本轮固定O、30新物理场景，未改变当前推荐old0:novalue，未自动展开其他实验。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    lock_json(args.output/'results.json',dict(reason='smoke_complete' if args.smoke else 'contribution_study_complete',
        physical_scenes=result['physical_scenes'],scientific_row_requests=len(rows),new_suffixes=integrity['new_closed_loop_suffixes'],
        prefix_tasks=len(tasks),training_episodes=0,recommended='old0:novalue'))
    plots(args,result,diag)

def plots(args,result,diag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=args.output/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(14,4));x=np.arange(len(UIDS))
    for i,phase in enumerate(('fixed','timed')):
        g=result['sections']['all'][phase]['mean']['summary'];axes[0].bar(x+(i-.5)*.35,[g[u]['success_rate']*100 for u in UIDS],.35,label=phase)
    axes[0].set(xticks=x,xticklabels=['S','F','G','G-no-NN','Geometry'],ylim=(0,105),ylabel='Legal double capture (%)');axes[0].legend()
    roots=diag['equal_input_roots'];axes[1].bar(x,[roots[u]['mean_ms'] or 0 for u in UIDS]);axes[1].set(xticks=x,xticklabels=['S','F','G','G-no-NN','Geometry'],ylabel='Equal-input fixed root time (ms)')
    for phase in ('fixed','timed'):
        g=result['sections']['all'][phase]['mean']['summary'];axes[2].plot([g[u]['mean_suffix_pursuit_ms'] for u in UIDS],[g[u]['success_rate']*100 for u in UIDS],'o',label=phase)
        for u in UIDS:axes[2].annotate(u,(g[u]['mean_suffix_pursuit_ms'],g[u]['success_rate']*100),fontsize=7)
    axes[2].set(xlabel='Suffix pursuit compute per requested game (ms)',ylabel='Success (%)');axes[2].legend()
    fig.tight_layout();fig.savefig(out/'contributions.png',dpi=180);plt.close(fig)

def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',nargs='?',default='pipeline',choices=['pipeline','prepare','check','prefix','fixed','timed','report','watch'])
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
                state.update(actual_workers=len([p for p in mp.active_children() if p.is_alive()]),
                    committed_suffixes=len(list((args.output/'cache_checksums').glob('*.lock.json'))),
                    committed_prefixes=len(list((args.output/'prefix_commits').glob('*.lock.json'))))
                o.write_json(args.output/'status.json',state)
        def heartbeat():
            while not stop.wait(5):status(state['phase'])
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:
            banks=prepare(args);status('prepared',physical_scenes=sum(map(len,banks.values())))
            if args.stage=='prepare':return
            check(args,status)
            if args.stage=='check':return
            tasks=prefix_tasks(args,banks)
            if args.stage in ('pipeline','prefix'):run_prefixes(args,tasks,status)
            if args.stage=='prefix':return
            choose_cpu(args);status('warmup',serial_cpu=list(os.sched_getaffinity(0)));warmup(args)
            fixed=requests(args,tasks,'fixed');timed=requests(args,tasks,'timed')
            if args.stage in ('pipeline','fixed'):run_stage(args,fixed,'fixed',status)
            if args.stage=='fixed':return
            if args.stage in ('pipeline','timed'):
                root_timing(args,tasks,status);run_stage(args,timed,'timed',status,fixed)
            status('report');all_items=fixed+timed;rows=[read_cached(args,i) for i in all_items];report(args,tasks,all_items,rows)
            o.write_json(args.output/'serial_environment_end.json',cpu_snapshot())
            status('complete',reason='smoke_complete' if args.smoke else 'contribution_study_complete',recommended='old0:novalue',elapsed_seconds=time.time()-state['started'])
        except BaseException as e:status('stopped',reason=f'{type(e).__name__}: {e}');raise
        finally:stop.set();thread.join(timeout=1)

if __name__=='__main__':main()
