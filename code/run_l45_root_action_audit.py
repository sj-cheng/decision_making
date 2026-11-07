"""Bounded root-action counterfactual audit. No training or model selection."""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import contextlib
import copy
import fcntl
import hashlib
import io
import itertools
import multiprocessing as mp
from pathlib import Path
import platform
import random
import shutil
import threading
import time

import numpy as np
import torch
torch.set_num_threads(1)
import run_l45_phase_handoff as h
from solvers.root_audit_puct import RootPUCT

o, s, f = h.o, h.s, h.f
lock_json, locked_read = h.lock_json, h.locked_read
ROOT, CODE = o.ROOT, o.CODE
SOURCE = h.DEFAULT_OUTPUT
DEFAULT = ROOT/'current/l45_root_action_audit_20261024'
SMOKE = ROOT/'current/l45_root_action_audit_smoke_20261024'


def config(args):
    return dict(schema=1, smoke=args.smoke, training_episodes=0, optimizer_updates=0, model_exports=0,
        budget=3 if args.smoke else 1000, opponent_budget=3 if args.smoke else 3000,
        depth=100, pursuit_betas=[.5,0.], opponent_betas=[0.,0.], candidates=4, streams=3,
        candidate_order=['selected','highest_other_Q','most_visited_if_new','farthest_active_action'],
        ties='root_order; selector_prefers_A0_on_success_tie', bootstrap_seed=49020261024,
        bootstrap_draws=20000, minimum_gain=.05, illegal_margin=.02, positive_streams=2,
        seed_bytes='SHA256 UTF8, first 8 bytes big endian; C++/NumPy low32, Torch low63, Python full64',
        decision='simultaneous_current_state; replace_one_pursuit_action; then_O',
        formal_roots=99, formal_suffix_cap=1188, recommended='old0:novalue', no_automatic_expansion=True)


def fingerprints():
    paths = [Path(__file__), CODE/'solvers/root_audit_puct.py', CODE/'build_root_action_audit.py',
        CODE/'cpp/build_root_audit/build_manifest.json', *list((CODE/'cpp/build_root_audit').glob('*.so'))]
    if len(paths) != 5:
        raise ValueError('Build exactly one private root extension')
    manifest=o.read_json(CODE/'cpp/build_root_audit/build_manifest.json')
    for section,base in (('original',CODE/'cpp'),('generated',CODE/'cpp/build_root_audit/source')):
        for name,digest in manifest[section].items():
            if o.sha(base/name)!=digest:
                raise ValueError('Private build source changed: '+name)
    if o.sha(paths[-1])!=manifest['binary']:
        raise ValueError('Private binary changed')
    return dict(files={str(p):o.sha(p) for p in paths}, inherited=h.fingerprints(),
        platform=[platform.platform(),platform.machine(),platform.python_version(),str(torch.__version__),np.__version__])


def registry(args):
    return {n:dict(models=str(args.output/'models'/n),model_hash=o.model_fingerprint(args.output/'models'/n))
            for n in ('O','L')}


def commit_pt(path, record):
    path = Path(path)
    if path.with_suffix('.lock.json').exists():
        old = read_pt(path)
        if not s.equal_tree(old,record):
            raise ValueError('Committed scientific record differs')
        return
    o.save_torch(path,record)
    lock_json(path.with_suffix('.json'),dict(sha256=o.sha(path)))


def read_pt(path):
    path = Path(path)
    if locked_read(path.with_suffix('.json'))['sha256'] != o.sha(path):
        raise ValueError('Cache checksum changed: '+str(path))
    return o.load_torch(path)


def prepare(args):
    if args.output == SOURCE or SOURCE in args.output.parents or args.output == ROOT:
        raise ValueError('Use an isolated output')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'experiment.lock.json').exists():
        verify(args)
        return locked_read(args.output/'bank.json')
    h.verify(h.parse_args([]))
    source_paths = [SOURCE/n for n in ('experiment.json','results.json','recommended.json','diagnostic_result.json')]
    for n in ('O','L'):
        dst = args.output/'models'/n
        dst.mkdir(parents=True,exist_ok=True)
        for model in o.MODEL_NAMES:
            origin = SOURCE/'models'/n/model
            source_paths.append(origin)
            shutil.copyfile(origin,dst/model)
    history = s.history_files(args)
    used = s.history_scenes(history)
    old_count = len(used)
    banks = dict(audit_four=[],audit_none=[],smoke_four=[],smoke_none=[])
    for phase,(name,bank) in enumerate(banks.items(),1601):
        layout = name.rsplit('_',1)[1]
        index = 0
        while not bank:
            spec = o.make_scenario(o.scenario_seed(phase,4902026102401 if args.smoke else 4902026102402,index),layout)
            index += 1
            identity = s.previous.physical_hash(spec)
            if identity not in used:
                used.add(identity);bank.append(spec)
    rows = []
    if args.smoke:
        # Independent controlled late one-capture states exercise every branch.
        for spec in banks['smoke_four']+banks['smoke_none']:
            p,state = o.restore_scenario(spec)
            state[[p.active_idxs[0],p.active_idxs[2]],0] = 0.
            state[p.time_idx,0] = p.tf-3*p.dt
            stats = h.initial_stats();stats.update(steps=77,first=77.,last=77.,capture_count=1)
            own,other = make_pair(p,registry(args),config(args))
            prefix = dict(state=state,stats=stats,own_rng=dict(rng=own.get_rng_state(p)),
                opponent_rng=other.get_rng_state(p),python_rng=o.rng_state(),branch=True)
            path = args.output/'smoke_prefixes'/(spec['episode_hash']+'.pt')
            commit_pt(path,prefix)
            rows.append(dict(spec=spec,prefix_path=str(path),prefix_sha=o.sha(path),historical_suffix=None))
    else:
        prefixes = h.diagnostic_tasks(h.parse_args([]))
        if len(prefixes) != 200:
            raise ValueError('Incomplete historical bank')
        for task in prefixes:
            if task['prefix'] != 'O':
                continue
            prefix = h.read_segment(task)
            for path in (Path(task['result']),Path(task['result']).with_suffix('.lock.json')):
                source_paths.append(path)
            if not prefix['branch']:
                continue
            suffix = next(t for t in h.suffix_tasks(h.parse_args([]),[task]) if t['after']=='O')
            h.read_segment(suffix)
            source_paths += [Path(suffix['result']),Path(suffix['result']).with_suffix('.lock.json')]
            rows.append(dict(spec=task['spec'],prefix_path=task['result'],prefix_sha=o.sha(task['result']),
                historical_suffix=suffix['result']))
        if len(rows) != 99:
            raise ValueError('Expected all 99 O first-capture branches')
    bank = dict(rows=rows,engineering=banks,total_physical=2 if args.smoke else 100,
        terminal_without_branch=0 if args.smoke else 1,history_exception=not args.smoke)
    lock_json(args.output/'scenarios.json',banks)
    lock_json(args.output/'bank.json',bank)
    lock_json(args.output/'experiment.json',dict(config=config(args),implementation=fingerprints(),
        source={str(p):o.sha(p) for p in source_paths},history={str(p):o.sha(p) for p in history},
        models=registry(args),bank_sha=o.sha(args.output/'bank.json'),scenario_sha=o.sha(args.output/'scenarios.json')))
    lock_json(args.output/'scenario_independence.json',dict(previous_physical_scenes=old_count,
        new_engineering_scenes=len(used)-old_count,overlap=0,source_history_reuse=not args.smoke))
    verify(args)
    return bank


def verify(args):
    record = locked_read(args.output/'experiment.json')
    if record['config'] != config(args) or record['implementation'] != fingerprints() or record['models'] != registry(args):
        raise ValueError('Protocol, model, build, implementation or platform changed')
    for p,digest in (record['source']|record['history']).items():
        if o.sha(p) != digest:
            raise ValueError('Frozen source or bank changed: '+p)
    bank = locked_read(args.output/'bank.json')
    if o.sha(args.output/'bank.json') != record['bank_sha'] or o.sha(args.output/'scenarios.json') != record['scenario_sha']:
        raise ValueError('Scenario manifest changed')
    ids = [s.previous.physical_hash(x) for values in bank['engineering'].values() for x in values]
    if len(ids) != len(set(ids)) or set(ids)&s.history_scenes([Path(p) for p in record['history']]):
        raise ValueError('Engineering scenario leakage')
    expected = 2 if args.smoke else 99
    if len(bank['rows']) != expected or len({x['spec']['episode_hash'] for x in bank['rows']}) != expected:
        raise ValueError('Missing or duplicate paired states')
    for row in bank['rows']:
        if o.sha(row['prefix_path']) != row['prefix_sha']:
            raise ValueError('Prefix changed')
    return record


def make_solver(p, model, method, role, cfg, export=False, seed=1, beta_value=None):
    if method == 'Policy Only':
        requested = dict(search_budget=cfg['budget'],search_only_budget=cfg['opponent_budget'],
            search_depth=100,beta_policy=.5,beta_value=0. if role else .1)
        return f.make_solver(p,model['models'],method,role,requested,seed)
    paths,value_path = h.oracle_paths(str(ROOT),iteration=45,model_dir=model['models'])
    with contextlib.redirect_stdout(io.StringIO()):
        policies,value = h.get_oracles(p,value_oracle_name='deterministic',value_oracle_path=value_path,
            policy_oracle_name='gaussian',policy_oracle_paths=paths)
    if method == 'Search Only':
        policies,value=[None]*p.num_robots,None
    beta = (0.,0.) if method=='Search Only' else (.5,0. if role else .1)
    if beta_value is not None and method!='Search Only':beta=(beta[0],beta_value)
    solver = RootPUCT(policies,value,cfg['opponent_budget'] if method=='Search Only' else cfg['budget'],*beta,export_root=export)
    solver.solver_wrapper.set_seed(int(seed))
    return solver


def make_pair(p, reg, cfg, export=False):
    for model in reg.values():
        if o.model_fingerprint(model['models']) != model['model_hash']:
            raise ValueError('Worker model changed')
    return (make_solver(p,reg['O'],'Search + Learning',1,cfg,export=export),
            make_solver(p,reg['L'],'Search Only',0,cfg))


def selected_candidates(table, p, state):
    table = np.asarray(table,dtype=np.float32)
    if table.ndim!=2 or table.shape[1]!=6+p.action_dim or not len(table):
        raise ValueError('Malformed root export')
    if not np.isfinite(table).all() or np.any(table[:,1]<=0) or np.sum(table[:,5])!=1:
        raise ValueError('Nonfinite or unvisited root entry or missing selected action')
    if not np.array_equal(table[:,0],np.arange(len(table))):
        raise ValueError('Root order changed')
    q = table[:,2].astype(float)+float(np.float32(p.gamma))*table[:,3].astype(float)/table[:,1]
    if np.any(np.abs(q-table[:,4])>1e-6*np.maximum(1.,np.abs(q))):
        raise ValueError('Q formula mismatch')
    dims = [j for robot in o.TEAMS[1] if p.is_active(state,robot) for j in p.action_idxs[robot]]
    executed = table[:,6:].copy()
    for robot in range(p.num_robots):
        if robot not in o.TEAMS[1] or not p.is_active(state,robot):
            executed[:,p.action_idxs[robot]]=0.
    keys = [tuple(float(x) for x in a) for a in executed]
    a0 = int(np.flatnonzero(table[:,5])[0])
    seen={keys[a0]};unique=[a0]
    for i in range(len(table)):
        if keys[i] not in seen:
            seen.add(keys[i]);unique.append(i)
    chosen=[a0]
    rest=[i for i in unique if i!=a0]
    if rest:
        chosen.append(max(rest,key=lambda i:(float(table[i,4]),-i)))
    most=max(unique,key=lambda i:(float(table[i,1]),-i))
    if most not in chosen:
        chosen.append(most)
    half=(np.asarray(p.action_lims)[dims,1]-np.asarray(p.action_lims)[dims,0])/2
    if not np.all(half>0):
        raise ValueError('Invalid action scale')
    normalized=executed[:,dims]/half
    while len(chosen)<4 and len(chosen)<len(unique):
        remain=[i for i in unique if i not in chosen]
        chosen.append(max(remain,key=lambda i:(min(float(np.linalg.norm(normalized[i]-normalized[j])) for j in chosen),-i)))
    return [dict(ordinal=i,action=executed[i].astype(float).reshape(-1,1),q=float(table[i,4]),
                 visits=int(table[i,1]),is_A0=i==a0) for i in chosen]


def tasks(args, bank):
    protocol=o.sha(args.output/'experiment.json')
    result=[]
    for row in bank['rows']:
        item=dict(kind='root',protocol=protocol,row=row,registry=registry(args),config=config(args))
        item['key']=o.json_hash(item)
        item['result']=str(args.output/'roots'/(item['key']+'.pt'))
        result.append(item)
    return result


def read_task(task):
    row=read_pt(task['result'])
    if row['key']!=task['key'] or row['task']!={k:v for k,v in task.items() if k!='result'}:
        raise ValueError('Cache task identity mismatch')
    return row


def extract_one(task):
    torch.set_num_threads(1)
    row=task['row']
    if o.sha(row['prefix_path'])!=row['prefix_sha']:
        raise ValueError('Prefix changed')
    prefix=read_pt(row['prefix_path'])
    p,_=o.restore_scenario(row['spec']);p.clear_closed_loop_cache()
    own,other=make_pair(p,task['registry'],task['config'],export=True)
    own.set_rng_state(p,prefix['own_rng']['rng']);other.set_rng_state(p,prefix['opponent_rng'])
    o.restore_rng(prefix['python_rng'])
    before=own.get_rng_state(p)
    state=prefix['state'].copy()
    action=own.policy(p,state,team=o.TEAMS[1])
    table=np.asarray(own.last_result.root_audit).copy()
    candidates=selected_candidates(table,p,state)
    if not np.array_equal(candidates[0]['action'],action):
        raise ValueError('Exported selected action differs from actual action')
    return dict(key=task['key'],task={k:v for k,v in task.items() if k!='result'},state=state,
        stats=prefix['stats'],root_table=table,candidates=candidates,own_before=before,
        own_after=own.get_rng_state(p),opponent_before=other.get_rng_state(p),python_rng=o.rng_state(),
        calls=dict(pursuer=own.calls,evader=other.calls))


def branch_tasks(args, roots):
    result=[]
    for root_task in roots:
        row=read_task(root_task)
        for index in range(len(row['candidates'])):
            for stream in range(3):
                item=dict(kind='suffix',protocol=o.sha(args.output/'experiment.json'),root_task=root_task,
                    root_sha=o.sha(root_task['result']),candidate=index,stream=stream)
                item['key']=o.json_hash(item)
                item['result']=str(args.output/'suffixes'/(item['key']+'.pt'))
                result.append(item)
    if len(result)>config(args)['formal_suffix_cap']:
        raise ValueError('Suffix budget exceeded')
    return result


def seed_for(identity,stream,role,component):
    text=f'l45-root-audit-v1|49020261024|{identity}|r{stream}|{role}|{component}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')


def initialize_stream(p, own, other, root, stream):
    if not stream:
        own.set_rng_state(p,root['own_after']);other.set_rng_state(p,root['opponent_before'])
        o.restore_rng(root['python_rng'])
        return
    identity=s.previous.physical_hash(root['task']['row']['spec'])
    for role,solver in (('pursuer',own),('evader',other)):
        solver.seeded_state(p,seed_for(identity,stream,role,'solver')&0xffffffff,
                            seed_for(identity,stream,role,'problem')&0xffffffff)
    random.seed(seed_for(identity,stream,'shared','python'))
    np.random.seed(seed_for(identity,stream,'shared','numpy')&0xffffffff)
    torch.manual_seed(seed_for(identity,stream,'shared','torch')&0x7fffffffffffffff)


def masked_policy(solver,p,state,role):
    return solver.policy(p,state,team=o.TEAMS[role])


def closed_loop(p,state,stats,own,other,first_action=None):
    trace=[];rewards=[]
    while not p.is_terminal(state):
        # Original TeamSolver evaluates the evader first from the same state.
        evader=masked_policy(other,p,state,0)
        pursuer=first_action if first_action is not None else masked_policy(own,p,state,1)
        first_action=None
        action=np.zeros((p.action_dim,1))
        for robot in o.TEAMS[0]:
            action[p.action_idxs[robot]]=evader[p.action_idxs[robot]]
        for robot in o.TEAMS[1]:
            action[p.action_idxs[robot]]=pursuer[p.action_idxs[robot]]
        reward=np.asarray(p.reward(state,action),float).reshape(-1)
        nxt=h.advance(p,state,action,stats)
        trace.append(dict(state=state.tolist(),action=action.tolist(),next_state=nxt.tolist()))
        rewards.append(reward.tolist());state=nxt
        if stats['steps']>int(p.tf/p.dt):
            raise ValueError('Remaining horizon reset')
    return state,trace,rewards


def branch_one(task):
    torch.set_num_threads(1)
    rt=task['root_task']
    if o.sha(rt['result'])!=task['root_sha']:
        raise ValueError('Root cache changed')
    root=read_task(rt)
    p,_=o.restore_scenario(rt['row']['spec']);p.clear_closed_loop_cache()
    own,other=make_pair(p,rt['registry'],rt['config'])
    initialize_stream(p,own,other,root,task['stream'])
    initial_rng=dict(own=own.get_rng_state(p),other=other.get_rng_state(p),python=o.rng_state())
    state=root['state'].copy();stats=copy.deepcopy(root['stats'])
    start=float(state[p.time_idx,0])
    state,trace,rewards=closed_loop(p,state,stats,own,other,root['candidates'][task['candidate']]['action'])
    metrics=h.metrics(p,state,stats)
    team=np.asarray(rewards,float)[:,list(o.TEAMS[1])] if rewards else np.zeros((0,2))
    mean=team.mean(axis=1)
    metrics.update(suffix_return=float(mean.sum()),discounted_suffix_return=float(np.dot(mean,p.gamma**np.arange(len(mean)))),
        suffix_return_team_sum=float(team.sum()),remaining_cost=float(metrics['completion_cost']-start))
    parity=False
    if task['candidate']==0 and task['stream']==0 and rt['row']['historical_suffix']:
        historical=read_pt(rt['row']['historical_suffix'])
        if h.science(metrics)!=h.science(historical['metrics']) or o.json_hash(trace)!=o.json_hash(historical['trajectory']) or \
            not np.array_equal(state,historical['state']) or own.get_rng_state(p)!=historical['own_rng']['rng'] or \
            other.get_rng_state(p)!=historical['opponent_rng']:
            raise ValueError('A0+r0 historical trajectory/RNG mismatch')
        parity=True
    return dict(key=task['key'],task={k:v for k,v in task.items() if k!='result'},metrics=metrics,
        trajectory=trace,rewards=rewards,state=state,initial_rng=initial_rng,own_rng=own.get_rng_state(p),
        opponent_rng=other.get_rng_state(p),historical_exact=parity,calls=dict(pursuer=own.calls,evader=other.calls))


def run_tasks(args,name,items,worker,pool,status):
    lock_json(args.output/name/'manifest.json',items)
    missing=[];calls=dict(pursuer=0,evader=0);committed=set()
    for task in items:
        if Path(task['result']).with_suffix('.lock.json').exists():
            row=read_task(task)
            committed.add(task['key'])
            for role in calls:calls[role]+=row['calls'][role]
        else:missing.append(task)
    done=len(items)-len(missing)
    status(name,completed=done,total=len(items),cached=done,mcts_calls=calls)
    iterator=pool.imap_unordered(worker,missing,chunksize=1) if pool else map(worker,missing)
    by_key={t['key']:t for t in items}
    physical={}
    for task in items:
        root_task=task['root_task'] if task['kind']=='suffix' else task
        identity=root_task['row']['spec']['episode_hash']
        physical.setdefault(identity,set()).add(task['key'])
    for row in iterator:
        commit_pt(by_key[row['key']]['result'],row)
        done+=1
        committed.add(row['key'])
        for role in calls:calls[role]+=row['calls'][role]
        if done%10==0 or done==len(items):
            status(name,completed=done,total=len(items),cached=len(items)-len(missing),mcts_calls=calls,
                completed_physical_scenes=sum(keys<=committed for keys in physical.values()),physical_scenes=len(physical))
    return [read_task(t) for t in items]


def choose_without_heldout(rows, held):
    candidates=sorted(rows)
    if candidates!=list(range(len(candidates))) or any(set(rows[i])!={0,1,2} for i in candidates):
        raise ValueError('Missing candidate/random-stream pairing')
    train=[r for r in range(3) if r!=held]
    base_illegal=sum(rows[0][r]['pursuer_illegal'] for r in train)
    eligible=[i for i in candidates if sum(rows[i][r]['pursuer_illegal'] for r in train)<=base_illegal]
    def order(i):
        return (-sum(rows[i][r]['legal_double_capture'] for r in train),0 if i==0 else 1,
                np.mean([rows[i][r]['completion_cost'] for r in train]),rows[i][train[0]]['ordinal'])
    return min(eligible,key=order)


def gate(delta, lower, streams, illegal):
    return bool(delta>=.05-1e-12 and lower>0 and sum(x>0 for x in streams)>=2 and illegal<=.02+1e-12)


def compute_statistics(groups,total_physical,draws=20000):
    contributions=[];details=[];agreements=[]
    for identity,group in sorted(groups.items()):
        selected=[];gains=[];illegal=[]
        for r in range(3):
            i=choose_without_heldout(group,r);selected.append(i)
            gains.append(group[i][r]['legal_double_capture']-group[0][r]['legal_double_capture'])
            illegal.append(group[i][r]['pursuer_illegal']-group[0][r]['pursuer_illegal'])
        contributions.append(gains)
        comparison=dict(q_return=[0,0,0],q_success=[0,0,0])
        for a,b in itertools.combinations(group,2):
            qdiff=group[a][0]['q']-group[b][0]['q']
            for key,metric in (('q_return','discounted_suffix_return'),('q_success','legal_double_capture')):
                diff=np.mean([group[a][r][metric]-group[b][r][metric] for r in range(3)])
                category=2 if qdiff==0 or diff==0 else 0 if qdiff*diff>0 else 1
                comparison[key][category]+=1
        agreements.append(comparison)
        details.append(dict(episode_hash=identity,chosen=selected,gains=gains,illegal=illegal,
            original_successes=[group[0][r]['legal_double_capture'] for r in range(3)],
            candidate_count=len(group),original_r0_success=group[0][0]['legal_double_capture'],agreement=comparison))
    if not details:
        raise ValueError('Empty scientific sample')
    delta=np.asarray(contributions,dtype=float)
    per_state=delta.mean(axis=1)
    rng=np.random.default_rng(49020261024)
    boot=np.empty(draws)
    for start in range(0,draws,1000):
        indices=rng.integers(0,len(details),(min(1000,draws-start),len(details)))
        boot[start:start+len(indices)]=per_state[indices].mean(axis=1)
    ci=np.quantile(boot,[.025,.975]).tolist()
    streams=delta.mean(axis=0).tolist()
    illegal=float(np.mean([d['illegal'] for d in details]))
    agreement={}
    for key in ('q_return','q_success'):
        fractions=[np.asarray(a[key],float)/sum(a[key]) for a in agreements if sum(a[key])]
        agreement[key]=dict(labels=['concordant','discordant','tied'],state_mean=np.mean(fractions,axis=0).tolist()
                            if fractions else None,states_with_pairs=len(fractions))
    return dict(delta=float(per_state.mean()),ci95=ci,per_stream_delta=streams,illegal_delta=illegal,
        passed=gate(float(per_state.mean()),ci[0],streams,illegal),conditional_states=len(details),
        total_physical=total_physical,all_scene_delta=float(per_state.sum()/total_physical),
        gained_stream_comparisons=int((delta>0).sum()),lost_stream_comparisons=int((delta<0).sum()),
        mean_gained_scenes_per_stream=float((delta>0).sum()/3),mean_lost_scenes_per_stream=float((delta<0).sum()/3),
        agreement=agreement,details=details,bootstrap_draws=draws,bootstrap_seed=49020261024)


def analyze(args,bank,roots,suffixes):
    verify(args)
    if roots!=tasks(args,bank):
        raise ValueError('Root state set changed')
    if not args.smoke and not locked_read(args.output/'audit_result.json')['passed']:
        raise ValueError('Engineering audit must pass before analysis')
    expected=branch_tasks(args,roots)
    if {x['key'] for x in suffixes}!={x['key'] for x in expected} or len(suffixes)!=len(expected):
        raise ValueError('Missing/duplicate suffix pairing')
    groups={};flat=[];parity=0;calls=dict(pursuer=len(roots),evader=0)
    for task in expected:
        row=read_task(task);root=read_task(task['root_task'])
        identity=task['root_task']['row']['spec']['episode_hash']
        candidate=root['candidates'][task['candidate']]
        metrics=dict(row['metrics'],ordinal=candidate['ordinal'],q=candidate['q'])
        groups.setdefault(identity,{}).setdefault(task['candidate'],{})[task['stream']]=metrics
        parity+=row['historical_exact']
        for role in calls:calls[role]+=row['calls'][role]
        flat.append(dict(episode_hash=identity,candidate=task['candidate'],stream=task['stream'],**metrics))
    if not args.smoke and parity!=99:
        raise ValueError('Missing historical exact replays')
    stats=compute_statistics(groups,bank['total_physical'])
    stats.update(smoke=args.smoke,passed=stats['passed'] and not args.smoke,historical_exact=parity,
        root_searches=len(roots),suffix_requests=len(expected),committed_suffixes=len(suffixes),mcts_calls=calls,
        recommended='old0:novalue',training_episodes=0,optimizer_updates=0,
        interpretation='Historical mechanism screen; future-informed selector; not deployment gain',
        reason='ranking_calibration_signal' if stats['passed'] and not args.smoke else 'smoke_only' if args.smoke else 'insufficient_ranking_signal')
    lock_json(args.output/'paired_results.json',flat)
    lock_json(args.output/'statistics.json',stats)
    return stats


def audit_pair(task):
    spec,role,opponent,cfg,reg=task['spec'],task['role'],task['opponent'],task['config'],task['registry']
    requested=dict(search_budget=cfg['budget'],search_only_budget=cfg['opponent_budget'],search_depth=100,
        beta_policy=.5,beta_value=0. if role else .1)
    old_trace=[];new_trace=[];scientific=[]
    for private,trace in ((False,old_trace),(True,new_trace)):
        p,state=o.restore_scenario(spec);p.clear_closed_loop_cache()
        if private:
            own=make_solver(p,reg['O'],'Search + Learning',role,cfg,True,spec['search_seeds'][role])
            other=make_solver(p,reg['L'],opponent,1-role,cfg,True,spec['search_seeds'][1-role],beta_value=.1)
            if opponent=='Policy Only':
                masked_other=other
            else:masked_other=o.RoleMaskedSolver(other,o.TEAMS[1-role])
            team=o.TeamSolver(masked_other,o.RoleMaskedSolver(own,o.TEAMS[role])) if role else \
                o.TeamSolver(o.RoleMaskedSolver(own,o.TEAMS[role]),masked_other)
        else:
            own=f.make_solver(p,reg['O']['models'],'Search + Learning',role,requested,spec['search_seeds'][role])
            opp_requested=dict(requested,beta_value=.1)
            other=f.make_solver(p,reg['L']['models'],opponent,1-role,opp_requested,spec['search_seeds'][1-role])
            # Same frozen original opponent configuration, including value .1.
            team=o.TeamSolver(other,own) if role else o.TeamSolver(own,other)
        stats=h.initial_stats()
        while not p.is_terminal(state):
            action=team.policy(p,state);nxt=h.advance(p,state,action,stats)
            trace.append(dict(state=state.tolist(),action=action.tolist(),next_state=nxt.tolist()));state=nxt
        scientific.append(h.science(h.metrics(p,state,stats)))
    if scientific[0]!=scientific[1] or o.json_hash(old_trace)!=o.json_hash(new_trace):
        raise ValueError(f'Original/private episode mismatch {role} {opponent} {spec["layout"]}')
    return dict(role=role,opponent=opponent,layout=spec['layout'],exact=True,episodes=2)


def audit(args,bank,pool,status):
    path=args.output/'audit_result.json'
    if path.with_suffix('.lock.json').exists():return locked_read(path)
    items=[dict(spec=bank['engineering']['audit_'+layout][0],role=role,opponent=opponent,
        config=config(args),registry=registry(args)) for layout in ('four','none') for role in (0,1) for opponent in o.OPPONENTS]
    records=[]
    for item in items:
        identity=o.json_hash(item);file=args.output/'audit_pairs'/(identity+'.json')
        if file.with_suffix('.lock.json').exists():records.append(locked_read(file))
    missing=[item for item in items if not (args.output/'audit_pairs'/(o.json_hash(item)+'.lock.json')).exists()]
    iterator=pool.imap_unordered(audit_pair,missing,chunksize=1) if pool else map(audit_pair,missing)
    for row in iterator:
        item=next(t for t in items if t['role']==row['role'] and t['opponent']==row['opponent'] and t['spec']['layout']==row['layout'])
        lock_json(args.output/'audit_pairs'/(o.json_hash(item)+'.json'),row);records.append(row)
        status('audit',completed=len(records),total=12)
    probes=[]
    for layout in ('four','none'):
        p,state=o.restore_scenario(bank['engineering']['audit_'+layout][0])
        for role in (0,1):
            for phase in (0,1):
                current=state.copy()
                if phase:current[[p.active_idxs[0],p.active_idxs[2]],0]=0.
                reg=registry(args)
                plain=make_solver(p,reg['O'],'Search + Learning',role,config(args),False)
                export=make_solver(p,reg['O'],'Search + Learning',role,config(args),True)
                initial=plain.get_rng_state(p)
                a=plain.policy(p,current,team=o.TEAMS[role]);b=export.policy(p,current,team=o.TEAMS[role])
                if not np.array_equal(a,b) or plain.get_rng_state(p)!=export.get_rng_state(p):
                    raise ValueError('Root export consumed RNG or changed action')
                if role==1:selected_candidates(export.last_result.root_audit,p,current)
                export.set_rng_state(p,initial)
                if not np.array_equal(a,export.policy(p,current,team=o.TEAMS[role])):
                    raise ValueError('RNG roundtrip changed action')
                probes.append(dict(layout=layout,role=role,phase=phase,exact=True))
    return lock_json(path,dict(passed=True,original_pairs=records,rng_action_probes=probes,engineering_episodes=24))


def report(args,bank,stats):
    passed=stats['passed'];ci=stats['ci95']
    text=f'''# 根动作评分核验结果

状态：完成；推荐继续为 old0:novalue。新增训练和优化器更新均为 0。

历史首捕后条件样本 {stats['conditional_states']} 个，冻结根搜索 {stats['root_searches']} 次，提交后缀 {stats['committed_suffixes']} 条，历史精确复现 {stats['historical_exact']} 条。

留一随机流评估净增 **{stats['delta']*100:+.2f} 个百分点**，普通配对 95% 区间 [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]。全 {stats['total_physical']} 场景口径 {stats['all_scene_delta']*100:+.2f} 点。

三个留出流增益：{[round(x*100,3) for x in stats['per_stream_delta']]} 点；自身非法变化 {stats['illegal_delta']*100:+.2f} 点。

轮换留出评估合计挽回 {stats['gained_stream_comparisons']} 个状态×流、丢失 {stats['lost_stream_comparisons']} 个；先在状态内对三个流平均，不能将这些当作独立物理场景。

投入评分校准门槛：{'通过，仅建议另行制定评分器计划' if passed else '未通过，本轮结束'}。原因：{stats['reason']}。

底层科学 MCTS 调用：{stats['mcts_calls']}；另列工程复现 24 局。根候选筛选和反事实后缀只是离线机制实验，未产生可部署收益或新模型。

本指标借助同一状态的另两个未来续跑选择动作，不能作为可部署选择器或新场景确认。仅研究首捕后一次动作、最多四个已有候选和固定 O 继续策略。未通过不能证明全局无更好动作。

Q 与实际折扣回报/合法双捕的两两排序比例（状态等权；顺序为一致、相反、平分）：{stats['agreement']}。

树内与真实闭环的时序、继续策略不同；排序不一致不单独证明随机 rollout 是根因。详细逐状态、逐候选、逐流结果见 paired_results.json 和 statistics.json，完整轨迹与原子缓存见 roots/、suffixes/。
'''
    (args.output/'REPORT.md').write_text(text)
    if passed:
        (args.output/'SCORER_NEXT_PLAN.md').write_text('''# 评分器下一轮草案（尚未启动）

本轮历史机制筛选达到研发投入门槛，尚无部署收益确认。

建议保持 O actor、环境、奖励、搜索预算不变，独立学习首捕后原根候选的合法完成概率与自身非法风险，以失败记 80 的完成成本作平分。输入只能来自当次部署可用观测与候选动作，不使用未来轨迹、对手身份或隐藏状态作为部署特征。

重新采集 O 实际搜索部署状态，用真实冻结对手闭环后果作为标签；按物理初态先划分训练、验证、确认，所有候选和随机流留在同一分区，先冻结所有银行。旧诊断数据不作为独立确认。

采用唯一候选重排规则与固定回退 O，保留 O 和原 l45 关闭旧价值两参考。完整计划需另行明确新增采样/训练上限、离线拟合与排序门槛、全局三对手验证、两参考独立配对确认、逃逸回归与单进程耗时。只在预登记门槛通过后考虑替换部署。本轮不自动采样或训练。
''')


def replay(args,bank):
    roots=tasks(args,bank);suffixes=branch_tasks(args,roots)
    counts=dict(roots=0,suffixes=0,historical_exact=0)
    for task in roots:read_task(task);counts['roots']+=1
    for task in suffixes:
        row=read_task(task);root=read_task(task['root_task'])
        p,_=o.restore_scenario(task['root_task']['row']['spec'])
        dims=[j for robot in o.TEAMS[1] for j in p.action_idxs[robot]]
        if not np.array_equal(np.asarray(row['trajectory'][0]['state']),root['state']):
            raise ValueError('Suffix initial state differs')
        if not np.array_equal(np.asarray(row['trajectory'][0]['action'])[dims],root['candidates'][task['candidate']]['action'][dims]):
            raise ValueError('Intervention action differs')
        for i,step in enumerate(row['trajectory']):
            if i and step['state']!=row['trajectory'][i-1]['next_state']:
                raise ValueError('Broken trajectory')
            state=np.asarray(step['state']);action=np.asarray(step['action'])
            for robot in range(p.num_robots):
                if not p.is_active(state,robot) and np.any(action[p.action_idxs[robot]]!=0):
                    raise ValueError('Inactive action was executed')
        counts['suffixes']+=1;counts['historical_exact']+=row['historical_exact']
    lock_json(args.output/'replay.json',dict(passed=True,**counts))
    return counts


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',nargs='?',default='pipeline',choices=['pipeline','prepare','audit','extract','branch','analyze','report','watch','replay'])
    p.add_argument('--workers',type=int,default=80)
    p.add_argument('--output',type=Path)
    p.add_argument('--smoke',action='store_true');p.add_argument('--once',action='store_true')
    args=p.parse_args(argv)
    if args.workers<1:p.error('workers must be positive')
    args.output=(args.output or (SMOKE if args.smoke else DEFAULT)).resolve()
    return args


def watch(args):
    while True:
        row=o.read_json(args.output/'status.json')
        row['pid_alive']=(Path('/proc')/str(row['pid'])).exists()
        live_workers=0
        if row['pid_alive']:
            for p in Path('/proc').glob('[0-9]*'):
                try:
                    child=f'PPid:\t{row["pid"]}\n' in (p/'status').read_text()
                    if child and b'spawn_main' in (p/'cmdline').read_bytes():live_workers+=1
                except (FileNotFoundError,ProcessLookupError,PermissionError):pass
        row['live_workers']=live_workers
        row['committed_roots']=len(list((args.output/'roots').glob('*.lock.json')))
        row['committed_suffixes']=len(list((args.output/'suffixes').glob('*.lock.json')))
        print(row,flush=True)
        if args.once or row['phase'] in ('complete','failed'):return row
        time.sleep(5)


def main(argv=None):
    args=parse_args(argv)
    if args.stage=='watch':return watch(args)
    args.output.mkdir(parents=True,exist_ok=True)
    writer=(args.output/'run.lock').open('w')
    fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();current={};mutex=threading.Lock();stop=threading.Event()
    def status(phase,**fields):
        with mutex:
            previous=current.copy()
            same=previous.get('phase')==phase
            phase_started=previous.get('phase_started',time.time()) if same else time.time()
            initial=previous.get('initial_completed',fields.get('completed',0)) if same else fields.get('completed',0)
            current.clear();current.update(phase=phase,pid=os.getpid(),workers=args.workers,updated_unix=time.time(),
                phase_started=phase_started,initial_completed=initial,elapsed_seconds=time.time()-started,**fields)
            if 'completed' in fields:
                rate=(fields['completed']-initial)/max(time.time()-phase_started,1e-9)
                current.update(rate=rate,eta_seconds=(fields['total']-fields['completed'])/rate if rate else None)
            o.write_json(args.output/'status.json',current)
        print(dict(current),flush=True)
    def heartbeat():
        while not stop.wait(5):
            with mutex:
                if current:
                    current['updated_unix']=time.time();o.write_json(args.output/'status.json',current)
    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        status('prepare');bank=prepare(args)
        if args.stage=='prepare':status('prepared');return
        if args.stage in ('report','replay','analyze'):
            roots=tasks(args,bank);suffixes=branch_tasks(args,roots)
            if args.stage=='replay':result=replay(args,bank)
            else:
                result=analyze(args,bank,roots,suffixes) if args.stage=='analyze' else locked_read(args.output/'statistics.json')
                report(args,bank,result)
            status('complete',stage=args.stage);return result
        with mp.get_context('spawn').Pool(args.workers) as pool:
            audit(args,bank,pool,status)
            if args.stage=='audit':status('audited');return
            root_tasks=tasks(args,bank)
            run_tasks(args,'extract',root_tasks,extract_one,pool,status)
            if args.stage=='extract':status('extracted');return
            suffix_tasks=branch_tasks(args,root_tasks)
            run_tasks(args,'branch',suffix_tasks,branch_one,pool,status)
            if args.stage=='branch':status('branched');return
        result=analyze(args,bank,root_tasks,suffix_tasks)
        report(args,bank,result);replay(args,bank);verify(args)
        lock_json(args.output/'results.json',dict(statistics_sha=o.sha(args.output/'statistics.json'),
            replay_sha=o.sha(args.output/'replay.json'),recommended='old0:novalue',passed=result['passed'],reason=result['reason']))
        status('complete',reason=result['reason'],recommended='old0:novalue',passed=result['passed'])
        return result
    except BaseException as exc:
        status('failed',reason=repr(exc));raise
    finally:
        stop.set();writer.close()


if __name__=='__main__':main()
