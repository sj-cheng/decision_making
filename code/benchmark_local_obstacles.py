"""Paired evaluation, profiling and pilot gates for local obstacle RL.

Global-state references are explicitly labelled and excluded from pilot gates.
Episode specs include map, exact speed limits and initial state. Seeds used
here are in separate namespaces from all training streams.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import csv
from dataclasses import asdict
import hashlib
from itertools import permutations
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from local_observation import PROTOCOL, TEAMS, make_episode, observe
from learning.local_rl import PolicyBundle
from solvers.belief_mcts import LocalController, BeliefMCTS, public_model, enable_fast_model

LOCAL_METHODS=('legacy','distilled','ppo','search_distilled','search_ppo','scratch','no_belief','local_search_only','oracle_search','mappo_v4')


def build_controller(spec,bundle,role,method,seed,budget,depth,beta_value=0.,time_budget_ms=None):
    if method in ('search_distilled','search_ppo','no_belief','local_search_only'):
        return BeliefMCTS(public_model(spec.obstacles),bundle,role,seed,simulations=budget,depth=depth,
                           beta_policy=0. if method=='local_search_only' else .5,beta_value=beta_value,
                           without_belief=True if method=='no_belief' else None,time_budget_ms=time_budget_ms)
    return LocalController(public_model(spec.obstacles),bundle,role,seed)


def evaluate_episode(spec,tested,reference,role,method='ppo',opponent_method='distilled',budget=1000,depth=8,beta_value=0.,time_budget_ms=None):
    from train_search_rl import _valid_robot
    p,state=spec.restore(); enable_fast_model(p)
    controllers=[None,None]
    controllers[role]=build_controller(spec,tested,role,method,spec.seed+17,budget,depth,beta_value,time_budget_ms)
    controllers[1-role]=build_controller(spec,reference,1-role,opponent_method,spec.seed+31,budget,depth)
    global_solver=None
    if method=='oracle_search':
        from benchmark_baselines import build_solver
        global_solver=build_solver(p,'Search Only',str(Path(__file__).resolve().parents[1]),budget)
        global_solver.solver_wrapper.set_seed(int(spec.seed)%2**32)
    elif method=='mappo_v4':
        from learning.mappo import MAPPOPolicySolver
        global_solver=MAPPOPolicySolver(p,path=str(Path(__file__).resolve().parents[1]/'current/baselines/mappo_50k_v4.pt'))
    captures=0; first_capture=None; capture_time=None; area=0.; returns=np.zeros(2)
    latency=[[],[]]; transitions=[0,0]; simulations=[0,0]; errors=[]; steps=0
    for step in range(80):
        if p.is_terminal(state):
            break
        joint=np.zeros((8,1)); area+=p.active_evader_count(state)*p.dt/2.
        for team in range(2):
            started=time.perf_counter()
            observation=observe(p,state,team)
            if team==role and global_solver is not None:
                controllers[team].prepare(observation)
                global_action=global_solver.policy(p,state,team=TEAMS[team])
                action=np.array([global_action[p.action_idxs[r],0] for r in TEAMS[team]])
            else:
                action,_=controllers[team].act(observation)
            latency[team].append((time.perf_counter()-started)*1000.)
            transitions[team]+=controllers[team].last_stats.get('transitions',0)
            simulations[team]+=controllers[team].last_stats.get('simulations',0)
            if team==role and method=='oracle_search':simulations[team]+=budget
            for local,robot in enumerate(TEAMS[team]):
                joint[p.action_idxs[robot],0]=action[local]
            if team==role:
                belief=controllers[team].belief
                slots=np.flatnonzero(belief.existence>.5)
                truth=np.array([state[p.state_idxs[r],0] for r in TEAMS[1-team] if p.is_active(state,r)])
                if len(slots)==len(truth) and len(slots):
                    means=belief.particles[:,slots,:2].mean(0)
                    ordering=min(permutations(range(len(truth))),key=lambda perm: sum(np.linalg.norm(means[i]-truth[j]) for i,j in enumerate(perm)))
                    for i,j in enumerate(ordering):
                        errors.append((float(belief.age[slots[i]]),float(np.linalg.norm(means[i]-truth[j])),float(belief.particles[:,slots[i],:2].var(0).sum())))
        nxt=p.step(state,joint,p.dt); reward=p.transition_reward(state,nxt)
        returns+=[np.mean(reward[list(TEAMS[t]),0]) for t in range(2)]
        current=2-p.active_evader_count(nxt)
        if current>captures:
            if first_capture is None: first_capture=float(nxt[p.time_idx,0])
            if current==2: capture_time=float(nxt[p.time_idx,0])
            captures=current
        state=nxt; steps+=1
    illegal=[any(not _valid_robot(p,state,r) for r in TEAMS[t]) for t in range(2)]
    obstacle=[any(p.check_obstacle_collision(state,r) for r in TEAMS[t]) for t in range(2)]
    oob=[any(np.any(state[p.state_idxs[r],0]<p.state_lims[p.state_idxs[r],0]) or
             np.any(state[p.state_idxs[r],0]>p.state_lims[p.state_idxs[r],1]) for r in TEAMS[t]) for t in range(2)]
    row=dict(seed=spec.seed,map_hash=hashlib.sha256(json.dumps(spec.obstacles).encode()).hexdigest(),
             training_seed=tested.metadata.get('training_seed',0),
             information_contract='global_state_reference' if global_solver is not None else 'local_shared_known_map',
             episode_hash=spec.fingerprint,layout=spec.layout,role=role,method=method,opponent=opponent_method,
             protocol=PROTOCOL,budget=budget,time_budget_ms=time_budget_ms,steps=steps,double_capture=float(captures==2 and not any(illegal)),
             raw_double_capture=float(captures==2),
             legal_survival=float(captures<2 and state[p.time_idx,0]>=p.tf and not any(illegal)),
             capture_time=capture_time if not any(illegal) else None,first_capture_time=first_capture,evader_alive_integral=area,
             pursuer_return=float(returns[1]),evader_return=float(returns[0]),
             pursuer_illegal=float(illegal[1]),evader_illegal=float(illegal[0]),
             pursuer_obstacle_collision=float(obstacle[1]),evader_obstacle_collision=float(obstacle[0]),
             pursuer_out_of_bounds=float(oob[1]),evader_out_of_bounds=float(oob[0]),
             tested_latency_ms=float(np.mean(latency[role])) if latency[role] else 0.,
             tested_latency_p95_ms=float(np.percentile(latency[role],95)) if latency[role] else 0.,
             tested_transitions=transitions[role],opponent_transitions=transitions[1-role],
             tested_simulations=simulations[role],opponent_simulations=simulations[1-role],
             reacquisition_count=len(controllers[role].belief.reacquisition_times),
             reacquisition_mean=float(np.mean(controllers[role].belief.reacquisition_times)) if controllers[role].belief.reacquisition_times else None)
    for label,lo,hi in [('visible',0,0),('age1_3',1,3),('age4_10',4,10),('age11_plus',11,float('inf'))]:
        selected=[e for e in errors if lo<=e[0]<=hi]
        row['tracking_'+label+'_error']=float(np.mean([e[1] for e in selected])) if selected else None
        row['tracking_'+label+'_variance']=float(np.mean([e[2] for e in selected])) if selected else None
        row['tracking_'+label+'_count']=len(selected)
    return row


def aggregate(rows):
    result={}
    for key in rows[0] if rows else []:
        values=[r[key] for r in rows if r[key] is not None and isinstance(r[key],(int,float))]
        if values:
            values=np.asarray(values,float)
            result[key]=dict(mean=float(values.mean()),n=len(values),ci95=float(1.96*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None)
    return result


def hierarchical_comparison(rows,repetitions=1000,grouped=False):
    """Paired bootstrap: training seeds, then maps, then episode pairs."""
    if not grouped:
        result={}
        for layout,opponent in sorted({(r['layout'],r['opponent']) for r in rows}):
            result[f'{layout}/{opponent}']=hierarchical_comparison([r for r in rows if r['layout']==layout and r['opponent']==opponent],repetitions,True)
        return result
    report={};rng=np.random.default_rng(7391)
    for baseline,candidate in [('distilled','ppo'),('search_distilled','search_ppo')]:
        for role in range(2):
            metric='double_capture' if role==1 else 'legal_survival'
            old={(r['training_seed'],r['episode_hash'],r['opponent']):r for r in rows if r['method']==baseline and r['role']==role}
            groups={}
            for r in rows:
                key=(r['training_seed'],r['episode_hash'],r['opponent'])
                if r['method']==candidate and r['role']==role and key in old:
                    groups.setdefault(r['training_seed'],{}).setdefault(r['map_hash'],[]).append(r[metric]-old[key][metric])
            if not groups: continue
            seeds=list(groups);draws=[]
            for _ in range(repetitions):
                seed_values=[]
                for seed in rng.choice(seeds,len(seeds),replace=True):
                    maps=list(groups[seed]);map_values=[]
                    for map_hash in rng.choice(maps,len(maps),replace=True):
                        values=groups[seed][map_hash]
                        map_values.append(np.mean(rng.choice(values,len(values),replace=True)))
                    seed_values.append(np.mean(map_values))
                draws.append(np.mean(seed_values))
            point=float(np.mean([np.mean([np.mean(v) for v in maps.values()]) for maps in groups.values()]))
            report[f'{candidate}_minus_{baseline}/role{role}']=dict(success_delta=point,
                 ci95=[float(v) for v in np.percentile(draws,[2.5,97.5])],training_seeds=len(seeds),
                 note='Single training seed cannot quantify training-seed uncertainty.' if len(seeds)<2 else '')
    return report


def validate_policy(bundle,anchor,role,episodes,training_seed):
    from train_search_rl import episode_seed
    rows=[]
    for layout_index,layout in enumerate(('two','four','random')):
        for index in range(episodes):
            seed=episode_seed(training_seed,80+layout_index,role,index)
            rows.append(evaluate_episode(make_episode(seed,layout),bundle,anchor,role,method='ppo'))
    score=np.mean([r['double_capture'] if role==1 else r['legal_survival'] for r in rows])
    # Task success dominates a bounded timing tie-break. Invalid endings do
    # not get rewarded for producing a short episode.
    if role==1:
        success=[r['capture_time'] for r in rows if r['double_capture']]
        timing=1.-np.mean(success)/80. if success else 0.
    else:
        timing=np.mean([r['evader_alive_integral'] if not r['evader_illegal'] else 0. for r in rows])/80.
    return dict(score=float(score+1e-4*timing),summary=aggregate(rows),episodes=rows)


def profile(args):
    from train_search_rl import episode_seed,write_json,implementation_hash
    bundle=PolicyBundle(args.model_dir,args.iteration,args.scratch)
    initial=Path(args.output)/'profile_initial.pt';bundle.save(initial)
    tasks=[]
    for i in range(100):
        spec=make_episode(episode_seed(args.seed,90,0,i),('two','four','random')[i%3])
        tasks.append(dict(spec=asdict(spec),checkpoint=str(initial),reference=str(initial),role=i%2,
                          method='legacy',opponent='distilled',budget=args.budget,depth=args.depth))
    started=time.perf_counter(); rows=[]
    if args.workers>1:
        with mp.get_context('spawn').Pool(args.workers) as pool:
            for row in pool.imap(evaluation_task,tasks):
                rows.append(row)
                if len(rows)%10==0:print(f'profile policy {len(rows)}/100',flush=True)
    else:
        rows=[evaluation_task(task) for task in tasks]
    policy_seconds=time.perf_counter()-started
    search=[]; transitions=[]
    serial_count=10 if args.workers>1 else 100
    for i in range(serial_count):
        spec=make_episode(episode_seed(args.seed,91,0,i),('two','four','random')[i%3])
        p,state=spec.restore(); enable_fast_model(p)
        controller=BeliefMCTS(public_model(spec.obstacles),bundle,i%2,spec.seed,simulations=args.budget,depth=args.depth)
        tick=time.perf_counter(); controller.act(observe(p,state,i%2))
        search.append(time.perf_counter()-tick); transitions.append(controller.last_stats['transitions'])
        if (i+1)%5==0: print(f'profile search {i+1}/100 mean={np.mean(search):.3f}s',flush=True)
    serial_search=search.copy();parallel_started=time.perf_counter()
    if args.workers>1:
        tasks=[]
        for i in range(serial_count,100):
            spec=make_episode(episode_seed(args.seed,91,0,i),('two','four','random')[i%3])
            tasks.append(dict(spec=asdict(spec),checkpoint=str(initial),role=i%2,budget=args.budget,depth=args.depth))
        with mp.get_context('spawn').Pool(args.workers) as pool:
            for result in pool.imap(profile_search_task,tasks):
                search.append(result['seconds']);transitions.append(result['transitions'])
                if len(search)%5==0:print(f'profile search {len(search)}/100',flush=True)
    parallel_seconds=time.perf_counter()-parallel_started
    typical_steps=float(np.mean([r['steps'] for r in rows])); mean_search=float(np.mean(search))
    per_search=parallel_seconds/(100-serial_count) if args.workers>1 else mean_search
    estimated=dict(distill_seconds=2*args.decisions*per_search,
                   ppo_seconds=2*args.episodes*(policy_seconds/100.+.1*typical_steps*per_search*args.opponent_budget/args.budget))
    result=dict(protocol=PROTOCOL,implementation_hash=implementation_hash(),policy_episodes=100,policy_seconds=policy_seconds,
                search_decisions=100,search_seconds_mean=mean_search,search_seconds_p95=float(np.percentile(search,95)),
                serial_search_decisions=serial_count,serial_search_seconds_mean=float(np.mean(serial_search)),
                parallel_search_seconds=parallel_seconds,
                search_transitions_mean=float(np.mean(transitions)),workers=args.workers,
                estimated_lower_bound=estimated,estimate_caveat='Excludes validation, fitting, IPC and CPU contention; measure actual parallel throughput during training.',
                cuda_available=torch.cuda.is_available(),policy_summary=aggregate(rows))
    write_json(Path(args.output)/'profile.json',result); print(json.dumps(result,indent=2),flush=True)
    return result


def profile_search_task(task):
    from local_observation import EpisodeSpec
    torch.set_num_threads(1)
    spec=EpisodeSpec(**task['spec']);bundle=PolicyBundle.load(task['checkpoint'])
    p,state=spec.restore();enable_fast_model(p)
    ctrl=BeliefMCTS(public_model(spec.obstacles),bundle,task['role'],spec.seed,simulations=task['budget'],depth=task['depth'])
    tick=time.perf_counter()
    try:ctrl.act(observe(p,state,task['role']))
    except RuntimeError as error:raise RuntimeError(f'Profile seed={spec.seed} layout={spec.layout} role={task["role"]}: {error}') from error
    return dict(seconds=time.perf_counter()-tick,transitions=ctrl.last_stats['transitions'])


def evaluation_task(task):
    from local_observation import EpisodeSpec
    from train_search_rl import worker_init
    worker_init()
    bundle=PolicyBundle.load(task['checkpoint'])
    reference=PolicyBundle.load(task['reference'])
    return evaluate_episode(EpisodeSpec(**task['spec']),bundle,reference,task['role'],task['method'],
                            task['opponent'],task['budget'],task['depth'],task.get('beta_value',0.),task.get('time_budget_ms'))


def run_evaluation(tasks,output,workers,signature):
    from train_search_rl import write_json,worker_init
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    config=output/'config.json'
    if config.exists() and json.loads(config.read_text())!=signature:
        raise ValueError('Evaluation output contains a different protocol/checkpoint/configuration')
    write_json(config,signature)
    chunks=output/'chunks'; chunks.mkdir(exist_ok=True)
    rows=[]; pending=[]
    for task in tasks:
        key=hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest()
        path=chunks/(key+'.json')
        if path.exists(): rows.append(json.loads(path.read_text()))
        else: pending.append((task,path))
    if workers>1 and pending:
        with mp.get_context('spawn').Pool(workers,initializer=worker_init) as pool:
            for (_,path),row in zip(pending,pool.imap(evaluation_task,[t for t,_ in pending])):
                write_json(path,row); rows.append(row)
                if len(rows)%20==0 or len(rows)==len(tasks):
                    print(f'evaluate completed={len(rows)}/{len(tasks)} method={row["method"]} role={row["role"]}',flush=True)
    else:
        for task,path in pending:
            row=evaluation_task(task); write_json(path,row); rows.append(row)
    rows.sort(key=lambda r:(r['method'],r['opponent'],r['role'],r['layout'],r['seed']))
    if rows:
        with (output/'episodes.csv').open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    grouped={}
    for row in rows:
        key=f'{row["method"]}/{row["opponent"]}/role{row["role"]}/{row["layout"]}'
        grouped.setdefault(key,[]).append(row)
    write_json(output/'summary.json',{key:aggregate(value) for key,value in grouped.items()})
    write_json(output/'paired_bootstrap.json',hierarchical_comparison(rows))
    return rows


def checkpoint_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pilot(args):
    from train_search_rl import episode_seed,write_json
    out=Path(args.output); distilled=out/'distilled.pt'; refined=out/'export/policy.pt'
    tasks=[]
    for layout_index,layout in enumerate(('two','four','random')):
        for index in range(args.pilot_episodes):
            spec=make_episode(episode_seed(args.seed,100+layout_index,0,index),layout)
            for role in range(2):
                for method,path in [('distilled',distilled),('ppo',refined),('search_distilled',distilled),('search_ppo',refined)]:
                    tasks.append(dict(spec=asdict(spec),checkpoint=str(path),reference=str(distilled),role=role,method=method,
                                      opponent='distilled',budget=args.budget,depth=args.depth))
    from train_search_rl import implementation_hash
    signature=dict(protocol=PROTOCOL,implementation_hash=implementation_hash(),seed=args.seed,budget=args.budget,depth=args.depth,episodes=args.pilot_episodes,
                   distilled=checkpoint_hash(distilled),refined=checkpoint_hash(refined))
    rows=run_evaluation(tasks,out/'pilot',args.workers,signature)
    report={'roles':{},'note':'Small-sample continuation gate, not evidence of statistical significance.'}
    gains=[]; safe=True
    for role in range(2):
        comparisons={}
        for baseline,candidate in [('distilled','ppo'),('search_distilled','search_ppo')]:
            old=[r for r in rows if r['role']==role and r['method']==baseline]
            new=[r for r in rows if r['role']==role and r['method']==candidate]
            metric='double_capture' if role==1 else 'legal_survival'
            delta=np.mean([r[metric] for r in new])-np.mean([r[metric] for r in old])
            own_illegal='pursuer_illegal' if role==1 else 'evader_illegal'
            illegal_delta=np.mean([r[own_illegal] for r in new])-np.mean([r[own_illegal] for r in old])
            timing_gain=0.
            if role==1:
                a=[r['capture_time'] for r in old if r['double_capture']]
                b=[r['capture_time'] for r in new if r['double_capture']]
                if a and b: timing_gain=1.-np.mean(b)/max(np.mean(a),1e-6)
            else:
                a=np.mean([r['evader_alive_integral'] if not r[own_illegal] else 0. for r in old])
                b=np.mean([r['evader_alive_integral'] if not r[own_illegal] else 0. for r in new])
                timing_gain=b/max(a,1e-6)-1.
            comparisons[candidate]=dict(success_delta=float(delta),illegal_delta=float(illegal_delta),timing_gain=float(timing_gain))
            if candidate=='search_ppo':
                gains.append(delta>=.05 or (delta>=-.02 and timing_gain>=.1))
                safe &= delta>=-.03 and illegal_delta<=.02
        report['roles'][str(role)]=comparisons
    report['continue_formal']=bool(any(gains) and safe)
    write_json(out/'pilot_gate.json',report); print(json.dumps(report,indent=2),flush=True)
    return report


def calibrate_value(args):
    """MC calibration followed by paired search ablation; never auto-enable."""
    from train_search_rl import episode_seed,rollout_task,write_json
    out=Path(args.output);bundle=PolicyBundle.load(args.input or out/'export/policy.pt')
    anchor=PolicyBundle.load(out/'distilled.pt');calibration=[];mc_pass=True
    for role in range(2):
        predictions=[];targets=[]
        for index in range(args.validation_episodes):
            result=rollout_task(dict(seed=episode_seed(args.seed,85,role,index),role=role,progress=1.,
                    kind='ppo',bundle=bundle.state_dict(),opponent=anchor.state_dict(),budget=args.budget,depth=args.depth))
            rows=result['rows'];discounted=0.;episode_targets=[]
            for row in reversed(rows):
                discounted=row['reward']+discounted*(1.-float(row['done']));episode_targets.append(discounted)
            with torch.no_grad():
                prediction=bundle.norms[role].denormalize(bundle.critics[role](torch.tensor(np.array([r['critic'] for r in rows]))).numpy())
            predictions.extend(prediction);targets.extend(reversed(episode_targets))
        mse=float(np.mean((np.asarray(predictions)-targets)**2));variance=float(np.var(targets))
        accepted=bool(mse<variance and np.isfinite(mse))
        calibration.append(dict(role=role,mse=mse,constant_mean_mse=variance,accepted=accepted));mc_pass &= accepted
    report=dict(mc=calibration,enabled=False)
    if mc_pass:
        candidate=bundle.copy();candidate.metadata['value_validated']=True
        changes=[]
        for role in range(2):
            old=[];new=[]
            for li,layout in enumerate(('two','four','random')):
                for index in range(args.validation_episodes):
                    spec=make_episode(episode_seed(args.seed,86+li,role,index),layout)
                    old.append(evaluate_episode(spec,bundle,anchor,role,'search_ppo',budget=args.budget,depth=args.depth))
                    new.append(evaluate_episode(spec,candidate,anchor,role,'search_ppo',budget=args.budget,depth=args.depth,beta_value=.1))
            metric='double_capture' if role==1 else 'legal_survival'
            delta=float(np.mean([r[metric] for r in new])-np.mean([r[metric] for r in old]))
            changes.append(dict(role=role,success_delta=delta))
        report['search_ablation']=changes
        if all(c['success_delta']>=0. for c in changes):
            report['enabled']=True;candidate.metadata['value_calibration']=report
            candidate.save(out/'value_calibrated.pt')
    write_json(out/'value_calibration.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint'); parser.add_argument('--reference')
    parser.add_argument('--aggregate',nargs='+',help='Merge evaluation directories from independent training seeds')
    parser.add_argument('--methods',nargs='+',choices=LOCAL_METHODS,default=['ppo','search_ppo'])
    parser.add_argument('--opponents',nargs='+',choices=['distilled','search_distilled','local_search_only'],default=['distilled','search_distilled'])
    parser.add_argument('--layouts',nargs='+',choices=['none','two','four','random'],default=['two','four','random'])
    parser.add_argument('--episodes',type=int,default=200); parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--budget',type=int,default=1000); parser.add_argument('--depth',type=int,default=8)
    parser.add_argument('--seed',type=int,default=0); parser.add_argument('--beta-value',type=float,default=0.)
    parser.add_argument('--time-budget-ms',type=float,help='End a tested local search after this elapsed budget (at least one simulation)')
    parser.add_argument('--output',required=True)
    args=parser.parse_args(); torch.set_num_threads(1)
    if min(args.episodes,args.workers,args.budget,args.depth)<=0:parser.error('episodes, workers, budget and depth must be positive')
    if args.time_budget_ms is not None and args.time_budget_ms<=0:parser.error('time-budget-ms must be positive')
    if args.aggregate:
        from train_search_rl import write_json
        rows=[]
        for directory in args.aggregate:
            for file in (Path(directory)/'chunks').glob('*.json'):
                rows.append(json.loads(file.read_text()))
        write_json(Path(args.output)/'paired_bootstrap.json',hierarchical_comparison(rows));return
    if not args.checkpoint or not args.reference:
        parser.error('--checkpoint and --reference are required unless --aggregate is used')
    from train_search_rl import episode_seed
    from train_search_rl import implementation_hash
    signature=vars(args).copy(); signature.pop('workers'); signature.update(protocol=PROTOCOL,implementation_hash=implementation_hash(),
                    checkpoint_hash=checkpoint_hash(args.checkpoint),reference_hash=checkpoint_hash(args.reference))
    manifest=[]; tasks=[]
    for li,layout in enumerate(args.layouts):
        for index in range(args.episodes):
            spec=make_episode(episode_seed(args.seed,110+li,0,index),layout); manifest.append(asdict(spec))
            for role in range(2):
                for method in args.methods:
                    for opponent in args.opponents:
                        source=args.reference if method in ('distilled','search_distilled') else args.checkpoint
                        tasks.append(dict(spec=asdict(spec),checkpoint=source,reference=args.reference,
                                          role=role,method=method,opponent=opponent,budget=args.budget,depth=args.depth,beta_value=args.beta_value,time_budget_ms=args.time_budget_ms))
    from train_search_rl import write_json
    write_json(Path(args.output)/'episode_manifest.json',manifest)
    run_evaluation(tasks,args.output,args.workers,signature)


if __name__=='__main__': main()
