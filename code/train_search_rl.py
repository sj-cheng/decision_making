"""Search distillation -> on-policy PPO -> export for local obstacle games.

Run ``python train_search_rl.py --help``. No legacy checkpoints are modified.
Every unit of work has a namespaced episode seed; ordered worker results and
saved RNG/optimizer states make continuation independent of worker count.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import copy
import io
import hashlib
from functools import lru_cache
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from torch.optim import Adam
from local_observation import (PROTOCOL, SCHEMA, TEAMS, OBS_DIM, EpisodeSpec,
                               curriculum_episode, observe)
from learning.local_rl import (PolicyBundle, advantages, atomic_torch_save,
                              critic_features, joint_log_prob, rng_state, restore_rng)
from solvers.belief_mcts import LocalController, BeliefMCTS, public_model, enable_fast_model

ROOT = Path(__file__).resolve().parents[1]
PPO_RESUME_KEYS = ['seed','episodes','depth','episodes_per_update','lr','critic_lr','clip','anchor',
                   'opponent_budget','eval_every','ppo_epochs','minibatch','entropy','scratch',
                   'without_belief','validation_episodes','input']


def episode_seed(training_seed, phase, role, index):
    return int(np.random.SeedSequence([int(training_seed), int(phase), int(role), int(index), 20261022]).generate_state(1)[0])


def worker_init():
    torch.set_num_threads(1)


def wire_weights(state):
    # Plain bytes use the Pool's pipes. Sending Tensor storages directly asks
    # PyTorch to open a Unix resource-sharing socket, unavailable in some
    # managed environments and unnecessary for these small models.
    stream=io.BytesIO(); torch.save(state,stream); return stream.getvalue()


def unwire_weights(state):
    return torch.load(io.BytesIO(state),map_location='cpu',weights_only=False) if isinstance(state,bytes) else state


def rollout_task(task):
    """Worker receives weights/config, builds sensor-only controllers locally."""
    seed = task['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    bundle = PolicyBundle(scratch=True); bundle.load_state_dict(unwire_weights(task['bundle']))
    opponent = PolicyBundle(scratch=True); opponent.load_state_dict(unwire_weights(task['opponent']))
    spec = curriculum_episode(seed, task['progress'])
    p, state = spec.restore(); enable_fast_model(p)
    role = task['role']; kind = task['kind']
    own_cls = BeliefMCTS if kind == 'distill' else LocalController
    own_kwargs = dict(simulations=task['budget'], depth=task['depth']) if kind == 'distill' else {}
    own = own_cls(public_model(spec.obstacles), bundle, role, seed, **own_kwargs)
    opp_cls = BeliefMCTS if task.get('search_opponent', False) else LocalController
    opp_kwargs = dict(simulations=task.get('opponent_budget', 200), depth=task['depth']) if task.get('search_opponent', False) else {}
    other = opp_cls(public_model(spec.obstacles), opponent, 1-role, seed+1, **opp_kwargs)
    rows = []; totals = np.zeros(2); elapsed = 0.; transitions = 0
    burn = int(np.random.default_rng(seed).integers(0, 25)) if kind == 'distill' else 0
    for step in range(80):
        if p.is_terminal(state):
            break
        obs = observe(p, state, role)
        if kind == 'distill' and step < burn:
            own.prepare(obs)
            chosen, raw, logp, features = own.actor_actions(obs, own.belief)
            info = dict(features=features)
        else:
            chosen, info = own.act(obs, stochastic=(kind == 'ppo'))
            transitions += own.last_stats.get('transitions', 0)
        opposing, _ = other.act(observe(p, state, 1-role))
        transitions += other.last_stats.get('transitions', 0)
        joint = np.zeros((8, 1))
        for t, actions in ((role, chosen), (1-role, opposing)):
            for local, robot in enumerate(TEAMS[t]):
                joint[p.action_idxs[robot], 0] = actions[local]
        nxt = p.step(state, joint, p.dt)
        reward = p.transition_reward(state, nxt)
        totals += [np.mean(reward[list(TEAMS[t]), 0]) for t in range(2)]
        if step >= burn:
            row = dict(obs=info['features'], active=obs.active.astype(np.float32), limits=obs.limits.astype(np.float32))
            if kind == 'distill':
                row['target'] = chosen.astype(np.float32)
            else:
                row.update(raw=info['raw'].astype(np.float32), old_logp=info['old_logp'],
                           critic=critic_features(p, state, info['features']),
                           reward=float(np.mean(reward[list(TEAMS[role]), 0])), done=bool(p.is_terminal(nxt)))
            rows.append(row)
        state = nxt; elapsed += p.dt
        if kind == 'distill' and len(rows) >= task.get('decisions', 16):
            break
    illegal = [any(not p.is_valid_robot(state, r) for r in TEAMS[t]) if hasattr(p, 'is_valid_robot') else
               any(not _valid_robot(p, state, r) for r in TEAMS[t]) for t in range(2)]
    stats = dict(seed=seed, role=role, steps=int(round(elapsed/p.dt)), double=float(p.active_evader_count(state) == 0 and not any(illegal)),
                 survival=float(p.active_evader_count(state)>0 and state[p.time_idx,0]>=p.tf and not any(illegal)),
                 illegal=illegal, returns=totals.tolist(), transitions=transitions,
                 reacquisition=own.belief.reacquisition_times, recoveries=own.belief.recovery_count)
    return dict(rows=rows, spec=asdict(spec), stats=stats)


def _valid_robot(p, state, robot):
    xy = state[p.state_idxs[robot], 0]
    lims = p.state_lims[p.state_idxs[robot]]
    return bool(np.all(xy >= lims[:,0]) and np.all(xy <= lims[:,1]) and not p.check_obstacle_collision(state, robot))


def run_tasks(pool, tasks):
    if pool:
        # Search opponents make episode costs very uneven; dispatch one at a
        # time rather than grouping an expensive episode with other episodes.
        return list(pool.map(rollout_task, tasks, chunksize=1))
    parent_rng=rng_state()
    try:
        return [rollout_task(t) for t in tasks]
    finally:
        restore_rng(parent_rng)


def append_manifest(directory, phase, results):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f'{phase}_episodes.jsonl').open('a') as out:
        for result in results:
            out.write(json.dumps(dict(spec=result['spec'], stats=result['stats'])) + '\n')


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp'); tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)); os.replace(tmp, path)


def configuration(args):
    config = vars(args).copy()
    config.update(protocol=PROTOCOL, observation_schema=SCHEMA, obs_dim=OBS_DIM,implementation_hash=implementation_hash())
    return config


def verify_resume(saved, args, keys, check_implementation=True):
    current = configuration(args)
    required = list(keys)+['protocol','observation_schema','obs_dim']
    if check_implementation:
        required.append('implementation_hash')
    for key in required:
        if saved.get(key) != current.get(key):
            raise ValueError(f'Resume configuration differs: {key}: {saved.get(key)!r} -> {current.get(key)!r}')


def fork_ppo(args):
    """Explicit continuation after a fix/batch change, retaining its provenance.

    Ordinary resume remains strict. A fork preserves optimizer, RNG, value
    normalization, opponent pool and episode counters, but is a new experiment
    segment, not an assertion of bit-exact continuation under changed code.
    """
    source=Path(args.fork_from).resolve();out=Path(args.output).resolve()
    if source.parent == out:
        raise ValueError('A fork needs a different output directory')
    fingerprint=hashlib.sha256(source.read_bytes()).hexdigest()
    manifest=out/'fork.json'
    if (out/'ppo_latest.pt').exists():
        if not manifest.exists() or json.loads(manifest.read_text())['source_sha256']!=fingerprint:
            raise ValueError('Output already contains a different experiment')
        return
    if list(out.glob('*.pt')):
        raise ValueError('Fork output already contains training artifacts')
    saved=torch.load(source,map_location='cpu',weights_only=False)
    verify_resume(saved['config'],args,[key for key in PPO_RESUME_KEYS if key not in ('episodes_per_update','minibatch')],
                  check_implementation=False)
    current=configuration(args)
    changes={key:dict(before=saved['config'].get(key),after=value) for key,value in current.items()
             if saved['config'].get(key)!=value}
    provenance=dict(source=str(source),source_sha256=fingerprint,source_counters=saved['counters'],
                    source_config=saved['config'],changes=changes,
                    note='Explicit fork; optimizer/RNG/history retained. Changed implementation or batch size is not exact resume.')
    for name in ('distilled.pt','best_role0.pt','best_role1.pt'):
        candidate=source.parent/name
        if candidate.exists():shutil.copy2(candidate,out/name)
    if not (out/'distilled.pt').exists() and args.stage=='ppo-pilot':
        raise ValueError('ppo-pilot requires distilled.pt next to the source checkpoint')
    saved['config']=current;saved['fork']=provenance;saved['needs_revalidation']=True
    write_json(manifest,provenance)
    atomic_torch_save(saved,out/'ppo_latest.pt')
    print(f'Forked PPO at episodes={saved["counters"]}; preserved source {source}',flush=True)


@lru_cache(maxsize=1)
def implementation_hash():
    digest=hashlib.sha256()
    for relative in ('local_observation.py','learning/local_rl.py','solvers/belief_mcts.py',
                     'train_search_rl.py','benchmark_local_obstacles.py','cpp/problems/example8.hpp','cpp/bindings.cpp','cpp/local_geometry.hpp',
                     'cpp/problems/problem.hpp','problems/example8.py','solvers/c_puct.py','benchmark_baselines.py'):
        digest.update(relative.encode());digest.update((ROOT/'code'/relative).read_bytes())
    from cpp.build import bindings
    digest.update(Path(bindings.__file__).read_bytes())
    return digest.hexdigest()


def distill(args, pool):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    progress_path = out / 'distill_progress.pt'
    if progress_path.exists():
        saved = torch.load(progress_path, weights_only=False)
        verify_resume(saved['config'], args, ['seed','decisions','budget','depth','model_dir','iteration','scratch','without_belief','distill_epochs','minibatch'])
        bundle = PolicyBundle(scratch=True); bundle.load_state_dict(saved['bundle'])
        data, counters, epochs_done = saved['data'], saved['counters'], saved['epochs_done']
        restore_rng(saved['rng'])
    else:
        bundle = PolicyBundle(args.model_dir, args.iteration, args.scratch)
        bundle.metadata['without_belief']=args.without_belief
        data = [[], []]; counters = [0, 0]; epochs_done = 0
    teacher_state = wire_weights(bundle.state_dict())
    while any(len(rows) < args.decisions for rows in data):
        tasks = []
        for _ in range(args.workers):
            role = min(range(2), key=lambda t: len(data[t]) + sum(x['decisions'] for x in tasks if x['role']==t))
            if len(data[role]) >= args.decisions:
                role = 1-role
            index = counters[role]; counters[role] += 1
            tasks.append(dict(seed=episode_seed(args.seed, 1, role, index), role=role,
                              progress=min(1., index*16/max(args.decisions,1)),
                              kind='distill', bundle=teacher_state, opponent=teacher_state,
                              budget=args.budget, depth=args.depth, decisions=min(16, args.decisions-len(data[role]))))
        results = run_tasks(pool, tasks)
        for task, result in zip(tasks, results):
            role = task['role']; data[role].extend(result['rows'][:args.decisions-len(data[role])])
        append_manifest(out, 'distill', results)
        atomic_torch_save(dict(config=configuration(args), bundle=bundle.state_dict(), data=data,
                               counters=counters, epochs_done=0, rng=rng_state()), progress_path)
        print(f'distill samples evader={len(data[0])}/{args.decisions} pursuer={len(data[1])}/{args.decisions}', flush=True)
    optimizers = [Adam(actor.parameters(), lr=3e-4) for actor in bundle.actors]
    if progress_path.exists():
        saved = torch.load(progress_path, weights_only=False)
        if 'optimizers' in saved:
            for optimizer, state in zip(optimizers, saved['optimizers']):
                optimizer.load_state_dict(state)
    for epoch in range(epochs_done, args.distill_epochs):
        losses = []
        for role in range(2):
            rows = data[role]
            for start in range(0, len(rows), args.minibatch):
                batch = rows[start:start+args.minibatch]
                for local, robot in enumerate(TEAMS[role]):
                    x = torch.as_tensor(np.array([r['obs'][local] for r in batch]), dtype=torch.float32)
                    limits = torch.as_tensor(np.array([r['limits'][local] for r in batch]), dtype=torch.float32)
                    target = torch.as_tensor(np.array([r['target'][local] for r in batch]), dtype=torch.float32)
                    active = torch.as_tensor(np.array([r['active'][local] for r in batch]), dtype=torch.float32)
                    dist = bundle.actors[robot].distribution(x, limits)
                    scale = ((limits[...,1]-limits[...,0])*.5).clamp_min(1e-6)
                    mean_loss = ((dist.mean-target)/scale).square().sum(-1)
                    target_std = .05*(limits[...,1]-limits[...,0])
                    var_loss = .05 * (dist.scale.log()-target_std.log()).square().sum(-1)
                    loss = ((mean_loss+var_loss)*active).sum()/active.sum().clamp_min(1.)
                    optimizers[robot].zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(bundle.actors[robot].parameters(), .5)
                    optimizers[robot].step(); losses.append(float(loss.detach()))
        bundle.refresh(); bundle.metadata.update(stage='distilled', training_seed=args.seed, source_iteration=args.iteration)
        atomic_torch_save(dict(config=configuration(args), bundle=bundle.state_dict(), data=data, counters=counters,
                               epochs_done=epoch+1, optimizers=[o.state_dict() for o in optimizers], rng=rng_state()), progress_path)
        print(f'distill epoch={epoch+1}/{args.distill_epochs} loss={np.mean(losses):.5f}', flush=True)
    bundle.save(out/'distilled.pt')
    return bundle


def ppo_update(bundle, anchor, role, episodes, actor_opt, critic_opt, args, progress):
    rows = [row for e in episodes for row in e['rows']]
    if not rows:
        raise RuntimeError('Empty PPO rollout')
    tensor = lambda name: torch.as_tensor(np.array([r[name] for r in rows]), dtype=torch.float32)
    obs, limits, raw, active = [tensor(k) for k in ('obs','limits','raw','active')]
    critic_x = tensor('critic'); old_logp = tensor('old_logp')
    norm = bundle.norms[role]; critic = bundle.critics[role]
    with torch.no_grad():
        predicted = norm.denormalize(critic(critic_x).numpy())
    offset = 0; advs = []; targets = []
    for episode in episodes:
        n = len(episode['rows']); chunk = episode['rows']
        adv, target = advantages([r['reward'] for r in chunk], predicted[offset:offset+n], [r['done'] for r in chunk])
        advs.extend(adv); targets.extend(target); offset += n
    adv = torch.tensor(np.asarray(advs), dtype=torch.float32)
    adv = (adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-6)
    old_mean, old_std = norm.mean, norm.std
    norm.update(np.asarray(targets))
    # Rebase the last layer so normalizer updates preserve physical values.
    last = critic.net[-1]; factor = old_std/norm.std
    with torch.no_grad():
        last.weight.mul_(factor); last.bias.mul_(factor).add_((old_mean-norm.mean)/norm.std)
    for parameter in (last.weight, last.bias):
        state = critic_opt.state.get(parameter, {})
        if 'exp_avg' in state:
            state['exp_avg'].mul_(factor); state['exp_avg_sq'].mul_(factor**2)
    returns = torch.tensor(norm.normalize(np.asarray(targets)), dtype=torch.float32)
    anchor_coef = args.anchor * max(0., 1.-progress/.2)
    losses = []; kls = []; stop = False
    for epoch in range(args.ppo_epochs):
        order = torch.randperm(len(rows))
        for start in range(0, len(rows), args.minibatch):
            idx = order[start:start+args.minibatch]
            logp = joint_log_prob(bundle.actors, obs[idx], raw[idx], limits[idx], active[idx], role)
            ratio = (logp-old_logp[idx]).exp()
            loss = -torch.minimum(ratio*adv[idx], ratio.clamp(1.-args.clip,1.+args.clip)*adv[idx]).mean()
            entropy = torch.zeros(()); penalty = torch.zeros(())
            for local, robot in enumerate(TEAMS[role]):
                dist = bundle.actors[robot].distribution(obs[idx,local], limits[idx,local])
                mask = active[idx,local]; denominator = mask.sum().clamp_min(1.)
                entropy += (dist.entropy().sum(-1)*mask).sum()/denominator
                with torch.no_grad():
                    reference = anchor.actors[robot].distribution(obs[idx,local],limits[idx,local]).mean
                scale = ((limits[idx,local,:,1]-limits[idx,local,:,0])*.5).clamp_min(1e-6)
                penalty += ((((dist.mean-reference)/scale).square().sum(-1))*mask).sum()/denominator
            loss = loss - args.entropy*entropy*.5 + anchor_coef*penalty*.5
            actor_opt.zero_grad(); loss.backward()
            parameters = [p for robot in TEAMS[role] for p in bundle.actors[robot].parameters()]
            torch.nn.utils.clip_grad_norm_(parameters, .5); actor_opt.step()
            value_loss = .5*(critic(critic_x[idx])-returns[idx]).square().mean()
            critic_opt.zero_grad(); value_loss.backward(); torch.nn.utils.clip_grad_norm_(critic.parameters(), .5); critic_opt.step()
            with torch.no_grad():
                kl = ((ratio-1)-(logp-old_logp[idx])).mean().item()
            losses.append((float(loss.detach()),float(value_loss.detach()))); kls.append(kl)
            if kl > .03:
                stop = True; break
        if stop:
            break
    bundle.refresh()
    return dict(actor_loss=float(np.mean(losses,axis=0)[0]), value_loss=float(np.mean(losses,axis=0)[1]),
                kl=float(np.mean(kls)), anchor=anchor_coef, samples=len(rows), reward=float(np.mean([r['reward'] for r in rows])))


def ppo(args, pool):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    path = out/'ppo_latest.pt'
    bundle = PolicyBundle(scratch=True)
    counters = [0,0]; history = []; best = [-float('inf')]*2; next_eval = [args.eval_every]*2
    if path.exists():
        saved = torch.load(path, weights_only=False)
        verify_resume(saved['config'], args, PPO_RESUME_KEYS)
        bundle.load_state_dict(saved['bundle']); counters=saved['counters']; history=saved['history']; best=saved['best']; next_eval=saved['next_eval']
        anchor = PolicyBundle(scratch=True); anchor.load_state_dict(saved['anchor'])
    else:
        if args.scratch:
            bundle = PolicyBundle(scratch=True)
        else:
            bundle = PolicyBundle.load(args.input or out/'distilled.pt')
        bundle.metadata['without_belief']=args.without_belief
        anchor = PolicyBundle.load(args.input) if args.scratch and args.input else bundle.copy()
    actor_opts = [Adam([p for robot in TEAMS[t] for p in bundle.actors[robot].parameters()], lr=args.lr) for t in range(2)]
    critic_opts = [Adam(bundle.critics[t].parameters(), lr=args.critic_lr) for t in range(2)]
    if path.exists():
        for optimizer, state in zip(actor_opts+critic_opts, saved['optimizers']):
            optimizer.load_state_dict(state)
        restore_rng(saved['rng'])
    fork_provenance=saved.get('fork') if path.exists() else None
    selector = np.random.default_rng(episode_seed(args.seed, 3, 0, 0))
    if path.exists():
        selector.bit_generator.state = saved['selector_rng']
        if saved.get('needs_revalidation'):
            # Old validation scores came from the previous implementation.
            # Compare the old best and current policy again on the same seeds.
            from benchmark_local_obstacles import validate_policy
            for role in range(2):
                previous=PolicyBundle.load(out/f'best_role{role}.pt') if (out/f'best_role{role}.pt').exists() else bundle
                old_validation=validate_policy(previous,anchor,role,args.validation_episodes,args.seed)
                validation=validate_policy(bundle,anchor,role,args.validation_episodes,args.seed)
                write_json(out/f'fork_previous_best_role{role}.json',old_validation)
                write_json(out/f'validation_role{role}_ep{counters[role]}.json',validation)
                chosen=previous if old_validation['score']>=validation['score'] else bundle
                chosen.save(out/f'best_role{role}.pt');best[role]=max(old_validation['score'],validation['score'])
                next_eval[role]=(counters[role]//args.eval_every+1)*args.eval_every
    else:
        from benchmark_local_obstacles import validate_policy
        for role in range(2):
            validation=validate_policy(bundle,anchor,role,args.validation_episodes,args.seed)
            best[role]=validation['score'];bundle.save(out/f'best_role{role}.pt')
            write_json(out/f'validation_role{role}_ep0.json',validation)
    updates_this_invocation=0
    while min(counters) < args.episodes:
        role = 0 if counters[0] <= counters[1] else 1
        count = min(args.episodes_per_update, args.episodes-counters[role])
        if count <= 0:
            role=1-role; count=min(args.episodes_per_update,args.episodes-counters[role])
        state = wire_weights(bundle.state_dict()); tasks=[]
        for offset in range(count):
            index=counters[role]+offset; draw=selector.random()
            opposing = history[int(selector.integers(len(history)))] if .45 <= draw < .9 and history else anchor.state_dict()
            tasks.append(dict(seed=episode_seed(args.seed,2,role,index),role=role,progress=index/max(args.episodes,1),
                              kind='ppo',bundle=state,opponent=wire_weights(opposing),search_opponent=draw>=.9,
                              opponent_budget=args.opponent_budget,budget=args.budget,depth=args.depth))
        started=time.perf_counter(); results=run_tasks(pool,tasks);collected=time.perf_counter()
        metrics=ppo_update(bundle,anchor,role,results,actor_opts[role],critic_opts[role],args,counters[role]/max(args.episodes,1))
        fitted=time.perf_counter()
        before=counters[role]; counters[role]+=count
        append_manifest(out,'ppo',results)
        if before//500 != counters[role]//500:
            history.append(copy.deepcopy(bundle.state_dict())); history=history[-8:]
        bundle.metadata.update(stage='ppo',training_seed=args.seed,episodes_per_role=counters.copy())
        if counters[role] >= next_eval[role] or counters[role] == args.episodes:
            from benchmark_local_obstacles import validate_policy
            validation=validate_policy(bundle,anchor,role,args.validation_episodes,args.seed)
            write_json(out/f'validation_role{role}_ep{counters[role]}.json',validation)
            score=validation['score']
            if score>best[role]:
                best[role]=score; bundle.save(out/f'best_role{role}.pt')
            next_eval[role]+=args.eval_every
        record=dict(role=role,episodes=counters.copy(),seconds=time.perf_counter()-started,
                    rollout_seconds=collected-started,update_seconds=fitted-collected,
                    episodes_per_second=count/max(collected-started,1e-9),workers=args.workers,
                    search_episodes=sum(t['search_opponent'] for t in tasks),**metrics)
        with (out/'train_metrics.jsonl').open('a') as log:
            log.write(json.dumps(record)+'\n')
        atomic_torch_save(dict(config=configuration(args),bundle=bundle.state_dict(),anchor=anchor.state_dict(),
                               counters=counters,history=history,best=best,next_eval=next_eval,
                               optimizers=[o.state_dict() for o in actor_opts+critic_opts],rng=rng_state(),
                               selector_rng=selector.bit_generator.state,
                               fork=fork_provenance),path)
        print('PPO '+json.dumps(record),flush=True)
        updates_this_invocation+=1
        if args.max_updates and updates_this_invocation>=args.max_updates and min(counters)<args.episodes:
            print('Stopped at requested checkpoint boundary; resume with the same output/configuration.',flush=True)
            return None
    bundle.save(out/'ppo_final.pt')
    return bundle


def export(args):
    source = Path(args.input) if args.input else Path(args.output)/'ppo_final.pt'
    bundle=PolicyBundle.load(source)
    selection=[]
    # Compose independently selected teams, retaining the corresponding critic
    # and normalizer rather than accidentally exporting the other team's state.
    for role in range(2):
        best=Path(args.output)/f'best_role{role}.pt'
        if best.exists():
            selected=PolicyBundle.load(best)
            for robot in TEAMS[role]:
                bundle.actors[robot].load_state_dict(selected.actors[robot].state_dict())
            bundle.critics[role].load_state_dict(selected.critics[role].state_dict())
            bundle.norms[role].__dict__.update(selected.norms[role].__dict__)
            selection.append(dict(role=role,checkpoint=str(best),episode=selected.metadata.get('episodes_per_role',[0,0])[role]))
    bundle.refresh(); bundle.metadata['source']=str(source)
    bundle.metadata['selected_roles']=selection
    bundle.save(Path(args.output)/'export/policy.pt')
    write_json(Path(args.output)/'export/metadata.json',bundle.metadata)
    return bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['distill','ppo','ppo-pilot','export','profile','pipeline','calibrate-value'],default='pipeline')
    parser.add_argument('--output',default=str(ROOT/'current/search_rl_v1/seed0'))
    parser.add_argument('--input'); parser.add_argument('--model-dir',default=str(ROOT/'current/memory_observation_v3/models'))
    parser.add_argument('--fork-from',help='Explicitly continue a PPO checkpoint in a new output after a code fix or batch-size change; preserve optimizer/RNG/counters')
    parser.add_argument('--iteration',type=int,default=45); parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--scratch',action='store_true')
    parser.add_argument('--without-belief',action='store_true',help='Train the feature-removal ablation; tracker/search still obey the local information boundary')
    parser.add_argument('--workers',type=int,default=16); parser.add_argument('--decisions',type=int,default=2000)
    parser.add_argument('--budget',type=int,default=1000); parser.add_argument('--depth',type=int,default=8)
    parser.add_argument('--distill-epochs',type=int,default=30); parser.add_argument('--episodes',type=int,default=5000)
    parser.add_argument('--episodes-per-update',type=int,default=16); parser.add_argument('--ppo-epochs',type=int,default=4)
    parser.add_argument('--minibatch',type=int,default=256); parser.add_argument('--lr',type=float,default=3e-5)
    parser.add_argument('--critic-lr',type=float,default=3e-4); parser.add_argument('--clip',type=float,default=.1)
    parser.add_argument('--anchor',type=float,default=.1); parser.add_argument('--entropy',type=float,default=.001)
    parser.add_argument('--opponent-budget',type=int,default=200); parser.add_argument('--eval-every',type=int,default=500)
    parser.add_argument('--validation-episodes',type=int,default=20)
    parser.add_argument('--pilot-episodes',type=int,default=20,help='Paired episodes per layout and role after training; reduce only for functional smoke tests')
    parser.add_argument('--max-updates',type=int,default=0,help='Optional graceful checkpoint boundary for controlled interruption tests')
    args=parser.parse_args()
    for name in ('workers','decisions','budget','depth','episodes','episodes_per_update','minibatch','eval_every','validation_episodes','pilot_episodes','distill_epochs','ppo_epochs'):
        if getattr(args,name)<=0:
            parser.error(f'--{name.replace("_","-")} must be positive')
    if args.max_updates<0 or args.seed<0:parser.error('seed and max-updates must be nonnegative')
    if args.fork_from and args.stage not in ('ppo','ppo-pilot'):
        parser.error('--fork-from is supported only for ppo or ppo-pilot')
    torch.set_num_threads(1); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if args.fork_from:fork_ppo(args)
    write_json(out/'invocation.json',configuration(args))
    if args.stage=='export':
        export(args); return
    if args.stage=='profile':
        from benchmark_local_obstacles import profile
        profile(args); return
    if args.stage=='calibrate-value':
        from benchmark_local_obstacles import calibrate_value
        calibrate_value(args);return
    with mp.get_context('spawn').Pool(args.workers,initializer=worker_init) if args.workers>1 else _NoPool() as pool:
        if args.stage in ('distill','pipeline'):
            distill(args,pool)
        if args.stage in ('ppo','ppo-pilot','pipeline'):
            result=ppo(args,pool)
            if result is None:
                return
    if args.stage in ('pipeline','ppo-pilot'):
        export(args)
        from benchmark_local_obstacles import pilot
        pilot(args)


class _NoPool:
    def __enter__(self): return None
    def __exit__(self,*args): return False


if __name__=='__main__':
    main()
