from __future__ import annotations
import json
import hashlib
import collections
import copy
import os
import pickle
import time
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from .core import Config,World,PublicMap,Belief,State,from_initial
from .semantics import GAMMA,team_reward,backed_return,SCHEMA,pursuit_potential,shaped_reward
from .provenance import bind_job,sha
from .methods import (Network,Controller,features,tensors,stack_features,normal_logprob,
                      geometric,candidate_pool,evader_action,evader_features,rootpuct_evader,MotionKernel,simulated_step,BRANCH_METHOD,METHOD_BRANCH,DIRECT_METHODS)


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False));os.replace(temporary,path)


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp');torch.save(data,temporary);os.replace(temporary,path)


def load_model(path,device='cpu'):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    if checkpoint.get('schema')!=SCHEMA:raise ValueError('Checkpoint architecture/protocol version mismatch: '+str(path))
    model=Network().to(device);model.load_state_dict(checkpoint['model']);model.eval()
    return model


@lru_cache(maxsize=12)
def read_case(path):
    case=json.loads(Path(path).read_text())
    return case,PublicMap(Config.scale(case['n']),case['boxes'])


def entries(bank,n,split):
    if split.startswith('sealed'):raise ValueError('Sealed maps are forbidden in training, teacher collection and validation')
    manifest=json.loads((Path(bank)/'manifest.json').read_text())
    selected=[row for row in manifest['entries'] if row['n']==n and row['split']==split]
    from .provenance import sha
    for row in selected:
        if sha(Path(bank)/row['path'])!=row['sha256']:raise RuntimeError('Map content changed: '+row['id'])
    return [str(Path(bank)/row['path']) for row in selected]


def teacher_labels(public,obs,belief,pool,rng,particles=4,depth=32):
    """Common public states, independent finite-horizon Monte Carlo action teacher.

    These truncated policy/rank labels never supervise the full-return critic.
    """
    world=World(public.cfg,public.boxes);scores=np.zeros(len(pool))
    seeds=rng.integers(2**31,size=particles)
    root_phi=pursuit_potential(public,State(obs['p'],belief.estimates(),obs['alive'],obs['t']),belief)
    for k,action in enumerate(pool):
        for seed in seeds:
            random=np.random.default_rng(seed);posterior=belief.fork(int(seed))
            targets,modes=posterior.sample_joint(random)
            state=State(obs['p'].copy(),targets,obs['alive'].copy(),obs['t']);rewards=[]
            phi=root_phi
            for d in range(depth):
                pa=action if d==0 else geometric(public,state.p,posterior.estimates(),state.alive,belief=posterior)
                nxt,modes=simulated_step(public,posterior,state,modes,pa,random)
                following_phi=pursuit_potential(public,nxt,posterior)
                rewards.append(shaped_reward(team_reward(public.cfg,state,nxt),phi,following_phi));phi=following_phi;state=nxt
                if world.done(state):break
            scores[k]+=backed_return(rewards)/particles
    return scores.astype(np.float32)


def collect_teacher_packets(job):
    """Uniform public states from complete training-map rollouts, independent of learner."""
    folder=Path(job['folder']);identity=bind_job(folder,job)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    rng=np.random.default_rng(np.random.SeedSequence([job['seed'],job.get('shard',0)]))
    paths=entries(job['bank'],job['n'],job['split']);packets=[];steps=0
    while len(packets)<job['queries']:
        path=paths[int(rng.integers(len(paths)))];case,public=read_case(path);world=World(public.cfg,public.boxes)
        state=from_initial(case,int(rng.integers(2)));belief=Belief(public,int(rng.integers(2**31)));motion=MotionKernel(public,rng)
        candidates=[]
        while not world.done(state):
            obs=world.observe(state);belief.update(obs)
            if state.t%job.get('stride',8)==0:
                candidates.append(dict(case=path,obs=obs,belief=belief_state(belief)))
            pa=geometric(public,state.p,belief.estimates(),state.alive,belief=belief)
            state=world.transition(state,pa,motion.actions(state));steps+=1
        count=min(len(candidates),job['queries']-len(packets))
        for index in sorted(rng.choice(len(candidates),size=count,replace=False)):
            packet=candidates[index];posterior=restore_belief(public,packet['belief']);obs=packet['obs']
            packet['x']=features(public,obs,posterior.estimates(),posterior.ages(),posterior)
            packet['pool']=candidate_pool(public,obs['p'],posterior.estimates(),obs['alive'],rng,belief=posterior).astype(np.float32)
            packet['teacher_seed']=int(rng.integers(2**31));packets.append(packet)
    path=folder/'packets.pt';save(path,dict(schema=SCHEMA,identity=identity['digest'],packets=packets))
    result=dict(identity=identity['digest'],path=str(path),queries=len(packets),supervised_collection_team_steps=steps)
    write_json(folder/'result.json',result);return result


def label_teacher_packet(job):
    folder=Path(job['folder']);identity=bind_job(folder,job)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    data=torch.load(job['packets'],map_location='cpu',weights_only=False)
    if data['schema']!=SCHEMA:raise ValueError('Packet schema mismatch')
    packet=data['packets'][job['packet_index']];_,public=read_case(packet['case']);belief=restore_belief(public,packet['belief'])
    started=time.time();rng=np.random.default_rng(packet['teacher_seed'])
    scores=teacher_labels(public,packet['obs'],belief,packet['pool'],rng,job['particles'],job['depth'])
    # Exact return ties use the existing geometry score as a deterministic
    # tie-break for the policy target; this is not a new reward or critic target.
    from .methods import geometry_scores
    tied=np.flatnonzero(scores==scores.max())
    if len(tied)>1:
        geometric_scores=geometry_scores(public,packet['obs']['p'],belief.estimates(),packet['obs']['alive'],packet['pool'][tied],belief)
        selected=tied[int(geometric_scores.argmax())]
    else:selected=int(tied[0])
    record=dict(x=packet['x'],pool=packet['pool'],scores=scores,action=packet['pool'][selected],
        public_state_id=hashlib.sha256(pickle.dumps(packet['obs'])).hexdigest())
    path=folder/'label.pt';save(path,dict(schema=SCHEMA,identity=identity['digest'],record=record))
    result=dict(identity=identity['digest'],path=str(path),queries=1,simulations=32*job['particles'],
        score_spread=float(np.ptp(scores)),seconds=time.time()-started)
    write_json(folder/'result.json',result);return result


def make_dataset(job):
    """Small serial entry using the same collection/label implementation as parallel jobs."""
    folder=Path(job['folder']);identity=bind_job(folder,job)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    collection=collect_teacher_packets(dict(job,folder=str(folder/'collection'),kind='teacher_collect'))
    records=[];labels=[]
    for i in range(job['queries']):
        result=label_teacher_packet(dict(kind='teacher_label',folder=str(folder/'labels'/str(i)),packets=collection['path'],packet_index=i,particles=job['particles'],depth=job['depth']))
        labels.append(result);records.append(torch.load(result['path'],weights_only=False)['record'])
    path=folder/'data.pt'
    result=dict(identity=identity['digest'],path=str(path),queries=len(records),simulations=sum(x['simulations'] for x in labels),
        supervised_collection_team_steps=collection['supervised_collection_team_steps'],informative_queries=sum(x['score_spread']>0 for x in labels))
    save(path,dict(schema=SCHEMA,identity=identity['digest'],records=records,result=result));write_json(folder/'result.json',result);return result


def supervised(model,optimizer,records,updates,rng,device,rank=False,use_value=True):
    model.train();losses=[]
    for _ in range(updates):
        rows=[records[i] for i in rng.integers(len(records),size=min(32,len(records)))]
        x=stack_features([r['x'] for r in rows],device)
        target=torch.as_tensor(np.stack([r['action'] for r in rows]),device=device)
        mean,_,value=model(x)
        loss=(mean-torch.atanh(target.clamp(-.999,.999))).square().mean()
        if rank:
            pool=torch.as_tensor(np.stack([r['pool'] for r in rows]),device=device)
            labels=torch.as_tensor(np.stack([r['scores'] for r in rows]),device=device)
            # Independent rollout teacher values, common pool and public states.
            prediction=model.scores(x,pool)
            distribution=torch.softmax(labels/.05,dim=-1)
            loss=loss-(distribution*torch.log_softmax(prediction,dim=-1)).sum(-1).mean()
        # Critic is learned only from real interaction returns, never truncated teacher scores.
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
        losses.append(float(loss.detach()))
    return dict(loss=float(np.mean(losses)),updates=updates)


def method_for(branch):return BRANCH_METHOD[branch]


def load_specs(specs,device='cpu'):
    cache={}
    result=[]
    for spec in specs:
        row=dict(spec)
        if row.get('checkpoint'):
            path=row['checkpoint']
            if row.get('sha256') and sha(path)!=row['sha256']:raise RuntimeError('Frozen opponent checkpoint changed')
            if path not in cache:cache[path]=load_model(path,device)
            row['model']=cache[path]
        else:row['model']=None
        result.append(row)
    return result


def validation_plan(paths,specs):
    """Fixed balanced Latin assignment: both initials on every validation map.

    Methods have equal scenario counts (+/-1); seeds/history snapshots within
    each method are cycled. This is a selection matrix, not the final full
    Cartesian test matrix. All opponents and per-opponent scores are retained.
    """
    methods=sorted({s.get('method','D_kernel') for s in specs});plan=[]
    for index,path in enumerate(paths):
        for initial in (0,1):
            scenario=2*index+initial;method=methods[scenario%len(methods)]
            choices=[i for i,s in enumerate(specs) if s.get('method','D_kernel')==method]
            selected=choices[(scenario//len(methods))%len(choices)]
            plan.append(dict(case=path,initial=initial,opponent_index=selected))
    return plan


def validation(model,job,device):
    paths=entries(job['bank'],job['n'],job['validation_split'])
    if job.get('verification'):paths=paths[:job.get('validation_maps',1)]
    elif len(paths)!=job['validation_maps']:raise ValueError('Validation bank does not match frozen map count')
    specs=load_specs(job.get('opponents') or [dict(method='AG' if job['role']=='evader' else 'D_kernel')],'cpu')
    plan=validation_plan(paths,specs)
    write_json(Path(job['folder'])/'validation_matrix.json',dict(scenarios=plan,opponents=job.get('opponents',[]),
        rule='all maps x two initials; balanced methods and cycling seed/history snapshots; worst opponent mean then overall mean'))
    groups=collections.defaultdict(list);episodes=[];model.eval()
    for index,item in enumerate(plan):
        case,public=read_case(item['case']);state=from_initial(case,item['initial']);world=World(public.cfg,public.boxes)
        spec=specs[item['opponent_index']];rng=np.random.default_rng(700_000+job['n']*1000+index)
        if job['role']=='pursuer':
            ctrl=Controller(public,method_for(job['branch']),17,model,job['validation_simulations'],depth=job['search_depth'])
        else:ctrl=Controller(public,spec['method'],17,spec['model'],job['validation_simulations'],depth=job['search_depth'])
        motion=MotionKernel(public,rng)
        while not world.done(state):
            pa=ctrl.act(world.observe(state))
            if job['role']=='evader':ea=evader_action(public,state,rng,model)
            elif spec['model'] is not None:ea=evader_action(public,state,rng,spec['model'])
            else:ea=motion.actions(state)
            state=world.transition(state,pa,ea)
        score=1-float(state.alive.mean()) if job['role']=='pursuer' else float(state.alive.mean())
        label=spec.get('method','D_kernel')
        if job['role']=='pursuer' and spec.get('checkpoint'):label+=f"_s{spec.get('seed')}_r{spec.get('round',0)}"
        groups[label].append(score);episodes.append(dict(case=case['id'],initial=item['initial'],opponent=label,score=score))
    means={k:float(np.mean(v)) for k,v in groups.items()}
    return dict(score=min(means.values()),mean=float(np.mean([e['score'] for e in episodes])),groups=means,episodes=episodes)


def belief_state(belief):
    return copy.deepcopy({k:v for k,v in belief.__dict__.items() if k not in ('public','rng')} | {'rng_state':belief.rng.bit_generator.state})


def restore_belief(public,data):
    b=Belief(public,0)
    b.__dict__.update({k:v for k,v in data.items() if k!='rng_state'})
    b.rng.bit_generator.state=data['rng_state']
    return b


def controller_state(controller):
    return dict(belief=belief_state(controller.belief),rng=controller.rng.bit_generator.state,last_seen=controller.last_seen,history_key=controller.history_key,last_history_t=controller.last_history_t,seed=controller.seed)


def restore_controller(controller,data):
    controller.belief=restore_belief(controller.public,data['belief'])
    controller.rng.bit_generator.state=data['rng'];controller.last_seen=data['last_seen'];controller.history_key=data['history_key'];controller.last_history_t=data['last_history_t'];controller.seed=data['seed']
    return controller



def load_records(path):
    data=torch.load(path,map_location='cpu',weights_only=False)
    if data.get('schema')!=SCHEMA:raise ValueError('Teacher dataset schema mismatch')
    return data['records']


def train(job):
    if job['gpu']==0:raise ValueError('GPU0 forbidden')
    os.sched_setaffinity(0,set(job['cpus']));torch.set_num_threads(1)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(job['seed']);np.random.seed(job['seed']);torch.use_deterministic_algorithms(True)
    folder=Path(job['folder']);identity=bind_job(folder,job)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    write_json(folder/'job.json',job)
    rng=np.random.default_rng(job['seed']);model=Network().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,eps=1e-5)
    records=load_records(job['dataset']) if job.get('dataset') else None
    checkpoint=folder/'latest.pt';steps=updates=iteration=teacher_queries=planner_simulations=0
    best_key=(-np.inf,-np.inf);opponent_samples=collections.Counter();saved={}
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        if saved.get('schema')!=SCHEMA or saved['identity']!=identity['digest']:raise ValueError('Resume identity mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        steps=saved['steps'];updates=saved['updates'];iteration=saved['iteration'];best_key=tuple(saved['best_key'])
        teacher_queries=saved['teacher_queries'];planner_simulations=saved['planner_simulations']
        opponent_samples.update(saved['opponent_samples'])
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'].cpu())
        if device.startswith('cuda'):torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        if records is not None and (folder/'online_teacher.pt').exists():
            records=load_records(folder/'online_teacher.pt')[:saved['teacher_records']]
    elif job.get('parent'):
        parent=load_model(job['parent'],device);model.load_state_dict(parent.state_dict())
    if not checkpoint.exists() and records and job['supervised_updates']:
        result=supervised(model,optimizer,records,job['supervised_updates'],rng,device,job['branch']=='rank',False)
        updates+=job['supervised_updates'];write_json(folder/'initialization.json',result)
    paths=entries(job['bank'],job['n'],job['train_split'])
    opponents=load_specs(job.get('opponents') or [dict(method='AG' if job['role']=='evader' else 'D_kernel')],'cpu')
    next_validation=(steps//job['validation_every']+1)*job['validation_every']
    initial_steps=steps;started=time.time();episode=None
    if saved.get('environment') is not None:
        restored=saved['environment'];path=restored['path'];case,public=read_case(path);world=World(public.cfg,public.boxes)
        state=restored['state'];selected=opponents[restored['opponent_index']]
        own=Controller(public,method_for(job['branch']),0,model,job['collection_simulations'],depth=job['search_depth']) if job['role']=='pursuer' else None
        opponent=Controller(public,selected['method'],0,selected['model'],job['opponent_simulations'],depth=job['search_depth']) if job['role']=='evader' else None
        if own is not None:restore_controller(own,restored['own'])
        if opponent is not None:restore_controller(opponent,restored['opponent'])
        bit_state=rng.bit_generator.state;motion=MotionKernel(public,rng,restored['motion_anchors'],restored['motion_modes']);rng.bit_generator.state=bit_state
        episode=True
    while steps<job['team_steps']:
        model.eval();rows=[];actions=[];raws=[];logps=[];values=[];rewards=[];dones=[];losses=[];kls=[]
        for _ in range(min(job['rollout'],job['team_steps']-steps)):
            if episode is None:
                path=paths[int(rng.integers(len(paths)))];case,public=read_case(path);world=World(public.cfg,public.boxes)
                state=from_initial(case,int(rng.integers(2)))
                methods=sorted({o.get('method','D_kernel') for o in opponents});method=methods[int(rng.integers(len(methods)))]
                choices=[o for o in opponents if o.get('method','D_kernel')==method]
                selected=choices[int(rng.integers(len(choices)))];opponent_samples[method]+=1
                own=Controller(public,method_for(job['branch']),int(rng.integers(2**31)),model,job['collection_simulations'],depth=job['search_depth']) if job['role']=='pursuer' else None
                opponent=Controller(public,selected['method'],int(rng.integers(2**31)),selected['model'],job['opponent_simulations'],depth=job['search_depth']) if job['role']=='evader' else None
                motion=MotionKernel(public,rng);episode=True
            obs=world.observe(state)
            if job['role']=='pursuer':
                # Search policy iteration: execute the declared search algorithm.
                # Search actions are imitation targets, never mislabeled Gaussian PPO samples.
                pa=own.act(obs);x=own.last_features;planner_simulations+=own.last_stats['simulations']
                with torch.no_grad():value=model(tensors(x,device))[2].item()
                actions.append(pa.astype(np.float32));raws.append(np.zeros_like(pa,np.float32));logps.append(0.)
                if records is not None and job['online_teacher_stride'] and steps%job['online_teacher_stride']==0:
                    pool=candidate_pool(public,state.p,own.belief.estimates(),state.alive,rng,kind='pool',belief=own.belief)
                    scores=teacher_labels(public,obs,own.belief,pool,rng,job['teacher_particles'],job['teacher_depth'])
                    from .methods import geometry_scores
                    tied=np.flatnonzero(scores==scores.max())
                    selected_teacher=tied[int(geometry_scores(public,state.p,own.belief.estimates(),state.alive,pool[tied],own.belief).argmax())] if len(tied)>1 else int(tied[0])
                    records.append(dict(x=x,pool=pool.astype(np.float32),scores=scores,action=pool[selected_teacher].astype(np.float32)))
                    teacher_queries+=1
                    save(folder/'online_teacher.pt',dict(schema=SCHEMA,records=records))
                ea=evader_action(public,state,rng,selected['model']) if selected['model'] is not None else motion.actions(state)
            else:
                x=evader_features(public,state)
                with torch.no_grad():
                    mean,logstd,v=model(tensors(x,device));raw=mean+torch.randn_like(mean)*logstd.exp()
                    ea=torch.tanh(raw)[0].cpu().numpy();value=v.item()
                    logp=normal_logprob(raw,mean,logstd,tensors(x,device)['actor_alive']).item()
                ea[~state.alive]=0.;pa=opponent.act(obs)
                raw_action=raw[0].cpu().numpy();raws.append(raw_action);logps.append(logp);actions.append(ea.astype(np.float32))
            controller=own if own is not None else opponent
            phi=pursuit_potential(public,state,controller.belief)
            nxt=world.transition(state,pa,ea);controller.observe(world.observe(nxt))
            following_phi=pursuit_potential(public,nxt,controller.belief)
            reward=shaped_reward(team_reward(public.cfg,state,nxt),phi,following_phi)
            if job['role']=='evader':reward=-reward
            rows.append(x);values.append(value);rewards.append(reward);dones.append(world.done(nxt))
            state=nxt;steps+=1
            if dones[-1]:episode=None
        last_value=0.
        if episode is not None:
            if own is not None:
                # Bootstrap without running or counting an additional planner action.
                own.observe(world.observe(state));x=features(public,world.observe(state),own.belief.estimates(),own.belief.ages(),own.belief)
            else:x=evader_features(public,state)
            with torch.no_grad():last_value=model(tensors(x,device))[2].item()
        advantages=np.zeros(len(rows),np.float32);carry=0.
        for i in reversed(range(len(rows))):
            following=last_value if i==len(rows)-1 else values[i+1]
            live=1.-float(dones[i]);delta=rewards[i]+GAMMA*following*live-values[i]
            carry=delta+GAMMA*.95*live*carry;advantages[i]=carry
        returns=advantages+values;normalized=(advantages-advantages.mean())/(advantages.std()+1e-8)
        model.train()
        for epoch in range(job['ppo_epochs']):
            epoch_kls=[]
            for chunk in np.array_split(rng.permutation(len(rows)),int(np.ceil(len(rows)/job['minibatch']))):
                x=stack_features([rows[i] for i in chunk],device)
                target=torch.as_tensor(np.asarray(returns)[chunk],device=device,dtype=torch.float32)
                mean,logstd,value=model(x)
                if job['role']=='evader':
                    raw=torch.as_tensor(np.asarray(raws)[chunk],device=device,dtype=torch.float32)
                    old=torch.as_tensor(np.asarray(logps)[chunk],device=device,dtype=torch.float32)
                    advantage=torch.as_tensor(normalized[chunk],device=device)
                    new=normal_logprob(raw,mean,logstd,x['actor_alive']);ratio=(new-old).clamp(-20,20).exp()
                    policy_loss=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()-.001*logstd.mean()
                    epoch_kls.append(float((old-new).mean().detach()))
                else:
                    action=torch.as_tensor(np.asarray(actions)[chunk],device=device)
                    policy_loss=(mean-torch.atanh(action.clamp(-.999,.999))).square().mean()
                loss=policy_loss+.5*(value-target).square().mean()
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
                updates+=1;losses.append(float(loss.detach()))
            kls.extend(epoch_kls)
            if epoch_kls and np.mean(epoch_kls)>.05*job['n']:break
        if records and job.get('auxiliary_updates'):
            supervised(model,optimizer,records,job['auxiliary_updates'],rng,device,job['branch']=='rank',False);updates+=job['auxiliary_updates']
        iteration+=1;diagnostic=None
        if not job.get('select_final') and (steps>=next_validation or steps==job['team_steps']):
            diagnostic=validation(model,job,device);next_validation+=job['validation_every']
            key=(diagnostic['score'],diagnostic['mean'])
            write_json(folder/'validation'/f'{steps}.json',diagnostic)
            save(folder/'checkpoints'/f'{steps}.pt',dict(schema=SCHEMA,identity=identity['digest'],model=model.state_dict(),steps=steps,validation=diagnostic,job=job))
            if key>best_key:
                best_key=key;save(folder/'best.pt',dict(schema=SCHEMA,identity=identity['digest'],model=model.state_dict(),steps=steps,validation=diagnostic,job=job))
        if job.get('select_final') and steps==job['team_steps']:
            best_key=(0.,0.);save(folder/'best.pt',dict(schema=SCHEMA,identity=identity['digest'],model=model.state_dict(),steps=steps,selection='predeclared final checkpoint; training maps only',job=job))
        environment=None if episode is None else dict(path=path,state=state,opponent_index=next(i for i,o in enumerate(opponents) if o is selected),
            own=controller_state(own) if own is not None else None,opponent=controller_state(opponent) if opponent is not None else None,
            motion_anchors=motion.anchors,motion_modes=motion.modes.copy())
        saved=dict(schema=SCHEMA,identity=identity['digest'],model=model.state_dict(),optimizer=optimizer.state_dict(),environment=environment,
            steps=steps,updates=updates,iteration=iteration,best_key=best_key,teacher_records=len(records) if records else 0,
            teacher_queries=teacher_queries,planner_simulations=planner_simulations,opponent_samples=dict(opponent_samples),
            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if device.startswith('cuda') else None,job=job)
        save(checkpoint,saved)
        row=dict(steps=steps,updates=updates,iteration=iteration,loss=float(np.mean(losses)),kl=float(np.mean(kls)) if kls else 0.,
            reward=float(np.mean(rewards)),validation=diagnostic['score'] if diagnostic else None,
            best_validation=best_key[0] if np.isfinite(best_key[0]) else None,teacher_queries=teacher_queries,planner_simulations=planner_simulations,
            seconds=time.time()-started,steps_per_second=(steps-initial_steps)/max(.01,time.time()-started),
            peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else 0.)
        with open(folder/'curve.jsonl','a') as f:f.write(json.dumps(row)+'\n')
        write_json(folder/'progress.json',row)
        if job.get('interrupt_after_rollouts') and iteration==job['interrupt_after_rollouts']:raise InterruptedError('Verification interruption after durable rollout checkpoint')
    result=dict(schema=SCHEMA,identity=identity['digest'],checkpoint=str(folder/'best.pt'),latest=str(checkpoint),team_steps=steps,updates=updates,
        teacher_queries=teacher_queries,planner_simulations=planner_simulations,best_validation=None if job.get('select_final') else best_key[0],best_validation_mean=None if job.get('select_final') else best_key[1],
        seed=job['seed'],n=job['n'],branch=job['branch'],role=job['role'],gpu=job['gpu'],seconds=time.time()-started,
        trained_from_legacy=False,opponent_episode_samples=dict(opponent_samples),checkpoint_sha256=sha(folder/'best.pt'),latest_sha256=sha(checkpoint),selection='fixed final' if job.get('select_final') else 'balanced validation matrix, worst opponent mean then overall mean')
    write_json(folder/'result.json',result);return result
