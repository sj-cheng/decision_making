"""Durable, validation-selected E49 adaptation, gated by frozen calibration H.

This module never reads test/development episodes for training. RL updates use
unmodified author R_MAPPO or SB3 PPO; source policies/optimizers stay preserved.
"""
from __future__ import annotations
import copy
import concurrent.futures as futures
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import sys
import time
import numpy as np
import torch


def _save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    if path.suffix=='.json':tmp.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
    else:torch.save(value,tmp)
    tmp.replace(path)


def _read(path):return json.loads(Path(path).read_text())


def _device():
    if not torch.cuda.is_available():raise RuntimeError('E49 adaptation requires reserved GPU3; no silent CPU/GPU0 fallback')
    # A caller may isolate physical3 with CUDA_VISIBLE_DEVICES=3.
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible not in (None,'','3'):raise RuntimeError('E49 training requires physical GPU3')
    if visible=='':raise RuntimeError('CUDA was disabled in this process; start a dedicated adaptation worker with CUDA_VISIBLE_DEVICES=3')
    return torch.device('cuda:0' if visible=='3' else 'cuda:3')


class PhysicalRollout:
    """One-second E49 interface shared by both original RL providers."""
    def __init__(self,entries,horizon,seed):
        self.entries=entries;self.horizon=horizon;self.rng=np.random.default_rng(seed)
    def reset(self):
        from e49_rules import create_environment,make_opponent
        from e49_controllers import make_controller,install
        from run_e49_benchmark import observation,reset_history
        install(self.horizon)
        self.entry=self.entries[int(self.rng.integers(len(self.entries)))];self.case=_read(self.entry['path'])
        self.boxes=np.asarray(self.case['map'],float).reshape(-1,2,2)
        self.env=create_environment(self.boxes,self.horizon)
        seed=int(self.rng.integers(1,2**31-1));self.seed=seed
        self.opponent_kind='rootpuct' if self.rng.random()<.5 else 'fixed_D_kernel'
        self.other=make_opponent(self.env,seed) if self.opponent_kind=='rootpuct' else None
        world=self.case['worlds'][int(self.rng.integers(2))]
        self.state=np.asarray(world['initial'],float).reshape(-1,1).copy()
        self.own=make_controller(self.case,dict(arm='A4',seed=seed))
        for frame in reset_history(self.case,self.horizon)[:-1]:self.own.observe(frame)
        self.obs=observation(self.state,self.boxes,self.horizon);self.own.observe(self.obs)
        if self.opponent_kind=='fixed_D_kernel':
            self.other=_FixedTargets(self.own.belief,seed)
            self.state[:4,0]=self.other.positions().ravel()
            # The sampled initial targets have the same hidden public report.
            actual=observation(self.state,self.boxes,self.horizon)
            if actual!=self.obs:raise RuntimeError('Fixed-D sample changed public handoff')
        self.done=False;self.utility=0.;self.steps=0
        return self.features()
    def features(self):
        samples=np.zeros((2,64,2));weights=np.zeros((2,64))
        for j,e in enumerate(self.own.belief.estimators):
            if self.obs['active_e'][j]:samples[j]=e.particles;weights[j]=e.particle_weights
        from e49_controllers import public_features
        return public_features(self.boxes,np.asarray(self.obs['positions']),self.obs['active_p'],self.obs['active_e'],samples,weights,self.own.belief.known,self.own.records,1.,self.obs['remaining'])
    def potential(self):
        from solvers.e45b_joint_nte import coverage_potential
        if self.done:return 0.
        x=self.features();points=x['particles'][:,:,:2]*20+x['public'][:2]*10;p=self.own.public
        return .05*coverage_potential(np.asarray(self.obs['positions']),np.asarray(self.obs['active_p']),np.asarray(self.obs['active_e']),points,x['weights'],1.,p.vertices,p.dist,self.boxes)
    def step(self,action):
        from run_e49_benchmark import observation
        from run_frozen_d_closed_loop import legal_action
        if self.done:raise RuntimeError('Reset terminal E49 training episode')
        raw=(self.other.actions(self.obs['active_e']) if self.opponent_kind=='fixed_D_kernel' else np.asarray(self.other.search(self.env,self.state.copy(),turn=0).best_action,float).reshape(-1)[:4].reshape(2,2))
        for j in range(2):
            if not self.obs['active_e'][j] or not legal_action(self.state[2*j:2*j+2,0],raw[j],1.5,self.boxes):raw[j]=0.
        self.state=self.env.step(self.state,np.r_[raw.ravel(),np.asarray(action).ravel()][:,None],1.)
        if not self.env.is_valid(self.state):raise RuntimeError('Invalid E49 training transition')
        self.steps+=1;self.done=not (self.state[9:11,0]>.5).any() or self.steps>=self.horizon
        reward=float(not (self.state[9:11,0]>.5).any())-.0025;self.utility+=reward
        self.obs=observation(self.state,self.boxes,self.horizon)
        self.own.observe(self.obs)
        return self.features(),reward,self.done

    def snapshot(self):
        result=dict(entry=self.entry,case=self.case,state=self.state,seed=self.seed,opponent_kind=self.opponent_kind,records=self.own.records,done=self.done,utility=self.utility,steps=self.steps,rng=self.rng.bit_generator.state)
        if self.opponent_kind=='rootpuct':result['opponent_rng']=self.other.get_rng_state(self.env)
        else:result['fixed']={k:getattr(self.other,k) for k in ('graphs','rng','ids','modes')}
        return copy.deepcopy(result)
    def restore(self,saved):
        from e49_rules import create_environment,make_opponent
        from e49_controllers import make_controller,install
        install(self.horizon)
        for key in ('entry','case','state','seed','opponent_kind','done','utility','steps'):setattr(self,key,saved[key])
        self.rng.bit_generator.state=saved['rng'];self.boxes=np.asarray(self.case['map'],float).reshape(-1,2,2)
        self.env=create_environment(self.boxes,self.horizon);self.own=make_controller(self.case,dict(arm='A4',seed=self.seed))
        for record in saved['records']:self.own.observe(record)
        self.obs=saved['records'][-1]
        if self.opponent_kind=='rootpuct':
            self.other=make_opponent(self.env,self.seed);self.other.set_rng_state(self.env,saved['opponent_rng'])
        else:
            self.other=_FixedTargets.__new__(_FixedTargets)
            for key,value in saved['fixed'].items():setattr(self.other,key,value)



class _FixedTargets:
    """Original D target kernel, reparameterized by the E49 public belief."""
    def __init__(self,belief,seed):
        from solvers.frozen_d_search_interface import root_sample,belief_view
        self.graphs=[e.graph for e in belief.estimators]
        if any(g is None for g in self.graphs):raise RuntimeError('Fixed-D training handoff requires hidden public history')
        self.rng=[np.array([(seed+94501+104729*j)%4294967296],np.int64) for j in range(2)]
        self.ids=[];self.modes=[]
        for j,e in enumerate(belief.estimators):
            _,i,m=root_sample(*belief_view(e),self.rng[j]);self.ids.append(i);self.modes.append(m)
    def positions(self):return np.array([g[0][i] for g,i in zip(self.graphs,self.ids)])
    def actions(self,active):
        from solvers.frozen_d_search_interface import target_step
        before=self.positions()
        for j,g in enumerate(self.graphs):
            if active[j]:self.ids[j],self.modes[j]=target_step(self.ids[j],self.modes[j],g[1],g[2],g[3],self.rng[j])
        return self.positions()-before


def _validate(out,entries,spec,horizon,stage,cpus):
    from run_e49_benchmark import batch,tasks_for
    validation_specs=[spec]
    if spec['arm']=='A3':validation_specs.append(dict(spec,arm='A5',name=spec['name'].replace('A3','A5')))
    rows=batch(out,tasks_for(entries,validation_specs,horizon,stage),cpus)
    groups={}
    for row in rows:groups.setdefault(row['task']['group'],[]).append(row['summary']['utility'])
    return float(np.mean([np.mean(v) for v in groups.values()]))


def _dataset(out,config,specs,horizon,cpus):
    """Frozen traces only; deterministic strata selection retains failures."""
    from run_e49_benchmark import batch,tasks_for,load_episode
    from e49_controllers import make_controller
    root=Path(out)/'adaptation/dataset';root.mkdir(parents=True,exist_ok=True)
    index=root/'index.json'
    if index.exists():return _read(index)
    result={};collection=[]
    # Four frozen geometric/controllers supply common states without learner
    # outcome filtering. Per-tier/first-capture strata are chosen after saving all.
    collectors=[next(s for s in specs if s['arm']==arm) for arm in ('A2','A4','AG','RouteAPF')]
    for split,limit in [('train',18000),('validation',2000)]:
        entries=[e for e in config['entries'] if e['split']==split]
        rows=batch(out,tasks_for(entries,collectors,horizon,'adaptation_collect_'+split),cpus)
        strata={};all_records=[]
        for row in sorted(rows,key=lambda r:json.dumps(r['task'],sort_keys=True)):
            payload=load_episode(row['episode']) if 'episode' in row else row
            task=payload['task'];case=_read(task['case']);prefix=[dict(f,remaining=horizon-f['step']) for f in case['history'][:-1]]
            for frame in payload['trace']:
                prefix.append(frame['observation']);phase='before' if sum(frame['observation']['active_e'])==2 else 'after'
                key=(str(task['obstacle_tier']),phase)
                record=dict(case=task['case'],records=copy.deepcopy(prefix),seed=task['seed'],phase=phase,obstacle_tier=task['obstacle_tier'],failed_episode=not payload['summary']['double_capture'])
                raw=json.dumps({k:record[k] for k in ('case','records','seed')},sort_keys=True).encode();record['id']=hashlib.sha256(raw).hexdigest()
                all_records.append(record);strata.setdefault(key,[]).append(record)
        # Save every failed trajectory reference; no failure-based exclusion.
        archive=root/(split+'_all_states.pkl');archive.write_bytes(pickle.dumps(all_records,protocol=5))
        chosen=[];seen=set()
        sampling_rng=np.random.default_rng(14920261108)
        for key in sorted(strata):sampling_rng.shuffle(strata[key])
        while len(chosen)<limit and any(strata.values()):
            for key in sorted(strata):
                while strata[key]:
                    item=strata[key].pop(0)
                    if item['id'] not in seen:seen.add(item['id']);chosen.append(item);break
                if len(chosen)>=limit:break
        target=root/(split+'.pkl');target.write_bytes(pickle.dumps(chosen,protocol=5))
        result[split]=dict(path=str(target),states=len(chosen),available=len(all_records),strata={f'{tier}/{phase}':sum(r['obstacle_tier']==tier and r['phase']==phase for r in chosen) for tier in ('sparse','16','24') for phase in ('before','after')})
    _save(index,result);return result


def _labels(out,dataset,horizon,family,cpus=None):
    """One 256-full-simulation geometry teacher query per public state/type."""
    import concurrent.futures as futures
    import multiprocessing as mp
    root=Path(out)/'adaptation/labels'/family;root.mkdir(parents=True,exist_ok=True)
    result={};jobs=[]
    for split,info in dataset.items():
        states=pickle.loads(Path(info['path']).read_bytes());paths=[]
        for record in states:
            path=root/(record['id']+'.npz');paths.append(str(path))
            if not path.exists():jobs.append(dict(record=record,path=str(path),family=family,horizon=horizon,out=str(out)))
        result[split]=paths
    if cpus and len(cpus)>1 and jobs:
        from run_e49_benchmark import worker_init
        context=mp.get_context('spawn');queue=context.Queue()
        for cpu in cpus:queue.put(cpu)
        with futures.ProcessPoolExecutor(len(cpus),mp_context=context,initializer=worker_init,initargs=(queue,)) as pool:
            for _ in pool.map(_label_one,jobs,chunksize=1):pass
    else:
        for job in jobs:_label_one(job)
    _save(root/'index.json',result);return result


def _label_one(job):
    from e49_controllers import make_controller,install
    from learning.e46_geometry_teacher import root_labels
    started=time.perf_counter();record=job['record'];install(job['horizon'],job['out'])
    path=Path(job['path'])
    if path.exists():return str(path)
    query=path.with_suffix('.query.json')
    if query.exists():raise RuntimeError('Incomplete prior teacher query requires engineering review: '+str(query))
    case=_read(record['case']);arm='A2' if job['family']=='L5' else 'GEOM_SEARCH'
    teacher=make_controller(case,dict(arm=arm,seed=record['seed']))
    for obs in record['records'][:-1]:teacher.observe(obs)
    _save(query,dict(status='started',public_state=record['id'],family=job['family'],time=time.time()))
    teacher.act(record['records'][-1],fixed_simulations=256)
    actions,weights=root_labels(teacher)
    labels=dict(teacher.features(teacher.root),label_actions=actions,label_weights=weights)
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**labels);tmp.replace(path)
    _save(path.with_suffix('.cost.json'),dict(public_state=record['id'],family=job['family'],complete_simulations=teacher.last_stats['simulations'],seconds=time.perf_counter()-started))
    _save(query,dict(status='complete',public_state=record['id'],family=job['family'],time=time.time()))
    return str(path)


def _supervised_train(out,family,seed,source,label_paths,target_epoch,device):
    from learning.e45b_joint_network import Network
    from learning.e46_geometry_teacher import RankNetwork
    from run_e46_geometry_teacher import student_loss
    folder=Path(out)/'adaptation'/family/str(seed);folder.mkdir(parents=True,exist_ok=True)
    resume=folder/'resume.pt';torch.manual_seed(seed)
    if family=='Single':
        with np.load(source,allow_pickle=False) as d:
            model=Network('single',int(d['width']));initial={k:torch.as_tensor(d[k],dtype=torch.float32) for k in d.files if k not in ('kind','width')}
    else:
        model=RankNetwork()
        with np.load(source,allow_pickle=False) as d:initial={k:torch.as_tensor(d[k],dtype=torch.float32) for k in d.files}
    model.load_state_dict(initial,strict=True);model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=1e-5);epoch=0;history=[]
    if resume.exists():
        saved=torch.load(resume,map_location=device,weights_only=False);model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);epoch=saved['epoch'];history=saved['history'];torch.set_rng_state(saved['rng'].cpu())
    def batches(paths,shuffle):
        order=torch.randperm(len(paths)).tolist() if shuffle else range(len(paths))
        for start in range(0,len(paths),128):
            rows=[]
            for i in list(order)[start:start+128]:
                with np.load(paths[i],allow_pickle=False) as d:rows.append({k:d[k] for k in d.files})
            yield {k:torch.as_tensor(np.stack([r[k] for r in rows]),dtype=torch.float32,device=device) for k in rows[0]}
    def val():
        model.eval()
        with torch.no_grad():return float(np.mean([float(student_loss(model,b,family=='L5')) for b in batches(label_paths['validation'],False)]))
    initial_loss=history[0]['initial_action_loss'] if history else val()
    while epoch<target_epoch:
        model.train();losses=[]
        for b in batches(label_paths['train'],True):
            optimizer.zero_grad();loss=student_loss(model,b,family=='L5');loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step();losses.append(float(loss.detach()))
        epoch+=1;path=folder/f'epoch_{epoch:02d}.npz';model.export(path)
        history.append(dict(epoch=epoch,initial_action_loss=initial_loss,train_loss=float(np.mean(losses)),validation_action_loss=val(),checkpoint=str(path)))
        _save(resume,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,history=history,rng=torch.get_rng_state()))
        _save(folder/'training.json',history)
    return history[-1]


def _mappo_train(out,seed,source,entries,horizon,target_steps,device):
    from e49_controllers import load_mappo,mappo_inputs,_clone
    from learning.e45i_official_mappo import settings,R_MAPPO
    from learning.e45j_repaired_mappo import finish_episodes,training_rewards
    from learning.e45j_adapter_repairs import route_apf_action
    folder=Path(out)/'adaptation/MAPPO_RouteAPF'/str(seed);folder.mkdir(parents=True,exist_ok=True)
    latest=folder/'resume.pt'
    torch.manual_seed(seed);model,payload=load_mappo(latest if latest.exists() else source)
    model.actor.to(device);model.critic.to(device);model.device=device
    model.actor.tpdv=dict(dtype=torch.float32,device=device);model.critic.tpdv=dict(dtype=torch.float32,device=device)
    trainer=R_MAPPO(settings(),model,device)
    # Preserve old optimizer state where no input tensor shape changed; expand
    # moments for the same semantic columns as the migrated model parameters.
    from e49_controllers import semantic_columns
    def optimizer_load(optimizer,state,old_dim):
        if state is None:return
        state=copy.deepcopy(state)
        for item in state['state'].values():
            for key,v in list(item.items()):
                if not torch.is_tensor(v) or not v.ndim:continue
                if v.shape[-1]==old_dim:
                    shape=list(v.shape);shape[-1]+=80;new=torch.zeros(shape,dtype=v.dtype);new[...,semantic_columns(old_dim)]=v;item[key]=new
        optimizer.load_state_dict(state)
    optimizer_load(model.actor_optimizer,payload.get('actor_optimizer'),844)
    optimizer_load(model.critic_optimizer,payload.get('critic_optimizer'),858)
    # finish_episodes is the unchanged complete-episode recurrent recipe,
    # with its central observation capacity explicitly widened by 80.
    finish=_clone(finish_episodes,[],{'CENT_DIM':938})
    steps=int(payload.get('e49_steps',0));update=int(payload.get('e49_update',0));history=payload.get('e49_history',[])
    rollout=PhysicalRollout(entries,horizon,seed+971)
    if 'e49_rollout_rng' in payload:rollout.rng.bit_generator.state=payload['e49_rollout_rng']
    if 'e49_torch_rng' in payload:torch.set_rng_state(payload['e49_torch_rng'].cpu())
    if 'e49_cuda_rng' in payload:torch.cuda.set_rng_state(payload['e49_cuda_rng'].cpu(),device)
    while steps<target_steps:
        trainer.prep_rollout();episodes=[];batch_steps=0
        # Complete episodes retain unbiased bootstrap semantics. To guarantee
        # the ceiling, stop before an episode that could exceed remaining H.
        while len(episodes)<16 and steps+batch_steps+horizon<=target_steps:
            rollout.reset();hidden=np.zeros((2,1,64),np.float32);records=[];phis=[];blocked=[];rewards=[];priv=[]
            while not rollout.done:
                features=rollout.features();x,critic_public=mappo_inputs(features,rollout.own.public,rollout.obs)
                before=hidden.copy();priv.append(rollout.state[:4,0].copy()/10.)
                with torch.no_grad():raw,logp,h=model.actor(x,hidden,np.ones((2,1),np.float32),deterministic=False)
                raw=raw.cpu().numpy();hidden=h.cpu().numpy()
                phis.append(float(rollout.potential()+.25*(2-sum(rollout.obs['active_e']))))
                physical=route_apf_action(rollout.own,rollout.obs,raw)
                short=rollout.own.last_stats['safety_shortened'];blocked.append(short/2)
                records.append(dict(obs=x,critic_public=critic_public,actor_hidden=before,raw=raw,logp=logp.cpu().numpy(),active=np.ones((2,1),np.float32)))
                _,reward,_=rollout.step(physical);rewards.append(reward)
            ep={k:np.stack([r[k] for r in records]) for k in records[0]}
            ep['privileged']=np.asarray(priv,np.float32);ep['reward']=training_rewards(rewards,phis,blocked)
            episodes.append(ep);batch_steps+=len(records)
        if not episodes:break
        data=finish(episodes,model);fraction=min(1.,steps/500000)
        for optimizer in (model.actor_optimizer,model.critic_optimizer):
            for group in optimizer.param_groups:group['lr']=3e-4*(1-fraction)+3e-5*fraction
        trainer.entropy_coef=.005*(1-fraction)+.0005*fraction
        trainer.prep_training();info=trainer.train(data)
        steps+=batch_steps;update+=1
        history.append(dict(steps=steps,update=update,episodes=len(episodes),loss={k:float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k,v in info.items()}))
        _save(latest,dict(actor=model.actor.state_dict(),critic=model.critic.state_dict(),actor_optimizer=model.actor_optimizer.state_dict(),critic_optimizer=model.critic_optimizer.state_dict(),e49_steps=steps,e49_update=update,e49_history=history,e49_rollout_rng=rollout.rng.bit_generator.state,e49_torch_rng=torch.get_rng_state(),e49_cuda_rng=torch.cuda.get_rng_state(device)))
        _save(folder/'training.json',history)
    frozen=folder/f'steps_{steps:06d}.pt'
    if latest.exists() and not frozen.exists():frozen.write_bytes(latest.read_bytes())
    return dict(checkpoint=str(frozen),steps=steps)


def _ppo_train(out,seed,source,entries,horizon,target_steps,device):
    dependencies=Path(__file__).resolve().parents[1]/'current/e49_persistent_pursuit_benchmark_20261108/python_deps'
    if dependencies.exists() and str(dependencies) not in sys.path:sys.path.insert(0,str(dependencies))
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    import learning.e45w_joint_ppo as provider
    from learning.e45g_ppo import safe_action
    folder=Path(out)/'adaptation/PPO_C'/str(seed);folder.mkdir(parents=True,exist_ok=True)
    latest=folder/'resume.zip';selection=Path(source).parent/'selection.json'
    selected_source=_read(selection)['checkpoint']
    # Loading legacy SB3 policy requires its original observation schema first.
    provider.OBS_SHAPES['maps']=(8,4);provider.OBS_SHAPES['map_mask']=(8,)
    old=PPO.load(selected_source,device=device)
    provider.OBS_SHAPES['maps']=(24,4);provider.OBS_SHAPES['map_mask']=(24,)
    class Env(gym.Env):
        def __init__(self):
            self.sim=PhysicalRollout(entries,horizon,seed+182)
            self.observation_space=provider.structured_observation_space();self.action_space=provider.latent_action_space()
        def reset(self,*,seed=None,options=None):
            super().reset(seed=seed)
            x=self.sim.reset();self.phi=self.sim.potential()
            return {k:np.asarray(v,np.float32) for k,v in x.items()},{}
        def step(self,latent):
            physical,_=safe_action(np.asarray(self.sim.obs['positions']),np.tanh(np.asarray(latent).reshape(2,2)),self.sim.obs['active_p'],1.,self.sim.boxes)
            x,reward,done=self.sim.step(physical);phi=self.sim.potential();shaped=reward+phi-self.phi;self.phi=phi
            return {k:np.asarray(v,np.float32) for k,v in x.items()},float(shaped),done,False,{}
    env=Env()
    if latest.exists():model=PPO.load(latest,env=env,device=device,force_reset=False)
    else:
        # All optimizer settings and learned actor/critic tensors hot-start.
        model=PPO(provider.StructuredJointPolicy,env,learning_rate=old.learning_rate,n_steps=old.n_steps,batch_size=old.batch_size,n_epochs=old.n_epochs,gamma=old.gamma,gae_lambda=old.gae_lambda,clip_range=old.clip_range,clip_range_vf=old.clip_range_vf,normalize_advantage=old.normalize_advantage,ent_coef=old.ent_coef,vf_coef=old.vf_coef,max_grad_norm=old.max_grad_norm,target_kl=old.target_kl,policy_kwargs=dict(actor_width=old.policy.actor_network.width),seed=seed,device=device,verbose=0)
        model.policy.load_state_dict(old.policy.state_dict(),strict=True);model.policy.optimizer.load_state_dict(old.policy.optimizer.state_dict())
    del old
    rng_path=folder/'sampling_rng.pt'
    if rng_path.exists():
        rng=torch.load(rng_path,map_location='cpu',weights_only=False)
        if rng['steps']!=model.num_timesteps:raise RuntimeError('PPO model/sampling checkpoint transaction mismatch')
        env.sim.rng.bit_generator.state=rng['environment'];torch.set_rng_state(rng['torch']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])
        if rng.get('simulation') is not None:env.sim.restore(rng['simulation']);env.phi=rng['phi']
        monitor=model.env.envs[0]
        if hasattr(monitor,'needs_reset'):monitor.needs_reset=False;monitor.rewards=rng.get('monitor_rewards',[])
        if rng.get('cuda') is not None:torch.cuda.set_rng_state(rng['cuda'],device)
    def save_rng():
        _save(rng_path,dict(steps=model.num_timesteps,monitor_rewards=list(getattr(model.env.envs[0],'rewards',[])),environment=env.sim.rng.bit_generator.state,simulation=env.sim.snapshot() if hasattr(env.sim,'snapshot') else None,phi=env.phi,torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None))
    completed=int(model.num_timesteps);remaining=target_steps-completed
    # SB3 collects whole rollout buffers: only authorize complete buffers that
    # fit within the hard500k cap; no partial rollouts or altered PPO updater.
    amount=(max(0,remaining)//model.n_steps)*model.n_steps
    class Save(BaseCallback):
        def _on_step(self):return True
        def _on_rollout_start(self):
            if self.model.num_timesteps>completed:
                self.model.save(latest);save_rng()
                _save(folder/'progress.json',dict(steps=self.model.num_timesteps,provider='Stable-Baselines3 PPO',horizon=horizon))
    if amount:model.learn(total_timesteps=amount,reset_num_timesteps=False,callback=Save())
    model.save(latest);save_rng();steps=int(model.num_timesteps)
    actor=folder/f'actor_{steps:06d}.pt';_save(actor,model.policy.actor_state_numpy())
    _save(folder/'progress.json',dict(steps=steps,provider='Stable-Baselines3 PPO',horizon=horizon))
    return dict(checkpoint=str(actor),steps=steps)


def _ppo_train_worker(payload):
    """Train one PPO seed in an isolated process and checkpoint namespace."""
    out,seed,source,entries,horizon,target_steps,cpus = payload
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    torch.set_num_threads(1)
    if cpus:
        os.sched_setaffinity(0,set(cpus))
    result=_ppo_train(out,int(seed),source,entries,int(horizon),int(target_steps),_device())
    return int(seed),result


def _ppo_train_parallel(out,seeds,sources,entries,horizon,target_steps,cpus):
    """Run independent PPO seeds concurrently with disjoint CPU affinity.

    Each worker owns a separate SB3 model, optimizer, environment sampler and
    RNG checkpoint. The only shared input is read-only training-map metadata.
    """
    seeds=tuple(int(s) for s in seeds)
    if set(seeds)!={17,29,43}:raise ValueError('E49 PPO parallel set must be 17,29,43')
    cpus=list(cpus);chunks=[cpus[i::len(seeds)] for i in range(len(seeds))]
    payloads=[(str(out),seed,
                sources[seed]['checkpoint'] if isinstance(sources[seed],dict) else sources[seed],
                entries,horizon,target_steps,chunks[i])
              for i,seed in enumerate(seeds)]
    context=mp.get_context('spawn');results={}
    with futures.ProcessPoolExecutor(max_workers=3,mp_context=context) as pool:
        pending={pool.submit(_ppo_train_worker,payload):payload[1] for payload in payloads}
        for future in futures.as_completed(pending):
            seed,result=future.result();results[seed]=result
    if set(results)!=set(seeds):raise RuntimeError('PPO parallel worker set incomplete')
    return results


def adapt(out,config,specs,horizon,decisions,cpus):
    """Adapt triggered families after frozen-H audit; return final22 specs.

    Resume checkpoints include optimizer and RNG states. Every candidate is
    evaluated on all24 independent validation maps in both worlds. Family
    pilot decision uses all3 seeds; step0 remains eligible on exact ties.
    """
    from e49_learning import continue_after_pilot,select_checkpoint
    out=Path(out);frozen=out/'horizon_frozen.json'
    # Parent may use this protocol spelling; require a concrete frozen record.
    alternatives=[frozen,out/'horizon_selection.json',out/'calibration/frozen_horizon.json',out/'horizon_calibration/frozen.json']
    frozen=next((p for p in alternatives if p.exists()),None)
    if frozen is None:raise RuntimeError('Calibrated H must be durably frozen before adaptation')
    if int(_read(frozen).get('horizon',_read(frozen).get('H',-1)))!=int(horizon):raise RuntimeError('Adaptation H differs from frozen calibration')
    device=None;torch.set_num_threads(1)
    from e49_controllers import install
    install(horizon,out)
    train=[e for e in config['entries'] if e['split']=='train'];validation=[e for e in config['entries'] if e['split']=='validation']
    if len(train)!=240 or len(validation)!=24:raise RuntimeError('Require registered240 training and24 validation maps')
    final=copy.deepcopy(specs);root=out/'adaptation';root.mkdir(parents=True,exist_ok=True)
    families={'Single':('SINGLE_PRIOR',),'L5':('A3','A5'),'MAPPO_RouteAPF':('MAPPO_RouteAPF',),'PPO_C':('PPO_C',)}
    dataset=None
    for family,arms in families.items():
        decision=decisions.get(family,decisions.get('MAPPO',{}) if family=='MAPPO_RouteAPF' else {})
        if not decision.get('triggered',decision.get('trigger',False)):continue
        choice_path=root/family/'selected.json'
        if choice_path.exists():
            chosen=_read(choice_path)
            for spec in final:
                if spec['arm'] in arms:spec['checkpoint']=chosen[str(spec['train_seed'])]['checkpoint'];spec['adapted']=chosen[str(spec['train_seed'])]['steps']>0;spec['adaptation_attempted']=True
            continue
        device=device or (_device() if family!='PPO_C' else None)
        if family in ('Single','L5'):
            dataset=dataset or _dataset(out,config,specs,horizon,cpus);labels=_labels(out,dataset,horizon,family,cpus)
        candidates={};pilot_results={}
        # PPO seeds share no state. Run the pilot target concurrently so each
        # seed owns its model, optimizer, sampler and checkpoint transaction.
        if family=='PPO_C':
            sources={seed:next(s for s in specs if s['arm']==arms[0] and s['train_seed']==seed)
                     for seed in (17,29,43)}
            histories={}
            for seed,source in sources.items():
                path=root/family/str(seed)/'validation.json'
                history=_read(path) if path.exists() else []
                if not history:
                    initial=_validate(out,validation,source,horizon,f'adapt_{family}_{seed}_step0',cpus)
                    history=[dict(steps=0,checkpoint=source['checkpoint'],validation_utility=initial)];_save(path,history)
                histories[seed]=history
            trained_map=_ppo_train_parallel(out,(17,29,43),sources,train,horizon,100000,cpus)
            for seed,source in sources.items():
                trained=trained_map[seed];steps=trained['steps'];history=histories[seed]
                row=next((r for r in history if r['steps']==steps),None)
                if row is None:
                    candidate=dict(source,checkpoint=trained['checkpoint'])
                    row=dict(steps=steps,checkpoint=trained['checkpoint'],
                             validation_utility=_validate(out,validation,candidate,horizon,f'adapt_{family}_{seed}_pilot',cpus))
                    history.append(row);_save(root/family/str(seed)/'validation.json',history)
                pilot_results[seed]=dict(utility_gain=row['validation_utility']-history[0]['validation_utility'],action_loss_reduction=0.)
                candidates[seed]=history
            proceed=continue_after_pilot(pilot_results,False)
            _save(root/family/'pilot_decision.json',dict(continue_training=proceed,seeds=pilot_results,parallel=True))
        else:
            proceed=None
        for seed in (() if family=='PPO_C' else (17,29,43)):
            source=next(s for s in specs if s['arm']==arms[0] and s['train_seed']==seed)
            history_path=root/family/str(seed)/'validation.json';history=_read(history_path) if history_path.exists() else []
            if not history:
                initial=_validate(out,validation,source,horizon,f'adapt_{family}_{seed}_step0',cpus)
                history=[dict(steps=0,checkpoint=source['checkpoint'],validation_utility=initial)];_save(history_path,history)
            pilot_limit=5 if family in ('Single','L5') else 100000
            row=next((r for r in history if 0<r['steps']<=pilot_limit),None)
            if row is None:
                if family in ('Single','L5'):
                    trained=_supervised_train(out,family,seed,source['checkpoint'],labels,5,device);steps=5
                    metadata=dict(action_loss_reduction=(trained['initial_action_loss']-trained['validation_action_loss'])/max(abs(trained['initial_action_loss']),1e-12))
                else:
                    trained=(_mappo_train if family=='MAPPO_RouteAPF' else _ppo_train)(out,seed,source['checkpoint'],train,horizon,100000,device);steps=trained['steps'];metadata={}
                candidate=dict(source,checkpoint=trained['checkpoint'])
                row=dict(steps=steps,checkpoint=trained['checkpoint'],validation_utility=_validate(out,validation,candidate,horizon,f'adapt_{family}_{seed}_pilot',cpus),**metadata);history.append(row);_save(history_path,history)
            pilot_results[seed]=dict(utility_gain=row['validation_utility']-history[0]['validation_utility'],action_loss_reduction=row.get('action_loss_reduction',0.))
            candidates[seed]=history
        if family!='PPO_C':
            proceed=continue_after_pilot(pilot_results,family in ('Single','L5'))
            _save(root/family/'pilot_decision.json',dict(continue_training=proceed,seeds=pilot_results))
        if proceed:
            targets=(10,15,20) if family in ('Single','L5') else (200000,300000,400000,500000)
            for target in targets:
                if family=='PPO_C':
                    sources={seed:next(s for s in specs if s['arm']==arms[0] and s['train_seed']==seed)
                             for seed in (17,29,43)}
                    trained_map=_ppo_train_parallel(out,(17,29,43),sources,train,horizon,target,cpus)
                    trained_rows=[(seed,sources[seed],trained_map[seed]) for seed in (17,29,43)]
                else:
                    trained_rows=[(seed,next(s for s in specs if s['arm']==arms[0] and s['train_seed']==seed),None) for seed in (17,29,43)]
                for seed,source,parallel_trained in trained_rows:
                    if family in ('Single','L5'):
                        trained=_supervised_train(out,family,seed,source['checkpoint'],labels,target,device);steps=target
                    else:
                        trained=parallel_trained;steps=trained['steps']
                    history=candidates[seed]
                    if not any(r['steps']==steps for r in history):
                        history.append(dict(steps=steps,checkpoint=trained['checkpoint'],validation_utility=_validate(out,validation,dict(source,checkpoint=trained['checkpoint']),horizon,f'adapt_{family}_{seed}_budget{target}',cpus)))
                        _save(root/family/str(seed)/'validation.json',history)
        selected={str(seed):select_checkpoint(history) for seed,history in candidates.items()};_save(choice_path,selected)
        for spec in final:
            if spec['arm'] in arms:spec['checkpoint']=selected[str(spec['train_seed'])]['checkpoint'];spec['adapted']=selected[str(spec['train_seed'])]['steps']>0;spec['adaptation_attempted']=True
    _save(root/'final_method_specs.json',final)
    return final


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cpus',required=True,help='Comma-separated authorized rollout CPUs')
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    from e49_controllers import method_specs
    config=_read(args.out/'bank_manifest.json');selection=_read(args.out/'horizon_selection.json');decision=_read(args.out/'adaptation_decisions.json')
    cpus=list(map(int,args.cpus.split(',')))
    os.sched_setaffinity(0,set(cpus));adapt(args.out,config,method_specs(),selection['horizon'],decision['families'],cpus)

if __name__=='__main__':main()
