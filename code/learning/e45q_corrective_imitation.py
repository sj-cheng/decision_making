"""Stable BC updates and public-only DAgger queries on actual learner states."""
import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import run_e45o_hasac_demonstration as o
from learning.e45p_actor_supervision import latent_bc_loss
from learning.e45p_execution_precision import environment_action
from solvers.e45f_baselines import GeometryController


def cosine_lr(step,total,high,low):
    fraction=min(1.,max(0.,step/max(1,total-1)))
    return low+(high-low)*.5*(1+np.cos(np.pi*fraction))


def sample_labels(original,corrections,rng,batch=256):
    n=batch//2 if corrections is not None and len(corrections['obs']) else batch
    ids=rng.integers(len(original['obs']),size=n)
    data={k:original[k][ids] for k in ('obs','action','active')}
    if n<batch:
        ids=rng.integers(len(corrections['obs']),size=batch-n)
        data={k:np.concatenate([data[k],corrections[k][ids]]) for k in data}
    order=rng.permutation(batch)
    return {k:v[order] for k,v in data.items()}


def stable_bc_step(model,teacher,lr):
    values=[]
    for i,actor in enumerate(model.actor):
        actor.turn_on_grad()
        for parameter in actor.actor.log_std_layer.parameters():parameter.requires_grad_(False)
        for group in actor.actor_optimizer.param_groups:group['lr']=float(lr)
        loss=latent_bc_loss(actor,teacher['obs'][:,i],teacher['action'][:,i],teacher['active'][:,i])
        actor.actor_optimizer.zero_grad();loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(actor.actor.parameters(),1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise RuntimeError('Nonfinite corrective BC gradient')
        actor.actor_optimizer.step();actor.turn_off_grad()
        values.append(float(loss.detach()))
    return values


def factory(case,task):
    own=o.factory(case,task);original_act=own.act;own.labels=[];own.oracle_records=[]
    expert=None
    if task.get('query_teacher',False):
        # Constructor gets public geometry and a seed, never the world's true
        # target state or future actions. Expert's simulated states stay local.
        expert=GeometryController(case['map'],case['speed'],task['seed'],
            use_policy=False,use_value=False,budget_ms=task.get('teacher_budget_ms',200.))
        original_observe=own.observe
        def observe(obs):original_observe(obs);expert.observe(obs)
        own.observe=observe
    def act(obs,fixed_simulations=None):
        proposed=original_act(obs,fixed_simulations)
        executed,error=environment_action(proposed,case['speed'])
        own.last_stats['environment_speed_roundoff_correction']=error
        if expert is not None:
            started=time.perf_counter()
            teacher_action=expert.act(obs,fixed_simulations=task.get('teacher_fixed_simulations'))
            teacher_action,_=environment_action(teacher_action,case['speed'])
            assert expert.belief.signature()==own.belief.signature()
            assert not expert.last_stats.get('assignment_used',False) and not expert.last_stats.get('roles_used',False)
            simulated=np.concatenate(expert.traces) if expert.traces else np.zeros((0,26))
            errors=o.j.common.base.audit_simulated(simulated,np.asarray(case['map']),case['speed'])
            if np.any(errors):raise RuntimeError('Corrective oracle simulated transition audit failed: '+str(errors))
            valid=expert.last_stats['simulations']>0
            if valid:
                sample=own.samples[-1]
                own.labels.append(dict(obs=sample['obs'].copy(),action=np.clip(teacher_action/case['speed'],-1,1).astype(np.float32),
                    active=sample['active'].copy()))
            record=dict(valid=bool(valid),teacher_queries=1,simulations=expert.last_stats['simulations'],
                simulated_transitions=len(simulated),query_ms=1000*(time.perf_counter()-started),
                normalized_teacher_action=(teacher_action/case['speed']).tolist(),
                public_belief_exact=True,simulation_audit_passed=True,teacher_actuates=False)
            own.oracle_records.append(record);own.last_stats['teacher_query']=record
        return executed
    own.act=act
    return own


def initialize_worker(folder,queue,parent):
    o.init_worker(folder,queue,parent);o.j.common.base.make_controller=factory


def pool_for(folder,cores):
    context=o.mp.get_context('spawn');queue=context.Queue()
    for core in cores:queue.put(core)
    return ProcessPoolExecutor(max_workers=len(cores),mp_context=context,initializer=initialize_worker,
        initargs=(str(folder),queue,os.getpid())),queue


def collect(job):
    result=o.rollout(job)
    own=o.CURRENT
    if not own.labels:raise RuntimeError('No valid corrective labels in complete episode')
    labels={key:np.stack([row[key] for row in own.labels]) for key in ('obs','action','active')}
    records=own.oracle_records
    path=o.FOLDER/'correction_labels'/(job['task']['key']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**labels)
    result['summary'].update(teacher_queries=len(records),valid_labels=len(labels['obs']),
        teacher_query_ms=sum(r['query_ms'] for r in records),teacher_simulations=sum(r['simulations'] for r in records),
        teacher_simulated_transitions=sum(r['simulated_transitions'] for r in records),
        labels_path=str(path),labels_sha256=o.j.sha(path),teacher_actuates=False)
    # Actual transition action in result['arrays'] is the learner's action.
    # Labels are a distinct supervised dataset, never relabelled RL actions.
    return dict(summary=result['summary'],labels=labels)
