"""Budget and migration helpers for E49; never train before H is frozen.

Providers: Single/L5 existing torch supervision; MAPPO author R_MAPPO;
PPO-C Stable-Baselines3. Callers must supply E49-only validated episodes.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import torch


@dataclass
class AdaptationBudget:
    family: str
    horizon: int
    horizon_frozen: bool
    supervised_states: int = 0
    epochs: int = 0
    team_steps: int = 0
    def consume(self,*,states=0,epochs=0,team_steps=0):
        if not self.horizon_frozen:raise RuntimeError('Freeze calibrated H before adaptation')
        if self.family not in ('Single','L5','MAPPO_RouteAPF','PPO_C'):raise ValueError(self.family)
        if min(states,epochs,team_steps)<0:raise ValueError('Negative budget consumption')
        if self.supervised_states+states>20000 or self.epochs+epochs>20 or self.team_steps+team_steps>500000:raise RuntimeError('E49 adaptation budget exceeded')
        self.supervised_states+=states;self.epochs+=epochs;self.team_steps+=team_steps


def adaptation_trigger(family,metrics):
    """Inputs already average seeds/maps and equally weight the three tiers."""
    comparisons={'Single':(('SINGLE_PRIOR','GEOM_SEARCH'),),'L5':(('A3','A2'),('A5','A4')),
                 'MAPPO_RouteAPF':(('MAPPO_RouteAPF','RouteAPF'),),'PPO_C':(('PPO_C','A4'),)}
    evidence=[]
    for method,reference in comparisons[family]:
        a,b=metrics[method],metrics[reference]
        evidence.append(dict(method=method,reference=reference,double_gap=b['double_capture']-a['double_capture'],utility_gap=b['utility']-a['utility']))
    return dict(trigger=any(e['double_gap']>=.1-1e-12 or e['utility_gap']>=.05-1e-12 for e in evidence),comparisons=evidence)


def continue_after_pilot(seed_results,supervised=False):
    if set(map(int,seed_results))!={17,29,43}:raise ValueError('Pilot requires all three model seeds')
    return any(r['utility_gain']>=.01 or (supervised and r['action_loss_reduction']>=.05) for r in seed_results.values())


def select_checkpoint(candidates):
    """Validation raw utility only; early candidate wins ties, including step0."""
    if not candidates:raise ValueError('Include old checkpoint as step 0')
    if not any(c['steps']==0 for c in candidates):raise ValueError('Old model must remain eligible')
    return max(candidates,key=lambda c:(c['validation_utility'],-c['steps']))


def supervised_hot_start(family,seed,source,train_batches,validation_batches,budget,output,epochs=5,lr=3e-4):
    """Bounded existing-architecture updates; label sets must exclude dev/test.

    Batches follow the established Single losses or L5 pairwise rank loss.
    Checkpoint selection for deployment still requires closed-loop validation
    raw utility; returned losses do not select the deployed checkpoint.
    """
    from learning.e45b_joint_network import Network,losses
    from learning.e46_geometry_teacher import RankNetwork
    if family not in ('Single','L5') or budget.family!=family:raise ValueError('Wrong supervised family')
    if int(seed) not in (17,29,43):raise ValueError('Unregistered model seed')
    budget.consume(epochs=epochs)
    torch.manual_seed(int(seed))
    with np.load(source,allow_pickle=False) as data:
        if family=='Single':
            model=Network('single',int(data['width']));state={k:torch.as_tensor(data[k],dtype=torch.float32) for k in data.files if k not in ('kind','width')}
        else:model=RankNetwork();state={k:torch.as_tensor(data[k],dtype=torch.float32) for k in data.files}
    model.load_state_dict(state,strict=True);optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
    def loss(batch):
        from run_e46_geometry_teacher import student_loss
        return student_loss(model,batch,rank=family=='L5')
    history=[];output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,epochs+1):
        model.train();training=[]
        for batch in train_batches:
            optimizer.zero_grad();value=loss(batch);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step();training.append(float(value.detach()))
        model.eval()
        with torch.no_grad():validation=[float(loss(batch)) for batch in validation_batches]
        path=output/f'epoch_{epoch:02d}.npz';model.export(path)
        history.append(dict(epoch=epoch,train_loss=float(np.mean(training)),validation_action_loss=float(np.mean(validation)),path=str(path)))
    return model,history


def migration_report(features,checkpoint,output=None):
    """Measure deterministic MAPPO action shift on identical sparse public input."""
    from e49_controllers import load_mappo,semantic_columns
    from learning.e45j_repaired_mappo import policy
    old,_=policy('MAPPO_APF');new,payload=load_mappo(checkpoint)
    old.actor.load_state_dict(payload['actor'],strict=True)
    x=np.asarray(features,np.float32)
    if x.shape[-1]!=844:raise ValueError('Expected old semantic actor input')
    expanded=np.zeros((*x.shape[:-1],924),np.float32);expanded[...,semantic_columns(844)]=x
    hidden=np.zeros((len(x),1,64),np.float32);mask=np.ones((len(x),1),np.float32)
    with torch.no_grad():
        before,_,_=old.actor(x,hidden,mask,deterministic=True)
        after,_,_=new.actor(expanded,hidden,mask,deterministic=True)
    delta=(after-before).numpy()
    report=dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                samples=len(x),max_abs_shift=float(np.max(np.abs(delta))),mean_abs_shift=float(np.mean(np.abs(delta))),
                input_old=844,input_new=924,critic_old=858,critic_new=938,
                semantics='old columns and LayerNorm affine parameters preserved; added linear columns zero; LayerNorm domain expands')
    if output:
        import json
        Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(report,indent=2)+'\n')
    return report


def input_migration_reports(output):
    """Record all 3 recurrent policies' measured sparse-input migration shift."""
    import e49_controllers as c
    from learning.e45i_official_mappo import PublicState
    from learning.e45j_repaired_mappo import inputs
    c.install(80,output)
    obs=dict(step=0,remaining=80,positions=[[-4.,-2.],[-4.,2.]],active_p=[True,True],active_e=[True,True],reports=[[[-1.,-2.],[-1.,-2.]],[[-1.,2.],[-1.,2.]]])
    state=PublicState(np.array([[[3.,4.],[3.,4.]]]),1.,17);state.observe(obs)
    expanded,_=c.mappo_inputs(state.features(obs),state.public,obs)
    old=expanded[:,c.semantic_columns(844)]
    return [migration_report(old,s['checkpoint'],Path(output)/f'mappo_input_migration_{s["train_seed"]}.json') for s in c.method_specs() if s['arm']=='MAPPO_RouteAPF']
