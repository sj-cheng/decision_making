"""Meaningful probability, public-input and deployment-path checks for E46."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import run_e46_geometry_teacher as run
from learning.e45b_joint_network import KEYS,Network,Runtime
from learning.e46_geometry_teacher import (MixedController,RankNetwork,RankRuntime,
    make_controller,prepare_public,root_labels,reference_values,legal)


@pytest.fixture(scope='module')
def packet(tmp_path_factory):
    out=tmp_path_factory.mktemp('e46_integration')
    run.configure(out)
    cfg=run.read(run.ROOT/'current/e45w_joint_ppo_and_planning_20261103/experiment.json')
    case=run.read(cfg['validation_entries'][0]['path'])
    return dict(id='acceptance',group='acceptance',map=case['map'],speed=case['speed'],
                records=case['history'],seed=146261105,phase='hidden')


def test_explicit_mixture_probabilities(monkeypatch):
    import learning.e46_geometry_teacher as impl
    own=object.__new__(MixedController)
    own.alpha=.5;own.source_counts=run.collections.Counter()
    own.proposal_rng=np.random.default_rng(99);own.public=SimpleNamespace(speed=1.)
    own.parameters=lambda node:None
    own.runtime=SimpleNamespace(sample=lambda params,rng:np.ones((2,2)))
    monkeypatch.setattr(impl,'draw_random',lambda *a:np.full((2,2),3.))
    monkeypatch.setattr(impl,'draw_geometry',lambda *a:np.full((2,2),2.))
    monkeypatch.setattr(impl,'legal',lambda *a:True)
    own.allowed=lambda:None
    node=SimpleNamespace(ap=np.ones(2,bool))
    for _ in range(20000):own.propose(node)
    frequencies={k:v/20000 for k,v in own.source_counts.items()}
    assert abs(frequencies['random']-.25)<.015
    assert abs(frequencies['geometry']-.375)<.015
    assert abs(frequencies['network']-.375)<.015


@pytest.mark.parametrize('arm',['PRIOR','GEOM','MIX','POOL_UNIFORM','POOL_GEOMETRIC'])
def test_real_public_path_legal_and_fixed_simulations(packet,arm):
    spec=dict(arm=arm,seed=packet['seed'],train_seed=17,alpha=.5)
    a,x,_=prepare_public(packet,spec,4)
    b,y,_=prepare_public(packet,spec,4)
    np.testing.assert_array_equal(x,y)
    assert a.last_stats['simulations']==b.last_stats['simulations']==4
    assert legal(a,a.root,x)
    assert a.last_stats['assignment_used'] is False
    assert a.last_stats['roles_used'] is False
    actions,weights=root_labels(a)
    assert actions.shape==(16,4) and weights.sum()==pytest.approx(1.)


def test_search_randomness_does_not_change_public_belief_input(packet):
    own,_,_=prepare_public(packet,dict(arm='PRIOR',seed=packet['seed'],search_seed=13),4)
    other,_,_=prepare_public(packet,dict(arm='GEOM',seed=packet['seed'],search_seed=37),4)
    for k in KEYS:np.testing.assert_array_equal(own.features(own.root)[k],other.features(other.root)[k])


def test_fixed_pool_teacher_retains_complete_joint_actions(packet):
    own,_,_=prepare_public(packet,dict(arm='POOL_UNIFORM',seed=packet['seed'],force_root=True),20)
    actions,weights=root_labels(own)
    assert len(own.root.actions)==16
    assert np.count_nonzero(weights)==16
    for action in own.root.actions:assert legal(own,own.root,action)


def test_rank_numpy_matches_torch_and_legal_runtime(packet,tmp_path):
    own,_,_=prepare_public(packet,dict(arm='POOL_UNIFORM',seed=packet['seed']),2)
    features=own.features(own.root)
    model=RankNetwork().double().eval();path=tmp_path/'rank.npz';model.export(path)
    actions=np.random.default_rng(0).uniform(-1,1,(16,4))
    x={k:torch.from_numpy(v[None]).double() for k,v in features.items()}
    with torch.no_grad():expected=model(x,torch.from_numpy(actions[None]).double())[0].numpy()
    np.testing.assert_allclose(RankRuntime(path)(features,actions),expected,rtol=1e-10,atol=1e-10)
    learner,a,_=prepare_public(packet,dict(arm='POOL_LEARNED',seed=packet['seed'],checkpoint=str(path)),4)
    assert legal(learner,learner.root,a)


def test_reference_uses_paired_worlds_and_is_replayable(packet):
    own,a,_=prepare_public(packet,dict(arm='GEOM',seed=packet['seed']),4)
    values=reference_values(packet,np.asarray([a,a]),2,871,horizon=2)
    np.testing.assert_array_equal(values[0],values[1])
    again=reference_values(packet,np.asarray([a,a]),2,871,horizon=2)
    np.testing.assert_array_equal(values,again)


def test_policy_only_loss_updates_encoder_but_not_value(packet):
    own,_,_=prepare_public(packet,dict(arm='GEOM',seed=packet['seed']),4)
    labels,weights=root_labels(own)
    arrays=dict(**own.features(own.root),label_actions=labels,label_weights=weights)
    batch={k:torch.from_numpy(v[None]).float() for k,v in arrays.items()}
    model=Network('single')
    loss=run.student_loss(model,batch);loss.backward()
    assert model.policy.weight.grad is not None
    assert model.value.weight.grad is None
    assert any(p.grad is not None for p in model.trunk.parameters())


def test_job_resume_and_dataset_teacher_separation(tmp_path):
    db=run.connect(tmp_path)
    a=dict(kind='label',stage='labels_GEOM_train',group='g',teacher='GEOM')
    b=dict(a,stage='labels_OLD_JOINT_train',teacher='OLD_JOINT')
    run.enqueue(db,a);run.enqueue(db,a);run.enqueue(db,b);db.commit()
    assert db.execute('select count(*) from jobs').fetchone()[0]==2
    assert run.digest(a)!=run.digest(b)
    db.close()


def test_dispatch_does_not_starve_data_behind_long_evaluation_stage(tmp_path):
    db=run.connect(tmp_path)
    for stage,count in (('A_evaluation',12),('B_training_data',2),('C_validation_data',2)):
        for i in range(count):run.enqueue(db,dict(kind='episode',stage=stage,group=str(i)))
    db.commit()
    class Pool:
        def __init__(self):self.submitted=[]
        def submit(self,fn,task,out):self.submitted.append(task['stage']);return object()
    pool=Pool();run.DISPATCH_CURSOR.clear()
    for _ in range(3):
        active={};run.dispatch(db,pool,active,1,tmp_path,'bulk')
    assert set(pool.submitted)=={'A_evaluation','B_training_data','C_validation_data'}
    db.close()
