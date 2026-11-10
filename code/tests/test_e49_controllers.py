"""Engineering acceptance for process-local E49 inference and migration."""
import copy
from pathlib import Path
import numpy as np
import unittest
import torch
import e49_controllers as c
from learning.e45b_joint_network import public_features as old_features,Runtime
from learning.e46_geometry_teacher import RankRuntime


def boxes24():
    return np.array([[[2.+(k%6),2.15+(k%6)],[3.+(k//6),3.15+(k//6)]] for k in range(24)])


def observation(h=40,step=0,one=False):
    return dict(step=step,remaining=h-step,positions=[[-4.,-2.],[-4.,2.]],active_p=[True,True],active_e=[True,not one],
                reports=[[[-1.,-2.],[-1.,-2.]], [None,None] if one else [[-1.,2.],[-1.,2.]]])


def test_method_inventory_sources():
    specs=c.method_specs()
    assert len(specs)==22 and len({s['arm'] for s in specs})==12
    assert all(Path(s['checkpoint']).is_file() for s in specs if s.get('checkpoint'))
    assert {s['train_seed'] for s in specs if s['arm']=='PPO_C'}=={17,29,43}


def check_full24_features_h_and_sparse_equivalence(h):
    c.install(h)
    obs=observation(h);positions=np.array(obs['positions']);samples=np.tile(np.array([[-1.,-2.],[-1.,2.]])[:,None,:],(1,64,1));weights=np.ones((2,64))/64
    args=(positions,obs['active_p'],obs['active_e'],samples,weights,[samples[0,0],samples[1,0]],[obs],1.,h)
    x=c.public_features(boxes24(),*args)
    assert x['maps'].shape==(24,4) and x['map_mask'].sum()==24
    assert x['public'][2]==1 and x['history'][0,13]==1
    if h==80:
        old=old_features(boxes24()[:8],*args);new=c.public_features(boxes24()[:8],*args)
        for key in old:
            if key in ('maps','map_mask'):assert np.array_equal(old[key],new[key][:8])
            else:assert np.array_equal(old[key],new[key])
        specs=c.method_specs()
        for arm,rt in [('SINGLE_PRIOR',Runtime),('A3',RankRuntime)]:
            runtime=rt(next(s['checkpoint'] for s in specs if s['arm']==arm))
            a=runtime(old) if arm=='SINGLE_PRIOR' else runtime(old,np.zeros((2,4)))
            b=runtime(new) if arm=='SINGLE_PRIOR' else runtime(new,np.zeros((2,4)))
            if arm=='SINGLE_PRIOR':
                for aa,bb in zip(a[0],b[0]):assert np.allclose(aa,bb,atol=1e-12)
                assert abs(a[1]-b[1])<1e-12
            else:assert np.allclose(a,b,atol=1e-12)


def check_all22_full24_inference_and_persistent_actions(one):
    c.install(40)
    for spec in c.method_specs():
        own=c.make_controller({'map':boxes24()},dict(spec,seed=917))
        # Negative public-history timestamps count physical transitions.
        own.observe(observation(40,-1,one))
        result=own.act(observation(40,0,one),fixed_simulations=1)
        assert result.shape==(2,2) and np.isfinite(result).all(),spec['name']
        assert abs(result).max()<=1.+1e-10,spec['name']
        assert np.array_equal(own.public.boxes,boxes24()),spec['name']
        assert len(own.belief.estimators)==2
        if one and spec['arm'] in ('IG','AG','RouteAPF','MAPPO_RouteAPF'):
            assert set(own.last_stats['assignment'])=={'0','1'}
        if hasattr(own,'target_map'):assert own.target_map.speed==1.5
        for evidence in getattr(own,'e49_simulated_evidence',[]):
            assert evidence['next_active_p'].all()


def test_semantic_flat_migration_is_exact_in_old_columns():
    c.install()
    spec=next(s for s in c.method_specs() if s['arm']=='MAPPO_RouteAPF')
    model,payload=c.load_mappo(spec['checkpoint'])
    for name,old_dim in [('actor',844),('critic',858)]:
        migrated=getattr(model,name).state_dict();columns=c.semantic_columns(old_dim)
        key='base.mlp.fc1.0.weight'
        assert torch.equal(migrated[key][:,columns],payload[name][key])
        added=np.setdiff1d(np.arange(old_dim+80),columns)
        assert torch.count_nonzero(migrated[key][:,added])==0
        for suffix in ('weight','bias'):
            key='base.feature_norm.'+suffix
            assert torch.equal(migrated[key][columns],payload[name][key])


def test_frozen_h_precedes_training_budget():
    from e49_learning import AdaptationBudget,continue_after_pilot,select_checkpoint
    budget=AdaptationBudget('Single',80,False)
    with unittest.TestCase().assertRaises(RuntimeError):budget.consume(epochs=1)
    budget.horizon_frozen=True;budget.consume(epochs=5,states=20000)
    with unittest.TestCase().assertRaises(RuntimeError):budget.consume(states=1)
    assert not continue_after_pilot({s:dict(utility_gain=0.,action_loss_reduction=.04) for s in c.SEEDS},True)
    assert select_checkpoint([dict(steps=0,validation_utility=.1),dict(steps=5,validation_utility=.1)])['steps']==0


class E49ControllersTests(unittest.TestCase):
    def test_sources(self):test_method_inventory_sources()
    def test_features(self):
        for h in (40,80,160):check_full24_features_h_and_sparse_equivalence(h)
    def test_all_models(self):
        for one in (False,True):check_all22_full24_inference_and_persistent_actions(one)
    def test_migration(self):test_semantic_flat_migration_is_exact_in_old_columns()
    def test_training_gate(self):test_frozen_h_precedes_training_budget()

if __name__=='__main__':unittest.main()
