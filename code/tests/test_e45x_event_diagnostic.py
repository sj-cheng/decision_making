import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from learning.e45x_event_diagnostic import (FrozenActor,prepare_public,initial_world,
    root_candidates,counterfactual,advance,EventController,Node,ETA,TIME_COST,LeafReadout)
from solvers.e45_team_adapter import public_observation
from corner_estimator_scenarios import original_model

W=Path(__file__).resolve().parents[2]/'current/e45w_joint_ppo_and_planning_20261103'


@pytest.fixture(scope='module')
def actor():
    cfg=json.loads((W/'experiment.json').read_text())
    return FrozenActor(cfg['models']['17']['pt'],'cpu')


def example():
    boxes=np.zeros((0,2,2)); positions=np.array([[-2.,0.],[2.,0.]])
    targets=np.array([[0.,2.],[0.,-2.]])
    obs=public_observation(79,positions,targets,np.ones(2,bool),np.ones(2,bool),boxes)
    root=dict(id='public-unit',controller_seed=3411,history=[obs],opponent='fixed_D_kernel')
    case=dict(map=boxes.tolist(),speed=1.1)
    return root,case


def test_posterior_world_and_fixed_candidates_are_public_and_reproducible(actor):
    root,case=example();prepared=prepare_public(root,case,actor)
    a,_,_=initial_world(root,prepared,2,'holdout');b,_,_=initial_world(root,prepared,2,'holdout')
    np.testing.assert_array_equal(a,b)
    candidates,sources=root_candidates(root,prepared,actor)
    again,_=root_candidates(root,prepared,actor)
    np.testing.assert_array_equal(candidates,again)
    assert candidates.shape==(6,2,2) and len(sources)==6
    assert any(np.max(np.abs(c))==0 for c in candidates)
    assert not any(k in root for k in ('state','target_truth','initial_hidden_state'))


def test_identical_interventions_have_identical_common_random_continuations(actor):
    root,case=example();prepared=prepare_public(root,case,actor)
    before={k:v.detach().cpu().numpy().copy() for k,v in actor.model.state_dict().items()}
    candidate=np.zeros((2,2))
    a=counterfactual(root,prepared,actor,candidate,5,'independent-reference',examples=True)
    b=counterfactual(root,prepared,actor,candidate,5,'independent-reference',examples=True)
    for key in ('utility','double_capture','first_capture','short_scores','initial_world_sha256'):
        assert a[key]==b[key]
    assert a['decisions']==1
    for k,v in actor.model.state_dict().items():np.testing.assert_array_equal(v.cpu().numpy(),before[k])


def test_physical_double_capture_retires_both_pairs_and_reward_is_once():
    p=original_model(np.zeros((0,2,2)),1.1)
    state=np.zeros((p.state_dim,1));state[:8,0]=[.5,0,3.5,0,0,0,3,0]
    state[8,0]=79;state[p.active_idxs,0]=1
    class Still:
        def seeded_state(self,*args):pass
        def search(self,*args,**kwargs):return SimpleNamespace(best_action=np.zeros(8))
    prep=SimpleNamespace(problem=p,case=dict(map=[],speed=1.1))
    nxt,obs,reward,done,captures,_=advance(prep,state,np.zeros((2,2)),Still(),[0,0],[0,0],'physical-unit')
    assert done and captures==2 and reward==1-TIME_COST
    assert not any(obs['active_p']) and not any(obs['active_e'])


def tiny_tree(targets,positions,arm='event16'):
    c=EventController.__new__(EventController);c.arm=arm;c.horizon=16;c.remaining=20;c.runtime=None;c.use_value=False
    c.candidates=np.zeros((1,2,2));c.root=Node(positions,[True,True],[True,True],
        [np.ones((1,9))/9,np.ones((1,9))/9],[None,None],0,[])
    c.public=SimpleNamespace(boxes=np.zeros((0,2,2)),speed=1.1)
    c.motion_boxes=np.zeros((0,2,2));c.graphs=[]
    for t in targets:
        c.graphs.append((np.asarray(t)[None],np.zeros((1,1),np.int32),np.ones(1,np.int32),
                         np.ones((1,1,9)),np.zeros((0,2,2)),0,None))
    c.rng=np.array([123],np.int64);c.world_rng=np.random.default_rng(15)
    c.allowed=lambda:None;c.operation=lambda f,*args:f(*args)
    c.potential=lambda node:-4. if node.depth==0 else -3.
    c.traces=[];c.nodes=1;c.parameters=lambda node:None
    return c


def test_terminal_double_capture_does_not_call_or_double_count_leaf():
    c=tiny_tree([[.5,0],[3.5,0]],[[0,0],[3,0]])
    c.leaf=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('Terminal must bypass leaf'))
    c.trajectory()
    assert np.isclose(c.root.totals[0],1-TIME_COST-ETA*(-4.))


def test_first_capture_bootstrap_uses_residual_public_node_and_cancels_potential():
    c=tiny_tree([[.5,0],[-8,-8]],[[0,0],[9,9]])
    called=[]
    def leaf(node,event=False):
        assert node.ap.sum()==1 and node.ae.sum()==1
        assert event
        called.append(node.depth)
        return .5-ETA*c.potential(node)
    c.leaf=leaf;c.trajectory()
    assert called==[1]
    assert np.isclose(c.root.totals[0],-TIME_COST+.5-ETA*(-4.))


def test_leaf_readout_obeys_raw_reward_bounds_and_declares_missing_residual(tmp_path):
    path=tmp_path/'head.npz'
    np.savez(path,mean=np.zeros(2),scale=np.ones(2),generic=np.array([0,0,10.]),
             residual=np.array([0,0,-10.]),residual_available=True)
    h=LeafReadout(path)
    assert h.raw(np.zeros(2),20)==1-TIME_COST
    assert h.raw(np.zeros(2),20,True)==-TIME_COST*20


def test_development_labels_cannot_change_fitted_heads(tmp_path):
    from run_e45x_event_diagnostic import fit_heads,write
    folder=tmp_path/'counterfactuals';folder.mkdir()
    train=[dict(z=[i/64.,(i%3)/3.],label=.2+i/200.,phase=1,remaining=20) for i in range(64)]
    dev=[dict(z=[.3,.5],label=-10.,phase=1,remaining=20)]
    write(folder/'train.json',dict(examples=train));write(folder/'development.json',dict(examples=dev))
    rows=[dict(key='train',split='train'),dict(key='development',split='development')]
    fit_heads(tmp_path,rows,dict(ridge_regularization=.01))
    a=LeafReadout(tmp_path/'leaf_readouts.npz').arrays
    dev[0]['label']=10.;write(folder/'development.json',dict(examples=dev))
    fit_heads(tmp_path,rows,dict(ridge_regularization=.01))
    b=LeafReadout(tmp_path/'leaf_readouts.npz').arrays
    for key in a:np.testing.assert_array_equal(a[key],b[key])
