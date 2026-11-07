"""Targeted scientific checks; runnable without pytest."""
import copy
import json
import time
import numpy as np
import torch
from solvers.branch_support import GridMap,Support,cell_certainly_visible,bounds,cell_id
from solvers.branch_nte import Controller,Search,candidate_bank
from solvers.local_pursuit import Observation,safe
from learning.branch_nte import Policy,Runtime,encoded,loss


def test_negative_visibility():
    from verify_l45_occlusion_scenarios import visible,legal
    boxes=np.array([[[-1.,1.],[-4.,4.]],[[2.,3.],[-1.,1.]]])
    rng=np.random.default_rng(63025);certified=0
    for _ in range(3000):
        at=rng.uniform(-9.8,9.8,2);i=int(rng.integers(6400));lo,hi=bounds(i)
        if not legal(at,boxes.tolist()):continue
        if cell_certainly_visible(at,lo,hi,boxes):
            certified+=1
            for _ in range(12):
                q=rng.uniform(lo,hi)
                assert visible(at.tolist(),q.tolist(),boxes.tolist())
    assert certified>50
    # The center can be visible while part of a cell is beyond sensor range.
    assert not cell_certainly_visible(np.array([0.,0.]),np.array([5.75,0.]),np.array([6.,.25]),np.empty((0,2,2)))


def test_motion_and_routes():
    p=GridMap([[[-1.,1.],[-5.,5.]]],1.3)
    s=Support(p);s.update(Observation(0,np.array([-3.,6.]),np.array([2.,6.]),80))
    assert s.covers(np.array([2.,6.]))
    for t in range(1,7):
        observer=np.array([-3.,6.-t]);s.update(Observation(t,observer,None,80-t))
        assert s.covers(np.array([2.,6.-t]))
    f,assignment,groups,_=s.branches()
    assert 2<=len(groups)<=4
    assert np.isclose(f[:,8].sum(),1)
    assert np.all(assignment[s.mask]>=0)
    samples,_=s.sample(np.array([42],np.int64),64)
    from verify_l45_occlusion_scenarios import visible
    assert all(not visible(s.at.tolist(),q.tolist(),p.boxes.tolist()) for q in samples)


def test_network_parity_and_losses():
    torch.manual_seed(0);model=Policy();p=GridMap([],1.1)
    history=[dict(step=0,position=[0.,0.],target=[5.8,1.],remaining=80),dict(step=1,position=[0.,.2],target=None,remaining=79)]
    s=Support(p)
    for r in history:s.update(Observation(r['step'],np.array(r['position']),None if r['target'] is None else np.array(r['target']),r['remaining']))
    f,_,_,_=s.branches();v=encoded(p,history,f)
    batch={k:torch.tensor(x[None],dtype=torch.float32) for k,x in v.items()}
    for mode in ('M','H'):
        with torch.no_grad():logits,mu,lv=model(**batch,mode=mode)
        w,m,l=Runtime(model,p,mode)(history,f)
        assert np.allclose(logits.softmax(-1)[0].numpy(),w,atol=2e-6)
        assert np.allclose(mu[0].numpy(),m,atol=2e-6)
        assert np.allclose(lv[0].numpy(),l,atol=2e-6)
        a=torch.tensor([[[-1.,0.],[1.,0.]]]);vis=torch.tensor([[10.,10.]])
        objective,nll=loss(logits,mu,lv,a,vis,torch.tensor([0]),mode)
        assert torch.isfinite(objective) and torch.isfinite(nll)
    # Opposite targets remain separate training actions, not a zero label.
    logits=torch.zeros((1,4));mu=torch.zeros((1,4,2));lv=torch.zeros_like(mu)
    _,nll=loss(logits,mu,lv,torch.tensor([[[-1.,0.],[1.,0.]]]),torch.ones((1,2)),torch.tensor([0]),'M')
    _,collapsed=loss(logits,mu,lv,torch.zeros((1,1,2)),torch.ones((1,1)),torch.tensor([0]),'M')
    assert nll>collapsed+.4


def test_bank_and_search_replay():
    p=GridMap([],1.1);f=np.zeros((4,12));f[:3,11]=1
    w=np.array([.7,.2,.1,0.]);mu=np.ones((4,2))*.25;lv=np.full((4,2),-4.)
    rng=np.array([5],np.int64)
    a,priors,sources,invalid,_,allocation=candidate_bank(p,np.zeros(2),f,mu,lv,w,rng)
    assert sum(allocation)==8 and allocation.tolist()==[3,3,2,0]
    assert len(a)<=16 and np.isclose(priors.sum(),1)
    for k in range(3):assert np.isclose(sum(priors[i] for i,ls in enumerate(sources) if k in ls),.5*w[k])
    assert np.isclose(sum(priors[i] for i,ls in enumerate(sources) if 4 in ls),.5)
    wall=GridMap([[[-1.,1.],[-5.,5.]]],1.1)
    # Many illegal proposals become the same zero action. Their merged node
    # must have the same uniform prior as each other unique action.
    z,prior,labels,bad,_,_=candidate_bank(wall,np.array([-1.01,0.]),f,np.ones((4,2)),np.full((4,2),-8.),w,np.array([5],np.int64),False,True)
    assert bad>1 and any(len(label)>1 for label in labels)
    assert np.allclose(prior,np.full(len(z),1/len(z)))
    c=Controller(p,'T',11,deadline_ms=0);c.update(Observation(0,np.zeros(2),np.array([4.,0.]),80))
    original=c.snapshot();obs=Observation(1,np.array([0.,.2]),np.array([4.,.5]),79)
    first=c.act(obs,fixed_simulations=9);root=copy.deepcopy(c.last_stats['root']);rng=c.rng.copy()
    assert sum(r['visits'] for r in root)==9
    c.restore(original);second=c.act(obs,fixed_simulations=9)
    assert np.array_equal(first,second) and root==c.last_stats['root'] and np.array_equal(rng,c.rng)


def test_puct_and_gate():
    import run_l45_branch_nte as run
    rows=[]
    for method in ('T-first','E-first'):
        for i in range(96):rows.append(dict(key=method+str(i),method=method,capture=int(i<90),completion_cost=20. if method=='T-first' else 19.,method_execution_failure=0))
    assert run.teacher_gate(rows)['passed']
    rows[-1]['method_execution_failure']=1
    assert not run.teacher_gate(rows)['passed']
    rows[-1]['method_execution_failure']=0
    for r in rows:
        if r['method']=='E-first':r['completion_cost']=19.01
    assert not run.teacher_gate(rows)['passed']


def test_expert_resume():
    p=GridMap([],1.1);own=Controller(p,'T',123,deadline_ms=0)
    own.update(Observation(0,np.array([0.,0.]),np.array([4.,0.]),80))
    obs=Observation(0,np.zeros(2),np.array([4.,0.]),80)
    search=Search(own,obs,np.array([4.,0.]));journal_copy=copy.deepcopy(search)
    for _ in range(3):
        search.one();journal_copy.apply_delta(copy.deepcopy(search.latest_delta))
    assert np.array_equal(search.c.rng,journal_copy.c.rng)
    assert search.root.visits==journal_copy.root.visits
    saved=copy.deepcopy(search);a,root=search.run(8,start=3)
    b,root2=saved.run(8,start=3)
    assert np.array_equal(a,b) and root==root2 and search.completed==8
    c,root3=journal_copy.run(8,start=3)
    assert np.array_equal(a,c) and root==root3


def run_unit_checks():
    results=[]
    for fn in (test_negative_visibility,test_motion_and_routes,test_network_parity_and_losses,test_bank_and_search_replay,test_puct_and_gate,test_expert_resume):
        started=time.perf_counter();fn();results.append(dict(name=fn.__name__,passed=True,seconds=time.perf_counter()-started))
    return results
