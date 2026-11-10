import pickle
import numpy as np
import unittest
from large_pursuit.core import Config,World,State,PublicMap,Belief,clear,generate_case,from_initial,visibility
from large_pursuit.methods import Controller,Network,features,tensors,candidate_pool
import torch


def setup(n=5,boxes=()):
    c=Config(n,40,8);w=World(c,boxes)
    return c,w


def state(p,e,n=5):
    pp=np.full((n,2),-15.);ee=np.full((n,2),15.)
    pp[:len(p)]=p;ee[:len(e)]=e
    return State(pp,ee,np.ones(n,bool))


def test_persistent_multikill_dedup_and_repeat():
    c,w=setup();s=state([[0,0],[0,.1]],[[.2,0],[.3,0],[2.,0]])
    z=np.zeros((5,2));q=w.transition(s,z,z)
    assert q.alive.tolist()==[False,False,True,True,True]
    a=z.copy();a[0]=[1,0]
    r=w.transition(q,a,z)
    assert r.alive[2]  # strict distance < 1
    r=w.transition(r,a,z)
    assert not r.alive[2]
    assert np.allclose(r.p[0],[2,0])
    assert w.observe(r)['seen'].shape==(5,)


def test_wall_capture_and_swept_contact():
    c,w=setup(boxes=[[[.2,.3],[-2,2]]]);s=state([[0,0]],[[.5,0]])
    z=np.zeros((5,2));q=w.transition(s,z,z)
    assert q.alive[0]
    a=z.copy();a[0]=[1,0];r=w.transition(q,a,z)
    assert r.p[0,0]<.2 and clear(q.p[0],r.p[0],w.boxes)


def test_terminal_and_timeout_and_invalid():
    c,w=setup();s=state([[0,0]],[[.1,0]]*5);z=np.zeros((5,2))
    q=w.transition(s,z,z);assert w.done(q)
    s.t=7;s.e[:]=15;assert w.done(w.transition(s,z,z))
    with unittest.TestCase().assertRaises(ValueError):w.transition(s,np.zeros((2,2)),z)
    with unittest.TestCase().assertRaises(ValueError):w.transition(s,np.full((5,2),np.nan),z)


def test_hidden_information_and_replay():
    c,w=setup();public=PublicMap(c,[]);s=state([[0,0]],[[10,10]])
    a=w.observe(s);b=Belief(public,17,8);b.update(a)
    s.t=1;obs=w.observe(s);other=s.copy();other.e[:]=[12,12]
    alt=w.observe(other)
    for k in obs:np.testing.assert_equal(obs[k],alt[k])
    from large_pursuit.training import belief_state,restore_belief
    snapshot=pickle.dumps(belief_state(b))
    b.update(obs);expected=b.samples.copy()
    b=restore_belief(public,pickle.loads(snapshot));b.update(alt)
    np.testing.assert_array_equal(expected,b.samples)


def test_same_transition_in_search_and_environment_and_16v16():
    case=generate_case(16,'post_field',923,120);c=Config.scale(16);w=World(c,case['boxes'])
    public=PublicMap(c,case['boxes']);s=from_initial(case,0);controller=Controller(public,'AG',1)
    assert controller.world.transition.__func__ is w.transition.__func__
    rng=np.random.default_rng(7)
    for _ in range(c.horizon):
        a=rng.uniform(-1,1,(16,2));b=rng.uniform(-1,1,(16,2))
        s=w.transition(s,a,b)
    assert w.done(s) and s.p.shape==(16,2)
    assert np.isfinite(s.p).all()


def test_common_candidates_and_network_variable_sets():
    c,w=setup();public=PublicMap(c,[]);s=state([[0,0]],[[10,10]])
    obs=w.observe(s);model=Network()
    x=features(public,obs,s.e);mean,_,value=model(tensors(x))
    assert mean.shape==(1,5,2) and value.shape==(1,)
    pool=candidate_pool(public,s.p,s.e,s.alive,np.random.default_rng(1))
    copies=[]
    for method in ('POOL_UNIFORM','POOL_GEOM','POOL_LEARNED'):
        ctrl=Controller(public,method,17,model,simulations=2,depth=2)
        action=ctrl.act(obs,candidates=pool)
        assert action.shape==(5,2) and ctrl.last_stats['simulations']==2
        copies.append(pool.copy())
    np.testing.assert_array_equal(copies[0],copies[2])


def test_padding_and_sealed_access_and_dead_actor_mask():
    from large_pursuit.methods import stack_features,normal_logprob
    from large_pursuit.training import entries
    c,w=setup();public=PublicMap(c,[]);ss=state([[0,0]],[[10,10]])
    x=features(public,w.observe(ss),ss.e);old=dict(x);old.pop('actor_alive')
    batch=stack_features([x,old],'cpu')
    assert batch['actor_alive'].shape==(2,5)
    raw=torch.ones((1,5,2));mean=torch.zeros_like(raw);logstd=torch.zeros_like(raw)
    assert normal_logprob(raw,mean,logstd,torch.zeros((1,5))).item()==0
    with unittest.TestCase().assertRaises(ValueError):entries('/nonexistent',5,'sealed_ID')


def test_rootpuct_and_public_replay():
    from large_pursuit.rootpuct import rootpuct_evader
    c,w=setup();public=PublicMap(c,[]);ss=state([[0,0]],[[10,10]])
    a=rootpuct_evader(public,ss,np.random.default_rng(17),simulations=8,depth=3)
    b=rootpuct_evader(public,ss,np.random.default_rng(17),simulations=8,depth=3)
    np.testing.assert_array_equal(a,b)
    assert a.shape==(5,2) and np.max(np.abs(a))<=1
    b=Belief(public,17,16);b.update(w.observe(ss))
    for t in range(1,4):
        ss.t=t;b.update(w.observe(ss))
    particles,modes=b._replay(0)
    assert len(particles)>=512 and modes.shape==(len(particles),)
    assert np.max(np.abs(particles-[10,10]))<=3+1e-8
    assert not visibility(ss.p,particles,public.boxes,c.sensing).any()


def test_continuous_pocket_without_grid_attachment():
    from large_pursuit.core import legal,move
    c=Config(5,40,160)
    boxes=np.array([[[0.,.1],[0.,.4]],[[0.,.4],[.2005,.4]],[[0.,.4],[0.,.1995]]])
    public=PublicMap(c,boxes);q=np.array([.2,.2]);goal=np.array([2.,2.])
    assert legal(q,boxes,c.size)
    root=public.nearest(q)
    assert not clear(q,public.points[root],boxes)
    for _ in range(12):
        action=public.action(q,goal)
        nxt=move(q[None],action[None],np.ones(1,bool),boxes,c.size)[0]
        assert legal(nxt,boxes,c.size) and clear(q,nxt,boxes)
        q=nxt
    np.testing.assert_allclose(q,goal,atol=1e-6)


if __name__ == "__main__":
    suite=unittest.TestSuite(unittest.FunctionTestCase(fn) for name,fn in list(globals().items()) if name.startswith("test_"))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
