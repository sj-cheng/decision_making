import copy
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from learning.memory_nte import Policy,features,inference,observation_array,weighted_nll
from memory_nte_scenarios import construct
from verify_l45_occlusion_scenarios import verify_case
from run_l45_memory_nte import gate


def test_encoder_masks_and_cpu_parity():
    torch.manual_seed(0);model=Policy().double()
    records=[dict(step=i,position=[-.1*i,0],target=[2.,3.] if i in (0,19) else None) for i in range(25)]
    array=observation_array(records);boxes=np.array([[[-1.,1.],[-3.,3.]]])
    for memory in (False,True):
        b,h,m,p=features(boxes,array,len(array),1.3,memory)
        assert m.sum()==(17 if memory else 1)
        assert np.all(h[m==0]==0)
        mu,lv=inference(boxes,array,len(array),1.3,memory,model.export())
        args=[torch.tensor(x).unsqueeze(0) for x in (b,np.ones(len(b)),h,m,p)]
        expected=model(*args)
        np.testing.assert_allclose(mu,expected[0].detach()[0],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(lv,expected[1].detach()[0],rtol=1e-10,atol=1e-10)
        poisoned=array.copy();poisoned[array[:,3]==0,4:]=1e6
        actual=inference(boxes,poisoned,len(array),1.3,memory,model.export())
        np.testing.assert_array_equal(actual[0],mu)


def test_weighted_labels_preserve_variance():
    actions=torch.tensor([[[-1.,0.],[1.,0.]]]);visits=torch.ones(1,2)
    a=weighted_nll(torch.zeros(1,2),torch.zeros(1,2),actions,visits)
    b=weighted_nll(torch.zeros(1,2),torch.full((1,2),-8.),actions,visits)
    assert b>a


def test_new_geometries_are_environment_valid():
    for family in ('long_wall','wall_corner','multi_route'):
        for H,height in ((4,5.54),(6,8.54)):
            e,p=construct(family,H,1.3,6225,height)
            assert verify_case(e,p)['passed']


def test_teacher_gate_exact_boundaries():
    def rows(g,t,e,gcost=40,tcost=40,ecost=40,fail=0):
        return [dict(key=f'{m}{i}',method=m,capture=int(i<n),completion_cost=c,method_execution_failure=int(m=='E' and i<fail))
                for m,n,c in (('G',g,gcost),('T',t,tcost),('E',e,ecost)) for i in range(96)]
    assert gate(rows(88,90,93))['passed']
    assert not gate(rows(88,90,92))['passed']
    assert gate(rows(90,90,90,ecost=36))['passed']
    assert not gate(rows(90,90,90,ecost=36.01))['passed']
    assert not gate(rows(88,90,93,fail=1))['passed']


def test_simulation_copy_changes_only_expansion_and_sensor_history():
    from solvers import local_pursuit as original
    from solvers import memory_nte as new
    import inspect
    old=inspect.getsource(original.simulation.py_func)
    actual=inspect.getsource(new.neural_simulation.py_func)
    # The common reward, opponent and rollout expressions must remain intact.
    for line in old.splitlines():
        if any(x in line for x in ('rewards[depth] =','other, _ =','action, _ = proposal','score = totals','width = max','totals[path[j]]')):
            assert line in actual


def test_neural_search_replay_and_limits():
    from solvers.local_pursuit import PublicMap,Observation
    from solvers.memory_nte import NeuralController
    torch.manual_seed(2);weights=Policy().export()
    public=PublicMap([],1.3)
    observation=Observation(0,np.array([-4.,0.]),np.array([0.,0.]),80)
    for method in ('N','H'):
        c=NeuralController(public,method,124,weights,simulations=7,deadline_ms=0.)
        before=c.snapshot()
        a=c.act(observation,fixed_simulations=7)
        after=c.snapshot();stats=c.last_stats.copy()
        c.restore(before)
        b=c.act(observation,fixed_simulations=7)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(after['rng'],c.rng)
        np.testing.assert_array_equal(after['particles'],c.particles)
        assert np.max(np.abs(a))<=1.3
        assert stats['simulations']==7 and stats['network_calls']>0
        assert sum(r['visits'] for r in stats['root'])==7
