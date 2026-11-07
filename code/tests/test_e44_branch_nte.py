import inspect
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from learning.e44_policy import Network,Runtime,action_log_density,weighted_nll,KINDS
from learning.nte_d_local import public_features
from solvers.e44_routes import route_arrays,MAX_ROUTES
from solvers.e44_branch_nte import Controller
from solvers.frozen_d_search_interface import loader
from audit_frozen_d_search_interface import observation,independent_traces
from corner_estimator_scenarios import construct
from run_e44_branch_nte import registered_latencies


class E44Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env,cls.packet=construct('wall_corner',4,1.1,944317,6.4,0,pause=0)

    def controller(self):
        p=self.packet;c=Controller(p['map'],p['motion']['pursuer_axis_limit'],317,'route','focused')
        for item in p['history'][:-1]:
            obs=observation(item);c.estimator.update(obs);c.observe_history(obs)
        c.act(observation(p['history'][-1]),fixed_simulations=8)
        c.deadline=float('inf')
        return c

    def test_public_interface_and_exact_repeat(self):
        a,b=self.controller(),self.controller()
        self.assertEqual(a.public_signature(),b.public_signature())
        self.assertEqual(a.last_stats['root_actions'],b.last_stats['root_actions'])
        for method in (Controller.propose,Controller.select,Controller.features,Controller.rollout_action):
            self.assertFalse(set(inspect.signature(method).parameters)&{'target','target_id','environment','state'})
        traces=np.concatenate(a.traces)
        self.assertFalse(any(independent_traces(traces,a.public.boxes,a.public.speed)))

    def test_route_permutation_and_legality(self):
        c=self.controller();x=c.features(c.root)
        samples=c.estimator.particles;weights=c.estimator.particle_weights;labels=c.estimator.labels
        args=(c.public.speed,c.public.vertices,c.public.dist,c.public.boxes,c.motion_boxes)
        result=route_arrays(c.root.at,samples,weights,labels,*args)
        order=np.random.default_rng(29).permutation(64)
        other=route_arrays(c.root.at,samples[order],weights[order],labels[order],*args)
        for a,b in zip(result[:5],other[:5]):np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
        self.assertTrue(result[1][0]);self.assertLessEqual(result[1].sum(),12)

    def test_common_density_finite_at_endpoints_and_multimodal(self):
        logits=torch.zeros(1,2,dtype=torch.double)
        mu=torch.tensor([[[-2.,0.],[2.,0.]]],dtype=torch.double)
        logs=torch.full((1,2,2),-.5,dtype=torch.double)
        points=torch.tensor([[[-1.,1.],[1.,-1.],[0.,0.]]],dtype=torch.double)
        self.assertTrue(torch.isfinite(action_log_density((logits,mu,logs),points)).all())
        x=torch.tensor([[[-.95,0.],[0.,0.],[.95,0.]]],dtype=torch.double)
        density=action_log_density((logits,mu,logs),x)[0]
        self.assertGreater(float(density[0]),float(density[1]))
        self.assertGreater(float(density[2]),float(density[1]))

    def test_registered_timing_charges_initialization_once(self):
        row=dict(initial_total_ms=213.,latencies_ms=[193.,197.])
        self.assertEqual(registered_latencies(row),[213.,197.])
        self.assertEqual(sum(t>200 for t in registered_latencies(row)),1)
        self.assertEqual(registered_latencies(dict(initial_total_ms=20.,latencies_ms=[])),[])

    def test_nine_architectures_parameter_match_numpy_and_gradient(self):
        c=self.controller();x=c.features(c.root)
        counts=[]
        with tempfile.TemporaryDirectory() as folder:
            for kind in KINDS:
                torch.manual_seed(17);m=Network(kind).double();path=Path(folder)/(kind+'.npz');m.export(path)
                counts.append(sum(p.numel() for p in m.parameters()))
                tx={k:torch.from_numpy(v)[None] for k,v in x.items()}
                with torch.no_grad():logits,location,logs=m(tx)
                expected=(torch.softmax(logits,-1).numpy()[0],location.numpy()[0],logs.numpy()[0])
                for a,b in zip(expected,Runtime(path)(x)):np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
                tx.update(label_actions=torch.tensor([[[-.8,0.],[.8,0.]]],dtype=torch.double),
                          label_weights=torch.tensor([[.5,.5]],dtype=torch.double))
                loss=weighted_nll(m,tx);loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters()))
        self.assertLess(max(counts)/min(counts),1.02)

    def test_set_permutation_preserves_physical_action_density(self):
        c=self.controller();x=c.features(c.root)
        y={k:v.copy() for k,v in x.items()}
        rng=np.random.default_rng(43);order=rng.permutation(64)
        y['particles'],y['weights']=y['particles'][order],y['weights'][order]
        order=rng.permutation(MAX_ROUTES)
        for k in ('routes','route_mask','anchors'):y[k]=y[k][order]
        for kind in KINDS:
            torch.manual_seed(17);m=Network(kind).double()
            p=torch.tensor([[[-.3,.2],[.5,-.6]]],dtype=torch.double)
            with torch.no_grad():
                a=action_log_density(m({k:torch.from_numpy(v)[None] for k,v in x.items()}),p)
                b=action_log_density(m({k:torch.from_numpy(v)[None] for k,v in y.items()}),p)
            np.testing.assert_allclose(a.numpy(),b.numpy(),atol=1e-12,rtol=0)


if __name__=='__main__':unittest.main()
