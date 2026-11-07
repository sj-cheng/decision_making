"""Meaningful public-feature and action interface invariants for the new arm."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from learning.nte_d_local import Network, Runtime, public_features
from solvers.nte_d_local import Controller, Node


class PublicFeatureTests(unittest.TestCase):
    def features(self):
        rng = np.random.default_rng(29)
        w = rng.random(64); w /= w.sum()
        return public_features([[[-1.,1.],[-5.,5.]]], rng.uniform(-8,8,(64,2)),w,
            np.array([-3.,0.]), [dict(step=3,position=[-3.,0.],target=None,remaining=77)],1.1,77,8,False)

    def test_weight_and_set_permutation_invariance(self):
        torch.manual_seed(17)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'net.npz'; Network('policy').export(path); net=Runtime(path)
            x=self.features(); a=net(x)
            order=np.random.default_rng(1).permutation(64)
            x['particles']=x['particles'][order]; x['weights']=x['weights'][order]
            np.testing.assert_allclose(np.asarray(a),np.asarray(net(x)),atol=1e-12,rtol=0)

    def test_nonuniform_weight_matters(self):
        torch.manual_seed(17)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'net.npz'; Network('policy').export(path); net=Runtime(path)
            x=self.features(); a=net(x)
            x['weights']=np.ones(64)/64.;x['particles'][:,2]=x['weights']
            self.assertGreater(float(np.max(np.abs(np.asarray(a)-np.asarray(net(x))))),1e-8)

    def test_invalid_weights_rejected(self):
        for w in (np.zeros(64), np.full(64,np.nan), np.ones(64)/32):
            with self.assertRaises(ValueError):
                public_features([],np.zeros((64,2)),w,np.zeros(2),[],1.1,80,8,False)

    def test_hidden_target_absent_from_action_API(self):
        import inspect
        for f in (Controller.features,Controller.propose,Controller.select,Controller.rollout_action):
            names=inspect.signature(f).parameters
            self.assertFalse(set(names)&{'target','state','environment','target_id','mode'})

    def test_node_condition_is_public_history(self):
        n=Node(np.zeros(2),np.ones((2,9))/18,-1,0,[],False)
        self.assertFalse(hasattr(n,'target'))
        self.assertFalse(hasattr(n,'mode'))
        self.assertEqual(n.observation,-1)


if __name__=='__main__':
    unittest.main()
