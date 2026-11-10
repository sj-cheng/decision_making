"""Protocol tests: no scientific training interaction is performed here."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import e49_adaptation as a
import e49_controllers as c

class E49AdaptationTests(unittest.TestCase):
    def config(self):return dict(entries=[dict(split='train') for _ in range(240)]+[dict(split='validation') for _ in range(24)])
    def test_requires_frozen_h(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):a.adapt(Path(tmp),self.config(),c.method_specs(),40,{},[0])
    def test_untriggered_uses_no_gpu_or_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'horizon_selection.json').write_text(json.dumps(dict(horizon=40)))
            with patch.object(a,'_device',side_effect=AssertionError('No GPU required')):
                result=a.adapt(out,self.config(),c.method_specs(),40,{},[0])
            self.assertEqual(len(result),22)
    def test_all_seeds_stop_and_old_model_wins_ties(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'horizon_selection.json').write_text(json.dumps(dict(horizon=40)))
            calls=[]
            def train(out,family,seed,source,labels,epochs,device):
                calls.append((seed,epochs))
                return dict(checkpoint=f'new_{seed}_{epochs}.npz',initial_action_loss=-2.,validation_action_loss=-2.02)
            with patch.object(a,'_device',return_value=torch.device('cpu')),patch.object(a,'_dataset',return_value={}),patch.object(a,'_labels',return_value={}),patch.object(a,'_supervised_train',side_effect=train),patch.object(a,'_validate',return_value=0.):
                result=a.adapt(out,self.config(),c.method_specs(),40,dict(Single=dict(triggered=True)),[0])
            self.assertEqual(calls,[(17,5),(29,5),(43,5)])
            self.assertTrue(all(not s['adapted'] for s in result if s['arm']=='SINGLE_PRIOR'))
            with patch.object(a,'_device',side_effect=AssertionError('Cached family needs no GPU')):
                a.adapt(out,self.config(),c.method_specs(),40,dict(Single=dict(triggered=True)),[0])
    def test_shared_l5_weights_and_early_validation_ties(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'horizon_selection.json').write_text(json.dumps(dict(horizon=40)))
            calls=[]
            def train(out,family,seed,source,labels,epochs,device):
                calls.append((seed,epochs));return dict(checkpoint=f'new_{seed}_{epochs}.npz',initial_action_loss=2.,validation_action_loss=1.8)
            def validate(out,entries,spec,horizon,stage,cpus):return .02 if spec['checkpoint'].startswith('new') else 0.
            with patch.object(a,'_device',return_value=torch.device('cpu')),patch.object(a,'_dataset',return_value={}),patch.object(a,'_labels',return_value={}),patch.object(a,'_supervised_train',side_effect=train),patch.object(a,'_validate',side_effect=validate):
                result=a.adapt(out,self.config(),c.method_specs(),40,dict(L5=dict(triggered=True)),[0])
            self.assertEqual(len(calls),12)
            for seed in c.SEEDS:
                paths={s['checkpoint'] for s in result if s['arm'] in ('A3','A5') and s['train_seed']==seed}
                self.assertEqual(paths,{f'new_{seed}_5.npz'})

if __name__=='__main__':unittest.main()
