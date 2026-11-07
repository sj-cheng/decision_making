"""Lookahead is an execution optimization, not a change to the PPO batches."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from learning.local_rl import PolicyBundle
from train_search_rl import episode_seed, wire_weights
from train_search_rl_lookahead import LookaheadRollouts


class LookaheadTests(unittest.TestCase):
    def test_no_speculation_across_opponent_pool_change(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            torch.save(dict(counters=[498,498],config=dict(seed=0),history=[]),path/'ppo_latest.pt')
            scheduler = LookaheadRollouts(path)
            tasks = [dict(seed=episode_seed(0,2,0,index),role=0,kind='ppo') for index in (498,499)]
            self.assertIsNone(scheduler.predict_next(tasks))

    def test_same_learned_models_rng_and_norms_as_sequential_training(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); torch.manual_seed(42); torch.set_num_threads(1)
            PolicyBundle().save(path/'initial.pt')
            common=['--stage','ppo','--input',str(path/'initial.pt'),'--episodes','6',
                    '--episodes-per-update','2','--workers','4','--ppo-epochs','1',
                    '--validation-episodes','1','--eval-every','6','--depth','1',
                    '--opponent-budget','1','--budget','1']
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/mpl')
            for script, name in [('train_search_rl.py','serial'),('train_search_rl_lookahead.py','overlap')]:
                result=subprocess.run([sys.executable,str(CODE/script),*common,'--output',str(path/name)],
                                      capture_output=True,text=True,env=env,timeout=120)
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            serial=torch.load(path/'serial/ppo_latest.pt',weights_only=False)
            overlap=torch.load(path/'overlap/ppo_latest.pt',weights_only=False)
            self.assertEqual(serial['counters'],[6,6]); self.assertEqual(overlap['counters'],[6,6])
            for kind in ('actors','critics'):
                for left,right in zip(serial['bundle'][kind],overlap['bundle'][kind]):
                    for key in left:torch.testing.assert_close(left[key],right[key],rtol=0.,atol=0.)
            self.assertEqual(serial['bundle']['norms'],overlap['bundle']['norms'])
            self.assertEqual(serial['selector_rng'],overlap['selector_rng'])
            torch.testing.assert_close(serial['rng']['torch'],overlap['rng']['torch'],rtol=0.,atol=0.)
            records=[json.loads(line) for line in (path/'overlap/lookahead_metrics.jsonl').read_text().splitlines()]
            self.assertTrue(any(r['reused_prefetch'] for r in records))
            self.assertEqual(overlap['config']['rollout_backend'],'paired_role_lookahead_v1')
            commands = [
                [sys.executable,str(CODE/'train_search_rl.py'),*common,'--output',str(path/'split'),'--max-updates','2'],
                [sys.executable,str(CODE/'train_search_rl_lookahead.py'),*common,'--output',str(path/'fork'),
                 '--fork-from',str(path/'split/ppo_latest.pt'),'--max-updates','1'],
                [sys.executable,str(CODE/'train_search_rl_lookahead.py'),*common,'--output',str(path/'fork'),
                 '--fork-from',str(path/'split/ppo_latest.pt')],
            ]
            for command in commands:
                result=subprocess.run(command,capture_output=True,text=True,env=env,timeout=120)
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            resumed=torch.load(path/'fork/ppo_latest.pt',weights_only=False)
            for kind in ('actors','critics'):
                for left,right in zip(serial['bundle'][kind],resumed['bundle'][kind]):
                    for key in left:torch.testing.assert_close(left[key],right[key],rtol=0.,atol=0.)
            self.assertEqual(serial['bundle']['norms'],resumed['bundle']['norms'])
            self.assertEqual(serial['selector_rng'],resumed['selector_rng'])
            torch.testing.assert_close(serial['rng']['torch'],resumed['rng']['torch'],rtol=0.,atol=0.)


if __name__ == '__main__':
    unittest.main()
