"""End-to-end reproducibility across interruption and worker counts."""
import os
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from learning.local_rl import PolicyBundle
from train_search_rl import rollout_task,run_tasks,wire_weights,episode_seed


class ContinuationTests(unittest.TestCase):
    def test_worker_wire_format_and_serial_rng_isolation(self):
        torch.set_num_threads(1);torch.manual_seed(31)
        bundle=PolicyBundle();state=wire_weights(bundle.state_dict())
        task=dict(seed=episode_seed(9,2,0,0),role=0,progress=0.,kind='ppo',bundle=state,opponent=state,budget=1,depth=1)
        before=torch.get_rng_state().clone();a=run_tasks(None,[task])[0]
        torch.testing.assert_close(before,torch.get_rng_state())
        b=run_tasks(None,[task])[0]
        self.assertEqual(a['stats'],b['stats'])
        for x,y in zip(a['rows'],b['rows']):
            np.testing.assert_array_equal(x['raw'],y['raw'])

    def test_exact_resume_with_different_worker_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);torch.manual_seed(42);PolicyBundle().save(root/'initial.pt')
            common=[sys.executable,str(CODE/'train_search_rl.py'),'--stage','ppo',
                    '--input',str(root/'initial.pt'),'--episodes','2','--episodes-per-update','1',
                    '--ppo-epochs','1','--validation-episodes','1','--eval-every','2',
                    '--depth','1','--opponent-budget','1','--budget','1']
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/mpl')
            commands=[common+['--output',str(root/'full'),'--workers','1'],
                      common+['--output',str(root/'split'),'--workers','1','--max-updates','2'],
                      common+['--output',str(root/'fork'),'--workers','2','--episodes-per-update','2',
                              '--fork-from',str(root/'split/ppo_latest.pt')],
                      common+['--output',str(root/'split'),'--workers','2']]
            for index,command in enumerate(commands):
                if index==2:source_before=hashlib.sha256((root/'split/ppo_latest.pt').read_bytes()).hexdigest()
                result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=120)
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
                if index==2:
                    self.assertEqual(source_before,hashlib.sha256((root/'split/ppo_latest.pt').read_bytes()).hexdigest())
                    manifest=json.loads((root/'fork/fork.json').read_text())
                    self.assertEqual(manifest['source_counters'],[1,1])
                    fork=torch.load(root/'fork/ppo_latest.pt',weights_only=False)
                    self.assertEqual(fork['counters'],[2,2])
                    self.assertEqual(fork['config']['episodes_per_update'],2)
                    self.assertEqual(fork['fork']['source_sha256'],source_before)
            full=torch.load(root/'full/ppo_latest.pt',weights_only=False)
            split=torch.load(root/'split/ppo_latest.pt',weights_only=False)
            self.assertEqual(full['counters'],[2,2]);self.assertEqual(split['counters'],[2,2])
            for kind in ('actors','critics'):
                for x,y in zip(full['bundle'][kind],split['bundle'][kind]):
                    for key in x:torch.testing.assert_close(x[key],y[key],rtol=0.,atol=0.)
            self.assertEqual(full['bundle']['norms'],split['bundle']['norms'])
            torch.testing.assert_close(full['rng']['torch'],split['rng']['torch'],rtol=0.,atol=0.)
            self.assertEqual(full['selector_rng'],split['selector_rng'])


if __name__=='__main__':unittest.main()
