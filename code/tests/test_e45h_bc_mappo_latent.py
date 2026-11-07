import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from learning.e45g_ppo import Actor
from learning.e45h_bc_latent import bc_loss, executed_targets
from learning.e45b_joint_network import KEYS
import run_e45h_bc_mappo_latent as run
from run_e45_team_pilot import make_case
from learning.e45g_ppo import Controller


class BCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        case=make_case('long_wall',0);own=Controller(case['map'],case['speed'],17,Actor())
        for obs in case['history']:own.observe(obs)
        features=own.features(case['history'][-1])
        cls.batch={key:torch.as_tensor(np.repeat(value[None],16,axis=0),dtype=torch.float32) for key,value in features.items()}

    def test_bc_positive_control_learns_nonzero_executed_action(self):
        torch.manual_seed(17);actor=Actor();batch={k:v.clone() for k,v in self.batch.items()}
        batch['label_actions']=torch.tensor([[[.7,-.4,-.6,.3]]]).repeat(16,1,1)
        optimizer=torch.optim.Adam(actor.parameters(),lr=1e-3)
        initial=float(bc_loss(actor,batch).detach())
        for _ in range(100):
            loss=bc_loss(actor,batch);optimizer.zero_grad();loss.backward();optimizer.step()
        self.assertLess(float(bc_loss(actor,batch).detach()),initial*.02)

    def test_executed_targets_align_public_clock_and_reject_mismatch(self):
        arrays={'public':np.zeros((4,8))};arrays['public'][:,2]=np.array([80,79,78,77])/80
        payload={'trace':[dict(observation=dict(remaining=80-i),action=[[.2,0.],[0.,-.3]]) for i in range(4)]}
        result=executed_targets(arrays,payload,1.1)
        self.assertEqual(result.shape,(4,1,4))
        arrays['public'][0,2]=.5
        with self.assertRaises(AssertionError):executed_targets(arrays,payload,1.1)

    def test_bc_loss_ignores_inactive_coordinate_labels(self):
        batch={k:v.clone() for k,v in self.batch.items()};batch['agents'][:,:,4]=torch.tensor([1.,0.])
        batch['label_actions']=torch.zeros((16,1,4));actor=Actor()
        first=bc_loss(actor,batch);batch['label_actions'][:,:,2:]=100
        torch.testing.assert_close(first,bc_loss(actor,batch))

    def test_new_statistics_have_bc_control_and_18_grouped_contrasts(self):
        rows=[]
        for group in range(3):
            for world in ('AB','BA'):
                for opponent in run.base.OPPONENTS:
                    for arm in run.ARMS:
                        for seed in (run.SEEDS if arm in run.LEARNED else (0,)):
                            rows.append(dict(key=str(len(rows)),group=str(group),world=world,family=str(group),suite='benchmark',opponent=opponent,
                                method=arm+str(seed) if arm in run.LEARNED else arm,train_seed=seed,
                                double_capture=True,utility=.8,completion_cost=20,latencies_ms=[5.]))
        stats=run.summarize(rows);self.assertEqual(stats['status'],'complete')
        self.assertEqual(stats['primary']['benchmark']['groups'],3)
        self.assertEqual(len(stats['primary']['benchmark']['comparisons']),18)
        self.assertIn('BC_MAPPO - BC_ONLY',[r['contrast'] for r in stats['primary']['benchmark']['comparisons']])


if __name__=='__main__':unittest.main()
