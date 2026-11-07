import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_e45e_value_off as e


class ValueOffTests(unittest.TestCase):
    def test_all_three_factories_use_same_joint_tree_and_disable_learned_value(self):
        case=dict(map=[],speed=1.1)
        for arm in e.ARMS:
            with patch.object(e,'Controller') as cls:
                e.factory(case,dict(method=arm+'17',seed=42,checkpoint='frozen.npz',budget_ms=200.))
                self.assertEqual(cls.call_args.args,([],1.1,42,'frozen.npz'))
                self.assertEqual(cls.call_args.kwargs,dict(use_policy=True,use_value=False,direct=False,budget_ms=200.))

    def test_request_preserves_c_world_seed_and_model_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            case=Path(folder)/'case.json';case.write_text('{}')
            entry=dict(id='history',path=str(case),family='long_wall',pattern='shared_entry',split='test')
            config=dict(checkpoints={a:{'17':a+'.npz'} for a in e.ARMS})
            for arm,old_arm in zip(e.ARMS,('SINGLE','FACT','PRIOR')):
                with patch.object(e,'read',return_value=config):
                    task=e.request(entry,'BA','rootpuct',arm,17,'evaluation')
                original=e.old.request(entry,'BA','rootpuct',old_arm+'17','evaluation',e.OUT,
                                       model_round=12,checkpoint=arm+'.npz')
                for field in ('seed','case','checkpoint','world','opponent','budget_ms','model_round'):
                    self.assertEqual(task[field],original[field])
                self.assertEqual(e.family(task),arm)

    def test_four_grouped_comparisons_and_complete_cell_requirement(self):
        rows=[]
        for g in range(5):
            for world in ('AB','BA'):
                for opponent in e.base.OPPONENTS:
                    for arm,utility in zip(e.ARMS,(.6,.8,.9)):
                        for seed in e.SEEDS:
                            rows.append(dict(group=str(g),world=world,opponent=opponent,method=arm+str(seed),
                                utility=utility,double_capture=True,completion_cost=20,latencies_ms=[195.]))
        with patch.object(e,'SMOKE',False):
            result=e.summarize(rows)
            self.assertTrue(result['complete_cells'])
            self.assertEqual(len(result['primary']),4)
            for r in result['primary']:
                self.assertAlmostEqual(r['utility_difference'],.3 if 'SINGLE' in r['contrast'] else .1)
                np.testing.assert_allclose(r['simultaneous95'],[r['utility_difference']]*2,atol=1e-14)
            self.assertIsNone(e.summarize(rows[:-1])['primary'])
            self.assertIsNone(e.summarize(rows+[rows[0]])['primary'])
            rows[0]['valid_episode']=False
            self.assertIsNone(e.summarize(rows)['primary'])


if __name__=='__main__':unittest.main()
