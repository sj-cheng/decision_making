import collections
import concurrent.futures as futures
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_e45c_budget as r


class BudgetTests(unittest.TestCase):
    def fake_entries(self,split,dose=None):
        n=r.counts()[dose] if split=='train' else 12
        return [dict(id=f'{split}_{i}',path=f'/case/{i}',family='long_wall',pattern='opposed_entries',split=split,family_index=i) for i in range(n)]

    def test_learning_dose_counts_and_common_random_streams(self):
        with patch.object(r,'SMOKE',False),patch.object(r,'entries',self.fake_entries),patch.object(r,'sha',return_value='1'*64):
            tasks={dose:r.collection_tasks(dose,5) for dose in r.DOSES}
        for dose,n in zip(r.DOSES,(24,120,480)):
            self.assertEqual(sum(t['split']=='train' for t in tasks[dose]),n*3)
            self.assertEqual(sum(t['split']=='validation' for t in tasks[dose]),12*4*3)
            self.assertTrue(all(t['budget_ms']==400 and t['model_round']==4 for t in tasks[dose]))
        def identity(t):return t['key'],t['seed'],t['opponent'],t['world']
        common={identity(t) for t in tasks['x1']}
        self.assertTrue(common<={identity(t) for t in tasks['x5']})
        self.assertTrue(common<={identity(t) for t in tasks['x20']})

    def test_value_switch_and_explicit_checkpoint_preserve_scientific_controller(self):
        case=dict(map=[],speed=1.1)
        for method,value,policy,direct in (('JOINT17',True,True,False),('PRIOR17',False,True,False),('VALUE_U17',True,False,False),('DIRECT17',True,True,True)):
            task=dict(method=method,budget_ms=10000,seed=43,checkpoint='/frozen/joint_17.npz')
            with patch.object(r,'Controller') as controller:
                r.make_controller(case,task)
                args,kw=controller.call_args
                self.assertEqual(args[3],Path(task['checkpoint']))
                self.assertEqual(kw,dict(use_policy=policy,use_value=value,direct=direct,budget_ms=10000))

    def test_budget_cache_paths_do_not_collide(self):
        old=r.base.OUT
        try:
            r.configure('/tmp/e45c_cache_unit')
            a=dict(round=0,stage='search/closed_loop/budget_200',key='same')
            b=a|dict(stage='search/closed_loop/budget_2000')
            self.assertNotEqual(r.base.paths(a)[0],r.base.paths(b)[0])
        finally:r.configure(old)

    def test_grouped_budget_comparison_keeps_map_as_unit(self):
        rows=[]
        for group in range(6):
            for dose,bonus in (('x1',0),('x5',.1),('x20',.2)):
                for seed in (17,29,43):
                    for world in ('AB','BA'):
                        rows.append(dict(group=str(group),opponent='rootpuct',_output=str(r.level(dose)),method=f'JOINT{seed}',utility=.01*group+bonus,budget_ms=200))
        pairs=[('rootpuct','JOINT','x5','x1',None,None),('rootpuct','JOINT','x20','x1',None,None)]
        result=r.compare(rows,pairs,'unit')
        self.assertEqual(result['groups'],6)
        np.testing.assert_allclose([x['utility_difference'] for x in result['comparisons']],[.1,.2],atol=1e-12)

    def test_overlapping_scheduler_completes_all_doses_and_resumes(self):
        history=[]
        class ImmediatePool:
            def __init__(self,count,training=False):self.training=training
            def submit(self,task):
                history.append((self.training,task['_output'],task['round']))
                f=futures.Future()
                f.set_result(task|dict(key=task.get('key','fit')))
                return f
            def close(self,failed=False):self_failed=failed
        def tasks(dose,round_):return [dict(key=str(i),_output=str(r.level(dose)),round=round_,stage='collection/train') for i in range(2)]
        def finish(dose,round_,records):
            self.assertEqual(len(records),9)
            folder=r.level(dose)/f'round_{round_:02d}'
            for seed in r.SEEDS:
                for kind in r.KINDS:
                    for ext in ('.npz','.pt'):
                        p=folder/'models'/f'{kind}_{seed}{ext}';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
            r.save(folder/'models_manifest.json',dict(files={}))
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'OUT',Path(tmp)),patch.object(r,'SMOKE',True),patch.object(r,'Pool',ImmediatePool),patch.object(r,'collection_tasks',tasks),patch.object(r.base,'cached',return_value=None),patch.object(r,'finish_models',finish),patch.object(r,'emit'),patch.object(r,'report'):
            r.learn()
            self.assertEqual(sum(t[0] for t in history),54)
            self.assertEqual(sum(not t[0] for t in history),12)
            self.assertEqual(len(r.read(r.OUT/'models_frozen.json')['files']),54)
            history.clear();r.learn();self.assertFalse(history)


if __name__=='__main__':unittest.main()
