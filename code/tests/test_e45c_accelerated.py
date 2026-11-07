import concurrent.futures as futures
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import run_e45c_budget as r
import run_e45c_accelerated as acceleration


class AccelerationTests(unittest.TestCase):
    def test_only_two_scheduling_constants_change(self):
        original=inspect.getsource(acceleration.ORIGINAL_LEARN)
        derived=acceleration.derive_learner(original,9)
        restored=derived.replace('gpu=Pool(9,True)','gpu=Pool(2,True)').replace('(True,gpu_queues,18,gpu)','(True,gpu_queues,4,gpu)')
        self.assertEqual(restored,original)
        self.assertIn('(False,cpu_queues,2*WORKERS,cpu)',derived)
        self.assertIn('gpu=Pool(9,True)',derived)

    def test_unexpected_frozen_scheduler_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'scheduler shape changed'):
            acceleration.derive_learner('def learn():pass',9)

    def test_derived_scheduler_completes_and_resumes_all_nine_jobs(self):
        history=[];pools=[]
        class ImmediatePool:
            def __init__(self,count,training=False):self.training=training;pools.append((count,training))
            def submit(self,task):
                history.append((self.training,task['_output'],task['round']))
                future=futures.Future();future.set_result(task|dict(key=task.get('key','fit')));return future
            def close(self,failed=False):pass
        def tasks(dose,round_):return [dict(key=str(i),_output=str(r.level(dose)),round=round_,stage='collection/train') for i in range(2)]
        def finish(dose,round_,records):
            self.assertEqual(len(records),9)
            folder=r.level(dose)/f'round_{round_:02d}'
            for seed in r.SEEDS:
                for kind in r.KINDS:
                    for ext in ('.npz','.pt'):
                        path=folder/'models'/f'{kind}_{seed}{ext}';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
            r.save(folder/'models_manifest.json',dict(files={}))
        with tempfile.TemporaryDirectory() as temp,patch.object(r,'OUT',Path(temp)),patch.object(r,'SMOKE',True),patch.object(r,'Pool',ImmediatePool),patch.object(r,'collection_tasks',tasks),patch.object(r.base,'cached',return_value=None),patch.object(r,'finish_models',finish),patch.object(r,'emit'),patch.object(r,'report'):
            namespace=dict(r.__dict__)
            exec(compile(acceleration.derive_learner(inspect.getsource(acceleration.ORIGINAL_LEARN),9),'<derived learner>','exec'),namespace)
            namespace['learn']()
            self.assertEqual(sum(x[0] for x in history),54)
            self.assertEqual(sum(not x[0] for x in history),12)
            self.assertIn((9,True),pools)
            self.assertIn((40,False),pools)
            history.clear();namespace['learn']();self.assertFalse(history)


if __name__=='__main__':unittest.main()
