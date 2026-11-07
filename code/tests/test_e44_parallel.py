import tempfile
import unittest
from pathlib import Path
import numpy as np
import run_e44_branch_nte as base
import run_e44_parallel as parallel


class ParallelExecutionTests(unittest.TestCase):
    def test_workers_use_distinct_physical_cores(self):
        cpus=parallel.choose_cpus(2)
        topology=parallel.core_topology()
        keys=[next(key for key,values in topology.items() if cpu in values) for cpu in cpus]
        self.assertEqual(len(set(keys)),2)

    def test_original_labels_seeds_and_all_nine_checkpoints_preserved(self):
        names=('OUT','SMOKE','BASE_SEED','SEEDS','KINDS','emit')
        previous={name:getattr(base,name) for name in names}
        old_official,old_parent=parallel.OFFICIAL,parallel.WORKER_PARENT
        base.torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as name:
                temp=Path(name);serial=temp/'serial';concurrent=temp/'parallel'
                _,packet=base.construct('wall_corner',4,1.1,944317,6.4,0,pause=0)
                rows=[]
                for split,n in (('train',2),('validation',1)):
                    for i in range(n):
                        rows.append(dict(id=f'{split}{i}',family='wall_corner',split=split))
                for out in (serial,concurrent):
                    base.save(out/'experiment.json',dict(train_states_per_map=1,validation_states_per_map=1,epochs=2))
                    base.save(out/'bank/manifest.json',dict(entries=rows))
                    base.save(out/'step1/selection.json',dict(selected='route_focused'))
                    for entry in rows:
                        base.save(out/'collection'/entry['split']/(entry['id']+'.json'),dict(packets=[packet]))
                base.OUT=serial;base.SMOKE=True;base.BASE_SEED=parallel.DEFAULT_SEED
                base.SEEDS=parallel.DEFAULT_SEEDS;base.KINDS=parallel.DEFAULT_KINDS;base.emit=lambda **_:None
                parallel.ORIGINAL_LABELS();parallel.ORIGINAL_TRAIN()
                parallel.OFFICIAL=concurrent;parallel.WORKER_PARENT=None
                for split in ('train','validation'):
                    for i,entry in enumerate(e for e in rows if e['split']==split):
                        parallel.label_job((split,i,entry))
                self.assertFalse((concurrent/'data/manifest.json').exists())
                for entry in rows:
                    relative=Path('data')/entry['split']/entry['id']/'000.npz'
                    with np.load(serial/relative) as a,np.load(concurrent/relative) as b:
                        self.assertEqual(a.files,b.files)
                        for key in a.files:np.testing.assert_array_equal(a[key],b[key])
                base.OUT=concurrent;base.BASE_SEED=parallel.DEFAULT_SEED
                manifest=parallel.ORIGINAL_LABELS()
                self.assertEqual(manifest['counts'],dict(train=2,validation=1))
                for seed in parallel.DEFAULT_SEEDS:
                    for kind in parallel.DEFAULT_KINDS:
                        parallel.training_job((kind,seed))
                        relative=Path('models')/f'{kind}_{seed}.npz'
                        with np.load(serial/relative) as a,np.load(concurrent/relative) as b:
                            self.assertEqual(a.files,b.files)
                            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
                        relative=Path('training')/f'{kind}_{seed}.json'
                        a,b=base.read(serial/relative),base.read(concurrent/relative)
                        self.assertEqual(a['log'],b['log'])
                        self.assertEqual(a['selected_epoch'],b['selected_epoch'])
        finally:
            for name,value in previous.items():setattr(base,name,value)
            parallel.OFFICIAL,parallel.WORKER_PARENT=old_official,old_parent


if __name__=='__main__':unittest.main()
