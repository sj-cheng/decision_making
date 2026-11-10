"""Parallel execution adapter; no changes to scientific sources or job identities."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['TZ']='Asia/Shanghai'
import argparse
import json
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=84)
    parser.add_argument('--gpu-jobs',type=int,default=24)
    parser.add_argument('--train-job',type=Path)
    args=parser.parse_args()
    from pursuit_parallel.runtime import verify_receipt,run_training
    root=args.output.resolve();verify_receipt(root)
    if args.train_job:
        run_training(json.loads(args.train_job.read_text()),json.loads(os.environ['PURSUIT_PARALLEL_CONFIG']))
        return
    import run_large_pursuit as driver
    import run_large_pursuit_full as full
    original=full.gpu_batch
    def parallel_batch(out,jobs,layout,max_jobs,stage):
        pending=[job for job in jobs if not (Path(job['folder'])/'result.json').exists()]
        count=min(len(pending),max_jobs,len(layout['training']))
        if not count:return original(out,jobs,layout,max_jobs,stage)
        groups={layout['training'][i]:[layout['training'][i]]+layout['training'][count+i::count]
                for i in range(count)}
        real_popen=driver.subprocess.Popen
        def launch(command,*positional,**kwargs):
            if (isinstance(command,list) and len(command)>=4 and
                    Path(command[1]).resolve()==Path(driver.__file__).resolve() and '--train-job' in command):
                path=Path(command[command.index('--train-job')+1]);job=json.loads(path.read_text())
                cpus=groups[job['cpus'][0]]
                config=dict(root=str(root),cpus=cpus,teacher_workers=min(8,len(cpus)),
                    validation_workers=min(len(cpus),max(2,16//count),8))
                command=[command[0],str(Path(__file__).resolve()),'--output',str(root),'--train-job',str(path)]
                kwargs['env']=dict(kwargs.get('env',os.environ),PURSUIT_PARALLEL_CONFIG=json.dumps(config))
                driver.event(out,'parallel_training_configuration',folder=str(path.parent),configuration=config)
            return real_popen(command,*positional,**kwargs)
        driver.subprocess.Popen=launch
        try:return original(out,jobs,layout,max_jobs,stage)
        finally:driver.subprocess.Popen=real_popen
    full.gpu_batch=parallel_batch
    sys.argv=[str(Path(full.__file__)),'--output',str(root),'--workers',str(args.workers),'--gpu-jobs',str(args.gpu_jobs)]
    full.main()


if __name__=='__main__':main()
