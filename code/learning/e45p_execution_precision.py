"""P-only amendment: record the speed-clipped action Example8 actually uses."""
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import run_e45o_hasac_demonstration as o


def environment_action(action, speed):
    action=np.asarray(action,dtype=np.float64)
    executed=np.clip(action,-float(speed),float(speed))
    error=float(np.max(np.abs(executed-action)))
    if error>2e-7:
        raise RuntimeError('Unexpected non-roundoff overspeed; do not hide an execution bug')
    return executed,error


def factory(case,task):
    own=o.factory(case,task);original=own.act
    def act(obs,fixed_simulations=None):
        proposed=original(obs,fixed_simulations)
        executed,error=environment_action(proposed,case['speed'])
        own.last_stats['environment_speed_roundoff_correction']=error
        return executed
    own.act=act
    return own


def initialize_worker(folder,queue,parent):
    o.init_worker(folder,queue,parent)
    o.j.common.base.make_controller=factory


def pool_for(folder,cores):
    ctx=o.mp.get_context('spawn');queue=ctx.Queue()
    for core in cores:queue.put(core)
    return ProcessPoolExecutor(max_workers=len(cores),mp_context=ctx,
        initializer=initialize_worker,initargs=(str(folder),queue,os.getpid())),queue
