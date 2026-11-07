"""Re-run a complete M checkpoint evaluation through actor-only deployment.

All eighteen recorded TRAINING-situation episodes are retained. Tests exact
state/action equivalence, not a new benchmark or a policy-selection exercise.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
from concurrent.futures import ProcessPoolExecutor
import ctypes
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import signal
import time
import traceback
import torch
import numpy as np
import run_e45m_hasac_task as m
from learning.e45_hasac_deployment import load_actors,Controller

OUT=m.ROOT/'current/e45_hasac_deployment_audit_20261101/closed_loop'
ACTORS=None


def factory(case,task):
    return Controller(case['map'],case['speed'],task['seed'],ACTORS,
                      stochastic=task['execution_mode']=='stochastic',policy_seed=task['policy_seed'])


def init_worker(queue,parent,checkpoint,digest):
    global ACTORS
    ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    if os.getppid()!=parent:raise RuntimeError('Audit parent exited')
    os.sched_setaffinity(0,{queue.get()});torch.set_num_threads(1)
    m.configure(OUT);m.j.common.base.make_controller=factory;m.j.common.base.PARENT=parent
    ACTORS,_=load_actors(checkpoint,digest)


def job(spec):
    task=spec['task']
    try:
        original=json.loads(gzip.decompress(Path(spec['original']).read_bytes()))
        assert m.j.sha(spec['original'])==spec['original_sha256']
        assert all(original['summary'][key]==value for key,value in task.items())
        payload,_=m.j.common.base.episode(task)
        m.j.save(OUT/'episodes'/(task['key']+'.json'),payload)
        assert len(payload['trace'])==len(original['trace'])
        for new,old in zip(payload['trace'],original['trace']):
            for key in ('state','next_state','action','opponent_action','reward','new_captures','remaining_targets'):
                np.testing.assert_array_equal(new[key],old[key],err_msg=key)
            assert new['belief_signature']==old['belief_signature']
        for key in ('utility','double_capture','first_capture','captures','decisions'):
            assert payload['summary'][key]==original['summary'][key],key
        return dict(key=task['key'],mode=task['execution_mode'],world=task['world'],
                    steps=len(payload['trace']),captures=payload['summary']['captures'],
                    matched=True,original=spec['original'],original_sha256=spec['original_sha256'])
    except BaseException as exc:
        m.j.save(OUT/'failures'/(task['key']+'.json'),dict(task=task,error=repr(exc),traceback=traceback.format_exc()));raise


def source_live():
    f=m.OUT/'runs/HASAC_17/status.json'
    if not f.exists():return False
    s=m.j.read(f);p=Path('/proc/'+str(s.get('pid'))+'/cmdline')
    return p.exists() and b'run_e45m_hasac_task.py train' in p.read_bytes().replace(b'\0',b' ')


def main():
    OUT.mkdir(exist_ok=True,parents=True);folder=m.OUT/'runs/HASAC_17';chosen=None
    m.j.common.emit(OUT,'waiting_for_first_M_20k_checkpoint')
    while chosen is None:
        hp=folder/'evaluation.json'
        history=m.j.read(hp) if hp.exists() else []
        candidates=[h for h in history if h['steps']>=20000 and (folder/'checkpoints'/f'steps_{h["steps"]}.pt').exists()]
        if candidates:chosen=candidates[0];break
        if not source_live():
            raise RuntimeError('Source trainer no longer live before required checkpoint; inspect source status, do not restart it')
        time.sleep(10)
    path=folder/'checkpoints'/f'steps_{chosen["steps"]}.pt';digest=m.j.sha(path)
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['steps']==chosen['steps'];m.verify(m.OUT)
    rows=[m.j.read(p) for p in (folder/'evaluation/cache').glob('*.json')]
    rows=[r for r in rows if r['stage']==f'hasac_eval_steps{chosen["steps"]}']
    assert len(rows)==18
    cfg=m.j.read(m.OUT/'experiment.json');entries={e['id']:e for e in cfg['entries']};specs=[]
    for row in rows:
        template=m.task_for(entries[row['group']],row['world'],row['seed'],17,row['execution_mode'],
                            row['policy_seed'],f'eval_steps{chosen["steps"]}')
        # Preserve the exact original packet, including unused APF/cache
        # metadata. Reconstructing defaults under another output directory
        # can change identity keys even though HASAC never uses those gains.
        task={key:row[key] for key in template}
        specs.append(dict(task=task,original=row['episode'],original_sha256=row['episode_sha256']))
    cpu=m.j.choose_cpus(4);ctx=mp.get_context('spawn');q=ctx.Queue()
    for c in cpu:q.put(c)
    m.j.common.emit(OUT,'running',checkpoint=str(path),checkpoint_sha256=digest,episodes=len(specs))
    result=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=ctx,initializer=init_worker,
                             initargs=(q,os.getpid(),str(path),digest)) as pool:
        for row in pool.map(job,specs):
            result.append(row)
            if len(result)%6==0:print('Matched complete episodes',len(result),'/',len(specs),flush=True)
    q.close()
    report=dict(passed=True,scope='closed-loop equivalence on all18 original training-situation evaluation episodes, not an independent test',
                checkpoint=str(path),checkpoint_sha256=digest,steps=chosen['steps'],episodes=len(result),
                transitions=sum(r['steps'] for r in result),all_states_actions_opponents_rewards_exact=True,
                online_actor_only=True,no_reward_or_critic_online=True,timing_performance_claim=False,
                source_sha256=m.j.sha(Path(__file__)),runtime_sha256=m.j.sha(m.ROOT/'code/learning/e45_hasac_deployment.py'),
                results=result)
    m.j.save(OUT/'equivalence.json',report);m.j.common.emit(OUT,'complete',episodes=len(result))


if __name__=='__main__':main()
