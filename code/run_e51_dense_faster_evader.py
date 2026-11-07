"""Small trajectory batch: original dense starts/RootPUCT, evader limit 1 -> 1.3."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TZ']='Asia/Shanghai'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import multiprocessing as mp
from pathlib import Path
import signal
import time
import traceback
import types
import numpy as np
import run_e48_dense_obstacles as dense

previous=dense.previous
ROOT=dense.ROOT
OUT=ROOT/'current/e51_dense_faster_evader_20261107'
EVADER_SPEED=1.3
ARMS=('A3','R2','A2','A4','A5','AG')


def environment(boxes,speed):
    from corner_estimator_scenarios import original_model
    p=original_model(boxes,speed)
    p.current_evader_speed_lim=EVADER_SPEED
    for robot in range(2):
        p.action_lims[p.action_idxs[robot]]=[[-EVADER_SPEED,EVADER_SPEED]]*2
        p.control_lims[p.action_idxs[robot]]=p.action_lims[p.action_idxs[robot]]/p.dt
    p.clear_closed_loop_cache()
    return p


def configure():
    dense.OUT=OUT
    dense.configure()


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'experiment.json'
    if path.exists():return previous.read(path)
    source=previous.read(ROOT/'current/e48_dense_obstacles_20261107/experiment.json')
    for group in source['protected'].values():
        for file,digest in group.items():assert previous.sha(file)==digest,file
    methods=[next(s for s in dense.specs()+previous.specs() if s['arm']==arm and s['train_seed']==17) for arm in ARMS]
    cfg=dict(experiment_id='E51',created=previous.stamp(),purpose='Original dense scene, only evader speed increased',
        entries=source['entries'][:4],methods=methods,episodes=24,budget_ms=200,
        conditions=dict(source='E48',obstacles=15,arena=[-10,10],teams='2v2',
            evader_axis_limit=EVADER_SPEED,old_evader_axis_limit=1.,pursuer_axis_limit=1.3,
            opponent='original RootPUCT, 3000 simulations, depth100, no neural oracles',
            initial_state='unchanged E48 ordinary visible starts; no synthetic withdrawal history',
            horizon=80,sensor_radius=6.,retirement='matched pursuer and evader',capture_radius=1.),
        historical_examples=2,protected=source['protected'],
        source_hashes={str(ROOT/'code'/n):previous.sha(ROOT/'code'/n) for n in
            ('run_e51_dense_faster_evader.py','e51_git_policy_worker.py','run_e48_dense_obstacles.py','run_e50_own_pursuers.py')},
        inherited_model_note='Frozen pursuer networks, belief and planning kernels unchanged; their original evader motion assumptions remain. No retraining.',
        scope='Four preselected starts, one fixed model seed; trajectory display, not a new confirmation study',
        cancelled_extensions='No E49 hard initial history or geometric evader; previous completed records retained as history')
    previous.write(path,cfg)
    (OUT/'docs').mkdir(exist_ok=True)
    (OUT/'docs/PLAN.md').write_text('''# E51：只改复杂场景与逃跑速度

按用户最新要求停止沿用E49/E50的困难接管初态和新增几何逃跑基线。复用E48最前两张15障碍地图、每图AB/BA普通初态；原RootPUCT逃跑算法3000模拟/深度100保留，仅将逃跑者每轴速度上限1.0提高到1.3，追捕仍为1.3。原80步、传感范围6、遮挡、捕获半径1、双方退休均不变。不新增初态筛选、逃跑策略、训练或运动平滑。

当前6种既有方法A3/R2/A2/A4/A5/AG，冻结种子17、原名义200ms接口，共24局，用于查看对局图。另保留地图00两初态的原Git历史方法参考2局，原1000模拟/全状态条件单列。所有初态与随机环境种子沿用E48，不按结果换图。16个独立物理CPU worker，保留至少20核；沿用此前通过的并发容量，不用GPU。

速度修改同步进入真实环境、原逃跑者C++搜索动作界限和执行合法性检查。追捕网络、信念、规划运动核保持冻结，因此仍保留原目标运动假设；这属于冻结控制器面对更快目标的展示，不声称已重新校准或最优适配。逐步核对实际位移、动作界限和合法性。既有E48/E49/E50文件不覆盖。当前只出少量对局及描述性统计，不扩充训练和确认实验。
''')
    return cfg


def verify(cfg):
    for group in [cfg['source_hashes'],*cfg['protected'].values()]:
        for file,digest in group.items():assert previous.sha(file)==digest,file
    for entry in cfg['entries']:assert previous.sha(entry['path'])==entry['sha256']


def tasks(cfg):
    for entry in cfg['entries']:
        for spec in cfg['methods']:
            yield previous.episode_task(entry,spec,'E51_nominal200ms',mode='wall',value=200,
                world=entry['world'],opponent='rootpuct',evader_speed=EVADER_SPEED)


def alarm(*_):raise TimeoutError('E51 per-game 600s watchdog')


def execute(task):
    target=OUT/'results'/task['stage']/(previous.digest(task)+'.json')
    if target.exists():return previous.read(target)
    import run_e45b_role_free as base
    from run_frozen_d_closed_loop import legal_action
    namespace=dict(base.episode.__globals__)
    opponents=[]
    class RecordedRoot(base.RootPUCT):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.raw=[];opponents.append(self)
        def search(self,p,state,turn=0):
            assert p.current_evader_speed_lim==EVADER_SPEED
            np.testing.assert_allclose(p.action_lims[:4],np.tile([-EVADER_SPEED,EVADER_SPEED],(4,1)))
            result=super().search(p,state,turn)
            self.raw.append(np.asarray(result.best_action).ravel()[:4].reshape(2,2).copy())
            return result
    # This private episode copy changes only the physical speed and its old
    # hardcoded evader execution bound. All pursuer code remains unchanged.
    namespace.update(original_model=environment,RootPUCT=RecordedRoot,
        legal_action=lambda at,action,speed,boxes:legal_action(at,action,EVADER_SPEED if speed==1. else speed,boxes))
    private=types.FunctionType(base.episode.__code__,namespace,'E51_episode',base.episode.__defaults__)
    stamp=time.perf_counter();signal.signal(signal.SIGALRM,alarm);signal.alarm(600)
    try:
        payload,_=private(dict(task,key=previous.digest(task)[:24]))
        raw=opponents[0].raw;assert len(raw)==len(payload['trace'])
        case=previous.read(task['case']);boxes=np.asarray(case['map']);above=0
        for frame,proposal in zip(payload['trace'],raw):
            frame['raw_opponent_action']=proposal.tolist()
            state=np.asarray(frame['state']);nxt=np.asarray(frame['next_state'])
            enemy=np.asarray(frame['opponent_action']);own=np.asarray(frame['action'])
            np.testing.assert_allclose(nxt[:8]-state[:8],np.r_[enemy.ravel(),own.ravel()],rtol=0,atol=1e-10)
            for j in range(2):assert legal_action(state[2*j:2*j+2],enemy[j],EVADER_SPEED,boxes)
            above+=int(np.any(np.abs(enemy)>1.+1e-7))
        payload['summary'].update(evader_speed=EVADER_SPEED,steps_with_evader_component_above_old_limit=above,
            opponent_shields=sum(f['opponent_shields'] for f in payload['trace']))
        path=OUT/'episodes'/task['stage']/(previous.digest(task)+'.pkl.gz')
        previous.save_pickle(path,payload)
        row=dict(valid=True,task=task,result=dict(summary=payload['summary'],episode=str(path),episode_sha256=previous.sha(path)),
            wall_seconds=time.perf_counter()-stamp,finished=previous.stamp())
    except Exception as exc:
        row=dict(valid=False,task=task,error=repr(exc),traceback=traceback.format_exc(),finished=previous.stamp())
    finally:signal.alarm(0)
    previous.write(target,row);return row


def worker_init(queue):
    os.sched_setaffinity(0,{queue.get()});configure()


def historical(entry):
    import subprocess,sys
    from solvers.root_audit_puct import RootPUCT
    from solvers.e45_team_adapter import public_observation
    from run_frozen_d_closed_loop import legal_action
    target=OUT/'historical'/(entry['id']+'_'+entry['world']+'.json')
    if target.exists():return previous.read(target)
    case=copy.deepcopy(previous.read(entry['path']));case['evader_speed']=EVADER_SPEED
    folder=OUT/'historical';folder.mkdir(exist_ok=True)
    case_path=folder/(entry['id']+'_'+entry['world']+'_case.json');previous.write(case_path,case)
    p=environment(np.asarray(case['map']),case['speed'])
    state=np.asarray(next(w['initial'] for w in case['worlds'] if w['id']==entry['world']))[:,None]
    seed=previous.episode_task(entry,cfg_git(),'E51_historical',world=entry['world'])['seed']
    enemy=RootPUCT([None]*4,None,3000,0.,0.);enemy.seeded_state(p,seed+17,seed+29)
    trace=[];first=None
    with target.with_suffix('.worker.log').open('w') as err:
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/e51_git_policy_worker.py'),str(case_path)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
        try:
            identity=__import__('json').loads(child.stdout.readline());assert identity['ready']
            for step in range(80):
                ap=state[p.active_idxs[2:],0]>.5;ae=state[p.active_idxs[:2],0]>.5
                obs=public_observation(int(state[8,0]),state[4:8,0].reshape(2,2),state[:4,0].reshape(2,2),ap,ae,np.asarray(case['map']))
                start=time.perf_counter();child.stdin.write(__import__('json').dumps(dict(state=state[:,0].tolist()))+'\n');child.stdin.flush()
                reply=__import__('json').loads(child.stdout.readline());action=np.asarray(reply['action']);raw=action.copy();shields=0
                for i in range(2):
                    if not ap[i]:action[i]=0.
                    elif not legal_action(state[4+2*i:6+2*i,0],action[i],case['speed'],np.asarray(case['map'])):action[i]=0.;shields+=1
                latency=1000*(time.perf_counter()-start)
                proposal=np.asarray(enemy.search(p,state.copy()).best_action).ravel()[:4].reshape(2,2)
                other=proposal.copy();eshields=0
                for j in range(2):
                    if not ae[j]:other[j]=0.
                    elif not legal_action(state[2*j:2*j+2,0],other[j],EVADER_SPEED,np.asarray(case['map'])):other[j]=0.;eshields+=1
                joint=np.r_[other.ravel(),action.ravel()][:,None];nxt=p.step(state,joint,1.)
                assert p.is_valid(nxt);np.testing.assert_allclose(nxt[:8],state[:8]+joint,rtol=0,atol=1e-10)
                if ae.sum()>np.sum(nxt[p.active_idxs[:2],0]>.5) and first is None:first=step+1
                trace.append(dict(state=state[:,0].tolist(),next_state=nxt[:,0].tolist(),observation=obs,
                    action=action.tolist(),raw_action=raw.tolist(),opponent_action=other.tolist(),raw_opponent_action=proposal.tolist(),
                    opponent_shields=eshields,pursuer_shields=shields,latency_ms=latency,stats=dict(simulations=1000)))
                state=nxt
                if p.is_terminal(state):break
            captures=int(2-np.sum(state[p.active_idxs[:2],0]>.5))
            summary=dict(captures=captures,double_capture=captures==2,decisions=len(trace),completion_cost=len(trace) if captures==2 else 80,
                utility=float(captures==2)-.0025*len(trace),first_capture=first,latencies_ms=[f['latency_ms'] for f in trace],
                online_overruns=sum(f['latency_ms']>200 for f in trace),
                pursuer_shields=sum(f['pursuer_shields'] for f in trace),opponent_shields=sum(f['opponent_shields'] for f in trace))
            task=dict(arm='GIT',case=str(case_path),entry=entry,world=entry['world'],train_seed=45,stage='E51_historical1000')
            path=folder/(entry['id']+'_'+entry['world']+'.pkl.gz');previous.save_pickle(path,dict(summary=summary,trace=trace))
            row=dict(valid=True,task=task,result=dict(summary=summary,episode=str(path),episode_sha256=previous.sha(path)),identity=identity)
            previous.write(target,row);return row
        finally:
            if child.poll() is None:
                try:child.stdin.write('{"quit":true}\n');child.stdin.flush();child.wait(timeout=5)
                except Exception:child.terminate()


def cfg_git():return dict(arm='GIT',name='GIT',train_seed=45)


def summarize(cfg):
    rows=[previous.read(OUT/'results'/t['stage']/(previous.digest(t)+'.json')) for t in tasks(cfg)]
    invalid=[r for r in rows if not r['valid']]
    table=[]
    for arm in ARMS:
        summaries=[r['result']['summary'] for r in rows if r['valid'] and r['task']['arm']==arm]
        table.append(dict(arm=arm,episodes=len(summaries),**{k:float(np.mean([s[k] for s in summaries])) for k in ('double_capture','completion_cost','decisions')}))
    previous.write(OUT/'final/results.json',dict(experiment='E51',table=table,invalid=invalid,scope=cfg['scope']))
    previous.write(OUT/'COMPLETE.json',dict(experiment='E51',finished=previous.stamp(),current_episodes=24,historical_examples=2,
        invalid=len(invalid),sha256=previous.sha(OUT/'final/results.json')))
    verify(cfg)


def run(cfg):
    verify(cfg);configure();os.sched_setaffinity(0,{10,11,12});ctx=mp.get_context('fork')
    queue=ctx.Queue()
    for cpu in range(16,32):queue.put(cpu)
    jobs=list(tasks(cfg));done=0
    with ProcessPoolExecutor(16,mp_context=ctx,initializer=worker_init,initargs=(queue,)) as pool:
        for future in as_completed([pool.submit(execute,t) for t in jobs]):
            row=future.result();done+=1
            previous.write(OUT/'progress.json',dict(phase='running',done=done,total=24,time=previous.stamp()))
            print(done,'/24',row['valid'],row.get('error',''),flush=True)
    os.sched_setaffinity(0,{14})
    for entry in cfg['entries'][:2]:historical(entry)
    summarize(cfg)
    previous.write(OUT/'progress.json',dict(phase='complete_with_invalid' if previous.read(OUT/'final/results.json')['invalid'] else 'complete',done=26,total=26,time=previous.stamp()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'));a=parser.parse_args()
    cfg=prepare()
    if a.mode=='run':run(cfg)
