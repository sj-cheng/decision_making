"""Own continuous pursuers against unchanged E49 evader; reuse data explicitly."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TZ']='Asia/Shanghai'
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import json
import multiprocessing as mp
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import run_e49_hard_initials as hard

previous=hard.previous;ROOT=hard.ROOT
OUT=ROOT/'current/e50_own_pursuers_20261107'
E49=ROOT/'current/e49_hard_initials_20261107'


def prepare():
    OUT.mkdir(exist_ok=True);path=OUT/'experiment.json'
    if path.exists():return previous.read(path)
    source=previous.read(E49/'experiment.json');hard.verify(source)
    methods=[s for s in previous.specs() if s['arm']=='R2']
    cfg=dict(experiment_id='E50',purpose='Own continuous pursuers, separate trajectory plots; no dynamics/angle changes',
        entries=source['entries'],new_methods=methods,opponent='geometry_escape',
        reused_arms=['A3','A5','A2'],reused_source=str(E49),new_prior_episodes=72,
        historical_examples=[source['entries'][0],source['entries'][1]],
        display_main=['A3','R2','A5'],geometry_reference='A2 separately; AG omitted from own-method main figures',
        historical='d47d188 original1000 simulations, depth100, beta_policy.9/value0, full-state reference; not equal-information/time comparison',
        zero_angle_filtering=True,interpolation=False,smoothing=False,
        scope='Exploratory extension on already-viewed E49 maps, no new independent confirmation',
        source_hashes={str(ROOT/'code'/name):previous.sha(ROOT/'code'/name) for name in ['run_e50_own_pursuers.py','e50_git_policy_worker.py','run_e49_hard_initials.py','solvers/e49_geometric_evader.py','run_e48_dense_obstacles.py']},
        protected=source['protected'])
    previous.write(path,cfg);(OUT/'docs').mkdir(exist_ok=True)
    (OUT/'docs/PLAN.md').write_text('''# E50：原有连续追捕方法单独展示\n\n按用户要求，保留E49有限前瞻八方向逃跑基线和困难初态。主展示为L5搜索、旧Single PRIOR搜索和L5直控，几何搜索单列参考，AG不再混在“自己的学习方法”主图。追捕动作、运动约束、网络和训练均不更改，不过滤角度，不插值/平滑成假轨迹。\n\nE49已有L5搜索/直控及几何搜索数据明确引用，不重复计作新样本。补跑旧Single PRIOR三种子、12初态、16模拟/200ms两个条件共72局；在E49原32路校准容量内使用独立物理核，保留20个逻辑CPU。另按预选地图00的AB/BA两个初态运行d47d188历史方法1000模拟参考，用隔离旧源码/库进程接入同一对手；全状态根与不同预算单独说明，原权重为目前可用V3第45轮，不能证明是提交时默认。公共执行端的非法动作停车计数单独报告，不冒称历史原仿真逐位重放。\n\n图按追捕/逃跑拆开：主图只突出追捕轨迹，另给双方完整图与实际位移方向分布。原始八方向逃跑线不作为追捕策略角度的证据。保留所有成功/失败，预选地图00/AB/模型17示例，不按结果换图。这里是已看过输入的探索性扩展，不进行新独立确认或重新选择主方法。\n''')
    return cfg


def verify(cfg):
    for group in [cfg['source_hashes'],*cfg['protected'].values()]:
        for f,digest in group.items():
            if previous.sha(f)!=digest:raise RuntimeError('Frozen dependency changed: '+f)


def configure():
    hard.OUT=OUT;hard.configure()


def init(queue):
    os.sched_setaffinity(0,{queue.get()});configure()


def prior_tasks(cfg):
    for mode in ('fixed16','nominal200ms'):
        for entry in cfg['entries']:
            for spec in cfg['new_methods']:
                yield previous.episode_task(entry,spec,'E50_'+mode,mode='sim' if mode=='fixed16' else 'wall',
                    value=16 if mode=='fixed16' else 200,world=entry['world'],opponent='geometry_escape')


def legacy_episode(entry):
    from corner_estimator_scenarios import original_model
    from solvers.e49_geometric_evader import GeometricEvader
    from solvers.e45_team_adapter import public_observation
    from solvers.e45b_joint_nte import simultaneous_capture
    from run_frozen_d_closed_loop import legal_action
    target=OUT/'historical'/(entry['case_id']+'.json')
    if target.exists():return previous.read(target)
    case=previous.read(entry['path']);p=original_model(np.asarray(case['map']),case['speed'])
    state=np.array(case['worlds'][0]['initial'])[:,None];enemy=GeometricEvader();trace=[];first=None
    logs=OUT/'historical';logs.mkdir(exist_ok=True)
    with (logs/(entry['case_id']+'.worker.log')).open('w') as err:
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'code/e50_git_policy_worker.py'),entry['path']],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
        try:
            line=child.stdout.readline()
            if not line:raise RuntimeError('Historical worker failed; see worker log')
            identity=json.loads(line);assert identity['ready']
            for step in range(68):
                ap=state[p.active_idxs[2:],0]>.5;ae=state[p.active_idxs[:2],0]>.5
                obs=public_observation(int(state[8,0]),state[4:8,0].reshape(2,2),state[:4,0].reshape(2,2),ap,ae,np.array(case['map']))
                start=time.perf_counter();child.stdin.write(json.dumps(dict(state=state[:,0].tolist()))+'\n');child.stdin.flush()
                line=child.stdout.readline()
                if not line:raise RuntimeError('Historical worker exited mid-game')
                reply=json.loads(line);action=np.array(reply['action']);raw=action.copy();shields=0
                for i in range(2):
                    if not ap[i]:action[i]=0.
                    elif not legal_action(state[4+2*i:6+2*i,0],action[i],case['speed'],np.array(case['map'])):action[i]=0.;shields+=1
                latency=1000*(time.perf_counter()-start)
                other=np.array(enemy.search(p,state.copy()).best_action[:4]).reshape(2,2)
                joint=np.r_[other.ravel(),action.ravel()][:,None];nxt=p.step(state,joint,1.)
                assert p.is_valid(nxt);np.testing.assert_allclose(nxt[:8],state[:8]+joint,rtol=0,atol=1e-10)
                nap,nae,_=simultaneous_capture(nxt[4:8,0].reshape(2,2),nxt[:4,0].reshape(2,2),ap,ae,np.array(case['map']))
                np.testing.assert_array_equal(nap,nxt[p.active_idxs[2:],0]>.5);np.testing.assert_array_equal(nae,nxt[p.active_idxs[:2],0]>.5)
                if ae.sum()>nae.sum() and first is None:first=step+1
                trace.append(dict(state=state[:,0].tolist(),next_state=nxt[:,0].tolist(),observation=obs,
                    action=action.tolist(),raw_action=raw.tolist(),opponent_action=other.tolist(),pursuer_shields=shields,
                    latency_ms=latency,worker_act_ms=reply['act_ms'],stats=dict(simulations=1000,historical_privileged_root=True)))
                state=nxt
                if p.is_terminal(state):break
            captures=int(2-np.sum(state[p.active_idxs[:2],0]>.5));complete=captures==2
            summary=dict(captures=captures,double_capture=complete,decisions=len(trace),completion_cost=len(trace) if complete else 68,
                utility=float(complete)-.0025*len(trace),first_capture=first,latencies_ms=[r['latency_ms'] for r in trace],
                pursuer_shields=sum(r['pursuer_shields'] for r in trace),privileged_root=True,budget_simulations=1000)
            row=dict(valid=True,task=dict(arm='GIT',method='GIT',entry=entry,case=entry['path'],world=entry['world'],
                opponent='geometry_escape',train_seed=45,stage='E50_historical1000'),result=dict(summary=summary),
                identity=identity,trace=trace,finished=previous.stamp())
            previous.write(target,row);return row
        finally:
            if child.poll() is None:
                try:child.stdin.write('{"quit":true}\n');child.stdin.flush();child.wait(timeout=5)
                except Exception:child.terminate()


def summarize(cfg):
    verify(cfg);rows=[]
    for task in prior_tasks(cfg):
        path=OUT/'results'/task['stage']/(previous.digest(task)+'.json')
        if not path.exists():raise RuntimeError('Incomplete PRIOR queue')
        rows.append(dict(source='new',path=str(path),row=previous.read(path)))
    for f in (E49/'results').glob('E49_*geometry_escape/*.json'):
        if 'engineering' in str(f):continue
        r=previous.read(f)
        if r['task']['arm'] in ('A3','A5','A2'):rows.append(dict(source='reused_E49',path=str(f),row=r))
    cells=collections.defaultdict(lambda:collections.defaultdict(list));invalid=[]
    for item in rows:
        r=item['row'];t=r['task']
        if not r['valid']:invalid.append(r);continue
        mode='fixed16' if t.get('fixed') is not None else 'nominal200ms';cells[mode,t['arm']][t['group']].append(r['result']['summary'])
    table=[]
    for (mode,arm),groups in sorted(cells.items()):table.append(dict(mode=mode,arm=arm,maps=len(groups),
        **{k:float(np.mean([np.mean([r[k] for r in v]) for v in groups.values()])) for k in ('double_capture','utility','completion_cost')}))
    previous.write(OUT/'episode_index.json',dict(episodes=[dict(source=x['source'],path=x['path']) for x in rows],
            new=72,reused=168,note='Reused E49 trajectories are not new independent evidence'))
    previous.write(OUT/'final/results.json',dict(experiment='E50',table=table,invalid_count=len(invalid),invalid=invalid,
        new_prior_episodes=72,reused_episodes=168,historical_examples=2,scope=cfg['scope']))
    lines=['# E50：原有追捕方法对固定E49逃跑基线','','主方法为L5搜索、旧Single PRIOR搜索、L5直控；同池几何搜索单列参考。无动作离散化、角度筛选或轨迹平滑。','',
        '|预算|方法|双捕率|效用|完成成本|','|---|---|---:|---:|---:|']
    for r in table:lines.append(f'|{r["mode"]}|{r["arm"]}|{100*r["double_capture"]:.2f}%|{r["utility"]:.5f}|{r["completion_cost"]:.2f}|')
    lines+=['',f'新增PRIOR72局，复用E49已有168局，工程失效{len(invalid)}局。没有新增独立测试集或训练。',
        'd47d188另有2个历史示例：原1000模拟/深度100，读取全状态，使用现存V3第45轮，不能纳入同信息/200ms排名。执行合法性屏蔽单独记录。',
        '逃跑方仍为8方向有限前瞻参考。追捕方连续动作不意味着真实小车动力学；原每轴限速和1s位移接口保留。']
    (OUT/'final/REPORT.md').write_text('\n'.join(lines)+'\n')
    previous.write(OUT/'COMPLETE.json',dict(experiment='E50',finished=previous.stamp(),new_prior=72,reused_E49=168,historical_examples=2,invalid=len(invalid),sha256=previous.sha(OUT/'final/results.json')))


def run(cfg):
    verify(cfg);configure();os.sched_setaffinity(0,{10,11,12});ctx=mp.get_context('fork')
    for mode in ('fixed16','nominal200ms'):
        tasks=[t for t in prior_tasks(cfg) if mode in t['stage']];pending=[t for t in tasks if not (OUT/'results'/t['stage']/(previous.digest(t)+'.json')).exists()]
        queue=ctx.Queue()
        for cpu in range(16,32):queue.put(cpu)
        done=len(tasks)-len(pending)
        with ProcessPoolExecutor(16,mp_context=ctx,initializer=init,initargs=(queue,)) as pool:
            fs=[pool.submit(hard.execute,t) for t in pending]
            for f in as_completed(fs):
                row=f.result();done+=1;previous.write(OUT/'progress.json',dict(experiment='E50',mode=mode,done=done,total=36,phase='running',time=previous.stamp()))
                print(mode,done,row['valid'],row.get('error',''),flush=True)
    os.sched_setaffinity(0,{14})
    for entry in cfg['historical_examples']:legacy_episode(entry)
    summarize(cfg)
    previous.write(OUT/'progress.json',dict(experiment='E50',phase='complete',time=previous.stamp()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run'));a=p.parse_args();cfg=prepare()
    if a.mode=='run':run(cfg)
