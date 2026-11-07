"""E48: frozen L5 vs geometry on image-inspired dense rectangle layouts."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLBACKEND']='Agg'
os.environ['TZ']='Asia/Shanghai'
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch
import run_l5_followup as previous

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'current/e48_dense_obstacles_20261107'
BASE=14820261107
LABELS=dict(A2='Geometry ranking + search',A3='L5 learned ranking + search',
            A4='Geometry top, direct',A5='L5 learned top, direct',AG='Assigned geometry (AG)')
ORIGINAL_FEATURES=None
_BARRIER=None


def dense_features(boxes,positions,active_p,active_e,samples,weights,known,records,speed,remaining):
    """Same encoder arithmetic, expanded token axis; no truncation or new weights."""
    boxes=np.asarray(boxes,float).reshape(-1,2,2)
    if len(boxes)<=8:
        return ORIGINAL_FEATURES(boxes,positions,active_p,active_e,samples,weights,known,records,speed,remaining)
    if len(boxes)>16:raise ValueError('E48 registered capacity16 exceeded')
    result=ORIGINAL_FEATURES(boxes[:8],positions,active_p,active_e,samples,weights,known,records,speed,remaining)
    center=positions[np.asarray(active_p,bool)].mean(0) if np.any(active_p) else positions.mean(0)
    maps=np.zeros((16,4));mask=np.zeros(16)
    maps[:len(boxes),:2]=(boxes.mean(2)-center)/20.
    maps[:len(boxes),2:]=(boxes[:,:,1]-boxes[:,:,0])/20.
    mask[:len(boxes)]=1.
    result.update(maps=maps,map_mask=mask)
    return result


def configure():
    global ORIGINAL_FEATURES
    previous.configure(OUT)
    import learning.e45b_joint_network as network
    import solvers.e45b_joint_nte as joint
    if ORIGINAL_FEATURES is None:ORIGINAL_FEATURES=network.public_features
    joint.public_features=dense_features
    import run_e45b_role_free as base
    from solvers.e45_team_adapter import TeamController
    def factory(case,task):
        if task['arm']=='AG':return TeamController(case['map'],case['speed'],task['seed'],'AG',budget_ms=200.)
        return previous.follow.make_controller(case,task)
    base.make_controller=factory
    torch.set_num_threads(1)


def specs():
    return [s for s in previous.specs() if s['arm'] in ('A2','A3','A4','A5')]+[dict(name='AG',arm='AG',train_seed=17)]


def geometry(index):
    # Fifteen blocks placed from the supplied illustration, not exact digitization.
    centers=np.array([[-7.6,4.5],[-3.5,7.6],[0.,6.2],[2.5,4.4],[5.8,4.8],
        [-2.5,2.8],[-7.,.4],[1.4,1.2],[4.4,0.],[7.7,1.6],
        [-3.1,-1.6],[1.9,-1.8],[-.9,-4.6],[-3.1,-6.5],[2.1,-6.9]])
    sizes=np.array([[1.2,1.5],[1.5,1.7],[1.5,1.6],[1.5,1.2],[1.5,2.0],
        [1.3,1.4],[1.4,1.6],[1.6,1.5],[1.6,1.3],[1.7,1.4],
        [1.8,1.6],[1.4,1.8],[1.8,1.4],[1.7,1.2],[1.6,1.7]])
    rng=np.random.default_rng(np.random.SeedSequence([BASE,index,1]))
    if index:
        centers+=rng.uniform(-.18,.18,centers.shape)
        sizes*=rng.uniform(.9,1.1,sizes.shape)
    return np.stack((centers-sizes/2,centers+sizes/2),axis=2)


def build_case(index):
    from corner_estimator_scenarios import original_model
    from solvers.e45_team_adapter import public_observation
    from solvers.frozen_d_search_interface import sensor_visible,PublicMap,route
    from run_frozen_d_closed_loop import legal_action
    boxes=geometry(index);p=original_model(boxes,1.3)
    public=PublicMap(boxes,1.3);worlds=[];rng=np.random.default_rng(np.random.SeedSequence([BASE,index,2]))
    for world in ('AB','BA'):
        for attempt in range(10000):
            pp=rng.uniform(-8.7,8.7,(2,2));angle=rng.uniform(-np.pi,np.pi,2);distance=rng.uniform(2.6,4.8,2)
            ee=pp+distance[:,None]*np.c_[np.cos(angle),np.sin(angle)]
            at=np.concatenate((ee,pp));s=np.zeros((p.state_dim,1));s[:8,0]=at.ravel();s[p.active_idxs,0]=1
            if not p.is_valid(s) or p.get_capture_pairs(s):continue
            if np.linalg.norm(ee[0]-ee[1])<2 or np.linalg.norm(pp[0]-pp[1])<2:continue
            if not all(sensor_visible(pp[i],ee[i],boxes) for i in range(2)):continue
            if any(route(a,b,1.3,public.vertices,public.dist,boxes,np.zeros(2,np.int64))[0]>=1000000 for a in pp for b in ee):continue
            p._update_observation_memory(s,s,0.)
            obs=public_observation(0,pp,ee,np.ones(2,bool),np.ones(2,bool),boxes)
            worlds.append(dict(id=world,initial=s[:,0].tolist(),scripted_states=[s[:,0].tolist()],history=[obs]))
            break
        else:raise RuntimeError('Outcome-blind initial-state construction exhausted')
    # Unlike E47's two hidden posterior worlds, these are two distinct starts.
    return dict(id=f'e48_dense_{index:02d}',family='image_inspired_dense15',map=boxes.tolist(),
        speed=1.3,worlds=worlds,history=worlds[0]['history'],suite='dense_exploratory',split='diagnostic',
        initial_condition='Both identified targets initially visible to a teammate; later occlusion from normal dynamics',
        acceptance='Legal states, no initial capture, visible paired starts and path connectivity only; no controller outcomes')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);cfg_path=OUT/'experiment.json'
    if cfg_path.exists():return previous.read(cfg_path)
    source_cfg=previous.read(previous.DEFAULT/'experiment.json');previous.verify(source_cfg)
    entries=[]
    for index in range(12):
        case=build_case(index)
        for world in case['worlds']:
            local=copy.deepcopy(case);local['history']=world['history']
            path=OUT/'bank'/f'{case["id"]}_{world["id"]}.json';previous.write(path,local)
            entries.append(dict(id=case['id'],family=case['family'],path=str(path),sha256=previous.sha(path),
                world=world['id'],suite=case['suite'],split=case['split'],map_index=index))
    cfg=dict(experiment_id='E48',title='Image-inspired dense15 zero-shot L5/geometry comparison',
        created=previous.stamp(),maps=12,starts_per_map=2,entries=entries,methods=specs(),episodes=432,
        budgets=['fixed16','nominal200ms'],opponent='rootpuct3000_full_state',
        conditions=dict(arena=[-10,10],obstacles=15,obstacle_shape='axis-aligned rectangles, approximate illustration',
            teams='2v2',evader_axis_limit=1.,pursuer_axis_limit=1.3,capture_radius=1.,
            horizon=80,sensor_radius=6.,occlusion=True,retirement='both matched participants',
            evader_goal_line=False),
        feature_adapter='16 obstacle slots, masked sum pooling; all15 visible to every controller; no new weights',
        training_performed=False,models='E46 L5 base seeds17/29/43 unchanged',
        primaries_200ms=[['A3','A2'],['A3','A5']],inference='exploratory map bootstrap, not powered confirmation',
        watchdog_seconds=600,resource_rule='Reserve >=20 logical CPU; independent physical timing cores; CPU only',
        protected={k:source_cfg[k] for k in ('source_hashes','legacy_source_hashes','model_hashes')},
        source_sha256=previous.sha(__file__))
    previous.write(cfg_path,cfg)
    plan='''# E48：图示风格密集障碍对照\n\n按用户确认，构造15个轴对齐矩形，位置/尺寸参考图片；不精确复刻旋转障碍，不改变为图中的多机器人或终点逃脱任务。20×20场地、2v2、80步、捕获双方退休、公开团队观察、RootPUCT3000对手均保留。\n\n12张预先生成的模板扰动地图，每张2个几何合法且起始可见的初态。该可见起点与E47隐藏接管不同，不能跨批归因。A2/A3/A4/A5及AG共5方法9实例，搜索16次/名义200ms两个条件，共432局。AG含显式目标分配，单列强参考；不预设学习胜出，不按输赢换图或换初态。\n\n冻结E46 L5基础三种子，不重训。神经地图编码器逐障碍编码后masked求和，实验适配只把输入行容量从8扩到16，参数形状不变；对原≤8障碍输入验证输出不变，15障碍不截断。属于障碍数量与布局分布外的零样本泛化诊断，不能概括为充分训练学习方法的极限。\n\n12地图聚类，先平均初态和固定模型；200ms下预定比较A3−A2、A3−A5的原始效用，地图bootstrap10000次同时区间仅作探索。不把重复种子当独立地图。固定16为机制诊断。记录失败、超时、零模拟和全部尝试，不把工程失败改作普通输局。\n\n先完成接口与固定工作计时校准，再跑全局；固定次数可高并行，200ms限通过校准的独立物理核。至少保留20逻辑CPU，推理为NumPy不需要GPU。每局600秒看门狗，超时保留失效证据。\n\n预先选地图00、初态AB、模型17绘制5方法×2预算10局轨迹，不按结果选图。全部运行数据与PDF只放本目录，既有E46/E47不改。\n'''
    (OUT/'docs').mkdir(exist_ok=True);(OUT/'docs/PLAN.md').write_text(plan)
    return cfg


def verify(cfg):
    for group in cfg['protected'].values():
        for path,expected in group.items():
            if previous.sha(path)!=expected:raise RuntimeError('Frozen artifact changed: '+path)
    if previous.sha(__file__)!=cfg['source_sha256']:raise RuntimeError('E48 source changed; record revision before resuming')


def acceptance(cfg):
    configure()
    import learning.e45b_joint_network as net
    case=previous.read(cfg['entries'][0]['path']);s=np.asarray(case['worlds'][0]['initial'])
    args=(s[4:8].reshape(2,2),[True,True],[True,True],np.repeat(s[:4].reshape(2,1,2),64,axis=1),
          np.full((2,64),1/64),[None,None],case['history'],1.3,80)
    x=ORIGINAL_FEATURES(np.asarray(case['map'])[:8],*args)
    same=dense_features(np.asarray(case['map'])[:8],*args)
    for k in x:np.testing.assert_array_equal(x[k],same[k])
    padded={k:v.copy() for k,v in x.items()};padded['maps']=np.pad(x['maps'],((0,8),(0,0)));padded['map_mask']=np.pad(x['map_mask'],(0,8))
    paths=[spec['checkpoint'] for spec in specs() if spec['arm']=='A3']
    for path in paths:
        rank=previous.follow.old_controller.__globals__['RankRuntime'](path)
        actions=np.zeros((16,4));np.testing.assert_allclose(rank(x,actions),rank(padded,actions),rtol=1e-13,atol=1e-13)
        dense=dense_features(np.asarray(case['map']),*args);assert dense['map_mask'].sum()==15
        assert np.isfinite(rank(dense,actions)).all()
    packet=dict(map=case['map'],speed=1.3,records=case['history'],seed=17048)
    hashes=[];timings=[]
    for arm in ('A2','A3','A4','A5'):
        spec=next(s for s in specs() if s['arm']==arm);start=time.perf_counter()
        own,action,cost=previous.follow.prepare_public(packet,dict(spec,seed=17048),2)
        hashes.append(own.last_stats['root_pool_sha256']);timings.append(dict(arm=arm,wall_s=time.perf_counter()-start,**cost))
    assert len(set(hashes))==1
    previous.write(OUT/'acceptance.json',dict(passed=True,all15_obstacles_encoded=True,
        original8_input_exact=True,zero_padding_rank_invariant=True,same_root_pool=True,
        timings=timings,source_sha256=previous.sha(__file__)))
    print('ACCEPTANCE',timings,flush=True)


def timeout(*args):raise TimeoutError('Registered E48 per-episode600s watchdog')


def worker_init(cpus):
    os.sched_setaffinity(0,{cpus.get()});configure()


def tasks(cfg,mode):
    for entry in cfg['entries']:
        for spec in cfg['methods']:
            yield previous.episode_task(entry,spec,'E48_'+mode,mode='sim' if mode=='fixed16' else 'wall',
                    value=16 if mode=='fixed16' else 200,world=entry['world'],opponent='rootpuct')


def episode(task):
    key=previous.digest(task);target=OUT/'results'/task['stage']/(key+'.json')
    if target.exists():return previous.read(target)
    start=time.time();signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    try:
        result=previous.e46.run_episode(task,OUT)
        row=dict(valid=True,task=task,result=result,finished=previous.stamp(),wall_s=time.time()-start)
    except Exception as exc:
        row=dict(valid=False,task=task,error=repr(exc),traceback=traceback.format_exc(),finished=previous.stamp(),wall_s=time.time()-start)
    finally:signal.alarm(0)
    previous.write(target,row);return row


def calibration_one(job):
    configure();entry=job['entry'];case=previous.read(entry['path'])
    packet=dict(map=case['map'],speed=case['speed'],records=case['history'],seed=17048)
    spec=next(s for s in specs() if s['arm']=='A3')
    previous.follow.prepare_public(packet,dict(spec,seed=17048),2)
    if _BARRIER is not None:_BARRIER.wait(timeout=180)
    values=[]
    for repeat in range(4):
        # Match the online budget scope: constructor and historical warmup
        # are separately reported by the episode driver, outside act().
        values.append(previous.profile_call(packet,dict(spec,seed=17048+repeat),200,4)['act_ms'])
    return values


def calibration_init(cpus,barrier):
    global _BARRIER
    worker_init(cpus);_BARRIER=barrier


def calibrate(cfg):
    path=OUT/'timing_calibration.json'
    if path.exists():return previous.read(path)['workers']
    ctx=mp.get_context('fork');old=os.sched_getaffinity(0);os.sched_setaffinity(0,{16})
    try:ref=calibration_one(dict(entry=cfg['entries'][0]))
    finally:os.sched_setaffinity(0,old)
    attempts=[];chosen=1
    for workers in (4,8,16,32):
        q=ctx.Queue()
        for cpu in range(16,16+workers):q.put(cpu)
        barrier=ctx.Barrier(workers)
        with ProcessPoolExecutor(workers,mp_context=ctx,initializer=calibration_init,initargs=(q,barrier)) as pool:
            values=np.asarray(list(pool.map(calibration_one,[dict(entry=cfg['entries'][0])]*workers))).ravel()
        median=float(np.median(values)/np.median(ref));p95=float(np.percentile(values,95)/np.percentile(ref,95))
        passed=median<=1.2 and p95<=1.3
        attempts.append(dict(workers=workers,median_ratio=median,p95_ratio=p95,passed=passed))
        if passed:chosen=workers
        else:break
    previous.write(path,dict(workers=chosen,reference_ms=ref,attempts=attempts,passed=True,
        note='Fixed-work complete act() calibration; constructor/history excluded as in episode budget; reserve20 logical CPU, no SMT on timing cores'))
    return chosen


def run(cfg):
    verify(cfg);configure();os.sched_setaffinity(0,{10,11,12})
    if not (OUT/'acceptance.json').exists():acceptance(cfg)
    workers=calibrate(cfg);ctx=mp.get_context('fork')
    for mode in ('fixed16','nominal200ms'):
        pending=[t for t in tasks(cfg,mode) if not (OUT/'results'/t['stage']/(previous.digest(t)+'.json')).exists()]
        q=ctx.Queue();count=32 if mode=='fixed16' else workers
        for cpu in range(16,16+count):q.put(cpu)
        done=216-len(pending)
        with ProcessPoolExecutor(count,mp_context=ctx,initializer=worker_init,initargs=(q,)) as pool:
            futures=[pool.submit(episode,t) for t in pending]
            for f in as_completed(futures):
                row=f.result();done+=1
                previous.write(OUT/'progress.json',dict(experiment='E48',phase='running',mode=mode,
                    completed_in_mode=done,total_in_mode=216,workers=count,time=previous.stamp(),pid=os.getpid()))
                print(mode,done,'valid',row['valid'],row.get('error',''),flush=True)
    summarize(cfg)


def summarize(cfg):
    verify(cfg);cells=collections.defaultdict(lambda:collections.defaultdict(list));invalid=[];allrows=[]
    for mode in ('fixed16','nominal200ms'):
        for task in tasks(cfg,mode):
            f=OUT/'results'/task['stage']/(previous.digest(task)+'.json')
            if not f.exists():raise RuntimeError('Incomplete scientific queue')
            row=previous.read(f);allrows.append(row)
            if not row['valid']:invalid.append(row);continue
            cells[mode,task['arm']][task['group']].append(row['result']['summary'])
    table=[]
    for (mode,arm),maps in sorted(cells.items()):
        table.append(dict(mode=mode,arm=arm,maps=len(maps),episodes=sum(map(len,maps.values())),
            **{k:float(np.mean([np.mean([r[k] for r in v]) for v in maps.values()])) for k in ('double_capture','utility','completion_cost')},
            decisions=sum(r['decisions'] for v in maps.values() for r in v),
            overruns=sum(r['online_overruns'] for v in maps.values() for r in v)))
    comparisons=[]
    if not invalid:
        delta=[]
        for a,b in cfg['primaries_200ms']:
            groups=sorted(cells['nominal200ms',a]);values=np.array([np.mean([x['utility'] for x in cells['nominal200ms',a][g]])-np.mean([x['utility'] for x in cells['nominal200ms',b][g]]) for g in groups]);delta.append(values)
            comparisons.append(dict(method=a,reference=b,mean=float(values.mean())))
        rng=np.random.default_rng(BASE);ids=rng.integers(12,size=(10000,12))
        boot=np.stack([v[ids].mean(1) for v in delta],1);means=np.array([c['mean'] for c in comparisons])
        radius=float(np.quantile(np.max(np.abs(boot-means),axis=1),.95))
        for c in comparisons:c['simultaneous95_exploratory']=[c['mean']-radius,c['mean']+radius]
    report=dict(experiment='E48',table=table,comparisons=comparisons,invalid_count=len(invalid),
                invalid=invalid,finished=previous.stamp(),scope='Small exploratory zero-shot dense15 study, not broad learning superiority')
    previous.write(OUT/'final/results.json',report)
    lines=['# E48：密集15障碍下冻结L5与几何的探索性比较','','12张模板扰动图、2个起始可见初态、冻结模型3种子；无训练。','',
        '|预算|方法|有效地图|双捕率|效用|失败惩罚完成成本|','|---|---|---:|---:|---:|---:|']
    for v in table:lines.append(f'|{v["mode"]}|{LABELS[v["arm"]]}|{v["maps"]}|{v["double_capture"]*100:.2f}%|{v["utility"]:.5f}|{v["completion_cost"]:.2f}|')
    lines+=['','200ms预定比较（地图bootstrap同时区间，仅作探索）：','']
    for c in comparisons:lines.append(f'- {c["method"]}−{c["reference"]}: {c["mean"]:+.5f}, {c["simultaneous95_exploratory"]}')
    lines+=['',f'工程失效任务：{len(invalid)}。失效不按普通输局混算；如非零，现有有效子集表不能用于方法优越性结论。',
        '', '几何与学习保留全部15障碍；网络扩展masked求和的输入行数，不改参数。该障碍数量不在原训练范围，结果只代表当前冻结模型。',
        '与E47相比还改变了起点可见性，不可用跨批胜率下降归因于障碍数量。图示的旋转障碍、更多机器人及终点逃脱规则未复刻。']
    (OUT/'final/REPORT.md').write_text('\n'.join(lines)+'\n')
    previous.write(OUT/'COMPLETE.json',dict(finished=previous.stamp(),tasks=len(allrows),valid=len(allrows)-len(invalid),
         invalid=len(invalid),sha256=previous.sha(OUT/'final/results.json')))
    previous.write(OUT/'progress.json',dict(experiment='E48',phase='complete' if not invalid else 'completed_with_invalid_tasks',
        completed=len(allrows),invalid=len(invalid),time=previous.stamp(),pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','acceptance','run','summarize'));args=p.parse_args()
    cfg=prepare()
    if args.mode=='acceptance':acceptance(cfg)
    elif args.mode=='run':run(cfg)
    elif args.mode=='summarize':summarize(cfg)
