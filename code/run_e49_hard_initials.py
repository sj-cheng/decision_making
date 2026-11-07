"""E49: hard hidden starts, frozen pursuers, paired fixed/geometry evaders."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TZ']='Asia/Shanghai'
import argparse
import collections
import concurrent.futures as futures
import copy
import multiprocessing as mp
from pathlib import Path
import signal
import time
import traceback
import types
import numpy as np
import run_e48_dense_obstacles as dense
from e49_hard_scenarios import construct,audit
from solvers.e49_geometric_evader import GeometricEvader,acceptance as evader_acceptance

previous=dense.previous
ROOT=dense.ROOT;OUT=ROOT/'current/e49_hard_initials_20261107'
BASE=14920261107
OPPONENTS=('rootpuct','geometry_escape')


def configure():
    dense.OUT=OUT;dense.configure()


def prepare():
    OUT.mkdir(exist_ok=True);path=OUT/'experiment.json'
    if path.exists():return previous.read(path)
    protected=previous.read(ROOT/'current/e48_dense_obstacles_20261107/experiment.json')
    entries=[]
    for index in range(6):
        for world in ('AB','BA'):
            case=construct(index,world);audit(case)
            filename=OUT/'bank'/(case['id']+'.json');previous.write(filename,case)
            entries.append(dict(id=f'e49_dense_map_{index:02d}',case_id=case['id'],family=case['family'],
                path=str(filename),sha256=previous.sha(filename),world=world,map_index=index,
                suite=case['suite'],split=case['split'],construction=case['construction']))
    source=[Path(__file__),ROOT/'code/e49_hard_scenarios.py',ROOT/'code/solvers/e49_geometric_evader.py',ROOT/'code/run_e48_dense_obstacles.py']
    cfg=dict(experiment_id='E49',title='Hard hidden handoffs and explicit geometry escape reference',
        created=previous.stamp(),maps=6,entries=entries,methods=dense.specs(),opponents=list(OPPONENTS),
        budgets=['fixed16','nominal200ms'],episodes=432,source_hashes={str(p):previous.sha(p) for p in source},
        protected=protected['protected'],model_training=False,
        conditions=dict(geometry='same first6 E48 layouts,15 axis-aligned obstacles; no outcome-based map choice',
            information='public team observations; full-state evaders as in Root reference',
            handoff_step=12,continuous_hidden_steps=4,min_euclidean_distance=8.,min_path_steps=7,
            history='visible anchors then legal scripted observer retreat; stationary targets before adaptive continuation',
            speed_ratio=1.3,horizon_total=80,remaining_steps=68),
        evader=dict(name='geometry_escape',depth=4,beam_width=8,actions='17 legal full/half compass plus stop',
            future_pursuer_model='map-aware direct chase of this target by every active pursuer',
            score='avoid predicted capture first; separation, geodesic distance, occlusion and modest boundary clearance',
            full_state=True,trained=False,optimality_claim=False,computation='ideal logical-time; not charged to pursuer budget'),
        comparisons='A3-A2 and A3-A5 at200ms separately per opponent;4 simultaneous exploratory map-bootstrap intervals',
        watchdog_seconds=600,resources='>=20 logical CPU reserved; fixed work32 physical; timing calibrated')
    previous.write(path,cfg)
    (OUT/'docs').mkdir(exist_ok=True)
    (OUT/'docs/PLAN.md').write_text('''# E49：困难接管初态与逃跑策略对照\n\n用户要求替换E48容易初态并指出逃跑方行为弱。本轮沿用E48最前6张地图（不按输赢选择）、15矩形、2v2、速度1.3/1.0、80步与双方退休规则。每图2个初态：先有合法可见锚点，追捕者按预定路线退出，目标在历史阶段静止；第12步接管时，两目标连续至少4步不可见、最近追捕距离至少8米、最短绕行至少7步，且历史曾有传感范围内的障碍遮挡。全部按几何、观测与动态合法性验收，不运行追捕策略筛选。\n\n保留RootPUCT3000，并加几何逃跑参考：完整真实双方位置、深度4/宽度8的有限前瞻，17个全/半步罗盘及停车动作；预测每名活跃追捕者按地图绕行追本目标，优先避免预计捕获，再评价距离、遮挡和边界余量。它是明确可核查的对手基线，不是最优或训练充分的逃逸专家；不读取本次追捕者将执行的动作。两类对手均采用理想逻辑时序，对手求解耗时另记，不向追捕预算收费。\n\n追捕A2/A3/A4/A5和AG，共9模型实例；6图×2初态×9实例×2对手×2预算=432局。冻结L5三个种子，不训练、不调整候选或信念。固定16模拟诊断与名义200ms分开；AG保留人工匹配参考身份。200ms下每个对手分别比较A3−A2/A3−A5，六图bootstrap10000次四项同时探索区间；不冒充充分样本确认。\n\n先检查公开历史逐步重放、同池与15障碍完整编码、逃跑者在可避免即捕反例下的合法避逃，再进行短前缀工程检查和计时校准。工程检查不按双捕率选策略。工程失效与600秒看门狗单独记账，不当普通失败局。\n\n预先选地图00、AB、模型17绘制5方法×2对手×2预算共20局；图中标注隐藏接管和历史。历史占用原总时域12步，剩余68步，不能与E48原始效用跨批直接相减当初态因果。需要对手作用时仅在本轮同初态内比较。所有新文档放本目录，既有E48不改。\n''')
    return cfg


def verify(cfg):
    for section in [cfg['source_hashes'],*cfg['protected'].values()]:
        for path,digest in section.items():
            if previous.sha(path)!=digest:raise RuntimeError('Frozen E49 dependency changed: '+path)
    for e in cfg['entries']:
        if previous.sha(e['path'])!=e['sha256']:raise RuntimeError('Scenario changed')


def make_tasks(cfg,mode,opponent,engineering=False):
    for entry in cfg['entries'][:1] if engineering else cfg['entries']:
        for spec in cfg['methods']:
            if engineering and spec['train_seed']!=17:continue
            task=previous.episode_task(entry,spec,'E49_'+mode+'_'+opponent,
                mode='sim' if mode=='fixed16' else 'wall',value=16 if mode=='fixed16' else 200,
                world=entry['world'],opponent=opponent)
            if engineering:task.update(stage='E49_engineering_'+opponent,limit=3)
            yield task


def alarm(*args):raise TimeoutError('Registered E49 per-episode600s watchdog')


def execute(task):
    target=OUT/'results'/task['stage']/(previous.digest(task)+'.json')
    if target.exists():return previous.read(target)
    import run_e45b_role_free as base
    configure();namespace=dict(base.episode.__globals__);opponents=[]
    def factory(*args,**kwargs):
        obj=GeometricEvader(*args,**kwargs) if task['opponent']=='geometry_escape' else base.RootPUCT(*args,**kwargs)
        opponents.append(obj);return obj
    namespace['RootPUCT']=factory
    private_episode=types.FunctionType(base.episode.__code__,namespace,'E49_episode',base.episode.__defaults__)
    start=time.perf_counter();signal.signal(signal.SIGALRM,alarm);signal.alarm(600)
    try:
        payload,_=private_episode(dict(task,key=previous.digest(task)[:24]))
        if task['opponent']=='geometry_escape':
            records=opponents[0].records;assert len(records)==len(payload['trace'])
            for frame,record in zip(payload['trace'],records):
                assert frame['opponent_shields']==0
                frame['opponent_stats']=record
            payload['summary']['opponent_ms']=[r['wall_ms'] for r in records]
        path=OUT/'episodes'/task['stage']/(previous.digest(task)+'.pkl.gz');previous.save_pickle(path,payload)
        result=dict(summary=payload['summary'],episode=str(path),episode_sha256=previous.sha(path))
        row=dict(valid=True,task=task,result=result,wall_seconds=time.perf_counter()-start,finished=previous.stamp())
    except Exception as exc:
        row=dict(valid=False,task=task,error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-start,finished=previous.stamp())
    finally:signal.alarm(0)
    previous.write(target,row);return row


def worker_init(queue):
    import ctypes
    ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    os.sched_setaffinity(0,{queue.get()});configure()


def acceptance(cfg):
    configure();previous.write(OUT/'opponent_acceptance.json',dict(passed=True,checks=evader_acceptance()))
    for e in cfg['entries']:audit(previous.read(e['path']))
    dense.acceptance(cfg)
    checks=[]
    for opponent in OPPONENTS:
        for task in make_tasks(cfg,'fixed16',opponent,engineering=True):
            row=execute(task);checks.append(dict(method=task['method'],opponent=opponent,valid=row['valid'],error=row.get('error')))
    previous.write(OUT/'engineering_acceptance.json',dict(checks=checks,passed=all(r['valid'] for r in checks)))
    if not all(r['valid'] for r in checks):raise RuntimeError('Engineering prefixes invalid; full batch not started')


def run(cfg):
    verify(cfg);configure();os.sched_setaffinity(0,{10,11,12})
    if not (OUT/'engineering_acceptance.json').exists():acceptance(cfg)
    if not previous.read(OUT/'engineering_acceptance.json')['passed']:raise RuntimeError('Engineering gate failed')
    # Reuse the checked E48 timer with E49 histories; input/state and CPU
    # profiles are measured anew, not copied from the prior easy-start run.
    workers=dense.calibrate(cfg);ctx=mp.get_context('fork')
    for mode in ('fixed16','nominal200ms'):
        for opponent in OPPONENTS:
            jobs=list(make_tasks(cfg,mode,opponent));pending=[t for t in jobs if not (OUT/'results'/t['stage']/(previous.digest(t)+'.json')).exists()]
            count=32 if mode=='fixed16' else workers;queue=ctx.Queue()
            for cpu in range(16,16+count):queue.put(cpu)
            done=len(jobs)-len(pending)
            with futures.ProcessPoolExecutor(count,mp_context=ctx,initializer=worker_init,initargs=(queue,)) as pool:
                active=[pool.submit(execute,t) for t in pending]
                for f in futures.as_completed(active):
                    row=f.result();done+=1;previous.write(OUT/'progress.json',dict(experiment='E49',phase='running',mode=mode,
                        opponent=opponent,completed_in_stage=done,total_in_stage=len(jobs),workers=count,time=previous.stamp(),pid=os.getpid()))
                    print(mode,opponent,done,'valid',row['valid'],row.get('error',''),flush=True)
    summarize(cfg)


def summarize(cfg):
    verify(cfg);cells=collections.defaultdict(lambda:collections.defaultdict(list));invalid=[]
    for mode in ('fixed16','nominal200ms'):
        for opponent in OPPONENTS:
            for task in make_tasks(cfg,mode,opponent):
                file=OUT/'results'/task['stage']/(previous.digest(task)+'.json')
                if not file.exists():raise RuntimeError('Incomplete E49 queue')
                row=previous.read(file)
                if not row['valid']:invalid.append(row);continue
                cells[mode,opponent,task['arm']][task['group']].append(row['result']['summary'])
    table=[]
    for (mode,opponent,arm),groups in sorted(cells.items()):
        table.append(dict(mode=mode,opponent=opponent,arm=arm,maps=len(groups),
            **{k:float(np.mean([np.mean([r[k] for r in v]) for v in groups.values()])) for k in ('double_capture','utility','completion_cost')},
            episodes=sum(map(len,groups.values()))))
    comparisons=[];delta=[]
    if not invalid:
        for opponent in OPPONENTS:
            for a,b in [('A3','A2'),('A3','A5')]:
                keys=sorted(cells['nominal200ms',opponent,a]);values=np.array([np.mean([r['utility'] for r in cells['nominal200ms',opponent,a][g]])-np.mean([r['utility'] for r in cells['nominal200ms',opponent,b][g]]) for g in keys])
                delta.append(values);comparisons.append(dict(opponent=opponent,method=a,reference=b,mean=float(values.mean())))
        rng=np.random.default_rng(BASE);ids=rng.integers(6,size=(10000,6));boot=np.stack([v[ids].mean(1) for v in delta],1)
        mean=np.array([r['mean'] for r in comparisons]);radius=float(np.quantile(np.max(np.abs(boot-mean),axis=1),.95))
        for c in comparisons:c['simultaneous95_exploratory']=[c['mean']-radius,c['mean']+radius]
    result=dict(experiment='E49',table=table,comparisons=comparisons,invalid_count=len(invalid),invalid=invalid,finished=previous.stamp())
    previous.write(OUT/'final/results.json',result)
    lines=['# E49：困难隐藏初态与逃跑基线对照','','6张密集地图、每图2个合法公开历史、冻结三模型；接管第12步，剩余68步。','',
        '|预算|对手|追捕方法|有效地图|双捕率|效用|完成成本|','|---|---|---|---:|---:|---:|---:|']
    for r in table:lines.append(f'|{r["mode"]}|{r["opponent"]}|{r["arm"]}|{r["maps"]}|{100*r["double_capture"]:.2f}%|{r["utility"]:.5f}|{r["completion_cost"]:.2f}|')
    lines+=['','200ms四项主要比较（6图同时bootstrap，探索性）：','']
    for c in comparisons:lines.append(f'- {c["opponent"]} {c["method"]}−{c["reference"]}: {c["mean"]:+.5f}, {c["simultaneous95_exploratory"]}')
    lines+=['',f'工程失效{len(invalid)}局，失效不作普通输局混算；如有失效，不据有效子集作方法优越性结论。',
        '几何逃跑是全状态有限前瞻参考，不是最优/学习专家；不读取真实追捕动作，预测追捕者仅为几何替代模型。',
        '新初态与E48不同，历史还消耗12步；跨批绝对分数不能作为单因素因果结论。两对手在本轮共用完全相同的初态与历史。',
        '固定16与名义200ms分别解释，所有原始轨迹和无效记录保留；不预设学习胜出。']
    (OUT/'final/REPORT.md').write_text('\n'.join(lines)+'\n')
    previous.write(OUT/'COMPLETE.json',dict(experiment='E49',finished=previous.stamp(),tasks=432,invalid=len(invalid),sha256=previous.sha(OUT/'final/results.json')))
    previous.write(OUT/'progress.json',dict(experiment='E49',phase='complete' if not invalid else 'completed_with_invalid_tasks',tasks=432,invalid=len(invalid),time=previous.stamp()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','acceptance','run','summarize'));args=parser.parse_args();cfg=prepare()
    if args.mode=='acceptance':acceptance(cfg)
    elif args.mode=='run':run(cfg)
    elif args.mode=='summarize':summarize(cfg)
