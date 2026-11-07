"""Execution-only E44 amendment: pinned processes and calibrated final timing.

The frozen runner and scientific algorithms remain unchanged. Each offline
label/model job calls the ORIGINAL functions in a private workspace; only
completed artifacts are copied into the original experiment directory.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import concurrent.futures as futures
import multiprocessing as mp
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import run_e44_branch_nte as base

ORIGINAL_STEP1=base.step1
ORIGINAL_LABELS=base.label_data
ORIGINAL_TRAIN=base.train
ORIGINAL_EVALUATE=base.evaluate
DEFAULT_SEED=base.BASE_SEED
DEFAULT_SEEDS=base.SEEDS
DEFAULT_KINDS=base.KINDS
WORKERS=32
CPUS=[]
WORKER_PARENT=None
OFFICIAL=None


def core_topology():
    cores={}
    for cpu in sorted(os.sched_getaffinity(0)):
        root=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=(int((root/'physical_package_id').read_text()),int((root/'core_id').read_text()))
        cores.setdefault(key,[]).append(cpu)
    return cores


def choose_cpus(number):
    cores=core_topology()
    def snapshot():
        result={}
        for line in Path('/proc/stat').read_text().splitlines():
            name,*fields=line.split()
            if name.startswith('cpu') and name[3:].isdigit():
                v=list(map(int,fields));result[int(name[3:])]=(sum(v[:8]),v[3]+v[4])
        return result
    a=snapshot();time.sleep(.25);b=snapshot()
    busy={cpu:1-(b[cpu][1]-a[cpu][1])/max(1,b[cpu][0]-a[cpu][0]) for cpu in b}
    selected=sorted(cores,key=lambda key:(max(busy[c] for c in cores[key]),key))[:number]
    return [min(cores[key],key=lambda c:(busy[c],c)) for key in selected]


def verify_execution():
    base.verify()
    p=base.OUT/'parallel_execution.json'
    config=base.read(p)
    assert base.sha(base.OUT/'experiment.json')==config['original_experiment_sha256']
    assert base.sha(base.OUT/'PROTOCOL.md')==config['original_protocol_sha256']
    for name,value in config['extension_hashes'].items():
        assert base.sha(name)==value,('Parallel extension changed',name)
    return config


def register():
    global CPUS,WORKERS
    path=base.OUT/'parallel_execution.json'
    if path.exists():
        config=verify_execution();CPUS=config['worker_cpus']
        assert config['workers']==WORKERS,'Resume with the registered worker count'
        return config
    CPUS=choose_cpus(WORKERS)
    if len(CPUS)<WORKERS:
        raise RuntimeError('Insufficient distinct available physical cores')
    sources=[Path(__file__).resolve(),base.ROOT/'code/tests/test_e44_parallel.py']
    config=dict(version='E44_execution_amendment_2',user_authorized='2026-10-27: increase parallelism, exploit abundant CPU/GPU resources',
        workers=WORKERS,worker_cpus=CPUS,threads_per_worker=1,physical_cores_distinct=True,
        parallel_stages=['Step1 development closed-loop (diagnostic only)','train/validation labels by map','nine independent model fits','final episodes ONLY after validation timing calibration'],
        serial_stages=['Step1 root diagnostics','timing reference on validation histories'],
        final_timing_calibration=dict(groups=3,methods=14,opponents=2,prefix_steps=2,
            worker_ladder=[n for n in (32,24,16,8,4,1) if n<=WORKERS],
            mean_simulation_ratio_minimum=.75,minimum_arm_simulation_ratio=.60,
            mean_cpu_wall_ratio_minimum=.85,additional_overrun_rate_maximum=.002,
            source='validation histories only; success and cost NEVER determine scheduling'),
        original_experiment_sha256=base.sha(base.OUT/'experiment.json'),
        original_protocol_sha256=base.sha(base.OUT/'PROTOCOL.md'),
        extension_hashes={str(p):base.sha(p) for p in sources},
        existing_development_episodes={str(p):base.sha(p) for p in (base.OUT/'step1/closed_loop/cache').glob('*.json')},
        algorithm_data_training_and_statistics_unchanged=True,original_12h_clock_preserved=True,
        serial_timing_requirement_explicitly_amended=True,
        development_timing_scope='mixed serial/parallel diagnostic timings; not final real-time evidence',
        episode_counts_and_model_hyperparameters_unchanged=True)
    base.save(path,config)
    for p in sources:
        target=base.OUT/'execution_snapshot'/p.name;target.parent.mkdir(exist_ok=True)
        target.write_bytes(p.read_bytes())
    text=('# E44 执行调度补充协议\n\n'
        '用户于2026-10-27再次要求充分利用资源加速。原 experiment.json、PROTOCOL.md、算法源码、地图、随机种子、教师选择、标签、九模型超参、1008局正式队列及12小时起算点保留。本文件明确补充原串行调度要求，不静默改写原注册。\n\n'
        f'最多{WORKERS}个单数值线程进程，绑定不同物理核。Step 1开发闭环并行，既有完整结果保留；该阶段时延只作诊断，不能混入最终实时收益。统一标签按地图并行，九个模型独立并行。每个任务在私有工作目录调用原函数，完成后原子复制归档，由协调进程生成共享清单。\n\n'
        'Step 1根诊断仍串行。正式测试前，在3类验证历史、14执行臂、2目标条件上分别跑2步串行计时参考及并行计时校准（84请求/模式）。优先32路，必要时24/16/8/4，仍不满足则串行；最大值受注册worker数限制。校准只读时延、CPU时间和模拟次数，不按成功率或成本选并行度。要求整体模拟次数比>=.75，各搜索臂>=.60，动作CPU/墙钟比>=.85，超时比例增幅<=.002。保存全部校准原始轨迹及结论；未通过不阻止训练或完整测试，只降低并发。\n\n'
        '通过后才并行执行完整1008局，每进程绑定不同物理核、每决策200ms墙钟不变，首次初始化仍计入，每帧额外记录CPU时间。最终时延为该并行负载下的结果，不能冒称原串行时延；原超时收益判断保留。失败不丢弃、不补抽样，不增加性能门槛。原源码摘要完整核验；新增调度器另行散列冻结。\n')
    (base.OUT/'EXECUTION_AMENDMENT.md').write_text(text)
    return config


def worker_emit(**values):
    record=dict(pid=os.getpid(),timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),cpu=list(os.sched_getaffinity(0)),**values)
    folder=OFFICIAL/'parallel_workers';folder.mkdir(exist_ok=True)
    base.save(folder/f'{os.getpid()}.json',record)
    with (folder/f'{os.getpid()}.jsonl').open('a') as f:
        import json
        f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')


def check_parent():
    if WORKER_PARENT is not None and os.getppid()!=WORKER_PARENT:
        raise RuntimeError('Coordinator exited; stop orphan offline work')


def worker_init(output,smoke,cpu_queue,parent,stage):
    global OFFICIAL,WORKER_PARENT
    OFFICIAL=Path(output);WORKER_PARENT=parent
    os.sched_setaffinity(0,{cpu_queue.get()})
    torch.set_num_threads(1)
    base.OUT=OFFICIAL;base.SMOKE=smoke;base.emit=worker_emit
    verify_execution()
    if stage!='training':
        # Warm compiled signatures outside every timed query. Discard controllers.
        chosen=[next(e for e in base.entries('validation') if e['family']==family)
            for family in ('long_wall','wall_corner','multi_route')]
        for entry in chosen:
            packet=base.read(entry['packet_path'])
            methods=base.METHODS if stage in ('evaluation','calibration') else base.MECHANISMS
            for method in methods:
                c,_=base.initialize(packet,method,317)
                c.act(base.observation(packet['history'][-1]),fixed_simulations=2)
            if stage=='labels':
                c.deadline=float('inf')
                if c.root is not None:
                    base.independent_values(c,[c.root.actions[0]],317,draws=1)
    worker_emit(stage='worker_ready',job_stage=stage)


def atomic_copy(source,target):
    source,target=Path(source),Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and base.sha(source)==base.sha(target):return
    temp=target.with_name(target.name+f'.copy-{os.getpid()}')
    shutil.copyfile(source,temp);temp.replace(target)


def workspace(name):
    path=OFFICIAL/'parallel_workspaces'/name;path.mkdir(parents=True,exist_ok=True)
    atomic_copy(OFFICIAL/'experiment.json',path/'experiment.json')
    return path


def committed_episode(req):
    cached=base.cached_episode(req)
    if cached is not None:return cached['key']
    timing=[]
    def initialize(packet,method,seed):
        controller,setup=base.initialize(packet,method,seed)
        original_act=controller.act
        def act(obs,fixed_simulations=None):
            cpu=time.process_time();wall=time.perf_counter()
            try:return original_act(obs,fixed_simulations=fixed_simulations)
            finally:timing.append(dict(act_cpu_ms=1000*(time.process_time()-cpu),act_wall_ms=1000*(time.perf_counter()-wall)))
        controller.act=act
        return controller,setup
    limit=req.get('engineering_limit',3 if base.SMOKE else None)
    fixed=[8]*limit if base.SMOKE and limit else None
    payload=base.run_episode(req,limit=limit,fixed=fixed,initializer=initialize)
    payload['summary']['execution_mode']='pinned_parallel' if WORKER_PARENT is not None else 'serial_calibration'
    payload['summary']['worker_cpu']=list(os.sched_getaffinity(0))
    for trace,values in zip(payload['trace'],timing):trace.update(values)
    payload['summary']['act_cpu_ms']=[r['act_cpu_ms'] for r in payload['trace']]
    payload['summary']['act_wall_ms']=[r['act_wall_ms'] for r in payload['trace']]
    if req['stage']=='evaluation' or req['stage'].startswith('timing/'):
        base.save(base.OUT/req['stage']/'episodes'/(req['key']+'.json'),payload)
        audit=base.audit_payload(payload)
        base.save(base.OUT/req['stage']/'audits'/(req['key']+'.json'),audit)
    row=base.commit_episode(req,payload)
    return row['key']


def episode_job(req):
    check_parent();base.OUT=OFFICIAL
    key=committed_episode(req)
    worker_emit(stage='development_episode',key=req['key'])
    return key


def label_job(task):
    check_parent();base.OUT=OFFICIAL;base.BASE_SEED=DEFAULT_SEED
    split,index,entry=task
    base.collect_packets(entry)  # ORIGINAL collection seed, before label offset.
    work=workspace(f'labels_{split}_{entry["id"]}')
    base.save(work/'bank/manifest.json',dict(entries=[entry]))
    atomic_copy(OFFICIAL/'step1/selection.json',work/'step1/selection.json')
    relative=Path('collection')/split/(entry['id']+'.json')
    atomic_copy(OFFICIAL/relative,work/relative)
    # Original label_data enumerates each split from zero. Preserve its original
    # global map index exactly while this private workspace contains one map.
    base.BASE_SEED=DEFAULT_SEED+index*1009;base.OUT=work
    ORIGINAL_LABELS()
    check_parent()
    folder=work/'data'/split/entry['id']
    for p in sorted(folder.iterdir()):
        if p.suffix in ('.json','.npz'):atomic_copy(p,OFFICIAL/'data'/split/entry['id']/p.name)
    count=len(list(folder.glob('*.npz')))
    worker_emit(stage='label_map_complete',split=split,group=entry['id'],states=count)
    return dict(split=split,group=entry['id'],states=count)


def training_job(task):
    check_parent();base.OUT=OFFICIAL;base.BASE_SEED=DEFAULT_SEED
    kind,seed=task;work=workspace(f'training_{kind}_{seed}')
    link=work/'data'
    if not link.exists():link.symlink_to(OFFICIAL/'data',target_is_directory=True)
    # Restore any earlier serial partial model without sharing writable files.
    if not (work/'models').exists():
        for folder in ('models','training'):
            for p in (OFFICIAL/folder).glob(f'{kind}_{seed}*'):
                if p.is_file():atomic_copy(p,work/folder/p.name)
    base.OUT=work;base.SEEDS=(seed,);base.KINDS=(kind,)
    ORIGINAL_TRAIN()
    check_parent()
    for folder in ('models','training'):
        for p in (work/folder).glob(f'{kind}_{seed}*'):
            if p.is_file() and p.suffix in ('.pt','.npz','.json'):atomic_copy(p,OFFICIAL/folder/p.name)
    record=base.read(work/'training'/f'{kind}_{seed}.json')
    worker_emit(stage='model_complete',kind=kind,seed=seed,epochs=record['epochs'])
    return record


def run_jobs(stage,callback,jobs,workers=None):
    if not jobs:return []
    count=min(WORKERS if workers is None else workers,len(jobs));ctx=mp.get_context('spawn');queue=ctx.Queue()
    for cpu in CPUS[:count]:queue.put(cpu)
    base.emit(stage='parallel_'+stage,completed=0,total=len(jobs),workers=count,worker_cpus=CPUS[:count])
    pool=futures.ProcessPoolExecutor(max_workers=count,mp_context=ctx,initializer=worker_init,
        initargs=(str(base.OUT),base.SMOKE,queue,os.getpid(),stage))
    done=[]
    try:
        pending=[pool.submit(callback,job) for job in jobs]
        for future in futures.as_completed(pending):
            done.append(future.result())
            base.emit(stage='parallel_'+stage,completed=len(done),total=len(jobs),workers=count)
        pool.shutdown(wait=True)
    except BaseException:
        processes=list((pool._processes or {}).values())
        pool.shutdown(wait=False,cancel_futures=True)
        for process in processes:
            if process.is_alive():process.terminate()
        for process in processes:process.join(timeout=5)
        raise
    finally:
        queue.close()
    return done


def parallel_step1():
    original_cached=base.cached_episode;dispatched=False
    def cached(req):
        nonlocal dispatched
        if req['stage']=='step1/closed_loop' and not dispatched:
            dispatched=True
            development=[entry for family in ('long_wall','wall_corner','multi_route')
                for entry in [e for e in base.entries('train') if e['family']==family][:1 if base.SMOKE else 4]]
            jobs=[base.request(e,w,m,'step1/closed_loop',o) for e in development for o in base.OPPONENTS
                for w in (('A',) if base.SMOKE else ('A','B')) for m in ('G',)+base.MECHANISMS]
            jobs=[job for job in jobs if original_cached(job) is None]
            run_jobs('development',episode_job,jobs)
        return original_cached(req)
    base.cached_episode=cached
    try:return ORIGINAL_STEP1()
    finally:base.cached_episode=original_cached


def parallel_labels():
    if not (base.OUT/'data/manifest.json').exists():
        jobs=[(split,i,entry) for split in ('train','validation') for i,entry in enumerate(base.entries(split))]
        run_jobs('labels',label_job,jobs)
    # Original function checks the completed common files, counts states and
    # commits the ONE shared manifest. It does not regenerate completed labels.
    return ORIGINAL_LABELS()


def parallel_train():
    if (base.OUT/'models_frozen.json').exists():return ORIGINAL_TRAIN()
    jobs=[(kind,seed) for seed in DEFAULT_SEEDS for kind in DEFAULT_KINDS
        if not (base.OUT/'training'/f'{kind}_{seed}.json').exists()]
    run_jobs('training',training_job,jobs)
    records=[base.read(base.OUT/'training'/f'{kind}_{seed}.json') for seed in DEFAULT_SEEDS for kind in DEFAULT_KINDS]
    assert len(records)==9
    for r in records:
        assert r['epochs']==base.read(base.OUT/'experiment.json')['epochs']
        assert base.sha(base.model_path(r['kind'],r['seed']))==r['model_sha256']
    base.save(base.OUT/'training.json',dict(started=True,completed=True,expected_models=9,records=records,
        train_states=len(list((base.OUT/'data/train').rglob('*.npz'))),
        validation_states=len(list((base.OUT/'data/validation').rglob('*.npz'))),test_outcomes_read=False))
    base.save(base.OUT/'models_frozen.json',dict(files={str(p):base.sha(p) for p in (base.OUT/'models').iterdir() if p.is_file()},
        selection='minimum validation Cartesian action NLL within fixed epochs; all nine models',test_outcomes_read=False))
    base.write_live_report('九个模型并行训练完成并冻结；开始验证集时延校准，然后完整测试。')
    return base.read(base.OUT/'training.json')


def timing_metrics(rows):
    decisions=[t for r in rows for t in base.registered_latencies(r)]
    arms={method:sum(sum(r['hidden_simulations']) for r in rows if r['method']==method)
        for method in base.METHODS if method not in ('G',) and not method.startswith('DIRECT')}
    cpu=sum(sum(r['act_cpu_ms']) for r in rows);wall=sum(sum(r['act_wall_ms']) for r in rows)
    return dict(decisions=len(decisions),p95_ms=float(np.percentile(decisions,95)),
        overrun_rate=sum(t>200. for t in decisions)/len(decisions),
        mean_cpu_wall_ratio=cpu/max(wall,1e-9),simulations_by_arm=arms)


def calibration_requests(stage):
    selected=[next(e for e in base.entries('validation') if e['family']==family)
        for family in ('long_wall','wall_corner','multi_route')]
    return [base.request(e,'A',m,stage,o)|dict(engineering_limit=2) for e in selected
        for o in base.OPPONENTS for m in base.METHODS]


def calibrate():
    path=base.OUT/'timing/decision.json'
    if path.exists():return base.read(path)['evaluation_workers']
    if base.SMOKE:
        base.save(path,dict(evaluation_workers=min(WORKERS,2),smoke=True,performance_conclusion=False))
        return min(WORKERS,2)
    serial_requests=calibration_requests('timing/serial')
    for i,req in enumerate(serial_requests):
        committed_episode(req)
        base.emit(stage='timing_serial_reference',completed=i+1,total=len(serial_requests))
    serial=timing_metrics([base.cached_episode(req) for req in serial_requests]);records=[]
    ladder=sorted({WORKERS}|{n for n in (32,24,16,8,4) if n<WORKERS},reverse=True)
    for n in ladder:
        if n==1:break
        requests=calibration_requests(f'timing/parallel_{n}')
        run_jobs('calibration',episode_job,[req for req in requests if base.cached_episode(req) is None],workers=n)
        metrics=timing_metrics([base.cached_episode(req) for req in requests])
        ratios={arm:metrics['simulations_by_arm'][arm]/max(1,v) for arm,v in serial['simulations_by_arm'].items() if v>0}
        ratio=sum(metrics['simulations_by_arm'].values())/max(1,sum(serial['simulations_by_arm'].values()))
        passed=bool(ratio>=.75 and min(ratios.values(),default=1)>=.60
            and metrics['mean_cpu_wall_ratio']>=.85 and metrics['overrun_rate']<=serial['overrun_rate']+.002)
        records.append(dict(workers=n,metrics=metrics,simulation_ratio=ratio,arm_ratios=ratios,passed=passed))
        base.save(base.OUT/'timing/calibration.json',dict(serial=serial,candidates=records,validation_only=True))
        if passed:
            base.save(path,dict(evaluation_workers=n,serial_reference=serial,candidates=records,
                selected_using_latency_and_throughput_only=True,test_outcomes_read=False))
            return n
    base.save(path,dict(evaluation_workers=1,serial_reference=serial,candidates=records,
        selected_using_latency_and_throughput_only=True,test_outcomes_read=False))
    return 1


def parallel_evaluate():
    verify_execution();n=calibrate()
    if n==1:return ORIGINAL_EVALUATE()
    requests=[]
    for i,entry in enumerate(base.entries('test')):
        for opponent in base.OPPONENTS:
            for wi,world in enumerate(('A',) if base.SMOKE else ('A','B')):
                offset=(i+wi)%len(base.METHODS)
                requests.extend(base.request(entry,world,m,'evaluation',opponent)
                    for m in base.METHODS[offset:]+base.METHODS[:offset])
    assert len(requests)==base.read(base.OUT/'experiment.json')['evaluation_games']
    base.save(base.OUT/'evaluation/requests.json',requests)
    run_jobs('evaluation',episode_job,[r for r in requests if base.cached_episode(r) is None],workers=n)
    rows=[base.cached_episode(req) for req in requests]
    for req in requests:
        assert (base.OUT/'evaluation/audits'/(req['key']+'.json')).exists()
    base.save(base.OUT/'evaluation/results.json',rows)
    result=base.summarize(rows)
    result['execution']=dict(evaluation_workers=n,physical_cores=CPUS[:n],
        timing_scope='pinned concurrent workload; not serial latency',validation_calibration=str(base.OUT/'timing/decision.json'))
    base.save(base.OUT/'evaluation/statistics.json',result)
    return result


def main():
    global WORKERS
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('pipeline','smoke'))
    parser.add_argument('--workers',type=int,default=32);parser.add_argument('--output',type=Path)
    args=parser.parse_args();WORKERS=args.workers
    if WORKERS<1:parser.error('--workers must be positive')
    if args.output:base.OUT=args.output.resolve()
    if args.stage=='smoke':base.SMOKE=True
    torch.set_num_threads(1)
    base.prepare();register();verify_execution()
    original_report=base.write_live_report
    def report(state):
        original_report(state)
        with (base.OUT/'REPORT.md').open('a') as f:
            f.write('\n执行调度已补充注册：[并行补充协议](EXECUTION_AMENDMENT.md)。最多'+str(WORKERS)
                +'个独立物理核进程；最终并行度由[验证集计时校准](timing/decision.json)确定。并行测试时延仅代表该并发负载，不能冒称原串行时延。\n')
    base.write_live_report=report
    base.step1=parallel_step1;base.label_data=parallel_labels;base.train=parallel_train;base.evaluate=parallel_evaluate
    base.pipeline();verify_execution()


if __name__=='__main__':main()
