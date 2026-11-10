"""E48 exploratory speed calibration; original E48--E51 picture batches stay archived."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TZ'] = 'Asia/Shanghai'
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/speed_difficulty_mpl')

import argparse
import collections
import concurrent.futures
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import signal
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'current/e48_speed_difficulty_20261107'
OLD = ROOT/'current/e46_geometry_teacher_20261105'
SPEEDS = (1., 1.3, 1.5)
ARMS = ('A4', 'AG', 'A2', 'A3')
FAMILIES = ('long_wall', 'wall_corner', 'multi_route', 'split_gate', 'staggered_gate', 'u_courtyard', 'post_field')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()


def specs():
    return [dict(arm=arm, method=arm, train_seed=17,
                 checkpoint=str(OLD/'training/L5_17_base/selected.npz') if arm=='A3' else None)
            for arm in ARMS]


def prepare():
    if (OUT/'experiment.json').exists():
        return read(OUT/'experiment.json')
    cfg = read(OLD/'experiment.json')
    chosen = []
    for family in FAMILIES:
        candidates = sorted((e for e in cfg['entries'] if e['split']=='development'
                             and e['family']==family and abs(e['speed']-1.3)<1e-10), key=lambda e:e['id'])
        if not candidates:
            raise RuntimeError('No outcome-blind 1.3-speed development map: '+family)
        entry = dict(candidates[0])
        case = read(entry['path'])
        target = OUT/'bank'/(entry['id']+'.json')
        write(target, case)
        chosen.append(dict(entry, original_path=entry['path'], original_sha256=entry['sha256'],
                           path=str(target), sha256=sha(target)))
    protected = dict(cfg['source_hashes'])
    e47 = read(ROOT/'current/l5_followup_20261107/experiment.json')
    protected.update(e47['source_hashes'])
    protected[str(OLD/'training/L5_17_base/selected.npz')] = sha(OLD/'training/L5_17_base/selected.npz')
    for path, digest in protected.items():
        if sha(path) != digest:
            raise RuntimeError('Historical protected artifact differs: '+path)
    own = {str(ROOT/'code'/name):sha(ROOT/'code'/name)
           for name in ('speed_difficulty_adapter.py', 'run_speed_difficulty.py')}
    for path in own:
        dest = OUT/'source_snapshot'/Path(path).name
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(Path(path).read_bytes())
    result = dict(experiment='E48-speed-difficulty-20261107', created=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                  entries=chosen, methods=specs(), evader_speeds=list(SPEEDS), optional_speed=1.8,
                  optional_rule='Run 1.8 only if best valid arm double-capture >=90% at 1.5; no further speed levels.',
                  pursuer_speed=1.3, opponent='rootpuct', opponent_simulations=3000,
                  worlds=['AB','BA'], nominal_ms=200, base_episodes=168, optional_episodes=56,
                  model_adaptation='Known evader bound; grid h=.5*v, refinement h=(.25,.125,.0625)*v; sigma=.35*v; route means and tree speed checks v.',
                  fixed_conditions='Original 2v2, horizon80, range6, exact shared observations, capture radius1 and paired retirement; lockstep execution.',
                  training=False, model_seed=17, protected=protected, source_hashes=own,
                  scope='Exploratory calibration on seven already-viewed development maps, one frozen model; not sealed confirmation or SOTA comparison.')
    write(OUT/'experiment.json', result)
    (OUT/'docs').mkdir(parents=True, exist_ok=True)
    (OUT/'docs/PLAN.md').write_text('''# E48：逃跑速度难度标定

用户授权重新实验以减轻高捕获率的天花板。新E48承接E47，旧临时E48—51仍只是归档演示，均不恢复为正式实验。

七个既有开发布局族各取按ID排序第一张追捕上限1.3的地图，不按结果筛图；AB/BA初态、公开历史及环境种子在速度档间配对。追捕每轴上限1.3；逃跑每轴上限1.0/1.3/1.5。A4、AG、A2、A3四法，L5仅冻结17模型，共168局。若1.5档最强有效方法双捕仍≥90%，按预定规则追加1.8档56局；不再继续提速。未找到非饱和档也如实报告，不挑赢家。

保留原RootPUCT3000逃跑者，独立响应各方法当前状态；不使用新SOTA标签，不训练。所有档保持原捕获/退休、感知、时域、锁步执行。A2/A3使用完整名义200ms，A4/AG直接返回且报告实际耗时；不宣称硬实时。

新进程局部适配把已知逃跑速度同步到环境、C++动作界限、D普通更新/历史恢复及树内模拟。固定相对网格分辨率h=.5v、噪声sigma=.35v，意味着模型的数值分辨率也按速度缩放，不能称逐位不变的原D；v=1保留原核。网络没有逃跑速度新输入且不重训，属于冻结网络的跨速度迁移。工程错误单列，不能算正常未捕获。

工程先检查v=1保留、快速移动合法边、恢复转移与树内速度一致，以及四法短对局。通过后冻结新源码与主任务列表，历史源码/权重逐项校验。保留原始轨迹、失败现场、完整返回时延及观测，公布双捕率、至少一捕率、目标捕获比例及首捕后失败分型。七图结果仅用于开发判断，不作显著性或最优性宣言。

CPU使用8个独立物理核并先作并发校准；不用GPU、至少保留20逻辑CPU。每局300秒看门狗，无限长尾作为工程失败保留；不修改旧结果、不提交Git。
''')
    return result


def verify(cfg):
    for section in ('protected', 'source_hashes'):
        for path, digest in cfg[section].items():
            if sha(path) != digest:
                raise RuntimeError('Frozen artifact changed: '+path)
    for e in cfg['entries']:
        if sha(e['path']) != e['sha256']:
            raise RuntimeError('Case changed')


def tasks(cfg, speeds=None, stage='main', limit=None):
    from learning.l5_followup import stable_seed
    for entry in cfg['entries']:
        for world in cfg['worlds']:
            seed = stable_seed('speed-calibration-20261107', entry['sha256'], world)
            for speed in speeds if speeds is not None else cfg['evader_speeds']:
                for spec in cfg['methods']:
                    yield dict(spec, entry=entry, case=entry['path'], group=entry['id'], family=entry['family'],
                               suite=entry['suite'], split='development', world=world, seed=seed,
                               opponent='rootpuct', stage=stage, round=0, model_round=12,
                               budget_ms=200, evader_speed=speed, limit=limit)


def diagnostics(payload):
    trace=payload['trace']; summary=payload['summary']; first=summary['first_capture']
    hidden=[]; longest=0; run=[0,0]; target_steps=0; above_old=0; shields=0
    for frame in trace:
        obs=frame['observation']; shields+=frame['opponent_shields']
        for j in range(2):
            if not obs['active_e'][j]:
                run[j]=0; continue
            target_steps+=1
            missing=not any(q is not None for q in obs['reports'][j])
            hidden.append(int(missing)); run[j]=run[j]+1 if missing else 0; longest=max(longest,run[j])
            above_old+=int(np.max(np.abs(frame['opponent_action'][j]))>1+1e-7)
    category='double_capture' if summary['double_capture'] else 'no_first_capture'
    if not summary['double_capture'] and first is not None:
        remaining=next(j for j in range(2) if trace[-1]['next_state'][9+j]>.5)
        post=trace[first:]
        seen=sum(any(q is not None for q in f['observation']['reports'][remaining]) for f in post)
        category='post_capture_never_seen' if not seen else 'post_capture_seen_not_caught'
    return dict(at_least_one=bool(summary['captures']>0), captured_fraction=summary['captures']/2.,
                hidden_target_steps=sum(hidden), active_target_steps=target_steps,
                longest_hidden_steps=longest, above_old_limit_steps=above_old,
                opponent_shields=shields, failure_category=category)


def alarm(*_):
    raise TimeoutError('Speed-calibration per-game 300s watchdog')


def execute(task):
    from speed_difficulty_adapter import install
    import run_e46_geometry_teacher as old
    target=OUT/'results'/task['stage']/(identity(task)+'.json')
    if target.exists():
        return read(target)
    start=time.perf_counter()
    signal.signal(signal.SIGALRM, alarm); signal.alarm(300)
    try:
        install(task['evader_speed'], OUT)
        result=old.run_episode(task, OUT)
        payload=old.load_pickle(result['episode'])
        extra=diagnostics(payload)
        case=read(task['case']); boxes=np.asarray(case['map'])
        from run_frozen_d_closed_loop import legal_action
        for frame in payload['trace']:
            for j in range(2):
                if not legal_action(np.asarray(frame['state'])[2*j:2*j+2],
                                    np.asarray(frame['opponent_action'][j]),task['evader_speed'],boxes):
                    raise RuntimeError('Illegal actual evader displacement')
        result['summary'].update(extra)
        row=dict(valid=True, task=task, result=result, wall_seconds=time.perf_counter()-start,
                 finished=time.strftime('%Y-%m-%d %H:%M:%S %z'),cpu=sorted(os.sched_getaffinity(0)))
    except Exception as exc:
        row=dict(valid=False, task=task, error=repr(exc), traceback=traceback.format_exc(),
                 wall_seconds=time.perf_counter()-start,finished=time.strftime('%Y-%m-%d %H:%M:%S %z'))
    finally:
        signal.alarm(0)
    write(target,row)
    return row


def cpu_list():
    selected=[]; physical=set()
    allowed=sorted(os.sched_getaffinity(0))
    for cpu in [c for c in allowed if c>=16]+[c for c in allowed if c<16]:
        topo=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key=((topo/'physical_package_id').read_text().strip(),(topo/'core_id').read_text().strip())
        if key not in physical:
            selected.append(cpu);physical.add(key)
        if len(selected)==8:
            break
    if len(allowed)-len(selected)<20:
        raise RuntimeError('Insufficient reserved CPU capacity')
    return selected


def worker_init(queue):
    os.sched_setaffinity(0,{queue.get()})


def calibration_probe(arg):
    case_path,cpu=arg
    os.sched_setaffinity(0,{cpu})
    from speed_difficulty_adapter import install
    from learning.l5_followup import make_controller
    install(1.3,OUT)
    case=read(case_path); samples=[]
    for repeat in range(4):
        own=make_controller(case,dict(arm='A2',seed=177,budget_ms=200))
        for obs in case['history'][:-1]:
            own.observe(obs)
        start=time.perf_counter(); own.act(case['history'][-1],fixed_simulations=16)
        elapsed=1000*(time.perf_counter()-start)
        if repeat:
            samples.append(elapsed)
    return samples


def calibrate(cpus,cfg):
    affinity=os.sched_getaffinity(0)
    reference=calibration_probe((cfg['entries'][0]['path'],cpus[0]))
    os.sched_setaffinity(0,affinity)
    attempts=[]
    while True:
        with concurrent.futures.ProcessPoolExecutor(len(cpus),mp_context=multiprocessing.get_context('fork')) as pool:
            parallel_results=list(pool.map(calibration_probe,[(cfg['entries'][0]['path'],c) for c in cpus]))
        pooled=[v for samples in parallel_results for v in samples]
        ratio=float(np.median(pooled)/np.median(reference))
        tail=float(np.percentile(pooled,95)/np.percentile(reference,95))
        passed=ratio<=1.35 and tail<=1.5
        attempts.append(dict(cpus=cpus,reference_ms=reference,parallel_ms=parallel_results,
                             median_ratio=ratio,p95_ratio=tail,passed=passed))
        write(OUT/'timing_calibration.json',dict(attempts=attempts,selected_cpus=cpus if passed else None))
        if passed:
            return cpus
        if len(cpus)==1:
            raise RuntimeError('Serial/concurrent timing calibration failed')
        cpus=cpus[:max(1,len(cpus)//2)]


def batch(jobs, cpus):
    context=multiprocessing.get_context('fork'); queue=context.Queue()
    for cpu in cpus:
        queue.put(cpu)
    count=0
    with concurrent.futures.ProcessPoolExecutor(len(cpus),mp_context=context,initializer=worker_init,initargs=(queue,)) as pool:
        for future in concurrent.futures.as_completed([pool.submit(execute,t) for t in jobs]):
            row=future.result();count+=1
            write(OUT/'progress.json',dict(stage=jobs[0]['stage'],completed=count,total=len(jobs),
                                         last_valid=row['valid'],last_error=row.get('error'),time=time.time()))
            print(f'{count}/{len(jobs)} {row["task"]["arm"]} v={row["task"]["evader_speed"]} valid={row["valid"]} '+row.get('error',''),flush=True)
    return [read(OUT/'results'/t['stage']/(identity(t)+'.json')) for t in jobs]


def summarize(cfg):
    rows=[read(p) for p in (OUT/'results/main').glob('*.json')]
    by=collections.defaultdict(list)
    for r in rows:
        by[r['task']['evader_speed'],r['task']['arm']].append(r)
    table=[]
    for (speed,arm),group in sorted(by.items()):
        valid=[r['result']['summary'] for r in group if r['valid']]
        n=len(group); good=len(valid)
        row=dict(evader_speed=speed,arm=arm,attempted=n,valid=good,engineering_failures=n-good)
        if good:
            for name in ('double_capture','at_least_one','captured_fraction','completion_cost','decisions','utility','longest_hidden_steps'):
                row[name]=float(np.mean([s[name] for s in valid]))
            row['failure_categories']=dict(collections.Counter(s['failure_category'] for s in valid))
            row['opponent_shields']=sum(s['opponent_shields'] for s in valid)
            row['above_old_limit_steps']=sum(s['above_old_limit_steps'] for s in valid)
            lat=[x for s in valid for x in s['latencies_ms']]
            row.update(p95_ms=float(np.percentile(lat,95)),max_ms=max(lat),
                       zero_simulations_note='A4/AG are direct; only A2/A3 simulation counts describe search.')
            wins=sum(s['double_capture'] for s in valid)
            row['double_failure_bounds']=[wins/n,(wins+n-good)/n]
        table.append(row)
    result=dict(table=table,rows=rows,scope=cfg['scope'],source_hashes=cfg['source_hashes'])
    write(OUT/'final/results.json',result)
    lines=['# E48：逃跑速度难度标定（探索性）','',cfg['scope'],'',
           '追捕每轴上限1.3；同一七张开发地图及AB/BA初态，RootPUCT3000；网络冻结17，名义200 ms锁步执行。',
           '捕获率分别报告“至少捕获一个目标的局比例”和“捕获目标数/2”。工程失败不当普通输局。','',
           '|逃跑上限|方法|有效/尝试|双捕|至少一捕|目标捕获比例|完成成本|工程失败|',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        vals=[f'{100*r[k]:.2f}%' for k in ('double_capture','at_least_one','captured_fraction')] if r['valid'] else ['—']*3
        cost=f'{r["completion_cost"]:.2f}' if r['valid'] else '—'
        lines.append(f'|{r["evader_speed"]}|{r["arm"]}|{r["valid"]}/{r["attempted"]}|'+ '|'.join(vals)+f'|{cost}|{r["engineering_failures"]}|')
    lines+=['','## 失败与执行诊断','', '|逃跑上限|方法|失败分型|P95 ms|最大 ms|对手动作被屏蔽次数|','|---|---|---|---:|---:|---:|']
    for r in table:
        if r['valid']:
            lines.append(f'|{r["evader_speed"]}|{r["arm"]}|{r["failure_categories"]}|{r["p95_ms"]:.2f}|{r["max_ms"]:.2f}|{r["opponent_shields"]}|')
    lines+=['','模型按已知速度重参数化，网格/噪声也随速度缩放；不能把结果解释为只变环境、估计器逐位不变。',
            '冻结网络未针对新速度训练；失败可能包含模型近似、跨速度迁移与运动能力限制，不能仅凭胜率判断哪项是唯一原因。',
            '本轮不证明几何已失效、网络优越、实车性能或统计显著性。既有E46/E47结果保持独立。']
    (OUT/'final/REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','engineering','run','report']);args=parser.parse_args()
    cfg=prepare();verify(cfg)
    if args.mode=='prepare':
        print(json.dumps(dict(path=str(OUT),episodes=cfg['base_episodes']),ensure_ascii=False));return
    if args.mode=='report':
        summarize(cfg);return
    from speed_difficulty_adapter import install, speed_lattice, speed_map_probabilities
    # Warm the new kernels outside online latency measurements and before fork.
    for speed in SPEEDS:
        install(speed,OUT)
        case=read(cfg['entries'][0]['path'])
        from solvers.frozen_d_search_interface import loader,Observation
        est=loader.load_estimator(case['map'],17)
        for obs in case['history']:
            positive=[(i,q) for i,q in enumerate(obs['reports'][0]) if q is not None]
            at,target=positive[0] if positive else (0,None)
            est.update(Observation(obs['step'],np.asarray(obs['positions'][at]),None if target is None else np.asarray(target),obs['remaining']))
    cpus=calibrate(cpu_list(),cfg)
    write(OUT/'resources.json',dict(cpus=cpus,workers=len(cpus),gpu=False,reserved_logical=len(os.sched_getaffinity(0))-len(cpus)))
    if args.mode=='engineering':
        jobs=list(tasks(dict(cfg,entries=cfg['entries'][:1],worlds=['AB']),stage='engineering',limit=3))
        rows=batch(jobs,cpus[:4]);ok=all(r['valid'] for r in rows)
        write(OUT/'ENGINEERING.json',dict(passed=ok,tasks=len(rows),source_hashes=cfg['source_hashes']))
        if not ok:
            raise RuntimeError('Engineering gate failed; main batch not authorized by gate')
        return
    if not read(OUT/'ENGINEERING.json')['passed']:
        raise RuntimeError('Engineering gate missing')
    rows=batch(list(tasks(cfg)),cpus)
    result=summarize(cfg)
    top=[r['double_capture'] for r in result['table'] if r['evader_speed']==1.5 and r['valid'] and not r['engineering_failures']]
    if len(top)==4 and max(top)>=.9:
        write(OUT/'optional_speed_trigger.json',dict(triggered=True,speed=1.8,criterion=max(top)))
        rows+=batch(list(tasks(cfg,speeds=[1.8])),cpus)
        summarize(cfg)
    verify(cfg)
    write(OUT/'COMPLETE.json',dict(finished=time.strftime('%Y-%m-%d %H:%M:%S %z'),attempted=len(rows),
                                 valid=sum(r['valid'] for r in rows),sha256=sha(OUT/'final/results.json')))


if __name__=='__main__':
    main()
