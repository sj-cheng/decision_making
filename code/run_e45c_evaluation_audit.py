"""Finish independent E45-C evaluations while retaining support failures.

This does NOT repair/refine D or change a controller. Known support errors
produce an explicit INVALID observation with null task outcomes. Unexpected
errors still stop execution. Planned primary inference requires every one of
its original cells; no failed maps are silently dropped or assigned rewards.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import collections
import concurrent.futures as futures
import fcntl
import re
import signal
import time
import traceback
from pathlib import Path
import numpy as np
import run_e45c_budget as r
import run_e45c_accelerated as acceleration
from e45c_belief_diagnostics import SUPPORT_ERRORS

MANIFEST='EVALUATION_FAILURE_CONTINUATION_20261028.json'
ORIGINAL_EPISODE_JOB=r.episode_job


def verify():
    config=r.read(r.OUT/MANIFEST)
    assert config['original_experiment_sha256']==r.sha(r.OUT/'experiment.json')
    for name in ('source_hashes','bound_artifacts'):
        for p,h in config[name].items():
            if r.sha(p)!=h:raise RuntimeError('Frozen evaluation-audit artifact changed: '+p)
    return config


def identity(row):return row['_output'],row['key']


def is_support_error(error):
    return any(error.startswith(prefix) for message in SUPPORT_ERRORS
        for prefix in (message,"RuntimeError('"+message,'RuntimeError("'+message))


def invalid_result(task,failure):
    if failure['task']!=task:raise RuntimeError('Failure record task mismatch')
    if task['stage']!='evaluation' or not is_support_error(failure['error']):
        raise RuntimeError('Not an eligible evaluation support failure')
    match=re.search(r'''public_history_evidence=([^\s'";)]+)''',failure['error'])
    if not match:raise RuntimeError('Known support failure lacks a replayable public history')
    stem=Path(match.group(1));evidence=stem.with_suffix('.json')
    if not evidence.resolve().is_relative_to((r.OUT/'belief_failures').resolve()):
        raise RuntimeError('Public evidence outside registered experiment')
    public=r.read(evidence);case=r.read(task['case'])
    if public['seed']!=task['seed'] or public['boxes']!=case['map'] or public.get('recovered'):
        raise RuntimeError('Support evidence identity mismatch')
    row=dict(**task,valid_episode=False,reason='belief_support_exhaustion',failure=failure['error'],
        double_capture=None,completion_cost=None,utility=None,first_capture=None,captures=None,decisions=None,
        latencies_ms=[],online_overruns=None,public_replay_exact=False,
        failure_public_step=public['failed_public_step'],remaining_at_takeover=case['history'][-1]['remaining'],
        public_evidence=str(evidence),public_evidence_sha256=r.sha(evidence),
        estimator_arrays=public['arrays'],estimator_arrays_sha256=r.sha(public['arrays']),
        frozen_models_sha256=r.sha(r.OUT/'models_frozen.json'),
        scope='Invalid evaluation; outcomes unknown, not a normal loss or a synthetic terminal reward')
    path=Path(task['_output'])/'evaluation/engineering_failures'/(task['key']+'.json')
    r.save(path,row);return row


def evaluation_job(task):
    r.configure(task['_output'])
    _,cache,_=r.base.paths(task);marker=cache.with_name(cache.stem+'.failure.json')
    if marker.exists():return invalid_result(task,r.read(marker))
    try:return ORIGINAL_EPISODE_JOB(task)
    except RuntimeError as exc:
        if task['stage']!='evaluation' or not is_support_error(str(exc)):raise
        if not marker.exists():raise
        return invalid_result(task,r.read(marker))


def init_worker(root,smoke,queue,parent,training):
    r.OUT=Path(root);verify()
    acceleration.init_worker(root,smoke,queue,parent,training)


class EvaluationPool(acceleration.AcceleratedPool):
    def __init__(self,count):
        self.training=False;ctx=r.mp.get_context('spawn');self.queue=ctx.Queue()
        for cpu in r.CPUS[:count]:self.queue.put(cpu)
        self.executor=futures.ProcessPoolExecutor(max_workers=count,mp_context=ctx,initializer=init_worker,
            initargs=(str(r.OUT),False,self.queue,os.getpid(),False))
    def submit(self,task):return self.executor.submit(evaluation_job,task)


def emit_progress(rows,total,workers):
    invalid=sum(row.get('valid_episode',True) is False for row in rows)
    r.emit('learning_budget_test',completed=len(rows),valid_completed=len(rows)-invalid,
        engineering_failures=invalid,total=total,workers=workers,continuation='explicit invalid outcomes')


def run_evaluations(tasks,workers):
    results=[];todo=collections.deque()
    for index,task in enumerate(tasks,1):
        r.configure(task['_output']);cached=r.base.cached(task)
        if cached is not None:results.append(cached)
        else:todo.append(task)
        if index%256==0:
            r.emit('evaluation_cache_verification',checked=index,total=len(tasks),reused_valid=len(results),pending=len(todo))
    r.configure(r.OUT);emit_progress(results,len(tasks),workers)
    if not todo:return results
    pool=EvaluationPool(workers);pending={};failed=True;last_report=0.
    try:
        while todo or pending:
            while todo and len(pending)<2*workers:
                task=todo.popleft();pending[pool.submit(task)]=task
            done,_=futures.wait(pending,timeout=30,return_when=futures.FIRST_COMPLETED)
            for future in done:
                task=pending.pop(future)
                try:results.append(future.result())
                except BaseException:r.save(r.OUT/'failed_task.json',task);raise
            emit_progress(results,len(tasks),workers)
            if time.monotonic()-last_report>30:
                report(results,len(tasks));last_report=time.monotonic()
        failed=False
    finally:pool.close(failed)
    return sorted(results,key=identity)


def learning_pairs():
    return [(o,m,d,'x1',None,None) for o in r.base.OPPONENTS for m in ('JOINT','PRIOR') for d in ('x5','x20')]


def primary_inference(rows,tasks):
    required={identity(t) for t in tasks if t['method'].startswith(('JOINT','PRIOR'))}
    observed={identity(x) for x in rows if x.get('valid_episode',True) is not False}
    missing=required-observed
    if missing:return dict(status='incomplete_due_to_engineering_failures',comparisons=[],missing_primary_cells=[list(x) for x in sorted(missing)],
        scope='All eight planned simultaneous primary comparisons withheld; no failed history dropped')
    result=r.compare([x for x in rows if x.get('valid_episode',True) is not False],learning_pairs(),
        'Original 36 histories and all primary cells complete; eight planned simultaneous comparisons')
    return result|dict(status='complete',missing_primary_cells=[])


def statistics(rows,tasks,search):
    failed=[x for x in rows if x.get('valid_episode',True) is False]
    valid=[x for x in rows if x.get('valid_episode',True) is not False]
    cells=collections.defaultdict(list)
    for x in rows:cells[(x['_output'],x['opponent'],x['method'])].append(x)
    arms=[]
    for (output,opponent,method),group in sorted(cells.items()):
        ok=[x for x in group if x.get('valid_episode',True) is not False];bad=[x for x in group if x.get('valid_episode',True) is False]
        wins=sum(x['double_capture'] for x in ok);n=len(group);lat=[v for x in ok for v in x['latencies_ms']]
        # Bounds describe unidentified outcomes; no scores are assigned to
        # failed episodes or injected into bootstrap or valid-only means.
        utility_lower=(sum(x['utility'] for x in ok)-sum(.0025*x['remaining_at_takeover'] for x in bad))/n
        utility_upper=(sum(x['utility'] for x in ok)+len(bad)*.9975)/n
        arms.append(dict(output=output,opponent=opponent,method=method,planned_games=n,valid_games=len(ok),engineering_failures=len(bad),
            capture_rate_bounds=[wins/n,(wins+len(bad))/n],utility_bounds=[utility_lower,utility_upper],
            valid_only_utility=float(np.mean([x['utility'] for x in ok])) if ok else None,
            valid_only_cost=float(np.mean([x['completion_cost'] for x in ok])) if ok else None,
            valid_only_p95_ms=float(np.percentile(lat,95)) if lat else None))
    sp=[(o,m,None,None,b,200) for o in r.base.OPPONENTS for m in ('JOINT','PRIOR','U') for b in (2000,10000)]
    result=dict(evaluation_cells=len(rows),valid_episodes=len(valid),engineering_failures=len(failed),
        learning=primary_inference(rows,tasks),search=r.compare(search,sp,'Original complete development search-budget comparisons'),
        failures=failed,arms=arms,invalid_outcomes_imputed=False,failed_histories_silently_dropped=False)
    r.save(r.OUT/'statistics_with_engineering_failures.json',result);return result


def report(rows,total,complete=False,stats=None):
    invalid=[x for x in rows if x.get('valid_episode',True) is False]
    lines=['# E45-C：训练完成，闭环评测与信念支持失败审计','',
        '三档各12轮训练全部完成，27个最终模型保持冻结；27648局采集与528局搜索预算闭环已完成。','',
        f'独立评测已登记 **{len(rows)}/{total}** 个任务：{len(rows)-len(invalid)}局正常完成、{len(invalid)}局显式信念支持失败。',
        '失败局的捕获、效用、完成成本均为null，不当作普通输局，不补虚构终值。','',
        '[实时进度](status.json) · [原冻结协议](PROTOCOL.md) · [本次精确诊断及执行修复](diagnosis_grid_exhaustion_20261028/REPORT.md) · [审计统计](statistics_with_engineering_failures.json)','',
        '第54步故障已用完整公开历史精确复现：0.5米格上25个一步可达点全被团队观测排除，但格点之间仍有合法隐藏位置。这是有限网格覆盖缺陷；本次继续评测不修改D、模型、运动学、预算或控制策略。','']
    if complete:
        lines+=['全部预定评测任务已尝试；原严格协议包含显式失败，不能宣称全部9072局均正常完成。',
            '八项预定学习主比较：'+('原始主比较单元全部有效，可按原协议进行同时区间推断。' if stats['learning']['status']=='complete' else '存在失败单元，整组主比较暂不作完整统计判定。'),'']
    lines+=['|档位|完成训练轮数|最终模型|','|---|---:|---:|','|x1|12/12|9|','|x5|12/12|9|','|x20|12/12|9|']
    (r.OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    torch=r.torch;torch.set_num_threads(1)
    lock=(r.OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify();r.verify();acceleration.verify_execution(r.OUT);r.verify_models(r.OUT/'models_frozen.json')
    r.CPUS=r.read(r.OUT/'execution.json')['physical_cores'];r.WORKERS=40;r.SMOKE=False
    remaining=int(r.WALL_CAP-(time.time()-r.read(r.OUT/'clock.json')['started_unix']))
    if remaining<=0:raise RuntimeError('Original wall budget exhausted')
    def stop(signum,*_):raise RuntimeError('Evaluation continuation interrupted by signal '+str(signum))
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(remaining)
    try:
        r.emit('evaluation_resume',note='Verifying preserved results; known support failures remain explicit invalid outcomes')
        tasks=r.evaluation_tasks();assert len(tasks)==9072
        search=r.read(r.OUT/'search/results.json');assert len(search)==528
        workers=r.read(r.OUT/'timing/decision.json')['workers'];assert workers==32
        rows=run_evaluations(tasks,workers);assert len(rows)==len(tasks) and {identity(x) for x in rows}=={identity(t) for t in tasks}
        r.save(r.OUT/'evaluation/results_with_engineering_failures.json',rows)
        stats=statistics(rows,tasks,search);verify();r.verify_models(r.OUT/'models_frozen.json')
        r.save(r.OUT/'evaluation_continuation_decision.json',dict(status='completed_with_engineering_failures' if stats['engineering_failures'] else 'complete',
            attempted=len(rows),valid_episodes=stats['valid_episodes'],engineering_failures=stats['engineering_failures'],primary_inference=stats['learning']['status'],
            scientific_controller_changed=False,trained_models_changed=False))
        report(rows,len(tasks),True,stats);r.emit('complete_with_engineering_failures' if stats['engineering_failures'] else 'complete',
            attempted=len(rows),valid_episodes=stats['valid_episodes'],engineering_failures=stats['engineering_failures'],primary_inference=stats['learning']['status'])
    except BaseException as exc:
        r.save(r.OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),previous_status=r.read(r.OUT/'status.json')))
        r.emit('execution_failure',error=repr(exc));raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
