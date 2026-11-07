"""E45-Q: stable-BC control, public DAgger, gated supervised HASAC."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import argparse
import ctypes
import fcntl
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import run_e45p_hasac_bc_anchor as p
from learning.e45m_author_hasac import AuthorHASAC
from learning.e45m_task_adapter import Replay
from learning.e45p_actor_supervision import public_teacher,initialize_bc_std,fit_metrics
from learning.e45q_corrective_imitation import pool_for,collect,cosine_lr,sample_labels,stable_bc_step
j=p.j;o=p.o
ROOT=p.ROOT
OUT=ROOT/'current/e45q_corrective_hasac_20261101'
ROUTES=['stable_bc','dagger']


def verify(root):
    root=Path(root);cfg=j.read(root/'experiment.json')
    for section in ('source_hashes','protected_hashes','upstream_hashes','bank_hashes'):
        for name,digest in cfg[section].items():
            if j.sha(name)!=digest:raise RuntimeError('Frozen Q artifact changed: '+name)
    assert j.sha(cfg['demonstration_file'])==cfg['demonstration_sha256']
    assert j.sha(cfg['teacher_manifest'])==cfg['teacher_manifest_sha256']
    return cfg


def register(root,smoke=False):
    root=Path(root).resolve()
    if (root/'experiment.json').exists():return verify(root)
    root.mkdir(parents=True,exist_ok=True)
    parent=p.verify(p.OUT);cfg=dict(parent)
    cfg.update(experiment='E45-Q',registered_at=time.strftime('%FT%T%z'),smoke=smoke,
        scope='stable BC versus public learner-state corrective imitation; conditional RL development only',
        algorithm='author HASAC networks/critic/sequential RL update; explicit corrective pretraining and auxiliary BC gradient in supervised arm',
        teacher='original public GEOM_SEARCH controls plus public queries on learner-visited states; never actuates',
        entries=parent['entries'][:1] if smoke else parent['entries'],routes=ROUTES,
        base_updates=2 if smoke else 4000,round_updates=2 if smoke else 2000,rounds=1 if smoke else 4,
        bc_updates=4 if smoke else 12000,bc_initial_std=.15,
        bc_optimizer='author Adam; base cosine1e-4->1e-5, each correction/control block3e-5->3e-6; gradient norm clipped1',
        correction_batch=256,original_fraction=.5,
        correction_policy='current deterministic learner executes every action; GEOM_SEARCH queries public visited states, never actuates',
        teacher_budget_ms=200.,teacher_fixed_simulations=2 if smoke else None,
        minimum_valid_teacher_fraction=.95,rounds_stable_required=2,
        smoke_bypasses_gate=bool(smoke),
        gate_scope='both seeds >=5/6 deterministic at final two correction/control round checkpoints',
        route_selection='first eligible common route in [stable_bc,dagger]; never choose route per seed',
        rl_actor_lr=1e-4,lr=1e-4,bc_actor_lr=1e-4,steps_per_model=1 if smoke else 20000,
        arms=['replay','replay_bc'],evaluation_every=5000,update_batch=32 if smoke else 256,
        batch_episodes=2 if smoke else 12,replay_capacity=256 if smoke else 50000,
        action_repeats=1 if smoke else 2,
        supervision='original labels plus all valid learner-state correction labels; separate from executed replay actions',
        selection='fixed budgets, final rounds gate, all checkpoints; no best-checkpoint rescue',
        budget='base4000+4x2000 BC updates/actor/route, 4x6 complete learner episodes/seed with <=1680 teacher queries/seed; optional RL20k online/arm/seed',
        held_out_test_used=False,validation_used=False,maximum_wall_seconds=12*3600)
    protected=dict(parent['protected_hashes']);protected.update(parent['source_hashes'])
    amendment=j.read(p.OUT/'precision_amendment.json');protected.update(amendment['source_hashes'])
    cfg['protected_hashes']=protected
    files=[Path(__file__).resolve(),ROOT/'code/learning/e45q_corrective_imitation.py',
        ROOT/'code/tests/test_e45q_corrective_imitation.py',ROOT/'code/E45Q_CORRECTIVE_HASAC_PROTOCOL_20261101.md',ROOT/'code/audit_e45q_smoke.py']
    cfg['source_hashes']={str(name):j.sha(name) for name in files}
    cfg['bank_hashes']={e['path']:j.sha(e['path']) for e in cfg['entries']}
    cfg['parent_P_experiment_sha256']=j.sha(p.OUT/'experiment.json')
    j.save(root/'experiment.json',cfg)
    for name in files:
        dst=root/'source_snapshot'/name.relative_to(ROOT/'code');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dst)
    return verify(root)


def labels_only(data):return {k:data[k] for k in ('obs','action','active')}
def join_labels(a,b):return {k:np.concatenate([a[k],b[k]]) for k in ('obs','action','active')}


def evaluate(pool,cfg,seed,model,tag,steps):
    result=dict(steps=steps,tag=tag)
    weights=o.actor_weights(model)
    for mode in ('deterministic','stochastic'):
        repeats=1 if mode=='deterministic' else cfg['action_repeats']
        jobs=[dict(actors=weights,training=False,task=o.task_for(e,w,env,seed,mode,
            o.l.seed_for('HASAC_eval',seed,e['id'],w,r),tag))
            for e,w,env in o.l.situations(cfg['entries']) for r in range(repeats)]
        result[mode]=o.l.score([r['summary'] for r in pool.map(o.rollout,jobs)])
    return result


def fit(model,original,corrections,steps,seed,block,folder):
    for step in range(steps):
        rng=np.random.default_rng(o.l.seed_for('Q_fit',seed,block,step))
        batch=sample_labels(original,corrections,rng)
        teacher=public_teacher(batch,model.device)
        high,low=(1e-4,1e-5) if block==0 else (3e-5,3e-6)
        lr=cosine_lr(step,steps,high,low);stable_bc_step(model,teacher,lr)
        if step%500==499 or step==steps-1:
            j.common.emit(folder,'supervised_fitting',block=block,steps=step+1,lr=lr)


def checkpoint(folder,model,history,cfg,steps):
    folder.mkdir(parents=True,exist_ok=True)
    o.checkpoint(folder,model,Replay(cfg['replay_capacity'],0),steps,len(history),history,j.sha(Path(cfg['root'])/'experiment.json'))


def imitation(args):
    root=Path(args.output);cfg=verify(root);cfg['root']=str(root)
    folder=root/'imitation'/f'seed_{args.seed}';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cores=list(map(int,args.cpus.split(',')));os.sched_setaffinity(0,{cores[-1]});torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    original=labels_only(p.load_data(cfg));original_teacher=public_teacher(original,args.device)
    models={};histories={};corrections=None;rows=[]
    pool,queue=pool_for(folder/'episodes',cores[:-1])
    try:
        base=folder/'base';base.mkdir(exist_ok=True)
        model=AuthorHASAC(obs_dim=845,device=args.device,seed=args.seed,retirement_mask=True)
        initialize_bc_std(model,.15)
        if not (base/'decision.json').exists():
            j.common.emit(folder,'base_fitting',steps=0)
            fit(model,original,None,cfg['base_updates'],args.seed,0,folder)
            row=evaluate(pool,cfg,args.seed,model,'Q_base',cfg['base_updates'])
            row['fit_original']=fit_metrics(model,original_teacher)
            checkpoint(base,model,[row],cfg,cfg['base_updates'])
            j.save(base/'decision.json',dict(status='complete',history=[row],checkpoint=j.read(base/'latest.json')))
        for route in ROUTES:
            target=folder/route;target.mkdir(exist_ok=True)
            own=AuthorHASAC(obs_dim=845,device=args.device,seed=args.seed,retirement_mask=True)
            source=target if (target/'latest.json').exists() else base
            _,_,history=o.restore(source,own,Replay(cfg['replay_capacity'],0),j.sha(root/'experiment.json'))
            models[route]=own
            histories[route]=j.read(target/'evaluation.json') if (target/'evaluation.json').exists() else [dict(history[-1],round=0)]
            if (target/'latest.json').exists():
                # Every route checkpoint has its complete round metadata saved
                # atomically before it is eligible to skip that round on resume.
                state=j.read(target/'round_state.json')
                assert histories[route]==history and history[-1]['round']==state['round']
                assert j.read(target/'latest.json')==state['checkpoint']
        correction_file=folder/'aggregate_corrections.npz'
        if correction_file.exists():
            with np.load(correction_file) as z:corrections={k:z[k].copy() for k in z.files}
        for round_index in range(1,cfg['rounds']+1):
            manifest=folder/f'round_{round_index}_collection.json'
            if manifest.exists():
                collection=j.read(manifest)
                for row in collection['episodes']:
                    assert j.sha(row['labels_path'])==row['labels_sha256']
            else:
                j.common.emit(folder,'collecting_corrections',steps=round_index)
                jobs=[]
                for index,(e,w,env) in enumerate(o.l.situations(cfg['entries'])):
                    task=o.task_for(e,w,env,args.seed,'deterministic',o.l.seed_for('Q_collect',args.seed,round_index,index),f'Q_correction_round{round_index}')
                    task.update(query_teacher=True,teacher_budget_ms=cfg['teacher_budget_ms'],teacher_fixed_simulations=cfg['teacher_fixed_simulations'])
                    jobs.append(dict(task=task,actors=o.actor_weights(models['dagger']),training=True))
                results=list(pool.map(collect,jobs));episode_rows=[r['summary'] for r in results]
                labels=results[0]['labels']
                for result in results[1:]:labels=join_labels(labels,result['labels'])
                label_path=folder/f'round_{round_index}_labels.npz';np.savez_compressed(label_path,**labels)
                collection=dict(episodes=episode_rows,labels=str(label_path),labels_sha256=j.sha(label_path),
                    queries=sum(r['teacher_queries'] for r in episode_rows),valid_labels=len(labels['obs']),
                    teacher_query_ms=sum(r['teacher_query_ms'] for r in episode_rows),teacher_actuates=False)
                j.save(manifest,collection)
            if collection['valid_labels']/collection['queries']<cfg['minimum_valid_teacher_fraction']:
                raise RuntimeError('Oracle valid-query fraction below registered quality gate; preserve collection')
            # Rebuild from complete immutable shards on every round/resume;
            # never append a round twice after a crash.
            corrections=None
            for index in range(1,round_index+1):
                packet=j.read(folder/f'round_{index}_collection.json')
                assert j.sha(packet['labels'])==packet['labels_sha256']
                with np.load(packet['labels']) as z:labels={k:z[k].copy() for k in z.files}
                corrections=labels if corrections is None else join_labels(corrections,labels)
            np.savez_compressed(correction_file,**corrections)
            rows.append(collection)
            for route in ROUTES:
                target=folder/route
                state=j.read(target/'round_state.json') if (target/'round_state.json').exists() else {'round':0}
                if state['round']>=round_index:continue
                model=models[route]
                fit(model,original,corrections if route=='dagger' else None,cfg['round_updates'],args.seed,round_index,folder)
                updates=cfg['base_updates']+round_index*cfg['round_updates']
                row=evaluate(pool,cfg,args.seed,model,f'Q_{route}_round{round_index}',updates)
                row.update(round=round_index,fit_original=fit_metrics(model,original_teacher),
                    fit_corrections=fit_metrics(model,public_teacher(corrections,model.device)))
                histories[route].append(row)
                j.save(target/'evaluation.json',histories[route])
                # Round state precedes checkpoint: if checkpoint is interrupted,
                # resume checks latest/history consistency and fails loudly.
                checkpoint(target,model,histories[route],cfg,updates)
                j.save(target/'round_state.json',dict(round=round_index,checkpoint=j.read(target/'latest.json')))
                j.common.emit(folder,'round_evaluated',steps=round_index,route=route,score=row['deterministic'])
        for route in ROUTES:
            labels=join_labels(original,corrections) if route=='dagger' else original
            path=folder/route/'supervision.npz';np.savez_compressed(path,**labels)
            j.save(folder/route/'decision.json',dict(status='complete',history=histories[route],
                checkpoint=j.read(folder/route/'latest.json'),supervision_path=str(path),supervision_sha256=j.sha(path),
                original_labels=len(original['obs']),correction_labels=len(corrections['obs']) if route=='dagger' else 0))
        verify(root);j.save(folder/'decision.json',dict(status='complete',routes=ROUTES,
            teacher_queries=sum(row['queries'] for row in rows),valid_labels=len(corrections['obs']),
            teacher_query_ms=sum(row['teacher_query_ms'] for row in rows)))
        j.common.emit(folder,'imitation_complete',steps=cfg['bc_updates'])
    except BaseException as exc:
        j.common.emit(folder,'failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:pool.shutdown(wait=True,cancel_futures=True);queue.close();lock.close()


def choose_route(root,cfg):
    eligible=[];details={}
    for route in ROUTES:
        passed=True;details[route]={}
        for seed in cfg['seeds']:
            decision=j.read(root/'imitation'/f'seed_{seed}'/route/'decision.json')
            late=[h for h in decision['history'] if h['round']>0][-cfg['rounds_stable_required']:]
            ok=len(late)==cfg['rounds_stable_required'] and all(h['deterministic']['episodes']==6 and h['deterministic']['double']>=5 for h in late)
            passed &= ok;details[route][str(seed)]=dict(passed=ok,last_scores=[h['deterministic'] for h in late])
        if passed:eligible.append(route)
    return (eligible[0] if eligible else None),details


def prepare_rl(root,cfg,route):
    for seed in cfg['seeds']:
        source=j.read(root/'imitation'/f'seed_{seed}'/route/'decision.json')
        folder=root/'bc'/f'HASAC_{seed}'
        j.save(folder/'decision.json',dict(status='complete',gate_passed=True,
            bc_updates=cfg['bc_updates'],history=source['history'],bc_only_checkpoint=source['checkpoint'],
            selected_route=route,supervision_path=source['supervision_path'],supervision_sha256=source['supervision_sha256']))


def rl(args):
    root=Path(args.output);cfg=verify(root)
    source=j.read(root/'bc'/f'HASAC_{args.seed}'/'decision.json')
    assert j.sha(source['supervision_path'])==source['supervision_sha256']
    with np.load(source['supervision_path']) as z:supervision={k:z[k].copy() for k in z.files}
    old_setup=p.setup
    def setup(arguments,folder):
        lock,model,pool,queue=old_setup(arguments,folder)
        for actor in model.actor:
            for group in actor.actor_optimizer.param_groups:group['lr']=cfg['rl_actor_lr']
        return lock,model,pool,queue
    p.verify=verify;p.setup=setup
    p.public_teacher=lambda data,device:public_teacher(supervision,device)
    p.o.pool_for=pool_for
    p.rl_train(args)


def report(root):
    root=Path(root);cfg=j.read(root/'experiment.json');status=j.read(root/'status.json') if (root/'status.json').exists() else {}
    lines=['E45-Q：稳定BC对照 / Actor状态纠错DAgger / 条件HASAC',time.strftime('%FT%T%z'),'状态：'+status.get('stage','starting'),
        '训练教师不执行动作；所有评估关闭教师；正式门槛：同一路线两种子末两轮均>=5/6。','',
        '路线/种子 | 轮数 | 确定性双捕 | 随机双捕']
    for route in ROUTES:
        for seed in cfg['seeds']:
            file=root/'imitation'/f'seed_{seed}'/route/'evaluation.json'
            if file.exists():
                row=j.read(file)[-1];a=row['deterministic'];b=row['stochastic']
                lines.append(f"{route}/{seed} | {row['round']} | {a['double']}/{a['episodes']} | {b['double']}/{b['episodes']}")
            else:lines.append(f'{route}/{seed} | queued | — | —')
    for seed in cfg['seeds']:
        file=root/'imitation'/f'seed_{seed}'/'status.json'
        if file.exists():lines.append(f"seed{seed}：{j.read(file).get('stage','—')}")
    if status.get('stage')=='rl_running':
        for arm in cfg['arms']:
            for seed in cfg['seeds']:
                file=root/'runs'/f'{arm}_{seed}'/'evaluation.json'
                if file.exists():
                    row=j.read(file)[-1];a=row['deterministic'];b=row['stochastic']
                    lines.append(f"{arm}/{seed} {row['steps']}步：{a['double']}/{a['episodes']}，{b['double']}/{b['episodes']}")
    temp=root/'LIVE.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(root/'LIVE.txt')


def run_group(root,cfg,args,command,started):
    combos=[('imitation',s) for s in cfg['seeds']] if command=='imitation' else [(a,s) for a in cfg['arms'] for s in cfg['seeds']]
    cores=j.choose_cpus(len(combos)*(args.workers+1));assert len(cores)==len(combos)*(args.workers+1)
    children=[];allocation=[]
    try:
        for i,(arm,seed) in enumerate(combos):
            folder=root/'imitation'/f'seed_{seed}' if command=='imitation' else root/'runs'/f'{arm}_{seed}'
            folder.mkdir(parents=True,exist_ok=True)
            if (folder/'decision.json').exists():continue
            use=cores[i*(args.workers+1):(i+1)*(args.workers+1)]
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),command,'--output',str(root),
                '--seed',str(seed),'--arm',arm,'--device',args.device,'--cpus',','.join(map(str,use)),'--parent',str(os.getpid())]
            with (folder/'run.log').open('a') as log:child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            children.append(child);allocation.append(dict(arm=arm,seed=seed,pid=child.pid,cores=use,device=args.device))
        j.save(root/f'execution_{command}.json',dict(allocation=allocation,workers=args.workers))
        while any(child.poll() is None for child in children):
            report(root)
            if any(child.poll() not in (None,0) for child in children):raise RuntimeError('Q child failed, inspect retained logs')
            if time.time()-started>cfg['maximum_wall_seconds']:raise RuntimeError('Registered wall cap reached')
            time.sleep(5)
        assert all(child.returncode==0 for child in children)
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()


def final_report(root,cfg,selected,details,rl_started):
    lines=['# E45-Q：稳定BC与纠错数据', '', '只用既有六个训练情形，不能作为独立泛化或最终强基线证明。', '',
        '|路线|种子|轮次|确定性双捕|随机双捕|原始教师动作MSE|', '|---|---|---:|---|---|---|']
    for route in ROUTES:
        for seed in cfg['seeds']:
            decision=j.read(root/'imitation'/f'seed_{seed}'/route/'decision.json')
            for row in decision['history']:
                a=row['deterministic'];b=row['stochastic']
                lines.append(f"|{route}|{seed}|{row['round']}|{a['double']}/{a['episodes']}|{b['double']}/{b['episodes']}|{row['fit_original']['action_mse']}|")
    lines+=['',f'统一选择路线：{selected}；正式RL启动：{rl_started}。',
        '两路线同初始稳定BC、同每轮监督更新预算；stable_bc只重复原163样本，dagger额外查询Actor真实访问状态，50%原始/50%纠错标签。',
        '纠错标签与实际执行动作分开保存，未用未执行教师动作伪造RL转移。教师公共信念一致性和模拟转移均在线检查。']
    if rl_started:
        lines+=['','|RL臂|种子|在线步|确定性双捕|随机双捕|','|---|---|---:|---|---|']
        for arm in cfg['arms']:
            for seed in cfg['seeds']:
                d=j.read(root/'runs'/f'{arm}_{seed}'/'decision.json');h=d['history'][-1];a=h['deterministic'];b=h['stochastic']
                lines.append(f"|{arm}|{seed}|{d['steps']}|{a['double']}/{a['episodes']}|{b['double']}/{b['episodes']}|")
    if cfg['smoke']:lines+=['','工程smoke显式绕过门槛，不作能力判断。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def pipeline(args):
    root=Path(args.output).resolve();cfg=register(root,args.smoke)
    lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time()
    try:
        if not cfg['smoke']:
            smoke=ROOT/'current/e45q_corrective_hasac_smoke_20261101';verify(smoke);assert j.read(smoke/'smoke_audit.json')['passed']
        j.common.emit(root,'imitation_running');run_group(root,cfg,args,'imitation',started)
        selected,details=choose_route(root,cfg);j.save(root/'route_gate.json',dict(selected=selected,details=details,smoke_bypass=cfg['smoke']))
        do_rl=selected is not None or cfg['smoke']
        if do_rl:
            prepare_rl(root,cfg,selected or 'dagger');j.common.emit(root,'rl_running');run_group(root,cfg,args,'rl',started)
        verify(root);final_report(root,cfg,selected,details,do_rl)
        status='complete' if do_rl else 'imitation_gate_failed'
        j.save(root/'decision.json',dict(status=status,selected_route=selected,rl_started=do_rl,smoke=cfg['smoke'],wall_seconds=time.time()-started))
        j.common.emit(root,status);report(root)
    except BaseException as exc:
        j.common.emit(root,'failed',error=repr(exc),traceback=traceback.format_exc());report(root);raise
    finally:lock.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['register','pipeline','imitation','rl','report'])
    parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--workers',type=int,default=12);parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--arm',default='replay_bc');parser.add_argument('--device',default='cuda:2');parser.add_argument('--cpus');parser.add_argument('--parent',type=int)
    args=parser.parse_args()
    if args.parent:
        ctypes.CDLL(None).prctl(1,signal.SIGTERM)
        if os.getppid()!=args.parent:raise RuntimeError('Q coordinator exited')
    if args.command=='register':register(args.output,args.smoke)
    elif args.command=='pipeline':pipeline(args)
    elif args.command=='imitation':imitation(args)
    elif args.command=='rl':rl(args)
    else:report(args.output)


if __name__=='__main__':main()
