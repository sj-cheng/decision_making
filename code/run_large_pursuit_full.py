"""Continue calibrated pilots into frozen alternating training and paired evaluation."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['TZ']='Asia/Shanghai'
import argparse
import collections
import fcntl
import hashlib
import json
import math
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from run_large_pursuit import (ROOT,DEFAULT,read,write,digest,event,resources,cpu_batch,gpu_batch,train_job,build_bank,protocol)
from large_pursuit.methods import BRANCHES,METHODS,DIRECT_METHODS,METHOD_BRANCH
from large_pursuit.protocol import POOL_METHODS,episode_count
from large_pursuit.acceptance import require_repair,require_pilot,acceptance_sources
from large_pursuit.semantics import SCHEMA
SEEDS=(17,29,43)
DIRECT=DIRECT_METHODS


def checkpoint_specs(results,n,all_methods=True):
    by_seed=collections.defaultdict(dict)
    for result in results:
        if result['n']==n:by_seed[result['seed']][result['branch']]=result['checkpoint']
    specs=[]
    for method in POOL_METHODS if all_methods else ('IG','AG','GEOM'):
        branch=METHOD_BRANCH.get(method)
        if branch:
            for seed,models in sorted(by_seed.items()):
                if branch in models:specs.append(dict(method=method,seed=seed,checkpoint=models[branch]))
        else:specs.append(dict(method=method,seed=17))
    return specs


def frozen_pool(out,name,specs):
    protected=[]
    for spec in specs:
        item=dict(spec)
        if item.get('checkpoint'):
            item['sha256']=hashlib.sha256(Path(item['checkpoint']).read_bytes()).hexdigest()
        protected.append(item)
    path=out/'pools'/f'{name}.json'
    value=dict(name=name,entries=protected,sha256=digest(protected),sampling='uniform method, then uniform seed and historical snapshot within method')
    if path.exists() and read(path)!=value:raise RuntimeError('Attempted to change a frozen common opponent pool')
    write(path,value)
    return protected


def verify_pool(specs):
    for item in specs:
        if item.get('checkpoint') and hashlib.sha256(Path(item['checkpoint']).read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError('Frozen opponent changed')


def bulk_audit(out,layout,manifest):
    path=out/'bulk_concurrency.json'
    if path.exists():return read(path)
    options=[c for c in (16,32,64,84) if c<=len(layout['bulk'])+len(layout['bulk_smt'])]
    cpus=layout['bulk']+layout['bulk_smt'];measurements=[]
    selected_entries=[e for e in manifest['entries'] if e['split']=='engineering'][:3]+[e for e in manifest['entries'] if e['split']=='engineering' and e['n']==16][:3]
    for concurrency in options:
        jobs=[]
        for index in range(128):
            entry=selected_entries[index%len(selected_entries)]
            jobs.append(dict(kind='calibration',case=str(out/'bank'/entry['path']),method='GEOM',seed=9017+index,
                simulations=16,steps=8,folder=str(out/'bulk_calibration'/f'w{concurrency}_{index}')))
        started=time.perf_counter();results=cpu_batch(out,jobs,cpus,concurrency,f'bulk_calibration_w{concurrency}')
        elapsed=time.perf_counter()-started
        service_times=[r['elapsed_ms']/1000+r['teacher_query_ms']/1000 for r in results]
        measurements.append(dict(workers=concurrency,seconds=elapsed,steady_team_steps_per_second=concurrency*np.mean([r['environment_team_steps'] for r in results])/max(.001,np.mean(service_times)),team_steps=sum(r['environment_team_steps'] for r in results),
            steps_per_second=sum(r['environment_team_steps'] for r in results)/max(.001,elapsed)))
    chosen=max(measurements,key=lambda r:r['steady_team_steps_per_second'])['workers']
    result=dict(workers=chosen,cpus=cpus[:chosen],measurements=measurements,selection='maximum measured steady-state service throughput; report startup-inclusive throughput separately',
        reserved_logical_cpus=layout['reserved'],timing_cpus_excluded=layout['timing']+layout['timing_siblings_idle'])
    write(path,result);return result


def timing_audit(out,layout,manifest):
    path=out/'timing_calibration.json'
    if path.exists():return read(path)
    results={}
    # All bulk and GPU jobs from the preceding phase have completed here.
    for concurrency in (1,2,4):
        jobs=[]
        for n in (5,16):
            selected=[e for e in manifest['entries'] if e['n']==n and e['split']=='engineering'][:3]
            for entry in selected:
                jobs.append(dict(kind='calibration',case=str(out/'bank'/entry['path']),method='GEOM',seed=9003,
                    simulations=64,steps=12,folder=str(out/'timing_calibration'/f'w{concurrency}_{entry["id"]}')))
        results[str(concurrency)]=cpu_batch(out,jobs,layout['timing'],concurrency,f'isolated_timing_w{concurrency}')
    reference={n:np.percentile([v for r in results['1'] if r['n']==n for v in r['latencies_ms']],95) for n in (5,16)}
    ratios={str(c):{str(n):float(np.percentile([v for r in results[str(c)] if r['n']==n for v in r['latencies_ms']],95)/reference[n]) for n in (5,16)} for c in (1,2,4)}
    selected=max(c for c in (1,2,4) if max(ratios[str(c)].values())<=1.30)
    result=dict(workers=selected,ratios=ratios,criterion='p95 GEOM64 latency <=1.30x isolated single-worker baseline in both scales',
                bulk_paused=True,training_paused=True,physical_cpus=layout['timing'],siblings_idle=layout['timing_siblings_idle'],measurements=results)
    write(path,result);return result


def pilot_alternation(out,layout,gpu_jobs,initial):
    # The independent pilot namespaces and seeds never become formal models.
    enemies=[]
    for n in (5,16):
        specs=frozen_pool(out,f'pilot_pursuers_{n}',checkpoint_specs(initial,n))
        job=train_job(out,'pilot',n,'evader',107,role='evader',round_index=1,team_steps=8192,
            supervised_updates=0,opponents=specs,opponent_simulations=2,validation_every=2048)
        enemies.append(job)
    evaders=gpu_batch(out,enemies,layout,min(2,gpu_jobs),'pilot_common_evaders')
    jobs=[]
    for n in (5,16):
        pool=frozen_pool(out,f'pilot_common_evaders_{n}',[dict(method='learned_evader',checkpoint=r['checkpoint'],seed=r['seed']) for r in evaders if r['n']==n])
        for result in initial:
            if result['n']!=n:continue
            jobs.append(train_job(out,'pilot',n,result['branch'],101,round_index=1,
                parent=result['checkpoint'],opponents=pool,dataset=str(out/'pilot_data'/str(n)/'data.pt'),
                team_steps=8192,validation_every=2048,supervised_updates=0))
    pursuers=gpu_batch(out,jobs,layout,gpu_jobs,'pilot_common_pool_pursuers')
    return evaders,pursuers


def pilot_acceptance(out,initial,evaders,pursuers):
    require_repair(out);diagnostics=[]
    for result in initial+evaders+pursuers:
        folder=Path(result['latest']).parent
        rows=[json.loads(line) for line in (folder/'curve.jsonl').read_text().splitlines()]
        if not rows or not all(np.isfinite(r['loss']) and np.isfinite(r['kl']) for r in rows):raise RuntimeError('Nonfinite pilot curve')
        checkpoints=[r for r in rows if r['validation'] is not None]
        if len(checkpoints)<3:raise RuntimeError('Pilot has fewer than three validation checkpoints')
        scores=[r['validation'] for r in checkpoints]
        diagnostics.append(dict(n=result['n'],branch=result['branch'],role=result['role'],seed=result['seed'],team_steps=result['team_steps'],
            validation_scores=scores,flat=max(scores)==min(scores),improvement=max(scores)-scores[0],
            throughput=float(np.median([r['steps_per_second'] for r in rows[-4:]])),gpu_mb=max(r['peak_gpu_mb'] for r in rows),
            interpretation='training adequacy remains uncertain if flat; no test-dependent budget extension'))
    data=[read(p) for p in (out/'pilot_data').glob('*/result.json')]
    if len(data)!=2 or any(r['queries']!=32 for r in data):raise RuntimeError('Pilot common teacher bank incomplete')
    audit=dict(status='fresh_pilot_reviewed',schema=SCHEMA,sources=acceptance_sources(),diagnostics=diagnostics,teacher_banks=data,
        common_frozen_pool=True,independent_pilot_maps=True,sealed_used=False,
        budget_rule='32x pilot pursuer steps per round, 16x pilot evader steps per round; four alternations; independent evaders 64x pilot steps; equal across branches',
        limitation='Engineering and bounded training diagnostics, not evidence that a method or opponent is optimal. Formal undertraining remains reportable.')
    write(out/'pilot_audit.json',audit);return audit


def require_audit_clear(out):
    return require_repair(out)


def freeze_budget(out,diagnostics,timing):
    require_audit_clear(out)
    require_pilot(out)
    path=out/'budget.frozen.json'
    if path.exists():return read(path)
    # Throughput is measured; interaction counts are bounded and equal across all
    # branches and seeds. No branch receives a performance-triggered extension.
    p_steps={r['team_steps'] for r in diagnostics['diagnostics'] if r['role']=='pursuer'}
    e_steps={r['team_steps'] for r in diagnostics['diagnostics'] if r['role']=='evader'}
    if p_steps!={8192} or e_steps!={8192}:raise RuntimeError('Unexpected pilot budget')
    p_round=32*p_steps.pop();e_round=16*e_steps.pop();rounds=4;rollout=256;epochs=4;minibatch=64
    formal=dict(version=1,scales=[5,16],pursuer_seeds=list(SEEDS),evader_pool_seeds=list(SEEDS),independent_evader_seeds=[61,73,89],
        rounds=rounds,pursuer_team_steps_per_round=p_round,pursuer_team_steps_per_branch_seed=rounds*p_round,
        shared_evader_team_steps_per_round=e_round,independent_evader_team_steps=524288,
        initial_teacher_queries_per_scale_seed=1024,teacher_pool_candidates=32,teacher_particles=4,teacher_depth=32,
        teacher_simulations_per_query=128,teacher_policy_ties='existing geometry score only, no new reward and no critic supervision',online_teacher_stride=256,online_teacher_queries_per_branch_seed=rounds*p_round//256,
        initial_supervised_updates=2048,online_auxiliary_updates_per_rollout=1,
        rollout_team_steps=rollout,ppo_epochs_max=epochs,minibatch=minibatch,
        ppo_update_limit_per_pursuer_round=p_round//rollout*epochs*(rollout//minibatch),
        auxiliary_update_limit_per_pursuer_round=p_round//rollout,
        gradient_update_limit_per_pursuer_branch_seed=2048+rounds*(p_round//rollout)*(epochs*(rollout//minibatch)+1),
        pursuer_update='search policy iteration: imitate executed search actions plus actual-return critic regression',evader_update='shared-parameter team PPO',ppo_early_stop='evaders only: mean joint KL > .05*n; actual update count retained',
        selection='all56 validation maps x two initials, balanced fixed opponent assignment; worst-opponent mean, then overall mean, earliest tie; every32768 steps; independent evaders fixed final checkpoint',
        heldout_selection='independent evaders 61/73/89 are created only after final pursuers are frozen',
        common_pools='opponents frozen at stage entry; all pursuer branches share all previous common evader snapshots of the same scale; evaders sample each pursuer method equally',
        total_pursuer_online_team_steps=2*3*len(BRANCHES)*rounds*p_round,
        total_common_evader_team_steps=2*3*rounds*e_round,total_independent_evader_team_steps=2*3*524288,
        teacher_queries_unique_initial=2*3*1024,
        teacher_queries_online=2*3*len(BRANCHES)*rounds*p_round//256,
        timing_workers=timing['workers'],bulk_concurrency=str(out/'bulk_concurrency.json'),pilot_budget_multiplier=128,estimated_steps_per_second=[dict(n=r['n'],branch=r['branch'],throughput=r['throughput']) for r in diagnostics['diagnostics']],test_episodes=episode_count(),
        test_budgets=['sim16','sim64','wall200','wall1000'],rootpuct_simulations=3000,rootpuct_depth=100,pursuer_search_depth=8,existing_potential_coefficient=.05,complete_simulation='expand to fresh leaf or terminal, maximum8 transitions; all leaf work included',collection_simulations=16,validation_simulations=64,validation_scenarios_per_checkpoint=112,independent_evader_selection='fixed final, training maps only',
        no_test_triggered_extension=True,no_days_deadline=True,pilot_audit_sha256=digest(diagnostics),timing_audit_sha256=digest(timing),
        storage_free_at_freeze_gb=shutil.disk_usage(out).free/2**30,
        limitation='This is one bounded formal run. Flat or poor formal learning remains an outcome and does not trigger extra training.')
    sources=[ROOT/'code/verify_large_pursuit.py',ROOT/'code/run_large_pursuit.py',ROOT/'code/run_large_pursuit_full.py',ROOT/'code/tests/test_large_pursuit.py',ROOT/'code/tests/test_large_pursuit_semantics.py',ROOT/'code/verify_large_pursuit_repair.py']+list((ROOT/'code/large_pursuit').glob('*.py'))
    formal['sources']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for source in sources:
        target=out/'frozen_source'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    write(path,formal);document=read(out/'protocol.draft.json');document['formal_budget_sha256']=digest(formal);document['formal_budget']=str(path)
    write(out/'protocol.frozen.json',document)
    text=f"""# 大规模持续追击正式预算（已冻结）

正式两规模分别训练，追击种子17/29/43；四轮交替，追击每轮{p_round}团队步、共同逃跑每轮{e_round}团队步。独立61/73/89逃跑者各524288步，在追击池冻结后，仅用训练图并固定最后检查点。

初始共同教师每规模每种子1024查询；每查询32候选×4公共后验根，最多32步，128次完整教师模拟。在线各分支每256团队步一次教师查询；每轮1024次。真实在线搜索收集每步16次完整模拟，最大深度8；这些成本另记。短教师回报只训练策略/排序，不训练长期价值。

追击更新为搜索策略迭代，逃跑为团队PPO；共享捕获比例减0.001步成本、折扣1及原有ETA=.05势差塑形约定；势函数按原始人数归一化，终止势与后续价值均为0，训练与树内叶值使用同一单位。每256步最多4轮×4小批，再1次辅助监督；初始2048监督更新。每追击分支每种子更新上限71680，实际更新与教师/搜索成本分别保存。

验证使用全部56图及各2初态，对共同池作固定、方法等权且循环训练seed/历史快照的112场安排；最差对手组均值优先，再整体均值，同分最早。每32768步验证，最终用最后交替轮的验证检查点。正式测试使用完整对手矩阵，不以对手平均掩盖退化。

固定公开方法和对应同检查点直控，测试共{episode_count()}局；模拟16/64、墙钟200/1000ms，RootPUCT3000次完整模拟及100个交替ply。包括额外监督采集步数、教师模拟上界、源哈希、吞吐、显存和选模安排的机器清单见budget.frozen.json。测试结果不触发追加训练。
"""
    (out/'预算清单.md').write_text(text);event(out,'formal_budget_frozen',path=str(path),pursuer_steps=formal['total_pursuer_online_team_steps'])
    return formal


def verify_sources(budget):
    for path,expected in budget['sources'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise RuntimeError('Source changed after formal freeze: '+path)


def formal_training(out,layout,gpu_jobs,workers,budget):
    require_audit_clear(out)
    verify_sources(budget)
    datasets=teacher_bank(out,'formal_data',layout,workers,SEEDS,'train',budget['initial_teacher_queries_per_scale_seed'],4,32)
    history={5:[],16:[]};previous={};final=[]
    for r in range(budget['rounds']):
        verify_sources(budget);jobs=[]
        for n in (5,16):
            pool=frozen_pool(out,f'common_evaders_{n}_before_p{r}',history[n])
            verify_pool(pool)
            for seed in SEEDS:
                for branch in BRANCHES:
                    jobs.append(train_job(out,'formal',n,branch,seed,round_index=r,
                        team_steps=budget['pursuer_team_steps_per_round'],opponents=pool,dataset=datasets[n,seed],
                        parent=previous.get((n,seed,branch)),supervised_updates=2048 if r==0 else 0))
        final=gpu_batch(out,jobs,layout,gpu_jobs,f'formal_pursuers_r{r}')
        previous={(x['n'],x['seed'],x['branch']):x['checkpoint'] for x in final}
        jobs=[]
        for n in (5,16):
            specs=frozen_pool(out,f'common_pursuers_{n}_r{r}',checkpoint_specs(final,n))
            for seed in SEEDS:
                parent=next((e['checkpoint'] for e in reversed(history[n]) if e['seed']==seed),None)
                jobs.append(train_job(out,'formal',n,'evader',seed,role='evader',round_index=r,
                    team_steps=budget['shared_evader_team_steps_per_round'],opponents=specs,parent=parent,supervised_updates=0))
        enemies=gpu_batch(out,jobs,layout,min(gpu_jobs,6),f'formal_common_evaders_r{r}')
        for result in enemies:history[result['n']].append(dict(method='learned_evader',seed=result['seed'],round=r,checkpoint=result['checkpoint']))
        write(out/f'round_{r}_complete.json',dict(pursuers=final,evaders=enemies))
    specs_by_n={n:frozen_pool(out,f'final_pursuers_{n}',checkpoint_specs(final,n)) for n in (5,16)}
    write(out/'pursuers.frozen.json',dict(pools=specs_by_n,models=final,independent_evaders_started=False))
    jobs=[]
    for n in (5,16):
        for seed in (61,73,89):
            jobs.append(train_job(out,'independent',n,'evader',seed,role='evader',team_steps=budget['independent_evader_team_steps'],
                opponents=specs_by_n[n],supervised_updates=0))
    enemies=gpu_batch(out,jobs,layout,min(gpu_jobs,6),'independent_evaders')
    write(out/'models.frozen.json',dict(pursuers=final,independent_evaders=enemies,
        pursuer_pools=specs_by_n,evader_pools={n:frozen_pool(out,f'independent_evaders_{n}',[dict(method='learned_evader',seed=r['seed'],checkpoint=r['checkpoint']) for r in enemies if r['n']==n]) for n in (5,16)}))
    return final,enemies


def formal_evaluation(out,layout,workers,budget,timing,pursuers,evaders,manifest):
    require_audit_clear(out)
    verify_sources(budget)
    frozen=read(out/'models.frozen.json')
    frozen_hashes={}
    for section in ('pursuer_pools','evader_pools'):
        for specs in frozen[section].values():
            verify_pool(specs)
            frozen_hashes.update({s['checkpoint']:s['sha256'] for s in specs if s.get('checkpoint')})
    # Seal is opened once, after model and budget freeze. Hash verification does
    # not expose outcome information and is repeated before dispatch.
    for entry in manifest['entries']:
        path=out/'bank'/entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('Map bank changed')
    write(out/'test_seal_opened.json',dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),models_sha256=digest(read(out/'models.frozen.json')),
        budget_sha256=digest(budget),checkpoint_selection_closed=True))
    result_paths=[]
    for mode,amount in (('direct',0),('sim',16),('sim',64),('wall',200),('wall',1000)):
        # Timing evaluations have no concurrent bulk collection or GPU training.
        cpus=layout['timing'] if mode=='wall' else layout['bulk']
        concurrency=timing['workers'] if mode=='wall' else workers
        for n in (5,16):
            models={(r['seed'],r['branch']):r['checkpoint'] for r in pursuers if r['n']==n}
            opponents=[dict(name='D_kernel'),dict(name='RootPUCT')]+[dict(name=f'learned_{r["seed"]}',checkpoint=r['checkpoint']) for r in evaders if r['n']==n]
            for opponent in opponents:
                jobs=[]
                for entry in manifest['entries']:
                    if entry['n']!=n or entry['split'] not in ('sealed_ID','sealed_OOD'):continue
                    for initial in (0,1):
                        environment_seed=int(digest([entry['id'],initial,opponent['name']])[:8],16)
                        for seed in SEEDS:
                            for method in METHODS:
                                if (mode=='direct')!=(method in DIRECT):continue
                                branch=METHOD_BRANCH.get(method)
                                label='direct' if mode=='direct' else f'{mode}{amount}'
                                identifier=f'{entry["id"]}_i{initial}_{method}_s{seed}_{opponent["name"]}_{label}'
                                folder=out/'evaluation'/f'{n}v{n}'/label/opponent['name']/identifier
                                job=dict(kind='evaluate',case=str(out/'bank'/entry['path']),initial=initial,method=method,seed=seed,
                                    checkpoint=models.get((seed,branch)),checkpoint_sha256=frozen_hashes.get(models.get((seed,branch))),opponent=opponent['name'],opponent_checkpoint=opponent.get('checkpoint'),opponent_checkpoint_sha256=frozen_hashes.get(opponent.get('checkpoint')),
                                    budget=label,simulations=amount if mode=='sim' else 64,budget_ms=amount if mode=='wall' else None,
                                    rootpuct_simulations=3000,environment_seed=environment_seed,folder=str(folder))
                                jobs.append(job);result_paths.append(str(folder/'result.json'))
                if shutil.disk_usage(out).free<10*2**30:raise RuntimeError('Less than 10GiB free; existing artifacts retained for storage recovery')
                cpu_batch(out,jobs,cpus,concurrency,f'evaluation_{n}_{mode}{amount}_{opponent["name"]}')
    if len(result_paths)!=episode_count():raise RuntimeError('Incomplete registered test matrix')
    write(out/'evaluation_index.json',dict(results=result_paths,episodes=len(result_paths)))
    return result_paths


def final_report(out,paths,models):
    from large_pursuit.evaluation import summarize
    raw=[read(path) for path in paths];rows=[]
    for row in raw:
        if row['budget']=='direct':
            for budget in ('sim16','sim64','wall200','wall1000'):rows.append(dict(row,budget=budget,direct_reused=True))
        else:rows.append(row)
    summarize(out,rows)
    table={(r['n'],r['map_id'],r['initial'],r['seed'],r['opponent'],r['budget'],r['method']):r for r in rows}
    failures=[]
    for r in rows:
        if r['method']!='FUSION':continue
        other=table.get((r['n'],r['map_id'],r['initial'],r['seed'],r['opponent'],r['budget'],'AG'))
        if other and r['capture_fraction']<other['capture_fraction']:failures.append(dict(n=r['n'],opponent=r['opponent'],gap=r['capture_fraction']-other['capture_fraction'],fusion=r['trajectory'],AG=other['trajectory']))
    failures.sort(key=lambda r:r['gap']);write(out/'failure_cases.json',failures[:100])
    write(out/'delivery.json',dict(status='complete',models=models,episodes=len(raw),summary=str(out/'summary.json'),
        failure_cases=str(out/'failure_cases.json'),protocol=str(out/'protocol.frozen.json'),budget=str(out/'budget.frozen.json'),
        test_feedback_used=False,conclusion='Report geometric superiority unchanged if observed; no retraining after tests'))
    (out/'结果说明.md').write_text(f'# 大规模持续追击实验结果\n\n已完成{len(raw)}局配对测试。逐局结果和完整轨迹见evaluation_index.json，按规模、ID/OOD、训练种子、对手和预算分别汇总于summary.json。主要比较采用按地图成组配对bootstrap，同时区间覆盖每个报告分层的预注册比较。失败完成成本按满时域计。\n\n模型和冻结记录见models.frozen.json、protocol.frozen.json、budget.frozen.json，训练曲线保存在各模型目录curve.jsonl。failure_cases.json完整保留融合输给AG的代表性案例；测试结果没有用于追加训练。\n')



def teacher_bank(out,stage,layout,workers,seeds,split,queries,particles,depth):
    """Separate public rollout collection from independent labels; parallelize labels."""
    import torch
    from large_pursuit.training import save
    from large_pursuit.provenance import bind_job,sha
    collections_jobs=[];metadata=[]
    shard_queries=4 if split=='pilot_train' else 16
    for n in (5,16):
        for seed in seeds:
            for shard,start in enumerate(range(0,queries,shard_queries)):
                count=min(shard_queries,queries-start)
                folder=out/(stage+'_packets')/f'{n}_{seed}_{shard}'
                job=dict(kind='teacher_collect',bank=str(out/'bank'),n=n,seed=seed,shard=shard,split=split,queries=count,stride=8,folder=str(folder))
                collections_jobs.append(job);metadata.append((n,seed,shard,count))
    collected=cpu_batch(out,collections_jobs,layout['bulk'],min(workers,len(collections_jobs)),stage+'_public_states')
    label_jobs=[];owners=[]
    for (n,seed,shard,count),result in zip(metadata,collected):
        for index in range(count):
            label_jobs.append(dict(kind='teacher_label',packets=result['path'],packet_index=index,particles=particles,depth=depth,
                folder=str(out/(stage+'_labels')/f'{n}_{seed}_{shard}_{index}')))
            owners.append((n,seed))
    labels=cpu_batch(out,label_jobs,layout['bulk'],workers,stage+'_independent_teacher')
    paths={}
    for n in (5,16):
        for seed in seeds:
            selected=[row for owner,row in zip(owners,labels) if owner==(n,seed)]
            name=str(n) if stage=='pilot_data' else f'{n}_{seed}'
            folder=out/stage/name
            specification=dict(kind='teacher_aggregate',n=n,seed=seed,queries=queries,
                parts=[dict(path=row['path'],sha256=sha(row['path'])) for row in selected])
            identity=bind_job(folder,specification)
            if not (folder/'result.json').exists():
                records=[torch.load(row['path'],map_location='cpu',weights_only=False)['record'] for row in selected]
                if len(records)!=queries:raise RuntimeError('Common teacher query count mismatch')
                steps=sum(row['supervised_collection_team_steps'] for meta,row in zip(metadata,collected) if meta[:2]==(n,seed))
                result=dict(schema=SCHEMA,identity=identity['digest'],path=str(folder/'data.pt'),queries=queries,
                    simulations=sum(row['simulations'] for row in selected),supervised_collection_team_steps=steps,
                    simulated_team_steps_upper_bound=queries*32*particles*depth,
                    informative_queries=sum(row['score_spread']>0 for row in selected),teacher_seconds=sum(row['seconds'] for row in selected))
                save(folder/'data.pt',dict(schema=SCHEMA,identity=identity['digest'],records=records,result=result));write(folder/'result.json',result)
            paths[n,seed]=str(folder/'data.pt')
    return paths


def reference_profiles(out,layout,workers,manifest):
    jobs=[]
    for entry in manifest['entries']:
        if entry['split']!='engineering':continue
        for simulations in (64,256,3000):
            jobs.append(dict(kind='reference_profile',case=str(out/'bank'/entry['path']),simulations=simulations,seed=9017,
                folder=str(out/'reference_profiles'/f'{entry["id"]}_{simulations}')))
    rows=cpu_batch(out,jobs,layout['bulk'],workers,'reference_cost_and_contract')
    write(out/'reference_audit.json',dict(rows=rows,
        scope='independent engineering maps only; legacy D distribution and two-team search contracts tested separately; no claim of global opponent optimality'))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--workers',type=int,default=84);parser.add_argument('--gpu-jobs',type=int,default=24)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    # No fresh directory, flag removal or old cached result bypasses this gate.
    require_repair(out)
    lock=open(out/'full_coordinator.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def stop_requested(*_):raise KeyboardInterrupt('Coordinator termination requested')
    signal.signal(signal.SIGTERM,stop_requested)
    layout=resources();workers=min(args.workers,len(layout['bulk'])+len(layout.get('bulk_smt',[])))
    layout['bulk']=layout['bulk']+layout.get('bulk_smt',[]);gpu_jobs=min(args.gpu_jobs,len(layout['training']))
    write(out/'full_coordinator.json',dict(pid=os.getpid(),start=time.strftime('%Y-%m-%d %H:%M:%S'),workers=workers,gpu_jobs=gpu_jobs,schema=SCHEMA))
    write(out/'resources.json',layout);protocol(out,layout)
    try:
        manifest=build_bank(out,layout,workers)
        calibration=[]
        jobs=[]
        for entry in manifest['entries']:
            if entry['split']!='engineering':continue
            for method in ('AG','GEOM'):
                for simulations in (16,64):
                    jobs.append(dict(kind='calibration',case=str(out/'bank'/entry['path']),method=method,seed=7001,simulations=simulations,steps=4,
                        folder=str(out/'calibration'/f'{entry["id"]}_{method}_{simulations}')))
        calibration=cpu_batch(out,jobs,layout['bulk'],workers,'engineering_calibration')
        write(out/'calibration.json',calibration);reference_profiles(out,layout,workers,manifest)
        data=teacher_bank(out,'pilot_data',layout,workers,(101,),'pilot_train',32,4,8)
        jobs=[train_job(out,'pilot',n,branch,101,dataset=data[n,101]) for n in (5,16) for branch in BRANCHES]
        initial=gpu_batch(out,jobs,layout,gpu_jobs,'fresh_pilot_pursuers')
        evaders,pursuers=pilot_alternation(out,layout,gpu_jobs,initial)
        audit=pilot_acceptance(out,initial,evaders,pursuers)
        timing=timing_audit(out,layout,manifest)
        throughput=bulk_audit(out,resources(),manifest)
        workers=min(args.workers,throughput['workers']);layout['bulk']=throughput['cpus'];write(out/'full_resources.json',dict(layout,bulk_workers=workers))
        budget=freeze_budget(out,audit,timing)
        models,independent=formal_training(out,layout,gpu_jobs,workers,budget)
        paths=formal_evaluation(out,layout,workers,budget,timing,models,independent,manifest)
        final_report(out,paths,dict(pursuers=models,independent_evaders=independent))
        write(out/'progress.json',dict(stage='complete',episodes=len(paths),running=False,pid=os.getpid()));event(out,'all_experiments_complete',episodes=len(paths))
    except BaseException:
        failure=dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),traceback=traceback.format_exc())
        write(out/'full_failure.json',failure)
        write(out/'progress.json',dict(stage='failed',pid=os.getpid(),running=False,error=failure['traceback'].splitlines()[-1],time=failure['time']))
        raise

if __name__=='__main__':main()
