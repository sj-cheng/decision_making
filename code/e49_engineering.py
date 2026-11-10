"""Frozen engineering fixtures and fail-closed E49 integration gates.

These cases are separate from scientific bank entries. Full integration uses
H40 and fixed2 pursuer simulations; it is not a nominal-budget scientific run.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import concurrent.futures as cf
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import numpy as np


def _runner_functions(text):
    names={'run_episode','observation','reset_history'}
    return {node.name:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
            for node in ast.parse(text).body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names}


def core_fingerprint():
    """Physics/controller/audit and runner bodies; report/CLI changes excluded."""
    root=Path(__file__).resolve().parents[1]
    modules={name:hashlib.sha256((root/'code'/name).read_bytes()).hexdigest()
             for name in ('e49_rules.py','e49_controllers.py','e49_audit.py')}
    native=root/'current/e49_persistent_pursuit_benchmark_20261108/cpp/source/problems/example8.hpp'
    return dict(modules=modules,runner_functions=_runner_functions((root/'code/run_e49_benchmark.py').read_text()),
                native_physics=hashlib.sha256(native.read_bytes()).hexdigest(),engineering_gate_version=2)


def prepare_engineering(out):
    from e49_bank import construct, geometry_key
    from e45f_scenarios import ID_FAMILIES,OOD_FAMILIES
    from e49_controllers import method_specs
    from run_e49_benchmark import write,read,sha
    out=Path(out);root=out/'engineering';manifest=root/'manifest.json'
    if manifest.exists():
        record=read(manifest)
        for entry in record['entries']:
            if sha(entry['case'])!=entry['sha256']:raise RuntimeError('Engineering fixture hash mismatch')
        return record
    entries=[];families=(*ID_FAMILIES,*OOD_FAMILIES)
    for tier in ('sparse','16','24'):
        for fi,family in enumerate(families):
            source_index=9000+fi
            case=construct(family,source_index,'development',tier,40)
            case['construction']['source_split']='development_namespace_separate_index9000+'
            case.update(id=f'e49_engineering_{tier}_{family}',split='engineering',horizon=40,
                        engineering_only=True,scientific_use=False)
            path=root/'cases'/tier/(family+'.json');write(path,case)
            entries.append(dict(tier=tier,family=family,case=str(path),sha256=sha(path),geometry_key=geometry_key(case['map'])))
    specs=method_specs();tasks=[]
    for tier in ('sparse','16','24'):
        tier_entries=[e for e in entries if e['tier']==tier]
        for i,spec in enumerate(specs):
            entry=tier_entries[i%len(tier_entries)]
            tasks.append(dict(spec,case=entry['case'],horizon=40,world='AB' if i%2==0 else 'BA',
                seed=490001+i,search_seed=490001+i,budget_ms=200.,fixed=2,
                opponent_simulations=3000,stage='engineering_full',obstacle_tier=tier,
                family=entry['family'],engineering_only=True))
    record=dict(version='e49_engineering_v1',entries=entries,tasks=tasks,method_instances=22,
                rule='Outcome-blind source indices9000+,seven families,three tiers; never scientific rows',
                horizon=40,fixed_pursuer_simulations=2,opponent_simulations=3000)
    write(manifest,record);return record


def paired_sampling(payload,boxes):
    """Independently replay fixed issued commands at .005, preserving captures.

    Ordinary within-grid capture-time changes and <=.02m stopped-target position
    changes are reported. Any target-set/terminal change or larger path impact
    blocks the gate; commands never get recomputed using finer observations.
    """
    from e49_audit import replay_motion
    trace=payload['trace'];initial=np.asarray(trace[0]['state'],float)
    p=initial[4:8].reshape(2,2).copy();e=initial[:4].reshape(2,2).copy()
    ap=initial[11:13]>.5;ae=initial[9:11]>.5
    events=[];mismatches=[];maximum=0.;active_maximum=0.
    for i,frame in enumerate(trace):
        action=np.asarray(frame['action'],float).copy();enemy=np.asarray(frame['opponent_action'],float).copy()
        action[~ap]=0.;enemy[~ae]=0.
        p,e,ap,ae,times,captors=replay_motion(p,e,action,enemy,ap,ae,np.asarray(boxes,float).reshape(-1,2,2),.005)
        reference=np.asarray(frame['next_state'],float)
        error=float(np.max(np.abs(np.r_[e.ravel(),p.ravel()]-reference[:8])))
        maximum=max(maximum,error)
        # Uncaught target and both pursuer paths should stay exactly identical.
        for j in range(2):
            if ae[j] and reference[9+j]>.5:active_maximum=max(active_maximum,float(np.max(abs(e[j]-reference[2*j:2*j+2]))))
        active_maximum=max(active_maximum,float(np.max(abs(p.ravel()-reference[4:8]))))
        if not np.array_equal(ae,reference[9:11]>.5):mismatches.append(dict(decision=i,reason='captured_set_changed'))
        for j in range(2):
            if times[j]>=0:events.append(dict(time=float(initial[8]+i+times[j]),target=j,pursuer=int(captors[j])))
    original=[event for frame in trace for event in frame['capture_events']]
    initial_post_capture=bool(np.any(initial[9:11]<.5))
    time_differences=[]
    for event in original:
        other=next((v for v in events if v['target']==event['target']),None)
        if other is None:mismatches.append(dict(reason='capture_missing',target=event['target']))
        else:time_differences.append(dict(target=event['target'],original_time=event['time'],fine_time=other['time'],difference=other['time']-event['time']))
    original_complete=bool(payload['summary']['double_capture']);fine_complete=bool(not ae.any())
    if original_complete!=fine_complete:mismatches.append(dict(reason='terminal_result_changed'))
    if maximum>.02+1e-9 or active_maximum>2e-8:mismatches.append(dict(reason='significant_path_change',max_error=maximum,active_error=active_maximum))
    return dict(passed=not mismatches,interval=.005,frames=len(trace),events=events,
                time_differences=time_differences,maximum_position_difference=maximum,
                maximum_active_path_difference=active_maximum,mismatches=mismatches,
                before_first_frames=sum(not initial_post_capture and not any(ev['time']<=f['state'][8] for ev in original) for f in trace),
                after_first_frames=sum(initial_post_capture or any(ev['time']<=f['state'][8] for ev in original) for f in trace))


def _job(task,out,replay=False):
    from run_e49_benchmark import execute,load_episode,read,run_episode,write,identity
    started=time.perf_counter();row=execute(task,out)
    if not row['valid']:return row
    payload=load_episode(row['episode']);case=read(task['case'])
    row['sampling_review']=paired_sampling(payload,case['map'])
    if replay:
        try:
            second=run_episode(task,out)
            for first,other in zip(payload['trace'],second['trace']):
                for key in ('action','opponent_action','opponent_raw_action','next_state'):
                    np.testing.assert_array_equal(first[key],other[key])
            if len(payload['trace'])!=len(second['trace']):raise AssertionError('Replay length differs')
            row['fixed_rng_replay']=True
        except Exception:
            row['fixed_rng_replay']=False;row['replay_error']=traceback.format_exc()
    row['engineering_seconds']=time.perf_counter()-started
    write(Path(out)/'engineering'/'checks'/(identity(task)+'.json'),row)
    return row


def _worker(cpu_queue):
    from run_e49_benchmark import worker_init
    worker_init(cpu_queue)


def run_tasks(out,tasks,workers=4,replay=False):
    from run_e49_benchmark import cpu_layout,write
    core=core_fingerprint()
    tasks=[dict(task,engineering_core=core) for task in tasks]
    layout=cpu_layout(workers);context=mp.get_context('spawn');queue=context.Queue()
    for cpu in layout['cpus']:queue.put(cpu)
    write(Path(out)/'engineering'/'resource_layout.json',layout)
    rows=[]
    with cf.ProcessPoolExecutor(len(layout['cpus']),mp_context=context,initializer=_worker,initargs=(queue,)) as pool:
        futures={pool.submit(_job,task,str(out),replay):task for task in tasks}
        for future in cf.as_completed(futures):
            try:row=future.result()
            except Exception:row=dict(valid=False,task=futures[future],error=traceback.format_exc())
            rows.append(row)
            print(json.dumps(dict(engineering_completed=len(rows),total=len(tasks),
                  method=row['task']['name'],valid=row['valid']),ensure_ascii=False),flush=True)
    return rows


def smoke(out,workers=4,limit=2,opponent_simulations=20):
    from e49_controllers import method_specs
    from run_e49_benchmark import write
    core_start=core_fingerprint()
    config=prepare_engineering(out)
    case=next(e['case'] for e in config['entries'] if e['family']=='long_wall' and e['tier']=='24')
    from run_e49_benchmark import read
    staged=read(case)
    staged['id']+='__staged_2v1';staged['engineering_state']='staged_2v1'
    for world in staged['worlds']:
        world['initial'][10]=0.
    for frame in staged['history']:
        frame['active_e'][1]=False;frame['reports'][1]=[None,None]
    staged_path=Path(out)/'engineering'/'cases'/'24'/'long_wall_staged_2v1.json'
    write(staged_path,staged)
    tasks=[]
    for phase,path in (('2v2',case),('2v1',str(staged_path))):
        tasks.extend(dict(spec,case=path,horizon=40,world='AB',seed=490000+i,search_seed=490000+i,
                    fixed=2,limit=limit,budget_ms=200.,opponent_simulations=opponent_simulations,
                    stage='engineering_prefix',engineering_only=True,engineering_phase=phase)
                     for i,spec in enumerate(method_specs()))
    rows=run_tasks(out,tasks,workers,replay=True)
    core_end=core_fingerprint()
    result=dict(core_start=core_start,core_end=core_end,prefix_only=True,limit=limit,rows=rows,passed=core_start==core_end and len(rows)==44 and all(r['valid'] and r.get('fixed_rng_replay') and r['sampling_review']['passed'] for r in rows),gate=False)
    write(Path(out)/'engineering'/'smoke.json',result);return result


def run_engineering(out,workers=4):
    from run_e49_benchmark import write,load_episode
    core_start=core_fingerprint()
    config=prepare_engineering(out);rows=run_tasks(out,config['tasks'],workers)
    core_end=core_fingerprint()
    expected={(t['name'],t['obstacle_tier']) for t in config['tasks']}
    observed=set();failed=[]
    for row in rows:
        task=row['task']
        if not row['valid']:failed.append(dict(method=task['name'],tier=task['obstacle_tier'],reason=row.get('error')));continue
        summary=row['summary']
        complete=summary['double_capture'] or summary['decisions']==40
        if complete and row['sampling_review']['passed']:observed.add((task['name'],task['obstacle_tier']))
        else:failed.append(dict(method=task['name'],tier=task['obstacle_tier'],reason='incomplete_or_sampling_mismatch'))
    from run_e49_benchmark import read
    smoke_path=Path(out)/'engineering'/'smoke.json'
    smoke_record=read(smoke_path) if smoke_path.exists() else {}
    replay_check=smoke_record.get('passed',False) and len(smoke_record.get('rows',[]))==44
    if replay_check and rows:
        replay_check=(smoke_record.get('core_start')==core_start and smoke_record.get('core_end')==core_start
                      and all(r['task'].get('engineering_core')==core_start for r in smoke_record['rows']))
    result=dict(gate=observed==expected and not failed and replay_check and core_start==core_end,
                core_start=core_start,core_end=core_end,core_unchanged=core_start==core_end,
                fixed_seed_replay_passed=replay_check,expected_full_episodes=len(expected),passed_full_episodes=len(observed),
                fixed_workload=True,horizon=40,fixed_pursuer_simulations=2,opponent_simulations=3000,
                missing=sorted(expected-observed),failures=failed,rows=rows,
                scientific_evaluation=False)
    write(Path(out)/'engineering_checks.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','smoke','full'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    result=prepare_engineering(args.out) if args.command=='prepare' else smoke(args.out,args.workers) if args.command=='smoke' else run_engineering(args.out,args.workers)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','tasks','entries')},ensure_ascii=False,default=str))


if __name__=='__main__':main()
