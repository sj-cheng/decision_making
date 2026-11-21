"""Reproducible E52 teacher/pilot/formal jobs with frozen paired initial states."""
import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
import repo_paths
repo_paths.install()
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import numpy as np
from run_e50_distributed_dynamic_smoke import run
from run_e52_train_nar_nae import initial_id


def job(spec):
    mode,case,out,cycles,checkpoint,seed,delay,drop = spec
    r=run(case,out,cycles=cycles,seed=seed,rootpuct=True,rootpuct_simulations=20,
        e52=mode!='e50', e52_checkpoint=checkpoint if mode in ('nar','nae','both') else None,
        e52_mode=mode,e52_depth=5 if mode=='teacher' else 3,
        e52_budget=1024 if mode=='teacher' else 256,e52_time_ms=1500. if mode=='teacher' else 200.,
        delay=delay, drop_cycles=drop,
        leaf_value_checkpoint='current/E系列实验输出/e50_deployment_audit_20261117/leaf_value_training/leaf_value_v1.npz' if mode=='e50' else None)
    times=[s['elapsed_ms'] for row in r['logs'] for s in row.get('search',{}).values()]
    positions=np.array([row['pursuers'] for row in r['logs']])
    movement=np.linalg.norm(np.diff(positions,axis=0),axis=-1)
    overlap=int(np.sum(np.linalg.norm(positions[:,0]-positions[:,1],axis=1)<1.))
    rediscovery=0
    for aid in ('0','1'):
        seen=[row['sensor_stats'][aid]['visible_targets']>0 for row in r['logs']]
        rediscovery+=sum(seen[i] and not seen[i-1] and any(seen[:i-1]) for i in range(1,len(seen)))
    captures=sorted(x['cycle'] for x in r['captures'])
    return dict(mode=mode,case=str(case),run=str(out),seed=seed,captures=len(captures),
        first_capture=captures[0] if captures else None,second_capture=captures[1] if len(captures)>1 else None,
        collision_cycles=sum(row['pursuer_collision'] for row in r['logs']),
        minco_infeasible=sum(not m['feasible'] for row in r['logs'] for m in row['minco'].values()),
        rootpuct_failures=r['rootpuct_failures'],terminal_opponent_skips=r['terminal_opponent_skips'],
        lifecycle_errors=sum(len(row['captures']) != len({c['target'] for c in row['captures']}) or any(track['target_id'] in {c['target'] for c in row['captures'] if c['cycle']<row['cycle']} for tracks in row.get('beliefs',{}).values() for track in tracks) for row in r['logs']),
        truncations=r['prefix_truncations'],
        stall_steps=int(np.sum(movement<.03)),overlap_cycles=overlap,rediscoveries=rediscovery,
        decision_p95_ms=float(np.percentile(times,95)) if times else None,
        peer_age_max=max([age for row in r['logs'] for ages in row['peer_ages'].values() for age in ages.values()] or [0]))


def summarize(out, manifest, rows):
    by_mode={}
    for mode in manifest['modes']:
        group=[r for r in rows if r['mode']==mode]
        if not group: continue
        by_mode[mode]=dict(episodes=len(group),double_captures=sum(r['captures']>=2 for r in group),
            single_captures=sum(r['captures']==1 for r in group),
            collisions=sum(r['collision_cycles'] for r in group),
            minco_infeasible=sum(r['minco_infeasible'] for r in group),
            rootpuct_failures=sum(r['rootpuct_failures'] for r in group),
            truncations=sum(r['truncations'] for r in group),
            decision_p95_max_ms=max((r['decision_p95_ms'] for r in group if r['decision_p95_ms'] is not None),default=None))
    formal=manifest['stage']=='formal' and set(manifest['modes'])=={'e50','geometry','nar','nae','both'} and len(rows)==len(manifest['jobs']) and len(rows)==150
    baseline=by_mode.get('e50',{}).get('double_captures',0)
    combined=by_mode.get('both',{})
    gates=dict(formal_complete=formal,
        capture_improvement=formal and combined.get('double_captures',0)>=baseline+3,
        historical_capture_target=formal and combined.get('double_captures',0)>=18,
        execution_safe=formal and combined.get('collisions',1)==0 and combined.get('minco_infeasible',1)==0,
        lifecycle=formal and all(r['lifecycle_errors']==0 and r['terminal_opponent_skips']==0 for r in rows),
        opponent_valid=formal and all(r['rootpuct_failures']==0 for r in rows),
        latency=formal and combined.get('decision_p95_max_ms') is not None and combined['decision_p95_max_ms']<=200.)
    report=dict(manifest=manifest,rows=rows,by_mode=by_mode,gates=gates)
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    lines=['# E52 experiment results','', 'Stage: '+manifest['stage'], '',
           'These results meet formal acceptance only when all formal gates pass.', '',
           '| Mode | Episodes | Double captures | Collisions | MINCO invalid | Max episode P95 ms |',
           '|---|---:|---:|---:|---:|---:|']
    for mode,r in by_mode.items():
        latency = '—' if r['decision_p95_max_ms'] is None else f"{r['decision_p95_max_ms']:.2f}"
        lines.append(f"| {mode} | {r['episodes']} | {r['double_captures']} | {r['collisions']} | {r['minco_infeasible']} | {latency} |")
    lines += ['', 'Gates: '+json.dumps(gates)]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def prepare_teacher_cases(cases, out):
    """Create distinct legal training starts using the fixed 5201 seed."""
    from e49_rules import _legal
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    generated=[]; rng=np.random.default_rng(5201)
    for i,path in enumerate(cases):
        case=json.loads(Path(path).read_text()); state=np.array(case['worlds'][0]['initial'],float)
        boxes=np.asarray(case['map'],float).reshape(-1,2,2)
        for j in range(4):
            original=state[2*j:2*j+2].copy()
            for _ in range(100):
                point=np.clip(original+rng.uniform(-1.5,1.5,2),-8,8)
                if _legal(point,boxes): state[2*j:2*j+2]=point; break
        for world in case['worlds']: world['initial']=state.tolist()
        target=out/f'train_case_{i}.json'; target.write_text(json.dumps(case)); generated.append(str(target))
    return generated


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=['teacher','smoke','pilot','formal'],default='smoke')
    p.add_argument('--modes',nargs='+',choices=['teacher','e50','geometry','nar','nae','both'])
    p.add_argument('--exclude-cases',nargs='+',default=[])
    p.add_argument('--cases',nargs='+'); p.add_argument('--checkpoint'); p.add_argument('--workers',type=int,default=8)
    p.add_argument('--delay',type=int,default=0); p.add_argument('--drop-cycle',action='append',type=int,default=[])
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    cases=sorted(a.cases or [str(p) for p in Path('current/E系列实验输出/e50_deployment_audit_20261117/formal_random_cases').glob('*/*.json')])
    if a.stage=='teacher':
        cases=prepare_teacher_cases(cases,out)
    elif a.stage=='smoke': cases=cases[:1]
    elif a.stage=='pilot' and not a.cases:
        cases=[str(p) for group in ('wall','corner','sparse') for p in sorted(Path('current/E系列实验输出/e50_deployment_audit_20261117/formal_random_cases',group).glob('*.json'))[:3]]
    if a.stage=='formal':
        if len(cases)!=30 or len({initial_id(json.loads(Path(p).read_text())) for p in cases})!=30:
            raise ValueError('formal suite requires 30 distinct frozen initial states')
        if any(sum(Path(p).parent.name==group for p in cases)!=10 for group in ('wall','corner','sparse')):
            raise ValueError('formal suite requires ten cases per map group')
    modes=a.modes or (['teacher'] if a.stage=='teacher' else (['geometry'] if a.stage=='smoke' else ['e50','geometry','nar','nae','both']))
    if a.stage=='teacher' and modes!=['teacher']: raise ValueError('teacher stage requires teacher mode')
    if any(mode in ('nar','nae','both') for mode in modes) and not a.checkpoint:
        raise ValueError('trained frozen checkpoint required for neural modes')
    if a.checkpoint:
        import torch
        metadata=torch.load(a.checkpoint,map_location='cpu',weights_only=False).get('metadata',{})
        used=set(metadata.get('train_initial_ids',[])+metadata.get('validation_initial_ids',[]))
        if any(initial_id(json.loads(Path(p).read_text())) in used for p in cases):
            raise ValueError('evaluation initial state overlaps network training/validation')
    cycles=2 if a.stage=='smoke' else 160
    specs=[(m,p,str(out/f'{m}_{i:02d}.json'),cycles,a.checkpoint,17,a.delay,a.drop_cycle) for i,p in enumerate(cases) for m in modes if p not in a.exclude_cases]
    manifest=dict(stage=a.stage,cases=cases,cycles=cycles,modes=modes,checkpoint=a.checkpoint,
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest() if a.checkpoint else None,
        jobs=specs,delay=a.delay,drop_cycles=a.drop_cycle)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(job,s):s for s in specs}
        for f in concurrent.futures.as_completed(futures):
            row=f.result(); rows.append(row)
            summarize(out,manifest,rows)
            print(json.dumps(row),flush=True)


if __name__=='__main__': main()
