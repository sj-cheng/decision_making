"""E55 paired evaluation, teacher collection and same-observation NAE ablation."""
import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
import repo_paths
repo_paths.install()
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
import sys

import numpy as np
from run_e50_distributed_dynamic_smoke import run
from run_e52_train_nar_nae import records_from_run
from solvers.e54_pursuer_coordination import coordinate

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'current/E系列实验输出'
E52 = ARCHIVE / 'e52_combined_20261118'
OUT = ARCHIVE / 'e55_slow_pursuit'
FROZEN = E52 / 'production_parallel_v3/frozen_nar_nae.pt'
BRANCH = E52 / 'nae_counterfactual_expanded_20261119/advantage.pt'
BASE = dict(e52=True, e52_budget=256, e52_depth=8, cycles=160, evader_hmcts=True,
            evader_minco=True, shared_max_speed=1.5, pursuer_speed_cap=1.125, max_acceleration=1.5,
            e53_depth=8, e53_simulations=256, e52_time_ms=None, e52_nae_strategy='leaf_residual')
ARMS = [('none',False),('nar',False),('nae',False),('nae',True),('both',False),('both',True)]


def save(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False))
    tmp.replace(path)


def config(method):
    mode, branch = ARMS[(method-1)//2]
    return dict(BASE,e52_mode=mode,e52_checkpoint=str(FROZEN) if mode!='none' else None,
                e52_advantage_checkpoint=str(BRANCH) if branch else None,
                pursuer_structure='coordinated' if method%2==0 else 'local')


def init():
    banks=ROOT/'current/trajectory_visualizations_20261107'
    cases=[]
    omitted=[]
    for name in ('dense_obstacles','hard_initials'):
        for source in sorted((banks/name/'bank').glob('*_[AB][AB].json')):
            data=json.loads(source.read_text())
            world=next(w for w in data['worlds'] if w['id']==source.stem[-2:])
            if np.max(np.abs(world['initial'][:8]))>9:
                omitted.append(str(source))
                continue
            target=OUT/'cases'/source.name
            save(target,dict(data,worlds=[world]))
            cases.append(target)
    def layout(p):
        return json.dumps(json.loads(p.read_text())['map'],sort_keys=True)
    excluded=set(map(layout,cases))
    rng=random.Random(55)
    train={}
    for split,n in [('train',96),('validation',20),('test',20)]:
        bank=ARCHIVE/'e49_persistent_pursuit_benchmark_20261108/bank'/split
        pool=sorted(bank.glob('*.json'))
        rng.shuffle(pool)
        selected=[]
        for p in pool:
            data=json.loads(p.read_text())
            initial=np.asarray(data['worlds'][0]['initial']).ravel()[:8]
            key=layout(p)
            if key in excluded or np.max(np.abs(initial))>9:
                continue
            excluded.add(key)
            selected.append(str(p))
            if len(selected)==n:break
        assert len(selected)==n
        train[split]=selected
    assert FROZEN.exists() and BRANCH.exists() and len(cases)==25
    save(OUT/'manifest.json',dict(cases=list(map(str,cases)),outside_motion_bounds=omitted,train=train,seeds=[17,43],
          base=BASE,arms={i:config(i) for i in range(1,13)},diagnostic_cases=list(map(str,cases[::4])),
          teacher_simulations=2048,teacher_cycles=48,teacher_seed=55))


def outcome(x):
    times=[a['time'] for a in x['captures']]
    cap=x['requested_cycles']
    assert len(times)==2 or len(x['logs'])==cap
    return dict(captures=len(times),first=(times+[cap])[0],double=(times+[cap,cap])[1],
                cycles=x['cycles'],collisions=sum(l['pursuer_collision'] for l in x['logs']))


def same(a,b):
    return (np.allclose(a.endpoint,b.endpoint,rtol=0,atol=1e-8) and
            abs(a.duration-b.duration)<1e-8 and a.pass_through==b.pass_through)


class Shadow:
    """Rerun the exact decision without NAE; optionally intervene once."""
    def __init__(self,method,intervene):
        self.method,self.intervene=method,intervene
        self.rows=[]
        self.flip=None

    def __call__(self,cycle,obs,controllers,plans):
        if self.intervene and self.flip is not None:return
        saved=[c.last_search for c in controllers.values()]
        shadow=[]
        audit=[]
        for i in range(2):
            c=controllers[i]
            net=c.nae
            try:
                c.nae=None
                shadow.append(c.act(obs[i]))
                audit.append(c.last_search)
            finally:
                c.nae=net
                c.last_search=saved[i]
        if self.method%2==0:
            shadow,_=coordinate(shadow,audit,[obs[i].belief for i in range(2)])
        changed=[not same(plans[i],shadow[i]) for i in range(2)]
        def rank(record):
            return sorted(range(len(record['root'])),key=lambda j:
                          (record['root'][j]['visits'],record['root'][j]['value']),reverse=True)
        self.rows.append(dict(cycle=cycle,queried=[s['nae_queries']>0 for s in saved],
                         rank_changed=[rank(saved[i])!=rank(audit[i]) for i in range(2)],
                         root_action_changed=[saved[i].get('selected_root_action') != audit[i].get('selected_root_action')
                                              for i in range(2)],
                         executed_plan_changed=changed))
        if any(changed) and self.flip is None:
            self.flip=cycle
            if self.intervene:
                plans.update(enumerate(shadow))


def evaluation(task):
    case,seed,method,phase=task
    folder=OUT/phase/f'm{method:02}'/f'{Path(case).stem}_{seed}'
    record=folder/'result.json'
    if record.exists():return json.loads(record.read_text())
    cfg=config(method)
    row=dict(case=Path(case).stem,seed=seed,method=method)
    if phase=='comparison':
        x=run(case,folder/'trace.json',seed=seed,**cfg)
        row.update(outcome(x))
    else:
        hook=Shadow(method,False)
        x=run(case,folder/'reference.json',seed=seed,decision_hook=hook,audit_nae=True,**cfg)
        row.update(outcome(x),flip=hook.flip,decisions=len(hook.rows)*2)
        for key in ("peer_groups","peer_reorders","leaf_winner_changes"):
            row[key]=sum(s[key] for l in x["logs"] for s in l["search"].values())
        for k in ('queried','rank_changed','executed_plan_changed'):
            row[k]=sum(sum(s[k]) for s in hook.rows)
        save(folder/'decisions.json',hook.rows)
        other=row['double']
        if hook.flip is not None:
            intervention=Shadow(method,True)
            y=run(case,folder/'intervention.json',seed=seed,decision_hook=intervention,**cfg)
            assert intervention.flip==hook.flip
            for a,b in zip(x['logs'][:hook.flip],y['logs'][:hook.flip]):
                assert a['pursuers']==b['pursuers'] and a['targets']==b['targets']
            other=outcome(y)['double']
        row['double_s_without_first_changed_action']=other
        row['seconds_saved_by_first_changed_action']=other-row['double']
    save(record,row)
    return row


class Alternative:
    def __init__(self,cycle,aid):
        self.cycle,self.aid=cycle,aid
        self.chosen=None

    def __call__(self,cycle,obs,controllers,plans):
        if cycle!=self.cycle:return
        i=self.aid
        plan=plans[i]
        options=controllers[i].last_search['root']
        choices=[(j,v) for j,v in enumerate(options) if v['visits'] and
                 np.linalg.norm(np.asarray(v['endpoint'])-plan.endpoint)>.1]
        if not choices:return
        j,v=max(choices,key=lambda pair:pair[1]['visits'])
        endpoint=np.asarray(v['endpoint'])
        plans[i]=replace(plan,action=endpoint-obs[i].pose,endpoint=endpoint,
                         duration=v['duration'],pass_through=v['pass_through'],candidate_id=j,
                         value=v['value'],visits=v['visits'])
        self.chosen=j


def teacher(task):
    split,case=task
    folder=OUT/'teacher'/split/Path(case).stem
    data=folder/'labels.json'
    if data.exists():return
    cfg=dict(config(1),e52_budget=2048,seed=55,cycles=48)
    reference=folder/'reference.json'
    x=json.loads(reference.read_text()) if reference.exists() else run(case,reference,**cfg)
    records=records_from_run(x)
    pairs=[]
    for target in (4,16,28):
        candidates=[r for r in records if target<=r['cycle']<target+8 and
                    sum(v>0 for v in r['policy'])>=2 and
                    x['logs'][r['cycle']]['sensor_stats'][str(r['agent_id'])]['belief_particles']]
        if not candidates:continue
        ref=candidates[0]
        t,i=ref['cycle'],ref['agent_id']
        hook=Alternative(t,int(i))
        y=run(case,folder/f'alt_{t}.json',**dict(cfg,cycles=t+8),decision_hook=hook)
        if hook.chosen is None:continue
        for a,b in zip(x['logs'][:t],y['logs'][:t]):
            assert a['pursuers']==b['pursuers'] and a['targets']==b['targets']
        alt=next(r for r in records_from_run(y) if r['cycle']==t and r['agent_id']==i)
        assert ref['features']==alt['features'] and ref['value_index']!=alt['value_index']
        pairs.append(dict(reference=ref,alternative=alt))
    save(data,dict(records=records,pairs=pairs))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['init','comparison','diagnosis','teacher'])
    parser.add_argument('--workers',type=int,default=1)
    a=parser.parse_args()
    if a.phase=='init':return init()
    m=json.loads((OUT/'manifest.json').read_text())
    if a.phase=='teacher':
        tasks=[(s,p) for s,cases in m['train'].items() for p in cases]
        fn=teacher
    else:
        diag=a.phase=='diagnosis'
        tasks=[(p,s,i,a.phase) for p in (m['diagnostic_cases'] if diag else m['cases'])
               for s in ([17] if diag else m['seeds']) for i in (range(5,13) if diag else range(1,13))]
        fn=evaluation
    errors=[]
    with ProcessPoolExecutor(a.workers) as pool:
        pending={pool.submit(fn,t):t for t in tasks}
        for j in as_completed(pending):
            try:j.result()
            except Exception as exc:
                errors.append((pending[j],repr(exc)))
                print(errors[-1],file=sys.stderr,flush=True)
    if errors:raise RuntimeError(errors)
    if a.phase!='teacher':
        rows=[json.loads(p.read_text()) for p in sorted((OUT/a.phase).glob('*/*/result.json'))]
        assert len(rows)==len(tasks)
        with (OUT/a.phase/'episodes.csv').open('w') as f:
            w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)
        summary={}
        for i in sorted({r['method'] for r in rows}):
            rs=[r for r in rows if r['method']==i]
            summary[i]=dict(n=len(rs),double=sum(r['captures']==2 for r in rs),
                            first_s=mean(r['first'] for r in rs),double_s=mean(r['double'] for r in rs))
            if a.phase=='diagnosis':
                den=sum(r['decisions'] for r in rs)
                summary[i].update({k:sum(r[k] for r in rs)/den for k in ('queried','rank_changed','executed_plan_changed')})
                changed=[r for r in rs if r['flip'] is not None]
                summary[i].update(changed_episodes=len(changed),mean_seconds_saved_on_changed=
                    mean(r['seconds_saved_by_first_changed_action'] for r in changed) if changed else None)
        save(OUT/a.phase/'summary.json',summary)
if __name__=='__main__': main()
