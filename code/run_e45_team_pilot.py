"""E45-A: bounded 2v2 integration and transfer pilot, no confirmation claims."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import argparse
import concurrent.futures as futures
import fcntl
import json
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
from solvers.e45_team_adapter import TeamController,TeamBelief,public_observation
from solvers.frozen_d_search_interface import root_sample,belief_view,target_step,loader
from solvers.root_audit_puct import RootPUCT,cpp
from run_frozen_d_closed_loop import save,sha,legal_action
from audit_frozen_d_search_interface import independent_traces
from estimator_audit_geometry import blocked
from corner_estimator_scenarios import construct,original_model
from prepare_l45_occlusion_scenarios import transform
from run_e44_parallel import choose_cpus

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'current/e45_2v2_pilot_20261027'
E44=ROOT/'current/e44_branch_conditioned_nte_20261027'
METHODS=('IG','AG','AFN17','AFN29','AFN43')
OPPONENTS=('fixed_D_kernel','rootpuct')
BASE_SEED=94520261027


def read(path):return json.loads(Path(path).read_text())
def checkpoint(method):return E44/'models'/f'route_{method[3:]}.npz' if method.startswith('AFN') else None
def emit(**kw):
    row=dict(pid=os.getpid(),timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw)
    save(OUT/'status.json',row)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(row,ensure_ascii=False),flush=True)


def mirror(q,frame):
    q=np.asarray(q)-np.asarray(frame['translation'])
    for _ in range(frame['quarter_turns']):q=np.array([q[1],-q[0]])
    return transform([q[0],-q[1]],frame)


def make_case(family,index):
    seed=int(np.random.SeedSequence([BASE_SEED,('long_wall','wall_corner','multi_route').index(family),index]).generate_state(1)[0])
    height=float(np.random.default_rng(seed).uniform(6.2,6.9));speed=1.1 if index==0 else 1.3
    old,packet=construct(family,4,speed,seed,height,index,pause=index)
    p=original_model(packet['map'],speed);worlds=[];shared=None
    for name,order in (('AB',(0,1)),('BA',(1,0))):
        a,b=[old['worlds'][i]['history_states'] for i in order]
        states=[];history=[]
        for x,y in zip(a,b):
            s=np.asarray(x['full_state'],float).reshape(-1,1).copy()
            s[p.state_idxs[1],0]=mirror(y['evader'],old['frame'])
            s[p.state_idxs[3],0]=mirror(y['pursuer'],old['frame'])
            s[p.active_idxs,0]=1.
            if not p.is_valid(s) or p.get_capture_pairs(s):raise RuntimeError('Illegal scripted 2v2 history')
            if states:
                prev=np.asarray(states[-1]).reshape(-1,1);joint=np.zeros((8,1))
                for robot in range(4):
                    delta=s[p.state_idxs[robot],0]-prev[p.state_idxs[robot],0]
                    assert legal_action(prev[p.state_idxs[robot],0],delta,1. if robot<2 else speed,np.asarray(packet['map']))
                    joint[p.action_idxs[robot],0]=delta
                nxt=p.step(prev,joint,1.)
                assert np.array_equal(nxt[p.active_idxs],s[p.active_idxs])
                for robot in range(4):np.testing.assert_allclose(nxt[p.state_idxs[robot]],s[p.state_idxs[robot]],atol=1e-10,rtol=0)
                s=nxt
            states.append(s[:,0].tolist())
            history.append(public_observation(int(s[8,0]),s[4:8,0].reshape(2,2),s[:4,0].reshape(2,2),
                np.ones(2,bool),np.ones(2,bool),np.asarray(packet['map'])))
        if shared is None:shared=history
        else:assert shared==history,'Worlds leaked into public history'
        assert all(all(q is None for q in reports) for reports in history[-1]['reports'])
        worlds.append(dict(id=name,initial=states[-1],scripted_states=states))
    return dict(id='team_'+old['id'],family=family,map=packet['map'],speed=speed,history=shared,worlds=worlds,
        scope='development pilot only; known identities, noiseless shared range6 sensor; paired capture retirement')


def prepare():
    if (OUT/'experiment.json').exists():return verify()
    OUT.mkdir(parents=True,exist_ok=True)
    entries=[]
    for family in ('long_wall','wall_corner','multi_route'):
        for i in range(2):
            case=make_case(family,i);folder=OUT/'bank';save(folder/(case['id']+'.json'),case)
            entries.append(dict(id=case['id'],family=family,path=str(folder/(case['id']+'.json'))))
    save(OUT/'bank/manifest.json',dict(entries=entries))
    files=[ROOT/'code'/s for s in ('run_e45_team_pilot.py','solvers/e45_team_adapter.py','tests/test_e45_team_adapter.py','E45_TEAM_PROTOCOL_20261027.md')]
    e44=read(E44/'experiment.json');protected=e44['source_hashes']|e44['protected_hashes']|read(E44/'models_frozen.json')['files']
    config=dict(experiment='E45-A',scope='2v2 development integration/transfer pilot, not joint-NTE confirmation',
        cases=6,worlds=['AB','BA'],opponents=list(OPPONENTS),methods=list(METHODS),episodes=120,
        team_budget_ms=200.,capture='both matched pursuer and evader retire; 2v2 becomes 1v1',
        shared_sensor='identified target position iff any active teammate sees it; both negative masks condition once-predicted belief',
        estimator='frozen D per-target kernel + new explicitly audited team-observation wrapper; factorized target beliefs',
        geometry='IG independently picks cheapest target; AG exact min-max one-to-one assignment, sum tie breaker',
        neural='all three frozen E44 route seeds; assignment + independent 2D NTE per pursuer, shared total200ms, mandatory G first option and checked G fallback',
        training_performed=False,joint_action_search=False,new_4d_joint_policy_trained=False,
        main_metrics=['double_capture','completion_cost','first_capture','post_first_capture_completion','redundant_assignment','latency'],
        no_performance_gate=True,source_hashes={str(p):sha(p) for p in files},protected_hashes=protected,
        bank_hashes={str(p):sha(p) for p in (OUT/'bank').glob('*.json')})
    save(OUT/'experiment.json',config)
    (OUT/'PROTOCOL.md').write_bytes((ROOT/'code/E45_TEAM_PROTOCOL_20261027.md').read_bytes())
    for p in files:
        q=OUT/'source_snapshot'/p.name;q.parent.mkdir(exist_ok=True);q.write_bytes(p.read_bytes())
    emit(stage='registered',episodes=120)
    return config


def verify():
    loader.verify();c=read(OUT/'experiment.json')
    for section in ('source_hashes','protected_hashes','bank_hashes'):
        for p,h in c[section].items():
            if sha(p)!=h:raise RuntimeError('Frozen source changed: '+p)
    return c


class FixedKernelTargets:
    def __init__(self,case,seed):
        b=TeamBelief(case['map'],seed)
        for obs in case['history']:b.update(obs)
        self.graphs=[e.graph for e in b.estimators]
        self.rng=[np.array([(seed+94501+104729*j)%4294967296],np.int64) for j in range(2)]
        self.ids=[];self.modes=[]
        for j,e in enumerate(b.estimators):
            _,i,m=root_sample(*belief_view(e),self.rng[j]);self.ids.append(i);self.modes.append(m)
    def positions(self):return np.array([g[0][i] for g,i in zip(self.graphs,self.ids)])
    def actions(self,active):
        before=self.positions()
        for j,g in enumerate(self.graphs):
            if active[j]:self.ids[j],self.modes[j]=target_step(self.ids[j],self.modes[j],g[1],g[2],g[3],self.rng[j])
        return self.positions()-before


def tasks():
    result=[]
    for entry in read(OUT/'bank/manifest.json')['entries']:
        for wi,world in enumerate(('AB','BA')):
            seed=int(np.random.SeedSequence([BASE_SEED,int(sha(entry['path'])[:8],16),wi]).generate_state(1)[0])
            for opponent in OPPONENTS:
                for method in METHODS:
                    result.append(dict(key=f'{entry["id"]}_{world}_{opponent}_{method}',case=entry['path'],world=world,
                        method=method,opponent=opponent,seed=seed,family=entry['family'],group=entry['id']))
    return result


def episode(task,limit=None,fixed=None):
    case=read(task['case']);boxes=np.asarray(case['map']);p=original_model(case['map'],case['speed'])
    start=time.perf_counter();own=TeamController(case['map'],case['speed'],task['seed'],task['method'],checkpoint(task['method']))
    for obs in case['history'][:-1]:own.observe(obs)
    setup=1000*(time.perf_counter()-start)
    state=np.asarray(next(w['initial'] for w in case['worlds'] if w['id']==task['world'])).reshape(-1,1)
    if task['opponent']=='fixed_D_kernel':
        other=FixedKernelTargets(case,task['seed']);state[:4,0]=other.positions().reshape(-1)
    else:
        other=RootPUCT([None]*4,None,3000,0.,0.);other.seeded_state(p,task['seed']+17,task['seed']+29)
    remaining=80-int(state[8,0]);trace=[];first=None;failure=None
    for k in range(min(remaining,limit) if limit else remaining):
        ap=state[p.active_idxs[2:],0]>.5;ae=state[p.active_idxs[:2],0]>.5
        obs=public_observation(int(state[8,0]),state[4:8,0].reshape(2,2),state[:4,0].reshape(2,2),ap,ae,boxes)
        # Independent sensor implementation validates every public measurement.
        for j in range(2):
            for i in range(2):
                seen=bool(ap[i] and ae[j] and np.linalg.norm(state[4+2*i:6+2*i,0]-state[2*j:2*j+2,0])<=6.+1e-12
                    and not blocked(state[4+2*i:6+2*i,0],state[2*j:2*j+2,0],boxes))
                assert seen==(obs['reports'][j][i] is not None)
        own.budget_ms=max(0.,200.-setup) if k==0 else 200.
        stamp=time.perf_counter();actions=own.act(obs,fixed_simulations=fixed);latency=1000*(time.perf_counter()-stamp)+(setup if k==0 else 0.)
        if task['opponent']=='fixed_D_kernel':raw=other.actions(ae)
        else:raw=np.asarray(other.search(p,state.copy(),turn=0).best_action).reshape(-1)[:4].reshape(2,2)
        opponent=raw.copy();shields=0
        for i in range(2):
            if not ae[i]:opponent[i]=0.
            elif not legal_action(state[2*i:2*i+2,0],opponent[i],1.,boxes):
                if task['opponent']=='fixed_D_kernel':raise RuntimeError('Illegal model target transition')
                opponent[i]=0.;shields+=1
            if not ap[i]:assert np.all(actions[i]==0)
            assert legal_action(state[4+2*i:6+2*i,0],actions[i],case['speed'],boxes)
        simulated=0
        for stats in own.last_stats['searches']:
            controller=own.models[stats['pursuer'],stats['target']]
            sims=np.concatenate(controller.traces) if controller.traces else np.zeros((0,10))
            assert not any(independent_traces(sims,boxes,case['speed']));simulated+=len(sims)
        joint=np.r_[opponent.reshape(-1),actions.reshape(-1)][:,None];nxt=p.step(state,joint,1.)
        assert p.is_valid(nxt)
        for robot in range(4):np.testing.assert_allclose(nxt[2*robot:2*robot+2,0],state[2*robot:2*robot+2,0]+joint[2*robot:2*robot+2,0],atol=1e-10,rtol=0)
        before=int(ae.sum());after=int((nxt[p.active_idxs[:2],0]>.5).sum())
        assert int((nxt[p.active_idxs[2:],0]>.5).sum())==after
        if before>after and first is None:first=k+1
        trace.append(dict(observation=obs,state=state[:,0].tolist(),next_state=nxt[:,0].tolist(),
            actions=actions.tolist(),opponent_actions=opponent.tolist(),stats=own.last_stats,latency_ms=latency,
            belief_signature=own.belief.signature(),simulated_transitions=simulated,opponent_shields=shields,
            remaining_targets=after,new_captures=before-after))
        state=nxt
        if p.is_terminal(state):break
    complete=not (state[p.active_idxs[:2],0]>.5).any()
    # Reconstruct every public belief without access to target truth.
    audit=TeamBelief(case['map'],task['seed'])
    for obs in case['history'][:-1]:audit.update(obs)
    for row in trace:
        audit.update(row['observation']);assert audit.signature()==row['belief_signature']
    summary=dict(**task,double_capture=bool(complete),completion_cost=len(trace) if complete else remaining,
        first_capture=first,captures=2-int((state[p.active_idxs[:2],0]>.5).sum()),
        after_first_cost=(len(trace)-first if complete else remaining-first) if first is not None else None,
        decisions=len(trace),latencies_ms=[r['latency_ms'] for r in trace],overruns=sum(r['latency_ms']>200 for r in trace),
        redundant_assignment=sum(len(r['stats']['assignment'])==2 and len(set(r['stats']['assignment'].values()))==1 for r in trace),
        failure=failure,reason='double_capture' if complete else 'engineering_prefix' if limit else 'timeout',
        public_replay_exact=True,simulated_transitions=sum(r['simulated_transitions'] for r in trace))
    return dict(summary=summary,trace=trace)


def initializer(output,queue):
    global OUT
    OUT=Path(output);os.sched_setaffinity(0,{queue.get()});verify()
    case=read(read(OUT/'bank/manifest.json')['entries'][0]['path'])
    for method in ('AG','AFN17'):
        c=TeamController(case['map'],case['speed'],317,method,checkpoint(method))
        for obs in case['history'][:-1]:c.observe(obs)
        c.act(case['history'][-1],fixed_simulations=2)


def worker(task):
    path=OUT/'episodes'/(task['key']+'.json');cache=OUT/'cache'/(task['key']+'.json')
    if cache.exists():
        row=read(cache);assert sha(path)==row['episode_sha256'];return row
    try:payload=episode(task)
    except BaseException as exc:
        save(OUT/'failures'/(task['key']+'.json'),dict(task=task,error=repr(exc)));raise
    save(path,payload);row=payload['summary']|dict(episode=str(path),episode_sha256=sha(path));save(cache,row);return row


def report(rows,complete=False):
    lines=['# E45-A：2v2 接入、强基线与分解 NTE 迁移试验','',
        '本阶段是开发试验，不是四维联合 NTE 已完成训练。捕获双方同时退场；首捕后转为1v1。',
        'IG为各自贪心选择目标，AG为一对一最小最大路径成本分配；AFN为AG分配加保留G候选的两个独立NTE，复用E44三种子。',
        '',f'完成 {len(rows)}/120 局；状态：'+('complete' if complete else 'running')+'。',
        '', '|对手|方法|双捕/局数|完成成本|首捕局数|决策P95 ms|超过200ms|','|---|---|---:|---:|---:|---:|---:|']
    for o in OPPONENTS:
        for m in METHODS:
            rr=[r for r in rows if r['opponent']==o and r['method']==m]
            if rr:
                lat=[x for r in rr for x in r['latencies_ms']]
                lines.append(f'|{o}|{m}|{sum(r["double_capture"] for r in rr)}/{len(rr)}|{np.mean([r["completion_cost"] for r in rr]):.2f}|{sum(r["first_capture"] is not None for r in rr)}|{np.percentile(lat,95):.2f}|{sum(r["overruns"] for r in rr)}|')
    lines+=['','固定D运动核只在目标首次重发现前保持生成核匹配；不将后续双捕称作全程模型匹配。',
        '本阶段不按收益筛选种子，不证明完整联合搜索或新神经架构贡献。后续E45-B需新四维策略训练和独立测试。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','run','smoke'));parser.add_argument('--workers',type=int,default=32)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output is not None:OUT=args.output.resolve()
    prepare();verify()
    if args.stage=='prepare':return
    if args.stage=='smoke':
        rows=[]
        for task in tasks()[::5]:
            for method in ('AG','AFN17'):
                t=task|dict(method=method,key=task['key']+'_smoke_'+method)
                result=episode(t,limit=2,fixed=2);rows.append(result['summary'])
        save(OUT/'smoke.json',dict(passed=True,prefixes=len(rows)));print('E45 smoke passed',len(rows));return
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    requests=tasks();save(OUT/'requests.json',requests);cores=choose_cpus(min(args.workers,len(requests)))
    save(OUT/'execution.json',dict(workers=len(cores),physical_cores=cores,threads=1,pid=os.getpid(),timing_scope='pinned parallel development load'))
    ctx=mp.get_context('spawn');queue=ctx.Queue()
    for c in cores:queue.put(c)
    rows=[];emit(stage='running',completed=0,total=120,workers=len(cores));report(rows)
    with futures.ProcessPoolExecutor(max_workers=len(cores),mp_context=ctx,initializer=initializer,initargs=(str(OUT),queue)) as pool:
        for f in futures.as_completed([pool.submit(worker,t) for t in requests]):
            rows.append(f.result());emit(stage='pilot_evaluation',completed=len(rows),total=120)
            if len(rows)%10==0:report(rows)
    rows.sort(key=lambda r:r['key']);save(OUT/'results.json',rows);verify();report(rows,True)
    save(OUT/'decision.json',dict(status='complete',episodes=len(rows),new_training=False,joint_NTE_validated=False,
        next_stage='E45-B task-route conditioned 4D joint proposals with shared beliefs and protected geometric support'))
    emit(stage='complete',episodes=len(rows))


if __name__=='__main__':main()
