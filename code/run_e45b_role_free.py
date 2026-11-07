"""E45-B: role-free joint belief MCTS policy iteration, resumable in tmux."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[_k]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import concurrent.futures as futures
import fcntl
import hashlib
import multiprocessing as mp
import signal
import time
from pathlib import Path
import numpy as np
import torch
from learning.e45b_joint_network import Network,Runtime,KINDS,KEYS,losses,augment
from solvers.e45b_joint_nte import Controller,coverage_potential,simultaneous_capture,ETA,TIME_COST
from solvers.e45_team_adapter import TeamController,TeamBelief,public_observation
from solvers.root_audit_puct import RootPUCT
from run_e45_team_pilot import FixedKernelTargets
from run_frozen_d_closed_loop import save,sha,legal_action
from corner_estimator_scenarios import original_model
from estimator_audit_geometry import blocked
from e45b_scenarios import make_case,audit_simulated
from e45b_sensor_audit import visible_batch
from run_e44_parallel import choose_cpus

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'current/e45b_role_free_joint_nte_20261027'
BASE_SEED=945220261027
SEEDS=(17,29,43)
OPPONENTS=('fixed_D_kernel','rootpuct')
METHODS=('IG','AG','U')+tuple(f'{name}{s}' for name in ('SINGLE','FACT','JOINT','DIRECT','VALUE_U','PRIOR') for s in SEEDS)
SMOKE=False
WORKERS=32
CPUS=[]
PARENT=None
WALL_CAP=12*3600
SOURCE_NAMES=('run_e45b_role_free.py','learning/e45b_joint_network.py','solvers/e45b_joint_nte.py',
    'e45b_scenarios.py','tests/test_e45b_joint_nte.py','E45B_ROLE_FREE_PROTOCOL_20261027.md',
    'e45b_sensor_audit.py','tests/test_e45b_sensor_audit.py','tests/fixtures/e45b_sensor_boundary_v1.npz')


def read(path):
    import json
    return json.loads(Path(path).read_text())


def emit(**values):
    import json
    r=dict(pid=os.getpid(),timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**values)
    save(OUT/'status.json',r)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(json.dumps(r,ensure_ascii=False),flush=True)


def worker_status(**values):
    save(OUT/'workers'/f'{os.getpid()}.json',dict(pid=os.getpid(),timestamp=time.time(),cpu=list(os.sched_getaffinity(0)),**values))


def entries(split):return [e for e in read(OUT/'bank/manifest.json')['entries'] if e['split']==split]
def model_file(kind,seed,round_,suffix='.npz'):
    return OUT/('initial_models' if round_==0 else f'round_{round_:02d}/models')/f'{kind}_{seed}{suffix}'


def prepare():
    if (OUT/'experiment.json').exists():return verify()
    OUT.mkdir(parents=True,exist_ok=True);rows=[];known=set()
    for old in (ROOT/'current/e44_branch_conditioned_nte_20261027/bank').rglob('controller_inputs/*.json'):
        known.add(hashlib.sha256(str(read(old)['map']).encode()).hexdigest())
    namespace=int(hashlib.sha256(OUT.name.encode()).hexdigest()[:8],16) if SMOKE else 0
    for split,n in (('train',1 if SMOKE else 8),('validation',1 if SMOKE else 2),('test',1 if SMOKE else 6)):
        for family in ('long_wall','wall_corner','multi_route'):
            for i in range(n):
                case=make_case(family,i,split,BASE_SEED+namespace);digest=hashlib.sha256(str(case['map']).encode()).hexdigest()
                if digest in known:raise RuntimeError('Map overlap')
                known.add(digest);path=OUT/'bank'/split/(case['id']+'.json');save(path,case)
                rows.append(dict(id=case['id'],path=str(path),family=family,pattern=case['pattern'],split=split))
    save(OUT/'bank/manifest.json',dict(entries=rows,selection='geometry and causal history only; no algorithm outcomes'))
    older=read(ROOT/'current/e45_2v2_pilot_20261027/experiment.json')
    protected=older['source_hashes']|older['protected_hashes']
    for name in ('experiment.json','decision.json','results.json','PROTOCOL.md'):
        for folder in ('e44_branch_conditioned_nte_20261027','e45_2v2_pilot_20261027'):
            p=ROOT/'current'/folder/name
            if p.exists():protected[str(p)]=sha(p)
    architecture={k:dict(width=Network(k).width,parameters=sum(p.numel() for p in Network(k).parameters())) for k in KINDS}
    config=dict(experiment='E45-B',role_free=True,manual_assignment=False,manual_role_labels=False,geometry_teacher=False,
        train_cases=3 if SMOKE else 24,validation_cases=3 if SMOKE else 6,test_cases=3 if SMOKE else 18,
        rounds=2 if SMOKE else 12,epochs_per_round=2 if SMOKE else 30,seeds=list(SEEDS),network_kinds=list(KINDS),architecture=architecture,
        methods=list(METHODS),opponents=list(OPPONENTS),evaluation_games=126 if SMOKE else 1512,
        train_decision_ms=400.,online_decision_ms=200.,max_states_per_episode=2 if SMOKE else 32,
        dataset='joint actor-generated replay per seed, last three rounds; SAME data/labels for all three policy families',
        actor_curriculum='first four rounds fixed root D kernel, then balanced fixed-D and full-state RootPUCT3000 pool; validation always both',
        learning='new joint-policy/team-value training; only pursuer team updated in this stage, not two-sided self-play',
        true_objective='terminal double capture reward1 minus .0025 per physical step; no individual role or target reward',
        potential=dict(eta=ETA,phi='negative sum over active targets of posterior expected nearest-pursuer obstacle path distance, each clipped80',
            terminal_phi=0.,full_return_invariance='shaped total = true team utility - eta*initial_phi',assignment=False,
            scope='explicit scalar geometry shaping for sparse-reward bootstrap, not zero-domain-prior learning'),
        tree=dict(horizon=8,selection='UCT c=.1',progressive_widening='min(16,max(2,ceil(2*(N+1)^.35)))',
            proposal='75% learned joint/factorized density +25% role-free uniform; uniform has 50% max-axis projection',
            observation_branch='joint action index + both public target detections + active masks',
            belief='factorized fixed-root exact grid marginals conditioned on all active teammates; 64-weighted public exports'),
        opponent_model_scope='tree uses fixed D motion kernels; adaptive RootPUCT mismatch remains, learned team value uses actual episode returns',
        timing='construction and each scripted history-frame update reported separately; persistent online act incl current belief/model/features/search/return has200ms wall cap',
        performance_gate=False,complete_rounds_and_test_regardless_of_efficacy=True,smoke=SMOKE,wall_cap_seconds=WALL_CAP,
        statistics='18 public-history grouped paired bootstrap10000;12 simultaneous utility contrasts (6 x2opponents); all worlds/seeds retained',
        comparisons=['JOINT-AG','JOINT-FACT','JOINT-SINGLE','JOINT-DIRECT','JOINT-VALUE_U','JOINT-PRIOR'],
        source_hashes={str(ROOT/'code'/s):sha(ROOT/'code'/s) for s in SOURCE_NAMES},protected_hashes=protected,
        bank_hashes={str(p):sha(p) for p in (OUT/'bank').rglob('*.json')})
    save(OUT/'experiment.json',config)
    (OUT/'PROTOCOL.md').write_bytes((ROOT/'code/E45B_ROLE_FREE_PROTOCOL_20261027.md').read_bytes())
    for s in SOURCE_NAMES:
        p=OUT/'source_snapshot'/s;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/'code'/s).read_bytes())
    emit(stage='registered',rounds=config['rounds'],evaluation_games=config['evaluation_games'])
    report('协议已冻结，工程验收与首次数据采集待执行。')
    return config


def verify():
    c=read(OUT/'experiment.json')
    amendment=OUT/'REPAIR_SENSOR_AUDIT_20261027.json'
    replacements={}
    if amendment.exists():
        repair=read(amendment)
        assert repair['original_experiment_sha256']==sha(OUT/'experiment.json')
        for p,item in repair['overrides'].items():
            assert c['source_hashes'][p]==item['original_sha256']
            replacements[p]=item['repaired_sha256']
        for p,h in repair['extra_hashes'].items():
            if sha(p)!=h:raise RuntimeError('Frozen repair artifact changed: '+p)
    for section in ('source_hashes','protected_hashes','bank_hashes'):
        for p,h in c[section].items():
            if section=='source_hashes':h=replacements.get(p,h)
            if sha(p)!=h:raise RuntimeError('Frozen artifact changed: '+p)
    for p in sorted(OUT.glob('round_*/models_manifest.json')):
        for path,h in read(p)['files'].items():
            if sha(path)!=h:raise RuntimeError('Frozen round model changed: '+path)
    return c


def initialize_models():
    folder=OUT/'initial_models';folder.mkdir(exist_ok=True)
    for seed in SEEDS:
        for kind in KINDS:
            path=model_file(kind,seed,0)
            if path.exists():continue
            torch.manual_seed(seed);m=Network(kind);m.export(path);torch.save(m.state_dict(),model_file(kind,seed,0,'.pt'))
    if not (folder/'manifest.json').exists():save(folder/'manifest.json',dict(random_only=True,files={str(p):sha(p) for p in folder.iterdir() if p.is_file()}))
    for p,h in read(folder/'manifest.json')['files'].items():assert sha(p)==h


def make_controller(case,task):
    m=task['method'];seed=task['seed'];budget=task.get('budget_ms',200.)
    if m in ('IG','AG'):return TeamController(case['map'],case['speed'],seed,m,budget_ms=budget)
    if m=='U':return Controller(case['map'],case['speed'],seed,budget_ms=budget)
    family=next(k for k in ('VALUE_U','SINGLE','FACT','JOINT','DIRECT','PRIOR') if m.startswith(k))
    train_seed=int(m[len(family):]);kind=dict(SINGLE='single',FACT='factorized',JOINT='joint',DIRECT='joint',VALUE_U='joint',PRIOR='joint')[family]
    return Controller(case['map'],case['speed'],seed,model_file(kind,train_seed,task['model_round']),
        use_policy=family!='VALUE_U',use_value=family!='PRIOR',direct=family=='DIRECT',budget_ms=budget)


def request(entry,world,opponent,method,stage,round_=0,model_round=0):
    seed=int(np.random.SeedSequence([BASE_SEED,int(sha(entry['path'])[:8],16),int(world=='BA')]).generate_state(1)[0])
    return dict(key=f'{entry["id"]}_{world}_{opponent}_{method}',case=entry['path'],group=entry['id'],family=entry['family'],pattern=entry['pattern'],
        split=entry['split'],world=world,opponent=opponent,method=method,seed=seed,stage=stage,round=round_,model_round=model_round,
        budget_ms=400. if stage.startswith('collection') else 200.)


def episode(task):
    case=read(task['case']);boxes=np.asarray(case['map']);p=original_model(boxes,case['speed'])
    started=time.perf_counter();own=make_controller(case,task);construct_ms=1000*(time.perf_counter()-started);history_ms=[]
    for obs in case['history'][:-1]:
        stamp=time.perf_counter();own.observe(obs);history_ms.append(1000*(time.perf_counter()-stamp))
    state=np.asarray(next(w['initial'] for w in case['worlds'] if w['id']==task['world'])).reshape(-1,1)
    if task['opponent']=='fixed_D_kernel':
        other=FixedKernelTargets(case,task['seed']);state[:4,0]=other.positions().ravel()
    else:
        other=RootPUCT([None]*4,None,3000,0.,0.);other.seeded_state(p,task['seed']+17,task['seed']+29)
    remaining=80-int(state[8,0]);limit=task.get('limit',3 if SMOKE else remaining);trace=[];labels=[];first=None
    for step in range(min(remaining,limit)):
        ap=state[p.active_idxs[2:],0]>.5;ae=state[p.active_idxs[:2],0]>.5
        obs=public_observation(int(state[8,0]),state[4:8,0].reshape(2,2),state[:4,0].reshape(2,2),ap,ae,boxes)
        for j in range(2):
            for i in range(2):
                a=state[4+2*i:6+2*i,0];q=state[2*j:2*j+2,0]
                seen=bool(ap[i] and ae[j] and visible_batch(a[None],q[None],boxes)[0])
                assert seen==(obs['reports'][j][i] is not None)
        wall=time.perf_counter();cpu=time.process_time()
        action=own.act(obs,fixed_simulations=task.get('fixed',4 if SMOKE else None))
        latency=1000*(time.perf_counter()-wall);cpu_ms=1000*(time.process_time()-cpu)
        for i in range(2):
            assert legal_action(state[4+2*i:6+2*i,0],action[i],case['speed'],boxes)
            if not ap[i]:assert np.max(abs(action[i]))==0
        simulated=np.concatenate(own.traces) if isinstance(own,Controller) and own.traces else np.zeros((0,26))
        errors=audit_simulated(simulated,boxes,case['speed'])
        if np.any(errors):
            evidence=OUT/'audit_failures'/f'{task["key"]}_r{task["round"]}_s{step}_{os.getpid()}'
            evidence.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(str(evidence)+'.npz',trace=simulated,boxes=boxes,speed=case['speed'],errors=errors)
            save(evidence.with_suffix('.json'),dict(task=task,step=step,observation=obs,stats=own.last_stats))
            raise RuntimeError('Independent joint simulation audit failed: '+str(errors)+'; evidence='+str(evidence))
        if task['opponent']=='fixed_D_kernel':raw=other.actions(ae)
        else:raw=np.asarray(other.search(p,state.copy(),turn=0).best_action).reshape(-1)[:4].reshape(2,2)
        enemy=raw.copy();shield=0
        for j in range(2):
            if not ae[j]:enemy[j]=0
            elif not legal_action(state[2*j:2*j+2,0],enemy[j],1.,boxes):
                if task['opponent']=='fixed_D_kernel':raise RuntimeError('Illegal fixed-model target transition')
                enemy[j]=0;shield+=1
        joint=np.r_[enemy.ravel(),action.ravel()][:,None];nxt=p.step(state,joint,1.)
        assert p.is_valid(nxt)
        np.testing.assert_allclose(nxt[:8],state[:8]+joint,atol=1e-10,rtol=0)
        nap,nae,_=simultaneous_capture(nxt[4:8,0].reshape(2,2),nxt[:4,0].reshape(2,2),ap,ae,boxes)
        np.testing.assert_array_equal(nap,nxt[p.active_idxs[2:],0]>.5);np.testing.assert_array_equal(nae,nxt[p.active_idxs[:2],0]>.5)
        complete=not nae.any();reward=float(complete)-TIME_COST
        if ae.sum()>nae.sum() and first is None:first=step+1
        if task['stage'].startswith('collection'):
            root=own.root;la=np.zeros((16,4));lw=np.zeros(16)
            keep=np.flatnonzero(np.asarray(root.visits)>0)
            if len(keep):
                la[:len(keep)]=np.asarray(root.actions)[keep].reshape(-1,4)/case['speed']
                lw[:len(keep)]=np.asarray(root.visits)[keep];lw/=lw.sum();source='completed_joint_search'
            else:la[0]=action.ravel()/case['speed'];lw[0]=1.;source='executed_fallback_no_search_claim'
            phi=root.phi
            if phi is None:
                phi=coverage_potential(root.positions,root.ap,root.ae,root.samples,root.weights,own.public.speed,own.public.vertices,own.public.dist,boxes)
            labels.append(dict(**own.features(root),label_actions=la,label_weights=lw,phi=phi,label_source=source))
        trace.append(dict(observation=obs,state=state[:,0].tolist(),next_state=nxt[:,0].tolist(),action=action.tolist(),
            opponent_action=enemy.tolist(),opponent_shields=shield,stats=own.last_stats,latency_ms=latency,cpu_ms=cpu_ms,
            reward=reward,new_captures=int(ae.sum()-nae.sum()),remaining_targets=int(nae.sum()),
            belief_signature=own.belief.signature(),simulated_transitions=len(simulated),simulation_errors=errors.tolist(),
            simulation_sha256=hashlib.sha256(simulated.tobytes()).hexdigest()))
        state=nxt
        if p.is_terminal(state):break
    # Full public replay is deliberately a scoring audit outside online return.
    belief=TeamBelief(boxes,task['seed'])
    for obs in case['history'][:-1]:belief.update(obs)
    for frame in trace:
        belief.update(frame['observation']);assert belief.signature()==frame['belief_signature']
    complete=not (state[p.active_idxs[:2],0]>.5).any();lat=[r['latency_ms'] for r in trace]
    row=dict(**task,double_capture=bool(complete),completion_cost=len(trace) if complete else remaining,
        utility=sum(r['reward'] for r in trace),first_capture=first,captures=2-int((state[p.active_idxs[:2],0]>.5).sum()),
        decisions=len(trace),latencies_ms=lat,construction_ms=construct_ms,history_ms=history_ms,
        online_overruns=sum(x>task['budget_ms'] for x in lat),history_overruns=sum(x>200 for x in history_ms),
        cpu_ms=[r['cpu_ms'] for r in trace],hidden_simulations=[r['stats'].get('simulations',0) for r in trace if all(all(q is None for q in z) for z in r['observation']['reports'])],
        failure=None,reason='double_capture' if complete else 'engineering_prefix' if len(trace)<remaining else 'timeout',
        simulated_transitions=sum(r['simulated_transitions'] for r in trace),public_replay_exact=True)
    arrays=None
    if labels:
        returns=np.cumsum([r['reward'] for r in trace][::-1])[::-1]
        for i,label in enumerate(labels):label['label_value']=returns[i]-ETA*label['phi']
        cap=read(OUT/'experiment.json')['max_states_per_episode'];chosen=np.unique(np.linspace(0,len(labels)-1,min(cap,len(labels))).astype(int))
        arrays={k:np.stack([labels[i][k] for i in chosen]) for k in KEYS+('label_actions','label_weights','label_value')}
        row['label_states']=len(chosen);row['search_label_states']=sum(labels[i]['label_source']=='completed_joint_search' for i in chosen)
    return dict(summary=row,trace=trace),arrays


def paths(task):
    folder=OUT/(f'round_{task["round"]:02d}' if task['stage'].startswith('collection') else '')/task['stage']
    return folder/'episodes'/(task['key']+'.json'),folder/'cache'/(task['key']+'.json'),folder/'labels'/(task['key']+'.npz')


def cached(task):
    path,cache,data=paths(task)
    if not cache.exists():return None
    row=read(cache)
    assert sha(path)==row['episode_sha256']
    if 'data_sha256' in row:assert sha(data)==row['data_sha256']
    return row


def episode_job(task):
    if PARENT is not None and os.getppid()!=PARENT:raise RuntimeError('Coordinator exited')
    row=cached(task)
    if row is not None:return row
    path,cache,data=paths(task);failure=cache.with_name(cache.stem+'.failure.json')
    if failure.exists():raise RuntimeError('Prior execution failure requires repair: '+str(failure))
    try:payload,arrays=episode(task)
    except BaseException as exc:save(failure,dict(task=task,error=repr(exc)));raise
    save(path,payload);row=payload['summary']|dict(episode=str(path),episode_sha256=sha(path))
    if arrays is not None:
        data.parent.mkdir(parents=True,exist_ok=True);temp=data.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(data)
        row.update(data=str(data),data_sha256=sha(data))
    save(cache,row);worker_status(stage=task['stage'],key=task['key']);return row


def cpu_init(output,smoke,queue,parent,model_round):
    global OUT,SMOKE,PARENT
    OUT=Path(output);SMOKE=smoke;PARENT=parent
    os.sched_setaffinity(0,{queue.get()});torch.set_num_threads(1);verify()
    for entry in [next(e for e in entries('validation') if e['family']==f) for f in ('long_wall','wall_corner','multi_route')]:
        case=read(entry['path'])
        task=dict(method='JOINT17',seed=317,model_round=model_round)
        own=make_controller(case,task)
        for obs in case['history'][:-1]:own.observe(obs)
        own.act(case['history'][-1],fixed_simulations=2)
        audit_simulated(np.zeros((0,26)),np.asarray(case['map']),case['speed'])
    worker_status(stage='cpu_ready',model_round=model_round)


def train_init(output,smoke,queue,parent):
    global OUT,SMOKE,PARENT,DEVICE
    OUT=Path(output);SMOKE=smoke;PARENT=parent;torch.set_num_threads(1)
    gpu=queue.get();DEVICE=f'cuda:{gpu}' if torch.cuda.is_available() and gpu>=0 else 'cpu'
    if DEVICE!='cpu':torch.cuda.set_device(gpu)
    torch.use_deterministic_algorithms(True);verify();worker_status(stage='trainer_ready',device=DEVICE)


def run_pool(stage,jobs,round_=0,training=False,workers=None):
    if not jobs:return []
    count=min(2 if training else WORKERS if workers is None else workers,len(jobs))
    ctx=mp.get_context('spawn');queue=ctx.Queue()
    for value in ([2,3][:count] if training else CPUS[:count]):queue.put(value)
    emit(stage=stage,round=round_,completed=0,total=len(jobs),workers=count,devices=['cuda:2','cuda:3'][:count] if training else None)
    pool=futures.ProcessPoolExecutor(max_workers=count,mp_context=ctx,
        initializer=train_init if training else cpu_init,
        initargs=(str(OUT),SMOKE,queue,os.getpid()) if training else (str(OUT),SMOKE,queue,os.getpid(),jobs[0].get('model_round',max(0,round_-1))))
    result=[]
    try:
        pending=[pool.submit(train_job if training else episode_job,job) for job in jobs]
        for f in futures.as_completed(pending):
            result.append(f.result());emit(stage=stage,round=round_,completed=len(result),total=len(jobs),workers=count)
        pool.shutdown(wait=True)
    except BaseException:
        processes=list((pool._processes or {}).values());pool.shutdown(wait=False,cancel_futures=True)
        for p in processes:
            if p.is_alive():p.terminate()
        for p in processes:p.join(timeout=5)
        raise
    finally:queue.close()
    return result


def collect(round_):
    path=OUT/f'round_{round_:02d}/collection_results.json'
    if path.exists():
        rows=read(path)
        for r in rows:
            assert sha(r['episode'])==r['episode_sha256'] and sha(r['data'])==r['data_sha256']
        return rows
    jobs=[]
    for split in ('train','validation'):
        for i,entry in enumerate(entries(split)):
            for s,seed in enumerate(SEEDS):
                opponent='fixed_D_kernel' if split=='train' and round_<=4 else OPPONENTS[(i+s+(round_ if split=='train' else 0))%2]
                world=('AB','BA')[(i+s+round_)%2]
                jobs.append(request(entry,world,opponent,f'JOINT{seed}',f'collection/{split}',round_,round_-1))
    rows=run_pool('collect',jobs,round_);rows.sort(key=lambda r:r['key']);save(path,rows)
    return rows


def data_for(seed,round_,split,device):
    rounds=range(max(1,round_-2),round_+1) if split=='train' else [round_]
    rows=[row for r in rounds for row in read(OUT/f'round_{r:02d}/collection_results.json')
        if row['split']==split and row['method']==f'JOINT{seed}']
    paths_=sorted(r['data'] for r in rows)
    if not paths_:raise RuntimeError('No complete training data')
    for r in rows:assert sha(r['data'])==r['data_sha256']
    arrays=[]
    for path in paths_:
        with np.load(path,allow_pickle=False) as data:arrays.append({k:data[k].copy() for k in data.files})
    tensors={k:torch.from_numpy(np.concatenate([d[k] for d in arrays])).float().to(device) for k in arrays[0]}
    return tensors,dict(files={r['data']:r['data_sha256'] for r in rows},states=len(tensors['public']))


def train_job(task):
    if PARENT is not None and os.getppid()!=PARENT:raise RuntimeError('Coordinator exited')
    kind,seed,round_=task['kind'],task['seed'],task['round']
    target=OUT/f'round_{round_:02d}/training/{kind}_{seed}.json'
    if target.exists():
        r=read(target);assert sha(model_file(kind,seed,round_))==r['model_sha256'];return r
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=Network(kind).to(DEVICE)
    model.load_state_dict(torch.load(model_file(kind,seed,round_-1,'.pt'),weights_only=True,map_location=DEVICE))
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=1e-5)
    train,train_manifest=data_for(seed,round_,'train',DEVICE);val,val_manifest=data_for(seed,round_,'validation',DEVICE)
    rng=np.random.default_rng(seed+1009*round_);epochs=read(OUT/'experiment.json')['epochs_per_round'];begin=1;logs=[];best=float('inf');selected=None
    resume=target.with_suffix('.resume.pt');resume.parent.mkdir(parents=True,exist_ok=True)
    model_file(kind,seed,round_).parent.mkdir(parents=True,exist_ok=True)
    if resume.exists():
        record=torch.load(resume,weights_only=False,map_location=DEVICE)
        model.load_state_dict(record['model']);optimizer.load_state_dict(record['optimizer'])
        rng.bit_generator.state=record['rng'];begin=record['epoch']+1;logs=record['logs'];best=record['best'];selected=record['selected']
    started=time.perf_counter()
    for epoch in range(begin,epochs+1):
        model.train();total=np.zeros(3);n=0
        for ids in np.array_split(rng.permutation(len(train['public'])),max(1,int(np.ceil(len(train['public'])/128)))):
            ids=torch.as_tensor(ids,device=DEVICE);batch={k:v[ids] for k,v in train.items()}
            batch=augment(batch,bool(rng.integers(2)),bool(rng.integers(2)))
            optimizer.zero_grad();loss,nll,vl=losses(model,batch)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite joint training objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
            total+=np.array([float(loss.detach()),float(nll.detach()),float(vl.detach())])*len(ids);n+=len(ids)
        model.eval()
        with torch.no_grad():value,vn,vv=[float(x) for x in losses(model,val)]
        if not np.isfinite(value):raise RuntimeError('Nonfinite validation objective')
        logs.append(dict(epoch=epoch,train_loss=float(total[0]/n),train_NLL=float(total[1]/n),train_value_Huber=float(total[2]/n),
            validation_loss=value,validation_NLL=vn,validation_value_Huber=vv))
        if value<best:
            best,selected=value,epoch;model.export(model_file(kind,seed,round_))
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},model_file(kind,seed,round_,'.pt'))
        payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,
            epoch=epoch,logs=logs,best=best,selected=selected)
        torch.save(payload,resume.with_suffix('.tmp'));resume.with_suffix('.tmp').replace(resume)
        worker_status(stage='training',kind=kind,seed=seed,round=round_,epoch=epoch,epochs=epochs,device=DEVICE)
    record=dict(kind=kind,seed=seed,round=round_,epochs=epochs,selected_epoch=selected,
        best_validation_loss=best,log=logs,train_manifest=train_manifest,validation_manifest=val_manifest,
        parameters=sum(p.numel() for p in model.parameters()),device=DEVICE,seconds=time.perf_counter()-started,
        model_sha256=sha(model_file(kind,seed,round_)),no_assignment_labels=True,test_outcomes_read=False)
    save(target,record);return record


def train_round(round_):
    manifest=OUT/f'round_{round_:02d}/models_manifest.json'
    if manifest.exists():verify();return read(OUT/f'round_{round_:02d}/training.json')
    records=run_pool('training',[dict(kind=k,seed=s,round=round_) for s in SEEDS for k in KINDS],round_,True)
    records.sort(key=lambda r:(r['seed'],r['kind']))
    assert len(records)==9 and all(r['epochs']==read(OUT/'experiment.json')['epochs_per_round'] for r in records)
    save(OUT/f'round_{round_:02d}/training.json',dict(completed=True,records=records,round=round_))
    save(manifest,dict(files={str(p):sha(p) for p in (OUT/f'round_{round_:02d}/models').iterdir() if p.is_file()},
        selected_using='validation policy NLL + .5 value Huber within fixed epochs; no test outcomes'))
    report(f'已完成第{round_}轮数据采集与九个策略/价值网络更新。')
    return records


def preflight():
    path=OUT/'interface.json'
    if path.exists():return read(path)
    records=[]
    for family in ('long_wall','wall_corner','multi_route'):
        entry=next(e for e in entries('validation') if e['family']==family)
        for method in ('U','SINGLE17','FACT17','JOINT17','DIRECT17','VALUE_U17','PRIOR17'):
            for opponent in OPPONENTS:
                task=request(entry,'AB',opponent,method,'engineering',model_round=0)|dict(limit=2,fixed=4)
                records.append(episode_job(task))
        emit(stage='interface',family=family,prefixes=len(records))
    result=dict(passed=True,prefixes=len(records),random_initial_models_only=True,no_test_outcomes=True,
        public_replay=True,independent_joint_geometry_and_sensor_audit=True,performance_gate=False)
    save(path,result);return result


def timing_metrics(rows):
    wall=sum(sum(r['latencies_ms']) for r in rows);cpu=sum(sum(r['cpu_ms']) for r in rows)
    lat=[t for r in rows for t in r['latencies_ms']]
    return dict(cpu_wall=cpu/max(wall,1e-9),overrun_rate=sum(t>200 for t in lat)/len(lat),p95_ms=float(np.percentile(lat,95)),
        simulations=sum(sum(r['hidden_simulations']) for r in rows))


def calibrate(final_round):
    path=OUT/'timing/decision.json'
    if path.exists():return read(path)['workers']
    if SMOKE:save(path,dict(workers=min(2,WORKERS),smoke=True,performance_conclusion=False));return min(2,WORKERS)
    selected=[next(e for e in entries('validation') if e['family']==f) for f in ('long_wall','wall_corner','multi_route')]
    methods=('AG','U','JOINT17','JOINT29','JOINT43','VALUE_U17')
    def jobs(stage):return [request(e,'AB',o,m,stage,model_round=final_round)|dict(limit=2) for e in selected for o in OPPONENTS for m in methods]
    serial=[]
    for i,task in enumerate(jobs('calibration/serial')):
        serial.append(episode_job(task));emit(stage='timing_serial',completed=i+1,total=36)
    reference=timing_metrics(serial);candidates=[]
    for n in sorted({WORKERS}|{i for i in (32,24,16,8,4) if i<WORKERS},reverse=True):
        if n==1:break
        rr=run_pool('calibration',jobs(f'calibration/parallel_{n}'),workers=n)
        metrics=timing_metrics(rr);ratio=metrics['simulations']/max(1,reference['simulations'])
        accepted=bool(metrics['cpu_wall']>=.85 and metrics['overrun_rate']<=reference['overrun_rate']+.002 and ratio>=.75)
        candidates.append(dict(workers=n,metrics=metrics,simulation_ratio=ratio,accepted=accepted))
        save(OUT/'timing/calibration.json',dict(reference=reference,candidates=candidates,validation_only=True))
        if accepted:
            save(path,dict(workers=n,reference=reference,candidates=candidates,test_outcomes_read=False));return n
    save(path,dict(workers=1,reference=reference,candidates=candidates,test_outcomes_read=False));return 1


def statistics(rows):
    groups=sorted({r['group'] for r in rows});comparators=('AG','FACT','SINGLE','DIRECT','VALUE_U','PRIOR')
    def methods(name):return [name] if name in ('AG','IG','U') else [f'{name}{s}' for s in SEEDS]
    def mean(group,opp,arm,key):return float(np.mean([r[key] for r in rows if r['group']==group and r['opponent']==opp and r['method'] in methods(arm)]))
    pairs=[(o,b) for o in OPPONENTS for b in comparators]
    delta=np.array([[mean(g,o,'JOINT','utility')-mean(g,o,b,'utility') for o,b in pairs] for g in groups])
    rng=np.random.default_rng(BASE_SEED+881);bootstrap=delta[rng.integers(len(groups),size=(10000,len(groups)))].mean(1)
    center=delta.mean(0);radius=float(np.quantile(np.max(np.abs(bootstrap-center),axis=1),.95));comparisons=[]
    for j,(opp,baseline) in enumerate(pairs):
        success=float(np.mean([mean(g,opp,'JOINT','double_capture')-mean(g,opp,baseline,'double_capture') for g in groups]))
        cost=float(np.mean([mean(g,opp,'JOINT','completion_cost')-mean(g,opp,baseline,'completion_cost') for g in groups]))
        involved=[r for r in rows if r['opponent']==opp and r['method'] in methods('JOINT')+methods(baseline)]
        overruns=sum(r['online_overruns']+r['history_overruns'] for r in involved)
        interval=[float(center[j]-radius),float(center[j]+radius)]
        per_seed=[]
        for seed in SEEDS:
            bb=baseline if baseline=='AG' else f'{baseline}{seed}'
            per_seed.append(float(np.mean([np.mean([r['utility'] for r in rows if r['group']==g and r['opponent']==opp and r['method']==f'JOINT{seed}'])
                -np.mean([r['utility'] for r in rows if r['group']==g and r['opponent']==opp and r['method']==bb]) for g in groups])))
        reliable=bool(interval[0]>0 and center[j]>=.01 and success>=0 and cost<=-1 and min(per_seed)>0 and overruns==0)
        comparisons.append(dict(opponent=opp,comparison='JOINT-'+baseline,utility_difference=float(center[j]),simultaneous95=interval,
            completion_cost_difference=cost,double_capture_difference=success,per_seed_utility=per_seed,overruns=overruns,reliable_gain=reliable))
    arms={}
    for opponent in OPPONENTS:
        arms[opponent]={}
        for method in METHODS:
            rr=[r for r in rows if r['opponent']==opponent and r['method']==method];lat=[x for r in rr for x in r['latencies_ms']]
            arms[opponent][method]=dict(games=len(rr),double_captures=sum(r['double_capture'] for r in rr),
                cost=float(np.mean([r['completion_cost'] for r in rr])),utility=float(np.mean([r['utility'] for r in rr])),
                p95_ms=float(np.percentile(lat,95)),maximum_ms=max(lat),online_overruns=sum(r['online_overruns'] for r in rr),
                history_overruns=sum(r['history_overruns'] for r in rr),startup_max_ms=max(r['construction_ms'] for r in rr))
    result=dict(games=len(rows),groups=len(groups),arms=arms,comparisons=comparisons,
        scope='role-free joint policy iteration against fixed opponent pool; not two-sided self-play or a global optimum proof',
        direction='analyze_full_results_no_automatic_retraining',smoke=SMOKE)
    save(OUT/'evaluation/statistics.json',result);return result


def evaluate(final_round):
    n=calibrate(final_round);jobs=[request(e,w,o,m,'evaluation',model_round=final_round)
        for e in entries('test') for w in (('AB',) if SMOKE else ('AB','BA')) for o in OPPONENTS for m in METHODS]
    assert len(jobs)==read(OUT/'experiment.json')['evaluation_games']
    save(OUT/'evaluation/requests.json',jobs)
    if n==1:
        rows=[]
        for i,t in enumerate(jobs):rows.append(episode_job(t));emit(stage='evaluation',completed=i+1,total=len(jobs),workers=1)
    else:rows=run_pool('evaluation',jobs,workers=n)
    rows.sort(key=lambda r:r['key']);save(OUT/'evaluation/results.json',rows)
    return statistics(rows)


def report(state):
    c=read(OUT/'experiment.json') if (OUT/'experiment.json').exists() else {}
    lines=['# E45-B：无预设分工的局部信念联合NTE','',state,'',
        '主方法直接输出两车四维连续动作；无任务分配器、无角色标签、无G动作教师。AG/IG仅用于评测基线。',
        '共享合法局部观测；原捕获双方退场规则；仅追捕团队在本阶段迭代学习，对手为冻结运动核与RootPUCT池。',
        '采用显式公共团队距离势函数处理稀疏奖励，终端清零使完整任务排序不变；不是无几何先验的学习。','',
        f'计划 {c.get("rounds",12)} 轮 × 三类网络 × 三种子，每轮各 {c.get("epochs_per_round",30)} epoch；最终 {c.get("evaluation_games",1512)} 局。',
        '', '- [冻结协议](PROTOCOL.md)','- [配置与摘要](experiment.json)','- [实时进度](status.json)',
        '- [资源](execution.json)','- [时延校准](timing/decision.json)','- [最终统计](evaluation/statistics.json)','',
        '|轮次|完整对局|双捕局数|监督状态|九模型更新|','|---|---:|---:|---:|---|']
    for r in range(1,c.get('rounds',12)+1):
        p=OUT/f'round_{r:02d}/collection_results.json'
        if p.exists():
            rows=read(p);done=(OUT/f'round_{r:02d}/models_manifest.json').exists()
            lines.append(f'|{r}|{len(rows)}|{sum(x["double_capture"] for x in rows)}|{sum(x.get("label_states",0) for x in rows)}|{"9/9" if done else "进行中"}|')
    if (OUT/'evaluation/statistics.json').exists():
        s=read(OUT/'evaluation/statistics.json');lines+=['',f'完整测试 {s["games"]} 局，独立公共历史 {s["groups"]} 组。','',
            '|对手|方法|双捕/局数|完成成本|团队效用|决策P95 ms|在线超时|','|---|---|---:|---:|---:|---:|---:|']
        for opp,arms in s['arms'].items():
            for m,v in arms.items():lines.append(f'|{opp}|{m}|{v["double_captures"]}/{v["games"]}|{v["cost"]:.2f}|{v["utility"]:.4f}|{v["p95_ms"]:.2f}|{v["online_overruns"]}|')
        lines+=['','|条件|比较|效用差（正为好）|12项同时95%区间|可靠收益|','|---|---|---:|---|---|']
        for r in s['comparisons']:lines.append(f'|{r["opponent"]}|{r["comparison"]}|{r["utility_difference"]:+.4f}|{r["simultaneous95"]}|{r["reliable_gain"]}|')
    lines+=['','启动/每帧历史吸收与持续在线控制分别完整报告。测试并发度由验证集计时校准决定，不冒称串行时延。',
        '高成功率或类似包抄的轨迹不单独证明自主协同；必须结合强AG、分解提议、同检查点直控、策略/价值消融及泛化结果判断。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


class BudgetExpired(BaseException):pass


def pipeline():
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=prepare();verify();clock=OUT/'clock.json'
    if not clock.exists():save(clock,dict(started_unix=time.time(),cap_seconds=WALL_CAP))
    started=read(clock)['started_unix'];remaining=int(WALL_CAP-(time.time()-started))
    if remaining<=0:raise RuntimeError('Registered wall budget exhausted')
    def alarm(*_):raise BudgetExpired('12h total budget exhausted')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(remaining)
    try:
        initialize_models();preflight()
        for r in range(1,c['rounds']+1):collect(r);train_round(r);verify()
        save(OUT/'models_frozen.json',dict(final_round=c['rounds'],trained_networks=9,
            actual_epochs_each=c['rounds']*c['epochs_per_round'],
            files={str(model_file(k,s,c['rounds'],ext)):sha(model_file(k,s,c['rounds'],ext)) for k in KINDS for s in SEEDS for ext in ('.npz','.pt')}))
        stats=evaluate(c['rounds']);verify()
        save(OUT/'decision.json',dict(status='complete',rounds=c['rounds'],final_models=9,evaluation_games=stats['games'],
            manual_assignment=False,two_sided_self_play=False,elapsed_seconds=time.time()-started,smoke=SMOKE))
        report('完整流水线已结束，按预注册比较解释结果。' if not SMOKE else '隔离工程smoke完成，不作方法收益结论。')
        emit(stage='complete',rounds=c['rounds'],models=9,evaluation_games=stats['games'],elapsed_seconds=time.time()-started)
    except BaseException as exc:
        import traceback
        save(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),previous_status=read(OUT/'status.json'),elapsed_seconds=time.time()-started))
        report('执行中断，现场已保留：'+repr(exc));emit(stage='budget_stop' if isinstance(exc,BudgetExpired) else 'execution_failure',error=repr(exc));raise
    finally:signal.alarm(0)


def main():
    global OUT,SMOKE,WORKERS,CPUS
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','pipeline','smoke','status'))
    parser.add_argument('--output',type=Path);parser.add_argument('--workers',type=int,default=32);args=parser.parse_args()
    if args.output:OUT=args.output.resolve()
    SMOKE=args.stage=='smoke';WORKERS=args.workers
    if WORKERS<1:parser.error('--workers must be positive')
    torch.set_num_threads(1)
    if args.stage=='prepare':prepare();return
    if args.stage=='status':print(read(OUT/'status.json'));return
    execution=OUT/'execution.json'
    if execution.exists():
        config=read(execution);CPUS=config['physical_cores'];assert config['workers']==WORKERS
    else:
        CPUS=choose_cpus(WORKERS)
        if len(CPUS)<WORKERS:raise RuntimeError('Not enough distinct physical cores')
        save(execution,dict(workers=WORKERS,physical_cores=CPUS,threads_per_worker=1,gpus=[2,3],pid=os.getpid(),
            timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),timing_scope='pinned concurrent workload after validation calibration'))
    pipeline()


if __name__=='__main__':main()
