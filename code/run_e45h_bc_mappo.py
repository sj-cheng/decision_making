"""E45-H: teacher BC initialization plus MAPPO, independent frozen experiment."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('TZ','Asia/Shanghai')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import collections
import concurrent.futures as futures
import fcntl
import hashlib
import json
import multiprocessing as mp
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from learning.e45g_ppo import Actor, Critic, Controller, TeamBelief, gae, latent_log_prob, install_belief, safe_action, execution_legal_action
from learning.e45b_joint_network import KEYS
from solvers.e45b_joint_nte import coverage_potential
import run_e45b_role_free as base
import run_e45f_generalization_benchmark as old
from e45c_capture_audit import audit_simulated
from run_e44_parallel import choose_cpus
from run_frozen_d_closed_loop import save, sha
from e45f_scenarios import ID_FAMILIES, OOD_FAMILIES, new_id_case, shifted_case, geometry_key
from e45g_training_acceleration import public_geometry_cache, parallel_collect
from learning.e45h_bc import fit_bc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'current/e45h_bc_mappo_20261031'
FROOT = ROOT / 'current/e45f_generalization_benchmark_20261029'
CROOT = ROOT / 'current/e45c_budget_scaling_20261028'
SEEDS = (17, 29, 43)
RL = ('BC_MAPPO',)
CHECKPOINT_ARMS = ('BC_MAPPO','BC_ONLY','MAPPO','IPPO')
GROOT = ROOT / 'current/e45g_rl_benchmark_20261030'
ARMS = ('BC_MAPPO','BC_ONLY','SINGLE_PRIOR','MAPPO','IPPO','SINGLE_DIRECT','AG','IG','U','GEOM_SEARCH')
LEARNED = ARMS[:6]
BASE_SEED = 945920261031
SOURCES = ('run_e45h_bc_mappo.py','learning/e45h_bc.py','tests/test_e45h_bc_mappo.py','E45H_BC_MAPPO_PROTOCOL_20261031.md')
ACTORS = {}
CAPTURE = None
WORKER = {}


def read(path):
    return json.loads(Path(path).read_text())


def emit(folder, stage, **values):
    row = dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'), pid=os.getpid(), stage=stage, **values)
    save(Path(folder) / 'status.json', row)
    with (Path(folder) / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


def tensor_batch(arrays, device):
    return {k: torch.as_tensor(arrays[k], dtype=torch.float32, device=device) for k in KEYS}


class RolloutController(Controller):
    def __init__(self, *args):
        super().__init__(*args)
        self.features_saved = []; self.raw = []; self.logp = []; self.active = []; self.phi = []

    def act(self, obs, fixed_simulations=None):
        action, features, raw, logp, _ = self.sample(obs)
        self.features_saved.append(features); self.raw.append(raw); self.logp.append(logp)
        self.active.append(np.asarray(obs['active_p'], np.float32))
        self.phi.append(float(coverage_potential(np.asarray(obs['positions']), np.asarray(obs['active_p']),
            np.asarray(obs['active_e']), np.asarray(features['particles'][:,:,:2] * 20 + features['public'][:2] * 10),
            features['weights'], self.public.speed, self.public.vertices, self.public.dist, self.public.boxes)))
        return action


class SafeWrapper:
    def __init__(self, own):
        self.own = own

    def __getattr__(self, key):
        return getattr(self.own, key)

    def observe(self, obs):
        return self.own.observe(obs)

    def act(self, obs, fixed_simulations=None):
        action = self.own.act(obs, fixed_simulations)
        result, changed = safe_action(np.asarray(obs['positions']), action / self.own.public.speed,
                                     obs['active_p'], self.own.public.speed, self.own.public.boxes)
        self.own.last_stats['outer_safety_shortened'] = changed
        return result


def load_actor(path):
    path = str(path)
    if path not in ACTORS:
        actor = Actor(); actor.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['actor'])
        actor.eval(); ACTORS[path] = actor
    return ACTORS[path]


def factory(case, task):
    global CAPTURE
    arm = task['method'].rstrip('0123456789')
    if arm in CHECKPOINT_ARMS:
        actor = WORKER.get('actor') if task['stage'] == 'training' else load_actor(task['checkpoint'])
        cls = RolloutController if task['stage'] == 'training' else Controller
        own = cls(case['map'], case['speed'], task['seed'], actor)
        if task['stage'] == 'training':
            CAPTURE = own
    else:
        own = old.factory(case, task)
    original_act = own.act
    def act(obs, fixed_simulations=None):
        action = original_act(obs, fixed_simulations)
        result, changed = safe_action(np.asarray(obs['positions']), action / own.public.speed,
                                     obs['active_p'], own.public.speed, own.public.boxes)
        own.last_stats['outer_safety_shortened'] = changed
        return result
    own.act = act
    # Keep the original class: the independent simulator audit dispatches on
    # the search-controller type and must see all tree transitions.
    return own


def configure(output):
    base.OUT = Path(output); base.SMOKE = False; base.make_controller = factory
    base.legal_action = execution_legal_action
    base.audit_simulated = audit_simulated
    old.OUT = OUT
    install_belief(Path(output) / 'belief_failures')
    # Fixed-kernel environment initialization keeps the historical target
    # generator. Only the pursuer/replay belief uses G's documented wrapper.


def verify():
    config = read(OUT / 'experiment.json')
    replacements = {}
    for name in ('REPAIR_BOUNDARY_20261030.json', 'EXECUTION_MULTIGPU_20261030.json', 'EXECUTION_SAMPLING_20261030.json', 'REPAIR_SEARCH_VALIDATION_20261030.json', 'SOURCE_FINALIZATION_20261031.json'):
        amendment = OUT / name
        if not amendment.exists():
            continue
        repair = read(amendment)
        if repair['original_experiment_sha256'] != sha(OUT / 'experiment.json'):
            raise RuntimeError('Source amendment registration changed: ' + name)
        for path, item in repair['overrides'].items():
            if replacements.get(path, config['source_hashes'].get(path)) != item['original_sha256']:
                raise RuntimeError('Source amendment chain mismatch: ' + path)
            replacements[path] = item['repaired_sha256']
        for path, digest in repair['extra_hashes'].items():
            if sha(path) != digest:
                raise RuntimeError('Boundary repair evidence changed: ' + path)
    for section in ('source_hashes', 'protected_hashes', 'bank_hashes'):
        for path, digest in config[section].items():
            if section == 'source_hashes':
                digest = replacements.get(path, digest)
            if sha(path) != digest:
                raise RuntimeError('Frozen E45-H artifact changed: ' + path)
    return config


def teacher_inventory(smoke):
    result={}
    for seed in SEEDS:
        splits={}
        reference=read(CROOT/f'learning/x20/round_12/training/single_{seed}.json')
        for split,rounds in [('train',(10,11,12)),('validation',(12,))]:
            rows=[row for round_ in rounds for row in read(CROOT/f'learning/x20/round_{round_:02d}/collection_results.json')
                  if row['split']==split and row['method']==f'JOINT{seed}']
            expected=reference['train_manifest' if split=='train' else 'validation_manifest']['files']
            if {row['data']:row['data_sha256'] for row in rows}!=expected:
                raise RuntimeError('Teacher lineage differs from frozen Single training replay')
            if smoke:
                rows=[next(row for row in rows if row['family']==family) for family in ID_FAMILIES]
            splits[split]=[dict(data=row['data'],data_sha256=row['data_sha256'],episode=row['episode'],episode_sha256=row['episode_sha256'],
                case=row['case'],split=split,group=row['group'],speed=read(row['case'])['speed']) for row in rows]
        if {row['group'] for row in splits['train']} & {row['group'] for row in splits['validation']}:
            raise RuntimeError('Teacher train/validation leakage')
        result[str(seed)]=splits
    return result


def prepare(smoke=False, steps=2_000_000):
    if (OUT / 'experiment.json').exists():
        return verify()
    OUT.mkdir(parents=True, exist_ok=True)
    inventory=teacher_inventory(smoke)
    save(OUT/'teacher_inventory.json',inventory)
    save(OUT/'sampling_execution.json',dict(geometry_cache=True,episode_parallel=False))
    protected = read(FROOT / 'experiment.json')['protected_hashes'].copy()
    import run_e45g_rl_benchmark as frozen_g
    frozen_g.verify()
    for p in (ROOT/'code/run_e45g_rl_benchmark.py',ROOT/'code/learning/e45g_ppo.py',ROOT/'code/e45g_training_acceleration.py',GROOT/'experiment.json',GROOT/'frozen_models.json',OUT/'teacher_inventory.json',OUT/'sampling_execution.json'):
        protected[str(p)]=sha(p)
    for model in read(GROOT/'frozen_models.json').values():protected[model['path']]=model['sha256']
    for relative in ('run_e45f_generalization_benchmark.py', 'run_e45f_capacity_compat.py',
                     'solvers/e45f_support_refinement.py', 'solvers/e45f_geometry_capacity.py', 'e45c_belief_diagnostics.py'):
        protected[str(ROOT / 'code' / relative)] = sha(ROOT / 'code' / relative)
    train_manifest = read(CROOT / 'bank/manifest.json')
    train = [r for r in train_manifest['entries'] if r['split'] == 'train']
    validation = [r for r in train_manifest['entries'] if r['split'] == 'validation']
    if smoke:
        train = [next(r for r in train if r['family'] == f) for f in ID_FAMILIES]
        validation = [next(r for r in validation if r['family'] == f) for f in ID_FAMILIES]
    known = set()
    for folder in (CROOT, FROOT, GROOT, ROOT / 'current/e45d_prior_correlation_20261029', ROOT / 'current/e45b_role_free_joint_nte_20261027'):
        for path in (folder / 'bank').glob('*/*.json'):
            packet = read(path)
            if 'map' in packet:
                known.add(geometry_key(packet['map']))
    entries = []; rejects = []
    for suite, families in [('benchmark', ID_FAMILIES), ('generalization', OOD_FAMILIES)]:
        for family in families:
            accepted = collections.Counter(); index = 0; target = 1 if smoke else 12
            while sum(accepted.values()) < target:
                if index > 1200:
                    raise RuntimeError('Scenario generation exhausted: ' + family)
                try:
                    builder = new_id_case if suite == 'benchmark' else shifted_case
                    case = builder(family, index, 'test', BASE_SEED + int(smoke) * 1_000_000)
                    cell = (case['pattern'], case['speed'], case['density']) if suite == 'generalization' else (case['pattern'], case['speed'])
                    if not smoke and accepted[cell] >= (1 if suite == 'generalization' else 2):
                        index += 1; continue
                    geometry = geometry_key(case['map'])
                    if geometry in known:
                        raise ValueError('Historical geometry duplicate')
                except (ValueError, AssertionError) as exc:
                    rejects.append(dict(family=family, index=index, reason=repr(exc))); index += 1; continue
                known.add(geometry); accepted[cell] += 1
                path = OUT / 'bank' / suite / (case['id'] + '.json'); save(path, case)
                entries.append(dict(id=case['id'], path=str(path), family=family, pattern=case['pattern'],
                    density=case['density'], suite=suite, split='test'))
                index += 1
    save(OUT / 'bank/manifest.json', dict(train=train, validation=validation, test=entries, rejections=rejects))
    for row in train + validation:
        protected[row['path']] = sha(row['path'])
    for seed in SEEDS:
        path = CROOT / f'learning/x20/round_12/models/single_{seed}.npz'
        protected[str(path)] = sha(path)
    config = dict(experiment='E45-H', smoke=smoke, seeds=list(SEEDS), algorithms=list(RL),
        arms=list(ARMS), train_maps=len(train), validation_maps=len(validation), test_groups=len(entries),
        training_steps_per_run=steps, training_runs=3, rollout_steps_per_worker=32 if smoke else 128,
        rollout_workers_per_run=2 if smoke else 16, epochs=3 if smoke else 5, bc_epochs=4 if smoke else 30, critic_warmup_updates=2 if smoke else 10, minibatch=512,
        learning_rate=3e-5, final_learning_rate=3e-6, clip=.2, max_grad_norm=.5,
        gae_lambda=.95, gamma=1., entropy_start=.001, entropy_final=.0001, potential_eta=.05,
        curriculum='balanced fixed_D_kernel/rootpuct3000 from first update after BC',
        validation_at=[steps] if smoke else [250_000, 500_000, 1_000_000, 1_500_000, steps],
        checkpoint_selection='BC step0 plus five MAPPO validation checkpoints; maximum raw utility, earliest on ties; full registered budget',
        actors='same public map/positions/64-weighted D/history packet; shared 2D actor with ego+ID; no truth or expert actions',
        critics='BC_MAPPO training-only true target coordinates; PopArt; initial critic-only warmup, no actor change',
        action='tanh Gaussian latent PPO; common deterministic target-free legal backtracking; no conditional rejection likelihood',
        reward='raw complete_double_capture-.0025/step +.05*(Phi_next-Phi); terminal Phi0; evaluate RAW utility',
        death_mask='actor losses only before retirement; team credit continues to global episode end, critic trained on both masks',
        sampling='complete episodes; no mid-episode rollout bootstrap; support failures archived explicitly, not scored as losses',
        evaluation='fresh84 histories disjoint C/B/D/F/G,2worlds,2opponents,22 arms; 7392 fresh episodes, no F supplement',
        primary='per suite BC_MAPPO minus nine families x2opponents;18 simultaneous95 group-bootstrap10000; BC_ONLY is primary fine-tuning control',
        paired_unit='public history; worlds/training seeds are within-group repeats',
        evaluation_budget_ms=200., protected_hashes=protected,
        source_hashes={str(ROOT/'code'/name):sha(ROOT/'code'/name) for name in SOURCES},
        bank_hashes={str(p):sha(p) for p in (OUT/'bank').rglob('*.json')})
    config['architecture'] = dict(actor_parameters=sum(p.numel() for p in Actor().parameters()),
        critic_parameters={kind:sum(p.numel() for p in Critic('MAPPO').parameters()) for kind in RL})
    save(OUT / 'experiment.json', config)
    (OUT / 'PROTOCOL.md').write_bytes((ROOT / 'code/E45H_BC_MAPPO_PROTOCOL_20261031.md').read_bytes())
    for name in SOURCES:
        dest = OUT / 'source_snapshot' / name; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/'code'/name).read_bytes())
    emit(OUT, 'registered', training_runs=3, steps_per_run=steps, fresh_test_tasks=len(entries)*88,
         training_maps=len(train), smoke=smoke)
    return config


def request(entry, world, opponent, arm, train_seed, stage, checkpoint=None, episode_seed=None):
    seed = int(np.random.SeedSequence([BASE_SEED, int(sha(entry['path'])[:8],16), int(world=='BA')]).generate_state(1)[0]) if episode_seed is None else int(episode_seed)
    checkpoint_sha256 = sha(checkpoint) if stage == 'validation' and checkpoint else None
    suffix = '_ckpt_' + checkpoint_sha256[:16] if checkpoint_sha256 else ''
    return dict(key=f'{entry["id"]}_{world}_{opponent}_{arm}{train_seed}_{seed}_{stage}'+suffix, case=entry['path'], group=entry['id'],
        family=entry['family'], suite=entry.get('suite',stage), pattern=entry['pattern'], density=entry.get('density','original'),
        split=entry['split'], world=world, opponent=opponent, method=arm+str(train_seed) if arm in LEARNED else arm,
        train_seed=train_seed, seed=seed, stage=stage, round=0, model_round=12, budget_ms=200., checkpoint=str(checkpoint) if checkpoint else None,
        _output=str(OUT), **(dict(checkpoint_sha256=checkpoint_sha256) if checkpoint_sha256 else {}))


def worker_init(output, cpus, parent, official):
    global OUT
    OUT = Path(official)
    import ctypes, signal
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() != parent:
        raise RuntimeError('Coordinator exited')
    os.sched_setaffinity(0, {cpus.get()}); torch.set_num_threads(1)
    configure(output); base.PARENT = parent
    # Compile / warm public sensor, capture and actor before timed evaluation.
    config = verify(); entry = read(OUT/'bank/manifest.json')['validation'][0]; case = read(entry['path'])
    torch.manual_seed(1); own = Controller(case['map'], case['speed'], 1, Actor())
    for obs in case['history'][:-1]:
        own.observe(obs)
    own.act(case['history'][-1])
    coverage_potential(np.asarray(obs['positions']), np.asarray(obs['active_p']), np.asarray(obs['active_e']),
                       np.array([e.particles for e in own.belief.estimators]), np.array([e.particle_weights for e in own.belief.estimators]),
                       case['speed'], own.public.vertices, own.public.dist, own.public.boxes)
    audit_simulated(np.zeros((0,26)), np.zeros((0,2,2)), 1.1)
    for arm in ('SINGLE_PRIOR','U','GEOM_SEARCH'):
        seed=17 if arm in LEARNED else 0
        model=CROOT/'learning/x20/round_12/models/single_17.npz' if arm in LEARNED else None
        request_=request(entry,'AB','fixed_D_kernel',arm,seed,'warmup',model)
        search=factory(case,request_)
        for obs in case['history'][:-1]:search.observe(obs)
        search.act(case['history'][-1],fixed_simulations=2)


def make_pool(output, cpus):
    ctx = mp.get_context('spawn'); queue = ctx.Queue()
    for cpu in cpus:
        queue.put(cpu)
    pool = futures.ProcessPoolExecutor(max_workers=len(cpus), mp_context=ctx, initializer=worker_init,
        initargs=(str(output), queue, os.getpid(), str(OUT)))
    return pool, queue


def training_request(task, rng, entries):
    entry = entries[int(rng.integers(len(entries)))]; seed = int(rng.integers(2**31))
    opponent = 'fixed_D_kernel' if rng.random() < .5 else 'rootpuct'
    return request(entry, 'AB' if rng.random()<.5 else 'BA', opponent, task['kind'], task['train_seed'], 'training', episode_seed=seed)


def rollout_job(task):
    plan = OUT/'sampling_execution.json'
    if plan.exists() and read(plan).get('geometry_cache', False):
        with public_geometry_cache():
            return original_rollout_job(task)
    return original_rollout_job(task)


def original_rollout_job(task):
    config = read(OUT/'experiment.json'); actor = Actor(); actor.load_state_dict(task['actor']); actor.eval()
    WORKER['actor'] = actor
    entries = read(OUT/'bank/manifest.json')['train']; rng = np.random.default_rng(task['seed'])
    arrays = {k:[] for k in KEYS}; extra = {k:[] for k in ('raw','logp','active','reward','privileged')}
    lengths = []; summaries = []; failures = []; steps = 0; attempts = 0
    while steps < task['steps']:
        if 'requests' in task and attempts >= len(task['requests']):
            break
        attempts += 1
        if attempts > 30:
            raise RuntimeError('Too many unresolved training episodes; no silent filtering')
        req = task['requests'][attempts-1] if 'requests' in task else training_request(task, rng, entries)
        seed = req['seed']; opponent = req['opponent']
        try:
            payload, _ = base.episode(req)
        except RuntimeError as exc:
            if not any(key in str(exc) for key in ('support','refinement','frozen sample budget')):
                raise
            record = dict(task=req, error=repr(exc), scored=False, used_for_ppo=False)
            save(base.OUT/'invalid_training'/(req['key']+'.json'), record); failures.append(record); continue
        own = CAPTURE; n = len(payload['trace']); phi = own.phi
        reward = np.array([r['reward'] for r in payload['trace']], np.float32)
        raw_total = float(reward.sum())
        reward += .05*(np.r_[phi[1:],0.] - np.array(phi))
        for key in KEYS:
            arrays[key].extend(x[key] for x in own.features_saved)
        extra['raw'].extend(own.raw); extra['logp'].extend(own.logp); extra['active'].extend(own.active)
        extra['reward'].extend(reward)
        extra['privileged'].extend(np.asarray(r['state'][:4],np.float32)/10 for r in payload['trace'])
        lengths.append(n); steps += n
        summary = payload['summary']
        summaries.append(dict(key=req['key'],case=req['case'],world=req['world'],opponent=opponent,seed=seed,decisions=n,
            raw_utility=raw_total,shaped_return=float(reward.sum()),phi_initial=phi[0],double_capture=summary['double_capture'],
            public_replay_exact=summary['public_replay_exact'],safety_shortened=sum(r['stats'].get('safety_shortened',0) for r in payload['trace'])))
    arrays = {k:np.asarray(v,np.float32) for k,v in (arrays|extra).items()}
    return dict(arrays=arrays,lengths=lengths,summaries=summaries,invalid=failures,
                digest=hashlib.sha256(b''.join(arrays[k].tobytes() for k in sorted(arrays))).hexdigest())


def eval_job(task):
    try:
        row = base.episode_job(task)
        if task.get('checkpoint_sha256') and row.get('checkpoint_sha256') != task['checkpoint_sha256']:
            raise RuntimeError('Validation cache checkpoint identity mismatch')
        return row
    except (RuntimeError,ValueError) as exc:
        if not any(key in str(exc) for key in ('support','refinement','frozen sample budget')):
            raise
        row = dict(**task,valid_episode=False,failure=repr(exc),double_capture=None,completion_cost=None,utility=None,
                   latencies_ms=[],cpu_ms=[],public_replay_exact=False)
        save(base.OUT/task['stage']/'invalid'/(task['key']+'.json'), row)
        return row


def validate_checkpoint(pool, kind, seed, model_path, steps, smoke):
    manifest=read(OUT/'bank/manifest.json')
    tasks=[request(e,w,o,kind,seed,'validation',model_path)
           for e in manifest['validation'] for w in ('AB','BA') for o in base.OPPONENTS]
    if smoke:tasks=[task|dict(limit=2,fixed=4) for task in tasks]
    result=list(pool.map(eval_job,tasks));good=[r for r in result if r.get('valid_episode',True)]
    return dict(steps=steps,checkpoint=str(model_path),checkpoint_sha256=sha(model_path),
        raw_utility=float(np.mean([r['utility'] for r in good])) if len(good)==len(result) else None,
        valid=len(good),total=len(result),
        double_capture=float(np.mean([r['double_capture'] for r in good])) if good else None,
        cache_identity='checkpoint_sha256_v2')


def training_worker(args):
    config=verify(); folder=OUT/'learning'/f'{args.algorithm}_{args.seed}'; folder.mkdir(parents=True,exist_ok=True)
    cpus=[int(x) for x in args.cpus.split(',')]; os.sched_setaffinity(0,set(cpus)); torch.set_num_threads(1)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Registered training GPU unavailable')
    device=torch.device(args.device); actor=Actor().to(device); critic=Critic('MAPPO').to(device)
    optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=config['learning_rate'],eps=1e-5)
    steps=0; update=0; best=-float('inf'); completed=[]; validation=[]
    checkpoint=folder/'resume.pt'; generator=np.random.default_rng(args.seed+107)
    bc_record=fit_bc(folder,read(OUT/'teacher_inventory.json')[str(args.seed)],args.seed,device,config['bc_epochs'],emit)
    def actor_digest(state):
        return hashlib.sha256(b''.join(state[key].detach().cpu().numpy().tobytes() for key in sorted(state))).hexdigest()
    bc_parameter_sha=actor_digest(torch.load(bc_record['checkpoint'],map_location='cpu',weights_only=False)['actor'])
    if not checkpoint.exists():actor.load_state_dict(torch.load(bc_record['checkpoint'],map_location=device,weights_only=False)['actor'])
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        actor.load_state_dict(saved['actor']);critic.load_state_dict(saved['critic']);optimizer.load_state_dict(saved['optimizer'])
        steps=saved['steps'];update=saved['update'];best=saved['best'];completed=saved['completed'];validation=saved['validation']
        generator.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),device)
        emit(folder,'resumed',steps=steps,update=update,checkpoint_sha256=sha(checkpoint),
             restored=['actor','critic','optimizer','numpy_rng','torch_rng','cuda_rng'],device=str(device))
    emit(folder,'training_start',steps=steps,update=update,device=str(device),rollout_cpus=cpus)
    pool,queue=make_pool(folder,cpus); started=time.time(); count_invalid=0
    sampling=read(OUT/'sampling_execution.json') if (OUT/'sampling_execution.json').exists() else {}
    try:
        if not checkpoint.exists():
            initial=validate_checkpoint(pool,args.algorithm,args.seed,bc_record['checkpoint'],0,config['smoke'])
            validation.append(initial);best=initial['raw_utility'] if initial['raw_utility'] is not None else -float('inf')
            if initial['raw_utility'] is not None:save(folder/'selected.json',initial|dict(selection='validation only; BC baseline eligible'))
            save(folder/'validation.json',validation)
            emit(folder,'bc_closed_loop_validated',steps=0,update=0,utility=initial['raw_utility'],double_capture=initial['double_capture'])
        if validation and any(row.get('cache_identity')!='checkpoint_sha256_v2' for row in validation):
            emit(folder,'validation_repair',steps=steps,update=update,checkpoints=len(validation),device=str(device))
            validation=[validate_checkpoint(pool,args.algorithm,args.seed,row['checkpoint'],row['steps'],config['smoke'])
                        for row in validation]
            valid=[row for row in validation if row['raw_utility'] is not None]
            best=max((row['raw_utility'] for row in valid),default=-float('inf'))
            if valid:
                selected=max(valid,key=lambda row:row['raw_utility'])
                save(folder/'selected.json',selected|dict(selection='validation only; no test outcomes'))
            else:
                (folder/'selected.json').unlink(missing_ok=True)
            save(folder/'validation.json',validation)
            saved.update(validation=validation,best=best)
            temp=checkpoint.with_suffix('.tmp.pt');torch.save(saved,temp);temp.replace(checkpoint)
            emit(folder,'validation_repaired',steps=steps,update=update,device=str(device),checkpoints=len(validation))
        while steps<config['training_steps_per_run']:
            begin=time.time();weights={k:v.detach().cpu() for k,v in actor.state_dict().items()}
            jobs=[dict(actor=weights,kind=args.algorithm,train_seed=args.seed,seed=int(generator.integers(2**31)),
                       progress=steps/config['training_steps_per_run'],steps=config['rollout_steps_per_worker'])
                  for _ in range(config['rollout_workers_per_run'])]
            rollouts=(parallel_collect(pool,jobs,read(OUT/'bank/manifest.json')['train'],training_request,rollout_job)
                      if sampling.get('episode_parallel') else list(pool.map(rollout_job,jobs)))
            rollout_seconds=time.time()-begin
            count_invalid+=sum(len(r['invalid']) for r in rollouts)
            data={k:np.concatenate([r['arrays'][k] for r in rollouts]) for k in rollouts[0]['arrays']}
            n=len(data['raw']);x=tensor_batch(data,device);priv=torch.as_tensor(data['privileged'],device=device)
            with torch.no_grad():values=critic(x,priv).cpu().numpy()
            adv=[];returns=[];offset=0
            for rollout in rollouts:
                for length in rollout['lengths']:
                    a,g=gae(data['reward'][offset:offset+length],values[offset:offset+length],lam=config['gae_lambda'])
                    adv.append(a);returns.append(g);offset+=length
            assert offset==n
            active=torch.as_tensor(data['active'],device=device); advantages=torch.as_tensor(np.concatenate(adv),device=device)
            valid=active>0;advantages=(advantages-advantages[valid].mean())/advantages[valid].std(unbiased=False).clamp_min(1e-6)
            target=torch.as_tensor(np.concatenate(returns),device=device);critic.update_normalization(target)
            raw=torch.as_tensor(data['raw'],device=device);old_logp=torch.as_tensor(data['logp'],device=device)
            old_value=torch.as_tensor(values,device=device)
            progress=min(1.,steps/config['training_steps_per_run'])
            lr=config['learning_rate']*(1-progress)+config['final_learning_rate']*progress
            entropy_coef=config['entropy_start']*(1-progress)+config['entropy_final']*progress
            for group in optimizer.param_groups:group['lr']=lr
            losses=[];kl=[];clip_fractions=[]
            for epoch in range(config['epochs']):
                for ids in torch.randperm(n,device=device).split(config['minibatch']):
                    xb={k:v[ids] for k,v in x.items()};mean,std=actor(xb);logp=latent_log_prob(mean,std,raw[ids])
                    logratio=(logp-old_logp[ids]).clamp(-20,20);ratio=logratio.exp();mask=active[ids];denom=mask.sum().clamp_min(1)
                    pg=torch.maximum(-advantages[ids]*ratio,-advantages[ids]*ratio.clamp(.8,1.2))
                    policy_loss=(pg*mask).sum()/denom
                    prediction=critic.normalized(xb,priv[ids]);scale=critic.value_scale;vmean=critic.value_mean
                    normalized_target=(target[ids]-vmean)/scale
                    clipped=((old_value[ids]-vmean)/scale)+(prediction-(old_value[ids]-vmean)/scale).clamp(-.2,.2)
                    value_loss=.5*torch.maximum((prediction-normalized_target).square(),(clipped-normalized_target).square()).mean()
                    entropy=(torch.distributions.Normal(mean,std).entropy().sum(-1)*mask).sum()/denom
                    actor_enabled=update>=config['critic_warmup_updates']
                    loss=value_loss+(policy_loss-entropy_coef*entropy if actor_enabled else 0.)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO loss')
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),.5);optimizer.step()
                    losses.append(float(loss.detach()));kl.append(float((((ratio-1)-logratio)*mask).sum().detach()/denom))
                    clip_fractions.append(float((((ratio-1).abs()>.2)*mask).sum().detach()/denom))
            steps+=n;update+=1
            for rollout in rollouts:
                for item in rollout['summaries']:item.update(update=update)
            summaries=[item for r in rollouts for item in r['summaries']]
            parameter_sha=actor_digest(actor.state_dict())
            if update<=config['critic_warmup_updates'] and parameter_sha!=bc_parameter_sha:
                raise RuntimeError('Critic-only warmup changed the frozen BC actor')
            record=dict(update=update,steps=steps,critic_warmup=update<=config['critic_warmup_updates'],actor_parameter_sha256=parameter_sha,device=str(device),agent_steps=float(active.sum()),episodes=len(summaries),invalid_episodes=sum(len(r['invalid']) for r in rollouts),
                training_raw_utility=float(np.mean([r['raw_utility'] for r in summaries])),double_capture=float(np.mean([r['double_capture'] for r in summaries])),
                loss=float(np.mean(losses)),approx_kl=float(np.mean(kl)),clip_fraction=float(np.mean(clip_fractions)),
                value_scale=float(critic.value_scale),learning_rate=lr,entropy_coef=entropy_coef,seconds=time.time()-begin,
                rollout_digests=[r['digest'] for r in rollouts],safety_shortened=sum(r['safety_shortened'] for r in summaries),
                rollout_seconds=rollout_seconds,optimization_seconds=time.time()-begin-rollout_seconds,
                rollout_workers=len(cpus),logical_streams=config['rollout_workers_per_run'])
            save(folder/'updates'/f'{update:05d}.json',record|dict(episodes_metadata=summaries))
            due=[point for point in config['validation_at'] if steps>=point and point not in completed]
            if due:
                model_path=folder/'checkpoints'/f'steps_{steps}.pt';model_path.parent.mkdir(parents=True,exist_ok=True)
                torch.save(dict(actor={k:v.detach().cpu() for k,v in actor.state_dict().items()},critic={k:v.detach().cpu() for k,v in critic.state_dict().items()},
                    kind=args.algorithm,seed=args.seed,steps=steps,training_only_privileged_critic=True),model_path)
                row=validate_checkpoint(pool,args.algorithm,args.seed,model_path,steps,config['smoke'])
                score=row['raw_utility']
                validation.append(row);completed.extend(due)
                if score is not None and score>best:
                    best=score
                    save(folder/'selected.json',row|dict(selection='validation only; no test outcomes'))
                save(folder/'validation.json',validation)
            payload=dict(actor=actor.state_dict(),critic=critic.state_dict(),optimizer=optimizer.state_dict(),steps=steps,update=update,
                best=best,completed=completed,validation=validation,numpy_rng=generator.bit_generator.state,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else torch.get_rng_state())
            temp=checkpoint.with_suffix('.tmp.pt');torch.save(payload,temp);temp.replace(checkpoint)
            emit(folder,'training',**record,registered_steps=config['training_steps_per_run'])
        if not (folder/'selected.json').exists():raise RuntimeError('No complete validation checkpoint')
        save(folder/'decision.json',dict(status='complete',steps=steps,updates=update,selected=read(folder/'selected.json'),
            convergence_claim=False,registered_budget_completed=True,wall_seconds=time.time()-started))
        emit(folder,'complete',steps=steps,updates=update,selected=read(folder/'selected.json'))
    finally:
        pool.shutdown(wait=True,cancel_futures=True);queue.close()


def evaluate(tasks, cpus, output):
    configure(output); done=[];pending=[]
    for task in tasks:
        cached=base.cached(task)
        if cached is None:
            invalid=Path(output)/task['stage']/'invalid'/(task['key']+'.json')
            if invalid.exists():done.append(read(invalid))
            else:pending.append(task)
        else:done.append(cached)
    if pending:
        pool,queue=make_pool(output,cpus)
        try:
            running={};source=iter(pending);exhausted=False;last=0
            while running or not exhausted:
                while not exhausted and len(running)<2*len(cpus):
                    task=next(source,None)
                    if task is None:exhausted=True;break
                    running[pool.submit(eval_job,task)]=task
                if not running:break
                completed,_=futures.wait(running,timeout=5,return_when=futures.FIRST_COMPLETED)
                for future in completed:
                    task=running.pop(future)
                    try:done.append(future.result())
                    except BaseException:
                        save(Path(output)/'failed_task.json',task);raise
                if time.time()-last>15 or len(done)==len(tasks):
                    emit(output,'evaluation',completed=len(done),total=len(tasks),workers=len(cpus),invalid=sum(not r.get('valid_episode',True) for r in done));last=time.time()
        finally:
            pool.shutdown(wait=True,cancel_futures=True);queue.close()
    return sorted(done,key=lambda r:r['key'])


def summarize(rows):
    cells=collections.defaultdict(list)
    for row in rows:cells[row['suite'],row['opponent'],row['method'].rstrip('0123456789')].append(row)
    summaries=[]
    for (suite,opponent,arm),rr in sorted(cells.items()):
        good=[r for r in rr if r.get('valid_episode',True)];lat=[t for r in good for t in r['latencies_ms']]
        summaries.append(dict(suite=suite,opponent=opponent,arm=arm,planned=len(rr),valid=len(good),
            double_capture=sum(r['double_capture'] for r in good),completion_cost=float(np.mean([r['completion_cost'] for r in good])) if good else None,
            utility=float(np.mean([r['utility'] for r in good])) if good else None,p95_ms=float(np.percentile(lat,95)) if lat else None,
            overrun_rate=float(np.mean(np.asarray(lat)>200)) if lat else None))
    primary={}
    for suite in sorted({r['suite'] for r in rows}):
        rr=[r for r in rows if r['suite']==suite];groups=sorted({r['group'] for r in rr})
        expected={(g,w,o,a,seed) for g in groups for w in ('AB','BA') for o in base.OPPONENTS for a in ARMS for seed in (SEEDS if a in LEARNED else (0,))}
        actual=[(r['group'],r['world'],r['opponent'],r['method'].rstrip('0123456789'),r['train_seed']) for r in rr]
        complete=set(actual)==expected and len(actual)==len(expected) and all(r.get('valid_episode',True) for r in rr)
        result=dict(status='complete' if complete else 'withheld_incomplete_cells',groups=len(groups),comparisons=[])
        if complete:
            comparisons=[(o,a) for o in base.OPPONENTS for a in ARMS[1:]]
            means={(g,o,a):np.mean([r['utility'] for r in cells[suite,o,a] if r['group']==g]) for g in groups for o in base.OPPONENTS for a in ARMS}
            delta=np.array([[means[g,o,'BC_MAPPO']-means[g,o,a] for o,a in comparisons] for g in groups]);mean=delta.mean(0)
            family={g:next(r['family'] for r in rr if r['group']==g) for g in groups};rng=np.random.default_rng(BASE_SEED+int(suite=='generalization'))
            bs=np.zeros((10000,len(comparisons)))
            for f in sorted(set(family.values())):
                ids=np.array([i for i,g in enumerate(groups) if family[g]==f]);bs+=delta[ids[rng.integers(len(ids),size=(10000,len(ids)))]].sum(1)/len(groups)
            radius=float(np.quantile(np.max(abs(bs-mean),axis=1),.95))
            result['comparisons']=[dict(opponent=o,contrast='BC_MAPPO - '+a,utility_difference=float(mean[k]),simultaneous95=[float(mean[k]-radius),float(mean[k]+radius)]) for k,(o,a) in enumerate(comparisons)]
        primary[suite]=result
    return dict(summaries=summaries,primary=primary,status='complete' if primary and all(r['status']=='complete' for r in primary.values()) else 'incomplete')


def report(rows=None,stats=None):
    status=read(OUT/'status.json');config=read(OUT/'experiment.json')
    lines=['# E45-H：局部2v2 BC＋MAPPO预训练与微调benchmark','',
        '状态：`'+status['stage']+'`。所有实验结果以完成记录为准。没有在线专家分配或残差教师。','',
        '[进度](status.json) · [冻结协议](PROTOCOL.md) · [配置](experiment.json) · [执行资源](execution.json) · [运行日志](run.log)','',
        f'BC初始化后MAPPO微调3种子，每模型{config["training_steps_per_run"]}环境步；{config["train_maps"]}训练地图、{config["validation_maps"]}验证地图；BC_ONLY并列。',
        '新独立测试36组新实例＋48组未训练障碍布局，10族22执行臂，共7392局；与旧C/B/D/F/G地图查重。',
        '所有方法共享公共团队观测、64带权信念、合法动作规则和200ms允许预算；纯策略不强行等待。MAPPO仅训练critic可见目标真位置，部署actor绝不可见。',
        '训练预算完成不自动代表收敛；学习曲线、最终验证趋势和种子差异均保留。信念失败不伪造输赢，对应主比较暂不完整。','',
        '|方法/种子|状态|环境步|验证最佳效用|','|---|---|---:|---:|']
    if status.get('error'):
        lines[3:3] = ['最后错误：`'+status['error']+'`。详见[failure.json](failure.json)。','']
    for kind in RL:
        for seed in SEEDS:
            folder=OUT/'learning'/f'{kind}_{seed}'
            state=read(folder/'status.json') if (folder/'status.json').exists() else dict(stage='pending',steps=0)
            selected=read(folder/'selected.json') if (folder/'selected.json').exists() else {}
            lines.append(f'|{kind}/{seed}|{state["stage"]}|{state.get("steps",0)}|{selected.get("raw_utility","pending")}|')
    if stats:
        if config['smoke']:lines+=['','**工程smoke：以下均为两步前缀，禁止作为科学性能结论。**']
        lines+=['','|测试套|对手|方法|有效/计划|双捕|完成成本|效用|P95毫秒|','|---|---|---|---:|---:|---:|---:|---:|']
        for r in stats['summaries']:
            lines.append(f'|{r["suite"]}|{r["opponent"]}|{r["arm"]}|{r["valid"]}/{r["planned"]}|{r["double_capture"]}|{r["completion_cost"]}|{r["utility"]}|{r["p95_ms"]}|')
        for suite,data in stats['primary'].items():
            lines+=['',f'## {suite}预定主比较：{data["status"]}','']
            for r in data['comparisons']:lines.append(f'- {r["opponent"]}，{r["contrast"]}：{r["utility_difference"]:+.5f}，同时95%区间{r["simultaneous95"]}')
        lines+=['','[完整结果](results.json) · [统计](statistics.json) · [学习曲线](learning_curves.png)']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def plot_curves():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for kind in RL:
        for seed in SEEDS:
            folder=OUT/'learning'/f'{kind}_{seed}';updates=[read(p) for p in sorted((folder/'updates').glob('*.json'))]
            if updates:
                axes[0].plot([r['steps'] for r in updates],[r['training_raw_utility'] for r in updates],alpha=.4,label=f'{kind}{seed}')
            validation=read(folder/'validation.json') if (folder/'validation.json').exists() else []
            if validation:axes[1].plot([r['steps'] for r in validation],[r['raw_utility'] if r['raw_utility'] is not None else np.nan for r in validation],marker='o',label=f'{kind}{seed}')
    for ax in axes:ax.set_xlabel('Actual team environment steps');ax.set_ylabel('Raw team utility');ax.legend();ax.grid(alpha=.2)
    axes[0].set_title('Training (curriculum changes)');axes[1].set_title('Frozen validation; checkpoint selection only')
    fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=150);plt.close(fig)


def final_tasks():
    config=verify();manifest=read(OUT/'bank/manifest.json');tasks=[]
    checkpoints={}
    for kind in RL:
        for seed in SEEDS:
            selected=read(OUT/'learning'/f'{kind}_{seed}'/'selected.json')
            if sha(selected['checkpoint'])!=selected['checkpoint_sha256']:raise RuntimeError('Selected model changed')
            checkpoints[kind,seed]=selected['checkpoint']
    for seed in SEEDS:
        checkpoints['BC_ONLY',seed]=str(OUT/'learning'/f'BC_MAPPO_{seed}'/'bc_selected.pt')
        for kind in ('MAPPO','IPPO'):checkpoints[kind,seed]=read(GROOT/'frozen_models.json')[kind+str(seed)]['path']
    freeze={kind+str(seed):dict(path=path,sha256=sha(path)) for (kind,seed),path in checkpoints.items()}
    if (OUT/'frozen_models.json').exists() and read(OUT/'frozen_models.json')!=freeze:
        raise RuntimeError('A previously frozen test model changed')
    save(OUT/'frozen_models.json',freeze)
    for entry in manifest['test']:
        for w in ('AB','BA'):
            for o in base.OPPONENTS:
                for arm in ARMS:
                    for seed in (SEEDS if arm in LEARNED else (0,)):
                        path=checkpoints[arm,seed] if arm in CHECKPOINT_ARMS else CROOT/f'learning/x20/round_12/models/single_{seed}.npz' if arm in LEARNED else None
                        tasks.append(request(entry,w,o,arm,seed,'evaluation',path))
    if config['smoke']:tasks=[t|dict(limit=2,fixed=4) for t in tasks]
    return tasks,checkpoints


def calibrate(cpus,checkpoints):
    # Dense held-out engineering layouts are independent of both C training
    # and G test; calibration is not restricted to the easier original maps.
    validation=read(FROOT/'bank/manifest.json')['validation']
    tasks=[]
    for entry in validation:
        for arm in ARMS:
            path=checkpoints[arm,17] if arm in CHECKPOINT_ARMS else CROOT/'learning/x20/round_12/models/single_17.npz' if arm in LEARNED else None
            for o in base.OPPONENTS:
                tasks.append(request(entry,'AB',o,arm,17 if arm in LEARNED else 0,'calibration',path)|dict(limit=2))
    reference=evaluate(tasks,cpus[:1],OUT/'timing/reference')
    def metrics(rows):
        if any(not r.get('valid_episode',True) for r in rows):raise RuntimeError('Calibration support failure')
        cpu=sum(sum(r['cpu_ms']) for r in rows);wall=sum(sum(r['latencies_ms']) for r in rows)
        search={a:sum(sum(r['hidden_simulations']) for r in rows if r['method'].rstrip('0123456789')==a) for a in ('SINGLE_PRIOR','U','GEOM_SEARCH')}
        lat=[t for r in rows for t in r['latencies_ms']]
        return dict(cpu_wall=cpu/max(wall,1e-9),search=search,overrun=float(np.mean(np.array(lat)>200)))
    serial=metrics(reference);records=[];selected=1
    for count in [n for n in (48,40,32,24,16,8,4,1) if n<=len(cpus)]:
        rows=evaluate(tasks,cpus[:count],OUT/f'timing/parallel_{count}');result=metrics(rows)
        ratios=[result['search'][a]/serial['search'][a] for a in serial['search'] if serial['search'][a]]
        total=sum(result['search'].values())/max(1,sum(serial['search'].values()))
        passed=result['cpu_wall']>=.85 and total>=.85 and min(ratios,default=1)>=.65 and result['overrun']<=serial['overrun']+.002
        records.append(dict(workers=count,metrics=result,simulation_ratio=total,passed=passed))
        if passed:selected=count;break
    save(OUT/'timing/decision.json',dict(workers=selected,reference=serial,candidates=records,outcomes_used=False))
    return selected


def supplement_report(extra):
    config=read(OUT/'experiment.json')
    if config['smoke']:
        (OUT/'f_supplement/REPORT.md').write_text('# 工程smoke\n\n24条两步前缀，不作性能结论。\n')
        return
    source=ROOT/'current/e45f_generalization_benchmark_20261029_support_repaired/results.json'
    reference=[r for r in read(source) if r['method'].rstrip('0123456789') in ARMS]
    evidence=[]
    for row in reference:
        if sha(row['episode'])!=row['episode_sha256']:
            raise RuntimeError('Historical F episode hash changed: '+row['key'])
        evidence.append(dict(key=row['key'],episode=row['episode'],sha256=row['episode_sha256']))
    merged=sorted(reference+extra,key=lambda r:r['key']);stats=summarize(merged)
    save(OUT/'f_supplement/merged_results.json',merged);save(OUT/'f_supplement/statistics.json',stats)
    save(OUT/'f_supplement/reference_integrity.json',dict(source=str(source),source_sha256=sha(source),
        historical_rows=len(reference),new_rl_rows=len(extra),episode_hashes=evidence,
        scope='historical reference, different wall-clock batches; no test-based training or tuning'))
    lines=['# 原E45-F地图上的MAPPO/IPPO追加对照','',
        'MAPPO/IPPO仅用C训练/验证；F地图未用于训练或选模。其余方法来自旧F修订结果，经逐局SHA核验。',
        '这是历史参考集的追加，不替代G新地图独立确认；旧F与新RL属于不同墙钟批次，保留来源及估计器包装差异。','',
        '|测试套|对手|方法|有效/计划|双捕|完成成本|效用|','|---|---|---|---:|---:|---:|---:|']
    for r in stats['summaries']:
        lines.append(f'|{r["suite"]}|{r["opponent"]}|{r["arm"]}|{r["valid"]}/{r["planned"]}|{r["double_capture"]}|{r["completion_cost"]}|{r["utility"]}|')
    lines+=['','[追加2016局](results.json) · [含历史参考](merged_results.json) · [完整性核验](reference_integrity.json)']
    (OUT/'f_supplement/REPORT.md').write_text('\n'.join(lines)+'\n')


def orchestrate(args):
    config=prepare(args.smoke,args.steps);cpus=choose_cpus(min(args.workers,48));device=args.device
    if not cpus:raise RuntimeError('No physical CPU available')
    if not (OUT/'execution.json').exists():
        devices=args.devices.split(',') if args.devices else [device]
        if any(not item.startswith('cuda:') or not item[5:].isdigit() for item in devices):
            raise ValueError('--devices requires comma-separated cuda:N device names')
        save(OUT/'execution.json',dict(cpus=cpus,device=device,maximum_workers=len(cpus),threads_per_worker=1,tmux='train:E45H',
            training_devices={f'{kind}_{seed}':devices[i%len(devices)] for i,(kind,seed) in enumerate((k,s) for k in RL for s in SEEDS)}))
    else:
        execution=read(OUT/'execution.json');cpus=execution['cpus'];device=execution['device']
    device_map=read(OUT/'execution.json').get('training_devices',{})
    sampling=read(OUT/'sampling_execution.json') if (OUT/'sampling_execution.json').exists() else {}
    per=config['rollout_workers_per_run'];concurrent=max(1,min(6,len(cpus)//per));pending=collections.deque((k,s) for k in RL for s in SEEDS)
    running=[];slots=collections.deque(range(concurrent));last=0
    try:
        while pending or running:
            while pending and slots:
                kind,seed=pending.popleft();folder=OUT/'learning'/f'{kind}_{seed}'
                if (folder/'decision.json').exists() and read(folder/'decision.json')['status']=='complete':continue
                slot=slots.popleft();cores=cpus[slot*per:(slot+1)*per]
                cores=sampling.get('rollout_cpus',{}).get(f'{kind}_{seed}',cores)
                training_device=device_map.get(f'{kind}_{seed}',device)
                folder.mkdir(parents=True,exist_ok=True);log=(folder/'run.log').open('a')
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--train-one','--algorithm',kind,'--seed',str(seed),
                     '--device',training_device,'--cpus',','.join(map(str,cores)),'--output',str(OUT)]
                process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running.append((process,log,slot,kind,seed))
            for item in list(running):
                process,log,slot,kind,seed=item
                if process.poll() is not None:
                    log.close();running.remove(item);slots.append(slot)
                    if process.returncode:raise RuntimeError(f'{kind}/{seed} trainer failed; see learning/{kind}_{seed}/run.log')
            if time.time()-last>20:
                emit(OUT,'training',active_runs=len(running),pending_runs=len(pending),
                     rollout_workers_per_run=len(next(iter(sampling['rollout_cpus'].values()))) if sampling.get('rollout_cpus') else per,
                     maximum_workers=sum(map(len,sampling['rollout_cpus'].values())) if sampling.get('rollout_cpus') else len(cpus),
                     training_devices=device_map,episode_parallel=sampling.get('episode_parallel',False));report();last=time.time()
            if running:time.sleep(2)
        verify();plot_curves();tasks,checkpoints=final_tasks();save(OUT/'requests.json',tasks)
        count=min(len(cpus),8) if config['smoke'] else calibrate(cpus,checkpoints)
        emit(OUT,'evaluation_start',total=len(tasks),workers=count);report()
        np.random.default_rng(BASE_SEED+99).shuffle(tasks)
        rows=evaluate(tasks,cpus[:count],OUT);save(OUT/'results.json',rows);stats=summarize(rows);save(OUT/'statistics.json',stats)
        extra=[]
        verify();plot_curves()
        invalid=sum(not r.get('valid_episode',True) for r in rows+extra)
        decision=dict(status='complete' if not invalid else 'complete_with_explicit_invalid_cells',primary_status=stats['status'],
            fresh_tasks=len(rows),supplement_tasks=len(extra),invalid=invalid,trained_models=3,training=False)
        export_completion_audit(rows,extra)
        save(OUT/'decision.json',decision);emit(OUT,decision['status'],**{k:v for k,v in decision.items() if k!='status'});report(rows,stats)
        update_ledger(decision)
    finally:
        if running:
            import signal
            for process,log,slot,kind,seed in running:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    folder=OUT/'learning'/f'{kind}_{seed}'
                    state=read(folder/'status.json') if (folder/'status.json').exists() else {}
                    emit(folder,'interrupted',steps=state.get('steps',0),update=state.get('update',0),
                         reason='coordinator stopped; resume.pt retained')
                log.close()


def export_completion_audit(rows,extra):
    models=read(OUT/'frozen_models.json')
    for row in models.values():
        if sha(row['path'])!=row['sha256']:raise RuntimeError('Frozen model mutated during test')
    for rows_,name in ((rows,'fresh'),(extra,'f_supplement')):
        if len({r['key'] for r in rows_})!=len(rows_):raise RuntimeError('Duplicate '+name+' cell')
        for row in rows_:
            if row.get('valid_episode',True):
                if not row['public_replay_exact'] or sha(row['episode'])!=row['episode_sha256']:
                    raise RuntimeError('Completed episode replay/hash mismatch')
    decisions=[read(OUT/'learning'/f'{k}_{s}'/'decision.json') for k in RL for s in SEEDS]
    save(OUT/'completion_audit.json',dict(fresh=len(rows),f_supplement=len(extra),trained_models=len(RL)*len(SEEDS),frozen_policy_checkpoints=len(models),
        actual_training_steps=sum(d['steps'] for d in decisions),registered_budget_all_completed=all(d['registered_budget_completed'] for d in decisions),
        all_models_hash_checked=True,all_valid_episodes_hash_and_public_replay_checked=True,convergence_claim=False))


def update_ledger(decision):
    if read(OUT/'experiment.json')['smoke']:return
    path=ROOT/'Git提交后实验完整汇总_20261025.md';text=path.read_text();anchor='<!-- E45H_STATUS -->'
    note=(anchor+'\n> **E45-H同条件RL benchmark：** '+decision['status']+'；BC＋MAPPO三训练种子，'
          +f'新独立评测{decision["fresh_tasks"]}局、无效{decision["invalid"]}局。'
          +'完整数据见[本轮报告](current/e45g_rl_benchmark_20261030/REPORT.md)。\n<!-- /E45H_STATUS -->\n')
    import re
    if anchor in text:text=re.sub(r'<!-- E45H_STATUS -->.*?<!-- /E45H_STATUS -->\n',lambda _:note,text,flags=re.S)
    else:text=text.replace('\n\n','\n\n'+note+'\n',1)
    path.write_text(text)


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--workers',type=int,default=48)
    parser.add_argument('--device',default='cuda:2');parser.add_argument('--steps',type=int,default=2_000_000)
    parser.add_argument('--devices',help='Round-robin training GPUs for a new execution plan, e.g. cuda:0,cuda:2,cuda:3')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--train-one',action='store_true');parser.add_argument('--algorithm',choices=RL);parser.add_argument('--seed',type=int);parser.add_argument('--cpus')
    args=parser.parse_args()
    if args.output:OUT=args.output.resolve()
    elif args.smoke:OUT=OUT.with_name(OUT.name+'_smoke')
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        if args.train_one:training_worker(args)
        elif args.prepare_only:prepare(args.smoke,args.steps)
        else:
            with (OUT/'coordinator.lock').open('w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);orchestrate(args)
    except BaseException as exc:
        folder=OUT/'learning'/f'{args.algorithm}_{args.seed}' if args.train_one else OUT
        folder.mkdir(parents=True,exist_ok=True)
        save(folder/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        state=read(folder/'status.json') if (folder/'status.json').exists() else {}
        emit(folder,'failed',error=repr(exc),**{k:state[k] for k in ('steps','update') if k in state})
        if not args.train_one and (OUT/'experiment.json').exists():report()
        raise


if __name__=='__main__':main()
