"""Bounded local-observation NTE+D baseline experiment with a teacher gate.

Run from this project's root with the dm_env_py311 interpreter. All deployed
sources and previous experiments are hashed, not edited. Tests precede the
six-hour data/train/evaluation budget. A failed teacher gate ends the trial.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import fcntl
import hashlib
import inspect
import json
import platform
import resource
import signal
import time
from pathlib import Path
import numpy as np
import torch
import run_search_value_validation as engine
from run_frozen_d_closed_loop import save, sha, legal_action
from corner_estimator_scenarios import construct
from verify_l45_occlusion_scenarios import verify_case
from audit_frozen_d_search_interface import observation, independent_traces, probability_audit
from solvers.frozen_d_search_interface import Controller as Geometry, loader, FROZEN
from solvers.nte_d_local import Controller, Node
from learning.nte_d_local import Network, Runtime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'current/l45_nte_d_local_baseline_20261027'
BASE_SEED = 93020261027
SEEDS = (17, 29, 43)
METHODS = ('G', 'M0')+tuple(f'{m}{s}' for m in ('NPI', 'NPIV', 'DIRECT') for s in SEEDS)
SOURCE_NAMES = ('run_nte_d_local_baseline.py', 'solvers/nte_d_local.py',
                'learning/nte_d_local.py', 'tests/test_nte_d_local.py')
ACTIVE = None
RANDOM_MODELS = False
PIPELINE_START = None


def emit(**values):
    save(OUT/'status.json', dict(pid=os.getpid(), **values))
    print(json.dumps(values, ensure_ascii=False), flush=True)


def read(path):
    return json.loads(Path(path).read_text())


def entries(split):
    return [e for e in read(OUT/'bank/manifest.json')['entries'] if e['split'] == split]


def prepare():
    if (OUT/'experiment.json').exists():
        return verify()
    known_m, known_h = set(), set()
    for p in (ROOT/'current').glob('**/controller_inputs/*.json'):
        d = read(p)
        if 'map' in d and 'history' in d:
            known_m.add(hashlib.sha256(json.dumps(d['map'], sort_keys=True).encode()).hexdigest())
            known_h.add(hashlib.sha256(json.dumps(dict(map=d['map'], history=d['history']), sort_keys=True).encode()).hexdigest())
    rows, checks = [], []
    previous = dict(maps=len(known_m), histories=len(known_h))
    for si, (split, repeats) in enumerate((('train', 8), ('validation', 2), ('test', 6))):
        for fi, family in enumerate(('long_wall', 'wall_corner', 'multi_route')):
            for i in range(repeats):
                seed = int(np.random.SeedSequence([BASE_SEED, si, fi, i]).generate_state(1)[0])
                rng = np.random.default_rng(seed)
                H, speed = (4 if i % 2 == 0 else 6), (1.1 if i % 4 < 2 else 1.3)
                height = float(rng.uniform(6.2, 6.9) if H == 4 else rng.uniform(8.15, 8.85))
                env, packet = construct(family, H, speed, seed, height, i % 2, pause=i % 2)
                check = verify_case(env, packet)
                assert check['passed'], check
                mh = hashlib.sha256(json.dumps(packet['map'], sort_keys=True).encode()).hexdigest()
                hh = hashlib.sha256(json.dumps(dict(map=packet['map'], history=packet['history']), sort_keys=True).encode()).hexdigest()
                assert mh not in known_m and hh not in known_h
                known_m.add(mh); known_h.add(hh)
                folder = OUT/'bank'/split
                for kind, data in (('controller_inputs', packet), ('environment', env)):
                    save(folder/kind/(env['id']+'.json'), data)
                rows.append(dict(id=env['id'], family=family, split=split,
                    packet_path=str(folder/'controller_inputs'/(env['id']+'.json')),
                    environment_path=str(folder/'environment'/(env['id']+'.json')),
                    map_sha256=mh, geometry_history_sha256=hh))
                checks.append(check)
    save(OUT/'bank/manifest.json', dict(entries=rows, counts=dict(train=24, validation=6, test=18),
        prior_counts=previous, disjoint_maps_and_histories=True,
        split_unit='whole map and all histories/hidden worlds', test_outcomes_read=False))
    save(OUT/'bank_acceptance.json', dict(passed=True, geometry_only=True, records=checks))
    protected = {}
    for name in ('param.py', 'problems/example8.py', 'solvers/c_puct.py',
                 'cpp/bindings.cpp', 'cpp/problems/example8.hpp'):
        p = ROOT/'code'/name
        protected[str(p)] = sha(p)
    for folder in ('l45_candidate_prior_30e_20261026', 'l45_candidate_prior_train_20261026',
                   'l45_search_value_20261026', 'l45_corner_turn_estimator_20261025/frozen_estimator'):
        for p in (ROOT/'current'/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.nbc', '.nbi'):
                protected[str(p)] = sha(p)
    reuse = ('solvers/search_value_belief.py', 'solvers/frozen_d_search_interface.py',
             'search_value_model.py', 'run_search_value_validation.py', 'run_frozen_d_closed_loop.py',
             'corner_estimator_scenarios.py', 'estimator_audit_geometry.py',
             'solvers/root_audit_puct.py', 'audit_frozen_d_search_interface.py')
    config = dict(version='local_belief_NTE_D_baseline_v1', date='2026-10-27',
        budget_ms=200., expert_budget_ms=2000., wall_cap_seconds=21600,
        train_maps=24, validation_maps=6, test_maps=18, worlds_per_history=2,
        train_seeds=list(SEEDS), epochs_per_seed=30, training_rounds=1,
        max_train_states=768, max_validation_states=192, max_test_games=396,
        methods=list(METHODS), planning_depth=8, discount=.95,
        reward='first rediscovery + capture; unchanged',
        opponent='original full-state RootPUCT3000 depth100 no networks; pre-action state only',
        estimator='accepted frozen D / 64 weighted positions / fixed-root planning kernel',
        local_information_for_all_pursuer_components=True, privileged_teacher=False,
        continuous_actions=True, tree_index='public action-observation history',
        progressive_widening='ceil(2*sqrt(node visits+1))', selection='Q+sqrt(2*log(N+1)/n)',
        beta_policy=.5, beta_value=.5, rejection_cap=64, fallback='checked stop, not G',
        policy='weighted particle DeepSets3-32-32, map4-32-32, history7-32-32, head105-64-64-4',
        value='independent same encoders/head105-64-64-1, sigmoid scaled to [0,2]',
        action_distribution='single diagonal Gaussian; tanh mean; logvar clamp[-6,1]; per-axis clip then legality conditioning',
        supervision='visit-weighted continuous-action first and second moments; independent public-model MC return for value',
        policy_loss='Gaussian NLL on visit-weighted action distribution, same moments as weighted mean label',
        value_loss='MSE independent public-model MC return; no true hidden trajectory labels',
        optimizer=dict(name='Adam', lr=.001, weight_decay=.0001, batch=64),
        checkpoint='separate minimum validation loss for each policy/value, within 30 epochs; no test selection',
        feature_capacity=dict(map_boxes=8, particles=64, recent_history=16, extra_last_seen=1),
        teacher_gate=dict(states=12, split='first four training maps per topology, handoff hidden observation',
            independent_searches=2, each_ms=2000., heldout_probe_ms=2000., max_paired_draws=96,
            actions='search A visit winner, A visit runner-up, public G reference',
            signal='held-out paired return winner-runner gap minus 1.96*SE >0; >=24 complete pairs',
            stability='L2 difference of two visit-weighted normalized mean actions <=0.5',
            required_pass_states=7, evidence='development reliability screen, not a calibrated search superiority test',
            failure='stop before formal labels, training and held-out closed-loop test'),
        collection='24/6 complete world-A trajectories, 80% G plus 20% legal uniform exploration; per-map evenly spaced public frames',
        primary_comparisons=['NPI - M0', 'NPIV - NPI', 'NPIV - DIRECT'],
        statistics=dict(groups=18, draws=10000, seed=BASE_SEED+1,
            correction='max absolute centered bootstrap simultaneous radius across 3 mean-cost differences'),
        source_hashes={str(ROOT/'code'/p): sha(ROOT/'code'/p) for p in SOURCE_NAMES+reuse},
        protected_hashes=protected,
        bank_hashes={str(p): sha(p) for p in (OUT/'bank').rglob('*.json')},
        deployment_changed=False, D_changed=False, new_geometry_generator=False, ppo=False,
        caveat='NTE belief/task adaptation with local teacher, not exact original-paper reproduction')
    save(OUT/'experiment.json', config)
    for p in SOURCE_NAMES+reuse:
        target = OUT/'source_snapshot'/p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/'code'/p).read_bytes())
    (OUT/'PLAN.md').write_bytes((ROOT/'NTE_D_局部观测研究计划_20261027.md').read_bytes())
    (OUT/'PROTOCOL.md').write_text(protocol_text(), encoding='utf-8')
    emit(stage='registered', train_maps=24, validation_maps=6, test_maps=18,
         teacher_gate_states=12, wall_cap_seconds=21600, training_started=False)
    return config


def protocol_text():
    return '''# 局部观测 NTE＋D 基线：执行协议

正式配置见 experiment.json，源代码和地图/历史在教师查询前冻结。全链路只使用公开地图、自身状态、合法观测历史及D信念。模拟真值只推进模拟、产生传感器观测和回报。

独立24/6/18地图，三类拓扑；测试18组历史各两个隐藏世界。三个训练种子17/29/43各最多30轮，单轮搜索监督。连续动作渐进扩展；不复用旧根先验网络或特权标签。策略和值独立编码，测试前按验证损失冻结。研究部署分离。

先用随机临时网络验收：带权分布、采样与运动核、集合置换、PyTorch/NumPy一致、合法动作、模拟转移、固定次数重放、同一公开历史隔离隐藏世界。编译预热单列。正式数据、训练、闭环与审计累计墙钟上限6小时，单线程CPU；在线200ms，离线查询2000ms。

教师屏障：12个预定训练地图的遮挡接管点，各两次独立目标随机流的2秒搜索。对A的最高访问动作与次高访问动作做独立、公共信念采样的配对MC，另带G参考；每次最多96对、2秒，至少24完整对。定义信号为配对回报差均值−1.96SE>0；稳定性为两次搜索访问加权归一化动作均值L2差≤0.5。至少7/12同时通过才生成正式标签并训练。少于此数即按计划停止，保留随机临时网络但明确它不是训练检查点，不消耗测试闭环。此屏障是开发可靠性筛查，不是正式多重检验结论。不得因失败切换G标签或更改奖励/深度。

通过屏障才采集每图完整轨迹：固定0.8G/0.2合法均匀探索，按地图/轨迹隔离，每训练图最多32状态、验证图最多32状态。每状态离线M0搜索≤2秒，保留全部根访问与回报统计，策略监督为访问加权动作均值/方差；值为独立MC回报。记录均值动作是否合法，不投影修改教师标签；网络执行时依法条件采样。不是声称精确复制论文Eq.13的训练损失。

G、M0、NPI×3、NPIV×3、DIRECT×3，共396局上限。可见阶段共用G追踪。三种搜索共享模型、树选择、rollout、深度和奖励，只有神经扩展/叶值开关不同。直接策略使用同一策略检查点合法采样。

完成成本：成功取实际步数、失败取接管剩余步数；并报合法捕获、重发现/时间、完整时延与超时。主要比较NPI−M0、NPIV−NPI、NPIV−DIRECT，按公共历史配对bootstrap10000次，三项同时95%区间。可靠收益要求区间上界<0、均值≤−1步、各种子同向、观察捕获率不下降及审计通过；捕获护栏不等于非劣。G完整保留，不以必须超过G作为框架建立条件。

完整决策含初始化、历史回放、D更新、特征、推理、搜索和动作检查；首次单独真实计账，保留超时。对手/环境另列联合周期。全部模型封存后一次性测试，不依据测试重训。不足预算、屏障失败或无可靠增益均停止，不自动扩容或添加模块。
'''


def verify():
    loader.verify()
    config = read(OUT/'experiment.json')
    for name in ('source_hashes', 'protected_hashes', 'bank_hashes'):
        for p, expected in config[name].items():
            assert sha(p) == expected, ('Frozen artifact changed', p)
    return config


def model_path(seed, kind):
    return OUT/('interface_random_models' if RANDOM_MODELS else 'models')/f'{kind}_{seed}.npz'


def initialize(packet, method, seed):
    global ACTIVE
    start = time.perf_counter()
    boxes, speed = packet['map'], packet['motion']['pursuer_axis_limit']
    if method == 'G':
        own = Geometry(boxes, speed, 'G', seed, budget_ms=200.)
    else:
        if method == 'M0':
            mode, p, v = 'M0', None, None
        else:
            mode = next(m for m in ('NPIV', 'NPI', 'DIRECT') if method.startswith(m))
            train_seed = int(method[len(mode):])
            p, v = model_path(train_seed, 'policy'), model_path(train_seed, 'value')
        own = Controller(boxes, speed, seed, mode, p, v if mode == 'NPIV' else None)
    static = 1000*(time.perf_counter()-start)
    start = time.perf_counter()
    for item in packet['history'][:-1]:
        obs = observation(item)
        own.estimator.update(obs)
        if hasattr(own, 'observe_history'):
            own.observe_history(obs)
    history = 1000*(time.perf_counter()-start)
    ACTIVE = own
    return own, dict(static_setup_ms=static, history_initialization_ms=history,
                     total_ms=static+history, budget_ms=200.)


def request(entry, world, method, stage):
    seeds = np.random.SeedSequence([BASE_SEED, int(entry['geometry_history_sha256'][:8], 16),
                                   int(world == 'B')]).generate_state(3)
    return dict(key=f'{entry["id"]}_{world}_{method}', group=entry['id'], family=entry['family'],
        stage=stage, world=world, method=method, opponent='rootpuct',
        packet_path=entry['packet_path'], environment_path=entry['environment_path'],
        seed=int(seeds[0]), opponent_seed=int(seeds[1]), environment_seed=int(seeds[2]))


def episode(req, **kwargs):
    old = engine.initialize
    engine.initialize = initialize
    try:
        return engine.episode(req, **kwargs)
    finally:
        engine.initialize = old


def signature(c, action):
    keys = ('root_actions', 'root_visits', 'root_totals', 'root_squares',
            'root_counts', 'terminations', 'nodes', 'value_calls')
    return dict(action=action.tolist(), public_sha=c.public_signature(),
                stats={k:c.last_stats[k] for k in keys})


def preflight():
    global RANDOM_MODELS
    if (OUT/'pretraining_interface.json').exists():
        return read(OUT/'pretraining_interface.json')
    RANDOM_MODELS = True
    torch.set_num_threads(1)
    folder = OUT/'interface_random_models'; folder.mkdir(exist_ok=True)
    for s in SEEDS:
        for kind in ('policy', 'value'):
            torch.manual_seed(s)
            Network(kind).export(folder/f'{kind}_{s}.npz')
    warm_start = time.perf_counter()
    packet = read(entries('validation')[0]['packet_path'])
    for m in ('M0', 'NPI17', 'NPIV17', 'DIRECT17'):
        c, _ = initialize(packet, m, 317)
        c.act(observation(packet['history'][-1]), fixed_simulations=2)
        independent_traces(np.concatenate(c.traces) if c.traces else np.zeros((0,10)), c.public.boxes, c.public.speed)
    save(OUT/'warmup.json', dict(seconds=time.perf_counter()-warm_start,
        excluded_from_decision_timing=True, test_data_used=False, includes_JIT_and_import_setup=True))
    records = []
    closed_loop_checks = []
    for family in ('long_wall', 'wall_corner', 'multi_route'):
        entry = next(e for e in entries('validation') if e['family'] == family)
        packet, env = read(entry['packet_path']), read(entry['environment_path'])
        obs = observation(packet['history'][-1])
        g, _ = initialize(packet, 'G', 317); g.act(obs)
        for m in ('M0', 'NPI17', 'NPIV17', 'DIRECT17'):
            replay = []
            for world in env['worlds']:
                # True world differs; only their independently verified common
                # public history enters initialize/act. No environment argument.
                c, _ = initialize(packet, m, 317)
                a = c.act(obs, fixed_simulations=16)
                replay.append(signature(c, a))
                assert legal_action(obs.position, a, c.public.speed, c.public.boxes)
                assert not any(independent_traces(np.concatenate(c.traces) if c.traces else np.zeros((0,10)),
                                                  c.public.boxes, c.public.speed))
                for name in ('particles', 'particle_weights', 'ids', 'weights', 'rng'):
                    np.testing.assert_array_equal(getattr(c.estimator, name), getattr(g.estimator, name))
                if c.root is not None:
                    c.deadline = float('inf')
                    x = c.features(c.root)
                    for kind in ('policy', 'value'):
                        runtime = Runtime(model_path(17, kind))
                        net = Network(kind).double()
                        net.load_state_dict({k:torch.from_numpy(v) for k,v in runtime.arrays.items()})
                        tx = {k:torch.from_numpy(v)[None] for k,v in x.items()}
                        with torch.no_grad():
                            expected = net(tx)
                        actual = runtime(x)
                        if kind == 'policy':
                            for aa, bb in zip(actual, expected):
                                np.testing.assert_allclose(aa, bb.numpy()[0], atol=1e-12, rtol=0)
                        else:
                            np.testing.assert_allclose(actual, expected.numpy()[0], atol=1e-12, rtol=0)
                        rng = np.random.default_rng(317)
                        xp = {k:v.copy() for k,v in x.items()}
                        order = rng.permutation(64)
                        xp['particles'], xp['weights'] = xp['particles'][order], xp['weights'][order]
                        order = rng.permutation(8)
                        xp['maps'], xp['map_mask'] = xp['maps'][order], xp['map_mask'][order]
                        np.testing.assert_allclose(np.asarray(runtime(xp)), np.asarray(actual), atol=1e-12, rtol=0)
            assert replay[0] == replay[1]
            c, setup = initialize(packet, m, 317)
            c.budget_ms = max(0., 200.-setup['total_ms'])
            start = time.perf_counter(); c.act(obs); ms = 1000*(time.perf_counter()-start)
            records.append(dict(family=family, method=m, replay_exact=True, two_hidden_worlds_isolated=True,
                D_matches_G=True, numpy_matches_torch=True, permutation_invariant=True,
                final_action_legal=True, simulation_legal=True, first_total_ms=setup['total_ms']+ms,
                act_ms=ms, initialization=setup, simulations=c.last_stats['simulations']))
            req = request(entry, 'A', m, 'engineering')
            first = episode(req, limit=6, fixed=[8]*6)
            second = episode(req, limit=6, fixed=[8]*6)
            fields = ('state','next_state','observation','own_action','opponent_action','particles','weights',
                      'estimator_sha256','search_rng_before','search_rng_after','simulation_sha256')
            exact = len(first['trace']) == len(second['trace']) and all(
                a[k] == b[k] for a,b in zip(first['trace'],second['trace']) for k in fields)
            assert first['summary']['failure'] is None and second['summary']['failure'] is None and exact
            save(OUT/'engineering/episodes'/(req['key']+'.json'), first)
            closed_loop_checks.append(dict(family=family, method=m, steps=len(first['trace']),
                fixed_simulations=8, exact_replay=exact, failure=None))
            emit(stage='interface', completed=len(records), total=12, method=m, family=family)
    c, _ = initialize(read(entries('validation')[0]['packet_path']), 'M0', 317)
    obs = observation(read(entries('validation')[0]['packet_path'])['history'][-1])
    c.act(obs, fixed_simulations=16)
    probability = probability_audit(c.estimator, obs, 317)
    assert all(probability[k] for k in ('root_sampling_passed', 'joint_sampling_passed',
        'transition_sampling_passed', 'filter_equivalent', 'frozen_probabilities_normalized'))
    assert probability['illegal_graph_edges'] == 0
    # Explicitly exercise public positive/negative observation conditioning.
    c.deadline = float('inf')
    prediction = c._prediction(c.root)
    points = c.estimator.graph[0]
    reachable = np.flatnonzero(prediction.sum(1)>0)
    known = int(reachable[0]); at = points[known]+np.array([0., .01])
    # Choose a lawful observation location, not a hidden-world-derived actor.
    if not legal_action(at, np.zeros(2), c.public.speed, c.public.boxes):
        at = points[known].copy()
    posterior = c._posterior(prediction, at, known)
    assert np.isclose(posterior[known].sum(), 1.)
    public_methods = [Controller.propose, Controller.select, Controller.rollout_action, Controller.features]
    assert not any(name in inspect.signature(f).parameters for f in public_methods
                   for name in ('target', 'target_id', 'mode', 'environment', 'state'))
    result = dict(passed=True, records=records, closed_loop_replays=closed_loop_checks, probability=probability,
        positive_observation_conditioning=True, no_hidden_parameters_in_action_API=True,
        random_models_only=True, training_started=False,
        actual_first_overruns=sum(r['first_total_ms'] > 200. for r in records),
        sensor_guard_disagreement_preserved=True)
    save(OUT/'pretraining_interface.json', result)
    RANDOM_MODELS = False
    return result


def teacher_query(packet, seed, trial):
    c, setup = initialize(packet, 'M0', seed)
    c.budget_ms = max(0., 2000.-setup['total_ms'])
    c.rng[:] = np.array([int(np.random.SeedSequence([seed, trial, BASE_SEED]).generate_state(1)[0])], np.int64)
    start = time.perf_counter()
    a = c.act(observation(packet['history'][-1]))
    elapsed = 1000*(time.perf_counter()-start)+setup['total_ms']
    traces = np.concatenate(c.traces) if c.traces else np.zeros((0, 10))
    errors = independent_traces(traces, c.public.boxes, c.public.speed)
    assert not np.any(errors)
    return c, dict(setup=setup, complete_ms=elapsed, over_2000ms=elapsed>2000.,
        action=a.tolist(), statistics=c.last_stats, simulated_transitions=len(traces),
        trace_sha256=hashlib.sha256(traces.tobytes()).hexdigest(), simulation_errors=errors.tolist())


def moments(c):
    if c.root is None or not sum(c.root.visits):
        raise RuntimeError('Teacher has no completed simulations')
    actions = np.asarray(c.root.actions)/c.public.speed
    w = np.asarray(c.root.visits, dtype=float); w /= w.sum()
    mean = w@actions
    variance = w@((actions-mean)**2)
    return mean, variance


def teacher_gate():
    path = OUT/'teacher_gate.json'
    if path.exists():
        return read(path)
    selected = [e for family in ('long_wall','wall_corner','multi_route')
                for e in [x for x in entries('train') if x['family'] == family][:4]]
    rows = []
    for i, entry in enumerate(selected):
        cached = OUT/'teacher_gate'/f'{i:02d}_{entry["family"]}.json'
        if cached.exists():
            rows.append(read(cached))
            continue
        packet = read(entry['packet_path'])
        seed = int(np.random.SeedSequence([BASE_SEED, i, 73]).generate_state(1)[0])
        a, qa = teacher_query(packet, seed, 0)
        b, qb = teacher_query(packet, seed, 1)
        ma, va = moments(a); mb, vb = moments(b)
        order = np.argsort(-np.asarray(a.root.visits), kind='stable')
        assert len(order) >= 2, 'Teacher has fewer than two legal proposals'
        best, runner = a.root.actions[int(order[0])], a.root.actions[int(order[1])]
        g, _ = initialize(packet, 'G', seed)
        ga = g.act(observation(packet['history'][-1]))
        a.rng[:] = np.array([int(np.random.SeedSequence([seed, 9401]).generate_state(1)[0])], np.int64)
        values, traces, probe_ms = a.independent_rollouts([best, runner, ga], draws=96, budget_ms=2000.)
        all_traces = np.concatenate(traces) if traces else np.zeros((0, 10))
        errors = independent_traces(all_traces, a.public.boxes, a.public.speed)
        assert not np.any(errors)
        n = len(values)
        diff = values[:, 0]-values[:, 1] if n else np.zeros(0)
        gap = float(diff.mean()) if n else 0.
        se = float(diff.std(ddof=1)/np.sqrt(n)) if n >= 2 else None
        lower = gap-1.96*se if se is not None else None
        distance = float(np.linalg.norm(ma-mb))
        signal = bool(n >= 24 and lower > 0.)
        stable = distance <= .5
        row = dict(group=entry['id'], family=entry['family'], seed=seed,
            query_A=qa, query_B=qb, visit_mean_A=ma.tolist(), visit_variance_A=va.tolist(),
            visit_mean_B=mb.tolist(), visit_variance_B=vb.tolist(), mean_action_distance=distance,
            mean_action_legal_A=legal_action(a.root.at, ma*a.public.speed, a.public.speed, a.public.boxes),
            probe_actions=[best.tolist(), runner.tolist(), ga.tolist()], probe_returns=values.tolist(),
            complete_paired_draws=n, probe_ms=probe_ms, heldout_gap=gap, heldout_gap_se=se,
            heldout_gap_lower95=lower, signal=signal, stable=stable, passed=signal and stable,
            probe_simulated_transitions=len(all_traces), simulation_errors=errors.tolist())
        rows.append(row)
        save(OUT/'teacher_gate'/f'{i:02d}_{entry["family"]}.json', row)
        emit(stage='teacher_gate', completed=i+1, total=12,
             passed_states=sum(r['passed'] for r in rows), signal=signal, stable=stable,
             paired_draws=n, gap=gap, lower95=lower, training_started=False)
    result = dict(passed=sum(r['passed'] for r in rows)>=7,
        passed_states=sum(r['passed'] for r in rows), required_states=7,
        signal_states=sum(r['signal'] for r in rows), stable_states=sum(r['stable'] for r in rows),
        states=12, records=rows, test_outcomes_read=False,
        decision='continue_to_labels' if sum(r['passed'] for r in rows)>=7 else 'stop_teacher_signal_insufficient')
    save(path, result)
    return result


def collect_and_label():
    class Collector(Geometry):
        def __init__(self, boxes, speed, seed):
            super().__init__(boxes, speed, 'G', seed, budget_ms=200.)
            self.records, self.packets = [], []
            self.behavior_rng = np.random.default_rng(seed+812)

        def observe_history(self, obs):
            self.records.append(dict(step=obs.step, position=obs.position.tolist(),
                target=None if obs.target is None else obs.target.tolist(), remaining=obs.remaining))

        def act(self, obs, fixed_simulations=None):
            self.observe_history(obs)
            action = super().act(obs, fixed_simulations=0)
            self.packets.append(dict(map=self.public.boxes.tolist(), motion=dict(pursuer_axis_limit=self.public.speed),
                                     history=list(self.records)))
            if self.behavior_rng.random() < .2:
                for _ in range(64):
                    candidate = self.behavior_rng.uniform(-self.public.speed, self.public.speed, 2)
                    if legal_action(obs.position, candidate, self.public.speed, self.motion_boxes):
                        return candidate
                return np.zeros(2)
            return action

    original_initialize = globals()['initialize']
    for split in ('train', 'validation'):
        for i, entry in enumerate(entries(split)):
            packet = read(entry['packet_path'])
            captured = {}
            def collection_initialize(packet, method, seed):
                start = time.perf_counter()
                c = Collector(packet['map'], packet['motion']['pursuer_axis_limit'], seed)
                for r in packet['history'][:-1]:
                    obs = observation(r); c.estimator.update(obs); c.observe_history(obs)
                captured['own'] = c
                ms = 1000*(time.perf_counter()-start)
                return c, dict(static_setup_ms=ms, history_initialization_ms=0., total_ms=ms, budget_ms=200.)
            old = engine.initialize; engine.initialize = collection_initialize
            try:
                payload = engine.episode(request(entry, 'A', 'G', 'collection'))
            finally:
                engine.initialize = old
            assert payload['summary']['failure'] is None
            save(OUT/'collection'/split/(entry['id']+'.json'), payload)
            packets = captured['own'].packets
            # Include public pre-handoff frames, including visible/just-lost.
            packets = [dict(map=packet['map'], motion=packet['motion'], history=packet['history'][:j+1])
                       for j in range(len(packet['history'])-1)]+packets
            indexes = np.unique(np.linspace(0, len(packets)-1, min(32,len(packets))).astype(int))
            for j in indexes:
                public_packet = packets[int(j)]
                c, query = teacher_query(public_packet, int(BASE_SEED+i*101+j), 0)
                if c.root is None:
                    # Same visible-stage geometric action; no claimed search label.
                    mu = np.asarray(query['action'])/c.public.speed
                    variance = np.zeros(2)
                    from learning.nte_d_local import public_features
                    x = public_features(c.public.boxes, c.estimator.particles,
                        c.estimator.particle_weights, np.asarray(public_packet['history'][-1]['position']),
                        public_packet['history'], c.public.speed, public_packet['history'][-1]['remaining'],
                        min(8, public_packet['history'][-1]['remaining']), True)
                    value, value_stats = 0., dict(source='visible branch excluded from value loss')
                    vx = x
                    value_mask = 0.
                else:
                    mu, variance = moments(c)
                    c.deadline = float('inf'); x = c.features(c.root)
                    # Train leaf values at public simulated nodes too, rather
                    # than training horizon-eight values then querying them at
                    # an unseen remaining horizon.
                    choices = [n for n in c.nodes if n.depth > 0]
                    if choices and int(j)%2:
                        desired = 1+int(j)%7
                        c.root = min(choices, key=lambda n:abs(n.depth-desired))
                    vx = c.features(c.root)
                    c.rng[:] = np.array([int(BASE_SEED+i*101+j+99991)], np.int64)
                    values, _, elapsed = c.independent_rollouts([c.rollout_action(c.root)], draws=64, budget_ms=2000.)
                    assert len(values) >= 8
                    value = float(values[:,0].mean())
                    value_stats = dict(source='independent public-model geometry MC', draws=len(values),
                        variance=float(values[:,0].var()), milliseconds=elapsed,
                        public_simulated_depth=c.root.depth, remaining_horizon=c.horizon-c.root.depth)
                    value_mask = 1.
                folder = OUT/'data'/split/entry['id']; folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder/f'{int(j):03d}.npz', **x,
                                    **{'v_'+k:v for k,v in vx.items()}, mean=mu, variance=variance,
                                    value=np.array(value), value_mask=np.array(value_mask))
                save(folder/f'{int(j):03d}.json', dict(group=entry['id'], public_packet=public_packet,
                    teacher=query, value_statistics=value_stats,
                    mean_action_legal=legal_action(np.asarray(public_packet['history'][-1]['position']),
                        mu*c.public.speed,c.public.speed,c.public.boxes)))
            emit(stage='labels_'+split, maps_completed=i+1, maps_total=len(entries(split)))
    save(OUT/'data/manifest.json', dict(files={str(p):sha(p) for p in (OUT/'data').rglob('*') if p.is_file()},
        test_data_used=False, unit='whole map/public history', privileged_labels=False))


def dataset(split):
    files = sorted((OUT/'data'/split).rglob('*.npz'))
    items = [dict(np.load(p, allow_pickle=False)) for p in files]
    return {k:torch.from_numpy(np.stack([d[k] for d in items])).float() for k in items[0]}


def train():
    train_data, val_data = dataset('train'), dataset('validation')
    folder = OUT/'models'; folder.mkdir(exist_ok=True)
    records = []
    save(OUT/'training.json', dict(started=True, completed=False, records=[], test_data_used=False))
    for seed in SEEDS:
        for kind in ('policy', 'value'):
            torch.manual_seed(seed)
            model = Network(kind)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
            rng = np.random.default_rng(seed)
            best, selected, log = float('inf'), None, []
            def loss(data):
                features = data if kind == 'policy' else {k[2:]:v for k,v in data.items() if k.startswith('v_')}
                answer = model(features)
                if kind == 'policy':
                    mu, logvar = answer
                    return .5*((((mu-data['mean'])**2+data['variance'])*torch.exp(-logvar)+logvar).sum(1)).mean()
                mask = data['value_mask']
                return (((answer-data['value'])**2)*mask).sum()/mask.sum().clamp_min(1.)
            for epoch in range(1,31):
                model.train(); total, count = 0., 0
                for batch in np.array_split(rng.permutation(len(train_data['mean'])),
                                             max(1,int(np.ceil(len(train_data['mean'])/64)))):
                    d = {k:v[batch] for k,v in train_data.items()}
                    optimizer.zero_grad(); value = loss(d); value.backward(); optimizer.step()
                    total += float(value.detach())*len(batch); count += len(batch)
                model.eval()
                with torch.no_grad():
                    validation = float(loss(val_data))
                log.append(dict(epoch=epoch, train_loss=total/count, validation_loss=validation))
                if validation < best:
                    best, selected = validation, epoch
                    model.export(folder/f'{kind}_{seed}.npz')
                    torch.save(model.state_dict(), folder/f'{kind}_{seed}.pt')
                emit(stage='training', seed=seed, kind=kind, epoch=epoch, validation_loss=validation)
            record = dict(seed=seed, kind=kind, epochs=30, selected_epoch=selected,
                          best_validation_loss=best, log=log)
            records.append(record); save(OUT/'training'/f'{kind}_{seed}.json', record)
    save(OUT/'training.json', dict(started=True, completed=True, records=records,
        train_states=len(train_data['mean']), validation_states=len(val_data['mean']), test_data_used=False))
    save(OUT/'models_frozen.json', dict(files={str(p):sha(p) for p in folder.iterdir() if p.is_file()},
        selection='validation only', test_outcomes_read=False))


def evaluate():
    rows = []
    requests = []
    for i, entry in enumerate(entries('test')):
        for wi, world in enumerate(('A','B')):
            offset = (i+wi)%len(METHODS)
            for method in METHODS[offset:]+METHODS[:offset]:
                requests.append(request(entry, world, method, 'evaluation'))
    save(OUT/'evaluation/requests.json', requests)
    for i, req in enumerate(requests):
        payload = episode(req)
        path = OUT/'evaluation/episodes'/(req['key']+'.json'); save(path, payload)
        row = payload['summary']|dict(episode=str(path), episode_sha256=sha(path))
        rows.append(row); save(OUT/'evaluation/cache'/(req['key']+'.json'), row)
        emit(stage='evaluation', completed=i+1, total=len(requests), group=req['group'], method=req['method'])
        if row['failure'] is not None:
            save(OUT/'evaluation/results.json', rows)
            raise RuntimeError('Closed-loop engineering failure; stop without retuning')
    save(OUT/'evaluation/results.json', rows)
    return summarize(rows)


def validate_trained_interfaces():
    records = []
    for family in ('long_wall','wall_corner','multi_route'):
        entry = next(e for e in entries('validation') if e['family']==family)
        packet = read(entry['packet_path']); obs = observation(packet['history'][-1])
        for method in METHODS:
            if method == 'G':
                continue
            copies = []
            for _ in range(2):
                c, _ = initialize(packet, method, 317)
                action = c.act(obs, fixed_simulations=16)
                copies.append(signature(c, action))
                assert legal_action(obs.position,action,c.public.speed,c.public.boxes)
                traces = np.concatenate(c.traces) if c.traces else np.zeros((0,10))
                assert not any(independent_traces(traces,c.public.boxes,c.public.speed))
            assert copies[0] == copies[1]
            c, setup = initialize(packet, method, 317)
            c.budget_ms = max(0.,200.-setup['total_ms'])
            start = time.perf_counter(); c.act(obs); ms = 1000*(time.perf_counter()-start)
            records.append(dict(family=family,method=method,replay_exact=True,legal=True,
                                first_total_ms=setup['total_ms']+ms))
    result = dict(passed=True,records=records,test_data_used=False,
                  model_hashes=read(OUT/'models_frozen.json'))
    save(OUT/'trained_interface.json',result)
    return result


def summarize(rows):
    groups = sorted({r['group'] for r in rows})
    methods = {}
    def q(x):
        return {name:float(np.percentile(x,p)) if x else None for name,p in
                (('p50',50),('p95',95),('p99',99),('max',100))}
    for method in METHODS:
        rr = [r for r in rows if r['method']==method]
        lat = [v for r in rr for v in r['latencies_ms']]
        first = [r['initial_total_ms'] for r in rr]
        methods[method] = dict(games=len(rr), captures=sum(r['capture'] for r in rr),
            rediscoveries=sum(r['first_seen'] is not None for r in rr),
            cost=float(np.mean([r['cost'] for r in rr])),
            rediscovery_cost=float(np.mean([r['rediscovery_cost'] for r in rr])),
            latency=q(lat), first=q(first), over_200ms=sum(v>200 for v in lat),
            first_over_200ms=sum(v>200 for v in first))
    def arm(group, method, field='cost'):
        names = [method] if method in ('G','M0') else [f'{method}{s}' for s in SEEDS]
        return float(np.mean([r[field] for r in rows if r['group']==group and r['method'] in names]))
    pairs = [('NPI','M0'),('NPIV','NPI'),('NPIV','DIRECT')]
    diff = np.array([[arm(g,a)-arm(g,b) for a,b in pairs] for g in groups])
    rng = np.random.default_rng(BASE_SEED+1)
    boot = diff[rng.integers(len(groups),size=(10000,len(groups)))].mean(1)
    mean = diff.mean(0)
    radius = float(np.quantile(np.max(np.abs(boot-mean),axis=1),.95))
    comparisons = []
    for j,(a,b) in enumerate(pairs):
        per_seed = [float(np.mean([np.mean([r['cost'] for r in rows if r['group']==g and r['method']==f'{a}{s}'])-
                   np.mean([r['cost'] for r in rows if r['group']==g and r['method']==(b if b=='M0' else f'{b}{s}')])
                   for g in groups])) for s in SEEDS]
        observed_capture = np.mean([arm(g,a,'capture')-arm(g,b,'capture') for g in groups])>=0
        reliable = bool(mean[j]+radius<0 and mean[j]<=-1 and all(x<0 for x in per_seed) and observed_capture)
        comparisons.append(dict(pair=f'{a} - {b}', mean=float(mean[j]),
            simultaneous95=[float(mean[j]-radius),float(mean[j]+radius)], per_seed=per_seed,
            observed_capture_no_drop=bool(observed_capture), reliable_task_gain=reliable,
            conclusion='reliable_gain' if reliable else 'worse' if mean[j]-radius>0 else 'insufficient_evidence'))
    result = dict(groups=len(groups), methods=methods, comparisons=comparisons,
                  group_differences=diff.tolist(), independent_unit='public initial history')
    save(OUT/'evaluation/statistics.json', result)
    return result


def alarm(signum, frame):
    raise TimeoutError('Registered six-hour pipeline budget reached')


def finalize(reason, gate=None, stats=None, error=None):
    verification = verify()
    elapsed = 0. if PIPELINE_START is None else time.monotonic()-PIPELINE_START
    decision = dict(reason=reason, stopped=True, error=error,
        teacher_gate_passed=None if gate is None else gate['passed'],
        formal_training_started=bool(read(OUT/'training.json').get('started', False)) if (OUT/'training.json').exists() else False,
        trained_checkpoints=sorted(str(p) for p in (OUT/'models').glob('*.npz')),
        heldout_games=len(list((OUT/'evaluation/cache').glob('*.json'))),
        test_retuned=False, deployment_changed=False, D_changed=False,
        pipeline_wall_seconds=elapsed, cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+
            resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        threads=1, automatic_expansion=False)
    save(OUT/'decision.json', decision)
    if not (OUT/'training.json').exists():
        save(OUT/'training.json', dict(started=False, completed=False, reason=reason,
                                      seeds=list(SEEDS), epochs_completed=0))
    report = ['# 局部观测 NTE＋D 适配基线执行报告', '',
        f'本轮结论：`{reason}`。本轮保留冻结D、原G、真实RootPUCT3000对手、运动模型、深度8和奖励。', '',
        '已实现连续动作神经树扩展、公共动作—观测树及策略/价值接口；它是当前追捕任务的局部信念适配，不能称原论文完整复现。', '',
        '教师、网络及树内追捕策略只读取公开地图、合法历史和信念。模拟隐藏样本仅用于模拟推进、观测和奖励。', '']
    if gate is not None:
        report += [f'预定教师屏障要求12个开发状态中至少7个通过；实际 **{gate["passed_states"]}/12**。', '',
            '|拓扑|状态|有区分信号|重复搜索均值稳定|同时通过|', '|---|---:|---:|---:|---:|']
        for family in ('long_wall','wall_corner','multi_route'):
            rr = [r for r in gate['records'] if r['family']==family]
            report.append(f'|{family}|{len(rr)}|{sum(r["signal"] for r in rr)}|{sum(r["stable"] for r in rr)}|{sum(r["passed"] for r in rr)}|')
        report += ['', '信号以独立公共模型配对MC评价最高/次高访问动作：至少24对且差值均值−1.96SE>0。稳定性采用两次独立目标随机流搜索的访问加权动作均值差≤0.5。此为事前开发屏障，不是正式搜索优越性的统计检验。', '']
        if not gate['passed']:
            report += ['**按预定停止条件结束：未生成正式训练标签，未启动三个种子的训练，未运行独立测试闭环。** 临时随机网络只验收接口，不能当作训练结果。测试地图已按几何构造封存，未查看算法测试结果。', '',
                '该负面结果说明当前固定奖励、8步规划模型与2秒教师配置未提供足够稳定且可区分的监督；不能推出NTE或学习普遍无效，也不能据此比较最终NTE＋D与G的捕获率。', '',
                '不改用G标签绕过屏障，不追加训练或改变深度/奖励。下一项机制变更需另立协议；本配置到此停止。', '']
    if stats is not None:
        report += ['|方法|捕获/局数|重发现|完成成本|决策P95ms|首次最大ms|', '|---|---:|---:|---:|---:|---:|']
        for m,r in stats['methods'].items():
            report.append(f'|{m}|{r["captures"]}/{r["games"]}|{r["rediscoveries"]}|{r["cost"]:.3f}|{r["latency"]["p95"]:.2f}|{r["first"]["max"]:.2f}|')
        report += ['', '主要成组配对比较：', '']
        for c in stats['comparisons']:
            report.append(f'- {c["pair"]}：成本差{c["mean"]:.3f}，三项同时95%区间{c["simultaneous95"]}；{c["conclusion"]}。')
    interface = read(OUT/'pretraining_interface.json') if (OUT/'pretraining_interface.json').exists() else {}
    report += ['', f'接口验收：{interface.get("passed",False)}；随机模型首次完整决策超200ms记录：{interface.get("actual_first_overruns","未完成")}。保留D传感器数值保护与精确模拟传感器的既有边界差异。', '',
        f'正式流水线墙钟{elapsed:.2f}秒，CPU累计{decision["cpu_seconds"]:.2f}秒，峰值RSS {decision["peak_rss_kib"]} KiB。单线程CPU；JIT预热单列，不与首次地图/历史初始化混淆。', '',
        '原部署源文件、冻结D及旧候选网络和搜索验证交付散列核对通过。', '',
        '- [执行协议](PROTOCOL.md)、[配置及源散列](experiment.json)、[地图历史划分](bank/manifest.json)。',
        '- [接口验收](pretraining_interface.json)、[教师屏障](teacher_gate.json)、[训练状态](training.json)、[停止决定](decision.json)。',
        '- [交付核对](delivery_checks.json)。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    checks = dict(passed=True, sources_unchanged=True, protected_unchanged=True,
        frozen_D_unchanged=True, bank_hashes_unchanged=True,
        protected_files=len(verification['protected_hashes']),
        training_not_started_after_gate_failure=bool(gate is None or gate['passed'] or not decision['formal_training_started']),
        test_not_consumed_after_gate_failure=bool(gate is None or gate['passed'] or decision['heldout_games']==0))
    assert checks['training_not_started_after_gate_failure'] and checks['test_not_consumed_after_gate_failure']
    save(OUT/'delivery_checks.json', checks)
    paths = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ('DELIVERY_SHA256.json','status.json','run.log','.run.lock')]
    save(OUT/'DELIVERY_SHA256.json', dict(files={str(p.relative_to(OUT)):sha(p) for p in paths}))
    emit(stage='complete', reason=reason, pipeline_wall_seconds=elapsed,
         trained_checkpoints=len(decision['trained_checkpoints']), heldout_games=decision['heldout_games'])


def main():
    global PIPELINE_START
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=('prepare','interface','all'), default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'.run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    config = prepare()
    if (OUT/'decision.json').exists():
        emit(stage='already_finished', decision=read(OUT/'decision.json')['reason'])
        return
    if args.stage == 'prepare':
        return
    preflight()
    if args.stage == 'interface':
        return
    PIPELINE_START = time.monotonic()
    signal.signal(signal.SIGALRM, alarm); signal.setitimer(signal.ITIMER_REAL, 21600.)
    gate = None
    try:
        gate = teacher_gate()
        if not gate['passed']:
            finalize('teacher_signal_insufficient', gate)
            return
        collect_and_label(); train(); validate_trained_interfaces()
        stats = evaluate()
        finalize('bounded_trial_complete', gate, stats)
    except Exception as exc:
        finalize('budget_stop' if isinstance(exc,TimeoutError) else 'execution_failure', gate, error=repr(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)


if __name__ == '__main__':
    main()
