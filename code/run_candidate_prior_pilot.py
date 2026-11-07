"""Bounded, resumable G-supervised candidate-prior training and closed-loop pilot."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import fcntl
import hashlib
import json
import time
from pathlib import Path
import numpy as np

import run_search_value_validation as engine
from run_frozen_d_closed_loop import save, sha
from corner_estimator_scenarios import construct
from verify_l45_occlusion_scenarios import verify_case
from audit_frozen_d_search_interface import observation
from solvers.frozen_d_search_interface import Controller as Geometry, geometric_candidates, loader
from solvers.learned_candidate_prior import Controller as Search, Direct, PublicHistory, candidate_data
from learning.candidate_prior import geometry_scores, SEEDS, EPOCHS, TEMPERATURE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'current/l45_candidate_prior_train_20261026'
METHODS = ('G', 'PUCT-U', 'PUCT-G')+tuple(f'NET{s}' for s in SEEDS)+tuple(f'PUCT-L{s}' for s in SEEDS)
BASE_SEED = 91020261026


def prepare():
    if (OUT/'experiment.json').exists():
        verify_frozen()
        return
    assert not (OUT/'evaluation').exists(), 'Cannot register after evaluation'
    known_histories, known_maps = set(), set()
    for p in (ROOT/'current').glob('**/controller_inputs/*.json'):
        packet = json.loads(p.read_text())
        if 'map' in packet and 'history' in packet:
            known_histories.add(hashlib.sha256(json.dumps(dict(map=packet['map'], history=packet['history']), sort_keys=True).encode()).hexdigest())
            known_maps.add(hashlib.sha256(json.dumps(packet['map'], sort_keys=True).encode()).hexdigest())
    entries, checks = [], []
    for split_index, (split, repetitions) in enumerate((('train', 8), ('validation', 2), ('test', 2))):
        for f, family in enumerate(('long_wall', 'wall_corner', 'multi_route')):
            for rep in range(repetitions):
                H = 4 if rep % 2 == 0 else 6
                speed = 1.1 if (rep//2+rep) % 2 == 0 else 1.3
                seed = int(np.random.SeedSequence([BASE_SEED, split_index, f, rep]).generate_state(1)[0])
                rng = np.random.default_rng(seed)
                height = float(rng.uniform(6.2, 6.9) if H == 4 else rng.uniform(8.15, 8.85))
                env, packet = construct(family, H, speed, seed, height, rep % 2, pause=rep % 2)
                check = verify_case(env, packet)
                assert check['passed'], (env['id'], check['reasons'])
                digest = hashlib.sha256(json.dumps(dict(map=packet['map'], history=packet['history']), sort_keys=True).encode()).hexdigest()
                map_digest = hashlib.sha256(json.dumps(packet['map'], sort_keys=True).encode()).hexdigest()
                assert digest not in known_histories and map_digest not in known_maps
                known_histories.add(digest); known_maps.add(map_digest)
                folder = OUT/'bank'/split
                for kind, value in (('controller_inputs', packet), ('environment', env)):
                    save(folder/kind/(env['id']+'.json'), value)
                entries.append(dict(id=env['id'], family=family, split=split, map_sha256=map_digest,
                    geometry_history_sha256=digest, packet_path=str(folder/'controller_inputs'/(env['id']+'.json')),
                    environment_path=str(folder/'environment'/(env['id']+'.json'))))
                checks.append(check)
    save(OUT/'bank/manifest.json', dict(entries=entries, split_unit='entire map and all public histories/worlds',
        counts=dict(train=24, validation=6, test=6), test_outcomes_read=False))
    save(OUT/'bank_acceptance.json', dict(passed=True, records=checks, geometry_only=True))
    names = ('run_candidate_prior_pilot.py', 'learning/candidate_prior.py', 'solvers/learned_candidate_prior.py',
        'tests/test_candidate_prior.py', 'L45_CANDIDATE_PRIOR_TRAIN_20261026.md')
    prior = ROOT/'current/l45_search_value_20261026'
    protected = json.loads((prior/'experiment.json').read_text())['source_hashes'].copy()
    protected.update(json.loads((prior/'experiment.json').read_text())['protected_sources'])
    protected.update({str(prior/p): sha(prior/p) for p in ('REPORT.md', 'final_decision.json', 'HANDOFF.md', 'DELIVERY_SHA256.json')})
    config = dict(version='candidate_prior_pilot_v1', train_seeds=list(SEEDS), epochs=EPOCHS,
        train_groups=24, validation_groups=6, test_groups=6, test_worlds=2, evaluation_games=108,
        methods=list(METHODS), budget_ms=200., horizon=8, cpuct=1.5, prior_floor=.1,
        root_prior_only=True, deeper_prior='uniform', reward='first rediscovery + capture', discount=.95,
        teacher='strong G hard action + soft score distribution; no FB labels',
        labels='public-only G, hidden decision frames in public prefixes and G closed loops',
        collection_opponent='unchanged RootPUCT3000, world A, one paired stream per map',
        optimizer=dict(name='Adam', lr=.001, batch_states=64, weight_decay=.0001),
        loss='soft CE(score temperature 1) + 0.5 hard CE(G argmax)',
        model='64 public features; shared 64-64-ReLU scalar candidate MLP; no candidate index input',
        checkpoint='minimum validation loss within fixed 80 epochs per seed; test untouched',
        evaluation_order='cyclic rotation across map/world, preserves complete paired blocks',
        training_device='cpu', inference_device='cpu NumPy exported MLP',
        evidence='exploratory held-out-map pilot, not independent confirmation',
        continuation_rule='seed-averaged PUCT-L cost <= both G and PUCT-G minus 1 step, no observed capture drop vs either, all 3 seed cost differences negative vs both, zero engineering failures',
        bootstrap=dict(draws=10000, seed=BASE_SEED+1, unit='public history; retain worlds and model seeds',
            primary_pairs=['PUCT-L vs G', 'PUCT-L vs PUCT-G'], correction='max absolute centered simultaneous radius for both cost comparisons'),
        source_hashes={str(ROOT/'code'/p): sha(ROOT/'code'/p) for p in names},
        protected_hashes=protected,
        bank_hashes={str(p): sha(p) for p in (OUT/'bank').rglob('*.json')},
        frozen_D_changed=False, value_network=False, ppo=False, deployment_changed=False)
    save(OUT/'experiment.json', config)
    for name in names:
        p = OUT/'source_snapshot'/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT/'code'/name).read_bytes())
    print(json.dumps(dict(stage='registered', train_groups=24, validation_groups=6, test_groups=6, games=108)), flush=True)


def verify_frozen():
    loader.verify()
    config = json.loads((OUT/'experiment.json').read_text())
    for name in ('source_hashes', 'protected_hashes', 'bank_hashes'):
        for path, expected in config[name].items():
            assert sha(path) == expected, ('Frozen file changed', path)
    return config


def entries(split):
    return [e for e in json.loads((OUT/'bank/manifest.json').read_text())['entries'] if e['split'] == split]


def model_path(seed):
    return OUT/'models'/f'prior_seed{seed}.npz'


class Teacher(Geometry):
    def __init__(self, boxes, speed, seed):
        super().__init__(boxes, speed, 'G', seed, budget_ms=200.)
        self.history = PublicHistory()
        self.examples = []

    def record(self, obs):
        if obs.target is not None:
            return
        e, p = self.estimator, self.public
        roots, best = geometric_candidates(obs.position, e.particles, e.particle_weights,
            e.regions[e.ids], p.speed, p.vertices, p.dist, p.boxes, self.motion_boxes)
        scores = geometry_scores(obs.position, roots, e.particles, e.particle_weights, p.speed, p.vertices, p.dist, p.boxes)
        assert np.array_equal(roots[int(np.argmax(scores))], best), 'Teacher mismatch'
        x = candidate_data(self, obs, roots)
        self.examples.append(dict(features=x, scores=scores, step=obs.step,
            public=dict(position=obs.position.tolist(), remaining=obs.remaining, target=None)))

    def act(self, obs, fixed_simulations=None):
        self.history.update(obs)
        answer = super().act(obs, fixed_simulations=0)
        self.record(obs)
        return answer


ACTIVE = None


def initialize(packet, method, seed):
    global ACTIVE
    started = time.perf_counter()
    boxes, speed = packet['map'], packet['motion']['pursuer_axis_limit']
    if method == 'TEACHER':
        own = Teacher(boxes, speed, seed)
    elif method == 'G':
        own = Geometry(boxes, speed, 'G', seed, budget_ms=200.)
    elif method.startswith('NET'):
        own = Direct(boxes, speed, seed, model_path(int(method[3:])))
    else:
        prior = {'PUCT-U': 'uniform', 'PUCT-G': 'geometry'}.get(method, 'learned')
        model = model_path(int(method[6:])) if prior == 'learned' else None
        own = Search(boxes, speed, seed, prior, model)
    static = 1000*(time.perf_counter()-started)
    stamp = time.perf_counter()
    for item in packet['history'][:-1]:
        obs = observation(item)
        own.estimator.update(obs)
        if hasattr(own, 'history'):
            own.history.update(obs)
        if method == 'TEACHER':
            own.record(obs)
    history = 1000*(time.perf_counter()-stamp)
    ACTIVE = own
    return own, dict(static_setup_ms=static, history_initialization_ms=history, total_ms=static+history, budget_ms=200.)


def task(entry, world, method, stage):
    wi = int(world == 'B')
    seeds = np.random.SeedSequence([BASE_SEED, int(entry['geometry_history_sha256'][:8], 16), wi]).generate_state(3)
    return dict(key=f'{entry["id"]}_{world}_{method}', group=entry['id'], family=entry['family'],
        world=world, method=method, opponent='rootpuct', stage=stage,
        packet_path=entry['packet_path'], environment_path=entry['environment_path'],
        seed=int(seeds[0]), opponent_seed=int(seeds[1]), environment_seed=int(seeds[2]))


def episode(request, **kwargs):
    old = engine.initialize
    engine.initialize = initialize
    try:
        return engine.episode(request, **kwargs)
    finally:
        engine.initialize = old


def status(**values):
    save(OUT/'status.json', dict(pid=os.getpid(), **values))
    print(json.dumps(values, ensure_ascii=False), flush=True)


def collect():
    if (OUT/'data/manifest.json').exists():
        for p, h in json.loads((OUT/'data/manifest.json').read_text())['files'].items():
            assert sha(p) == h
        return
    for split in ('train', 'validation'):
        for i, entry in enumerate(entries(split)):
            path = OUT/'data'/split/(entry['id']+'.npz')
            if path.exists():
                continue
            request = task(entry, 'A', 'TEACHER', 'collection')
            payload = episode(request)
            assert payload['summary']['failure'] is None, payload['summary']
            examples = ACTIVE.examples
            x = np.zeros((len(examples), 64, 64), np.float32)
            scores = np.full((len(examples), 64), -1000., np.float64)
            mask = np.zeros((len(examples), 64), bool)
            for j, sample in enumerate(examples):
                n = len(sample['scores'])
                x[j, :n] = sample['features']; scores[j, :n] = sample['scores']; mask[j, :n] = True
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, x=x, scores=scores, mask=mask,
                steps=np.array([s['step'] for s in examples]), group=np.array(entry['id']))
            temporary.replace(path)
            save(OUT/'collection/episodes'/(request['key']+'.json'), payload)
            status(stage='collect_'+split, completed=i+1, total=len(entries(split)), states=len(examples), group=entry['id'])
    paths = list((OUT/'data').rglob('*.npz'))
    manifest = dict(files={str(p): sha(p) for p in paths}, public_only_features=True, test_data_collected=False)
    save(OUT/'data/manifest.json', manifest)


def dataset(split):
    data = [np.load(OUT/'data'/split/(e['id']+'.npz'), allow_pickle=False) for e in entries(split)]
    result = tuple(np.concatenate([d[k] for d in data]) for k in ('x', 'scores', 'mask'))
    for d in data:
        d.close()
    return result


def train():
    for p, h in json.loads((OUT/'data/manifest.json').read_text())['files'].items():
        assert sha(p) == h
    if (OUT/'models_frozen.json').exists():
        for p, h in json.loads((OUT/'models_frozen.json').read_text())['files'].items():
            assert sha(p) == h
        return
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tx, ts, tm = dataset('train'); vx, vs, vm = dataset('validation')
    mean = tx[tm].mean(axis=0); scale = tx[tm].std(axis=0); scale[scale < .0001] = 1.
    xt, xv = [torch.from_numpy(np.clip((x-mean)/scale, -10., 10.).astype(np.float32)) for x in (tx, vx)]
    st, sv = torch.from_numpy(ts), torch.from_numpy(vs)
    mt, mv = torch.from_numpy(tm), torch.from_numpy(vm)
    def loss_fn(logits, scores, mask):
        logits = logits.masked_fill(~mask, -1e9)
        target = torch.softmax((scores/TEMPERATURE).masked_fill(~mask, -1e9), dim=-1)
        logp = torch.log_softmax(logits, dim=-1)
        return -(target*logp).sum(-1).mean()+.5*nn.functional.cross_entropy(logits, scores.argmax(-1))
    records = []
    for seed in SEEDS:
        metadata = OUT/'models'/f'prior_seed{seed}.json'
        if metadata.exists():
            record = json.loads(metadata.read_text()); assert sha(model_path(seed)) == record['sha256']
            records.append(record); continue
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=.001, weight_decay=.0001)
        rng = np.random.default_rng(seed)
        best_loss, best, logs = float('inf'), None, []
        for epoch in range(EPOCHS):
            net.train(); order = rng.permutation(len(tx)); losses = []
            for start in range(0, len(order), 64):
                idx = order[start:start+64]
                logits = net(xt[idx]).squeeze(-1)
                loss = loss_fn(logits, st[idx], mt[idx])
                opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
            net.eval()
            with torch.no_grad():
                pred = net(xv).squeeze(-1)
                vl = float(loss_fn(pred, sv, mv))
                accuracy = float((pred.masked_fill(~mv, -1e9).argmax(-1) == sv.argmax(-1)).float().mean())
            logs.append(dict(epoch=epoch+1, train_loss=float(np.mean(losses)), validation_loss=vl, validation_hard_accuracy=accuracy))
            if vl < best_loss:
                best_loss = vl; best = {k: v.detach().clone() for k, v in net.state_dict().items()}; best_epoch = epoch+1
        net.load_state_dict(best)
        arrays = dict(mean=mean.astype(float), scale=scale.astype(float))
        for i, layer in enumerate((net[0], net[2], net[4])):
            arrays[f'w{i}'] = layer.weight.detach().numpy().astype(float)
            arrays[f'b{i}'] = layer.bias.detach().numpy().astype(float)
        model_path(seed).parent.mkdir(parents=True, exist_ok=True)
        np.savez(model_path(seed), **arrays)
        from learning.candidate_prior import Network
        exported = Network(model_path(seed))
        differences = []
        with torch.no_grad():
            predictions = net(xv).squeeze(-1).numpy()
        for j in range(len(vx)):
            differences.append(float(np.max(np.abs(exported(vx[j])-predictions[j]))))
        max_difference = max(differences)
        assert max_difference < .001, max_difference
        logits = np.stack([exported(x) for x in vx]); logits[~vm] = -1e9
        labels = vs.argmax(-1); picks = logits.argmax(-1)
        regret = vs.max(-1)-vs[np.arange(len(vs)), picks]
        record = dict(seed=seed, epochs=EPOCHS, selected_epoch=best_epoch, sha256=sha(model_path(seed)),
            validation_loss=best_loss, validation_hard_accuracy=float(np.mean(labels == picks)),
            validation_G_score_regret=float(regret.mean()), numpy_export_max_logit_difference=max_difference,
            train_states=len(tx), validation_states=len(vx), test_read=False, logs=logs)
        save(metadata, record); records.append(record)
        status(stage='trained', seed=seed, selected_epoch=best_epoch, validation_accuracy=record['validation_hard_accuracy'])
    save(OUT/'training.json', dict(records=records, complete=True, model_seeds_are_not_independent_map_samples=True))
    save(OUT/'models_frozen.json', dict(files={str(p): sha(p) for p in (OUT/'models').glob('*')},
        test_outcomes_read=False, training_complete=True))


def evaluation_tasks():
    tasks = []
    for i, entry in enumerate(entries('test')):
        for wi, world in enumerate(('A', 'B')):
            shift = (2*i+wi) % len(METHODS)
            for method in METHODS[shift:]+METHODS[:shift]:
                tasks.append(task(entry, world, method, 'evaluation'))
    return tasks


def preflight():
    import importlib.util
    spec = importlib.util.spec_from_file_location('candidate_prior_preflight', ROOT/'code/tests/test_candidate_prior.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = module.run_preflight(OUT)
    save(OUT/'preflight.json', result)
    assert result['passed'], result


def evaluate():
    frozen = json.loads((OUT/'models_frozen.json').read_text())
    for p, h in frozen['files'].items():
        assert sha(p) == h
    preflight()
    requests = evaluation_tasks(); save(OUT/'evaluation/requests.json', requests)
    rows = []
    for i, request in enumerate(requests):
        cache = OUT/'evaluation/cache'/(request['key']+'.json')
        if cache.exists():
            row = json.loads(cache.read_text()); assert sha(row['episode']) == row['episode_sha256']
        else:
            payload = episode(request)
            path = OUT/'evaluation/episodes'/(request['key']+'.json')
            save(path, payload)
            row = payload['summary'] | dict(episode=str(path), episode_sha256=sha(path))
            save(cache, row)
        rows.append(row)
        status(stage='evaluation', completed=i+1, total=len(requests), method=row['method'], capture=row['capture'],
            cost=row['cost'], failures=sum(r['failure'] is not None for r in rows))
        if row['failure']:
            save(OUT/'evaluation/partial_results.json', rows)
            raise RuntimeError(('Engineering failure; preserved completed results', row['key'], row['failure']))
    save(OUT/'evaluation/results.json', rows)


def report():
    from audit_search_value_validation_v2 import audit
    checked = audit(OUT, 'evaluation')
    save(OUT/'evaluation/postrun_audit_v2.json', checked)
    verify_frozen()
    rows = json.loads((OUT/'evaluation/results.json').read_text())
    summary = {}
    for method in METHODS:
        selected = [r for r in rows if r['method'] == method]
        latencies = [v for r in selected for v in r['latencies_ms']]
        initial = [r['initial_total_ms'] for r in selected]
        summary[method] = dict(games=len(selected), captures=sum(r['capture'] for r in selected),
            rediscovered=sum(r['first_seen'] is not None for r in selected),
            cost=float(np.mean([r['cost'] for r in selected])), rediscovery_cost=float(np.mean([r['rediscovery_cost'] for r in selected])),
            decision_p95_ms=float(np.quantile(latencies, .95)), decision_max_ms=float(max(latencies)),
            initial_max_ms=max(initial), initial_overruns=sum(v > 200. for v in initial),
            decision_overruns=sum(v > 200. for v in latencies), engineering_failures=sum(r['failure'] is not None for r in selected),
            hidden_simulations=float(np.mean([v for r in selected for v in r['hidden_simulations']])))
    groups = [e['id'] for e in entries('test')]
    def value(group, methods, metric):
        return float(np.mean([float(r[metric]) for r in rows if r['group'] == group and r['method'] in methods]))
    learned = [f'PUCT-L{s}' for s in SEEDS]
    raw = {}
    for baseline in ('G', 'PUCT-G', 'PUCT-U'):
        raw[baseline] = {metric: np.array([value(g, learned, metric)-value(g, [baseline], metric) for g in groups])
            for metric in ('cost', 'rediscovery_cost', 'capture')}
    rng = np.random.default_rng(BASE_SEED+1)
    indices = rng.integers(0, len(groups), (10000, len(groups)))
    bootstrap = {b: {m: x[indices].mean(-1) for m, x in vals.items()} for b, vals in raw.items()}
    radius = float(np.quantile(np.max(np.stack([np.abs(bootstrap[b]['cost']-raw[b]['cost'].mean()) for b in ('G', 'PUCT-G')]), axis=0), .95))
    comparisons = {b: {m: dict(difference=float(x.mean()), ci95=np.quantile(bootstrap[b][m], [.025, .975]).tolist(),
        simultaneous_ci95=[float(x.mean()-radius), float(x.mean()+radius)] if b in ('G', 'PUCT-G') and m == 'cost' else None)
        for m, x in vals.items()} for b, vals in raw.items()}
    seed_checks = {str(s): {b: summary[f'PUCT-L{s}']['cost']-summary[b]['cost'] for b in ('G', 'PUCT-G')} for s in SEEDS}
    audit_passed = bool(checked.get('passed', not checked.get('errors')))
    timing_passed = all(not s['decision_overruns'] and not s['initial_overruns'] for s in summary.values())
    signal = bool(audit_passed and timing_passed and all(comparisons[b]['cost']['difference'] <= -1. and
        comparisons[b]['capture']['difference'] >= 0. for b in ('G', 'PUCT-G')) and
        all(v < 0. for d in seed_checks.values() for v in d.values()))
    statistics = dict(summary=summary, comparisons=comparisons, seed_cost_differences=seed_checks,
        bootstrap=dict(groups=6, draws=10000, preserve='two worlds and all three model seeds', simultaneous_radius=radius),
        independent_confirmation=False, model_seeds_not_sample_size=True)
    save(OUT/'evaluation/statistics.json', statistics)
    decision = dict(continue_to_independent_confirmation_design=signal, audit_passed=audit_passed,
        timing_passed=timing_passed, training_complete=True, evaluation_complete=True,
        independent_confirmation=False, deployment_changed=False,
        reason='pilot_signal_requires_new_confirmation' if signal else 'pilot_did_not_meet_frozen_continuation_rule')
    save(OUT/'decision.json', decision)
    lines = ['# 强G监督的候选先验网络：训练与闭环试验', '',
        '本轮由用户明确授权小规模训练。上一轮无网络搜索的停止结论保留；本轮冻结D、强G及部署。', '',
        '**当前结论：'+('出现开发信号，需独立确认。**' if signal else '未达到预登记继续门槛。**'), '',
        '24张训练地图、6张验证地图、6张全新测试地图；完整地图及其全部历史/世界不跨分区。三种已知拓扑的新实例，不是未见拓扑。',
        '每个模型仅学习强G；64维公开特征，64-64候选打分MLP。三个种子固定80轮，以验证损失选择检查点。无价值网络、PPO或奖励修改。',
        '搜索均为8步FB、200ms总预算、相同PUCT公式、深层均匀先验，仅根先验不同。G和网络直接控制共用合法几何候选；可见目标均使用相同几何跟踪。',
        '初次决策扣除地图、模型加载和公开历史回放；推理与先验准备计入决策。未对G补等待。', '',
        '|方法|捕获/12|重发现/12|失败惩罚完成成本|重发现成本|决策P95/最大ms|首次总计最大ms|决策/首次超200ms|隐藏帧平均模拟|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m, s in summary.items():
        lines.append(f'|{m}|{s["captures"]}|{s["rediscovered"]}|{s["cost"]:.2f}|{s["rediscovery_cost"]:.2f}|{s["decision_p95_ms"]:.2f}/{s["decision_max_ms"]:.2f}|{s["initial_max_ms"]:.2f}|{s["decision_overruns"]}/{s["initial_overruns"]}|{s["hidden_simulations"]:.1f}|')
    lines += ['', 'PUCT-L为三个种子的平均；每个种子完整保留12局。统计独立单位为6组公共历史，A/B与种子不增加样本量。', '',
        '|学习先验搜索相对基线|平均完成成本差|同时95%区间（前两项）|捕获率差|', '|---|---:|---|---:|']
    for b, v in comparisons.items():
        lines.append(f'|{b}|{v["cost"]["difference"]:+.2f}|{v["cost"]["simultaneous_ci95"]}|{100*v["capture"]["difference"]:+.2f}pp|')
    lines += ['', f'独立逐帧审计通过：{audit_passed}；实际总时限门槛通过：{timing_passed}。',
        '失败成本保留原剩余步数；不能用成功子样本或训练损失替代闭环效果。成组bootstrap区间仅用于探索，小样本及天花板不能证明捕获非劣。',
        '当前对手仍是全状态RootPUCT3000；网络只接收公开地图、历史摘要和冻结D带权信念的特征。', '',
        '- [冻结协议](experiment.json)、[几何验收](bank_acceptance.json)、[数据散列](data/manifest.json)。',
        '- [训练与验证](training.json)、[冻结模型](models_frozen.json)、[前置检查](preflight.json)。',
        '- [108局结果](evaluation/results.json)、[配对统计](evaluation/statistics.json)、[独立审计](evaluation/postrun_audit_v2.json)、[决定](decision.json)。', '',
        '复现：在项目根目录运行 `/home/sjc/miniconda3/envs/dm_env_py311/bin/python code/run_candidate_prior_pilot.py pipeline`。已完成局逐文件校验后复用；冻结源或输入改变则拒绝续跑。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    paths = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ('DELIVERY_SHA256.json', 'status.json', 'pipeline.log')]
    save(OUT/'DELIVERY_SHA256.json', {str(p): sha(p) for p in paths})
    status(stage='complete', **decision)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=('prepare', 'collect', 'train', 'evaluate', 'report', 'pipeline'))
    args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'current/.l45_search_value_timing.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prepare()
        if args.command == 'pipeline':
            collect(); train(); evaluate(); report()
        elif args.command != 'prepare':
            globals()[args.command]()


if __name__ == '__main__':
    main()
