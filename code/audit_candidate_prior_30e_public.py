"""Post-run public-input policy reconstruction; never changes frozen decisions."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np
from run_frozen_d_closed_loop import save, sha
from solvers.frozen_d_search_interface import Controller as Geometry, geometric_candidates, move
from solvers.learned_candidate_prior import PublicHistory, candidate_data
from learning.candidate_prior import Network, mixed_prior, geometry_scores
from audit_frozen_d_search_interface import observation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'current/l45_candidate_prior_30e_20261026'


def register():
    path = OUT/'supplement_registration.json'
    if path.exists():
        expected = json.loads(path.read_text())
        assert sha(__file__) == expected['source_sha256']
        return
    assert not (OUT/'evaluation/results.json').exists(), 'Register before complete outcomes'
    save(path, dict(source=str(Path(__file__).resolve()), source_sha256=sha(__file__),
        purpose='reconstruct policies using public observations only; descriptive G disagreement and score regret',
        complete_test_results_read=False, changes_to_primary_statistics_or_decision=False,
        wall_cap_seconds=300, added_training_or_evaluation_games=0))
    (OUT/'source_snapshot/audit_candidate_prior_30e_public.py').write_bytes(Path(__file__).read_bytes())


def audit():
    registration = json.loads((OUT/'supplement_registration.json').read_text())
    assert sha(__file__) == registration['source_sha256']
    def timeout(signum, frame):
        raise TimeoutError('Supplementary read-only public policy audit exceeded 300 seconds')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(300)
    started = time.perf_counter()
    rows = json.loads((OUT/'evaluation/results.json').read_text())
    models = {s: Network(OUT/'models'/f'prior_seed{s}.npz') for s in (17, 29, 43)}
    errors, records = [], []
    frames = hidden_frames = capture_frames = 0
    invisible_captures = 0
    for number, row in enumerate(rows):
        payload = json.loads(Path(row['episode']).read_text())
        task = payload['task']; packet = json.loads(Path(task['packet_path']).read_text())
        c = Geometry(packet['map'], packet['motion']['pursuer_axis_limit'], 'G', task['seed'])
        c.history = PublicHistory()
        for item in packet['history'][:-1]:
            obs = observation(item); c.estimator.update(obs); c.history.update(obs)
        e, p = c.estimator, c.public
        method = row['method']; hidden = disagree = network_disagree = 0
        regrets, network_regrets, root_coverage = [], [], []
        for frame in payload['trace']:
            # No state/next_state or hidden coordinates cross this actor block.
            obs = observation(frame['observation']); e.update(obs); c.history.update(obs)
            stats = frame['statistics']; recorded = np.asarray(frame['own_action'])
            if obs.target is not None:
                expected = move(obs.position, obs.target, p.speed, p.vertices, p.dist, p.boxes, np.zeros(2, np.int64))
            else:
                hidden += 1; hidden_frames += 1
                roots, base = geometric_candidates(obs.position, e.particles, e.particle_weights,
                    e.regions[e.ids], p.speed, p.vertices, p.dist, p.boxes, c.motion_boxes)
                scores = geometry_scores(obs.position, roots, e.particles, e.particle_weights,
                    p.speed, p.vertices, p.dist, p.boxes)
                nearest = int(np.argmin(np.max(np.abs(roots-recorded), axis=1)))
                if not np.allclose(roots[nearest], recorded, atol=1e-10, rtol=0.):
                    errors.append(dict(key=row['key'], step=obs.step, kind='not_a_public_legal_candidate'))
                disagree += int(not np.allclose(base, recorded, atol=1e-10, rtol=0.))
                regrets.append(float(scores.max()-scores[nearest]))
                if method == 'G':
                    expected = base
                elif method.startswith('NET'):
                    logits = models[int(method[3:])](candidate_data(c, obs, roots))
                    expected = roots[int(logits.argmax())]
                    if not np.allclose(logits, stats['root_logits'], atol=1e-10, rtol=1e-12):
                        errors.append(dict(key=row['key'], step=obs.step, kind='direct_logits_not_public_reproducible'))
                else:
                    if method == 'PUCT-U':
                        prior = np.ones(len(roots))/len(roots)
                    elif method == 'PUCT-G':
                        prior = mixed_prior(scores)
                    else:
                        logits = models[int(method[6:])](candidate_data(c, obs, roots))
                        prior = mixed_prior(logits)
                        pick = int(logits.argmax())
                        network_disagree += int(not np.allclose(roots[pick], base, atol=1e-10, rtol=0.))
                        network_regrets.append(float(scores.max()-scores[pick]))
                    if stats['root_prior'] and not np.allclose(prior, stats['root_prior'], atol=1e-12, rtol=1e-12):
                        errors.append(dict(key=row['key'], step=obs.step, kind='search_prior_not_public_reproducible'))
                    visits = np.asarray(stats['root_visits'])
                    root_coverage.append(float(np.mean(visits > 0)))
                    expected = roots[int(visits.argmax())] if stats['simulations'] else base
            if not np.allclose(expected, recorded, atol=1e-10, rtol=0.):
                errors.append(dict(key=row['key'], step=obs.step, kind='action_selection_mismatch'))
            frames += 1
            # Scoring-only terminal visibility check, after the policy block.
            if frame['capture']:
                capture_frames += 1
                invisible_captures += int(not frame['post_seen'])
        records.append(dict(key=row['key'], method=method, family=row['family'], capture=row['capture'],
            hidden_frames=hidden, decisions_different_from_G=disagree,
            mean_G_score_regret=float(np.mean(regrets)) if regrets else 0.,
            network_alone_disagreements=network_disagree,
            network_alone_mean_G_score_regret=float(np.mean(network_regrets)) if network_regrets else None,
            mean_root_fraction_visited=float(np.mean(root_coverage)) if root_coverage else None))
        if (number+1) % 24 == 0:
            print(json.dumps(dict(stage='public_policy_audit', completed=number+1, total=len(rows))), flush=True)
    summary = {}
    for method in sorted({r['method'] for r in records}):
        selected = [r for r in records if r['method'] == method]
        n = sum(r['hidden_frames'] for r in selected)
        summary[method] = dict(hidden_frames=n,
            G_action_disagreement_rate=sum(r['decisions_different_from_G'] for r in selected)/max(1, n),
            G_score_regret=sum(r['mean_G_score_regret']*r['hidden_frames'] for r in selected)/max(1, n),
            root_fraction_visited=float(np.mean([r['mean_root_fraction_visited'] for r in selected
                if r['mean_root_fraction_visited'] is not None])) if method.startswith('PUCT') else None)
    result = dict(passed=not errors and invisible_captures == 0, errors=errors, episodes=len(rows),
        public_policy_frames=frames, hidden_policy_frames=hidden_frames, captures=capture_frames,
        invisible_captures=invisible_captures, elapsed_seconds=time.perf_counter()-started,
        inputs='only public observations, frozen D, public map, model weights; selected search action checked against logged visits',
        causal_limit='public replay verifies inputs/selection; G score regret does not establish true task value or causal failure source',
        summary=summary, records=records)
    save(OUT/'public_policy_audit.json', result)
    lines = ['# 公开输入策略重放与描述性诊断', '',
        f'重放{len(rows)}局、{frames}个真实决策，其中{hidden_frames}个隐藏目标决策。通过：{result["passed"]}。',
        f'所有{capture_frames}次捕获的终态目标均可见；隔墙不可见捕获：{invisible_captures}。', '',
        '从公开观测独立重建冻结D、几何候选、网络特征/logits和根先验，核对网络直接动作及搜索按访问次数选出的动作。'
        '未将日志中的隐藏坐标交给策略重建。搜索模拟的整体随机过程由原冻结实现及在线几何检查保证，本审计不声称重放全部搜索。', '',
        '|方法|隐藏决策数|实际动作不同于同历史G|同历史G评分平均损失|平均访问根比例|', '|---|---:|---:|---:|---:|']
    for m, s in summary.items():
        coverage = '-' if s['root_fraction_visited'] is None else f'{100*s["root_fraction_visited"]:.1f}%'
        lines.append(f'|{m}|{s["hidden_frames"]}|{100*s["G_action_disagreement_rate"]:.1f}%|{s["G_score_regret"]:.4f}|{coverage}|')
    lines += ['', '上述G评分是描述性同历史对照，并非真实捕获价值。动作分歧能定位网络/搜索对几何策略的偏离，'
        '不能单凭分歧率或评分损失断言失败原因。没有根据这些结果重选模型或改变主要统计、停止决定。', '',
        '[完整重放记录](public_policy_audit.json)、[测试完成前登记的补充审计](supplement_registration.json)。']
    (OUT/'PUBLIC_POLICY_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'summary')}, ensure_ascii=False), flush=True)
    assert result['passed'], errors[:10]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('register', 'audit'))
    if parser.parse_args().command == 'register': register()
    else: audit()
