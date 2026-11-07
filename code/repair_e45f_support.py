"""Audited E45-F completion; preserve original results and replace one invalid cell."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['TZ'] = 'Asia/Shanghai'
import fcntl
import json
import time
import traceback
from pathlib import Path
import numpy as np

import run_e45f_generalization_benchmark as f
import run_e45f_capacity_compat as capacity
import run_e45b_role_free as base
from solvers.e45f_support_refinement import RefinedTeamBelief, install as install_refinement

ORIGINAL = f.ROOT / 'current/e45f_generalization_benchmark_20261029'
OUT = ORIGINAL.with_name(ORIGINAL.name + '_support_repaired')


def status(stage, **kwargs):
    row = dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'), stage=stage, **kwargs)
    f.save(OUT / 'status.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def verify_original():
    previous = f.OUT
    try:
        f.OUT = ORIGINAL
        return capacity.verify()
    finally:
        f.OUT = previous


def audit_reused(rows):
    """Original episode SHA binds its successful independent public replay."""
    proofs = []
    for k, row in enumerate(rows):
        if not row.get('valid_episode', True):
            continue
        path = Path(row['episode'])
        if f.sha(path) != row['episode_sha256']:
            raise RuntimeError('Original episode hash mismatch: ' + row['key'])
        payload = f.read(path)
        if not row['public_replay_exact'] or payload['summary']['failure'] is not None:
            raise RuntimeError('Original replay audit missing')
        if payload['summary']['key'] != row['key'] or not payload['trace']:
            raise RuntimeError('Original identity or trace missing')
        if any(frame['stats'].get('team_belief', {}).get('adaptive_grid_refinement')
               for frame in payload['trace']):
            raise RuntimeError('Unexpected refined original episode')
        proofs.append(dict(key=row['key'], episode=str(path), sha256=row['episode_sha256'],
                           independent_public_replay_exact=True))
        if k % 500 == 0:
            status('checking_original_caches', checked=len(proofs), total=len(rows) - 1)
    return proofs


def complete():
    config = verify_original()
    original_rows = f.read(ORIGINAL / 'results.json')
    tasks = f.read(ORIGINAL / 'requests.json')
    if len(original_rows) != 6720 or {r['key'] for r in original_rows} != {t['key'] for t in tasks}:
        raise RuntimeError('Original planned identities are incomplete')
    bad = [r for r in original_rows if not r.get('valid_episode', True)]
    if len(bad) != 1 or bad[0]['reason'] != 'belief_support_exhaustion':
        raise RuntimeError('Unexpected original failure set')
    original_bad = bad[0]
    for path, digest in [(original_bad['public_evidence'], original_bad['public_evidence_sha256']),
                         (original_bad['arrays'], original_bad['arrays_sha256'])]:
        if f.sha(path) != digest:
            raise RuntimeError('Failure evidence changed')
    proof = f.read(original_bad['public_evidence'])
    manifest = dict(original=str(ORIGINAL), original_results_sha256=f.sha(ORIGINAL / 'results.json'),
                    original_statistics_sha256=f.sha(ORIGINAL / 'statistics.json'),
                    original_experiment_sha256=f.sha(ORIGINAL / 'experiment.json'),
                    amended_estimator='E45-F zero-support-only 0.25 lattice refinement',
                    scope='Every method uses the same refinement; successful arithmetic and RNG paths unchanged',
                    reuse_basis='Only the original fatal zero-support branch changes; successful independent replay audits are SHA-bound',
                    resolution=.25, sigma=.35, intent_persistence=.9, exported_samples=64,
                    full_negative_history_replayed=True, training=False, sources={})
    for relative in ['repair_e45f_support.py', 'solvers/e45f_support_refinement.py',
                     'tests/test_e45f_support_refinement.py', 'tests/fixtures/e45f_empty_support_public.json']:
        path = f.ROOT / 'code' / relative
        manifest['sources'][str(path)] = f.sha(path)
    f.save(OUT / 'AMENDMENT.json', manifest)
    f.save(OUT / 'original_invalid.json', original_bad)
    proofs = audit_reused(original_rows)
    f.save(OUT / 'reuse_audit.json', dict(verified=len(proofs), episodes=proofs,
                                        no_successful_episodes_reexecuted=True))

    f.OUT = OUT
    f.configure()
    install_refinement(OUT / 'belief_failures')
    # Warm the exact public failure before measuring online execution.
    warm = RefinedTeamBelief(proof['boxes'], proof['seed'])
    for obs in proof['history']:
        warm.update(obs)
    if len(warm.refinements) != 1:
        raise RuntimeError('Public failure no longer matches amendment')
    task = next(t for t in tasks if t['key'] == original_bad['key']) | dict(_output=str(OUT))
    own = f.factory(f.read(task['case']), task)
    case = f.read(task['case'])
    for obs in case['history'][:-1]:
        own.observe(obs)
    own.act(case['history'][-1], fixed_simulations=2)
    status('rerunning_failed_episode', reused=len(proofs), total=6720, key=task['key'])
    row = base.episode_job(task)
    payload = f.read(row['episode'])
    history = case['history'][:-1] + [frame['observation'] for frame in payload['trace']]
    if history[:len(proof['history'])] != proof['history']:
        raise RuntimeError('Rerun diverged before original failure; cannot silently replace')
    # Joint controllers do not expose estimator diagnostics in last_stats.
    # Reconstruct the intervention from a second exact public replay instead
    # of assuming that optional telemetry was written to the original runner.
    replay = RefinedTeamBelief(case['map'], task['seed'])
    for obs in case['history'][:-1]:
        replay.update(obs)
    interventions = []
    for frame in payload['trace']:
        before = len(replay.refinements)
        replay.update(frame['observation'])
        if replay.signature() != frame['belief_signature']:
            raise RuntimeError('Amended independent replay signature mismatch')
        if len(replay.refinements) > before:
            interventions.append(dict(step=frame['observation']['step'],
                                      refinements=replay.refinements[before:], latency_ms=frame['latency_ms']))
    if len(interventions) != 1 or interventions[0]['step'] != 79:
        raise RuntimeError('Unexpected refined execution path')
    row.update(valid_episode=True, support_amendment=str(OUT / 'AMENDMENT.json'),
               support_refinements=interventions, original_invalid_evidence=original_bad['public_evidence'])
    rows = [row if old['key'] == row['key'] else old for old in original_rows]
    stats = f.summarize(rows)
    if stats['primary_status'] != 'complete' or len(rows) != 6720:
        raise RuntimeError('Repaired primary comparisons remain incomplete')
    if stats['primary']['benchmark'] != f.read(ORIGINAL / 'statistics.json')['primary']['benchmark']:
        raise RuntimeError('Unaffected benchmark statistics changed')
    for path, digest in manifest['sources'].items():
        if f.sha(path) != digest:
            raise RuntimeError('Repair source changed during execution')
    verify_original()
    f.save(OUT / 'results.json', rows); f.save(OUT / 'statistics.json', stats)
    for suite in ['benchmark', 'generalization']:
        f.save(OUT / suite / 'results.json', [r for r in rows if r['suite'] == suite])
    f.save(OUT / 'completion_audit.json', dict(planned=6720, valid=6720, invalid=0,
           reused=len(proofs), rerun=1, all_keys_unique=len({r['key'] for r in rows}) == len(rows),
           original_public_history_prefix_exact=True, original_benchmark_statistics_exact=True,
           original_artifacts_hashes_verified=True, repaired_episode_public_replay_exact=row['public_replay_exact'],
           refinements=interventions, primary_comparisons={s:len(v['comparisons']) for s,v in stats['primary'].items()},
           original_result='6719 valid + 1 invalid, retained without edits', training=False))
    status('complete', completed=6720, total=6720, valid=6720, invalid=0, reused=6719, rerun=1,
           primary_status='complete')
    # The original report renderer is retained; its protocol links point back
    # to the original frozen inputs rather than copying or modifying them.
    f.report(rows, stats)
    report_path = OUT / 'REPORT.md'
    text = report_path.read_text()
    for name in ['PROTOCOL.md', 'experiment.json', 'layout_gallery.png', 'coverage_audit.json', 'timing/decision.json', 'run.log']:
        text = text.replace('](' + name + ')', '](' + str(ORIGINAL / name) + ')')
    notice = ('\n**2026-10-30修订完成：** 原批次保留6719有效＋1无效；本目录为明确修订后的估计器结果。'
              '原0.5网格在第79步漏掉合法遮挡区域，仅空支持时用0.25网格按全部公开负观测重新过滤。'
              '这一分辨率修订改变离散估计器，不称原冻结D完全不变。SHA核验复用6719局，失败1局按原地图、模型和种子重跑；'
              '到原失败点的公共历史逐项一致，修复局独立公共回放通过。6720/6720有效，两套各18项主比较完整。'
              '修复计算计入在线时延，任何超200ms如实保留。\n\n'
              '[修订说明](AMENDMENT.json) · [完成核对](completion_audit.json) · [原始失败](original_invalid.json)'
              ' · [原始批次](' + str(ORIGINAL / 'REPORT.md') + ')\n')
    report_path.write_text(text.replace('\n\n', '\n' + notice + '\n', 1))
    f.save(OUT / 'decision.json', dict(status='complete', primary_status='complete', attempted=6720,
                                     valid=6720, invalid=0, reused=6719, rerun=1, training=False))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'coordinator.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / 'failure.json').exists():
            (OUT / 'failure.json').replace(OUT / ('precompletion_failure_' + str(time.time_ns()) + '.json'))
        try:
            complete()
        except BaseException as exc:
            f.save(OUT / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
            status('failed', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
