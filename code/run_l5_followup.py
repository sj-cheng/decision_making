"""Resumable execution of the authorized L5 attribution plan, 2026-11-07.

Frozen E46 artifacts are read-only. Scientific tasks use an isolated directory.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['TZ'] = 'Asia/Shanghai'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Existing NumPy inference needs no GPU.
os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse
import collections
import concurrent.futures as futures
import copy
import ctypes
import fcntl
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import pickle
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import run_e46_geometry_teacher as e46
from learning import l5_followup as follow
from learning.e46_geometry_teacher import (
    OLD_MODELS, make_controller as legacy_controller, prepare_public as legacy_public,
    draw_geometry, reference_values,
)

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'current/e46_geometry_teacher_20261105'
DEFAULT = ROOT/'current/l5_followup_20261107'
PLAN = ROOT/'总结文档/L5同池归因与执行验证实验计划_20261107.md'
SEEDS = (17, 29, 43)
OPPONENTS = ('fixed_D_kernel', 'rootpuct')
BASE = 1470520261107
OUT = None
CURSOR = collections.defaultdict(int)

read = e46.read
write = e46.write
sha = e46.sha
digest = e46.digest
save_pickle = e46.save_pickle
load_pickle = e46.load_pickle
stamp = e46.stamp


def event(out, kind, **kw):
    row = dict(time=stamp(), event=kind, **kw)
    with (Path(out)/'events.jsonl').open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False)+'\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


def configure(out):
    global OUT
    OUT = Path(out)
    torch.set_num_threads(1)
    e46.configure(out)
    import run_e45b_role_free as base
    base.make_controller = follow.make_controller


def specs():
    result = []
    for arm in ('A1', 'A2', 'A3', 'A4', 'A5', 'R1', 'R2'):
        for seed in (SEEDS if arm in ('A3', 'A5', 'R2') else (17,)):
            r = dict(name=f'{arm}_{seed}' if arm in ('A3', 'A5', 'R2') else arm,
                     arm=arm, train_seed=seed)
            if arm in ('A3', 'A5'):
                r['checkpoint'] = str(OLD/f'training/L5_{seed}_base/selected.npz')
            elif arm == 'R2':
                r['checkpoint'] = str(OLD_MODELS/f'single_{seed}.npz')
            result.append(r)
    assert len(result) == 13
    return result


def episode_task(entry, spec, stage, mode='sim', value=16, world='AB', opponent='rootpuct', **kw):
    seed = follow.stable_seed(BASE, entry['sha256'], world, 'environment')
    task = dict(kind='episode', entry=entry, group=entry['id'], stage=stage,
                case=entry['path'], world=world, opponent=opponent, seed=seed,
                repeat=0, method=spec['name'], arm=spec['arm'],
                train_seed=spec.get('train_seed', 17), checkpoint=spec.get('checkpoint'),
                model_round=12, round=0, split=entry['split'], family=entry['family'],
                suite=entry['suite'], mode=mode, budget_ms=value if mode == 'wall' else 200,
                fixed=value if mode == 'sim' else None, control_value=value)
    task.update(kw)
    return task


def packet_specs(packet, profile=False):
    return [dict(s, seed=packet['seed'], budget_ms=200, profile=profile) for s in specs()]


def stratified_refs(cfg, count):
    result=[]
    for suite in ('ID','OOD'):
        groups=collections.defaultdict(list)
        for ref in cfg['roots']:
            if ref['suite']==suite:groups[ref['phase']].append(ref)
        selected=[]
        while len(selected)<count//2 and any(groups.values()):
            for phase in ('visible','just_lost','hidden','post_capture'):
                if groups[phase] and len(selected)<count//2:selected.append(groups[phase].pop(0))
        result.extend(selected)
    return result


def select_roots(cfg, smoke=False):
    result = []
    for suite in ('ID', 'OOD'):
        maps = sorted([e for e in cfg['entries'] if e['split'] == 'development' and e['suite'] == suite], key=lambda e:e['id'])
        groups = []
        for entry in maps:
            packets = load_pickle(OLD/'packets/development'/(entry['id']+'.pkl.gz'))
            by_phase = collections.defaultdict(list)
            for p in packets:
                p = dict(p, suite=suite, family=entry['family'])
                by_phase[p['phase']].append(p)
            rows = []
            while any(by_phase.values()):
                for phase in ('visible', 'just_lost', 'hidden', 'post_capture'):
                    if by_phase[phase]:
                        rows.append(by_phase[phase].pop(0))
            groups.append(rows)
        chosen = []
        target = 2 if smoke else 128
        while len(chosen) < target and any(groups):
            for rows in groups:
                if rows and len(chosen) < target:
                    chosen.append(rows.pop(0))
        if len(chosen) != target:
            raise RuntimeError('Insufficient public roots for '+suite)
        result.extend(chosen)
    return result


def prepare(out, smoke=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cfg_path = out/'experiment.json'
    if cfg_path.exists():
        cfg = read(cfg_path)
        assert cfg['smoke'] == smoke
        verify(cfg)
        return cfg
    old = read(OLD/'experiment.json')
    e46.verify(OLD, old)
    roots = select_roots(old, smoke)
    refs = []
    for index, packet in enumerate(roots):
        path = out/'roots'/f'{index:03d}.pkl.gz'
        save_pickle(path, packet)
        refs.append(dict(index=index, path=str(path), sha256=sha(path),
                         group=packet['group'], suite=packet['suite'], phase=packet['phase']))
    models = {s['checkpoint']:sha(s['checkpoint']) for s in specs() if s.get('checkpoint')}
    for seed in SEEDS:
        info = read(OLD/f'training/L5_{seed}_base/result.json')
        assert sha(info['path']) == info['sha256']
    names = ('run_l5_followup.py', 'learning/l5_followup.py', 'l5_delay_evaluation.py', 'tests/test_l5_followup.py',
             'tests/fixtures/l5_retirement_regression_20261107.json')
    source = {str(ROOT/'code'/name):sha(ROOT/'code'/name) for name in names}
    dev = [e for e in old['entries'] if e['split'] == 'development']
    if smoke:
        dev = [next(e for e in dev if e['suite'] == suite) for suite in ('ID', 'OOD')]
    cfg = dict(experiment='L5-followup-20261107', created=stamp(), smoke=smoke,
               plan=str(PLAN), plan_sha256=sha(PLAN), roots=refs, development=dev,
               methods=specs(), source_hashes=source, model_hashes=models,
               legacy_source_hashes=old['source_hashes'],
               legacy_config=str(OLD/'experiment.json'), legacy_config_sha256=sha(OLD/'experiment.json'),
               max_states_per_episode=old.get('states_per_train_map',100),
               primary_comparisons=[['A3', 'A2'], ['A3', 'A5']], delta=.02,
               fixed_simulations=2 if smoke else 16, reference_worlds=2 if smoke else 32,
               confirmation_base=24 if smoke else 288, confirmation_cap=24 if smoke else 576,
               nominal_wall_ms=200, training_authorized=False,
               semantic_version='public-node-keyed-pools-v1; original nominal deadlines',
               gpu_preference=[3, 2, 1], gpu_forbidden=[0])
    for original in source:
        target = out/'source_snapshot'/Path(original).relative_to(ROOT/'code')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(original).read_bytes())
    write(cfg_path, cfg)
    write(out/'remaining_scope.json', dict(
        P6b=dict(status='engineering_pending', reason='Event-time delay adapter must pass zero-delay equivalence before scientific execution'),
        P7a=dict(status='interface_check_pending', candidate='existing Search+Learning, iteration79, 18_lidar_2obstacle'),
        P7b=dict(status='external_prerequisites_missing', reason='No verified current robot/bridge/control-period available; no vehicle commands issued'),
    ))
    return cfg


def verify(cfg):
    for group in ('source_hashes', 'model_hashes', 'legacy_source_hashes'):
        for path, expected in cfg[group].items():
            if sha(path) != expected:
                raise RuntimeError('Frozen artifact changed: '+path)
    if sha(cfg['legacy_config']) != cfg['legacy_config_sha256']:
        raise RuntimeError('Legacy registration changed')


def root_menu(packet):
    own, _, _ = follow.prepare_public(packet, dict(arm='A1', seed=packet['seed']), 0)
    own.deadline = float('inf')
    menu = own.pool(own.root).copy()
    return menu


def root_job(task, cfg):
    packet = load_pickle(task['packet'])
    menu = root_menu(packet)
    rows = []
    for spec in packet_specs(packet):
        if not spec['arm'].startswith('A'):
            continue
        for repeat in range(1 if spec['arm'] in ('A4', 'A5') or cfg['smoke'] else 3):
            seed = follow.stable_seed(packet['id'], repeat, BASE, 'search')
            own, action, cost = follow.prepare_public(packet, dict(spec, injected_root=menu.tolist(), search_seed=seed), cfg['fixed_simulations'])
            if own.last_stats.get('root_pool_sha256') != follow.pool_hash(menu):
                raise RuntimeError('Injected root menu changed across arms')
            rows.append(dict(method=spec['name'], arm=spec['arm'], train_seed=spec['train_seed'], repeat=repeat,
                             action=action.tolist(), stats=own.last_stats, **cost))
    menu_path = Path(OUT)/'menus'/f'{task["index"]:03d}.npz'
    e46.save_npz(menu_path, dict(actions=menu))
    return dict(packet=task['packet'], pool_sha256=follow.pool_hash(menu), menu=str(menu_path),
                rows=rows, phase=packet['phase'], suite=packet['suite'])


def profile_call(packet, spec, budget, fixed=None):
    construction_start=time.perf_counter()
    own = follow.make_controller(dict(map=packet['map'], speed=packet['speed']), dict(spec, seed=packet['seed'], budget_ms=budget, profile=True))
    construction_ms=1000*(time.perf_counter()-construction_start)
    started = time.perf_counter()
    for obs in packet['records'][:-1]:
        own.observe(obs)
    history = 1000*(time.perf_counter()-started)
    started = time.perf_counter(); cpu = time.process_time()
    action = own.act(packet['records'][-1], fixed_simulations=fixed)
    return dict(action=action.tolist(), stats=own.last_stats, history_ms=history,construction_ms=construction_ms,
                act_ms=1000*(time.perf_counter()-started), cpu_ms=1000*(time.process_time()-cpu))


def profile_job(task, cfg):
    packet = load_pickle(task['packet']); rows = []
    for spec in specs():
        for budget in ([200] if cfg['smoke'] else [100, 200]):
            for repeat in range(1 if cfg['smoke'] else 3):
                rows.append(dict(method=spec['name'], arm=spec['arm'], budget=budget,
                                 repeat=repeat, **profile_call(packet, spec, budget)))
    return dict(rows=rows, packet_id=packet['id'], phase=packet['phase'])


def anomaly_job(task, cfg):
    source = read(task['source'])
    payload = load_pickle(source['result']['episode'])
    worst = int(np.argmax([t['latency_ms'] for t in payload['trace']]))
    case = read(source['task']['case'])
    original = source['task']
    own = legacy_controller(case, dict(original, arm='POOL_LEARNED'))
    own = follow.instrument(own)
    for obs in case['history'][:-1]:
        own.observe(obs)
    before = []
    for frame in payload['trace'][:worst]:
        start = time.perf_counter()
        own.act(frame['observation'])
        before.append(1000*(time.perf_counter()-start))
    start = time.perf_counter(); cpu = time.process_time()
    action = own.act(payload['trace'][worst]['observation'])
    return dict(mode='full_public_observation_prefix_replay; wall-dependent search RNG/cache not exact saved runtime state',
                original_latency_ms=payload['trace'][worst]['latency_ms'], source=task['source'],
                observation_step=payload['trace'][worst]['observation']['step'],
                replay_prefix=len(before), prefix_act_ms=before,
                act_ms=1000*(time.perf_counter()-start), cpu_ms=1000*(time.process_time()-cpu),
                action=action.tolist(), stats=own.last_stats)


def calibration_job(task, cfg):
    refs = cfg['roots'][:1] if cfg['smoke'] else stratified_refs(cfg,6)
    rows = []
    for ref in refs:
        packet = load_pickle(ref['path']); samples = []
        # Untimed warmup excludes import/Numba first compilation from comparison.
        profile_call(packet, dict(arm='R1'), 200, 2)
        for _ in range(3):
            samples.append(profile_call(packet, dict(arm='R1'), 200, cfg['fixed_simulations'])['act_ms'])
        rows.append(dict(group=str(ref['index']), samples_ms=samples))
    return dict(rows=rows)


def reference_job(task, cfg):
    packet = load_pickle(task['packet'])
    root_result = read(task['root_result'])['result']
    with np.load(root_result['menu'], allow_pickle=False) as f:
        menu = f['actions']
    extra = []
    for arm in ('GEOM', 'PRIOR'):
        own, _, _ = legacy_public(packet, dict(arm=arm, seed=packet['seed'], train_seed=17), 0)
        own.deadline = float('inf')
        own.proposal_rng = np.random.default_rng(follow.stable_seed(packet['id'], arm, 'reference-menu'))
        for _ in range(16):
            extra.append(draw_geometry(own, own.root) if arm == 'GEOM' else own.propose(own.root))
    actions = list(menu.copy())
    for a in extra:
        if not any(np.max(np.abs(a-b)) < 1e-10 for b in actions):
            actions.append(a)
    actions = np.asarray(actions)
    worlds = cfg['reference_worlds']
    select = reference_values(packet, actions, worlds, follow.stable_seed(packet['id'], 'selection'))
    report = reference_values(packet, actions, worlds, follow.stable_seed(packet['id'], 'report'))
    best = int(np.argmax(select.mean(1)))
    pool_best = int(np.argmax(select[:len(menu)].mean(1)))
    selected = []
    for arm in ('A4', 'A5', 'A3'):
        row = next(r for r in root_result['rows'] if r['arm'] == arm and r['train_seed'] == 17 and r['repeat'] == 0)
        selected.append(dict(source=arm, action=row['action']))
    selected.append(dict(source='model_reference_selected', action=actions[best].tolist()))
    unique = []
    for r in selected:
        match = next((v for v in unique if np.max(np.abs(np.asarray(v['action'])-r['action'])) < 1e-10), None)
        if match:
            match['sources'].append(r['source'])
        else:
            unique.append(dict(action=r['action'], sources=[r['source']]))
    target = Path(OUT)/'reference'/f'{task["index"]:03d}.npz'
    e46.save_npz(target, dict(actions=actions, selection_returns=select, report_returns=report))
    return dict(path=str(target), packet=task['packet'], menu_count=len(menu), reference_count=len(actions),
                worlds=worlds, reference_selected=best, pool_selected=pool_best,
                reference_report=float(report[best].mean()), pool_report=float(report[pool_best].mean()),
                coverage_difference=float(report[best].mean()-report[pool_best].mean()),
                interventions=unique,
                scope='model-local, independent selection/report worlds; finite reference not global optimum')


def bank_job(task, cfg):
    from e45f_scenarios import ID_FAMILIES, OOD_FAMILIES, new_id_case, shifted_case, geometry_key
    out = Path(OUT); known = set(); scanned = 0
    for bank in sorted((ROOT/'current').glob('*/bank')):
        if bank.parent == out:
            continue
        for path in bank.rglob('*.json'):
            try:
                item = read(path)
                if isinstance(item, dict) and 'map' in item:
                    known.add(geometry_key(item['map'])); scanned += 1
            except (ValueError, KeyError, TypeError):
                continue
    entries = []; attempts = []
    quantities = dict(confirmation=(cfg['confirmation_cap']//6, cfg['confirmation_cap']//8),
                      delay=(4, 3), opponent=(4, 3))
    for split, (idn, oodn) in quantities.items():
        for family in (*ID_FAMILIES, *OOD_FAMILIES):
            if cfg['smoke'] and family not in (ID_FAMILIES[0],OOD_FAMILIES[0]):continue
            is_id = family in ID_FAMILIES
            count = idn if is_id else oodn
            if cfg['smoke']:
                count = 1
            accepted = 0
            for index in range(20000):
                if accepted == count:
                    break
                try:
                    case = (new_id_case if is_id else shifted_case)(family, index, 'test', follow.stable_seed(BASE, split))
                    key = geometry_key(case['map'])
                    if key in known:
                        raise ValueError('duplicate geometry')
                except (AssertionError, ValueError) as exc:
                    attempts.append(dict(split=split, family=family, index=index, reason=str(exc)))
                    continue
                known.add(key)
                case.update(id=f'l5f_{split}_{family}_{accepted:03d}', split=split,
                            suite='ID' if is_id else 'OOD')
                target = out/'bank'/split/(case['id']+'.json'); write(target, case)
                entries.append(dict(id=case['id'], split=split, suite=case['suite'], family=family,
                                    family_index=accepted, path=str(target), sha256=sha(target),
                                    geometry_sha256=key, speed=case['speed']))
                accepted += 1
                write(out/'bank_progress.json', dict(time=stamp(), split=split, family=family,
                      accepted=accepted, target=count, all_maps=len(entries), old_maps_scanned=scanned))
            if accepted != count:
                raise RuntimeError('Could not construct registered new maps')
    manifest = dict(entries=entries, attempts=attempts, old_maps_scanned=scanned,
                    acceptance='geometry/history legality/connectivity only; controller outcomes never queried')
    write(out/'bank/manifest.json', manifest)
    return dict(manifest=str(out/'bank/manifest.json'), maps=len(entries), old_maps_scanned=scanned)


def intervention_job(task, cfg):
    """One root action intervention, paired posterior world and fixed continuation."""
    from corner_estimator_scenarios import original_model
    from solvers.e45_team_adapter import public_observation
    from solvers.e45b_joint_nte import Controller
    from solvers.frozen_d_search_interface import target_step
    from solvers.root_audit_puct import RootPUCT
    from run_frozen_d_closed_loop import legal_action
    packet = load_pickle(task['packet']); boxes = np.asarray(packet['map'])
    seed = follow.stable_seed(BASE, packet['id'], task['posterior_world'], 'intervention-world')
    sampler, _, _ = legacy_public(packet, dict(arm='PRIOR', seed=packet['seed'], train_seed=17), 0)
    sampler.world_rng = np.random.default_rng(seed)
    obs0 = packet['records'][-1]
    p = original_model(packet['map'], packet['speed'])
    state = np.zeros((p.state_dim, 1)); state[4:8, 0] = np.asarray(obs0['positions']).ravel()
    state[8, 0] = obs0['step']; state[p.active_idxs, 0] = np.r_[obs0['active_e'], obs0['active_p']]
    ids, modes = sampler.sample_world()
    for j in range(2):
        state[2*j:2*j+2, 0] = sampler.graphs[j][0][ids[j]] if obs0['active_e'][j] else [-9.8, -9.8]
    p._update_observation_memory(state, state, 0.)
    actual = public_observation(int(state[8,0]), state[4:8,0].reshape(2,2), state[:4,0].reshape(2,2),
                                np.asarray(obs0['active_p']), np.asarray(obs0['active_e']), boxes)
    if actual['reports'] != obs0['reports'] or not p.is_valid(state) or p.get_capture_pairs(state):
        raise RuntimeError('Posterior world is inconsistent with public root or active-capture rules; not silently resampled')
    own = Controller(packet['map'], packet['speed'], packet['seed'],
                     OLD_MODELS/'single_17.npz', use_policy=True, use_value=False, direct=True)
    for obs in packet['records'][:-1]:
        own.observe(obs)
    opponent = RootPUCT([None]*4, None, 3000, 0., 0.) if task['opponent'] == 'rootpuct' else None
    if opponent:
        opponent.seeded_state(p, seed+17, seed+29)
    transition = [np.array([(seed+104729*j+1) % 4294967296], np.int64) for j in range(2)]
    rows = []; remaining = int(obs0['remaining'])
    for step in range(min(remaining, 2 if cfg['smoke'] else remaining)):
        ap = state[p.active_idxs[2:],0] > .5; ae = state[p.active_idxs[:2],0] > .5
        obs = public_observation(int(state[8,0]), state[4:8,0].reshape(2,2), state[:4,0].reshape(2,2), ap, ae, boxes)
        proposal = own.act(obs, fixed_simulations=0)
        action = np.asarray(task['action']) if step == 0 else proposal
        if not all(legal_action(state[4+2*i:6+2*i,0], action[i], packet['speed'], boxes) for i in range(2)):
            raise RuntimeError('Illegal intervention/continuation action')
        if opponent:
            enemy = np.asarray(opponent.search(p, state.copy(), turn=0).best_action).reshape(-1)[:4].reshape(2,2).copy()
            shields = 0
            for j in range(2):
                if not ae[j]: enemy[j] = 0.
                elif not legal_action(state[2*j:2*j+2,0], enemy[j], 1., boxes):
                    enemy[j] = 0.; shields += 1
        else:
            enemy = np.zeros((2,2)); shields = 0
            for j in range(2):
                if ae[j]:
                    g = sampler.graphs[j]
                    ids[j], modes[j] = target_step(ids[j], modes[j], g[1], g[2], g[3], transition[j])
                    enemy[j] = g[0][ids[j]]-state[2*j:2*j+2,0]
                    if not legal_action(state[2*j:2*j+2,0], enemy[j], 1., boxes):
                        raise RuntimeError('Illegal fixed-kernel intervention target motion')
        nxt = p.step(state, np.r_[enemy.ravel(),action.ravel()][:,None], 1.)
        if not p.is_valid(nxt):
            raise RuntimeError('Intervention reached illegal physical state')
        rows.append(dict(observation=obs, action=action.tolist(), opponent_action=enemy.tolist(),
                         opponent_shields=shields, state=state[:,0].tolist(), next_state=nxt[:,0].tolist()))
        state = nxt
        if p.is_terminal(state): break
    complete = not (state[p.active_idxs[:2],0] > .5).any()
    path = Path(OUT)/'interventions'/(digest(task)+'.pkl.gz'); save_pickle(path, rows)
    return dict(utility=float(complete)-.0025*len(rows), double_capture=bool(complete),
                completion_cost=len(rows) if complete else remaining, decisions=len(rows),
                trace=str(path), trace_sha256=sha(path), posterior_seed=seed,
                scope='paired posterior action intervention; public Single17 direct continuation; not actual-world ground truth')


def sl_opponent(problem):
    from learning.oracles import get_oracles
    from solvers.root_audit_puct import RootPUCT
    folder = ROOT/'current/historymodels/18_lidar_2obstacle'
    paths = [str(folder/f'model_policy_l79_i{i}.pt') for i in range(4)]
    value = str(folder/'model_value_l79.pt')
    policies, critic = get_oracles(problem, policy_oracle_name='gaussian',
                                  policy_oracle_paths=paths, value_oracle_name='deterministic', value_oracle_path=value)
    return RootPUCT(policies, critic, 3000, .5, .1)


def opponent_interface_job(task, cfg):
    from corner_estimator_scenarios import original_model
    from run_frozen_d_closed_loop import legal_action
    entry = cfg['development'][0]; case = read(entry['path']); p = original_model(case['map'],case['speed'])
    folder = ROOT/'current/historymodels/18_lidar_2obstacle'
    files = [folder/f'model_policy_l79_i{i}.pt' for i in range(4)]+[folder/'model_value_l79.pt']
    try:
        fingerprints = {str(q):sha(q) for q in files}
        other = sl_opponent(p); other.seeded_state(p, BASE+17, BASE+29)
        state = np.asarray(case['worlds'][0]['initial']).reshape(-1,1)
        action = np.asarray(other.search(p,state.copy(),turn=0).best_action).reshape(-1)[:4].reshape(2,2)
        if not np.isfinite(action).all(): raise RuntimeError('Nonfinite legacy opponent')
        legal = [bool(legal_action(state[2*j:2*j+2,0],action[j],1.,np.asarray(case['map']))) for j in range(2)]
        return dict(supported=True, files=fingerprints, action=action.tolist(), legal_before_existing_shield=legal,
                    information='existing full-state Search+Learning opponent; not pursuer input',
                    exposure='not tuned on this followup; historical checkpoint exposure cannot be claimed absent')
    except Exception as exc:
        return dict(supported=False, reason=repr(exc), traceback=traceback.format_exc(),
                    scope='interface unavailable; do not replace with a new opponent algorithm')


def holdout_episode(task, out):
    import run_e45b_role_free as base
    from corner_estimator_scenarios import original_model
    original = base.RootPUCT
    case = read(task['case'])
    # base.episode's non-kernel branch calls this constructor; all other physics,
    # shielding, public belief replay and scoring remain unchanged.
    base.RootPUCT = lambda *args, **kwargs:sl_opponent(original_model(case['map'],case['speed']))
    try:
        return e46.run_episode(task,out)
    finally:
        base.RootPUCT = original


def delay_interface_job(task,cfg):
    from l5_delay_evaluation import evaluate
    import run_e45b_role_free as base
    entry=cfg['development'][0];case=read(entry['path']);checks=[]
    try:
        for arm in ('A4','A3'):
            spec=next(s for s in cfg['methods'] if s['arm']==arm)
            t=episode_task(entry,spec,'engineering_delay_zero',value=2,world='AB',opponent='rootpuct',limit=3)
            ordinary,_=base.episode(t)
            zero=evaluate(t,case,'zero',limit=3,fixed=2)
            np.testing.assert_allclose(zero['final_state'],ordinary['trace'][-1]['next_state'],atol=1e-10,rtol=0)
            for x,y in zip(zero['decisions'],ordinary['trace']):
                np.testing.assert_array_equal(x['action'],y['action'])
            checks.append(dict(arm=arm,zero_delay_equal=True,steps=len(ordinary['trace'])))
        return dict(supported=True,checks=checks,semantics=dict(control_period_s=1.,deadline_s=.2,
                    timeout='plant stops at deadline, result discarded, actual solver compute retained',
                    sensors='integer epochs, queued public observations consumed in sequence',
                    opponent='same Root3000, ideal logical-time updates at integer epochs'))
    except Exception as exc:
        return dict(supported=False,checks=checks,reason=repr(exc),traceback=traceback.format_exc())


def delay_job(task,cfg):
    from l5_delay_evaluation import evaluate
    result=evaluate(task,read(task['case']),task['delay_mode'],limit=2 if cfg['smoke'] else None)
    target=Path(OUT)/'delay_episodes'/(digest(task)+'.pkl.gz');save_pickle(target,result)
    return dict(summary=result['summary'],episode=str(target),episode_sha256=sha(target))


def job_alarm(signum, frame):
    raise TimeoutError('Registered per-job watchdog 1800 s; result retained as engineering invalid/censored')


def run_job(task, out):
    global OUT
    OUT = Path(out); cfg = read(OUT/'experiment.json'); identity = digest(task)
    target = OUT/'results'/task['stage']/(identity+'.json')
    if target.exists():
        return read(target)
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, job_alarm)
    signal.alarm(1800 if not cfg['smoke'] else 300)
    try:
        kind = task['kind']
        if kind == 'episode':
            result = holdout_episode(task,OUT) if task['opponent'] == 'legacy_SL79' else e46.run_episode(task, OUT)
        elif kind == 'root':
            result = root_job(task, cfg)
        elif kind == 'profile':
            result = profile_job(task, cfg)
        elif kind == 'anomaly':
            result = anomaly_job(task, cfg)
        elif kind == 'calibration':
            result = calibration_job(task, cfg)
        elif kind == 'reference':
            result = reference_job(task, cfg)
        elif kind == 'bank':
            result = bank_job(task, cfg)
        elif kind == 'intervention':
            result = intervention_job(task, cfg)
        elif kind == 'opponent_interface':
            result = opponent_interface_job(task, cfg)
        elif kind == 'delay_interface':
            result = delay_interface_job(task,cfg)
        elif kind == 'delay':
            result = delay_job(task,cfg)
        else:
            raise ValueError(kind)
        row = dict(valid=True, task=task, result=result, wall_seconds=time.perf_counter()-started,
                   finished=stamp(), pid=os.getpid(), cpu=list(os.sched_getaffinity(0)))
    except Exception as exc:
        row = dict(valid=False, task=task, error=repr(exc), traceback=traceback.format_exc(),
                   wall_seconds=time.perf_counter()-started, finished=stamp(), pid=os.getpid())
        write(OUT/'failures'/task['stage']/(identity+'.json'), row)
    finally:
        signal.alarm(0)
    write(target, row)
    return row


def worker_init(out, cpus):
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    os.sched_setaffinity(0, {cpus.get()})
    configure(out)


def initialize(db, out, cfg):
    for ref in cfg['roots']:
        e46.enqueue(db, dict(kind='root', stage='P1_shared_roots', group=ref['group'],
                            packet=ref['path'], index=ref['index']))
    for entry in cfg['development']:
        for spec in cfg['methods']:
            for world in (('AB',) if cfg['smoke'] else ('AB','BA')):
                for opponent in (('fixed_D_kernel',) if cfg['smoke'] else OPPONENTS):
                    e46.enqueue(db, episode_task(entry, spec, 'P2_development_sim', world=world,
                                  opponent=opponent, value=cfg['fixed_simulations'], **({'limit':2} if cfg['smoke'] else {})))
    profiles = cfg['roots'][:1] if cfg['smoke'] else stratified_refs(cfg,32)
    for ref in profiles:
        e46.enqueue(db, dict(kind='profile', stage='P3_normal_profiles', group=ref['group'],
                            packet=ref['path'], index=ref['index']), 'timing')
    if not cfg['smoke']:
        for name in ('7c8ef269f7ff7abb716fbe464a111e884841367541f07434cb652f963930e7dd',
                     '9bf5c8233c55abb36b476481e8d2b30705dfe2d3abbe89bd1f35200f42526755',
                     '45ab394f44c4665eb72707ad53608b1849065be3e80aab4fbb7cfecd0dd6ea9f'):
            for repeat in range(3):
                e46.enqueue(db, dict(kind='anomaly', stage='P3_anomaly_replay', group=name,
                              source=str(OLD/'results/E8_final_wall'/(name+'.json')), repeat=repeat), 'timing')
    e46.enqueue(db, dict(kind='bank', stage='new_map_preparation', group='bank'))
    e46.enqueue(db, dict(kind='opponent_interface', stage='P7_opponent_interface', group='legacySL'))
    e46.enqueue(db, dict(kind='delay_interface', stage='P6_delay_interface', group='zero_delay_equivalence'))
    db.commit()


def dispatch(db, executor, active, capacity, out, pool_name, only=None):
    stages = [r[0] for r in db.execute('select distinct stage from jobs where pool=? and status="pending" order by stage', (pool_name,))]
    if only:
        stages = [s for s in stages if s in only]
    while len(active) < capacity and stages:
        stage = stages[CURSOR[pool_name] % len(stages)]; CURSOR[pool_name] += 1
        found = db.execute('select id,payload from jobs where pool=? and status="pending" and stage=? order by created,id limit 1', (pool_name, stage)).fetchone()
        if not found:
            stages.remove(stage); continue
        identity, raw = found
        db.execute('update jobs set status="running",updated=? where id=?', (time.time(), identity))
        active[executor.submit(run_job, json.loads(raw), str(out))] = identity
    db.commit()


def done_stage(db, stage, expected=None):
    rows = dict(db.execute('select status,count(*) from jobs where stage=? group by status', (stage,)))
    return bool(rows) and not any(rows.get(s, 0) for s in ('failed', 'pending', 'running')) and (expected is None or rows.get('done') == expected)


def enqueue_wall_development(db, cfg):
    for entry in cfg['development']:
        for spec in cfg['methods']:
            for world in (('AB',) if cfg['smoke'] else ('AB','BA')):
                for opponent in (('fixed_D_kernel',) if cfg['smoke'] else OPPONENTS):
                    e46.enqueue(db, episode_task(entry, spec, 'P2_development_wall', mode='wall', value=200,
                                  world=world, opponent=opponent, **({'limit':2} if cfg['smoke'] else {})), 'timing')


def freeze_sample(db, out, cfg):
    rows = e46.completed(db, 'P2_development_wall')
    cells = collections.defaultdict(lambda:collections.defaultdict(list))
    suite = {e['id']:e['suite'] for e in cfg['development']}
    for row in rows:
        t, s = row['task'], row['result']['summary']
        cells[t['arm']][t['group']].append(s['utility'])
    estimates = []; required = cfg['confirmation_base']
    for a, b in cfg['primary_comparisons']:
        sd = {}
        for ss in ('ID', 'OOD'):
            diffs = [float(np.mean(cells[a][g])-np.mean(cells[b][g])) for g in suite if suite[g] == ss]
            sd[ss] = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.
        sigma = math.sqrt((sd['ID']**2+sd['OOD']**2)/2)
        n = math.ceil(((2.2414+.8416)*sigma/cfg['delta'])**2)
        required = max(required, n)
        estimates.append(dict(contrast=f'{a}-{b}', map_sd=sd, effective_sd=sigma, approximate_N=n))
    rounded = 24*math.ceil(required/24)
    actual = min(cfg['confirmation_cap'], max(cfg['confirmation_base'], rounded))
    frozen = dict(time=stamp(), N=actual, required_before_cap=rounded, cap=cfg['confirmation_cap'],
                  precision_limited=rounded > cfg['confirmation_cap'], estimates=estimates,
                  scope='fixed once from development; no confirmation outcomes inspected',
                  source_hash=sha(out/'experiment.json'))
    write(out/'sample_size_frozen.json', frozen)
    return frozen


def advance(db, out, cfg):
    reference_indices = set(r['index'] for r in stratified_refs(cfg,2 if cfg['smoke'] else 128))
    for row in e46.completed(db, 'P1_shared_roots'):
        t = row['task']
        if t['index'] in reference_indices:
            e46.enqueue(db, dict(kind='reference', stage='P4_model_reference', group=t['group'],
                          index=t['index'], packet=t['packet'], root_result=str(out/'results/P1_shared_roots'/(digest(t)+'.json'))))
    for row in e46.completed(db, 'P4_model_reference'):
        t = row['task']
        for index, action in enumerate(row['result']['interventions']):
            for opponent in OPPONENTS:
                for world in range(1 if cfg['smoke'] else 16):
                    e46.enqueue(db, dict(kind='intervention', stage='P4_real_interventions_r1', group=t['group'],
                                  packet=t['packet'], index=t['index'], candidate=index,
                                  action=action['action'], sources=action['sources'], opponent=opponent, posterior_world=world))
    calibrated = (out/'timing_calibration.json').exists() and read(out/'timing_calibration.json').get('passed')
    if calibrated:
        enqueue_wall_development(db, cfg)
    delay_interface=e46.completed(db,'P6_delay_interface')
    if delay_interface:
        scope=read(out/'remaining_scope.json');supported=delay_interface[0]['result']['supported']
        delay_complete=done_stage(db,'P6_event_time_delay',12 if cfg['smoke'] else 672)
        scope['P6b']=dict(status=('executed' if delay_complete else 'zero_delay_equivalence_passed') if supported else 'engineering_equivalence_failed',
                        evidence=str(out/'results/P6_delay_interface'/(digest(delay_interface[0]['task'])+'.json')),
                        reason=delay_interface[0]['result'].get('reason'))
        write(out/'remaining_scope.json',scope)
        if supported and calibrated and done_stage(db,'P5_confirmation_wall') and (out/'bank/manifest.json').exists():
            for entry in read(out/'bank/manifest.json')['entries']:
                if entry['split']!='delay':continue
                for spec in cfg['methods']:
                    if spec['arm'] not in ('A2','A3','A5'):continue
                    if cfg['smoke'] and spec['train_seed']!=17:continue
                    for world in (('AB',) if cfg['smoke'] else ('AB','BA')):
                        for mode in ('zero','measured'):
                            t=episode_task(entry,spec,'P6_event_time_delay',mode='wall',value=200,
                                           world=world,opponent='rootpuct',kind='delay',delay_mode=mode)
                            e46.enqueue(db,t,'timing')
    interface = e46.completed(db, 'P7_opponent_interface')
    if interface:
        scope = read(out/'remaining_scope.json')
        supported = interface[0]['result']['supported']
        opponent_complete=done_stage(db,'P7_holdout_opponent',336)
        scope['P7a'] = dict(status=('executed' if opponent_complete else 'interface_passed') if supported else 'interface_unavailable',
                           evidence=str(out/'results/P7_opponent_interface'/(digest(interface[0]['task'])+'.json')),
                           reason=interface[0]['result'].get('reason'))
        write(out/'remaining_scope.json',scope)
        if supported and calibrated and done_stage(db,'P5_confirmation_wall') and (out/'bank/manifest.json').exists():
            for entry in read(out/'bank/manifest.json')['entries']:
                if entry['split'] != 'opponent': continue
                for spec in cfg['methods']:
                    if spec['arm'] not in ('A2','A3','A5'): continue
                    for world in ('AB','BA'):
                        e46.enqueue(db,episode_task(entry,spec,'P7_holdout_opponent',mode='wall',value=200,
                                      world=world,opponent='legacy_SL79',**({'limit':2} if cfg['smoke'] else {})),'timing')
    count = len(cfg['development'])*(1 if cfg['smoke'] else 4)*13
    if done_stage(db, 'P2_development_wall', count) and (out/'bank/manifest.json').exists():
        frozen = read(out/'sample_size_frozen.json') if (out/'sample_size_frozen.json').exists() else freeze_sample(db, out, cfg)
        n = frozen['N']; entries = read(out/'bank/manifest.json')['entries']
        selected = [e for e in entries if e['split'] == 'confirmation' and e['family_index'] < n//(6 if e['suite'] == 'ID' else 8)]
        if cfg['smoke']:
            selected = [e for e in entries if e['split'] == 'confirmation']
        if not (out/'confirmation_manifest.json').exists():
            write(out/'confirmation_manifest.json', dict(frozen_at=stamp(), entries=selected, N=len(selected), before_test=True))
        for entry in selected:
            for spec in cfg['methods']:
                for world in (('AB',) if cfg['smoke'] else ('AB','BA')):
                    for opponent in (('fixed_D_kernel',) if cfg['smoke'] else OPPONENTS):
                        e46.enqueue(db, episode_task(entry, spec, 'P5_confirmation_wall', mode='wall', value=200,
                                      world=world, opponent=opponent, **({'limit':2} if cfg['smoke'] else {})), 'timing')
    db.commit()


def progress(out, db, phase='running'):
    stages = {}
    for stage, status, n in db.execute('select stage,status,count(*) from jobs group by stage,status'):
        stages.setdefault(stage, {})[status] = n
    report = dict(experiment='L5-followup-20261107', phase=phase, time=stamp(), pid=os.getpid(), stages=stages,
                  failures=sum(v.get('failed', 0) for v in stages.values()),
                  archived_failures=sum(v.get('archived_failed',0) for v in stages.values()),
                  no_new_training=True, gpu_jobs=[], remaining_scope=read(out/'remaining_scope.json'))
    if (out/'timing_calibration.json').exists():
        report['timing_calibration'] = read(out/'timing_calibration.json')
    if (out/'sample_size_frozen.json').exists():
        report['sample_size'] = read(out/'sample_size_frozen.json')
    write(out/'progress.json', report)
    lines = ['# L5后续实验实时进展', '', f'更新时间：{report["time"]}；状态：{phase}。', '',
             '|阶段|完成|运行|待运行|失败|', '|---|---:|---:|---:|---:|']
    for s, v in sorted(stages.items()):
        lines.append(f'|{s}|{v.get("done",0)}|{v.get("running",0)}|{v.get("pending",0)}|{v.get("failed",0)}|')
    lines += ['', '未完成的延迟/对手/实车前置见remaining_scope.json；没有运行就不标记完成。',
              f'当前失败：{report["failures"]}；已保留的修订前失效记录：{report["archived_failures"]}（见revisions）。',
              '固定次数/参考可高并行；墙钟批次仅使用校准通过的独立计时核。']
    (out/'LIVE.md').write_text('\n'.join(lines)+'\n')
    return report


def timing_gate(db, out, cfg, bulk_active):
    path = out/'timing_calibration.json'
    state = read(path) if path.exists() else dict(passed=False, round=0, workers=1, attempts=[], bulk_paused=False)
    if state['passed']:
        return state
    if not done_stage(db, 'P3_normal_profiles') or (not cfg['smoke'] and not done_stage(db, 'P3_anomaly_replay')):
        return state
    if not (out/'execution_version_frozen.json').exists():
        verify(cfg)
        write(out/'execution_version_frozen.json',dict(time=stamp(),version=cfg['semantic_version'],
              revision='no E46 algorithm mutation; common public-node-keyed pool adapter',
              deadline_scope='nominal act deadlines, not a hard real-time guarantee',
              P3_evidence='all registered profiling/replay tasks valid; inclusive components retained',
              source_hashes=cfg['source_hashes']))
    stage = f'P6a_calibration_{state["round"]}'
    if state.get('bulk_paused') and bulk_active:
        return state
    for repeat in range(state['workers']):
        e46.enqueue(db, dict(kind='calibration', stage=stage, group=str(repeat), repeat=repeat), 'timing')
    state['stage'] = stage; write(path, state); db.commit()
    if not done_stage(db, stage, state['workers']):
        return state
    rows = e46.completed(db, stage)
    samples = [t for r in rows for v in r['result']['rows'] for t in v['samples_ms']]
    if not (out/'timing_serial_reference.json').exists():
        write(out/'timing_serial_reference.json', dict(median=float(np.median(samples)), p95=float(np.percentile(samples,95)), rows=rows))
    ref = read(out/'timing_serial_reference.json')
    median = float(np.median(samples))/ref['median']; p95 = float(np.percentile(samples,95))/ref['p95']
    good = median <= 1.2 and p95 <= 1.3
    state['attempts'].append(dict(workers=state['workers'], median_ratio=median, p95_ratio=p95, passed=good, bulk_paused=state['bulk_paused']))
    capacity = 1 if cfg['smoke'] else 8
    if good and state['workers'] < capacity:
        state['workers'] *= 2; state['round'] += 1
    elif good:
        state['passed'] = True
    elif state['workers'] > 1:
        state['workers'] //= 2
        # The prior worker count passed under the same registered bulk load.
        state['passed'] = True
    elif not state['bulk_paused']:
        state['bulk_paused'] = True; state['round'] += 1
    else:
        raise RuntimeError('Fixed-work timing calibration failed even in isolation')
    write(path, state)
    return state


def summarize_final(db, out, cfg):
    rows = []
    for source in e46.completed(db, 'P5_confirmation_wall'):
        t, s = source['task'], source['result']['summary']
        rows.append(dict(arm=t['arm'], method=t['method'], group=t['group'], suite=t['suite'],
                         world=t['world'], opponent=t['opponent'], seed=t['train_seed'],
                         double=float(s['double_capture']), utility=s['utility'], cost=s['completion_cost'],
                         decisions=s['decisions'], overruns=s['online_overruns']))
    cells = collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rows:
        cells[r['arm']][r['group']].append(r)
    maps = {r['group']:r['suite'] for r in rows}
    deltas = []; comparisons = []
    for a, b in cfg['primary_comparisons']:
        d = {suite:np.asarray([np.mean([r['utility'] for r in cells[a][g]])-np.mean([r['utility'] for r in cells[b][g]])
                              for g in sorted(maps) if maps[g] == suite]) for suite in ('ID','OOD')}
        comparisons.append(dict(method=a, reference=b, mean=float(sum(v.mean() for v in d.values())/2)))
        deltas.append(d)
    rng = np.random.default_rng(BASE+900)
    indices = {suite:rng.integers(len(deltas[0][suite]),size=(10000,len(deltas[0][suite]))) for suite in ('ID','OOD')}
    boot = np.stack([sum(v[suite][indices[suite]].mean(1) for suite in ('ID','OOD'))/2 for v in deltas],1)
    radius = float(np.quantile(np.max(np.abs(boot-np.asarray([c['mean'] for c in comparisons])),axis=1),.95))
    for c in comparisons:
        c['simultaneous95'] = [c['mean']-radius, c['mean']+radius]
    table = []
    for arm in sorted(cells):
        for suite in ('ID','OOD'):
            groups = [g for g in cells[arm] if maps[g] == suite]
            table.append(dict(arm=arm,suite=suite,maps=len(groups),
                **{key:float(np.mean([np.mean([r[key] for r in cells[arm][g]]) for g in groups])) for key in ('double','utility','cost')}))
    write(out/'final/results.json',dict(rows=rows,table=table,comparisons=comparisons,smoke=cfg['smoke'],
          scope='two predeclared overall utility contrasts, map-stratified family intervals; fixed three training seeds'))
    lines = ['# L5同池独立确认结果'+('（工程验证，不作科学结论）' if cfg['smoke'] else ''),'','|方法|范围|双捕率|效用|完成成本|','|---|---|---:|---:|---:|']
    for r in table:
        lines.append(f'|{r["arm"]}|{r["suite"]}|{100*r["double"]:.2f}%|{r["utility"]:.5f}|{r["cost"]:.2f}|')
    lines += ['', '主要效用比较：', '', '|比较|均值差|同时95%区间|','|---|---:|---|']
    for c in comparisons:
        lines.append(f'|{c["method"]}−{c["reference"]}|{c["mean"]:+.5f}|{c["simultaneous95"]}|')
    (out/'final/REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'final/CORE_COMPLETE.json',dict(finished=stamp(),episodes=len(rows),smoke=cfg['smoke'],sha256=sha(out/'final/results.json'),
          scope='P5 core confirmation only; optional/external prerequisites separately tracked'))


def idle_next_action(db, gate):
    """An empty queue between calibration levels is not a terminal workflow."""
    if db.execute('select count(*) from jobs where status in ("pending","running")').fetchone()[0]:
        return 'wait'
    if db.execute('select count(*) from jobs where status="failed"').fetchone()[0]:
        return 'stopped_incomplete'
    if not gate.get('passed'):
        return 'wait'
    return 'finish_core' if done_stage(db,'P5_confirmation_wall') else 'stopped_incomplete'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args(); out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'coordinator.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=prepare(out,args.smoke)
    if args.prepare_only:
        print(str(out/'experiment.json'));return
    verify(cfg);configure(out)
    layout=e46.resource_layout();layout['bulk']=sorted(layout['bulk']+layout['training'])
    layout['gpu_preference']=[3,2,1];layout['gpu_forbidden']=[0]
    write(out/'resources.json',layout)
    # The coordinator itself also stays off the twenty reserved logical CPUs.
    os.sched_setaffinity(0,set(layout['training']))
    if not (out/'timing_serial_reference.json').exists():
        previous=os.sched_getaffinity(0);os.sched_setaffinity(0,{layout['timing'][0]})
        try:
            reference=calibration_job({},cfg)
            values=[v for r in reference['rows'] for v in r['samples_ms']]
            write(out/'timing_serial_reference.json',dict(time=stamp(),median=float(np.median(values)),
                  p95=float(np.percentile(values,95)),rows=reference['rows'],scope='single worker before this experiment bulk starts'))
        finally:os.sched_setaffinity(0,previous)
    db=e46.connect(out);db.execute('update jobs set status="pending" where status="running"');db.commit()
    initialize(db,out,cfg)
    bulk_cpus=layout['bulk'][:2] if cfg['smoke'] else layout['bulk']
    timing_cpus=layout['timing'][:1] if cfg['smoke'] else layout['timing']
    # All kernels are CPU-only and torch/BLAS single-threaded before fork.
    ctx=mp.get_context('fork');bq=ctx.Queue();tq=ctx.Queue()
    for cpu in bulk_cpus:bq.put(cpu)
    for cpu in timing_cpus:tq.put(cpu)
    bulk=futures.ProcessPoolExecutor(len(bulk_cpus),mp_context=ctx,initializer=worker_init,initargs=(str(out),bq))
    timing=futures.ProcessPoolExecutor(len(timing_cpus),mp_context=ctx,initializer=worker_init,initargs=(str(out),tq))
    ba={};ta={};last=0.;phase='running'
    event(out,'coordinator_started',bulk_workers=len(bulk_cpus),timing_capacity=len(timing_cpus),reserved=layout['reserved'])
    try:
        while True:
            e46.collect_finished(db,ba);e46.collect_finished(db,ta)
            if time.time()-last>=15:
                advance(db,out,cfg);progress(out,db);last=time.time()
            gate=timing_gate(db,out,cfg,bool(ba))
            timing_pending=db.execute('select count(*) from jobs where pool="timing" and status in ("pending","running")').fetchone()[0]
            if not gate.get('bulk_paused') or (gate.get('passed') and not timing_pending):
                dispatch(db,bulk,ba,len(bulk_cpus),out,'bulk')
            if gate.get('passed'):
                if not gate.get('bulk_paused') or not ba:
                    dispatch(db,timing,ta,gate['workers'],out,'timing')
            else:
                only={gate['stage']} if gate.get('stage') else {'P3_normal_profiles','P3_anomaly_replay'}
                capacity=gate.get('workers',1) if gate.get('stage') else min(4,len(timing_cpus))
                dispatch(db,timing,ta,capacity,out,'timing',only)
            if not ba and not ta and not db.execute('select count(*) from jobs where status="pending"').fetchone()[0]:
                advance(db,out,cfg)
                decision=idle_next_action(db,gate)
                if decision=='wait':
                    time.sleep(1)
                    continue
                if decision=='finish_core':
                    verify(cfg);summarize_final(db,out,cfg);phase='core_finished';break
                phase='stopped_incomplete';break
            time.sleep(1)
    except BaseException as exc:
        phase='stopped_incomplete'
        event(out,'coordinator_interrupted',error=repr(exc))
        raise
    finally:
        progress(out,db,phase);bulk.shutdown(wait=True);timing.shutdown(wait=True)


if __name__=='__main__':
    main()
