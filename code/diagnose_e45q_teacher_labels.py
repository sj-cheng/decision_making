"""Read-only E45-Q label provenance and bounded teacher-search diagnostics.

No learner, optimizer, training rollout, or label rewrite is constructed.
All saved input artifacts are hashed before and after the audit.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import copy
import gzip
import hashlib
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / 'current/e45o_hasac_demonstration_20261101'
Q = ROOT / 'current/e45q_corrective_hasac_20261101'
LABEL_KEYS = ('obs', 'action', 'active')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')


def load_labels(path):
    with np.load(path, allow_pickle=False) as packet:
        return {key: packet[key].copy() for key in LABEL_KEYS}


def concatenate(parts):
    return {key: np.concatenate([part[key] for part in parts]) for key in LABEL_KEYS}


class Audit:
    def __init__(self):
        self.inputs = {}
        self.roots = {}
        self.query_count = 0
        self.simulated_transitions = 0

    def protect(self, path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        if expected is not None:
            assert actual == expected, str(path)
        self.inputs[str(path)] = actual
        return path

    def packet(self, path, expected=None):
        return read(self.protect(path, expected))

    def label_packet(self, path, expected=None):
        return load_labels(self.protect(path, expected))

    def metadata(self, row, payload, indices, source, train_seed=None, round_index=None):
        case = self.packet(row['case'])
        result = []
        for index in indices:
            frame = payload['trace'][index]
            stats = frame['stats']
            teacher = stats if source == 'original' else stats['teacher_query']
            prefix = case['history'][:-1] + [t['observation'] for t in payload['trace'][:index + 1]]
            item = dict(source=source, train_seed=train_seed, round=round_index,
                case=row['case'], group=row['group'], world=row['world'],
                environment_seed=row['seed'], trace_index=index,
                step=frame['observation']['step'], teacher_simulations=teacher['simulations'],
                observation_sha256=digest(frame['observation']),
                public_prefix_sha256=digest(prefix), belief_signature=frame['belief_signature'],
                episode=row.get('episode', row.get('source')),
                normalized_action=(np.asarray(frame['action']) / case['speed']).tolist()
                    if source == 'original' else teacher['normalized_teacher_action'])
            result.append(item)
            if index == 0:
                key = (row['group'], row['world'])
                self.roots.setdefault(key, dict(case=case, seed=row['seed'],
                    observation=frame['observation'], records=[]))['records'].append(item)
        return result

    def original(self):
        manifest = self.packet(O / 'demonstrations/manifest.json')
        parts, metadata = [], []
        for entry in manifest['episodes']:
            payload = self.packet(entry['source'], entry['source_sha256'])
            labels = self.label_packet(entry['archive'], entry['archive_sha256'])
            assert len(payload['trace']) == len(labels['obs']) == entry['transitions']
            for i, frame in enumerate(payload['trace']):
                expected = (np.asarray(frame['action']) / self.packet(entry['case'])['speed']).astype(np.float32)
                np.testing.assert_array_equal(labels['action'][i], expected)
                np.testing.assert_array_equal(labels['active'][i, :, 0], frame['observation']['active_p'])
            metadata += self.metadata(entry, payload, range(len(payload['trace'])), 'original')
            parts.append(labels)
        merged = concatenate(parts)
        frozen = self.label_packet(O / 'demonstrations/data.npz')
        for key in LABEL_KEYS:
            np.testing.assert_array_equal(merged[key], frozen[key])
        return merged, metadata

    def corrections(self, seed):
        folder = Q / 'imitation' / f'seed_{seed}'
        parts, metadata, queries, invalid = [], [], 0, 0
        for round_index in range(1, 5):
            manifest = self.packet(folder / f'round_{round_index}_collection.json')
            round_parts = []
            for row in manifest['episodes']:
                payload = self.packet(row['episode'], row['episode_sha256'])
                labels = self.label_packet(row['labels_path'], row['labels_sha256'])
                frames = payload['trace']
                valid_indices = [i for i, frame in enumerate(frames) if frame['stats']['teacher_query']['valid']]
                queries += len(frames)
                invalid += len(frames) - len(valid_indices)
                assert len(valid_indices) == len(labels['obs']) == row['valid_labels']
                for label_index, frame_index in enumerate(valid_indices):
                    frame = frames[frame_index]
                    teacher = frame['stats']['teacher_query']
                    assert teacher['simulations'] > 0 and teacher['public_belief_exact']
                    assert teacher['simulation_audit_passed'] and not teacher['teacher_actuates']
                    expected = np.clip(teacher['normalized_teacher_action'], -1, 1).astype(np.float32)
                    np.testing.assert_array_equal(labels['action'][label_index], expected)
                    np.testing.assert_array_equal(labels['active'][label_index, :, 0], frame['observation']['active_p'])
                metadata += self.metadata(row, payload, valid_indices, 'correction', seed, round_index)
                round_parts.append(labels)
            combined = concatenate(round_parts)
            frozen = self.label_packet(manifest['labels'], manifest['labels_sha256'])
            for key in LABEL_KEYS:
                np.testing.assert_array_equal(combined[key], frozen[key])
            parts.append(combined)
        merged = concatenate(parts)
        aggregate = self.label_packet(folder / 'aggregate_corrections.npz')
        for key in LABEL_KEYS:
            np.testing.assert_array_equal(merged[key], aggregate[key])
        return merged, metadata, dict(queries=queries, valid=len(merged['obs']), invalid=invalid)


def groups_for(data, actor):
    groups = {}
    for i, observation in enumerate(data['obs'][:, actor]):
        if data['active'][i, actor, 0] > 0:
            groups.setdefault(observation.tobytes(), []).append(i)
    return list(groups.values())


def lower_bound(data, groups, actor, weights, latent):
    targets = data['action'][:, actor]
    if latent:
        targets = np.arctanh(np.clip(targets, np.float32(-1 + 1e-4), np.float32(1 - 1e-4)))
    targets = targets.astype(np.float64)
    numerator = 0.
    for ids in groups:
        w = weights[ids]
        mean = np.average(targets[ids], axis=0, weights=w)
        numerator += float(np.sum(w[:, None] * (targets[ids] - mean)**2))
    denominator = 2 * np.sum(weights * data['active'][:, actor, 0])
    return numerator / denominator


def summarize(data, metadata, original_count):
    assert np.isfinite(data['obs']).all() and np.isfinite(data['action']).all()
    assert np.max(np.abs(data['action'])) <= 1
    np.testing.assert_array_equal(data['obs'][:, :, -1:], data['active'])
    assert np.max(np.abs(data['action'][data['active'][..., 0] == 0]), initial=0) == 0
    n = len(data['obs'])
    out = dict(rows=n, original_rows=original_count, correction_rows=n-original_count, actors=[])
    weights = np.ones(n) / n
    mixed = weights.copy()
    if 0 < original_count < n:
        mixed[:original_count] = .5 / original_count
        mixed[original_count:] = .5 / (n-original_count)
    for actor in range(2):
        groups = groups_for(data, actor)
        duplicates = [ids for ids in groups if len(ids) > 1]
        conflicts = []
        for ids in duplicates:
            actions = data['action'][ids, actor]
            if np.max(np.ptp(actions, axis=0)) <= 1e-5:
                continue
            meta = [dict(global_index=i, **metadata[i]) for i in ids]
            conflicts.append(dict(indices=ids, actions=actions.tolist(),
                max_component_span=float(np.max(np.ptp(actions, axis=0))),
                same_joint_actor_inputs=all(np.array_equal(data['obs'][i], data['obs'][ids[0]]) for i in ids),
                same_public_prefix=len({m['public_prefix_sha256'] for m in meta}) == 1,
                same_observation=len({m['observation_sha256'] for m in meta}) == 1,
                same_belief_signature=len({m['belief_signature'] for m in meta}) == 1,
                same_environment_seed=len({m['environment_seed'] for m in meta}) == 1,
                all_at_episode_start=all(m['trace_index'] == 0 for m in meta),
                uniform_action_mean=actions.mean(0).tolist(),
                mixed_latent_optimum=np.tanh(np.average(np.arctanh(np.clip(actions,
                    np.float32(-1+1e-4), np.float32(1-1e-4))), axis=0, weights=mixed[ids])).tolist(),
                provenance=meta))
        out['actors'].append(dict(actor=actor, active_rows=int(data['active'][:, actor].sum()),
            unique_inputs=len(groups), duplicate_groups=len(duplicates), conflict_groups=len(conflicts),
            conflict_rows=sum(len(c['indices']) for c in conflicts), conflicts=conflicts,
            uniform_action_loss_floor=lower_bound(data, groups, actor, weights, False),
            uniform_latent_loss_floor=lower_bound(data, groups, actor, weights, True),
            mixed_50_50_action_loss_floor=lower_bound(data, groups, actor, mixed, False),
            mixed_50_50_latent_loss_floor=lower_bound(data, groups, actor, mixed, True)))
    return out


def fixed_search(audit, output):
    from solvers.e45f_baselines import GeometryController
    from learning.e45j_repaired_mappo import inputs
    from e45g_training_acceleration import public_geometry_cache
    results = []
    for (group, world), entry in audit.roots.items():
        case, seed, obs = entry['case'], entry['seed'], entry['observation']
        reference = load_labels(O / 'demonstrations' / f'{group}_{world}.npz')['obs'][0]
        def construct():
            controller = GeometryController(case['map'], case['speed'], seed,
                use_policy=False, use_value=False, budget_ms=200.)
            for old in case['history'][:-1]:
                controller.observe(old)
            return controller
        sequences = []
        started = time.perf_counter()
        with public_geometry_cache():
            for repeat in range(2):
                controller = construct()
                controller.act(obs, fixed_simulations=1)
                public, _ = inputs(controller.features(controller.root), controller.public, obs)
                public = np.c_[public, np.asarray(obs['active_p'], np.float32)[:, None]].astype(np.float32)
                np.testing.assert_array_equal(public, reference)
                sequence = []
                for count in range(1, 129):
                    if count > 1:
                        controller.trajectory()
                    best = int(np.argmax(controller.root.visits))
                    sequence.append(dict(simulations=count,
                        action=(controller.root.actions[best] / case['speed']).tolist(),
                        root_actions=(np.asarray(controller.root.actions) / case['speed']).tolist(),
                        visits=list(controller.root.visits), totals=list(controller.root.totals),
                        belief_signature=controller.belief.signature()))
                sequences.append(sequence)
                audit.query_count += 1
                audit.simulated_transitions += sum(len(t) for t in controller.traces)
        assert sequences[0] == sequences[1], 'Fixed-budget fresh reconstructions disagree'
        historical = []
        for record in entry['records']:
            count = record['teacher_simulations']
            assert 1 <= count <= 128
            predicted = np.asarray(sequences[0][count-1]['action'])
            error = float(np.max(np.abs(predicted - record['normalized_action'])))
            assert record['belief_signature'] == sequences[0][count-1]['belief_signature']
            historical.append(dict(source=record['source'], train_seed=record['train_seed'],
                round=record['round'], simulations=count, maximum_action_error=error,
                reproduced_at_same_count=error <= 2e-6))
        rng_probes = []
        # Change only SEARCH randomness, keeping belief seed/history unchanged.
        if any(not np.allclose(entry['records'][0]['normalized_action'], r['normalized_action'], atol=1e-5)
               for r in entry['records'][1:]):
            with public_geometry_cache():
                for offset in (0, 1, 2):
                    controller = construct()
                    controller.rng = np.array([(seed+94522027+offset) % 4294967296], np.int64)
                    controller.export_rng = np.array([(seed+94522028+offset) % 4294967296], np.int64)
                    controller.proposal_rng = np.random.default_rng(seed+94522029+offset)
                    controller.world_rng = np.random.default_rng(seed+94522030+offset)
                    action = controller.act(obs, fixed_simulations=64)
                    public, _ = inputs(controller.features(controller.root), controller.public, obs)
                    public = np.c_[public, np.asarray(obs['active_p'], np.float32)[:, None]].astype(np.float32)
                    np.testing.assert_array_equal(public, reference)
                    rng_probes.append(dict(search_seed_offset=offset, simulations=64,
                        normalized_action=(action/case['speed']).tolist(),
                        belief_signature=controller.belief.signature()))
                    audit.query_count += 1
                    audit.simulated_transitions += sum(len(t) for t in controller.traces)
        item = dict(group=group, world=world, seed=seed, repeats=2, fixed_simulations=128,
            independent_repeats_exact=True, actor_input_exact=True,
            historical=historical, search_rng_only_probes=rng_probes,
            wall_seconds=time.perf_counter()-started, sequence=sequences[0])
        results.append(item)
        save(output/'fixed_search.json', results)
        print(json.dumps(dict(stage='fixed_search', group=group, world=world,
            historical_reproduced=sum(r['reproduced_at_same_count'] for r in historical),
            historical_total=len(historical), seconds=item['wall_seconds'])), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
        default=ROOT/'current/e45q_teacher_label_audit_20261102')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    audit = Audit()
    for path in [Path(__file__), ROOT/'code/learning/e45q_corrective_imitation.py',
                 ROOT/'code/learning/e45p_actor_supervision.py', ROOT/'code/solvers/e45b_joint_nte.py',
                 ROOT/'code/solvers/e45f_baselines.py', ROOT/'code/learning/e45m_task_adapter.py',
                 ROOT/'code/learning/e45j_repaired_mappo.py', ROOT/'code/learning/e45b_joint_network.py']:
        audit.protect(path)
    original, original_meta = audit.original()
    result = dict(scope='Existing training labels plus bounded public-state search probes only',
        optimizer_updates=0, environment_rollouts=0, existing_labels_rewritten=False,
        original=summarize(original, original_meta, len(original['obs'])), seeds={})
    for seed in (17, 29):
        corrections, metadata, counts = audit.corrections(seed)
        combined = concatenate([original, corrections])
        for route, expected in [('stable_bc', original), ('dagger', combined)]:
            frozen = audit.label_packet(Q/'imitation'/f'seed_{seed}'/route/'supervision.npz')
            for key in LABEL_KEYS:
                np.testing.assert_array_equal(frozen[key], expected[key])
        result['seeds'][str(seed)] = dict(counts=counts,
            corrections=summarize(corrections, metadata, 0),
            combined=summarize(combined, original_meta+metadata, len(original['obs'])),
            final_fit=audit.packet(Q/'imitation'/f'seed_{seed}'/'dagger/evaluation.json')[-1])
    save(args.output/'static_audit.json', result)
    print(json.dumps(dict(stage='static_complete', seeds={s:v['counts'] for s,v in result['seeds'].items()})), flush=True)
    fixed = fixed_search(audit, args.output)
    unchanged = all(sha(path) == expected for path, expected in audit.inputs.items())
    assert unchanged, 'An audit input changed during diagnosis'
    result.update(inputs_unchanged=unchanged, protected_files=len(audit.inputs),
        fixed_search_independent_repeats_exact=all(x['independent_repeats_exact'] for x in fixed),
        historical_root_labels_reproduced=sum(r['reproduced_at_same_count'] for x in fixed for r in x['historical']),
        historical_root_labels_total=sum(len(x['historical']) for x in fixed),
        new_teacher_queries=audit.query_count, new_simulated_transitions=audit.simulated_transitions,
        wall_seconds=time.perf_counter()-started)
    save(args.output/'input_hashes.json', audit.inputs)
    save(args.output/'summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('original','seeds')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
