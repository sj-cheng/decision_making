"""Generate fixed, held-out starts for paired E52 diagnostics."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from e49_rules import _legal
from run_e52_train_nar_nae import initial_id


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRODUCTION_DIR = ROOT / 'current/E系列实验输出/e52_combined_20261118/production_parallel_v3'
SCENES = ('wall', 'corner', 'sparse')
CASES_PER_SCENE = 3


def _case_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _pose_key(case):
    # Disallow an identical geometric start even if non-pose state metadata differs.
    payload = [case['map'], np.round(case['worlds'][0]['initial'][:8], 8).tolist()]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _perturb(source, rng):
    boxes = np.asarray(source['map'], float).reshape(-1, 2, 2)
    clearance = boxes + np.array([-.08, .08])
    formations = [np.asarray(world['initial'][:8], float).reshape(4, 2)
                  for world in source['worlds']]
    for _ in range(300):
        angle = rng.uniform(-np.pi, np.pi)
        translation = rng.uniform(.4, 1.1) * np.array([np.cos(angle), np.sin(angle)])
        shifted = [formation + translation for formation in formations]
        if not all(
            np.max(np.abs(poses[:2])) <= 8.0 and np.max(np.abs(poses[2:])) <= 8.8
            and all(_legal(point, clearance) for point in poses)
            and np.min(np.linalg.norm(poses[:2, None] - poses[None, 2:], axis=2)) >= 1.1
            for poses in shifted
        ):
            continue
        case = deepcopy(source)
        case['horizon'] = 160
        case['history'] = []
        case['history_clock'] = 'fresh start without scripted observation history'
        for world, poses in zip(case['worlds'], shifted):
            initial = poses.ravel().tolist() + world['initial'][8:]
            world['initial'] = initial
            world['scripted_states'] = [initial]
            world['history'] = []
        return case, translation.tolist()
    return None


def generate_diagnostic_cases(output_dir, *, production_dir=DEFAULT_PRODUCTION_DIR, seed=521119):
    """Write three cases per scene and return their paths and provenance manifest.

    Each case shifts all four agents together, preserving their relative roles
    and spacing. Selection depends on fixed source paths and legality only.
    """
    output_dir = Path(output_dir).resolve()
    production_dir = Path(production_dir).resolve()
    formal_manifest = production_dir / 'formal/summary.json'
    frozen_manifest = production_dir / 'frozen_checkpoint.json'
    paths = sorted(_case_path(path) for path in _read(formal_manifest)['manifest']['cases'])
    formal = {path: _read(path) for path in paths}
    formal_ids = {initial_id(case) for case in formal.values()}
    if (len(paths) != 30 or len(formal_ids) != 30
            or any(sum(path.parent.name == scene for path in paths) != 10 for scene in SCENES)):
        raise ValueError('expected 30 distinct v3 formal starts spanning all three map groups')

    frozen = _read(frozen_manifest)
    metadata = frozen['metadata']
    train_ids = set(metadata['train_initial_ids'])
    validation_ids = set(metadata['validation_initial_ids'])
    teacher_paths = [_case_path(_read(_case_path(log))['case']) for log in metadata['logs']]
    teacher = [_read(path) for path in teacher_paths]
    teacher_ids = {initial_id(case) for case in teacher}
    if train_ids & validation_ids or teacher_ids != train_ids | validation_ids or len(teacher_ids) != 30:
        raise ValueError('teacher logs must match the frozen training and validation starts')
    if formal_ids & teacher_ids:
        raise ValueError('formal and frozen teacher starts overlap')

    seen_ids = formal_ids | teacher_ids
    seen_poses = {_pose_key(case) for case in [*formal.values(), *teacher]}
    rng = np.random.default_rng(seed)
    generated = []
    for scene in SCENES:
        sources = [path for path in paths if path.parent.name == scene]
        selected = 0
        for index in rng.permutation(len(sources)):
            source_path = sources[index]
            source = formal[source_path]
            proposal = _perturb(source, rng)
            if proposal is None:
                continue
            case, translation = proposal
            case_id = initial_id(case)
            pose_key = _pose_key(case)
            if case_id in seen_ids or pose_key in seen_poses:
                continue
            case['id'] = f'e52_nae_diagnostic_{scene}_{selected:02d}'
            target = output_dir / scene / f"{case['id']}.json"
            row = dict(path=str(target), scene=scene, initial_id=case_id,
                       spatial_id=pose_key, source_case=str(source_path),
                       source_initial_id=initial_id(source), translation=translation)
            generated.append((target, case, row))
            seen_ids.add(case_id)
            seen_poses.add(pose_key)
            selected += 1
            if selected == CASES_PER_SCENE:
                break
        if selected != CASES_PER_SCENE:
            raise RuntimeError(f'could only generate {selected} legal distinct starts for {scene}')

    manifest = dict(seed=seed, horizon=160, cases_per_scene=CASES_PER_SCENE,
                    source_formal_summary=str(formal_manifest), source_checkpoint_manifest=str(frozen_manifest),
                    source_checkpoint_sha256=frozen['sha256'], formal_source_ids=sorted(formal_ids),
                    teacher_train_source_ids=sorted(train_ids),
                    teacher_validation_source_ids=sorted(validation_ids),
                    teacher_source_cases=[str(path) for path in teacher_paths],
                    cases=[row for _, _, row in generated])
    for target, case, _ in generated:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(case, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return [path for path, _, _ in generated], manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--production-dir', type=Path, default=DEFAULT_PRODUCTION_DIR)
    parser.add_argument('--seed', type=int, default=521119)
    args = parser.parse_args()
    paths, _ = generate_diagnostic_cases(args.output, production_dir=args.production_dir, seed=args.seed)
    print(json.dumps([str(path) for path in paths], ensure_ascii=False))


if __name__ == '__main__':
    main()
