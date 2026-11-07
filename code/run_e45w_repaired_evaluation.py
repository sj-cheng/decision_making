"""Unified final evaluation under the explicitly amended W support filter."""
import argparse
from pathlib import Path
import torch
from learning.e45w_support_recovery import install, REVISION
install()
import run_e45w_frozen_evaluation as base
install(base.OUT/'final_evaluation_repair1/belief_failures')

ORIGINAL_FINGERPRINT = base.environment_fingerprint


def fingerprint():
    paths = [Path(__file__), Path(__file__).parent/'learning/e45w_support_recovery.py']
    return base.digest(dict(original=ORIGINAL_FINGERPRINT(), revision=REVISION,
                            repair_sources={str(p):base.sha(p) for p in paths}))


def completed_models(manifest):
    registry = base.read(base.OUT/'repair_coordinator/registry.json')
    result = []
    for arm in ('B','C'):
        for seed in manifest['seeds']:
            directory = Path(registry[f'{arm}_{seed}']['directory'])
            status_path = directory/'status.json'; status = base.read(status_path)
            if status['status'] != 'complete' or not status.get('training_budget_audit_passed'):
                raise RuntimeError('Repaired final test remains locked: '+str(directory))
            repair = status.get('repair_budget')
            target = repair['target_logical_steps'] if repair else 1998848
            assert status['counts']['team_transitions'] == target
            if repair: assert repair['maximum_lifetime_team_steps'] <= 1998848
            selection = base.read(directory/'selection.json')
            terminal = base.read(directory/'terminal_checkpoint.json')
            for stage, checkpoint in [('selected', Path(selection['checkpoint'])),
                                       ('terminal', Path(terminal['model']))]:
                path = directory/('selected_actor.pt' if stage=='selected' else 'terminal_actor.pt')
                model = base.descriptor(f'{arm}_{seed}_{stage}',arm,seed,stage,path)
                archived = torch.load(checkpoint.parent/'actor.pt',map_location='cpu',weights_only=False)
                assert base.tensor_hash(archived) == model['actor_tensor_sha256']
                model.update(training_status_sha256=base.sha(status_path),
                    selection_sha256=base.sha(directory/'selection.json'),
                    full_model_checkpoint=str(checkpoint),full_model_sha256=base.sha(checkpoint),
                    repair_budget=repair, evaluation_revision=REVISION)
                result.append(model)
    return result


base.environment_fingerprint = fingerprint
base.completed_learning_models = completed_models


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=40)
    parser.add_argument('--cpus',default=','.join(map(str,range(40))))
    args=parser.parse_args()
    args.mode='final'; args.smoke=False
    args.output=base.OUT/'final_evaluation_repair1'
    install(args.output/'belief_failures')
    base.run(args)
