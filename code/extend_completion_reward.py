"""Continue the near-tied completion-bonus candidate without altering selection.

The preselected floor experiment keeps running independently. This extension
is declared after short validation but before either held-out confirmation.
"""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_k]='1'
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import torch
import run_search_rl_reward_experiment as study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=study.trainer.ROOT/'current/search_rl_reward_20261022')
    parser.add_argument('--workers',type=int,default=16)
    opts=parser.parse_args();parent=opts.parent.resolve();out=parent/'completion_extension';out.mkdir(exist_ok=True)
    original=json.loads((parent/'experiment.json').read_text());config=original['config']
    selected=json.loads((parent/'selection.json').read_text())['selected']
    if selected=='completion_bonus_v1':raise ValueError('Completion bonus already in the selected main continuation')
    filenames=['training_state.pt','metrics.jsonl','episodes.jsonl','latest.pt',
               f'policy_ep{config["short_episodes"]}.pt']
    sources={}
    for seed in config['seeds']:
        folder=parent/f'seed{seed}'/'completion_bonus_v1'
        for filename in filenames:sources[str(folder/filename)]=study.digest(folder/filename)
    declaration=dict(parent_experiment_sha256=study.digest(parent/'experiment.json'),
        reason='Reward candidates differ by one short-validation success; extend both before held-out confirmation.',
        chosen_without_using_confirmation=True,source_hashes=sources,script_sha256=study.digest(__file__),
        reward_runner_sha256=study.digest(study.__file__),workers=opts.workers)
    if (out/'extension.json').exists() and json.loads((out/'extension.json').read_text())!=declaration:
        raise ValueError('Different extension provenance')
    study.trainer.write_json(out/'extension.json',declaration)
    for name in ('initial.pt','distilled.pt'):
        if not(out/name).exists():shutil.copy2(parent/name,out/name)
    for seed in config['seeds']:
        source=parent/f'seed{seed}'/'completion_bonus_v1';dest=out/f'seed{seed}'/'completion_bonus_v1'
        if not(dest/'training_state.pt').exists():
            dest.mkdir(parents=True,exist_ok=True)
            for item in source.iterdir():
                if item.is_file():shutil.copy2(item,dest/item.name)
    args=argparse.Namespace(**config);args.workers=opts.workers;args.output=out
    initial=study.PolicyBundle.load(out/'initial.pt');anchor=study.PolicyBundle.load(out/'distilled.pt')
    torch.set_num_threads(1)
    def status(phase):
        study.trainer.write_json(out/'status.json',dict(phase=phase,updated_unix=time.time()));print('STATUS',phase,flush=True)
    with mp.get_context('spawn').Pool(args.workers,initializer=study.trainer.worker_init) as pool:
        status('longer_completion_bonus_training')
        study.train_to(args,out,pool,('completion_bonus_v1',),args.long_episodes,initial,anchor)
        checkpoints=[('completion_bonus_v1',seed,out/f'seed{seed}'/'completion_bonus_v1'/f'policy_ep{args.long_episodes}.pt') for seed in args.seeds]
        status('completion_bonus_policy_confirmation')
        policy=study.evaluate(args,out,pool,checkpoints,'long_confirmation',610,args.confirmation_per_layout,roles=(0,1))
        status('completion_bonus_full_search_confirmation')
        search=study.evaluate(args,out,pool,checkpoints,'search_confirmation',620,args.search_per_layout,methods=('search_ppo',),roles=(0,1))
    study.trainer.write_json(out/'results.json',dict(policy=study.summarize(policy),search=study.summarize(search),
                                                   continue_formal=False,preselected_main_reward=selected))
    assert all(study.digest(path)==value for path,value in sources.items())
    status('complete')


if __name__=='__main__':main()
