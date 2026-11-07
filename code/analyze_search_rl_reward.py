"""Read-only model integrity and fixed-scene learning curves for reward study."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_k]='1'
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
import run_search_rl_reward_experiment as study


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=study.trainer.ROOT/'current/search_rl_reward_20261022')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--curve-per-layout',type=int,default=20)
    args=p.parse_args();out=args.output.resolve()
    manifest=json.loads((out/'experiment.json').read_text());config=manifest['config']
    selection=json.loads((out/'selection.json').read_text());chosen=selection['selected']
    initial=study.PolicyBundle.load(out/'initial.pt')
    metrics=[];episodes={};frozen=[];owners={}
    for seed in config['seeds']:
        for arm in study.ARMS:
            owner=out/'completion_extension' if arm=='completion_bonus_v1' and (out/'completion_extension/extension.json').exists() else out
            owners[arm]=owner
            folder=owner/f'seed{seed}'/arm
            rows=[json.loads(l) for l in (folder/'metrics.jsonl').read_text().splitlines()]
            expected=config['long_episodes'] if arm in ('legacy',chosen) or owner!=out else config['short_episodes']
            assert rows[-1]['episodes']==expected
            assert len({r['episodes'] for r in rows})==len(rows)
            bundle=study.PolicyBundle.load(folder/'latest.pt')
            assert study.state_hash(bundle,0)==study.state_hash(initial,0)
            frozen.append(dict(seed=seed,arm=arm,evader_unchanged=True))
            metrics.extend(rows)
            for l in (folder/'episodes.jsonl').read_text().splitlines():
                r=json.loads(l);key=(seed,r['training_index'])
                signature=(hashlib.sha256(json.dumps(r['spec'],sort_keys=True).encode()).hexdigest(),r['opponent'])
                if key in episodes:assert signature==episodes[key]
                else:episodes[key]=signature
    source=Path(config['source'])
    assert all(study.digest(source/name)==sha for name,sha in manifest['source'].items())
    assert study.digest(Path(study.__file__))==manifest['script_sha256']
    assert study.trainer.implementation_hash()==manifest['base_implementation']
    assert max(r['old_logp_error'] for r in metrics)<2e-3
    observed={}
    for r in metrics:
        key=(r['seed'],r['start_index'],r['own_state_before']) if r['shared_rollouts'] else (r['seed'],r['start_index'],r['arm'])
        observed[key]=True
    integrity=dict(passed=True,original_models_unchanged=True,implementation_unchanged=True,
                   frozen_evaders=frozen,updates=len(metrics),logical_episodes=len(metrics)*config['batch'],
                   physical_sampled_episodes=len(observed)*config['batch'],paired_map_opponents=True,
                   old_logp_max_error=max(r['old_logp_error'] for r in metrics))
    study.trainer.write_json(out/'integrity.json',integrity)
    curve_arms=[arm for arm in study.ARMS if arm in ('legacy',chosen) or owners[arm]!=out]
    points=sorted({config['short_episodes'],256,512,768,config['long_episodes']})
    points=[i for i in points if config['short_episodes']<=i<=config['long_episodes'] and i%config['batch']==0]
    checkpoints=[('initial',None,out/'initial.pt')]
    for n in points:
        for seed in config['seeds']:
            for arm in curve_arms:
                path=owners[arm]/f'seed{seed}'/arm/f'policy_ep{n}.pt'
                assert path.exists()
                checkpoints.append((f'{arm}_ep{n}',seed,path))
    torch.set_num_threads(1)
    options=argparse.Namespace(search_budget=1000,search_depth=8)
    with mp.get_context('spawn').Pool(args.workers,initializer=study.trainer.worker_init) as pool:
        rows=study.evaluate(options,out,pool,checkpoints,'learning_curve_confirmation',630,args.curve_per_layout)
    result=study.summarize(rows)
    study.trainer.write_json(out/'learning_curve.json',dict(points=points,rows=result,
        note='Same new phase-630 scenes at every checkpoint. Post-hoc diagnostic, not used to select reward or model.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    base=next(r for r in result if r['label']=='initial')
    specs=[('success','Legal double capture'),('one_capture_timeout','One capture then timeout'),('own_illegal','Own illegal ending')]
    for ax,(metric,title) in zip(axes,specs):
        for arm,color in zip(curve_arms,('tab:blue','tab:green','tab:orange')):
            values=[]
            for seed in config['seeds']:
                line=[next(r[metric]/r['n'] for r in result if r['label']==f'{arm}_ep{n}' and r['seed']==seed) for n in points]
                values.append(line);ax.plot(points,line,alpha=.35,color=color,marker='.')
            ax.plot(points,np.mean(values,axis=0),color=color,marker='o',label=arm)
        ax.axhline(base[metric]/base['n'],color='black',linestyle='--',label='Frozen ep512 baseline')
        ax.set_title(title);ax.set_xlabel('Additional pursuer PPO episodes');ax.set_ylim(0,1);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Fixed-scene post-hoc trajectory across training; two sampling seeds, one pretrained model')
    fig.tight_layout();fig.savefig(out/'learning_curve.png',dpi=150);plt.close(fig)
    print(json.dumps(integrity,indent=2),flush=True)


if __name__=='__main__':main()
