"""Full phase/replay audit and actual empty-boundary resume reproducibility."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
import run_e45n_hasac_curriculum as n


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


def model_difference(a,b):
    values=[]
    for aa,bb in zip(a['actors'],b['actors']):
        values.extend(float((v-bb[k]).abs().max()) for k,v in aa.items())
    for key in ('critic','critic2','target_critic','target_critic2'):
        values.extend(float((v-b[key][k]).abs().max()) for k,v in a[key].items())
    return max(values)


def main():
    torch.set_num_threads(1)
    root=n.ROOT/'current/e45n_hasac_curriculum_smoke_20261101'
    assert n.j.read(root/'decision.json')['status']=='complete'
    cfg=n.verify(root);folder=root/'runs/HASAC_Curriculum_17'
    d=n.j.read(folder/'decision.json');assert d['gradient_updates']==6 and len(d['gates'])==3
    assert all(g['smoke_bypass'] for g in d['gates'])
    rows=[n.j.read(p) for p in sorted((folder/'updates').glob('*.json'))]
    assert [r['phase_index'] for r in rows]==[0,1,2]
    starts=[];ends=[];details=[]
    for i,row in enumerate(rows):
        base=folder/'phase_snapshots'/f'phase_{i}'/'checkpoints'
        paths=sorted(base.glob('*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
        assert len(paths)==2
        start,end=map(load,paths);starts.append((paths[0],start));ends.append(end)
        assert start['gradient_updates']==2*i and end['gradient_updates']==2*(i+1)
        assert start['replay_path'] is None
        assert end['history'][-1]['phase_index']==i
        archives=[]
        for e in row['episodes']:
            assert e['opponent']==cfg['phases'][i]['opponent']
            assert e['case']==n.phase_entry(cfg['entries'][0],cfg['phases'][i])['path']
            assert n.j.sha(e['replay_episode'])==e['replay_sha256']
            with np.load(e['replay_episode']) as z:archives.append({k:z[k].copy() for k in z.files})
        assert len(archives)==2
        with np.load(end['replay_path']) as z:
            size=int(z['size'])
            assert size==sum(len(a['obs']) for a in archives)
            for key in archives[0]:np.testing.assert_array_equal(z[key],np.concatenate([a[key] for a in archives]))
        if i:
            assert model_difference(ends[i-1],start)==0
            assert start['steps']==ends[i-1]['steps']
        details.append(dict(phase=cfg['phases'][i]['name'],training_episodes=2,
                            phase_transitions=size,empty_replay_at_start=True,
                            old_phase_samples_absent=True,actor_and_critic_retained=True))
    # Resume from an EMPTY moving-phase replay checkpoint. It must reproduce
    # all later behavior/gradient updates, not only decode the saved tensors.
    resume=root/'resume_check';resume.mkdir(exist_ok=True)
    shutil.copy2(root/'experiment.json',resume/'experiment.json')
    rf=resume/'runs/HASAC_Curriculum_17';rf.mkdir(exist_ok=True,parents=True)
    mid_path,mid=starts[1]
    if not (rf/'decision.json').exists():
        n.j.save(rf/'latest.json',dict(path=str(mid_path),sha256=n.j.sha(mid_path),steps=mid['steps'],
                                     phase_index=1,phase_steps=0,replay_rng=mid['replay_rng']))
        n.j.save(rf/'gates.json',[d['gates'][0]])
        cores=n.j.choose_cpus(3)
        cmd=[sys.executable,'-u',str(n.ROOT/'code/run_e45n_hasac_curriculum.py'),'train','--output',str(resume),
             '--seed','17','--device','cuda:1','--cpus',','.join(map(str,cores)),'--parent',str(os.getpid())]
        print('Checking actual resume from empty moving-phase replay...',flush=True)
        with (resume/'resume.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    pointer=n.j.read(rf/'latest.json');resumed=load(pointer['path'])
    assert resumed['steps']==ends[-1]['steps'] and resumed['gradient_updates']==6
    difference=model_difference(ends[-1],resumed)
    assert difference<=1e-6,difference
    resumed_decision=n.j.read(rf/'decision.json');assert len(resumed_decision['gates'])==3
    assert [x['phase'] for x in resumed_decision['gates']]==[0,1,2]
    assert not list(root.glob('runs/*/failures/*.json'))
    result=dict(passed=True,scope='engineering smoke, gate bypassed, not learning evidence',
                phases=details,gradient_updates=6,real_resume_from_empty_phase=True,
                all_actor_and_critic_resume_max_difference=difference,
                earlier_gate_history_preserved=True,source_sha256=n.j.sha(Path(__file__)))
    n.j.save(root/'smoke_audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
