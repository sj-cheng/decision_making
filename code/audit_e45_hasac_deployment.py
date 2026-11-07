"""Exact deployment-vs-development behavior on recorded training histories.

No new test cases or controller performance selection. Source M/N artifacts
remain read-only. Also checks independence from unrelated Torch RNG draws.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import gzip
import hashlib
import json
from pathlib import Path
import torch
import numpy as np
import run_e45m_hasac_task as m
from learning.e45_hasac_deployment import load_actors,Controller


def main():
    torch.set_num_threads(1)
    out=m.ROOT/'current/e45_hasac_deployment_audit_20261101';out.mkdir(exist_ok=True)
    m.configure(out/'public_replay')
    checkpoint=m.OUT/'runs/HASAC_17/checkpoints/steps_0.pt'
    actors,metadata=load_actors(checkpoint,m.j.sha(checkpoint))
    checks=[]
    # Verify complete deterministic AND stochastic policy sequences against
    # actual original M step0 archives, with the original public packet/RNG.
    cache=m.OUT/'runs/HASAC_17/evaluation/cache'
    rows=[m.j.read(p) for p in cache.glob('*.json')]
    for mode in ('deterministic','stochastic'):
        row=sorted([r for r in rows if r['execution_mode']==mode and 'eval_steps0' in r['stage']],key=lambda r:r['key'])[0]
        assert m.j.sha(row['episode'])==row['episode_sha256']
        payload=json.loads(gzip.decompress(Path(row['episode']).read_bytes()))
        case=m.j.read(row['case'])
        own=Controller(case['map'],case['speed'],row['seed'],actors,
                       stochastic=mode=='stochastic',policy_seed=row['policy_seed'])
        for obs in case['history'][:-1]:own.observe(obs)
        maximum=0.
        for t in payload['trace']:
            # Unrelated framework users must not alter this policy's stream.
            torch.randn(13)
            action=own.act(t['observation'])
            maximum=max(maximum,float(np.max(np.abs(action-np.asarray(t['action'])))))
            assert own.belief.signature()==t['belief_signature']
        assert maximum==0,maximum
        checks.append(dict(mode=mode,source=row['episode'],steps=len(payload['trace']),max_action_error=maximum,
                           extra_torch_draws_do_not_change_policy=True))
    # Successful public trajectories supply actual first-capture/retired
    # observations absent from the random policy's initial archives. Compare
    # actor outputs on those packets; teacher actions are never training data.
    controls=m.l.OUT/'positive_control/evaluation/episodes'
    retired_frames=0
    for path in sorted(controls.glob('*.json'))[:2]:
        payload=m.j.read(path);row=payload['summary'];case=m.j.read(row['case'])
        for mode in ('deterministic','stochastic'):
            original=m.Controller(case['map'],case['speed'],row['seed'],actors,
                                  stochastic=mode=='stochastic',policy_seed=2791)
            for obs in case['history'][:-1]:original.observe(obs)
            expected=[original.act(t['observation']) for t in payload['trace']]
            deployed=Controller(case['map'],case['speed'],row['seed'],actors,
                                stochastic=mode=='stochastic',policy_seed=2791)
            for obs in case['history'][:-1]:deployed.observe(obs)
            for target,t in zip(expected,payload['trace']):
                actual=deployed.act(t['observation']);np.testing.assert_array_equal(actual,target)
                active=np.asarray(t['observation']['active_p'],bool)
                if not active.all():
                    assert np.max(np.abs(actual[~active]))==0;retired_frames+=1
    assert retired_frames>0
    # Check integrity guard rather than accidentally evaluating a changed model.
    try:load_actors(checkpoint,'0'*64)
    except ValueError:pass
    else:raise AssertionError('Changed checkpoint accepted')
    report=dict(passed=True,scope='actor-only deployment equivalence on reused training observations; not an independent benchmark',
                metadata=metadata,recorded_action_checks=checks,retired_frames_checked=retired_frames,
                source_sha256=m.j.sha(Path(__file__)),runtime_sha256=m.j.sha(m.ROOT/'code/learning/e45_hasac_deployment.py'),
                actor_only=True,critic_or_reward_called_online=False,integrity_guard_checked=True,
                no_teacher_training=True)
    m.j.save(out/'equivalence.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
