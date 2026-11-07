"""Synthetic GPU construction/update check, not task training or performance."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from learning.e45m_author_hasac import AuthorHASAC,UPSTREAM


def main():
    torch.set_num_threads(1);rng=np.random.default_rng(43017)
    model=AuthorHASAC(device='cuda:1');n=64
    obs=rng.normal(0,.1,(2,n,844)).astype(np.float32)
    central=rng.normal(0,.1,(n,858)).astype(np.float32)
    actions=rng.uniform(-1,1,(2,n,2)).astype(np.float32)
    reward=rng.normal(0,.1,(n,1)).astype(np.float32)
    done=np.zeros((n,1),np.float32);done[-8:]=1
    valid=np.ones((2,n,1),np.float32);valid[0,n//2:]=0
    batch=(central,obs,actions,None,reward,done,valid,done.copy(),central*.9,obs*.9,None,np.ones((n,1),np.float32))
    before=[{k:v.clone() for k,v in a.actor.state_dict().items()} for a in model.actor]
    cbefore={k:v.clone() for k,v in model.critic.critic.state_dict().items()}
    for _ in range(3):model.update(batch)
    assert all(any(not torch.equal(v,model.actor[i].actor.state_dict()[k]) for k,v in state.items()) for i,state in enumerate(before))
    assert any(not torch.equal(v,model.critic.critic.state_dict()[k]) for k,v in cbefore.items())
    with torch.no_grad():
        for i,a in enumerate(model.actor):
            act,logp=a.get_actions_with_logprobs(obs[i]);assert act.shape==(n,2) and logp.shape==(n,1)
            assert (act.abs()<=1).all() and torch.isfinite(logp).all()
    bad=list(batch);bad[6]=np.zeros_like(valid)
    try:model.update(tuple(bad))
    except ValueError:pass
    else:raise AssertionError('All-retired minibatch was accepted')
    files={}
    for p in sorted(UPSTREAM.rglob('*.py')):
        if '.git' in p.parts:continue
        data=p.read_bytes();rel=str(p.relative_to(UPSTREAM))
        blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        expected=subprocess.check_output(['git','rev-parse','HEAD:'+rel],cwd=UPSTREAM,text=True).strip()
        assert blob==expected
        files[str(p)]=hashlib.sha256(data).hexdigest()
    root=Path(__file__).resolve().parents[1]/'current/e45_marl_macro_audit_20261101';root.mkdir(exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip()
    provenance=dict(url='https://github.com/PKU-MARL/HARL',commit=commit,source_hashes=files,
                    python_files=len(files),modified_author_files=False,simulator_assets_excluded=True)
    (root/'harl_upstream.json').write_text(json.dumps(provenance,indent=2)+'\n')
    result=dict(passed=True,scope='synthetic GPU smoke only; not hidden-task training',device='cuda:1',updates=3,
                both_actors_updated=True,twin_critic_updated=True,invalid_retirement_batch_rejected=True,
                author_commit=commit,author_python_files=len(files),config=model.config,
                missing_before_task_training=['retirement-safe replay and next public observation adapter',
                    'whole-episode environment rollout and reward/terminal checks','registered task training protocol'])
    (root/'hasac_synthetic_smoke.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
