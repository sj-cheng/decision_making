"""Freeze new estimator validation histories before reading outcomes.
Refuses to overwrite an existing frozen bank; geometric rejection only.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from estimator_validation_scenarios import construct
from verify_l45_occlusion_scenarios import verify_case
root=Path(__file__).resolve().parent.parent;out=root/'current/l45_multimodal_estimator_20261025';out.mkdir(exist_ok=True)
if (out/'bank_frozen.json').exists():raise RuntimeError('Frozen bank exists')
known=set()
for p in (root/'current').glob('**/controller_inputs/*.json'):
 if out in p.parents:continue
 try:d=json.loads(p.read_text());known.add(hashlib.sha256(json.dumps(dict(map=d['map'],history=d['history']),sort_keys=True).encode()).hexdigest())
 except (KeyError,ValueError):pass
entries=[];checks=[];artifacts={}
for f,family in enumerate(['long_wall','wall_corner','multi_route']):
 for h,H in enumerate([4,6]):
  for v,speed in enumerate([1.1,1.3]):
   for variant in range(2):
    seed=int(np.random.SeedSequence([71020261025,f,h,v,variant]).generate_state(1)[0]);rng=np.random.default_rng(seed)
    height=float(rng.uniform(6.15,6.65) if H==4 else rng.uniform(8.15,8.65))
    env,pub=construct(family,H,speed,seed,height,variant)
    check=verify_case(env,pub)
    if not check['passed']:raise RuntimeError((env['id'],check['reasons']))
    key=hashlib.sha256(json.dumps(dict(map=pub['map'],history=pub['history']),sort_keys=True).encode()).hexdigest()
    if key in known:raise RuntimeError('Historical public history overlap')
    known.add(key)
    for kind,d in [('controller_inputs',pub),('environment',env)]:
     p=out/'bank'/kind/(env['id']+'.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2));artifacts[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    entries.append(dict(id=env['id'],family=family,public=f'controller_inputs/{env["id"]}.json',environment=f'environment/{env["id"]}.json',geometry_history_sha256=key))
    checks.append(check)
(out/'bank/manifest.json').write_text(json.dumps(dict(entries=entries,split='unseen_confirmation',outcomes_evaluated=False),indent=2))
(out/'bank_acceptance.json').write_text(json.dumps(dict(passed=True,groups=len(checks),checks=checks),indent=2))
(out/'bank_frozen.json').write_text(json.dumps(dict(seed=71020261025,groups=24,trajectories=96,artifacts=artifacts,algorithm_results_read=False),indent=2))
print('Frozen 24 new public histories, 96 scripted trajectories; independent geometry accepted; no estimator outcomes evaluated.')
