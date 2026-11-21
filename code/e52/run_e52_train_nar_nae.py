"""Offline teacher collection and grouped NAR/NAE training from simulation logs.

Eight-cycle NAE targets are on-policy returns for the executed root action;
unexecuted candidates receive no fabricated value label. The NAR policy uses
all teacher root visits. Episodes shorter than eight future steps are excluded.
"""
import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
import repo_paths
repo_paths.install()
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from solvers.e52_nar_nae import NARNAE


def initial_id(case):
    return hashlib.sha256(json.dumps([case['map'],case['worlds'][0]['initial']],sort_keys=True).encode()).hexdigest()


def records_from_run(data):
    logs = data['logs']; records = []
    terminal = bool(logs and len({x["target"] for x in logs[-1]["captures"]}) >= 2)
    for t in range(len(logs)):
        if t+8 > len(logs) and not terminal: break
        for aid in ('0','1'):
            root = logs[t]['search'][aid]['root']
            counts = np.array([c['visits'] for c in root], float)
            if not len(root) or counts.sum() == 0: continue
            selected = int(logs[t]['plans'][aid]['candidate_id'])
            if selected < 0: continue
            value = 0.
            for k in range(min(8, len(logs)-t)):
                row = logs[t+k]; prev = logs[t+k-1] if t+k else None
                new_captures = sum(x['cycle']==row['cycle'] for x in row['captures'])
                seen = row['sensor_stats'][aid]['visible_targets']
                previous_seen = prev['sensor_stats'][aid]['visible_targets'] if prev else 0
                tracks = row.get('beliefs',{}).get(aid, [])
                pose = np.array(row['pursuers'][int(aid)])
                old_pose = np.array(prev['pursuers'][int(aid)]) if prev else pose
                progress = 0.
                if tracks:
                    positions = np.array([x['position'] for x in tracks])
                    progress = float(np.min(np.linalg.norm(positions-old_pose,axis=1)) - np.min(np.linalg.norm(positions-pose,axis=1)))
                interception = 0.
                if tracks:
                    intercepts = np.array([np.array(x["position"])+np.array(x["velocity"]) for x in tracks])
                    interception = float(np.min(np.linalg.norm(intercepts-old_pose,axis=1))-np.min(np.linalg.norm(intercepts-pose,axis=1)))
                # Blocking proxy: progress toward a one-cycle predicted intercept.
                # Shared capture event, own camera reacquisition and local progress.
                reward = 4.*new_captures + .5*float(seen>0 and previous_seen==0) + progress + .2*interception - .05
                if np.linalg.norm(pose-old_pose)<.03: reward -= .1
                value += .98**k * reward
                if len({x["target"] for x in row["captures"]}) >= 2: break
            records.append(dict(features=[c['features'] for c in root], policy=(counts/counts.sum()).tolist(),
                                returns=[value]*len(root), value_index=selected, cycle=t, agent_id=aid))
    return records


def main():
    p=argparse.ArgumentParser(); p.add_argument('--logs',nargs='+',required=True)
    p.add_argument('--output',required=True); p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--seed',type=int,default=52); a=p.parse_args()
    groups={}
    for path in a.logs:
        data=json.loads(Path(path).read_text()); case=json.loads(Path(data['case']).read_text())
        groups.setdefault(initial_id(case),[]).extend(records_from_run(data))
    keys=sorted(groups)
    if len(keys)<2: raise ValueError('need at least two distinct initial states for train/validation split')
    rng=np.random.default_rng(a.seed); rng.shuffle(keys)
    n=max(1,len(keys)//5); validation_keys=keys[:n]; train_keys=keys[n:]
    train=[r for k in train_keys for r in groups[k]]; validation=[r for k in validation_keys for r in groups[k]]
    if not train or not validation: raise ValueError('no full eight-cycle returns in a split')
    import torch
    torch.set_num_threads(1); torch.manual_seed(a.seed)
    model=NARNAE(); loss=model.fit_records(train,epochs=a.epochs,seed=a.seed,progress=lambda row: print(json.dumps(row),flush=True))
    def metrics(records):
        mse=[]; ce=[]
        with torch.no_grad():
            for r in records:
                x=torch.tensor(r['features'],dtype=torch.float32); i=r['value_index']
                mse.append(float((model.nae(x)[i,0]-r['returns'][i])**2))
                ce.append(float(-(torch.tensor(r['policy'])*torch.log_softmax(model.nar(x).flatten(),0)).sum()))
        return dict(value_mse=float(np.mean(mse)),policy_cross_entropy=float(np.mean(ce)),records=len(records))
    metadata=dict(schema='e52_teacher_dataset_v1',seed=a.seed,epochs=a.epochs,horizon=8,
        labels='teacher visits; executed-action eight-cycle discounted returns',
        reward=dict(capture=4.,reacquisition=.5,progress=1.,predicted_interception=.2,step=-.05,stall=-.1),
        architecture=dict(hidden=[128,128],activation='ReLU',input_dim=14,optimizer='Adam',learning_rate=.001),
        teachers=[dict(path=path,depth=json.loads(Path(path).read_text()).get('e52_depth'),budget=json.loads(Path(path).read_text()).get('e52_budget')) for path in a.logs],
        train_initial_ids=train_keys,validation_initial_ids=validation_keys,
        logs=a.logs,train=metrics(train),validation=metrics(validation),loss=loss)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    model.save(out,metadata=metadata); out.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata))


if __name__=='__main__': main()
