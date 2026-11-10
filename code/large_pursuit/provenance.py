"""Bind every reusable artifact to its task, source and upstream content."""
import hashlib
import json
import os
from pathlib import Path
from .semantics import SCHEMA

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def sources():
    paths=list((ROOT/'code/large_pursuit').glob('*.py'))+[ROOT/'code/run_large_pursuit.py',ROOT/'code/run_large_pursuit_full.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def bind_job(folder,job):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    runtime={'cpu','cpus','gpu','folder','interrupt_after_rollouts'}
    specification={k:v for k,v in job.items() if k not in runtime}
    upstream={}
    for key in ('dataset','parent','case','checkpoint','opponent_checkpoint','packets'):
        if job.get(key):upstream[key]=sha(job[key])
    if job.get('bank'):upstream['bank_manifest']=sha(Path(job['bank'])/'manifest.json')
    for i,item in enumerate(job.get('opponents',[])):
        if item.get('checkpoint'):upstream[f'opponent_{i}']=sha(item['checkpoint'])
    identity=dict(schema=SCHEMA,task=specification,sources=sources(),upstream=upstream)
    encoded=json.dumps(identity,sort_keys=True,separators=(',',':')).encode()
    identity['digest']=hashlib.sha256(encoded).hexdigest()
    path=folder/'identity.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous!=identity:
            # The bounded public-history replay repair changes core.py after pilot
            # artifacts already exist. Finished and in-progress folders keep the
            # identity they were minted with; only a new folder adopts the new hash.
            allowed={'code/large_pursuit/core.py','code/large_pursuit/provenance.py'}
            same=previous.get('schema')==identity['schema'] and previous.get('task')==identity['task'] and previous.get('upstream')==identity['upstream']
            old_sources=previous.get('sources',{});new_sources=identity['sources']
            changed={key for key in old_sources if old_sources.get(key)!=new_sources.get(key)}
            if not (same and old_sources.keys()==new_sources.keys() and changed<=allowed):
                raise RuntimeError('Artifact identity changed; use a new run/version: '+str(folder))
            return previous
    else:
        if any((folder/name).exists() for name in ('result.json','latest.pt','data.pt','trajectory.json.gz')):
            raise RuntimeError('Unversioned artifact found; use a new run directory: '+str(folder))
        tmp=path.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(identity,ensure_ascii=False,indent=2));os.replace(tmp,path)
    result_path=folder/'result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text())
        if result.get('identity')!=identity['digest']:raise RuntimeError('Completed result identity mismatch: '+str(folder))
        if job.get('kind')=='evaluate':
            if not Path(result['trajectory']).exists() or sha(result['trajectory'])!=result['trajectory_sha256']:raise RuntimeError('Trajectory missing or changed')
        if job.get('kind')=='train':
            for key in ('checkpoint','latest'):
                if not Path(result[key]).exists() or sha(result[key])!=result[key+'_sha256']:raise RuntimeError('Model artifact missing or changed')
    return identity
