"""Durable launch: require passed smoke, run the registered pilot, then report."""
import os
from pathlib import Path
import subprocess
import sys
import run_e45o_hasac_demonstration as o


if __name__=='__main__':
    root=o.OUT;root.mkdir(parents=True,exist_ok=True)
    smoke=o.ROOT/'current/e45o_hasac_demonstration_smoke_20261101'
    o.verify(smoke)
    audit=o.j.read(smoke/'smoke_audit.json')
    assert audit['passed'] and len(audit['runs'])==4
    o.verify(root)
    o.j.save(root/'launch.json',dict(pid=os.getpid(),smoke_audit=str(smoke/'smoke_audit.json'),
        smoke_audit_sha256=o.j.sha(smoke/'smoke_audit.json'),
        launcher_sha256=o.j.sha(Path(__file__).resolve()),
        analysis_source_sha256=o.j.sha(o.ROOT/'code/analyze_e45o_demonstration.py')))
    with (root/'pipeline.log').open('a') as log:
        result=subprocess.run([sys.executable,'-u',str(o.ROOT/'code/run_e45o_hasac_demonstration.py'),
            'pipeline','--output',str(root),'--workers','4','--device','cuda:1'],
            cwd=o.ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        sys.exit(result.returncode)
    subprocess.run([sys.executable,str(o.ROOT/'code/analyze_e45o_demonstration.py'),
        '--output',str(root)],cwd=o.ROOT,check=True)
