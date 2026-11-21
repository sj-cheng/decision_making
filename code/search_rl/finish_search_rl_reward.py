"""Run integrity checks and merged reports after both detached studies finish."""
import sys
from pathlib import Path
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
import repo_paths
repo_paths.install()
import json
from pathlib import Path
import subprocess
import sys
import time
from watch_search_rl_reward import read


def main():
    code=Path(__file__).resolve().parent
    root=code.parents[1]/'current/search_rl_reward_20261022'
    def status(phase,**extra):
        path=root/'postprocess_status.json';temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(phase=phase,updated_unix=time.time(),**extra),indent=2));temp.replace(path)
    status('waiting_for_training_and_search')
    while not all(read(root/p,{}).get('phase')=='complete' for p in ('status.json','completion_extension/status.json')):
        time.sleep(15)
    with (root/'postprocess.log').open('a',buffering=1) as log:
        for script in ('analyze_search_rl_reward.py','summarize_search_rl_reward.py'):
            status('running',script=script)
            result=subprocess.run([sys.executable,str(code/script)],cwd=code.parent,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                status('failed',script=script,exit_code=result.returncode)
                return
    status('complete',report=str(root/'FINAL_REPORT.md'))


if __name__=='__main__':main()
