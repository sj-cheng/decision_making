"""Share matching strong-evader baseline outcomes between the budget-64 queues."""
import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

from run_1000_cruise import H80

SCALE = H80/'network_scale_trials'
OUT = SCALE/'advanced'
FIELDS = ('captures','first','double','cycles','collisions','search_ms')


def read(path):
    return json.loads(path.read_text())


def live(folder):
    for file in folder.glob('*/trace.progress.json'):
        try:
            progress = read(file)
            if progress['state']=='running' and Path(f"/proc/{progress['pid']}").exists():
                return True
        except FileNotFoundError:
            pass
    return False


def publish(path, row):
    # The legacy workers already in flight may finish while this helper runs.
    # Never replace a completed record or their staging filename.
    path.parent.mkdir(parents=True,exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, mode='w', delete=False) as tmp:
        json.dump(row,tmp)
    try:
        os.link(tmp.name,path)
    except FileExistsError:
        pass
    finally:
        Path(tmp.name).unlink()


def timing_cases(cases):
    layouts = {}
    for case in cases:
        name=case['case'];layouts.setdefault(name.rsplit('_',1)[0],[]).append(name)
    return {case for names in layouts.values() for case in sorted(names)[:2]}


def share(case,timed):
    name=case['case']
    scale=SCALE/'episodes/expanded_large_52/b64'/name/'result.json'
    teacher=OUT/'eval/evader_teacher'/name/'result.json'
    if scale.exists() and teacher.exists():
        a,b=read(scale),read(teacher)
        if any(a[k]!=b[k] for k in FIELDS if k!='search_ms'):
            raise ValueError(f'overlapping teacher outcomes disagree: {name}')
        return
    if scale.exists() and name not in timed and not live(teacher.parent):
        row=read(scale)
        if (row['case'],row['method'],row['budget'])!=(name,'expanded_large_52',64):
            raise ValueError(f'incompatible cached budget row: {scale}')
        publish(teacher,{**{k:row[k] for k in FIELDS},'elapsed':None,'evader_ms':None,
                         'simulations':None,'reused_from':str(scale),'timing_sample':False})
    elif teacher.exists() and not live(scale.parent):
        row=read(teacher)
        publish(scale,{**{k:row[k] for k in FIELDS},'case':name,'method':'expanded_large_52',
                       'budget':64,'reused_from':str(teacher)})


def main(watch):
    cases=read(SCALE/'manifest.json')['dev'];timed=timing_cases(cases)
    publish(OUT/'teacher_timing_cases.json',sorted(timed))
    while True:
        for case in cases:share(case,timed)
        if not watch or all((OUT/'eval/evader_teacher'/c['case']/'result.json').exists()
                            and (SCALE/'episodes/expanded_large_52/b64'/c['case']/'result.json').exists()
                            for c in cases):break
        time.sleep(10)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true')
    main(parser.parse_args().watch)
