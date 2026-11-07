"""Final full-artifact audit and descriptive summary for the frozen L run."""
from pathlib import Path
import gzip
import json
import numpy as np
import run_e45l_marl_diagnosis as l


def main():
    root=l.OUT;cfg=l.verify(root);assert l.j.read(root/'status.json')['stage']=='complete'
    records={};total_steps=total_train=total_eval=0
    for folder in sorted((root/'runs').iterdir()):
        done=l.j.read(folder/'decision.json');assert done['status']=='complete'
        rows=[l.j.read(p) for p in sorted((folder/'updates').glob('*.json'))]
        episodes=[e for r in rows for e in r['episodes']]
        steps=sum(e['decisions'] for e in episodes);assert steps==done['steps']
        assert len(episodes)==len(list((folder/'training/episodes').glob('*.gz')))
        visible=first=double=0;visible_frames=frames=0
        for e in episodes:
            assert l.j.sha(e['episode'])==e['episode_sha256']
            payload=json.loads(gzip.decompress(Path(e['episode']).read_bytes()))
            trace=payload['trace'];assert len(trace)==e['decisions'] and payload['summary']['public_replay_exact']
            assert sum(t['new_captures'] for t in trace)==e['captures']
            v=[any(q is not None for reports in t['observation']['reports'] for q in reports) for t in trace]
            visible+=any(v);visible_frames+=sum(v);frames+=len(v)
            first+=e['captures']>0;double+=e['double_capture']
        history=l.j.read(folder/'evaluation.json');assert len(history)==6
        caches=[l.j.read(p) for p in (folder/'evaluation/cache').glob('*.json')]
        assert len(caches)==108
        for h in history:
            assert l.j.sha(h['checkpoint'])==h['sha256']
            for mode in ('deterministic','stochastic'):
                selected=[r for r in caches if r['checkpoint_sha256']==h['sha256'] and r['execution_mode']==mode]
                assert len(selected)==h[mode]['episodes']
                assert sum(r['double_capture'] for r in selected)==h[mode]['double']
                assert sum(r['captures']>0 for r in selected)==h[mode]['first']
                assert abs(np.mean([r['utility'] for r in selected])-h[mode]['utility'])<1e-10
                for r in selected:
                    p=folder/'evaluation/episodes'/(r['key']+'.json');payload=l.j.read(p)
                    assert payload['summary']['public_replay_exact'] and not r['failure']
                    assert len(payload['trace'])==r['decisions']
        assert not list((folder/'failures').glob('*.json'))
        records[folder.name]=dict(steps=steps,updates=len(rows),training_episodes=len(episodes),
            evaluation_episodes=len(caches),ever_visible=visible,first_capture=first,double_capture=double,
            visible_frames=visible_frames,frames=frames,initial=history[0],final=history[-1])
        total_steps+=steps;total_train+=len(episodes);total_eval+=len(caches)
    result=dict(passed=True,scope='frozen L descriptive audit, no new evaluation or test',runs=records,
                total_steps=total_steps,training_episodes=total_train,evaluation_episodes=total_eval,
                experiment_sha256=l.j.sha(root/'experiment.json'),audit_source_sha256=l.j.sha(Path(__file__)))
    l.j.save(root/'final_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2),flush=True)
    for name,r in records.items():
        print(name,{k:v for k,v in r.items() if k not in ('initial','final')},flush=True)


if __name__=='__main__':main()
