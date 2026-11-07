import json
from pathlib import Path
P=Path('/data2/sjc/decision_making/current/e45w_joint_ppo_and_planning_20261103')


def main():
    f=P/'repair_coordinator/progress.json'
    if not f.exists():
        print('W repair coordinator initializing');return
    d=json.loads(f.read_text());print('E45-W | public-history support repair r1 | tmux:train')
    print(d['timestamp'],d['state'])
    for name,row in d['registry'].items():
        path=Path(row['directory'])/'status.json'
        s=json.loads(path.read_text()) if path.exists() else {}
        c=s.get('counts',{});b=s.get('repair_budget',{})
        target=b.get('target_logical_steps',1998848);step=c.get('team_transitions',0)
        print(f"{name}: {s.get('status','initializing')} | {step:,}/{target:,} | Adam {c.get('optimizer_steps',0):,} | repair attempt {row['attempt']}")
        v=s.get('latest_validation')
        if v:
            q=v['stochastic'];print(f"  public validation @{v['team_transitions']:,}: double {q['double_captures']}/{q['episodes']}")
        if b:print(f"  charged lost/crash sampling: {b['charged_lost_steps']:,}; total cap {b['cap']:,}; fresh environments")
        if s.get('error') or row.get('terminal_error'):print('  ERROR:',s.get('error') or row.get('terminal_error'))
    w=d.get('wall_status') or {}
    print('Wall/cost full new run:',w.get('completed',0),'/',w.get('planned',2520),'invalid',w.get('invalid',0),'exit',d['wall_exit_code'])
    print('Unchanged experiments 2/3 complete; experiment 5 passed; original failures preserved.')
    print('Unified repaired final test:',(d.get('final_status') or {}).get('status','waiting for all six frozen learners'))
    print('Test scores unlocked:',d['test_scores_unlocked'])


if __name__=='__main__':main()
