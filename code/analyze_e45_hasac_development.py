"""Read-only combined HASAC analysis; never selects or changes a policy."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/e45_hasac_development_plot')
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'current/e45_hasac_development_review_20261101'
SOURCES=[('M_from_scratch',ROOT/'current/e45m_hasac_task_20261101'),
         ('N_curriculum',ROOT/'current/e45n_hasac_curriculum_20261101')]


def read(p):return json.loads(p.read_text())


def analyze():
    OUT.mkdir(exist_ok=True,parents=True)
    cache=read(OUT/'exposure_cache.json') if (OUT/'exposure_cache.json').exists() else {}
    records=[];statuses={}
    lines=['# 无分配强化学习基线：HASAC开发结果总览','',
           f'更新时间：{time.strftime("%FT%T%z")}。仅训练内开发结果，无独立测试。','',
           'M为从头隐藏训练；N为同作者HASAC的可见到隐藏课程。N额外课程成本单列，不称等总预算。', '',
           '|实验/种子|状态|已交互步|最新评估阶段|当前阶段确定性双捕|原隐藏Root确定性双捕|',
           '|---|---|---:|---|---:|---:|']
    for name,root in SOURCES:
        cfg=read(root/'experiment.json');stage=read(root/'status.json') if (root/'status.json').exists() else {}
        statuses[name]=stage.get('stage','starting')
        for seed in cfg['seeds']:
            folder=root/'runs'/(f'HASAC_{seed}' if name.startswith('M_') else f'HASAC_Curriculum_{seed}')
            s=read(folder/'status.json') if (folder/'status.json').exists() else {}
            history=read(folder/'evaluation.json') if (folder/'evaluation.json').exists() else []
            updates=[read(p) for p in sorted((folder/'updates').glob('*.json'))]
            episodes=[]
            for u in updates:
                for e in u['episodes']:
                    key=e['episode_sha256']
                    if key not in cache:
                        payload=json.loads(gzip.decompress(Path(e['episode']).read_bytes()))
                        vis=[any(q is not None for reports in t['observation']['reports'] for q in reports)
                             for t in payload['trace']]
                        cache[key]=dict(ever_visible=any(vis),frames=len(vis),visible_frames=sum(vis))
                    episodes.append(e|cache[key]|dict(phase=u.get('phase','hidden_root')))
            metrics={}
            for phase in sorted({e['phase'] for e in episodes}):
                rr=[e for e in episodes if e['phase']==phase]
                metrics[phase]=dict(episodes=len(rr),first=sum(e['captures']>0 for e in rr),
                                   double=sum(e['double_capture'] for e in rr),
                                   ever_visible=sum(e['ever_visible'] for e in rr),
                                   steps=sum(e['decisions'] for e in rr))
            decision=read(folder/'decision.json') if (folder/'decision.json').exists() else None
            record=dict(experiment=name,seed=seed,status=s.get('stage','queued'),steps=s.get('steps',0),
                        history=history,training=metrics,decision=decision)
            records.append(record)
            phase='待评估';current=hidden='—'
            if history:
                h=history[-1];phase=h.get('phase','hidden_root');d=h['deterministic'];p=h.get('hidden_probe',d)
                current=f'{d["double"]}/{d["episodes"]}';hidden=f'{p["double"]}/{p["episodes"]}'
            lines.append(f'|{name}/{seed}|{record["status"]}|{record["steps"]}|{phase}|{current}|{hidden}|')
    lines+=['','可见课程的成功率与原隐藏Root成功率分别列出；不能把前者当成主任务结果。', '',
            '## 实际反馈密度','',
            '|实验/种子|训练阶段|训练转移|局数|当前局曾看到目标|至少首捕|双捕|',
            '|---|---|---:|---:|---:|---:|---:|']
    for r in records:
        for phase,d in r['training'].items():
            lines.append(f'|{r["experiment"]}/{r["seed"]}|{phase}|{d["steps"]}|{d["episodes"]}|{d["ever_visible"]}|{d["first"]}|{d["double"]}|')
    lines+=['','不同阶段/策略的轨迹分布不同，以上为描述性统计，不能把每个帧或局当成独立训练种子。', '',
            '## 状态与解释','']
    for name,stage in statuses.items():lines.append(f'- {name}协调器：{stage}。')
    for r in records:
        if r['decision']:
            lines.append(f'- {r["experiment"]}/{r["seed"]}结束结果：{r["decision"]["status"]}；'
                         f'梯度更新{r["decision"]["gradient_updates"]}。')
            if r['decision']['status']=='gate_failed':
                lines.append('  可见课程门槛未过，尚未执行后续隐藏训练，不能记作完成主任务训练后失败。')
    lines+=['','若从头轨迹很少见到目标而课程能改善，只支持当前课程与额外预算组合，需等总步数对照才能单独归因。'
            '若可见已掌握但隐藏仍弱，继续检查探索迁移、记忆/表示与对抗难度。软件异常与学习门槛失败分别保留。', '',
            '固定80%训练内目标不是发表门槛。无论本轮通过与否，都不能仅据这些少量训练情形宣称广分布强基线或所有RL不适用。', '',
            '## 所有固定检查点','',
            '|实验/种子|累计新增步|阶段|阶段步|当前确定性双捕|当前随机双捕|隐藏Root确定性探针|',
            '|---|---:|---|---:|---:|---:|---:|']
    for r in records:
        for h in r['history']:
            d=h['deterministic'];s=h['stochastic'];p=h.get('hidden_probe',d)
            lines.append(f'|{r["experiment"]}/{r["seed"]}|{h["steps"]}|{h.get("phase","hidden_root")}|{h.get("phase_steps",h["steps"])}|'
                         f'{d["double"]}/{d["episodes"]}|{s["double"]}/{s["episodes"]}|{p["double"]}/{p["episodes"]}|')
    lines+=['','没有最好分选模，所有固定检查点公开。曲线横轴包含N的额外课程交互。', '',
            '![隐藏任务开发曲线](hidden_learning_curves.png)', '',
            '[M原始记录](../e45m_hasac_task_20261101/) · [N原始记录](../e45n_hasac_curriculum_20261101/)']
    result=dict(updated_at=time.strftime('%FT%T%z'),coordinators=statuses,runs=records,
                independent_test_used=False,policy_selection=False,
                analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for filename,data in [('analysis.json',result),('exposure_cache.json',cache)]:
        temp=OUT/(filename+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(OUT/filename)
    temp=OUT/'分析.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(OUT/'开发结果分析.md')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,(name,_) in zip(axes,SOURCES):
        for r in [x for x in records if x['experiment']==name]:
            color='#2563eb' if r['seed']==17 else '#d97706'
            hs=r['history']
            ax.plot([h['steps']/1000 for h in hs],
                    [100*h.get('hidden_probe',h['deterministic'])['double']/h.get('hidden_probe',h['deterministic'])['episodes'] for h in hs],
                    marker='o',color=color,label=f'Seed {r["seed"]}, deterministic Root')
            # Never misplot visible-phase stochastic success as hidden success.
            rr=[h for h in hs if h.get('phase','hidden_root')=='hidden_root']
            ax.plot([h['steps']/1000 for h in rr],
                    [100*h['stochastic']['double']/h['stochastic']['episodes'] for h in rr],
                    marker='x',ls='--',color=color,label=f'Seed {r["seed"]}, stochastic Root')
        ax.set_title(name);ax.set_ylim(-3,103);ax.grid(alpha=.2)
        ax.set_xlabel('All additional transitions (thousands)');ax.legend(fontsize=7,loc='upper left')
    axes[0].set_ylabel('Original hidden-task double capture (%)')
    fig.suptitle('HASAC development only: original hidden training situations')
    fig.tight_layout();fig.savefig(OUT/'hidden_learning_curves.png',dpi=160);fig.savefig(OUT/'hidden_learning_curves.pdf');plt.close(fig)
    print(result['updated_at'],statuses,flush=True)
    return all(stage in ('complete','failed') for stage in statuses.values())


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args();started=time.monotonic()
    while True:
        terminal=analyze()
        if terminal or not args.watch or time.monotonic()-started>13*3600:break
        time.sleep(60)


if __name__=='__main__':main()
