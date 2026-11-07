"""Read-only L learning and sensor-exposure analysis; never selects a policy."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/e45l_analysis_plot')
import argparse
import gzip
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'current/e45l_marl_diagnosis_20261101'


def read(p):return json.loads(p.read_text())


def analyze(root):
    cfg=read(root/'experiment.json');status=read(root/'status.json');records=[]
    cache=read(root/'exposure_cache.json') if (root/'exposure_cache.json').exists() else {}
    lines=['# E45-L：初始化、探索与主任务学习诊断', '',
           f'更新时间：{time.strftime("%FT%T%z")}；协调器：{status["stage"]}。', '',
           '同6个训练情形、固定RootPUCT3000，原隐藏观察和双捕规则；以下仅开发证据。', '',
           '|初始化/种子|完成步数|最新评估步|确定性双捕|随机双捕|最近10批首捕/双捕|',
           '|---|---:|---:|---:|---:|---|']
    for init in cfg['initializations']:
        for seed in cfg['seeds']:
            f=root/'runs'/f'{init}_{seed}'
            if not (f/'evaluation.json').exists():continue
            h=read(f/'evaluation.json');updates=[read(p) for p in sorted((f/'updates').glob('*.json'))]
            summaries=[]
            for u in updates:
                for e in u['episodes']:
                    key=e['episode_sha256']
                    if key not in cache:
                        payload=json.loads(gzip.decompress(Path(e['episode']).read_bytes()))
                        vis=[any(q is not None for reports in t['observation']['reports'] for q in reports)
                             for t in payload['trace']]
                        cache[key]=dict(ever_visible=any(vis),visible_frames=sum(vis),frames=len(vis),
                                        first_visible=next((i for i,v in enumerate(vis) if v),None))
                    summaries.append(e|cache[key]|dict(update=u['update']))
            def exposure(part):
                if not part:return dict(episodes=0)
                return dict(episodes=len(part),ever_visible=sum(e['ever_visible'] for e in part),
                            first_capture=sum(e['captures']>0 for e in part),double_capture=sum(e['double_capture'] for e in part),
                            frames=sum(e['frames'] for e in part),visible_frames=sum(e['visible_frames'] for e in part))
            final=h[-1];last_updates={u['update'] for u in updates[-10:]}
            record=dict(initialization=init,seed=seed,steps=updates[-1]['steps'] if updates else 0,history=h,
                        training_all=exposure(summaries),training_first10=exposure([e for e in summaries if e['update']<=10]),
                        training_last10=exposure([e for e in summaries if e['update'] in last_updates]),
                        last_gate=final['deterministic']['double']/final['deterministic']['episodes']>=.8,
                        stable_last3=len(h)>=3 and all(x['deterministic']['double']/x['deterministic']['episodes']>=.8 for x in h[-3:]))
            records.append(record);d=final['deterministic'];s=final['stochastic'];recent=record['training_last10']
            lines.append(f'|{init}/{seed}|{record["steps"]}|{final["steps"]}|{d["double"]}/{d["episodes"]}|'
                         f'{s["double"]}/{s["episodes"]}|{recent.get("first_capture",0)}/{recent.get("double_capture",0)}，共{recent["episodes"]}局|')
    lines+=['','## 真实传感器暴露机会', '',
            '从完整已提交训练轨迹统计：曾看到目标、至少首捕、最终双捕。只统计当前局真实报告，不将历史预热的旧测量算作当前可见。', '',
            '|初始化/种子|阶段|局数|曾有当前目标可见|至少首捕|双捕|可见目标帧占团队决策帧|',
            '|---|---|---:|---:|---:|---:|---:|']
    for r in records:
        for part,label in [('training_first10','最早10批'),('training_last10','最近10批')]:
            x=r[part]
            if not x['episodes']:continue
            lines.append(f'|{r["initialization"]}/{r["seed"]}|{label}|{x["episodes"]}|{x["ever_visible"]}|'
                         f'{x["first_capture"]}|{x["double_capture"]}|{x["visible_frames"]/x["frames"]:.1%}|')
    lines+=['','曾可见多但首捕少提示追捕/控制学习不足；可见也少则还需检查探索与重发现。'
            '这些指标受当前策略影响，只能定位现象，不单独证明因果，也不能据此断定强化学习不适用。', '',
            '## 训练内能力门槛', '',
            '最新评估至少5/6确定性双捕是本轮80%开发目标；额外列出最后3次评估均过线作为稳定性描述，不改注册结果。']
    for r in records:
        lines.append(f'- {r["initialization"]}/{r["seed"]}：最新门槛{r["last_gate"]}；最后3次均过线{r["stable_last3"]}。')
    lines+=['','若尚未完成，不将中途成绩当最终验收。即使训练内通过，也需广任务、更多种子和独立测试。', '',
            '![学习曲线](learning_diagnosis.png)', '',
            '[注册协议](source_snapshot/E45L_MARL_DIAGNOSIS_PROTOCOL_20261101.md) · [实时状态](LIVE.txt)']
    data=dict(updated_at=time.strftime('%FT%T%z'),pipeline_stage=status['stage'],runs=records,
              test_used=False,policy_selection=False)
    for filename,obj in [('diagnosis_analysis.json',data),('exposure_cache.json',cache)]:
        temp=root/(filename+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');temp.replace(root/filename)
    temp=root/'诊断分析.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(root/'诊断分析.md')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for r in records:
        color='#2563eb' if r['initialization']=='random' else '#d97706'
        for ax,mode in zip(axes,('deterministic','stochastic')):
            ax.plot([h['steps']/1000 for h in r['history']],
                    [100*h[mode]['double']/h[mode]['episodes'] for h in r['history']],
                    marker='o',color=color,ls='-' if r['seed']==17 else '--',
                    label=f'{r["initialization"]}, seed {r["seed"]}')
            ax.set_title(mode+' policy');ax.set_xlabel('Additional transitions (thousands)')
            ax.set_ylim(-3,103);ax.grid(alpha=.2)
    axes[0].set_ylabel('Original-task double capture (%)');axes[1].legend(fontsize=8)
    fig.suptitle('E45-L: same six hidden training situations, fixed RootPUCT')
    fig.tight_layout();fig.savefig(root/'learning_diagnosis.png',dpi=160);fig.savefig(root/'learning_diagnosis.pdf');plt.close(fig)
    print(data['updated_at'],status['stage'],flush=True)
    return status['stage']


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    started=time.monotonic()
    while True:
        stage=analyze(ROOT)
        if not args.watch or stage in ('complete','failed') or time.monotonic()-started>9*3600:break
        time.sleep(60)


if __name__=='__main__':main()
