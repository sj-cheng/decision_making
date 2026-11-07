"""Merge the predeclared reward study and near-tie extension for review."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_k]='1'
import argparse
import json
from pathlib import Path
import numpy as np
import run_search_rl_reward_experiment as study


def load_rows(root,split):
    folder=root/split
    manifest=json.loads((folder/'manifest.json').read_text())
    rows=[json.loads((folder/(r['key']+'.json')).read_text()) for r in manifest]
    assert len({r['key'] for r in rows})==len(rows)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=study.trainer.ROOT/'current/search_rl_reward_20261022')
    args=p.parse_args();out=args.output.resolve();extension=out/'completion_extension'
    experiment=json.loads((out/'experiment.json').read_text());config=experiment['config'];seeds=config['seeds']
    assert json.loads((out/'status.json').read_text())['phase']=='complete'
    assert json.loads((extension/'status.json').read_text())['phase']=='complete'
    short=load_rows(out,'short_validation')
    long=load_rows(out,'long_confirmation')+load_rows(extension,'long_confirmation')
    search=load_rows(out,'search_confirmation')+load_rows(extension,'search_confirmation')
    unique_sets=[]
    for rs in (short,long,search):unique_sets.append({r['episode_hash'] for r in rs})
    assert all(unique_sets[i].isdisjoint(unique_sets[j]) for i,j in ((0,1),(0,2),(1,2)))
    assert all(r['tested_simulations']==config['search_budget']*r['steps'] for r in search)
    comparisons=[]
    for rs,method in ((long,'ppo'),(search,'search_ppo')):
        for candidate in study.ARMS[1:]:
            for baseline in ('initial','legacy'):
                for role in (0,1):
                    comparisons.append(study.paired(rs,baseline,candidate,seeds,role,method))
    timing=[]
    for candidate in study.ARMS[1:]:
        for seed in seeds:
            old={r['episode_hash']:r for r in search if r['label']=='legacy' and r['diagnostic_seed']==seed and r['role']==1}
            pairs=[(old[r['episode_hash']],r) for r in search if r['label']==candidate and r['diagnostic_seed']==seed and r['role']==1]
            both=[(a,b) for a,b in pairs if a['double_capture'] and b['double_capture']]
            timing.append(dict(candidate=candidate,seed=seed,n_joint_success=len(both),
                               capture_time_difference=float(np.mean([b['capture_time']-a['capture_time'] for a,b in both])) if both else None,
                               note='Candidate minus same-seed legacy; conditioned on both succeeding.'))
    summaries={k:study.summarize(v) for k,v in [('short',short),('long',long),('search',search)]}
    exploratory=[]
    for arm in study.ARMS[1:]:
        pure=next(c for c in comparisons if c['candidate']==arm and c['baseline']=='legacy' and c['role']==1 and c['method']=='ppo')
        full=next(c for c in comparisons if c['candidate']==arm and c['baseline']=='legacy' and c['role']==1 and c['method']=='search_ppo')
        exploratory.append(dict(arm=arm,pure_positive_supported=pure['ci95'][0]>0 and min(pure['per_seed_delta'])>=0,
                                search_positive_supported=full['ci95'][0]>0 and min(full['per_seed_delta'])>=0))
    result=dict(summaries=summaries,comparisons=comparisons,search_timing=timing,
                exploratory_signals=exploratory,continue_formal=False,
                counts=dict(short_evaluations=len(short),long_evaluations=len(long),full_search_evaluations=len(search)),
                scope='Same pretrained ep512 start, two sampling seeds, pursuer-only reward intervention; all MCTS rewards unchanged.',
                decision='No automatic 50k run. Require robust longer-window gains and independent training-seed replication.')
    study.trainer.write_json(out/'combined_results.json',result)
    lines=['# 第二次捕获奖励：较长训练与完整搜索验证','',
           '本轮已完成。保留第512局双方最佳模型为冻结起点，仅更新追捕。三组最终均追加训练1024局，网络、学习率、PPO参数、课程和固定对手相同。',
           '短验证两个候选只差1局，因此在读取保留确认结果前补充续训双捕奖励候选；原主流程选择记录不变。',
           '模型起点的Adam统一重置；128局之后连续保留各组优化器。所有组使用同一旧环境评估，完整搜索仍用原奖励、1000模拟、深度8、beta_value=0。','',
           '## 较长训练独立确认','',
           '|组|采样种子|角色|成功|自身非法|只抓一个超时|原环境回报|','|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries['long']:
        lines.append(f'|{r["label"]}|{r["seed"]}|{r["role"]}|{r["success"]:g}/{r["n"]}|{r["own_illegal"]:g}|{r["one_capture_timeout"]}|{r["environment_return"]:.3f}|')
    lines+=['','## 完整预算搜索确认','','|组|采样种子|角色|成功|自身非法|只抓一个超时|','|---|---:|---:|---:|---:|---:|']
    for r in summaries['search']:
        lines.append(f'|{r["label"]}|{r["seed"]}|{r["role"]}|{r["success"]:g}/{r["n"]}|{r["own_illegal"]:g}|{r["one_capture_timeout"]}|')
    lines+=['','## 相对原奖励组的追捕变化','','|候选|执行|成功率差|两个采样种子的差|配对95%探索区间|','|---|---|---:|---|---|']
    for c in comparisons:
        if c['baseline']!='legacy' or c['role']!=1:continue
        lines.append(f'|{c["candidate"]}|{c["method"]}|{c["delta"]:+.1%}|{", ".join(f"{v:+.1%}" for v in c["per_seed_delta"])}|[{c["ci95"][0]:+.1%}, {c["ci95"][1]:+.1%}]|')
    lines+=['','![最终结果](combined_results.png)','','## 解释边界与当前决策','',
            '奖励语义：legacy为原奖励；completion_bonus_v1在合法双捕事件额外加1；capture_floor_v1将合法捕获的时间衰减部分补足到每名0.5。额外奖励不用于评估分数。',
            '角色1为追捕、角色0为逃逸。逃逸网络冻结，但其搜索使用的对手预测网络可能因追捕策略改变而改变，所以仍实际评估两角色的搜索。',
            f'完成短验证{len(short)}局、较长确认{len(long)}局、完整预算搜索{len(search)}局。每模型每角色的搜索只有12局，是完整预算的小样本验证，不是正式大样本benchmark。',
            '两个采样种子共享一个预训练种子，并复用相同配对场景；不能把两种子的评估当作两倍独立场景。配对区间保留场景跨种子相关性，但仍属探索结论。',
            '新奖励改变了回报尺度，训练reward和critic loss不能直接跨组解释为任务提升。主要看合法双捕、只抓一个超时和非法率。',
            'continue_formal=false；没有启动每阵营5万局，也没有替换原模型。旧模型哈希、on-policy、冻结逃逸与配对完整性见integrity.json。','',
            '详细方案见PLAN.md；旧轨迹奖励对齐审计见reward_alignment_audit.json；所有配对差和时间指标见combined_results.json。',
            '阶段学习曲线见learning_curve.png/learning_curve.json（如已生成）；这些检查点不用于重新挑奖励或模型。']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['initial',*study.ARMS];colors=['black','tab:blue','tab:green','tab:orange']
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,key,title in zip(axes,('long','search'),('1024 additional PPO episodes','Full MCTS: 1000 simulations, depth 8')):
        for seed in seeds:
            yy=[next(r['success']/r['n'] for r in summaries[key] if r['label']==label and r['role']==1 and r['seed']==(None if label=='initial' else seed)) for label in labels]
            ax.plot(range(4),yy,'o-',label=f'sampling seed {seed}')
        ax.set_xticks(range(4),labels,rotation=20,ha='right');ax.set_title(title);ax.set_ylim(0,1.05);ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Legal double-capture rate')
    fig.tight_layout();fig.savefig(out/'combined_results.png',dpi=150);plt.close(fig)
    print(json.dumps(result['counts']),flush=True)


if __name__=='__main__':main()
