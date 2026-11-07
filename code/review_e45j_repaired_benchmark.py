"""Progress/export and final artifact audit; never changes model selection."""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import time


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def optional(path):return read(path) if path.exists() else {}
def write(path,value):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def review(root):
    cfg=read(root/'experiment.json');status=optional(root/'status.json');done=optional(root/'decision.json')
    execution=optional(root/'execution.json');training=[];rows=[]
    for path in sorted((root/'test/cache').glob('*.json')):
        if path.name.endswith('.failure.json'):continue
        r=read(path)
        if not r['public_replay_exact']:raise RuntimeError('Public replay mismatch '+str(path))
        rows.append(r)
    if len({r['key'] for r in rows})!=len(rows):raise RuntimeError('Duplicate benchmark key')
    for kind in ('RMAPPO','MAPPO_APF'):
        for seed in cfg['seeds']:
            folder=root/'learning'/f'{kind}_{seed}';s=optional(folder/'status.json');resume=optional(folder/'resume.json')
            selection=optional(folder/'selected.json');decision=optional(folder/'decision.json')
            updates=sorted((folder/'updates').glob('*.json'));last=optional(updates[-1]) if updates else {}
            gates=read(folder/'gates.json') if (folder/'gates.json').exists() else []
            run=next((r for r in execution.get('training',[]) if r['kind']==kind and r['seed']==seed),{})
            pid=run.get('pid');alive=pid is not None and Path(f'/proc/{pid}').exists()
            stage=s.get('stage','initial_validation' if alive else 'queued')
            if status.get('stage')=='failed' and not alive and stage!='complete':stage='interrupted'
            elif alive and stage=='failed':stage='resuming'
            training.append(dict(kind=kind,label=cfg['method_labels'][kind],seed=seed,
                stage=stage,phase=last.get('phase','initial_validation'),
                steps=max(last.get('steps',0),resume.get('steps',0),s.get('steps',0)),
                updates=last.get('update',0),selected_steps=selection.get('steps'),validation_capture=selection.get('double_capture'),
                recent_training_capture=last.get('training_capture'),recent_blocked_fraction=last.get('blocked_fraction'),
                gates=gates,learning_gate_failed=decision.get('learning_gate_failed',False),pid=pid,alive=alive))
    grouped=collections.defaultdict(list)
    for r in rows:grouped[r['suite'],r['opponent'],r['arm'],r['train_seed']].append(r)
    records=[]
    for (suite,opp,arm,seed),rr in sorted(grouped.items()):
        records.append(dict(suite=suite,opponent=opp,arm=arm,label=cfg['method_labels'].get(arm,arm),seed=seed,
            independent_groups=len({r['group'] for r in rr}),episodes=len(rr),captures=sum(r['double_capture'] for r in rr),
            capture_rate=sum(r['double_capture'] for r in rr)/len(rr),utility=sum(r['utility'] for r in rr)/len(rr),
            completion_cost=sum(r['completion_cost'] for r in rr)/len(rr)))
    if records:
        with (root/'per_seed_results.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    result=dict(updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),stage=status.get('stage'),
        test_episodes=len(rows),expected=cfg['formal_tasks'],all_public_replay_exact=all(r['public_replay_exact'] for r in rows),
        training=training,training_steps=sum(r['steps'] for r in training),complete=bool(done),convergence_claim=False)
    if done:
        checked={}
        for r in rows:
            if sha(r['episode'])!=r['episode_sha256']:raise RuntimeError('Episode SHA mismatch')
            if r.get('checkpoint'):
                checked.setdefault(r['checkpoint'],sha(r['checkpoint']))
                if checked[r['checkpoint']]!=r['checkpoint_sha256']:raise RuntimeError('Model SHA mismatch')
        for section in ('source_hashes','protected_hashes','bank_hashes','upstream_hashes'):
            for path,digest in cfg[section].items():
                if sha(path)!=digest:raise RuntimeError('Frozen source/bank SHA mismatch '+path)
        if (root/'amendment_01.json').exists():
            amendment=read(root/'amendment_01.json')
            for path,digest in amendment['source_hashes'].items():
                if sha(path)!=digest:raise RuntimeError('Amendment source SHA mismatch '+path)
            if done.get('amendment_sha256')!=sha(root/'amendment_01.json'):raise RuntimeError('Amendment identity mismatch')
        if sha(root/'apf_selected.json')!=read(root/'apf_freeze.json')['sha256']:raise RuntimeError('Gains changed')
        if len(rows)!=cfg['formal_tasks']:raise RuntimeError('Incomplete benchmark')
        result.update(full_episode_hashes_verified=len(rows),frozen_artifacts_verified=True)
    fixed=optional(root/'fixed_repair_benchmark.json')
    lines=['# E45-J：修复后baseline与新鲜benchmark','',
        f'更新时间：{result["updated_at"]}。阶段：`{result["stage"]}`；完整对局 **{len(rows)}/{cfg["formal_tasks"]}**；累计训练 **{result["training_steps"]:,}** 步。','',
        '新鲜42组公共历史（18 ID/24 OOD），两隐藏世界、两已知对手。主九族2856局；旧版与初始化辅助对照2184局。地图组是独立单位，网络种子和隐藏世界为组内重复。','',
        '新学习族为 **RMAPPO-Clearance、MAPPO＋RouteAPF**；固定执行器为 **RouteAPF**。它们包含明确的适配/课程改变，不能称为原版论文性能复现。原I代码、模型和结果保持冻结。','',
        '每模型最多50万步，包含全部课程交互；基础课程不达门槛会提前停止并保留失败标记。势差之外增加非法意图训练成本，最终仍用原始效用。选模只看原隐藏验证集，不看测试。','',
        '|方法|种子|状态/阶段|训练步|近期课程成功|缩幅比例|选中步|主验证双捕|门槛失败|','|---|---:|---|---:|---:|---:|---:|---:|---|']
    fmt=lambda x:'—' if x is None else f'{x:.2%}'
    for r in training:
        lines.append(f'|{r["label"]}|{r["seed"]}|{r["stage"]}/{r["phase"]}|{r["steps"]}|{fmt(r["recent_training_capture"])}|{fmt(r["recent_blocked_fraction"])}|{r["selected_steps"]}|{fmt(r["validation_capture"])}|{r["learning_gate_failed"]}|')
    if fixed:
        lines+=['','固定执行器的完整336局结果（两方法，非新学习族最终成绩）：','',
            '|suite|opponent|method|captures|rate|','|---|---|---|---:|---:|']
        for r in fixed['rows']:lines.append(f'|{r["suite"]}|{r["opponent"]}|{r["label"]}|{r["double_capture"]}/{r["episodes"]}|{r["double_capture_rate"]:.2%}|')
    lines+=['','[预注册协议](PROTOCOL.md) · [配置/SHA](experiment.json) · [固定执行器结果](fixed_repair_benchmark.md) · [九族主表](nine_family_benchmark.md) · [全部对照](all_controls_benchmark.md) · [配对统计](paired_statistics.json) · [逐种子CSV](per_seed_results.csv) · [机器状态](review_status.json)','',
        '链接目标在对应阶段完成后生成。训练日志中的单目标课程成功率不能当作最终2v2双捕率。只报告实际完成步数、门槛和独立测试，不声明收敛。']
    if (root/'amendment_01.json').exists():
        amendment=read(root/'amendment_01.json')
        lines+=['','**15:15支持耗尽后的续训修订：** 见[修订01](AMENDMENT_01.md)。前'+str(amendment['earlier_training_steps'])+
            '步沿用旧适配器，之后从同一优化器/RNG/课程检查点续训。0.5/0.25支持耗尽时，以共同的0.125网格仅凭公开历史恢复；这会改变离散后验，不能称全程冻结离散估计器。所有最终比较臂和公共重放共同启用，原336局逐帧签名一致后复用。恢复可能耗时数秒，在线超时如实计入。',
            '', '[同一失败任务完整复现](support125_fault_replay.json) · [336局不变性凭据](support125_existing_trace_proof.json) · [修订源码SHA](amendment_01.json)']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');write(root/'review_status.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        if (args.output/'experiment.json').exists():
            r=review(args.output);print(json.dumps({k:r[k] for k in ('stage','test_episodes','training_steps','complete')}),flush=True)
            if r['complete'] or r['stage']=='failed' or not args.watch:break
        elif not args.watch:raise RuntimeError('Experiment is not registered')
        time.sleep(20)


if __name__=='__main__':main()
