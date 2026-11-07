"""Read-only experiment audit and report export; never changes training choices."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review(root):
    config=read(root/'experiment.json');rows=[]
    stages=('static_test','test')
    for stage in stages:
        for path in sorted((root/stage/'cache').glob('*.json')):
            row=read(path)
            if not row['public_replay_exact']:
                raise RuntimeError('Public replay audit failed: '+str(path))
            rows.append(row)
    training=[]
    for kind in ('RMAPPO','MAPPO_APF'):
        for seed in config['seeds']:
            folder=root/'learning'/f'{kind}_{seed}'
            status=read(folder/'status.json') if (folder/'status.json').exists() else {}
            selected=read(folder/'selected.json') if (folder/'selected.json').exists() else {}
            training.append(dict(kind=kind,seed=seed,steps=status.get('steps',0),
                stage=status.get('stage','queued'),selected_steps=selected.get('steps'),
                validation_capture=selected.get('double_capture')))
    grouped=collections.defaultdict(list)
    for row in rows:grouped[row['suite'],row['opponent'],row['arm'],row['train_seed']].append(row)
    records=[]
    for (suite,opponent,arm,seed),rr in sorted(grouped.items()):
        latency=[v for r in rr for v in r['latencies_ms']]
        records.append(dict(suite=suite,opponent=opponent,arm=arm,seed=seed,
            independent_groups=len({r['group'] for r in rr}),episodes=len(rr),captures=sum(r['double_capture'] for r in rr),
            capture_rate=float(np.mean([r['double_capture'] for r in rr])),
            utility=float(np.mean([r['utility'] for r in rr])),
            cost=float(np.mean([r['completion_cost'] for r in rr])),
            p95_ms=float(np.percentile(latency,95)),overruns=sum(v>200 for v in latency)))
    if records:
        with (root/'per_seed_results.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    state=read(root/'status.json')
    lines=['# E45-I：九族基线先导','',
        f'更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %z")}。阶段：`{state["stage"]}`。',
        '',f'已保存基准对局 **{len(rows)}/{config["formal_tasks"]}**；新RL累计 **{sum(r["steps"] for r in training):,}** 步。',
        '', '本轮先导为14组公共历史（6组ID、8组OOD）、两隐藏世界、两已知对手。网络种子与隐藏世界是组内重复；不是952张独立地图。每模型上限10万步，不宣称已充分调优或收敛。',
        '', '|方法|种子|状态|已登记步数|选中步数|验证双捕|','|---|---:|---|---:|---:|---:|']
    for r in training:
        rate='—' if r['validation_capture'] is None else f'{r["validation_capture"]:.2%}'
        lines.append(f'|{r["kind"]}|{r["seed"]}|{r["stage"]}|{r["steps"]}|{r["selected_steps"]}|{rate}|')
    lines += ['', '已有六族＋固定APF的完整结果见 [static_benchmark.md](static_benchmark.md)；九族完成后输出 [nine_family_benchmark.md](nine_family_benchmark.md)。逐种子明细见 [CSV](per_seed_results.csv)。未生成的报告表示该阶段尚未完成。',
        '', '代码直接使用[MAPPO作者仓库](https://github.com/marlbenchmark/on-policy)的actor/critic、GRU、分布和PPO更新；上游commit：`'+config['upstream_commit']+'`。APF/MAPPO+APF为本任务的公开信念适配，不是完整OPEN/DACOOP复现。',
        '', '工程检查：4项单元检查、6个三步前缀，以及两种方法各一次真实完整对局＋GPU官方PPO更新。工程记录见 [engineering](../e45i_engineering_v2_20261031/engineering_updates.json)，不是正式性能证据。',
        '', '[注册配置](experiment.json) · [协议](PROTOCOL.md) · [APF验证选参](apf_selected.json) · [执行资源](execution.json) · [状态](status.json)',
        '', '最终性能表按各对手、ID/OOD分别报告。已有G/H模型和源文件保持原记录；测试不参与选模。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    result=dict(updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),test_episodes=len(rows),
        expected=config['formal_tasks'],all_public_replay_exact=all(r['public_replay_exact'] for r in rows),
        training=training,performance_scope='bounded pilot, not converged MARL')
    # The final audit also verifies full episode files and selected checkpoints.
    if (root/'decision.json').exists():
        hashes={}
        for row in rows:
            path=row['episode']
            if sha(path)!=row['episode_sha256']:raise RuntimeError('Episode hash mismatch: '+path)
            path=row.get('checkpoint')
            if path:
                hashes.setdefault(path,sha(path))
                if hashes[path]!=row['checkpoint_sha256']:raise RuntimeError('Checkpoint hash mismatch: '+path)
        for section in ('source_hashes','protected_hashes','bank_hashes','upstream_hashes'):
            for path,digest in config[section].items():
                if sha(path)!=digest:raise RuntimeError('Frozen artifact changed: '+path)
        if len(rows)!=config['formal_tasks']:raise RuntimeError('Incomplete final benchmark')
        result['full_episode_hashes_verified']=len(rows)
        result['frozen_sources_and_models_verified']=True
    (root/'review_status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        result=review(args.output)
        print(json.dumps(dict(episodes=result['test_episodes'],steps=sum(r['steps'] for r in result['training'])),ensure_ascii=False),flush=True)
        if not args.watch or (args.output/'decision.json').exists():break
        time.sleep(20)


if __name__=='__main__':main()
