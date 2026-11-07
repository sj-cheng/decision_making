"""Report all E45-O training checkpoints and the predeclared stability gate."""
import argparse
from pathlib import Path
import run_e45o_hasac_demonstration as o


def analyze(root):
    cfg=o.verify(root)
    demo=o.j.read(root/'demonstrations/manifest.json')
    lines=['# E45-O 示范回放 HASAC：开发结果', '',
        '本报告只描述六个固定训练情形，不是新ID/OOD测试，也不提供独立样本显著性结论。', '',
        f"示范：{len(demo['episodes'])}条完整轨迹、{demo['transitions']}团队转移，"
        f"历史教师搜索动作耗时合计{demo['teacher_online_ms']/1000:.3f}秒、"
        f"模拟转移{demo['teacher_simulated_transitions']}；新教师查询0次。", '',
        'guided每批25%示范，无BC/KL损失；baseline使用原HASAC回放。全部评估关闭教师。', '',
        '|组别|种子|在线步|含示范计费步|更新数|终点确定性双捕|终点随机双捕|后期连续三次≥5/6|',
        '|---|---:|---:|---:|---:|---|---|---|']
    rows=[]
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            folder=root/'runs'/f'{arm}_{seed}'
            decision=o.j.read(folder/'decision.json')
            history=decision['history'];last=history[-1]
            late=[h for h in history if h['steps']>0][-3:]
            stable=len(late)==3 and all(h['deterministic']['double']>=5 and
                h['deterministic']['episodes']==6 for h in late)
            a=last['deterministic'];b=last['stochastic']
            lines.append(f"|{arm}|{seed}|{decision['steps']}|{decision['total_accounted_steps']}|"
                f"{decision['gradient_updates']}|{a['double']}/{a['episodes']}|{b['double']}/{b['episodes']}|{'通过' if stable else '未通过'}|")
            rows.append(dict(decision,arm=arm,seed=seed,stable_training_gate=stable))
    for row in rows:
        lines += ['', f"## {row['arm']}，种子{row['seed']}", '',
            '|在线步|确定性双捕|随机双捕|确定性首捕|确定性原始效用|',
            '|---:|---|---|---|---:|']
        for h in row['history']:
            a=h['deterministic'];b=h['stochastic']
            lines.append(f"|{h['steps']}|{a['double']}/{a['episodes']}|{b['double']}/{b['episodes']}|"
                f"{a['first']}/{a['episodes']}|{a['utility']:.6f}|")
    passed=all(r['stable_training_gate'] for r in rows if r['arm']=='guided')
    lines += ['', '## 后续判断', '',
        ('guided两个种子通过本轮训练内稳定性门槛，可以进一步设计更广训练分布和全新ID/OOD确认。'
         if passed else 'guided尚未跨两个种子通过训练内稳定性门槛。保留完整结果，先检查成功经验利用和退化，不自动追加预算或BC损失。'), '',
        '六条既有教师轨迹的覆盖有限；失败不能推出所有示范辅助方法无效。'
        '实际整批预算超出已在表中披露，历史教师数据跨种子复用；各模型可访问成本与全实验新增采集成本有区别。']
    o.j.save(root/'analysis.json',dict(runs=rows,guided_stable_gate_passed=passed,
        independent_test=False,demonstration_manifest=demo))
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(str(root/'REPORT.md'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=o.OUT)
    analyze(parser.parse_args().output)
