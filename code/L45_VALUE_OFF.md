# 现成模型与旧价值配置独立确认

入口 `confirm_l45_value_off.py` 只评测，不含训练调用。默认输出为
`current/l45_value_off_confirm_20261023`，历史 l45 和实验结果保持不变。

七套冻结模型（原 l45、旧 PPO 三个 2,048 局最终模型、A 组三个锁定的
512 局模型）分别比较 beta_value=0.1/0。追捕预算为 1,000 次、深度 100，
beta_policy=0.5。对手始终保持原配置；Search Only 对手为 3,000 次。
关闭价值只作用于追捕部署，逃逸回归使用原 beta_value=0.1。

正式确认使用新的 200 四障碍、100 无障碍场景和三个对手，共 12,600 局。
历史三个原环境实验中的银行及可恢复 rollout 初态均去重。19 项比较按物理
场景共同做 20,000 次 bootstrap，正式门槛使用最大中心化偏差的同时 95%
下界；单检查点统计不重采样训练种子。原 baseline 增益至少 5 个百分点、
下界大于零、自身非法增加不超过 2 个百分点、无障碍下降不超过 3 个百分点
才进入逃逸回归。允许单个独立确认有效的检查点入选，全部训练种子另列。

smoke 与正式银行使用不同种子命名空间。正式准备额外排除本轮 smoke，以及
首次启动后因命名空间重合而中止的归档银行；这些历史清单也纳入来源指纹。

逃逸回归在新 100 四障碍/100 无障碍场景、三个对手下进行，生存率下降
不超过 3 个百分点，自身非法增加不超过 2 个百分点。通过后关闭本实验
并行 pool，对候选及 baseline 预热并做 30 场景单进程计时；成功率优先，
耗时只报告并在成功率/非法/完成用时平分时作为选择依据。

从仓库根目录运行：

```bash
/home/sjc/miniconda3/envs/dm_env_py311/bin/python -u code/confirm_l45_value_off.py --workers 80
/home/sjc/miniconda3/envs/dm_env_py311/bin/python code/confirm_l45_value_off.py --stage watch
/home/sjc/miniconda3/envs/dm_env_py311/bin/python code/confirm_l45_value_off.py --stage watch --once
```

先使用隔离输出做功能 smoke：

```bash
/home/sjc/miniconda3/envs/dm_env_py311/bin/python -u code/confirm_l45_value_off.py --smoke --output /tmp/l45_value_off_smoke --workers 2
```

阶段包括 `prepare/audit/confirm/guard/timing/report/watch/replay`，默认 `pipeline`
自动串联所有需要的阶段。重复相同命令可按原子缓存续跑，workers 可改变；
源码、模型、场景或科学配置变化会拒绝混合续跑。smoke 使用每银行 1 场景、
3 次模拟，深度仍为 100；它永不产生正式提升结论。

`recommended.json` 和 `recommended/models/` 是可加载结果，包含追捕/逃逸
独立配置、模型指纹、来源和性能证据。无模型通过则显式回退原 l45。
`load_recommended_solver(args, problem, role, search_seed)` 校验指纹后构造原
C++ 搜索；`--stage replay` 重放确认集首场景与原纯策略对手，检查科学结果
完全相同，不向确认集添加样本。计时原始每步数据在 `serial_timing`；
并行耗时只用于进度估计。

本轮不自动追加训练、修改奖励、增加预算或恢复局部观测方案。未通过时
保留失败结论；完整结果见输出目录的 REPORT.md、results.json 和阶段报告。
