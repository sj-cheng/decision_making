# l45 搜索指导微调

独立入口 `run_l45_search_feedback.py`，默认结果目录为
`current/l45_search_feedback_20261023`。历史训练器、环境和搜索算法不修改。

两组从当前推荐 `old0:novalue` 的第 2048 局完整状态续接：

| 组别 | 本轮 actor 损失 |
|---|---|
| S | L_search |
| J | L_search + 0.1 × (L_PPO_clip − 0.001H) |

PPO 仅使用网络实际采样执行的限幅前动作及概率；教师动作只进入监督项。
每批 64 局，从实际轨迹按首捕前/后分层选取最多 1024 个不同决策状态，
用本组当前批次冻结快照运行 MCTS 生成联合动作标签。下一批刷新快照。
动作均值监督按本局动作半范围归一化，屏蔽失活机器人，不另设方差标签。

训练对手两组均为 16 局原纯策略、16 局原 Search＋Learning、32 局 Search Only；
后两者固定为原模型、1000/3000 次、深度 100。追捕教师及部署为
1000 次、深度 100、βπ=0.5、βv=0；逃逸部署保持 βv=0.1。

先运行种子 0 两组各 512 局，每 256 局搜索验证。相对当前推荐和原 l45
关闭价值两个参考均达到 +3 pp、强对手子项改善且回归可接受，才补齐两组
种子 1、2。正式上限 3072 局、49152 次教师查询，不自动延长。

验证使用每对手 100 个四障碍及 50 个无障碍场景；独立确认使用全新
400/100 个场景。锁定六个检查点后，16 项比较共同做 20000 次物理场景
配对 bootstrap。新训练模型须对两个参考分别 +3 pp 且同时下界大于零。
J−S 的同时下界大于零才确认该固定检查点的额外 PPO 贡献。

逃逸回归使用新 100/100 场景，角色配置保持原样。通过后关闭实验并行池，
在 30 个新四障碍场景上做单进程计时。无候选通过则保留当前推荐。

## 命令

在仓库根目录运行：

```bash
/home/sjc/miniconda3/envs/dm_env_py311/bin/python -m unittest discover -s code/tests -p test_l45_search_feedback.py -v
/home/sjc/miniconda3/envs/dm_env_py311/bin/python -u code/run_l45_search_feedback.py --smoke --output current/l45_search_feedback_smoke_20261023 --workers 8
/home/sjc/miniconda3/envs/dm_env_py311/bin/python -u code/run_l45_search_feedback.py --workers 80
/home/sjc/miniconda3/envs/dm_env_py311/bin/python code/run_l45_search_feedback.py --stage watch --once
/home/sjc/miniconda3/envs/dm_env_py311/bin/python code/run_l45_search_feedback.py --stage replay
```

阶段入口：pipeline / prepare / audit / pilot / expand / confirm / guard / timing /
report / watch / replay。`--workers` 可在中断后改变；科学配置和源码变化
拒绝混合续跑。确认和扩展必须有锁定的前置通过记录。

smoke 采用独立场景命名空间、每组一个 64 局批次、32 个教师状态和低搜索预算，
仅验证功能，不宣称性能提升，也不会自动扩大。正式场景排除全部可恢复
历史 l45 场景、实际 rollout、smoke 及归档；去重范围不包含无法恢复的 NTE
预训练状态。

完整状态和采样/教师缓存按批次提交。中断后重用已完成样本和标签；
缓存、模型、源码或场景指纹不符会停止。`REPORT.md`、`results.json`、
`recommended.json` 和 `recommended/models` 为最终交付，原模型不覆盖。
