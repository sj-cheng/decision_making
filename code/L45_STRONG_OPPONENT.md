# l45 强对手短续训

独立入口 `run_l45_strong_opponent.py`，不调用历史 pipeline，不修改原环境或搜索实现。
唯一训练变量：A 对手配比 32/32/0，B 配比 16/16/32；每批各 64 局合并采样、分别更新。
两组与三个续训随机种子都从推荐的旧 PPO seed0 第 2048 局完整 actor、critic、Adam 状态开始。
三个续训种子共享一个原始训练起点，不能据此推断跨预训练起点稳定性。

使用 `/home/sjc/miniconda3/envs/dm_env_py311/bin/python`。默认正式输出
`current/l45_strong_opponent_20261023`，默认 80 workers，每进程单线程。

```bash
python code/run_l45_strong_opponent.py --stage pipeline --workers 80
python code/run_l45_strong_opponent.py --stage watch --once
python code/run_l45_strong_opponent.py --stage replay
```

所有阶段可单独调用：`prepare audit pilot expand confirm guard timing report watch replay`。
`expand` 只接受锁定的通过试点；`confirm` 只接受锁定的六个选模结果。
更换 `--workers` 可以续跑；更改源码、协议、场景、来源、模型或已提交缓存会拒绝混合。
异常立即停止并保留完整续跑状态，不自动修改干预或增加训练。

正式部署追捕为 1000 次模拟、深度 100、βπ=.5、βv=0；逃逸与固定对手保持 βπ=.5、βv=.1。
Search Only 为 3000 次、深度 100、两个 β 均为 0。训练 critic 不导出给搜索。
PPO 沿用原 3e-5/3e-4、4 epochs、256 minibatch、clip .1、γ/λ=1/.95、entropy .001、梯度裁剪 .5。
每 epoch 整批精确高斯 KL 超过 .02 后停止本批后续 actor 更新。

先种子0两组各512局，新增0/256/512局做实际 MCTS 验证。
选优后至少提高3个百分点、强对手子项改善且非法率及其他场景不过界，才补齐两组的种子1、2。
最多3072局。试点不通过即结束；512局预算不等同于收敛证明。

全部场景启动前冻结：验证60/30、确认200/100、逃逸回归100/100、计时30个四障碍场景，另设独立预热。
按初态、速度和动作边界去重全部可恢复的 `current/l45*/**/scenarios.json` 和实际 `**/rollouts/*.json`，
包括 smoke、首次启动归档。A/B 共享训练场景是预先规定的唯一例外；不宣称覆盖全部NTE预训练状态。

确认六个选优模型与两个对照最多7200局，同权重同配置共享缓存。
15项事先登记比较、20000次物理场景配对 bootstrap，统计种子45020261023，
保留同场景三个对手与全部模型，不重采样训练种子。
相对当前推荐至少+3个百分点且同时95%下界>0，并满足强对手改善和回归限制才可采用。
B对同种子A的比较只解释对手配比贡献，不额外限制更好模型的采用。
逃逸回归通过后关闭全部本实验并行 worker，交错单进程计时。
最终排序按成功率、非法率、完成代价、单步耗时、A优先、种子、较早检查点。

```bash
python -m unittest discover -s code/tests -p test_l45_strong_opponent.py -v
python code/run_l45_strong_opponent.py --smoke --workers 8 \
  --output current/l45_strong_opponent_smoke_20261023
```

最小 smoke 只做种子0两组各一个完整64局PPO批次及实际搜索验证，始终禁止自动扩展及正式质量结论。
确认、回归、计时、导出与失败回退通过临时目录中的受控功能测试覆盖。

中断恢复以 `training/{A,B}/seedN/latest.json` 的原子检查点提交为准。
每条完整轨迹保存到不可变摘要校验缓存，任务含组别、种子、更新轮次、场景、采样和搜索种子。
更新前重算动作概率并屏蔽失活机器人；丢失的进度投影可从检查点与轨迹缓存重建。
完整模型、优化器、RNG、待验证标记和选模历史均保留。

最终查看 `REPORT.md`、`results.json`、`recommended.json`、`recommended/models`。
未确认新收益时继续采用 `old0:novalue`，保留既有历史提升；从不覆盖原模型。
推荐加载入口是 `load_recommended_solver(args, problem, role, search_seed)`，`--stage replay` 验证原搜索精确重放。
