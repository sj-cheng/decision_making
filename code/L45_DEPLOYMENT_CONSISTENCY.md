部署一致性与动作选择可学习性实验

入口 `run_l45_deployment_consistency.py`，默认输出 `current/l45_deployment_consistency_20261024`。完整冻结协议在输出目录 `PLAN.md` 与 `experiment.json`。本轮是新方法研发的有界前置试验，使用历史局面明确登记复用；不把条件续跑或交叉验证包装成新场景确认。

运行：

```bash
/home/sjc/miniconda3/envs/dm_env_py311/bin/python code/build_deployment_search.py
/home/sjc/miniconda3/envs/dm_env_py311/bin/python -m unittest discover -s code/tests -p test_l45_deployment_consistency.py -v
/home/sjc/miniconda3/envs/dm_env_py311/bin/python code/run_l45_deployment_consistency.py pipeline --smoke --workers 4
/home/sjc/miniconda3/envs/dm_env_py311/bin/python code/run_l45_deployment_consistency.py pipeline --workers 80
/home/sjc/miniconda3/envs/dm_env_py311/bin/python code/run_l45_deployment_consistency.py watch --once
```

阶段支持 `prepare、audit、diagnose、pilot、fit、report、replay、watch、pipeline`；参数支持 `--workers、--output、--smoke、--once`。`--workers` 不纳入科学协议，允许改变 worker 数恢复；重跑只执行未原子提交的任务。源码、私有二进制、平台、模型、源数据、缓存或配对改变则停止。`fit` 的场景分区、拟合和统计均确定化；重启从头重算小型拟合，不读取确认数据。

`cpp/solvers/simultaneous_puct.hpp` 是同步基线：节点代表联合状态，双方分别按团队回报执行 decoupled UCT 与渐进拓宽，同步选择后推进一个时间步。双方选择不访问对方本步动作。连续动作先验仍来自冻结 actor，随机 rollout 也联合推进。它不是已证明收敛的均衡算法。

`build_deployment_search.py` 从已冻结根审计扩展的源码副本构建 `cpp/build_deployment`，历史源码、模型和共享库均不覆盖。私有副本中新增环境转移计数，包括安全 rollout 采样器内部调用的 `step`；分别统计策略/价值网络评估。旧搜索加计数后须保持动作、根统计和随机流不变。

正式续跑固定 99 个已有 O 首捕局面，O/L 各比较交替与同步，共 396 条；真实 Search Only 对手仍是原搜索。每种配置完整遵守真实剩余时间。O 交替续跑与历史 A0/stream0 的完整轨迹、科学指标及双方最终 RNG 逐一比较。L 的条件结果不代表 L 自己形成前缀时的整局效果。

`learning/deployment_scorer.py` 使用已有成对 A0/AQ 三流结果，外层 5 折按物理场景评估，内层 4 折选择 Ridge/单分裂树与替换阈值。均值方差、阈值及方法选择只用当前训练分区。输入限原 actor 观测、当前动作和根统计。配对 bootstrap 在状态内先对三个流平均，再按物理场景重采样。门槛失败就停止这条评分器分支，不自动新增数据或训练 actor。

报告 `REPORT.md`，逐任务缓存 `semantic/、continuation/`，分区和预测 `scorer_result.json`，拟合输入与标签 `scorer_dataset.json`，工程证据 `audit_result.json、replay.json`。看板 `status.json` 列实际 worker PID、线程数和已提交任务；并行耗时仅作成本审计，不能宣称单进程加速。主目录中的启动失败 smoke 归档保留用于追踪，不参与效果结论，其场景仍被后续银行排除。
