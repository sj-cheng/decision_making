# 首捕前后交叉接管实验

仅冻结模型推理。O=推荐 old0:novalue（PPO seed0 原第2048局），S=硬KL搜索监督新增第512局，L=原l45。S继承PPO起点。本轮训练、教师查询、优化器更新均为0。

唯一候选H：真实首捕后的下一次追捕决策由O切为S。整次搜索固定一个模型，模拟树内不切换。原环境、奖励、观测与搜索算法保持；追捕1000次、深度100、βπ=.5、βv=0；逃逸完整沿用O，βπ=.5、βv=.1。原对手保持，Search Only为3000次。SO只诊断，不参与选模。

## 运行

使用环境 `/home/sjc/miniconda3/envs/dm_env_py311/bin/python`。

```bash
python code/build_phase_search.py
python -m unittest discover -s code/tests -p test_l45_phase_handoff.py
python code/run_l45_phase_handoff.py --smoke --workers 8
python code/run_l45_phase_handoff.py --workers 80
python code/run_l45_phase_handoff.py --stage watch --once
```

正式输出默认`current/l45_phase_handoff_20261023`，smoke默认独立的`current/l45_phase_handoff_smoke_20261023`。入口支持pipeline、prepare、audit、diagnose、validate、confirm、guard、timing、report、watch、replay，以及--workers、--output、--smoke、--once。一个输出目录同时只允许一个写进程。正式使用独立tmux `l45-phase-handoff`，每worker一个线程。

## 数据与随机流

私有扩展只在复制的绑定中增加搜索engine及problem随机状态的导出/恢复，不改原C++算法、历史库或已有训练器。私有类型module_local。模型切换接续搜索随机流，不重新设seed；对手从原快照继续按新状态决策。

诊断复用最新硬KL验证的全部100个四障碍物理场景和Search Only对手。O、S各运行到首捕；每条前缀接O、S各一个后缀。最多200前缀、400后缀。完整状态包含53维物理量和观测记忆、原场景参数、累计奖励、步数、终止记录及双方C++/Python/NumPy/Torch随机状态。剩余时间不重置。零捕、非法、直接双捕不生成实际续跑，仍保留原结果。

OO、SS必须精确重现历史科学指标；审计另逐动作核对连续与分段运行。前缀和后缀按任务原子提交pt与外部摘要，整局缓存另有锁定摘要。未提交任务可重放，已提交内容改变则停止。源码、平台、构建产物、模型、配对种子或场景指纹变化时拒绝混合恢复。worker数不参与科学身份。

后续所有新场景启动前冻结，排除所有可恢复历史银行、实际rollout场景、smoke、归档和未使用场景银行，阶段间去重；不声称穷尽NTE预训练数据。诊断是预登记的历史复用例外。

## 门槛与预算

- 诊断：OS对OO净增至少1/100个合法双捕，自身非法增加≤2点。历史诊断仅决定继续投入；不过即停。报告全100口径、实际分支口径及挽回/丢失；不把相关后缀当独立物理样本。
- 新验证：H、O、L，每对手100四障碍/50无障碍，最多1350请求。
- 独立确认：锁定唯一H后，每对手全新400/100，最多4500请求。预登记H−O、H−L；20000次物理场景配对bootstrap，种子48020261023，共同保留三个对手，普通95%区间及最大中心化偏差同时95%下界。
- 验证/确认均要求：H对两个参考各四障碍平均≥+3点、Search Only严格提高；其他两个对手分别下降≤3点；无障碍平均下降≤3点；两个地图各自身非法增加≤2点。确认还要求两个同时下界均>0。
- 确认通过才回归：每对手全新100/100场景，最多1800请求。逃逸每地图对两个参考各合法生存率下降≤3点、自身非法增加≤2点。H/O有效逃逸配置完全相同，复用缓存，最多1200唯一对局。
- 回归通过才退出并行池，用独立预热及全新30个四障碍场景、三个对手、H/O/L轮换顺序单进程计时（270局）。报告平均/P95单步、接管决策及整局耗时；耗时增长不直接否决质量改善。

失败立即结束，不追加样本、不更换候选、不延长训练或调整KL/奖励/搜索预算。确认或回归失败保留O。smoke独立场景和低搜索预算，不允许宣称或采用效果；条件分支由受控测试覆盖。

## 产物

experiment/scenarios/diagnostic_bank锁定协议、模型来源、实现与平台指纹及全部场景。prefix_cache/suffix_cache保存完整快照与逐动作轨迹。cache及cache_checksums保存整局科学结果与校验。各阶段manifest区分请求和唯一执行身份。diagnostic_result、validation_result、candidate_lock、confirmation_result、guard_result、timing_result逐门槛提交。

最终REPORT、results、recommended及replay给出全部结论和采用原因。推荐导出包括O/S/L权重、前后阶段配置、私有库和适配器副本。`load_recommended_solver(args, problem, role, seed)`可加载并校验导出；H追捕使用O→S、逃逸完整O，失败导出O。不会覆盖历史模型。

单个固定检查点组合即使有效，也不宣称跨种子稳定或PPO独立贡献。跨前缀差异不归因为单一几何因素，两后缀都失败不证明局面无解。
