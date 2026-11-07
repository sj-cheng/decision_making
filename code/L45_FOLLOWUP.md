# l45 PPO 后续闭环

入口 run_l45_ppo_followup.py 使用独立输出 current/l45_ppo_followup_20261023。科学设置和阶段门槛见该目录 PLAN.md。

默认 pipeline 先核对来源与原配置复现，再执行搜索筛选；没有搜索收益时才开始 A/B 奖励试验。所有原环境、训练器和搜索源码只读复用。旧 make_solver 固定参数不修改，新适配器独立传入本方配置，并将对手保持在原配置。

常用命令（仓库根目录）：
    python code/run_l45_ppo_followup.py --workers 80
    python code/run_l45_ppo_followup.py --stage watch
    python code/run_l45_ppo_followup.py --stage watch --once
    python code/run_l45_ppo_followup.py --smoke --output /tmp/l45_followup_check --workers 4

--stage 可独立执行 prepare、audit、search-screen、search-confirm、reward-train、reward-confirm、guard、report。确认需要已有的锁定选择；guard 需要已通过追捕确认的候选。确认数据不参与选择。

cache 以张量内容、场景、实际本方/对手配置、角色、模式和计时上下文为键；别名保留在各阶段 manifest 中，因此模型选优与最终检查点相同时复用计算。每次读缓存都核对身份字段。source、experiment、scenarios 和共享初始权重签名覆盖续跑；原子检查点指针恢复 actor/critic/优化器/RNG/进度/选模状态。

完整报告仅在 pipeline 结束后生成；质量未通过与运行错误分别记录。结果不会自动覆盖原 l45 或启动额外长训练。
