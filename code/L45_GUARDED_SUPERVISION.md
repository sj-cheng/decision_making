# 硬 KL 下的单组搜索监督试点

从 `old0:novalue` 的 PPO 第2048局完整状态开始，只训练 S、续训种子0，
新增训练上限512局，每批64局，不追加PPO actor损失、不扩种子。
actor/critic及原Adam状态完整继承；逃逸actor、原价值网络冻结，critic不导出。
本轮检验该固定微调配置的部署收益，不判断新增PPO贡献或硬KL单独的因果贡献。

使用原环境、奖励和原C++求解器。追捕部署1000次/深度100/βπ=.5/βv=0，
逃逸部署βπ=.5/βv=.1。对手固定原纯策略16、原S+L16、原Search Only32局。
每批采样结束后，从首捕前后分层选择最多1024个状态，由本批冻结快照的
真实部署联合best_action生成标签；动作均值损失按动作半范围归一化并屏蔽失活者。
阶段标记与完整状态中的双方活跃标记交叉核对，教师总查询上限8192次。

每个actor minibatch都以整批初始高斯分布为参照计算累计KL。只有KL≤.02才
接受，超限恢复参数与完整Adam状态，步长减半，最多缩步12次；仍失败则回退
并停止本批剩余actor更新，critic完成剩余4 epochs。无标签minibatch跳过actor。
学习率3e-5/3e-4、minibatch256、梯度裁剪.5；critic的γ/λ=1/.95。

每批原子提交完整状态。采样、标签均独立缓存，中断于更新内部则从上一批
检查点和原随机状态重放整批，不把部分更新误计为完成；修改worker数允许恢复。
模型、源码、协议、场景或缓存摘要变化时拒绝混合续跑，历史程序和结果不改。

## 阶段与采用规则

启动前冻结全流程新场景，排除所有可恢复历史l45/PPO场景，包括smoke、归档、
rollout及未使用场景银行；阶段之间按物理初态和速度去重。

- 验证：每对手100四障碍/50无障碍，在新增0/256/512局运行实际MCTS。
  第0局复用当前推荐结果。先排除相对两个参考的非法率/分对手/无障碍回归，
  再按双捕率、较低非法率、较低完成成本选优；失败按80步，完全平分取更早。
- 参考：冻结当前推荐和原l45关闭追捕旧价值；不把不同历史场景的成功率直接比较。
- 仅选定训练检查点对两个参考分别至少+3pp、Search Only子项均提高，且自身
  非法率增加≤2pp、另外两个对手各下降≤3pp、无障碍平均下降≤3pp，才确认。
- 独立确认：锁定唯一候选后，每对手400四障碍/100无障碍，三模型最多4500局。
  两项比较、20000次物理场景配对bootstrap，种子47020261023，三个对手共同保留；
  最大中心化偏差给出同时95%下界。采用要求对两个参考均+3pp且同时下界>0，
  并满足上述限制。不追加样本、不再选检查点。
- 通过后逃逸回归：每对手100/100新场景，完整模型的逃逸角色保持原配置，
  相对两个参考生存率下降各≤3pp、自身非法率增加≤2pp。通过才退出并行池，
  对候选与两个参考用30个新四障碍场景及独立预热交错单进程计时。

任何门槛失败则保留old0:novalue。本轮结束不加入J、不扫奖励或学习率、
不扩大训练或搜索预算；单种子结果不宣称跨种子稳定。

## 命令与输出

在仓库根目录执行：

```bash
PY=/home/sjc/miniconda3/envs/dm_env_py311/bin/python
$PY -m unittest discover -s code/tests -p test_l45_guarded_supervision.py -v
$PY -u code/run_l45_guarded_supervision.py --smoke --output current/l45_guarded_supervision_smoke_20261023 --workers 8
$PY -u code/run_l45_guarded_supervision.py --workers 80
$PY code/run_l45_guarded_supervision.py --stage watch --once
$PY code/run_l45_guarded_supervision.py --stage replay
```

阶段：pipeline/prepare/audit/pilot/confirm/guard/timing/report/watch/replay。
支持--output/--workers/--smoke/--once；smoke必须显式使用独立目录。
smoke仅一个完整64局批次、最多32标签和低搜索预算，不产生科学收益结论。

默认结果目录 `current/l45_guarded_supervision_20261023`。输出REPORT.md、results.json、
recommended.json、recommended/models、锁定选模与两项统计、完整检查点和逐批KL轨迹。
看板核对真实进程和缓存，并显示接受步数、缩步次数、累计KL及actor停止原因。
未执行单进程计时时明确标记未知，不能用并行吞吐替代部署耗时。
