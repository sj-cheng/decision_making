# 固定模型的双捕完成搜索效用实验

本轮接续 `current/l45_search_feedback_20261023/NEXT_PLAN.md`，不再追加S/J训练。
只检验本方MCTS内部完成双捕时额外1.0的效用；原真实环境、奖励、三个对手、
1000次模拟和深度100固定。两个固定模型为旧PPO第2048局与原l45，各做开/关。

私有构建不修改历史C++、`.so`或原适配器。生成源码、逐行差异和编译指纹位于
`code/cpp/build_completion/`。完成事件对两个追捕分量分别加1.0，逃逸分量不变；
树边与随机rollout均处理，首捕、非法状态、未完成超时和重复终止均不发放。
计数为新生成树边与rollout中的合法完成事件，不能解释为独立成功样本。

从仓库根目录运行：

```bash
PY=/home/sjc/miniconda3/envs/dm_env_py311/bin/python
$PY code/build_completion_search.py
$PY -m unittest discover -s code/tests -p test_l45_completion_search.py -v
$PY -u code/run_l45_completion_search.py --smoke --workers 8
$PY -u code/run_l45_completion_search.py --workers 80
$PY code/run_l45_completion_search.py --stage watch --once
$PY code/run_l45_completion_search.py --stage replay
```

正式结果目录：`current/l45_completion_search_20261023`。
入口支持pipeline/prepare/audit/screen/confirm/guard/timing/report/watch/replay，
以及--output/--workers/--smoke/--once。改变worker数允许续跑；协议、权重、源码、
场景或缓存摘要变化则拒绝混合。阶段清单与结果原子保存，已完成逐局缓存可复用。

先在独立60个四障碍、30个无障碍物理场景筛选，四配置三个对手共1080局。
新配置相对两个参考分别至少+3pp、Search Only均提高、且回归限制通过才锁定。
锁定后最多400/100新场景、6000局独立确认；未入选不补评。
预登记5项比较，20000次物理场景配对bootstrap，统计种子46020261023。
两候选均确认时取最大中心偏差同时95%下界；仅一个候选入选时用固定5项上限的
Bonferroni单侧99%下界。未评候选不补造结果。

确认通过后检查逃逸角色在原效用下的回归；进程池退出后再做30场景单进程计时。
推荐包含权重、角色参数和新适配器加载入口；仅复制权重不能迁移新搜索效用。
没有新收益则导出R0（old0:novalue），原历史推荐保留。筛选失败即停止，
不追加样本、不扫完成奖励系数、不加搜索预算、不自动进入下一轮网络训练。

输出：REPORT.md、results.json、recommended.json、recommended/models、
screen_result.json及实际执行的后续阶段记录。单进程计时未执行时明确报告未知。

## 已保存批次上的KL工程验收

`learning/kl_guard.py`提供硬接受条件：每个minibatch更新后，相对整批初始
分布计算累计KL；超限就恢复参数和完整Adam状态，以1/2步长重试，最多12次。
没有可接受步则完整回退并停止该批actor更新，critic可在未来调用方继续更新。
历史训练器和检查点保持原样；这不是一次新的成功率实验，也未自动接入旧流水线。

```bash
$PY -m unittest discover -s code/tests -p test_kl_guard.py -v
$PY code/verify_l45_guarded_update.py
```

工程回放只读取S试点第一个真实批次、完整原Adam状态与已有教师标签，在内存中
用相同minibatch顺序比较一轮原更新和受约束更新；不新增环境回合、教师查询或
部署搜索，不导出候选模型。其标签拟合变化不能代替部署成功率确认。
