# 有障碍局部观测：搜索蒸馏与 PPO

执行信息为已知静态地图、本队位置和共享语义 LiDAR；控制器不接收实际全局状态。
64 个联合粒子维护两个匿名目标。策略输入 293 维：原 270 维（历史六维由因果估计重建）、
两个目标的 18 个历史估计量、队友位置及活跃标记 3 维、自身动作上限 2 维。
新输入权重初始化为零，复用 V7 第 45 轮网络。

局部协议按最近角度将目标分到唯一波束，Python/C++ 使用相同归属规则。
这修复了 C++ 浮点角度比较在相邻波束交界处漏检的问题；旧协议保留原判断。

负观测按传感器实际的 64 束角度与障碍回波判断，不能用理想直线可见性替代；
障碍拐角可能出现“直线可见、最近离散波束被挡住”的情况。
边界附近的量化回波先在角度区间内恢复合法位置，必要时投影到已知自由空间；
这些步骤均不读取目标真实坐标。很窄的未观测区域补充已知障碍角点附近的合法先验支持。

新协议 `example8_local_shared_known_map_swept_v1` 使用位移线段碰撞，保持旧协议默认关闭。
每局 manifest 存储真实初始状态、地图及速度参数，**仅用于环境恢复和评测**。
执行策略不能读取 manifest 中隐藏位置。地图生成器在独立种子空间中采样连通的 2—4 障碍地图。

## 构建与验证

使用 Python 3.11 环境（与原 pybind11 扩展匹配）：

```bash
cd /data2/sjc/decision_making
PYTHON=/home/sjc/miniconda3/envs/dm_env_py311/bin/python
"$PYTHON" code/build_local_rl.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
  "$PYTHON" -m unittest discover -s code/tests -p 'test_*.py' -v
```

构建先写入独立目录，再原子替换扩展；首次替换前保留旧二进制。
测试包含隐藏真值扰动不改变执行、目标重现、负观测、粒子恢复、地图缓存、Python/C++ 几何、
特征及高斯分布一致性、真实 PPO 更新，以及从 1 worker 中断后切换 2 workers 的精确续训。

## 一键实验

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
  "$PYTHON" -u code/run_search_rl_pipeline.py --workers 16
```

顺序为测试 → 100 局策略与 100 次搜索测吞吐 → 每阵营 2,000 个联合专家决策点 →
每阵营 5,000 局 PPO → 导出 → 480 局配对试验。
默认搜索总预算 1,000、深度 8；不是每个粒子分别跑 1,000。
`status.json` 显示阶段，`run.log` 保存全过程。恢复使用相同命令；使用新输出目录开始独立实验。

只有 `pilot_gate.json` 的 `continue_formal` 为 true，才自动开始 3 个训练种子的正式训练及消融。
`--pilot-only` 可将实验限制在试验阶段。正式阶段每阵营 50,000 局，包含随机初始化 PPO、
重新训练的历史特征移除消融、不同搜索预算、无障碍退化检查和单独标注的全局状态参考。
这可能需要长时间运行；以 `profile.json` 的实测吞吐估计，不能把功能 smoke 的时间当正式训练时间。

也可单独运行：

```bash
"$PYTHON" code/train_search_rl.py --stage distill --output current/search_rl_v1/manual
"$PYTHON" code/train_search_rl.py --stage ppo --output current/search_rl_v1/manual
"$PYTHON" code/train_search_rl.py --stage export --output current/search_rl_v1/manual
"$PYTHON" code/benchmark_local_obstacles.py \
  --checkpoint current/search_rl_v1/manual/export/policy.pt \
  --reference current/search_rl_v1/manual/distilled.pt \
  --methods distilled ppo search_distilled search_ppo \
  --output current/search_rl_v1/manual/evaluation
```

## 算法与实验边界

搜索使用根粒子采样的开放环动作树，所有粒子共享 Q 统计；模拟内从观测更新历史，
没有按隐藏位置分支的策略。对手使用本地策略模型，不复制真实对手历史或每局隐藏速度。
这是有限深度滚动规划，尚不包含完整 POMCP 的观测分支。

训练使用四个独立 actor、两个集中式 critic。PPO 只使用策略亲自执行的样本；
行为克隆使用同一次搜索的团队联合标签。高斯分布在物理动作空间采样，保存限幅前动作计算概率。
critic 使用原环境奖励的固定团队均值，actor 屏蔽失活机器人。

默认禁用价值截断。可运行 `--stage calibrate-value`：先验证真实 Monte Carlo 回报校准，
再进行配对搜索消融；通过后单独输出 `value_calibrated.pt`。显式使用此模型和
`--beta-value 0.1` 才启用截断；普通模型会拒绝启用。

报告按地图、阵营、对手和训练 seed 分组。`paired_bootstrap.json` 先按训练 seed、
再按地图、再按配对回合重采样；单个训练 seed 不代表训练随机性的置信区间。
`--aggregate` 合并多个训练 seed 的评估目录，同一次合并必须保持搜索预算相同。
补充相同时间预算评估可用 `--time-budget-ms 100 --workers 1`；它至少执行一次模拟，
不保证硬实时截止，实际时间与模拟次数仍记录在结果中。
非法终止、合法生存、捕获和存活数量积分分别记录。

检查点包含模型、critic 归一化、优化器、对手池、计数、随机状态和实现指纹。
默认精确续训要求算法参数和源码指纹不变；worker 数可改变。
`--max-updates N` 在批次检查点边界退出，便于受控中断。
功能 smoke 可缩小 `--decisions`、`--episodes`、`--budget`、`--depth`、
`--validation-episodes`、`--pilot-episodes`；这些输出不能当作正式结果。
修改预算、训练总局数或代码应创建新实验目录，不能覆盖并混合旧结果。

## 2026-10-22 预算与并行检查

当前四个 actor 均为 `293 → 64 → 64 → 64 → 4`，每个 27,396 参数；
两个 critic 均为 `635 → 128 → 128 → 1`，每个 98,049 参数，总计 305,682。
先保留网络大小、64 个粒子、专家搜索 1,000 次/深度 8、搜索对手 200 次。
蒸馏每阵营 2,000 个联合决策点足以做试验，泛化实验可另外比较 5,000—10,000 点；
尚无证据支持直接扩大网络或将搜索提高到数千次。

旧试验蒸馏完成，PPO 每阵营完成 2,208 局后因上述波束盲缝中断。
其纯策略验证曾在 512 局达到追捕双捕获率 63.3%（初始 53.3%），
到 2,000 局回到 53.3%；逃逸合法生存率从初始 10% 到 2,000 局的 5%。
每次每阵营仅 60 局验证，不能据此宣称最终增益或扩大到正式长训练。
完成每阵营 5,000 局试验和独立搜索配对评估后再决定是否进入 50,000 局/3 seeds。

机器有 56 个物理核、112 个逻辑核。早期沙箱检测的 CUDA 不可见不能代表宿主机：
后续在宿主机核实有 4 张 RTX 4090，CUDA 可用，GPU 1/2 基本空闲。
固定 96 个任务、每次搜索
100 次/深度 8，16/32/48 个单线程 worker 的稳态耗时分别为 8.21/5.49/5.39 秒，
各完成相同的 75,957 次模拟转移。推荐 32 workers；48 的额外收益约 2%。
这些是短搜索吞吐，不能作为整局 PPO 的等比例加速保证。

PPO 每批只有 16 局时，提高 worker 数的收益受限。建议新分支使用每批 64 局，
保留 minibatch 256、4 轮 PPO、原学习率和 clip。
这会改变策略更新频率，属于显式的新训练配置，不是仅改变执行速度。
搜索对手约占 10% 回合，耗时远大于普通策略对手，整批仍会等待最慢的搜索回合。

`--fork-from` 将已有 PPO 检查点显式迁移到新目录，保留优化器、归一化、随机状态、
对手池和局数，在 `fork.json` 保存来源哈希及配置变化；旧文件保持不变。
它允许代码修复及采样批量/minibatch 变更，其余训练参数仍检查一致性。
迁移后用当前协议重新验证旧最佳与当前模型，避免直接沿用旧实现的验证分数。
普通续训仍严格检查源码指纹。

以下命令跳过已完成的蒸馏，从旧检查点迁移或继续新分支，完成 PPO、导出和配对试验，
不自动启动正式 3 seeds 长训练：

```bash
cd /data2/sjc/decision_making
nohup env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
  /home/sjc/miniconda3/envs/dm_env_py311/bin/python -u code/train_search_rl.py \
  --stage ppo-pilot \
  --fork-from current/search_rl_v1/pilot_seed0/ppo_latest.pt \
  --output current/search_rl_v2_parallel/pilot_seed0 \
  --workers 32 --episodes-per-update 64 \
  >> current/search_rl_v2_parallel/train.log 2>&1 &
```

此目录已由就绪验证创建。重复运行同一命令会续训，启动前应确认前一个训练进程已经退出。
查看 `train.log`、`pilot_seed0/train_metrics.jsonl` 和最终 `pilot_seed0/pilot_gate.json`。
新增日志分开记录采样时间、更新耗时、每秒局数和搜索对手局数，便于根据实际批次估计剩余时间。

就绪检查已通过 46 项测试，并重跑原失败 seed `2480020934` 完整一局。
新配置实际完成两个 64 局批次，耗时分别 80.0/160.5 秒，网络更新各不到 1 秒；
已在检查点边界退出，当前两个阵营均为 2,272 局。
两批外推剩余 PPO 约 2.85 小时，建议预留 2—4 小时；后续搜索配对评估需另外计时。
完整机器信息、搜索并行对照、测试和实际训练记录见
`current/search_rl_v2_parallel/resource_audit.json`。

## 运行中的终端看板

训练使用 nohup 后，终端 Ctrl+C 退出 `tail -f` 不会终止后台训练。
可在任意终端查看实时进度：

```bash
/home/sjc/miniconda3/envs/dm_env_py311/bin/python \
  /data2/sjc/decision_making/code/watch_search_rl.py \
  --output /data2/sjc/decision_making/current/search_rl_v2_parallel/pilot_seed0
```

看板每两秒刷新，显示实际完成的追捕/逃逸局数、训练进度条、近期批次外推的剩余时间、
训练主进程及 worker CPU 活动、蒸馏基线/最近验证/已保留最佳成功率。
它自动适配小终端，并在 PPO 完成后切换至配对评估进度。
局数只在整批写入后更新；旧训练器未记录批内逐局完成数，看板不会伪造这一进度。
搜索评估的完成数由已保存结果计算，也可能因有序收集而暂时停留在同一数值。
控制台中的活动图标只表示程序存活，CPU 活动表示 worker 在工作，均不等于胜率改善。
纯策略验证大约每 500 局更新，不能把训练进度百分比当作胜率。

看板使用标准库，只读日志和 `/proc`，不加载 checkpoint、不修改训练实现指纹、不重启训练。
`Ctrl+C` 仅关闭看板；`--once` 打印一帧，`--json` 导出一次机器可读快照。

## 利用空闲 CPU 重叠双阵营采样

`train_search_rl_lookahead.py` 沿用原训练器全部参数、PPO 更新、检查点和导出逻辑，
仅将另一阵营的下一批轨迹提前提交给同一 worker 池。
一批约 10% 回合使用搜索对手，其他回合很快完成；提前提交可利用这些空闲 worker。
当前阵营更新时，另一阵营的 actor 保持不变，提前采集的数据仍满足 on-policy 条件。
实际消费前核对种子、配置、本队 actor 和对手 actor 权重；预测不匹配时回退正常采样。
当前更新会在 500 局边界改变对手池时，禁用这一批的提前采集。
提前采集使用独立的对手选择 RNG 副本；只有消费的批次才推进已保存 RNG。
因此在批次之间退出、丢弃尚未消费的预采样后，仍可精确续训。

新的执行入口单独记录实现指纹和 `rollout_backend`，通过 `--fork-from` 写入新目录，
旧检查点保留。基础算法指纹相同的调度迁移复用验证结果，不额外消耗训练 RNG。
单元测试对比完整训练及跨入口迁移/中断恢复，模型、归一化、对手选择 RNG 和 Torch RNG
均需逐位相同。实际搜索预算下的性能及模型一致性记录另见资源加速报告。

```bash
/home/sjc/miniconda3/envs/dm_env_py311/bin/python -u code/train_search_rl_lookahead.py \
  --stage ppo-pilot \
  --fork-from current/search_rl_v2_parallel/pilot_seed0/ppo_latest.pt \
  --output current/search_rl_v3_overlap/pilot_seed0 \
  --workers 32 --episodes-per-update 64
```

这是运行入口示例；若 tmux 中已有同目录训练，应直接查看看板，避免重复启动。
已使用另一阵营预采样的批次中，`rollout_seconds` 记录消费时还需等待的时间；
`lookahead_metrics.jsonl` 另记复用与提前提交的批量数。
这一调度不会将单个 MCTS 的依赖步骤并行化，也不保证填满所有 CPU。
显卡适合后续批量化推理或模拟；当前网络更新每批不到 1 秒，单次 cProfile 中网络前向约占
6.4%，主要时间在 CPU 的粒子更新、观测解码和搜索循环，单独迁移网络不能消除这些耗时。

2026-10-22 的实际检查点对照：固定同一份模型、优化器及 RNG，各采集相同 128 局
（预算 200、深度 8、32 workers、每阵营批量 64）。原调度耗时 186.93 秒，
双阵营重叠耗时 130.69 秒，约 1.43 倍速度、减少 30.1% 耗时。
模型、优化器、归一化、全部随机状态及对局 manifest 均逐位相同。
这是单个固定批次、原训练同时运行时的局部测量，不保证完整训练或后续搜索评估同比加速。
记录在 `current/search_rl_v3_overlap/acceleration_report.json`。

扩展首先提供 CPU 正确性实现，热点几何、传感器、环境转移和特征编码使用 C++。
正式性能结论必须等待完整训练和配对评估；结构测试或少量 PPO 更新不能证明胜率提升。
