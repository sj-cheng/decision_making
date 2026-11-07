# 有障碍物 baseline benchmark

入口为 `code/run_benchmark_obstacles.sh`，复用 `code/benchmark_baselines.py` 的逐局指标、阵营耗时、分块并行和断点续跑。默认使用两个固定矩形障碍物，结果写入 `current/benchmarks/obstacles_two_v1/`。

## 正式运行

```bash
cd /data2/sjc/decision_making
bash code/run_benchmark_obstacles.sh --workers 80
```

当前机器有 112 个逻辑 CPU，可增加到 96 workers；每个 worker 的 PyTorch、OpenMP、MKL 和 OpenBLAS 线程数均为 1，MAPPO 使用 CPU。

```bash
bash code/run_benchmark_obstacles.sh --workers 96
```

默认自动使用 `$HOME/miniconda3/envs/dm_env_py311/bin/python`；可通过 `PYTHON=/path/to/python` 覆盖。脚本会切到自己的目录，可从任意目录启动。

后台运行并保存日志：

```bash
mkdir -p current/benchmarks
nohup bash code/run_benchmark_obstacles.sh --workers 96 \
  > current/benchmarks/obstacles_two_v1.log 2>&1 &
```

终端会报告实际启用的 worker 数、场景协议、对局进度和已完成的 chunks。每 10 局一个 chunk，默认共 1,780 个任务；实际 worker 数受任务数和逻辑 CPU 数限制。中断后重新执行相同命令即可续跑，允许调整 `--workers`。改变障碍地图、checkpoint、模型目录、预算或 chunk 大小时，需改用新的 `--output`。

## 协议与模型

五种方法进行完整 5 × 5 阵营对战：Policy Only、Search Only、Search + Learning、MAPPO、Large-budget MCTS。普通对局 5 seeds × 200 局；涉及 Large-budget MCTS 的对局 1 seed × 200 局，总计 **17,800 局**。

| 项目 | 默认值 |
| --- | --- |
| Search Only 搜索预算 | 3,000 |
| Search + Learning 搜索预算 | 1,000 |
| Large-budget MCTS 搜索预算 | 50,000 |
| β policy / β value | 0.5 / 0.1，与 V7 一致 |
| Policy / value 模型 | `current/memory_observation_v3/models`，iteration 45 |
| MAPPO 模型 | `current/baselines/mappo_50k_v4.pt`，与更新后的无障碍表一致 |
| 默认障碍 1 | x ∈ [-4, -2]，y ∈ [-3.5, -1.5] |
| 默认障碍 2 | x ∈ [2, 4]，y ∈ [1.5, 3.5] |
| 动力学 | 关闭 MINCO，沿用 direct propagation |
| 出生方式 | 沿用当前 Example8；不启用障碍附近出生，初始状态满足障碍间距检查 |

每种地图先统一生成初始状态，再共享给所有方法；同一个 seed/episode 在普通对局和 Large-budget 对局间使用相同初始状态。父进程生成初始状态和子进程执行对局都显式配置同一张地图，C++ 搜索也接收这些障碍物。

这里延续既有全局状态搜索评测入口：search 使用完整 simulator state 和障碍地图，value 使用物理状态编码。Policy Only / Search + Learning 的策略先验仍使用原有语义 LiDAR 和记忆，加入障碍后会发生遮挡；MAPPO 保持 checkpoint 的 ego-relative 输入和动作解码器，其 actor 输入没有新增障碍几何。这是现有 checkpoint 在障碍地图上的迁移评测，没有重新训练模型，不能直接解读为全部方法都经过有障碍训练的比较。历史 NTE 结果不混入新地图结果。

障碍碰撞沿用 Example8 的离散步末端判定：位置进入矩形（含边界）即为非法终止；本实验没有改为连续轨迹碰撞检测。碰撞和越界分别记录，并区分追捕方、逃逸方，不将逃逸方撞障碍算作捕获。

## 快速验证

以下运行完整 25 组对战，每组 2 局，使用小搜索预算验证模型加载和多进程，不作为正式结果：

```bash
bash code/run_benchmark_obstacles.sh \
  --workers 4 --seeds 1 --large-seeds 1 --episodes 2 --chunk-episodes 1 \
  --search-only-budget 20 --search-learning-budget 10 --large-budget 30 \
  --output /tmp/benchmark_obstacles_smoke --no-progress
```

验证障碍初始化、碰撞阵营归因、Python/C++ 编码与碰撞奖励一致性、续跑地图隔离：

```bash
CUDA_VISIBLE_DEVICES='' MPLCONFIGDIR=/tmp/mpl \
  /home/sjc/miniconda3/envs/dm_env_py311/bin/python \
  -m unittest discover -s code/tests -p test_benchmark_obstacles.py -v
```

## 四障碍地图

```bash
OBSTACLE_LAYOUT=four bash code/run_benchmark_obstacles.sh --workers 96
```

四障碍在默认两个障碍的基础上增加左上和右下两个对称矩形，结果自动写入 `current/benchmarks/obstacles_four_v1/`。直接调用 Python 时可用 `--obstacle-layout none|two|four`；原 Python 入口默认仍为 `none`。

## 输出

- `config.json`：协议名称、障碍物坐标、模型、预算、seed、并行参数及观测约定。
- `chunks/*.csv`：原子写入的已完成分块，供断点续跑。
- `pursuer-*_evader-*.csv`：按 seed/episode 排序的逐局结果。
- `pursuer-*_evader-*_summary.csv`：每组对战的均值、标准差、95% 区间半宽和样本数。

除既有双捕获率、对局长度、回报与各阵营耗时外，新增 `obstacle_collision`、`out_of_bounds` 及其 `pursuer_*`、`evader_*` 指标。各指标是该局是否发生的 0/1 标记；summary 中均值就是发生率。大量 worker 下的决策耗时反映并发争用，单独比较推理耗时时应另开输出目录，以 `--workers 1` 运行所需对战。
