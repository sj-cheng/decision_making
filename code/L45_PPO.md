# 原始 l45 直接追加 PPO

本轮保留原固定四障碍、初始化与速度随机化、奖励、端点碰撞、270 维雷达/历史编码以及原 C++ 搜索。直接加载 `train.py` 迭代训练的四份 l45 策略，不扩维、不重新蒸馏、不加入粒子信念或局部搜索。仅追捕 actor 2/3 做 PPO，逃逸 actor 0/1 与原 `model_value_l45.pt` 冻结；新增集中式 critic 只辅助训练，不导入搜索。

正式配置为 PPO 种子 0/1/2，每个 2,048 局、每批 64 局，每 256 局验证选优。训练对手各占一半：原 l45 纯策略、原 l45 引导搜索。搜索保持 1,000 次模拟、深度 100、βπ=0.5、βv=0.1；Search Only 评测对手为 3,000 次、深度 100。三个 PPO 种子共享同一预训练 l45，不是三个独立预训练种子。

## 启动、续训与看板

从仓库根目录执行，先确保测试与独立 smoke 已通过。正式 pipeline 完成正确性检查、单进程测速、训练、验证、确认及报告；不会自动启动 50,000 局，也不会恢复旧局部观测实验。

```bash
cd /data2/sjc/decision_making
/home/sjc/miniconda3/envs/dm_env_py311/bin/python -u code/train_l45_ppo.py --stage pipeline --workers 32
```

长任务应放在独立 tmux 窗口中，并将 stdout/stderr 保存到输出目录 `run.log`。默认输出为 `current/l45_ppo_control_20261022`。中断后运行**同一命令和配置**即可续训；检查点含优化器、随机状态和计数。模型、源码、配置或场景指纹变化会拒绝混合续训，应先检查变化原因，不能删除指纹绕过检查。`--stage train` 只训练，`--stage evaluate` 使用已选定模型完成确认，`--stage report` 汇总现有结果。

```bash
/home/sjc/miniconda3/envs/dm_env_py311/bin/python code/watch_l45_ppo.py
/home/sjc/miniconda3/envs/dm_env_py311/bin/python code/watch_l45_ppo.py --once
```

看板只读 JSON 和 `/proc`，不依赖 Torch。它核对真实 Python runner 及其输出目录；旧 `status.json` 不会被当成正在运行。ETA 仅估计剩余训练采样与更新，**不含验证和后续完整搜索评测**。关闭看板不会关闭训练。

## 独立功能检查

```bash
/home/sjc/miniconda3/envs/dm_env_py311/bin/python -u code/train_l45_ppo.py --smoke --output /tmp/l45_ppo_smoke --workers 2
```

smoke 仅验证完整流程，主动缩小模拟次数和样本量，仍保留原搜索深度；必须使用独立输出，不能将其结果当正式结论。正式 `--stage check` 会校验零更新动作、原 C++ 搜索加载一致性及冻结模型。进一步的单元测试覆盖 PPO 更新、概率记录和续训等条件。

## 结果与原基准兼容

输出包括 `experiment.json`（配置、源码和模型指纹）、`scenarios.json`（固定配对场景）、各 seed 的进度/更新/检查点，以及最终 `REPORT.md` 和 `results.json`。最佳模型仅通过验证集选择，最终检查点另行保留；确认集不能用于挑选模型。

`seed0/best_models`、`seed1/best_models`、`seed2/best_models` 及相应 `final_models` 保留原 `model_policy_l45_i{0..3}.pt`、`model_value_l45.pt` 文件名和 state_dict 格式。原基准可通过 `--model-dir <该目录> --model-iteration 45` 加载；比较时还须保持 `--beta-policy 0.5 --beta-value 0.1` 和原障碍、预算等协议，不应使用基准脚本其他默认值替代本实验配置。

主结论比较同条件下合法双捕率，同时检查非法终止、首捕后超时、无障碍和逃逸回归。网络直接执行与网络引导搜索分别报告。相同搜索预算下不预先宣称单步更快；单进程决策耗时与并发吞吐分别记录。原模型和旧实验保留，不覆盖。

## 本轮评测并行调整

2026-10-22 三个种子训练完成后，按用户要求将剩余评测从 32 进程提高到 48 进程。同一组 96 个任务耗时从 44.56 秒降至 35.86 秒，非计时结果完全一致；详见输出目录的 `parallel_probe.json` 和 `parallel_adjustment.json`。

随后按用户要求补测 80 进程：同一组 96 个任务耗时 34.51 秒，相对 48 进程的 35.86 秒，本次实测约快 4%，任务结果仍完全一致。单次短测不能保证持续获得相同增益，详见 `parallel_probe_80.json`；当前使用 80 进程续跑。

当前续跑命令为下列命令，直接读取已有缓存和已选定模型，不重新训练。`workers` 不属于训练/场景指纹，但其他参数仍必须一致。并发计时会受到进程数影响，部署耗时继续只看独立单进程评测。

```bash
/home/sjc/miniconda3/envs/dm_env_py311/bin/python -u code/train_l45_ppo.py --stage evaluate --workers 80
```
