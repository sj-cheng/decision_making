# E45-U：官方 HASAC 原生任务完整训练链校准

本协议只实施流程重审的第一优先级；不续训旧追逃模型，不修改旧实验，不自动启动全观测/隐藏追逃任务。

## 冻结来源与环境

- HARL commit：`b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`，独立逐文件 Git 对象副本，115 个源文件及 SHA256 见 `current/e45u_official_hasac_calibration_20261102/source_manifest.json`。原 checkout 不改动。
- 对应 `tuned_configs/pettingzoo_mpe/simple_spread_v2-continuous/hasac/config.json`；调用作者完整构造、`run()`、`warmup()`、`train()`、`insert()`、官方评估与保存流程。
- 独立 venv 使用 Python 3.10，PettingZoo 1.22.2、SuperSuit 3.7.0、Gymnasium 0.26.3、NumPy 1.23.5、pygame 2.1.2。复用现有 Python 3.10 环境的 PyTorch 2.7.0/CUDA 12.6 等底层库，不修改父环境；完整依赖/安装记录保留。
- 原生任务默认 3 agents、3 landmarks；本地观察18维，集中状态54维，每 agent 五维 [0,1] 连续动作，25步时限。原适配器在26步原生时限前于25步截断并标 `bad_transition`，保持不改。奖励是原生各 agent 奖励之和再复制给每个 agent；评估只计一次团队奖励。
- 冻结作者源码未提供完整 lockfile；这是指定源码/公开配方的原生任务校准，不是逐版本复刻作者机器，不承诺论文曲线数值一致。

## 训练配方和计数

- 正式种子1（公开配置种子）和2；同时两张空闲GPU，各20个训练进程、20个官方评估进程，torch线程4；不为加速而改变并行环境数。
- 每种子完整公开预算：`num_env_steps=10,000,000`，官方 `run` 在热身后执行此预算，因此每种子实际团队转移 **10,010,000**，其中热身10,000；两种子合计20,020,000。
- 20路环境，每50向量tick新增1000团队转移，调用完整 `train()` 50次；每种子500,000 train调用；batch1000、replay1,000,000、n_step20、gamma0.99、自动温度、actor/critic lr均5e-4等全部保留。
- 自动温度实际从log_alpha=0、alpha=1开始，不能把配置里的alpha0.2当作有效初值。
- 每种子训练向量tick500,000+热身500；单agent转移30,030,000。正式每train对应3个Actor Adam、1个合并双Q Adam、3个Actor温度Adam、1个Critic温度Adam，共8个optimizer step；全部实际计数，不能把train调用与optimizer step混称。
- 作者每10000向量tick评估，即每200000训练团队转移一次，含热身的横轴从210000到10010000。每次官方评估40局，参数不变。
- 额外观测计数、固定评估和归档不得改变学习数据、样本抽样、随机数状态或作者更新。`run`/`train`来自冻结源；instrumentation只委托调用。

## 工程校验先于正式训练

原生环境检查后，在同GPU上依次跑未包裹的官方Runner、带记录的Runner；smoke统一只将训练预算改为1000团队步、eval_interval改为50向量tick，以覆盖完整热身、50次更新和一次官方评估。其余训练超参保留。比较初始/终点Actor、双Q及target、全部Adam、温度、Replay、全局RNG摘要；要求两者完全相同，实际单位计数通过。smoke单独计费，不拼入正式曲线。

## 评估、选模与阳性判定

- 额外独立验证：每次官方评估后，在固定原生reset seeds 2000001..2000100上运行确定性Actor，每次100局；step0同样评估。每次额外评估完整保存/恢复Python、NumPy、Torch CPU/CUDA RNG，不触碰训练/官方评估环境。
- 按验证团队回报均值选择checkpoint；严格大于才替换，平局选较早，包括step0。两种子规则一致；保存全部曲线、所有官方评估checkpoint及终点，禁止删失败种子。
- 固定最终测试reset seeds 3000001..3000200，训练和选模结束后一次性比较初始Actor、验证所选Actor、终点Actor；不用于选模/调整超参。每模型200局；同种子配对回报差报告均值和正态近似95%CI，该CI只反映局间变化，不能冒充训练种子总体不确定性。
- 正向证据以两个训练种子各自所选与终点相对step0的测试改进为依据；改进CI下界均为正可作为本轮原生链学习阳性。若仅所选改进而终点漂移，则报告“学到过但稳定性不足”；若证据不一致，不宣告校准通过，先审查原生曲线和配方，不转入追逃任务。
- 本轮目的不是证明已得到本任务强基线，也不要求达到任意80%门槛。标准任务阳性不能替代追逃任务的适配验收。
- 审计保存模型、优化器及RNG；未保存全部worker物理状态/回放的快照明确标注非精确续训检查点，不声称具备精确恢复能力。

## 运行与交付

所有实验置于现有 `tmux:train` 的独立窗口，不占用用户原shell。两种子各日志、实时status/计数、验证JSONL、选择记录、最终封存测试、源码/环境哈希保留。训练未完成时明确报告运行中，不预报阳性。

来源：冻结HARL仓库原README、官方runner和对应tuned config。借用GPU PyTorch与旧依赖的版本差异须披露，工程smoke不能代替学习阳性。
