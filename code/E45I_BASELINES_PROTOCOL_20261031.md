# E45-I：作者RMAPPO、公共信念APF与九族基准先导

用户于2026-10-31要求立即接公开仓库、写代码并做几个baseline/benchmark。本轮是明确限额的先导，不预设训练收敛或胜率；不修改G/H、原模型及原始结果。

作者MAPPO仓库： https://github.com/marlbenchmark/on-policy 。下载后在experiment.json登记commit与全部上游Python源、许可SHA。actor、critic、Gaussian、RNN与PPO训练器均直接调用上游原代码，不复制自写PPO冒称作者复现。任务观测适配、完整对局采集和变长序列缓冲由本项目实现。

OPEN仓库： https://github.com/thu-uav/Multi-UAV-pursuit-evasion 。完整仿真依赖Isaac Sim 2022.2.0及特定torchrl/tensordict。本轮不将其UAV动力学代码硬接入二维位移任务；APF/MAPPO+APF按论文类别实施本地公开信念适配，不冒称完整DACOOP或OPEN复现。

## 方法和信息

- 主任务沿用20×20、2v2、80步总时域、相同线段碰撞和捕获双方退场。每次接管公共历史剩余时间不同，失败成本记剩余时间。
- 所有部署actor只能读相同公共团队包：地图、团队状态、D的64带权样本、16帧显式历史，并允许RNN记住连续公共观测。actor输入展平后增加ego ID。critic训练时另加真实目标坐标，两个追捕者估计同一个团队value。部署不调用critic。
- RMAPPO：上游共享64宽MLP+GRU，二维Gaussian latent经tanh和公共合法动作映射执行。MAPPO_APF：同一RMAPPO训练器，四维Gaussian调吸引、障碍排斥、机间排斥、切向绕障增益，每个增益为固定系数乘exp(log4*tanh(raw))。log-prob只计算原始Gaussian动作。
- APF是公开信念驱动的自有适配：目标可见用测量，不可见用带权后验均值；按公开期望距离做分配，结合AABB/边界排斥、切向场与机间分离。合力按任务逐坐标速度界归一化，近目标按公开距离缩幅。没有AG动作回退、隐藏目标读取或搜索。固定APF与学习APF共用执行器。
- 三组固定APF系数仅在C的12验证地图、两世界两对手上按原始效用选择，最早平局胜；选择结果单独冻结。两种学习方法共用该固定执行器参考值。
- 其余六族Single PRIOR、AG、IG、Single DIRECT、U、GEOM_SEARCH复用已冻结实现和C20×第12轮Single三种子，不在本轮重新训练。

## 学习先导

RMAPPO和MAPPO_APF各17/29/43三种子，每模型100,000团队环境转移；完整对局可能轻微越过上限，不截断团队信用。训练地图480、验证12，保持原分区。前1/3预算固定D核，之后均衡固定D/RootPUCT3000。训练奖励与G/H一致：原始团队回报加0.05公开几何势差；gamma1、lambda0.95、PPO5epochs、2minibatches、GRU chunk10、学习率3e-4、熵系数0.01线性至0.001。首捕不终止episode；退休actor不参与策略损失。critic无agent ID，退场后仍预测团队价值。无BC或额外专家标签。

RNN训练chunk不跨episode；尾块补零并在策略/价值损失同时mask。每个活跃真实样本每个epoch访问一次；全退休agent的块可排除，环境转移仍完整登记。actor使用公共历史预热；critic从接管时零hidden开始。未修改上游更新规则；本轮关闭ValueNorm/PopArt，避免归一化更新混入尾块补零。短预算结果不能称充分调优强MARL。

初始step0及跨25k/50k/100k的首个更新检查点按验证原始团队效用选模，允许选择初始策略；测试不参与选模。每次保存actor/critic、各自优化器、NumPy/Torch/CUDA RNG；用于评测的检查点不可覆盖，续训使用单独latest。验证缓存含检查点SHA与APF参数身份。

## 先导benchmark

新生成6组ID、8组OOD公共历史，去除C/F/G/H等历史几何重复；合法性、不可区分历史与连通性验收，不根据控制结果筛图。每组两隐藏世界、两对手；六个学习执行臂来自三种子Single PRIOR/Single DIRECT，另六个新RL执行臂，APF/AG/IG/U/GEOM_SEARCH各一臂，共17执行臂，预计952完整对局。

第一阶段先完成固定APF与已有六族的616局对比，输出static_benchmark；六个新RL训练器完成后冻结验证选模结果，再追加336局，合成nine_family_benchmark。两对手均出现在训练中，不能标为对手OOD。独立推断单位仅6/8组公共历史，先报告描述统计和种子明细，不把952局当952独立地图，不据此声称总体优越性。

报告双捕、原始效用、失败惩罚成本、完整在线P95/超时、公开信念独立重放。纯策略无需等待200ms；搜索按共同200ms上限运行，超限实报。并发固定8路评测，与训练阶段分开，不冒称串行实时时延或硬实时保证。

任何未处理异常停止本轮阶段并保留现场，不静默记输。工程smoke只跑三步前缀并明确标为接口验证。完整部署规则、Gaussian概率、RNN序列和退场信用通过单元检查后，才启动正式先导。

## 资源与证据

六训练器各6个不同物理核采样＋1个更新核，使用GPU2/3，保留其他任务；最多占42物理核，静态/最终评测另用8核且与训练分阶段。记录准确CPU绑定、PID、GPU、commit、源SHA、银行与模型SHA。原G/H冻结代码和模型只读复用。本轮属于当前任务的自建统一benchmark，尚未做OPEN或GraphChase公共平台迁移。
