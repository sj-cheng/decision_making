# E45-W：六项实验并行执行协议

用户明确授权尽可能并行启动六项实验、提高CPU/GPU并行度。任务统一登记到current/e45w_joint_ppo_and_planning_20261103；所有工作放在现有tmux:train的独立窗口，不更改旧B–V冻结结果。

## 共用任务和隔离

使用原公共局部观测、团队共享历史、64带权粒子和Single结构化特征；保留原物理、障碍、速度、双方配对退休和80步总时限/原接管剩余时间。真实逃逸对手是fixed_D_kernel与RootPUCT3000，均衡训练和分别报告。无显式目标指派、预设角色或部署专家回退。

训练复用C的480图；新增验证12组，最终测试18组原拓扑新实例+24组未训练布局。新geometry按去平移/90度旋转及1e-4精度与旧B/C/F/G/J/V去重，按几何合法、因果历史和测试队形×速度配额验收，所有拒绝保留，不依据控制器成绩筛图。

AB/BA、动作种子和模型种子在公共历史组内，不当额外独立地图。所有方法固定环境/controller seed公式相同。stochastic评估两个动作种子仅改变Actor采样，环境种子不变。planning/frozen A可以先运行最终测试，但结果不用于PPO配置开发；六个学习模型和选模完成前不查看测试效能、不改科学配方。工程错误修复留痕，不用测试胜负挑配置。

## 实验1／3／4：冻结规划评测

Single PRIOR三个模型17/29/43，GEOM_SEARCH和U各一臂；模型C-x20-round12冻结，不更新。实验1预算50/100/200ms，实验3固定16/64/128完整模拟、不强制200ms返回；各42组×2world×2opponent×5执行臂×3预算=2520局，共5040局。

实验4复用实验1的同局数据，不重复跑：端到端时延为权威，另列初始化/历史吸收和嵌套方法计时；嵌套时间不可相加，原operation_ms是最大回调耗时估计，不冒称累计耗时。

墙钟池独立分配物理CPU0–15，按验证两步前缀串行/并发的CPU-wall、完整模拟吞吐和超时变化选最高合格并发，候选不看捕获结果。固定模拟池先用物理CPU16–27十二路，可在墙钟池空闲后仅改变调度扩展，不改变模拟数/任务。两个评测均预热，保存非法/支持错误，主统计单元不完整则相应确认比较不完整，不静默删除。

每套12项Single−GEOM/U效用对比（2opponent×3budget×2comparator），地图family分层、公共历史成组bootstrap10000，给同时95%区间。ID/OOD与wall/sim分别解释，不跨家族挑显著。

## 实验2：冻结Single直控A

采用相同Torch Actor接口的三个既有Single模型，只采一次原latent Gaussian，经tanh+目标无关安全缩幅，无树搜索。新的A/C step0统一同Torch sampler和动作种子；不能照搬旧NumPy Runtime的12/24数值。

验证每模型96局stochastic为主、48局deterministic为诊断；测试每模型336局stochastic+168局deterministic。所有模式事先固定，不在测试后选择。结果按完整任务与同一物理初始化匹配，旧64次合法拒绝采样只是历史执行接口参照。

## 实验5：PPO接口工程验收

独立Py311 venv复用现有Torch2.10/CUDA环境，固定stable-baselines3==2.7.1、gymnasium==1.2.3。结构化Single Actor与独立随机Critic，后者不继承旧学习叶价值。SB3 PPO.train/collect/GAE使用原库，不能冒称作者RMAPPO。

样本保存未压缩latent z；当前活跃坐标参与Gaussian logp/entropy，退休坐标零并排除密度。环境仅做tanh和安全映射。SB3要求有限Box，因此用±1e6工程guard并在policy/env断言绝不触界，确保库clip为no-op；不对饱和动作反推atanh。

验收包含原Single NumPy/Torch参数及固定噪声分布一致、同策略ratio1/KL0、tanh与latent密度比、退休掩码、B/C初始critic一致、梯度隔离、原物理/信息语义及真实更新/save-load。还须通过8路真实任务smoke再允许实验6。不同PRNG不能声称逐局动作序列完全一致。

首个多进程smoke在step0评估发现RootPUCT pybind只读最佳动作view不能被退休mask改写；保留失败现场，新W环境显式copy后再mask，与原episode raw.copy语义一致，加只读回归测试，在新attempt目录重跑。没有改旧源码，没有因此更换任务或删除科学失败。

## 实验6：B／C六路正式训练

B随机初始化，C加载同seed的Single17/29/43；Actor结构/数据/训练配方相同，Critic按相同seed独立随机初始化，不加载旧叶值。集中式四维团队PPO，公共观察，非去中心化MARL复现。

每模型8个采样环境，256 vector ticks组成2048团队步rollout，976rollouts=1,998,848团队步；六模型合计11,993,088，低于12M上限，禁止自动追加。batch256、10epoch、lr3e-4、clip.2、gamma1、GAE.95、entropy.001、vf.5、grad.5、target-KL.03、normalize_advantage。KL早停使用SB3原逻辑，实际epoch与minibatch Adam step计数，不将rollout、epoch和optimizer step混称。

奖励raw double−.0025每步，加eta.05公共coverage势差、真终点Phi0；无blocked额外罚、无额外捕获进度奖励、无教师动作/持续BC。与旧HASAC同时改变表示/优化/初始化机会，是新baseline开发，不冒称单因素因果对照。

验证每100rollouts=204800新增团队步，另step0与终点。固定96局随机执行原始效用严格更高才替换，平局较早；deterministic48局只诊断。完整保留全部种子、终点与初始，C对A才检验RL增益，C对B主要检验初始化。冻结模型zip含optimizer，另存RNG审计；不含worker物理状态，不能声称精确环境断点恢复。

初期GPU1/2/3各两learner，6模型各8个采样worker（合计48），分配不与wall/sim重叠的24物理core及其超线程；训练/评估source均单数值线程。规划池释放后可按日志扩大现有进程CPU亲和性，不改变n_envs/rollout或训练预算。独立A评测用CPU52–55及其超线程。最终评测在训练与选模冻结后扩大空闲CPU并行。

## 依赖、结果与计费

实验1/2/3可独立并行，实验4附着1；5在工程池验收，6自动等待5通过后启动。算力充足不等于墙钟计时能任意超额并发，因此墙钟性能池有校准与独立CPU分区，训练和固定模拟优先扩展其他CPU。

科学定义在共享experiment.json登记；各组件源码/provider/model/case哈希在正式运行前独立冻结，避免工程未完成就伪称已通过。记录所有工程smoke实际物理步、toy步、优化更新，与正式训练及评估分开。

训练六路结束后统一冻结A、B/C所选与终点15个Actor，进行公共新测试。每Actor随机336+确定168局；A已运行可按actor/spec/源哈希精确复用。报告逐种子、逐对手、ID/OOD、双捕/效用/成本与分组区间，不把7560对局当7560独立地图。若新学习对照更强也完整保留。

一项软件失败先停止该项并保留证据，不把异常记正常输局；独立不依赖任务可继续。主协调器输出各实验真实stage，不将排队或工程验收当作正式训练已经启动。
