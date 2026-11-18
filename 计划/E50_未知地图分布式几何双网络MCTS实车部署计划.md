# E50：未知地图、信念与双网络 MCTS 的 2v2 实车部署计划

状态：2026-11-10，路线已调整为“先做已有强方法的实车适配审计”。暂不新增 AMBUSH 相似的双网络方法，不改变已冻结的 E46–E49 结果。

## 目标与固定边界

最终目标是在真实机器人上运行 2v2 追逃：SLAM 提供共享位姿和增量地图，LIDAR 提供局部障碍观测，摄像头/LIDAR 在目标可见时提供坐标，D 估计器在目标丢失时维护信念。规划采用集中联合搜索，执行采用两车各自的轨迹跟踪与安全层；不预设 attacker/hider 角色。

当前阶段不把 NAR/NAE 作为新增主方法。先测 E46–E49 已有强方法在 LIDAR 射线观测和 MINCO/多项式执行下的闭环表现，再根据保留下来的方法决定是否需要新增搜索或网络结构。

保持两条仿真线：现有已知地图、1 秒直接位移基线；面向实车的未知地图线，仍以 1 秒重规划，但在每个决策周期内以 50–100 Hz 执行多项式轨迹。

## 实施阶段

1. **固定比较矩阵**：冻结并复用 E46 的 A2/A3/A4/A5/AG、E47 的 L5/同池方法、E48 的密集障碍方法和 E49 的完整 2v2 方法；不重训、不新增网络。
2. **LIDAR 射线观测**：实车直接复用现有 SLAM/定位；仿真只实现车体中心向外发射固定角度射线，返回最近障碍距离，并加入量程、噪声、漏检和延迟。控制器不能读取完整障碍列表。
3. **MINCO 复演**：复用 `Example8` 已有 MINCO/闭环多项式实现，把每个方法输出的 1 秒动作送入同一轨迹后端；同时保留直接位移版本，隔离“方法能力”和“执行器变化”。
4. **未知地图适配**：先做局部射线障碍代理和安全检查，再决定哪些方法需要局部地图记忆、哪些方法只能在可见障碍范围内运行。未知区域不得被默认为自由空间。
5. **分布执行审计**：中心保持联合决策，两车分别执行轨迹；加入通信延迟、丢包、位姿延迟和单车停车降级，记录每个方法的安全和协同退化。
6. **方法筛选**：只有在未知观测、MINCO执行和分布式扰动下仍稳定的方法，才进入后续 NAR/NAE 或其他新方法研究。

## 首轮方法优先级

- 第一批：A2、A3/L5、A4、AG、GEOM_SEARCH、RouteAPF。
- 第二批：IG、SINGLE_PRIOR、A5、MAPPO+RouteAPF、PPO-C。
- 不把固定任务分配、全局隐藏目标坐标或已知全局地图写入适配接口。
- 首轮不比较新增 NAR/NAE；先找出在真实部署约束下仍然有效的现有方法。

## 验收与实验

先通过已知地图回归、射线观测合法性、MINCO 轨迹合法性、Python/C++ 转移一致性、未知区域保守处理和通信降级，再进入首轮方法比较。

必须比较原始 E46–E49 方法在已知地图/射线观测、直接位移/MINCO、无扰动/通信扰动下的双捕率、二捕完成率、重新发现时间、轨迹平滑度、碰撞/急停率、规划延迟和 MINCO 耗时。把“方法失败”“观测不足”“轨迹不可执行”“通信降级”分开记账。

## 当前第一批实现

已新增 `code/solvers/e50_interfaces.py`、`e50_candidate_adapter.py`、`e50_lidar_map.py`、`e50_local_obstacle.py`、`e50_endpoint_trajectory.py`、`e50_minco_audit.py`、`e50_minco_transition.py`、`e50_minco_corrected.py`、`e50_minco_math_audit.py`、`e50_compat_smoke.py` 和 `run_e50_compat_one_step.py`：提供公开 LIDAR 射线、信念/轨迹数据结构、保守局部射线安全判断、复用 E46 `candidate_pool` 的兼容桥接、终点语义多项式轨迹、Example8 MINCO 轨迹审计、同步多车 MINCO 推进、E50 隔离的转置修正版 MINCO，以及直接位移/MINCO 成对 smoke 评测。已对 A2、A3、A4、AG、GEOM_SEARCH、RouteAPF 完成一张 E49 开发图的一步执行诊断；它们都能产生合法动作，但 MINCO 端点与原位移目标有明显偏差。审计还确认 Example8 自由终端条件存在 `M^-1`/`M^-T` 转置错误；E50 已隔离修正版，修正只带来约0.02m变化，主问题仍是默认闭环响应过慢。终点语义 smoke 进一步显示，约1米的一秒动作在当前加速度上限下通常不可行，不能静默裁剪后继续比较；同一诊断动作在3秒终点窗口下才通过当前动态约束，后续必须决定采用更长的重规划轨迹还是改变动作语义。以上不改变现有控制器，也不新增 AMBUSH 式搜索器。

## 2026-11-10 实施进度

- 已用 `PYTHONPATH=code /home/sjc/miniconda3/envs/dm_env_py311/bin/python -m unittest discover -s code/tests -p 'test_e50*.py'` 通过 35 项 E50 单元验收。
- 在 `current/e50_deployment_audit_20261110/compat_long_wall_0001.json` 对首批六种方法完成同一 E49 验证场景的端点时域审计：六种方法的 1 秒端点均不可行，2 秒和 3 秒均可行；没有碰撞，MINCO 轨迹的最大速度约 0.51–0.55 m/s、最大加速度约 0.70–0.75 m/s²。
- 在 `current/e50_deployment_audit_20261110/rolling_long_wall_0001.json` 完成 2 秒规划、每秒执行并重规划的两周期滚动 smoke：A2、A3、A4、AG、GEOM_SEARCH、RouteAPF 全部两周期可行、无碰撞、两车局部 LIDAR 安全检查均通过。该 smoke 仍是已知地图和冻结逃跑者诊断，不作为未知地图捕获结果。
- 在 `current/e50_deployment_audit_20261110/dynamic_limit_sweep.json` 完成速度/加速度参数网格审计；在本场景下 2 秒端点是可行性分界，结果仅用于仿真筛选，实车参数必须重新标定。

当前决定：首轮执行语义采用 **2 秒终点轨迹、1 秒滚动重规划**；保留 1 秒直接位移作为能力对照，不把不可行的 1 秒 MINCO 轨迹静默裁剪后纳入比较。下一步进入未知地图适配：把控制器输入替换为局部射线/扫描记忆，加入通信延迟、丢包、位姿延迟和单车停车降级，再在同一首批方法矩阵上跑无扰动与扰动对照。

本轮继续完成：新增 `code/solvers/e50_distributed_audit.py` 和 `code/run_e50_distributed_smoke.py`。分布式审计明确记录每个机器人的命令延迟、丢包、位姿延迟和 `stop_degraded` 事件；在首个滚动 smoke 上生成 `current/e50_deployment_audit_20261110/distributed_smoke.json`。名义条件四条命令全部送达；单周期延迟或丢包均产生两车停车降级事件，未复用陈旧命令。E50 单元验收现为 38 项通过。该结果仍是执行边界诊断，尚未宣称未知地图闭环性能。

随后新增 `--unknown-map` 严格 smoke 路径：控制器拿到空障碍地图，真实障碍只用于独立 LiDAR 射线生成和最终碰撞审计。产物为 `current/e50_deployment_audit_20261110/rolling_unknown_map_long_wall_0001.json`。六种方法两周期的实际执行前 1 秒均无碰撞且 MINCO 可行；GEOM_SEARCH 的第二周期 2 秒计划后半段存在未来碰撞，但该段在 1 秒滚动重规划前不会执行。smoke 已将 `collision` 与 `planned_trajectory_collision` 分开记录。下一步必须把完整计划碰撞作为执行前安全门，或证明通信延迟下不会继续执行该后半段；当前仍不能宣称未知地图闭环验收完成。该路径仍冻结逃跑者，尚未构成捕获率实验。

已加入完整计划碰撞安全门：若 2 秒轨迹任意后半段进入障碍，当前周期不执行原动作，改为停车并记录 `safety_override=stop_future_collision`，等待下一周期重新规划。该门在 GEOM_SEARCH 案例触发一次，避免了未来段碰撞被通信或执行器继续放大；这只是保守安全机制，仍需在多场景和通信扰动闭环中验证其代价。

已扩展到 3 张 E49 验证图、每图 4 个滚动周期，产物为 `current/e50_deployment_audit_20261110/unknown_map_batch.json`。六种方法共 72 个周期均未出现实际执行段碰撞或动力学不可行；但多种方法频繁触发未来轨迹碰撞安全门（A2/A3/A4/AG/RouteAPF 在部分场景各触发 1–2 次，GEOM_SEARCH 触发 0–1 次）。这说明当前未知地图路径在保守安全门下可以避免立即碰撞，但代价是停车降级，尚未达到“稳定协同执行”的方法筛选标准；下一步需要把这些降级事件与通信扰动结合并统计任务性能。

通信扰动已接入真实滚动状态推进，而不再只是离线命令重放：`run_e50_rolling_smoke.py` 支持命令延迟和丢包，实际状态由送达动作推进。3 张地图、每图 4 周期、1 周期命令延迟并在第 1 周期丢包的结果位于 `current/e50_deployment_audit_20261110/unknown_map_delay_drop_batch.json`；六种方法均无执行碰撞和动力学不可行，但这是因为首周期停车、丢包周期停车，尚未证明任务协同能力。位姿延迟仍需接入控制器时间序列，双捕获/重新发现等任务指标也尚未完成。

位姿延迟已接入滚动控制：控制器读取延迟一周期的历史位置，执行仍从真实位置推进。3 张地图、每图 4 周期、命令延迟 1、丢包 1、位姿延迟 1 的结果位于 `current/e50_deployment_audit_20261110/unknown_map_delay_drop_pose_batch.json`。当前六种方法均无执行碰撞和动力学不可行，但短周期实验中动作大量退化为停车/延迟送达，尚未形成捕获或重新发现指标；需要继续扩展到完整任务时域并补齐方法级性能记账。

已扩展到每图 12 个滚动周期，并记录规划耗时、MINCO 峰值速度/加速度、执行距离、通信停车和安全门触发。结果位于 `current/e50_deployment_audit_20261110/unknown_map_long_horizon_12cycles.json`：六种方法实际执行段碰撞均为 0，但 3 张图合计安全门触发 8–21 次、通信停车各 3 次；A3/GEOM_SEARCH 执行距离较高，A2/A4/AG/RouteAPF 因保守停车明显退化。该结果已经能区分“安全通过”和“有效协同”，但仍是冻结逃跑者的执行诊断，尚未提供双捕获率或重新发现时间。

已加入确定性合法移动逃跑者并延长到每图 20 周期，生成 `current/e50_deployment_audit_20261110/unknown_map_moving_evader_20cycles.json`。该诊断链路已能记录捕获事件、执行距离、规划耗时和安全降级；三张图、六种方法在命令/位姿扰动下均无执行碰撞，但 120 个方法-场景组合没有完成捕获，安全门触发 21–45 次，说明当前保守停车策略仍不足以支持追捕任务。该逃跑者是工程诊断策略，不替代正式 E49 RootPUCT 对手；正式双捕获率仍需接入冻结对手和完整 80 步任务循环。

已完成单场景完整 80 步诊断，产物为 `current/e50_deployment_audit_20261110/unknown_map_moving_evader_80cycles.json`。六种方法均无实际执行碰撞，但捕获数仍为 0；安全门触发 A2/A4=71、A3=67、AG=73、GEOM_SEARCH=37、RouteAPF=73 次，执行距离分别约 7.7、16.1、7.7、3.6、47.2、2.5 m。结果确认当前瓶颈是未知地图下的安全门导致持续停车，而不是动力学不可行；在接入正式 RootPUCT 对手前，需要先降低无效安全门或改进局部地图记忆，否则双捕获率不会有意义。

已修正未知地图安全门的信息边界：`--unknown-map` 模式不再用真实障碍地图的完整轨迹碰撞参与决策，真实地图只保留为事后审计；执行前安全门只检查当前 LiDAR 射线代理覆盖的整条候选轨迹。20 周期复核产物为 `current/e50_deployment_audit_20261110/unknown_map_lidar_gate_20cycles.json`：实际执行段仍无碰撞，但 A2/A3/A4/AG/RouteAPF 各触发约 15 次 LiDAR 安全门，GEOM_SEARCH 触发 7 次，说明当前局部射线代理本身过于保守，下一步需要引入扫描记忆/局部地图膨胀校准，不能直接进入正式捕获率比较。

已增加 LiDAR 邻束范围插值模式，并保留原邻束最小值模式作为回归基线；39 项 E50 单测通过。插值模式 20 周期结果位于 `current/e50_deployment_audit_20261110/unknown_map_lidar_interp_20cycles.json`：无执行碰撞，GEOM_SEARCH 仍为 7 次安全门，A3 降至 13 次，其余方法约 15 次，说明单纯插值只能小幅缓解过保守问题。下一步需要接入真正的滚动扫描记忆和局部占据更新，再决定是否调整膨胀参数。

已接入 `LocalScanMemory` 的滚动命中点记忆，并增加轨迹级记忆安全查询；40 项 E50 单测通过。20 周期结果位于 `current/e50_deployment_audit_20261110/unknown_map_scan_memory_20cycles.json`：执行碰撞仍为 0，GEOM_SEARCH 安全门降至 6 次，其余方法仍约 15 次，说明当前命中点距离规则仍偏保守，记忆层已接通但尚未解决主要停车退化。下一步应基于时间戳和射线方向做局部占据栅格，而不是继续扩大点距离阈值。

已将扫描记忆升级为带自由射线和占据命中格子的局部占据栅格，并加入方向性轨迹检查；41 项 E50 单测通过。20 周期结果位于 `current/e50_deployment_audit_20261110/unknown_map_grid_20cycles.json`：执行碰撞仍为 0，但 A2/A3/A4/AG/RouteAPF 仍约 15–16 次安全门、GEOM_SEARCH 6 次，说明主要过保守来源在当前帧射线代理而非历史点记忆。下一步需要校准机器人膨胀半径、射线采样和轨迹安全判据，之后再进入正式对手任务评测。

已完成 3 种射线数（32/64/128）× 3 种膨胀半径（0.02/0.05/0.10）的 12 周期标定，产物为 `current/e50_deployment_audit_20261110/lidar_safety_sweep.json`。本场景中 9 组参数对每个方法的安全门次数完全一致，说明停车退化不是简单射线分辨率或膨胀半径造成，而是候选轨迹与局部射线几何的系统性不匹配。暂时固定 64 射线、0.05 膨胀作为审计基线，下一步转向让候选生成器直接使用局部占据/射线可达性，而不是先生成全局目标轨迹再由安全层大量拒绝。

已加入 `project_action_to_local_scan`：当未知地图候选被 LiDAR 安全门拒绝时，在当前可见方向上生成有限半径和轴向逃逸候选，选择最接近原动作的局部安全动作，再重新进行 MINCO 审计；无候选时才停车。42 项 E50 单测通过。单场景 12 周期结果位于 `current/e50_deployment_audit_20261110/unknown_map_projection_12cycles.json`：六种方法实际执行碰撞均为 0，原先的停车被 3–8 次 `local_action_projection` 替代，执行距离提高到约 5.2–7.7 m。该结果说明局部候选投影能缓解安全门退化，但仍需多场景长时域和正式对手验证。

已扩展局部候选投影到 3 张地图、每图 20 周期、移动逃跑者、命令延迟/丢包/位姿延迟，产物为 `current/e50_deployment_audit_20261110/unknown_map_projection_batch_20cycles.json`。六种方法实际执行碰撞和捕获均为 0；LiDAR 安全拒绝全部通过局部投影化解（A2/A3/A4/AG/GEOM_SEARCH/RouteAPF 分别 47/45/47/48/33/48 次投影），没有未来碰撞停车；剩余停车只来自通信丢包（各方法 3 或 6 次）。执行距离提高到约 20.5–39.4 m，说明局部投影已消除主要的安全门停车退化，但正式 RootPUCT 对手和双捕获率仍未接入。

已接入冻结 E49 RootPUCT 逃跑者接口，完成 4 周期和 12 周期联调；12 周期产物为 `current/e50_deployment_audit_20261110/rootpuct_unknown_12cycles.json`。RootPUCT 搜索、未知地图控制器、局部动作投影、通信扰动和捕获事件记录链路均正常，六种方法均无执行碰撞，但 12 周期内尚无捕获；每种方法仅有 1 次通信停车，局部投影 5–9 次。当前 RootPUCT 使用 20 次搜索作为接口联调预算，尚不能作为正式 E49 强对手统计；下一步扩展到完整 80 步并提高搜索预算，同时记录双捕获和重新发现时间。

80 步 RootPUCT 联调首次暴露局部投影只检查终点、长轨迹仍可能碰撞的问题，已修正为检查完整 2 秒 MINCO 采样轨迹的 LiDAR 射线和占据栅格；同时 RootPUCT 偶发失败改为显式计数并使用零逃跑动作降级，不能静默当作成功搜索。修正后的完整 80 步产物为 `current/e50_deployment_audit_20261110/rootpuct_unknown_80cycles_budget50_safe.json`：六种方法均跑满 80 步，实际执行碰撞和捕获均为 0，RootPUCT 搜索失败计数为 0；局部投影 1–7 次，通信停车 1–2 次。该结果证明长轨迹安全门已稳定，但捕获率仍为 0，尚未达到正式方法筛选标准。

为诊断零捕获原因，已在每周期记录最小追捕距离、RootPUCT 实际逃跑位移和活动状态。12 周期扰动结果 `current/e50_deployment_audit_20261110/rootpuct_distance_12cycles.json` 显示最小距离约 2.55–3.58 m 后重新拉开；20 周期无扰动结果 `current/e50_deployment_audit_20261110/rootpuct_unknown_nominal_20cycles.json` 仍无捕获，最小距离约 2.78–4.05 m。当前场景中逃跑者速度上限 1.5、追捕者执行速度约 1.0，且初始间距较大，零捕获有明确动力学原因，不能归因于捕获判定 bug。后续正式比较需把双捕获率与“未捕获/速度差不足”分开记账，并补充同一方法在可捕获初态和正式 E49 对手预算下的对照。

捕获事件时序已修正为每周期先检查执行后位置，再推进逃跑者；这样不会因逃跑动作先行而漏记周期起点已进入捕获半径的事件。现有 E50 单测 42 项仍全部通过。可捕获初态的正式构造还需要同步重建合法的 E49 public history 和目标可见性记录，避免把初始化失败误记为方法失败。

已新增 `run_e50_method_accounting.py`，对 80 步 RootPUCT 结果生成分项记账 `current/e50_deployment_audit_20261110/rootpuct_unknown_80cycles_budget50_safe_accounting.json`，分别统计决策数、捕获、执行碰撞、轨迹不可行、通信停车/丢包、局部投影和 RootPUCT 搜索失败。当前 80 步样本六种方法均为 80 次有效决策、0 捕获、0 执行碰撞、1–2 次通信停车、1–7 次局部投影；这份记账不再把零捕获与安全或搜索失败混为一类。

第二批方法已接入同一未知地图滚动链路：IG、SINGLE_PRIOR、A5、MAPPO_RouteAPF、PPO-C 在 4 周期和 12 周期审计中均能初始化并完成执行。12 周期产物为 `current/e50_deployment_audit_20261110/second_batch_unknown_12cycles.json`：五种方法执行碰撞和捕获均为 0，通信停车各 1 次，局部投影 0–2 次，执行距离约 3.5–7.8 m。它们尚未完成 80 步 RootPUCT 正式对照，暂不能纳入最终方法筛选。

第二批五种方法已完成 80 步 RootPUCT 全时域接口对照，产物为 `current/e50_deployment_audit_20261110/second_batch_rootpuct_unknown_80cycles_budget20.json`。IG、SINGLE_PRIOR、A5、MAPPO_RouteAPF、PPO-C 均完成 80 次有效决策，RootPUCT 搜索失败和执行碰撞均为 0，捕获均为 0；局部投影次数为 1/8/7/1/9，通信停车各 1 次，执行距离约 3.7–45.8 m。该结果完成第二批的适配审计，但仍是单场景、20 次 RootPUCT 搜索预算，不能替代正式多场景统计。

已完成 11 种方法 × 3 张验证地图 × 每图 12 周期的多场景未知地图统计，产物为 `current/e50_deployment_audit_20261110/all_methods_unknown_3maps_12cycles.json`。33 个方法-场景组合全部有效，执行碰撞和捕获均为 0；每个方法通信停车均为 3 次（每图一次丢包/延迟起始退化），局部投影累计 1–5 次，执行距离累计约 8.1–19.7 m。该结果证明 11 种方法都能通过当前未知观测/MINCO/分布执行接口，但由于场景仍以高速度逃跑者和短时域诊断为主，尚不能据此宣称正式双捕获性能。

由于 E49 验证库初态最近追捕距离均约 4.7 m 以上，当前没有天然可捕获初态；已单独完成捕获指标层 smoke `current/e50_deployment_audit_20261110/capture_gate_smoke.json`，覆盖双捕获、单捕获和遮挡线段不捕获三种情况，确认捕获配对、遮挡判定和双捕获标志正确。该 smoke 不计入方法结果，正式可捕获初态仍需重建合法 public history 后再做。

已完成一张“可捕获诊断初态”副本的 4 周期方法链验证，产物为 `current/e50_deployment_audit_20261110/capturable_diagnostic_4cycles.json`。只改变诊断副本的初始追捕者位置，正式 E49 地图、控制器和原始 history 文件不改；A2/A4 在首周期捕获 1 个目标，A3/AG/GEOM_SEARCH/RouteAPF 在首周期完成双捕获，证明 MINCO 执行、捕获时序和双捕获记账链路可工作。由于该副本的当前状态与原始 history 不同，结果仅作链路验收，不能作为方法性能结果；下一步需生成状态与 history 完全一致的正式可捕获任务包。

已按 E49 `reports=[target][observer]` 语义重建状态与 history 一致的可捕获诊断包，产物为 `current/e50_deployment_audit_20261110/capturable_consistent_4cycles.json`。首批六种方法均在首周期完成双捕获；第二批五种方法产物为 `current/e50_deployment_audit_20261110/capturable_consistent_second_batch_4cycles.json`，也均完成双捕获。该包保留正式地图和 public history 格式，只把诊断副本初始追捕者放到目标 0.25 m 范围内，因此可以证明方法链在可捕获条件下的双捕获指标正常；它仍是诊断初态，不替代正式 E49 分布抽样结果。

已扩展为 3 个 history 一致的可捕获初态（追捕者偏移 0.25/0.5/0.75 m）、11 种方法、每个 4 周期，产物为 `current/e50_deployment_audit_20261110/capturable_consistent_batch.json`。33 个组合全部有效；A2/A3/A4/AG/GEOM_SEARCH/RouteAPF/IG/A5/MAPPO_RouteAPF/PPO-C 各完成 3/3 双捕获，SINGLE_PRIOR 完成 2/3；执行碰撞为 0，局部投影仅少量触发。该结果首次给出可捕获条件下的双捕获对照，但仍属于诊断初态，正式论文指标仍需按冻结 E49 初态分布和完整预算运行。

已将同一可捕获初态对照延长到 12 周期，产物为 `current/e50_deployment_audit_20261110/capturable_consistent_batch_12cycles.json`。11 种方法在 3 个初态上均完成双捕获（33/33），执行碰撞为 0；A2/A3/A4/SINGLE_PRIOR/A5/PPO-C 主要通过局部投影继续执行，AG/RouteAPF/IG/MAPPO_RouteAPF 在目标捕获后进入停车状态。该结果确认 SINGLE_PRIOR 的短时域差异不是捕获链路错误，但停车统计必须按“目标已捕获后的正常停止”和“通信降级停车”分开，不能混报。

已修正停车记账：所有目标捕获后的零动作现在标记为 `post_capture_stop`，不再计入通信降级。重跑后的 12 周期结果仍为 33/33 双捕获、0 执行碰撞、所有方法通信降级停车均为 0；捕获后的停止与通信扰动已完全分离。

已补齐独立 Python 转移一致性验收，产物为 `current/e50_deployment_audit_20261110/transition_consistency_100.json`：100 个固定随机状态/动作/障碍案例中，加速 `joint_transition` 与参考 `reference_joint_transition` 的最大位置误差为 0，活动标志误差为 0，全部通过。该结果满足计划中的 Python 转移一致性门槛；C++ 绑定仍需在实际部署构建环境中单独复核。

已完成 C++ 绑定转移复核，产物为 `current/e50_deployment_audit_20261110/cpp_transition_consistency_20.json`：20 个交替空/有障碍状态和随机动作通过 Python `E49Environment.step` 与原生 `Problem_Wrapper.step` 对照，最大状态误差约 1.38e-6，失败 0。Python/C++ 转移一致性验收现已同时通过。

已生成汇总报告 `current/e50_deployment_audit_20261110/REPORT.md`，集中链接多场景未知地图、80 步 RootPUCT、可捕获初态和 Python/C++ 一致性证据，并明确当前诊断预算、初态和真实硬件接入限制。该报告是当前 E50 审计交付物，不等同于正式论文实验结论。

已新增 `code/solvers/e50_runtime_adapter.py`，提供 ROS 无关的 SLAM 位姿、双车 LiDAR、目标信念运行时边界；对缺失输入、传感器过期和时间戳错位统一输出 `degraded` 停车状态。配套 E50 单测现为 45 项通过，后续可直接把真实 ROS/通信驱动接到该边界，而不改变控制器和审计接口。

运行时边界 smoke 已生成 `current/e50_deployment_audit_20261110/runtime_adapter_smoke.json`，四种情形均按预期区分：live、`stale_sensor`、`timestamp_skew`、`missing_input`。这完成了硬件输入边界的离线验收；真实 ROS/SLAM/LiDAR 驱动接入仍是实车阶段工作。

交叉验收再次通过：E49 `test_e49_rules.py` 的 7 项规则/原生 C++ 等价性测试全部通过，E50 专项测试 45 项全部通过。当前仿真部署审计、Python/C++ 转移一致性和运行时输入边界均有独立证据。

运行时适配器已增加 `gate_command`：只要输入快照为 `degraded`，强制输出两车零动作；live 快照才放行控制命令。E50 专项测试现为 46 项通过，实车接入的停车安全门已有独立接口和回归测试。

运行时适配器新增 `record`，把快照原因、请求命令、门控命令和是否发生覆盖导出为 JSON-safe 记录；E50 专项测试现为 47 项通过，硬件联调具备可回放的降级证据格式。

`runtime_adapter_smoke.json` 已更新为包含四种输入故障的完整请求/门控命令记录，可直接作为硬件断链回放样例。

`runtime_adapter_smoke.json` 已更新为同时记录请求动作和门控后动作，live 情形原样放行，三种 degraded 情形均输出零动作；运行时降级停车现在有直接产物证据。

已生成实车交接文档 `current/e50_deployment_audit_20261110/HARDWARE_HANDOFF.md`，固定 SLAM 位姿、LiDAR、信念和命令的数据形状、时间新鲜度门槛、降级原因和上线前检查项。真实硬件接入仍未完成，文档明确保留这一未完成项。

已生成首轮方法筛选账本 `current/e50_deployment_audit_20261110/method_screening.json`。筛选规则要求：3 张未知地图有效、执行碰撞为 0、可捕获初态双捕获至少 2/3、并且已有 80 步 RootPUCT 长时域证据。A2、A3、A4、AG、GEOM_SEARCH、RouteAPF 满足全部条件，进入下一阶段；IG、SINGLE_PRIOR、A5、MAPPO_RouteAPF、PPO-C 的多场景和可捕获证据已通过，但 80 步 RootPUCT 长时域证据尚缺，暂缓进入正式筛选，不把缺失证据误判为失败。

第二批 80 步 RootPUCT 重跑已完成，产物为 `current/e50_deployment_audit_20261110/second_batch_rootpuct_unknown_80cycles_budget20_retry.json`：IG、SINGLE_PRIOR、A5、MAPPO_RouteAPF、PPO-C 均完成 80 次有效决策，执行碰撞、捕获和 RootPUCT 搜索失败均为 0。合并记账位于 `current/e50_deployment_audit_20261110/all_methods_rootpuct_80_accounting.json`，更新后的筛选账本显示 11 种方法全部满足当前审计门槛，均进入下一阶段；这仍是诊断场景和 20–50 次 RootPUCT 预算，不等于正式论文性能结论。

已生成逐项完成审计 `current/e50_deployment_audit_20261110/completion_audit.json`：计划门槛 11 项中 9 项已有仿真或接口证据，仍缺真实 ROS/SLAM/LiDAR/通信硬件接入和冻结 E49 正式预算下的双捕获统计。审计明确标记 `complete=false`，当前交付不会被误报为实车完成。

已生成冻结基线与部署审计并列结果 `current/e50_deployment_audit_20261110/frozen_e49_vs_e50_comparison.json`：保留 E49 正式已知地图双捕获率，同时单独列出 E50 未知地图场景数、执行碰撞和可捕获诊断双捕获，不混合两个实验口径。该结果可直接用于后续正式实验设计和方法筛选说明。

2026-11-12 修正未知地图接线：未知地图滚动循环现在先生成模拟 LiDAR 角度—距离扫描，再将命中点写入局部扫描记忆，并把占据命中聚合成不超过 24 个局部障碍 patch，作为控制器本周期的规划地图；之后才调用方法 `act()`，再统一送入 MINCO 轨迹和 LiDAR 安全门。未知图重建控制器时同步按当前局部几何过滤历史目标报告，避免全局地图报告与局部 belief 几何冲突；未把全局障碍列表作为规划输入。未知区域不生成障碍 patch，现有 E49 方法通过“局部 LiDAR patch → 旧几何 map 接口”的兼容桥接迁移。`test_e50*.py` 当前 53 项通过；默认 LiDAR 已调整为 256 根/360°（1.40625°），局部栅格 0.1 m、命中聚类距离 0.25 m；11 个方法在长墙两周期联调均能完成规划，暂无执行碰撞。该结果证明 LiDAR 局部障碍已进入规划调用前，但仍是兼容桥接，不等同于正式多地图双捕获率结论；下一步跑完整多地图、动态目标和正式 RootPUCT 预算对照，并统计停车/投影/捕获指标。

2026-11-12 完成 256 根 LiDAR、3 张验证地图、11 个方法、每图 12 周期、动态 RootPUCT 20 次的首轮分层诊断：396 个执行周期碰撞为 0，但计划轨迹安全门触发 55 次、未来碰撞停车 55 次、局部投影 11 次、MINCO 不可行 5 次、捕获 0 次。长墙贡献绝大多数安全门事件；稀疏多路径无计划碰撞但有 3 次 MINCO 不可行；AG/IG 规划耗时和不可行事件偏高。分析记录见 `current/e50_deployment_audit_20261110/lidar256_3maps_12cycles_analysis.md`。当前先定位为局部地图/安全门与轨迹语义问题，暂不按捕获数给方法排序。

2026-11-12 针对长墙安全门停车问题增加决策层局部绕障候选：在原直线/四轴投影之外加入目标方向旋转和最近 LiDAR 命中点两侧切向动作；仅当原动作全部被局部扫描挡住时启用，避免改变本来可行的动作。54 项 E50 单测通过。长墙 6 方法×4 周期联调执行碰撞为 0，未触发安全覆盖；Example8 默认密集图（17 个内部障碍，连同边界为常用 18 障碍口径）6 方法×4 周期执行碰撞为 0，只有少数未来轨迹安全门事件。

2026-11-12 在 Example8 常用密集图（17 个内部矩形，按边界计为 18 障碍口径）上完成 256 根 LiDAR、11 方法、12 周期动态 RootPUCT 联调：132 个执行周期碰撞为 0；AG、RouteAPF、IG 各产生 1 次捕获事件，安全覆盖主要为少量局部投影/未来停车，未出现长墙式系统性停车。该图暂不需要直接切换到 MINCO 带障碍优化；先保留决策层切向绕障候选，并继续做正式双捕获和更高对手预算对照。

2026-11-12 使用现有 E50 滚动执行链在 Example8 密集障碍图上为 11 个方法各跑一局 4 周期可视化，记录完整 MINCO 执行采样轨迹、逃跑者位置和 LiDAR 命中点；总览图及单方法图位于 `current/e50_deployment_audit_20261110/example8_dense17_trajectory_plots/`，原始轨迹记录为 `example8_dense17_trajectory_episode.json`。

2026-11-12 修正轨迹可视化口径：E50 当前每周期规划 2 s MINCO 端点轨迹，但实际只执行前 1 s 后滚动重规划；旧图把未执行的后 1 s 规划尾段也画成已执行轨迹，造成断裂和急转的视觉误判。输出现已区分 executed prefix 与 planned tail，绘图脚本兼容旧记录并默认只绘制真实执行前缀。进一步确认当前执行器是端点约束五次最小 jerk 兼容轨迹，不是带障碍全局 MINCO；单段内平滑，跨周期动作方向仍可突变，需后续加入跨周期动作/速度变化率约束。

2026-11-12 完成跨周期平滑首轮验证：新增可控的相邻周期目标位移变化率限制，默认平滑实验阈值 1.2 m/周期；首周期不改动作，后续以实际送达动作作为连续性参考，并记录原始 controller_action 与平滑后的 action。Example8 密集图 11 方法×4 周期无执行碰撞；原始/平滑逐方法对比图见 `current/e50_deployment_audit_20261110/raw_vs_smoothed_trajectory_comparison.png`。A2/A3/A4/GEOM_SEARCH 的轨迹有可见但有限的绕向变化，AG、RouteAPF、IG、SINGLE_PRIOR、PPO-C 基本重合；当前未发现需要改变方法语义或立即引入带障碍 MINCO 的证据。

2026-11-13 复用旧几何逃跑轨迹流程完成完整 68 步接管对照：使用 `e49_hard_dense_00_AB`、旧 `GeometricEvader`、11 个当前方法、256 根 LiDAR、局部地图和 MINCO 执行。首版完整尾段安全门结果为 A3/A5 双捕，其余方法被未执行的 2 s 轨迹尾段安全门大量截断；随后改为只对实际执行的 1 s 前缀做安全门，完整尾段仅记账。前缀门版 11 方法均跑满 68 步、执行碰撞 0；A2/A3/A4/A5/IG 各捕获 1 个目标，尚未完成双捕，AG/GEOM_SEARCH/RouteAPF/SINGLE_PRIOR/MAPPO_RouteAPF/PPO-C 仍有较多前缀安全停车。说明“走两步停”首先来自短 smoke/未执行尾段安全门，其次仍有 MINCO 动力学时间语义和局部 LiDAR 安全门退化。旧 atlas 绘图代码已直接复用，图册目录为 `current/e50_deployment_audit_20261110/e50_legacy_atlas_dense00_AB_prefixgate/`，GIF 为 `e50_lidar_minco_geometry_escape_11methods_prefixgate.gif`。

2026-11-16 复核：在 Example8 dense17 验证图上用当前滚动执行链重新跑 11 个方法、每方法 12 周期、RootPUCT 接口预算 20，产物为 `current/e50_deployment_audit_20261116/unknown_map_dense17_rootpuct_12cycles.json`，摘要为 `SUMMARY.md`/`SUMMARY.json`。132 个执行周期全部有效，执行碰撞、MINCO 不可行和 RootPUCT 搜索失败均为 0；本轮冻结逃跑者未推进，捕获数为 0，不作为捕获率结论。对同一动作日志追加名义、1 周期通信/位姿延迟、丢包和延迟+丢包审计，产物为 `distributed_unknown_map_dense17_12cycles.json`。这补充了当前代码的可重复长时域安全证据，但仍未完成真实 ROS/SLAM/LiDAR/通信硬件接入，也未替代冻结 E49 正式多地图完整预算双捕获统计。

2026-11-16 动态目标复核：在同一 Example8 dense17 未知地图上，用动态 RootPUCT 逃跑者（20 次接口预算）运行首批六种方法、每方法 12 周期，产物为 `current/e50_deployment_audit_20261116/unknown_map_dense17_moving_rootpuct_12cycles_first6.json`。6/6 方法完成全部周期；执行碰撞和 RootPUCT 搜索失败均为 0。本轮记录了动态目标下的捕获事件与最小距离，结果仍是单图、短时域接口复核，不能替代正式多地图完整预算统计。

2026-11-16 多地图动态目标复核：在 long-wall、wall-corner、sparse-multi-route 三张验证图上，首批六种方法各运行 12 周期，动态 RootPUCT 预算 20。三张图共 216 个方法-周期组合全部完成，执行碰撞和搜索失败均为 0；捕获仅在 dense17 复核中出现，三张新增图短时域内未捕获。原始结果分别为 `current/e50_deployment_audit_20261116/unknown_map_{wall,corner,sparse}_moving_rootpuct_12cycles_first6.json`。这完成了多地图动态接口复核，但还不是完整 80 步、正式 RootPUCT 预算的论文统计。

2026-11-16 修复 RouteAPF 长时域初始化：稀疏 LiDAR 局部 patch 覆盖冻结 RouteAPF 的 public anchor 时，增加仅针对 `No lawful public anchor` 的空兼容地图重试；实时 LiDAR/MINCO 安全层仍保留。long-wall 2 周期回归成功，E50 专项测试 76 项通过。该兼容回退需在后续正式实验中单独记账，不能当作完整局部地图规划证据。

2026-11-16 RouteAPF 交叉长时域复核：sparse-multi-route 未知地图、动态 RootPUCT、80 周期运行成功，RouteAPF 80/80 周期有效，执行碰撞 0、搜索失败 0、捕获 0。RouteAPF 的空兼容地图初始化回退已在 long-wall 与 sparse-multi-route 两种地图验证。

2026-11-16 长时域汇总审计：`current/e50_deployment_audit_20261116/LONG_HORIZON_SUMMARY.md` 汇总了当前 80 周期记录。A2/A3/AG/GEOM_SEARCH 在 long-wall 记录中无执行碰撞；RouteAPF 在修复后的 long-wall 与 sparse-multi-route 记录中均 80/80 有效且无碰撞。需要特别保留一个未通过项：IG 的 long-wall 80 周期旧批次出现 52 次执行碰撞和 992 次 RootPUCT 搜索失败，不能纳入稳定方法筛选；该结果需要后续单独修复或解释，不能被短时域 12 周期结果覆盖。

2026-11-16 IG 长时域复核修复：针对 IG 在第 28 周期后出现的 LiDAR 不安全前缀仍被执行，增加 `stop_lidar_prefix` 执行前安全覆盖并重新运行 long-wall 动态 RootPUCT 80 周期。结果 `current/e50_deployment_audit_20261116/unknown_map_wall_moving_rootpuct_80cycles_ig_safety_gate.json`：80/80 周期有效，执行碰撞 0、RootPUCT 搜索失败 0、捕获 0，安全覆盖触发 57 次。该结果说明碰撞/搜索失败被安全层隔离，但 IG 在该场景仍有较高停车退化，不能直接视为高效方法。

2026-11-16 IG 跨地图复核：sparse-multi-route 上重新运行 IG 80 周期，结果 `current/e50_deployment_audit_20261116/unknown_map_sparse_moving_rootpuct_80cycles_ig_safety_gate.json`。80/80 周期有效，执行碰撞 0、RootPUCT 搜索失败 0、捕获 1，LiDAR 前缀安全覆盖 18 次；与 long-wall 的 57 次相比，退化明显依赖长墙几何。

2026-11-16 IG 长墙投影复核：将 LiDAR 不安全执行前缀从直接停车改为局部安全动作投影后，long-wall 动态 RootPUCT 80 周期结果为 `current/e50_deployment_audit_20261116/unknown_map_wall_moving_rootpuct_80cycles_ig_projection.json`：80/80 有效，执行碰撞 0、搜索失败 0，局部动作投影 16 次，直接停车 0 次。相较此前安全门版本的 57 次停车，长墙退化显著降低。

2026-11-16 分布式独立 MCTS 回归修复：修正局部 MCTS 在小预算下选择未访问候选的问题；局部候选预算固定为 8，所有返回计划的 `visits` 均为正。修正未发现目标时的价值先验，使车辆执行局部探索，而不是因零目标信念原地不动。修正冻结 E49 RootPUCT 在单目标退役后仍被调用的问题：双目标仍使用 RootPUCT，单目标阶段切换到显式单目标运动模型，并单独记录步数。最终 dense17 名义、dense17 一周期延迟加第 20 周期丢包、半障碍延迟丢包三条 80 周期回归均为执行碰撞 0、MINCO 不可行 0、RootPUCT 搜索失败 0；捕获分别为 1、1、1。汇总见 `current/e50_deployment_audit_20261116/DISTRIBUTED_FINAL_COMPARISON.md`。该结果是分布式链路回归，不替代冻结 E49 正式多地图双捕获率；单目标 RootPUCT 原生绑定、LeafValueNet 正式训练、真实 ROS/SLAM/LiDAR/通信接入仍待完成。

2026-11-16 单目标 RootPUCT 生命周期继续修复：确认 E49 原生绑定可以接受第二逃跑者 inactive 的状态；此前异常来自单目标运动模型将目标推进场地边界或障碍物内部，以及在 `time==tf` 后仍调用搜索。新增仿真状态合法性投影（边界余量与最近自由点）、终端搜索跳过和异常周期记录；单目标阶段现在继续调用 RootPUCT，状态修复、终端跳过和捕获后的生命周期分别记账。最终三条 80 周期回归（dense17 名义、dense17 延迟丢包、半障碍延迟丢包）RootPUCT 搜索失败均为 0、执行碰撞均为 0、MINCO 不可行均为 0；捕获为 1/1/1。半障碍状态投影 29 次、终端跳过 12 次，均不再混入方法失败。77 项 E50 测试通过。汇总已更新至 `current/e50_deployment_audit_20261116/DISTRIBUTED_FINAL_COMPARISON.md`。正式多地图双捕获率、LeafValueNet 训练和真实硬件接入仍未完成。

2026-11-16 多地图执行语义修正与 80 周期统计：发现 dynamic runner 在 MINCO 前缀推进后又把同一追捕者动作送入 E49 `step`，造成追捕者重复推进、越界和伪搜索失败；现已改为 MINCO 前缀作为唯一追捕者执行，E49 `step` 只负责逃跑者推进与捕获状态更新，并增加基于当前局部 LiDAR 的执行前缀截断。6 个场景共 480 周期的统一统计见 `current/e50_deployment_audit_20261116/DISTRIBUTED_MULTIMAP_80.md`：RootPUCT 搜索失败、执行碰撞和 MINCO 不可行均为 0；dense17、半障碍、wall-corner 出现捕获，long-wall 和 sparse-multi-route 未捕获且前缀截断分别为 140、141 次。该结果说明长墙/稀疏图的主要瓶颈已定位为局部 LiDAR 候选与执行前缀的停滞退化，不能直接进入最终双捕获率比较；下一步应先改进局部绕障候选/轨迹执行，再做多初态重复统计。77 项 E50 测试继续通过。

2026-11-16 局部候选与执行一致性修正：长墙/稀疏图的前缀截断根因进一步确认是候选生成没有把仿真场地边界作为局部执行约束，以及候选线段检查使用 8 点而 MINCO 评价使用更密采样。现已加入 ±9m 边界约束，并将候选检查采样统一为 32 点。重新跑 6 个场景、每场景 80 周期（共 480 周期）后，RootPUCT 搜索失败和执行碰撞均为 0；long-wall、sparse-multi-route 前缀截断均降为 0。dense17、半障碍、wall-corner 各出现 1 个捕获；wall-corner 仅第 12 周期出现 1 次局部 MINCO 碰撞/前缀截断，需后续单独复核。汇总见 `current/e50_deployment_audit_20261116/DISTRIBUTED_MULTIMAP_80.md`，77 项 E50 测试通过。该结果仍不是冻结 E49 多初态正式双捕获率。

2026-11-16 多地图可捕获初态链路验收：新增 `code/run_e50_distributed_capturable_batch.py`，在 6 张地图上按 0.25/0.5/0.75m 追捕者偏移生成 18 个可捕获诊断初态，使用当前 256 根 LiDAR、分布式局部 MCTS、MINCO 和 RootPUCT 跑 8 周期。18/18 初态完成双捕获，执行碰撞、MINCO 不可行和 RootPUCT 失败均为 0；结果见 `current/e50_deployment_audit_20261116/DISTRIBUTED_CAPTURABLE_MULTIMAP.md`。该批次验证捕获配对、双捕获记账和执行链路，不替代冻结 E49 随机初态的正式双捕获率。

2026-11-16 wall-corner 冲突复核：将局部候选安全检查采样从 32 点提高到 128 点，重新跑 6 场景、每场景 80 周期。wall-corner 的第 12 周期局部 MINCO 碰撞/前缀截断消失；6 场景共 480 周期的 RootPUCT 搜索失败、执行碰撞、MINCO 不可行和前缀截断均为 0。wall-corner 在统一口径下完成 2 个目标捕获，dense17/半障碍各捕获 1 个目标，long-wall/sparse-multi-route 仍在该初态 80 周期内未捕获。最终汇总更新至 `current/e50_deployment_audit_20261116/DISTRIBUTED_MULTIMAP_80.md`。该结果仍需多初态重复后才能形成正式双捕获率。

2026-11-16 随机初态重复：新增 `code/build_e50_distributed_random_cases.py`，为 long-wall、wall-corner、sparse-multi-route 各生成 3 个合法随机平移初态，并在当前统一口径下运行 9 个 80 周期任务。9 个任务执行碰撞、RootPUCT 失败、MINCO 不可行和前缀截断均为 0；wall 0/3 捕获，corner 3/3 捕获 1 个目标，sparse 1/3 捕获 1 个目标，双捕获 0/9。结果见 `current/e50_deployment_audit_20261116/DISTRIBUTED_RANDOM_INITIALS_80.md`。该结果说明主要瓶颈已经从安全性转为 long-wall/sparse 的追捕效率和初态依赖；正式随机双捕获率仍未达标。

2026-11-16 目标观测与航向修复：确认分布式 runner 丢弃了同一帧中第二个目标报告，且车辆航向一直固定为 0，导致前向相机在目标位于后方时无法通过运动转向发现；无目标时探索扇区也固定优先向 +x。现已保留全部目标报告，将执行位移更新为车辆航向并传入相机/FOV，按周期旋转未知目标探索扇区，未训练价值先验改为靠近最近可见 belief 支持。修复后 sparse-multi-route 3 个随机初态均捕获 1 个目标（修复前 1/3），long-wall 仍 0/3；6 个任务 RootPUCT 失败、执行碰撞、MINCO 不可行和前缀截断均为 0。结果见 `current/e50_deployment_audit_20261116/DISTRIBUTED_RANDOM_INITIALS_80_AFTER_FIX.md`。该修复改善了目标发现，但随机初态双捕获仍未完成，long-wall 仍需继续处理。

2026-11-16 long-wall 追捕诊断：在目标报告保留、航向/FOV 接入和旋转探索修复后，sparse-multi-route 80 周期动态复核捕获 1 个目标；long-wall 仍未捕获，但所有安全/搜索指标为 0 失败。进一步尝试仅按局部开阔度给端点评分会导致长墙前左右振荡，已回退该未通过方案。当前 long-wall 的下一步应实现基于 LiDAR 命中边缘的切向绕行候选，并在候选生成层解决遮挡墙体绕行，不能继续叠加端点开阔度权重。

2026-11-16 局部墙边切向候选：在 `e50_distributed_geometry.py` 增加基于当前 LiDAR obstacle patch 中心、左右切向和扩展角点的局部绕行候选，并继续由 MINCO 全轨迹安全检查过滤。long-wall 20 周期定向回归显示追捕者动作从固定直行变为沿墙边切向绕行，RootPUCT 失败、MINCO 不可行、前缀截断和执行碰撞均为 0；完整 80 周期仍未捕获，但链路安全稳定。该结果说明候选生成已开始利用局部墙体几何，long-wall 的剩余瓶颈转为目标重新发现/绕墙后追击效率，需继续做相机搜索与切向候选联合实验。

2026-11-16 long-wall 传感器证据：在动态 runner 中增加每周期可见目标数、belief 粒子数、相机状态和云台角度记账。标准 80 周期与延长 horizon=160 诊断均显示 long-wall 可见目标数始终为 0，RootPUCT 失败、MINCO 不可行、前缀截断和执行碰撞均为 0；说明目标从未进入相机/FOV 可见侧，问题发生在遮挡墙体绕行和搜索覆盖阶段。将局部候选预算从 4 提高到 8 后动作覆盖更完整，但 80 周期仍无可见目标。下一步需要基于局部墙体端点建立可达 frontier 路径，而不是继续调节目标追击价值。

2026-11-16 frontier 连续绕行：将上一周期实际执行方向作为无目标 belief 时的连续候选，避免局部探索扇区每周期换向；局部候选预算固定为 8。long-wall 80 周期复核首次在第 20 周期进入相机可见区，第 28 周期捕获 1 个目标，可见目标周期数 59；RootPUCT 失败、MINCO 不可行、前缀截断和执行碰撞均为 0。该结果证明局部墙边连续绕行已能把追捕者带到目标可见侧，下一步继续验证第二目标追击和多初态重复。

2026-11-16 frontier 随机初态复核：long-wall 的 3 个随机初态在同一 frontier 连续绕行版本下均进入相机可见区，首次可见周期分别为 34、60、32，并各捕获 1 个目标；RootPUCT 失败、执行碰撞、MINCO 不可行和前缀截断均为 0。结果见 `current/e50_deployment_audit_20261116/FRONTIER_RANDOM_WALL_80.md`。当前剩余问题是捕获一个目标后的第二目标持续信念和队友协同，双捕获尚未完成。

2026-11-16 双捕获 active 状态修复：确认 dynamic runner 将逃跑者 active 数组误用于追捕者 active 状态；第一个目标捕获后，第二辆追捕车被错误置为 INACTIVE，导致双捕获失败。现已拆分 `active_e` 与 `active_p`，相机、MINCO 执行和捕获记账分别使用正确状态。long-wall frontier 80 周期复核完成双捕获：第 28 周期捕获目标 1，第 39 周期捕获目标 0；RootPUCT 搜索失败、执行碰撞和 MINCO 不可行均为 0。77 项 E50 测试通过。该修复是双车协同链路的关键正确性修复，需继续在随机初态和其他地图复跑确认。

2026-11-16 active 修复后随机 long-wall 复核：3 个随机初态在 frontier 连续绕行版本下首次可见周期为 34、60、32；其中 2/3 完成双捕获，剩余 1 个捕获 1 个目标。RootPUCT 失败、执行碰撞、MINCO 不可行和前缀截断均为 0。结果见 `current/e50_deployment_audit_20261116/ACTIVE_FIX_RANDOM_WALL_80.md`。相比 active 修复前的 0/3 双捕获，当前提升到 2/3；剩余问题是少数初态下第二目标在 80 周期内未完成追捕。

2026-11-17 LiDAR 自身回波修复与随机复核：corner 随机初态第 3 组曾出现 58 次 MINCO 前缀截断，逐周期日志显示追捕车在贴近墙体后，零距离 LiDAR 命中被障碍 patch 膨胀并包含当前传感器位置，导致后续静止轨迹也被判碰撞。`LidarPlanningMap` 现记录最新传感器位姿并过滤包含自身位姿的 patch；8 项 LiDAR 单测和 77 项 E50 专项测试通过。该组 80 周期重跑后前缀截断、MINCO 不可行/碰撞、执行碰撞和 RootPUCT 失败均为 0，捕获 1 个目标。合并 wall/corner/sparse 的 9 个随机初态结果为 8/9 双捕获、9/9 安全与搜索指标通过，汇总见 `current/e50_deployment_audit_20261116/ACTIVE_FIX_RANDOM_CORNER_SPARSE_80.md`。仍需保留：这是诊断随机初态，尚未替代冻结 E49 正式预算、多初态统计；LeafValueNet 尚未训练，真实 ROS/SLAM/LiDAR/通信仍未接入。

2026-11-17 长时域与捕获生命周期复核：将 corner random 2 和 long-wall random 0 延长到 160 周期。corner random 2 在第 27 周期捕获 target1、第 151 周期捕获 target0，执行碰撞、MINCO 不可行/碰撞、前缀截断和 RootPUCT 失败均为 0；long-wall random 0 在 160 周期仍捕获 1 个目标，但同样所有安全/搜索指标为 0。复核中发现并修复执行层手动捕获未同步 E49 active 标志导致的重复捕获事件；现已同步 active 状态并以目标集合去重。原始结果和统计见 `current/e50_deployment_audit_20261117/LONG_HORIZON_RECHECK.md`。当前随机诊断 9/9 安全通过，corner 未双捕获初态在延长时域补齐双捕获；long-wall 个别初态仍存在第二目标追击效率不足，需继续做协同/信念保持实验。

2026-11-17 长时域与捕获生命周期复核：将 corner random 2 和 long-wall random 0 延长到 160 周期。corner random 2 在第 27 周期捕获 target1、第 151 周期捕获 target0，执行碰撞、MINCO 不可行/碰撞、前缀截断和 RootPUCT 失败均为 0；long-wall random 0 在 160 周期仍捕获 1 个目标，但同样所有安全/搜索指标为 0。复核中发现并修复执行层手动捕获未同步 E49 active 标志导致的重复捕获事件；现已同步 active 状态并以目标集合去重。原始结果和统计见 `current/e50_deployment_audit_20261117/LONG_HORIZON_RECHECK.md`。当前随机诊断 9/9 安全通过，corner 未双捕获初态在延长时域补齐双捕获；long-wall 个别初态仍存在第二目标追击效率不足，需继续做协同/信念保持实验。

2026-11-17 分布式信念交换与捕获后重搜：发现原 PeerMessage 只交换位姿/意图，没有交换目标 belief；加入有限窗口的 belief 粒子传输与接收端合并。首轮复核又发现历史粒子和延迟消息会使车辆追击已捕获目标，现已加入捕获周期屏障：清空两车 belief、忽略捕获前 peer 消息并短暂抑制旧 belief 发送，同时保留目标级唯一捕获记账。77 项 E50 测试通过。长墙随机初态 80 周期复核完成双捕获（target0@50、target1@61），执行碰撞、MINCO 不可行/碰撞、前缀截断和 RootPUCT 失败均为 0，结果见 `current/e50_deployment_audit_20261117/BELIEF_EXCHANGE_LONG_WALL_80.md`。这证明分布式队友信念交换已在仿真链路生效；仍需在 corner/sparse 多初态上复跑通信版本，并完成 LeafValueNet 训练、正式冻结预算统计及真实 ROS/SLAM/LiDAR/通信接入。

2026-11-17 belief 新鲜度与跨地图抽样：在长墙通信成功基础上加入本地观测时间戳、5 周期过期清除、捕获周期屏障及过期消息抑制。wall random0 80 周期完成双捕获（target0@50、target1@61）；corner random0 捕获 1 个目标，sparse random0 未捕获，但三者执行碰撞、MINCO/前缀异常和 RootPUCT 失败均为 0。结果见 `current/e50_deployment_audit_20261117/BELIEF_EXCHANGE_CROSS_MAP_80.md`。当前结论：分布式信念交换链路已工作，长墙问题得到解决；corner/sparse 仍暴露局部候选/搜索覆盖停滞，不能宣称所有地图迁移完成。下一步应优先改进无目标或过期 belief 时的 frontier/free-ray 候选，并在修复后重跑多初态通信统计。

2026-11-17 LiDAR 自由射线候选修复：确认自由射线候选最初虽已加入 `LidarPlanningMap`，却在 `_CycleProxy` 中被丢失，局部 MCTS 实际拿不到；现已在 proxy 保留最新自由射线端点，并优先于大批切向候选进入候选预算。对 sparse random0 延长到 160 周期后，target1@89、target0@104 完成双捕获，执行碰撞、MINCO 不可行/碰撞、前缀截断和 RootPUCT 失败均为 0。结果见 `current/e50_deployment_audit_20261117/FREE_RAY_SPARSE_160.md`。E50 专项测试增至 78 项并通过。下一步复跑 corner 多初态，确认自由射线候选的跨地图效果。

2026-11-17 自由射线候选 corner 复核：corner random0 延长到 160 周期后，target1@42、target0@50 完成双捕获；执行碰撞、MINCO 不可行/碰撞、前缀截断和 RootPUCT 失败均为 0，结果见 `current/e50_deployment_audit_20261117/FREE_RAY_CORNER_160.md`。结合 sparse random0 的 target1@89、target0@104，局部 LiDAR frontier/free-ray 候选已在 corner、sparse 两类困难地图上恢复持续追捕。剩余工作转为多初态统计、LeafValueNet 正式训练/消融、真实 ROS/SLAM/LiDAR/通信接入和最终 GIF 可视化回归。

2026-11-17 新版轨迹可视化：基于 belief 交换、捕获后信念清除和 LiDAR 自由射线候选的最新日志生成 3 份 GIF：`current/e50_deployment_audit_20261117/wall_belief_exchange.gif`（80 帧）、`corner_free_ray.gif`（160 帧）、`sparse_free_ray.gif`（160 帧）。图中绘制的是每周期实际执行位置，分别对应长墙双捕获、corner 双捕获和 sparse 双捕获结果。

2026-11-17 LeafValueNet 训练与闭环：新增 `run_e50_leaf_value_ablation.py`，用 8000 个公开局部候选样本（包含 belief 追击与无 belief frontier 探索）训练 `e50_leaf_value_v2` checkpoint。留出集 RMSE 0.0961、Pearson 0.99980、排序相关 0.96176；训练值只作为几何/MINCO 评分的小幅修正，无 belief 时保留自由射线探索先验。sparse random0 160 周期加载 checkpoint 后 target1@89、target0@104 双捕获，执行碰撞、MINCO/前缀异常和 RootPUCT 失败均为 0。详细证据见 `current/e50_deployment_audit_20261117/LEAF_VALUE_TRAINING.md`；LeafValueNet 正式训练项已具备离线和闭环证据，仍需多初态 value-vs-untrained 消融与真实部署接入。

2026-11-17 LeafValueNet 多初态闭环统计：使用训练 checkpoint、256 根 LiDAR、belief 交换、自由射线候选和 RootPUCT 20 次，在 wall/corner/sparse 各 3 个随机初态运行 160 周期。9/9 任务执行碰撞、MINCO 不可行/碰撞、前缀截断和 RootPUCT 失败均为 0；双捕获 6/9（wall 2/3、corner 3/3、sparse 1/3），单捕获 1/9，零捕获 2/9。结果见 `current/e50_deployment_audit_20261117/LEAF_VALUE_MULTIMAP_160.md`。这完成了训练值网络的初步多初态闭环审计，但正式论文统计仍需固定随机种子/预算和更大样本量，真实 ROS/SLAM/LiDAR/通信接入仍未完成。

2026-11-17 LeafValueNet 版本可视化：为 9 个训练 value 多初态日志生成代表性 160 帧 GIF：`current/e50_deployment_audit_20261117/wall_leafvalue.gif`、`corner_leafvalue.gif`、`sparse_leafvalue.gif`。图中使用实际执行位置和最新捕获记账，可用于检查训练值网络对轨迹连续性和搜索方向的影响。

2026-11-17 LeafValueNet 严格消融完成：对同一 wall/corner/sparse 9 个随机初态、160 周期统一运行训练 checkpoint 与未训练 prior。两者均为 6/9 双捕获、1/9 单捕获、2/9 零捕获；两组执行碰撞、MINCO/前缀异常和 RootPUCT 失败均为 0。训练值网络已安全接入，但当前样本没有捕获率增益，不能宣称其优于未训练先验。结果见 `current/e50_deployment_audit_20261117/LEAF_VALUE_ABLATION_160.md`；需要更丰富的动态 belief/遮挡/协同标签后再做增益训练。

2026-11-17 更新完成审计：`run_e50_completion_audit.py` 现同时检查自由射线 frontier、LeafValueNet 训练、value-vs-untrained 消融和最新 GIF。当前 15 项计划门槛中 13 项有证据，仍明确缺少真实 ROS/SLAM/LiDAR/通信硬件接入和冻结 E49 正式采样/预算下的双捕获统计；机器可读审计为 `current/e50_deployment_audit_20261117/completion_audit.json`，`complete=false`。

2026-11-17 history 同步版正式随机统计：修正随机样本生成器，使初始状态、scripted_states、public history positions/reports 同步平移；三类地图各生成 10 个合法初态，共 30 个，统一训练 LeafValueNet、256 根 LiDAR、belief 交换、自由射线候选、MINCO、RootPUCT 20 次、160 周期运行。结果为 wall 1/10、corner 10/10、sparse 4/10 双捕获，合计 15/30；执行碰撞 0、MINCO 异常 0、RootPUCT 失败 0，仅 wall 1 个前缀截断。该结果首次具备 history 同步的正式多初态统计口径，但也明确暴露 wall/sparse 的初态依赖，不能把单个成功 GIF 当作整体性能。报告见 `current/e50_deployment_audit_20261117/FORMAL_RANDOM_30_160.md`。

2026-11-17 完成审计更新：history 同步版 30 初态、160 周期正式统计已纳入机器审计，当前 15 项门槛中 14 项有仿真/接口证据；唯一仍 pending 的是实际 ROS/SLAM/LiDAR/通信硬件接入。`completion_audit.json` 仍保持 `complete=false`，因为仿真证据不能替代实车联调。

2026-11-17 ROS 消息适配回放：`e50_runtime_adapter.py` 新增 ROS1/ROS2 时间戳、PoseArray、LaserScan 和 belief 消息转换入口；新增 `run_e50_ros_adapter_smoke.py`，验证 live 消息放行、stale 消息两车停车，且不依赖 ROS 包即可回放。E50 专项测试增至 79 项并通过。该项完成代码边界和回放验收，真实 ROS 节点、话题和车载设备联调仍需现场硬件。

2026-11-17 ROS2 bridge 骨架：新增 `code/ros2/e50_runtime_bridge/`，提供 PoseArray、两路 LaserScan、Float64MultiArray belief 的消息转换和 `/e50/cmd0,/e50/cmd1` 门控发布；ROS 未安装环境下完成 py_compile、belief 解码和 79 项 E50 测试。交接说明见 `current/e50_deployment_audit_20261117/ROS2_BRIDGE_HANDOFF.md`。当前仍不能宣称真实硬件完成，现场还需在车载 ROS2 镜像中 colcon build、接真实话题并做低速断链/停车验收。

2026-11-17 ROS2 bridge 命令链补齐：bridge 新增 `/e50/requested_cmd0` 和 `/e50/requested_cmd1` 请求命令入口，MCTS/控制器输出先进入请求话题，再由同步快照门控后发布 `/e50/cmd0/1`；新增有限值/维度校验和 3 项 bridge 单测。E50 专项测试增至 82 项并通过。真实 ROS 话题连接仍需在车载镜像现场验证。

2026-11-17 队友分散覆盖修复：失败日志显示 wall/sparse 中两车常汇合到同一 frontier。局部候选现在避开最新队友位姿 0.75m，并对自由射线顺序按 agent/cycle 错位。wall random0 定向复核由 0 捕获改善为双捕获（target1@51、target0@71），碰撞、MINCO、前缀截断和 RootPUCT 失败均为 0；sparse random1 改善为单捕获，240 周期仍未发现第二目标。证据见 `current/e50_deployment_audit_20261117/PEER_DIVERSITY_TARGETED.md`。完整 30 初态统计需在该版本冻结后重新跑，旧 15/30 结果仅作为上一版本基线。

2026-11-17 队友分散候选完整消融：将“候选避开队友0.75m + 自由射线顺序错位”扩展到 history 同步 30 初态、160 周期，双捕获为 14/30（wall2/10、corner9/10、sparse3/10），执行碰撞/MINCO/RootPUCT均为0，但前缀截断2次；已验证基线为15/30、前缀截断1次。该策略不合入主线，代码已恢复基线候选逻辑；定向 wall 改善只作为消融证据。报告见 `current/e50_deployment_audit_20261117/PEER_DIVERSITY_ABLATION.md`。

2026-11-17 捕获后 previous_action 消融：尝试捕获后清空两车 previous_action 以强制重新探索；sparse random1 延长 240 周期仍 0 捕获，且原本成功的 sparse random0 由双捕获退化为 0 捕获。该策略已回滚，主线保留连续动作候选；捕获后的 belief/active/过期消息生命周期修复继续保留。记录见 `current/e50_deployment_audit_20261117/POST_CAPTURE_SEARCH_ABLATION.md`。

2026-11-17 无目标扇区优先消融：将两车错开的探索扇区置于自由射线之前，sparse 两个定向初态均双捕获（失败 sparse1 改为 target0@36、target1@79），但 wall random0 退化为 0 捕获并产生 1 次前缀截断。该策略不具备跨地图稳定性，已回滚为自由射线优先主线；sparse 改善仅作为后续地图自适应搜索线索，报告见 `current/e50_deployment_audit_20261117/SECTOR_FIRST_ABLATION.md`。

2026-11-17 LiDAR 自由射线角度自适应消融：按局部自由射线角度覆盖动态切换扇区优先/自由射线优先；sparse random1 仍单捕获、wall random0 仍零捕获，未改善基线，已回滚。报告见 `current/e50_deployment_audit_20261117/ADAPTIVE_SECTOR_ABLATION.md`，主线保持简单的自由射线优先候选。

2026-11-17 ROS2 静态 preflight：新增 `run_e50_ros2_preflight.py`，在无 ROS 环境下检查 package.xml、setup entrypoint、PoseArray、两路 LaserScan、belief、请求命令、输出命令和 degraded gate 共 8 项，8/8 通过；车载环境执行命令为 `colcon build --packages-select e50_runtime_bridge`。当前完成审计增加到 17/18，唯一实测缺口仍是真实 ROS/SLAM/LiDAR/通信设备联调。

2026-11-17 ROS2 包安装前检查：在当前无 ROS 的环境中，`setup.py --name --version` 成功识别 `e50_runtime_bridge 0.1.0`，node.py py_compile 通过，E50 专项 82 项全部通过，代码差异检查通过。真实车载镜像仍需执行 colcon build 和实际话题联调。
