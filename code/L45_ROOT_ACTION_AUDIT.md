# L45 根动作评分核验

正式协议：[PLAN.md](../current/l45_root_action_audit_20261024/PLAN.md)。本轮只核验冻结 O 的首捕后一次决策；不会训练或替换模型。

入口 `run_l45_root_action_audit.py`，私有构建 `build_root_action_audit.py`，适配器 `solvers/root_audit_puct.py`。历史源码和扩展保留。根导出在原动作选定后读取团队 Q、回传值、边奖励、访问次数及动作，不参与搜索。

```bash
/home/sjc/miniconda3/envs/dm_env_py311/bin/python code/build_root_action_audit.py
/home/sjc/miniconda3/envs/dm_env_py311/bin/python -m unittest discover -s code/tests -p test_l45_root_action_audit.py -v
/home/sjc/miniconda3/envs/dm_env_py311/bin/python code/run_l45_root_action_audit.py --smoke --workers 8
/home/sjc/miniconda3/envs/dm_env_py311/bin/python code/run_l45_root_action_audit.py --workers 80
/home/sjc/miniconda3/envs/dm_env_py311/bin/python code/run_l45_root_action_audit.py watch --once
```

阶段：pipeline、prepare、audit、extract、branch、analyze、report、watch、replay。支持 --output、--smoke、--once、--workers。smoke 的受控晚期状态来自独立新场景，只检查工程分支；正式阶段只使用全部 99 个历史 O 首捕快照。原 100 场景中另一个零捕终止仅进入全场景统计。

根缓存和后缀缓存以 task 指纹命名，`.pt` 原子写入后以独立摘要提交，JSON 锁同时保存内容和摘要。未提交任务可重放；已提交内容、来源、配对或指纹变化则停止。各 worker 的科学随机流只依赖物理场景、随机流编号和角色，不依赖进程数量或任务顺序。

正式每状态至多四候选、三个随机流，最多 1188 条后缀。r0 从保存的根搜索后本方 RNG 和原对手决策前 RNG 开始，A0 必须复现旧 OO 完整轨迹。r1/r2 的辅助 C++ problem RNG 使用该平台 default_random_engine（minstd_rand0）的有效种子状态，分布参数保留；转换规则和平台/构建一同锁定。双方当前步都看干预前状态，后续按新状态各自决策。

每状态用两个随机流选择动作、留出第三流评价，轮换三次。主指标按物理状态等权，bootstrap 共同保留候选和三个流。本指标仍使用其他流的未来信息，不是部署选择器；历史诊断通过只能支持另行研究动作评分。

工件：experiment.json、bank.json、scenario_independence.json、audit_result.json、extract/manifest.json、branch/manifest.json、roots/、suffixes/、paired_results.json、statistics.json、results.json、replay.json、REPORT.md、status.json。通过投入门槛时另交付 SCORER_NEXT_PLAN.md 草案，不自动训练。
