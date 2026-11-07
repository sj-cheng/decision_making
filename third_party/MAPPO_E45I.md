# E45-I upstream dependency

The unmodified MAPPO author repository is checked out in `mappo_on_policy/`.

- Repository: https://github.com/marlbenchmark/on-policy
- Commit: `de66d7a4b23fac2513f56f96f73b3f5cb96695ac`
- License: upstream MIT; keep `mappo_on_policy/LICENSE` with redistributed code.
- This task imports the original R_MAPPOPolicy, R_Actor, R_Critic and R_MAPPO.
- The E45-I registration fingerprints every upstream Python file and license.

Restore the source with `git clone https://github.com/marlbenchmark/on-policy.git mappo_on_policy` followed by `git -C mappo_on_policy checkout de66d7a4b23fac2513f56f96f73b3f5cb96695ac` from this directory. The project adapter inserts that directory into its own Python path; no editable upstream install is required.

Execution environment: `/home/sjc/miniconda3/envs/dm_env_py311`. Existing PyTorch, NumPy and Gym are reused; `absl-py==2.5.1` was added because upstream package initialization imports `absl.flags`. The configured mirror failed TLS; installation succeeded from the official PyPI index. No upstream source patch was applied. CUDA workers set the current device before calling upstream Gaussian code.

The OPEN repository was separately read at commit `c1041e822e3e0fafbaad687de4e5093db0742e0b`. Its complete simulator requires Isaac Sim and pinned torchrl/tensordict dependencies. E45-I does not install that simulator or claim its APF adapter is a full OPEN reproduction.
