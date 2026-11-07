
This repository contains code for the following publication, please cite our work if you use this repo:

B. Riviere, W. Hönig, M. Anderson, S-J. Chung. "Neural Tree Expansion for Multi-Robot Planning in Non-Cooperative Environments" in IEEE Robotics and Automation Letters (RA-L) June 2021. 


## Dependencies:
Developed on Ubuntu 20.04. Python dependencies in `~/environment.yml`, can be batch installed with: 

```
conda env create -f environment.yml
```
If package install fails, try removing specific versions in `.yml` file: e.g. if 
```
ResolvePackageNotFound: 
  - libgfortran-ng==7.5.0=hdf63c60_6
```  
change `libgfortran-ng==7.5.0=hdf63c60_6` to `libgfortran-ng=7.5.0`. 

Then:
```
conda activate dm_env
```

## Compiling:
from `~/code/cpp`:
```
mkdir build
cd build
cmake -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release ..
make
```

## Script Examples:  
Run individual problems and solvers from `~\code` by modifying `code/param.py` and then:
```
python run.py
```
Run batch examples from `~\code\tests`: 
```
python regression.py
```
Run waypoint planning from `~\code\tests`: 
```
python waypoint_planning.py
```
Train neural networks by modifying parameters in `code/train.py` then, from `~\code`:
```
python train.py
```

## Baseline benchmark

For the obstacle-map benchmark with V7 models, updated MAPPO, 80 CPU workers,
and resumable episode chunks, run from the repository root:

```bash
bash code/run_benchmark_obstacles.sh --workers 80
```

Use `OBSTACLE_LAYOUT=four` for four obstacles (default: two). Full commands,
protocol details, and smoke checks are in
[`code/BENCHMARK_OBSTACLES.md`](code/BENCHMARK_OBSTACLES.md).

The reproducible first-stage benchmark is implemented in
`code/benchmark_baselines.py`. It evaluates Random, Greedy, Policy Only,
Search Only, Search + Learning, MAPPO and Large-budget MCTS against fixed
strong opponents, and writes both per-episode CSV files and summary files.
The benchmark keeps the current Example8 global/no-obstacle protocol: direct
propagation with the active Example8 speed and spawn settings; the protocol
name is stored in `config.json`.
Per-episode records now contain separate `evader_*_latency_ms` and
`pursuer_*_latency_ms` statistics.  The legacy `mean_latency_ms` fields are
the complete joint action-call latency and should not be reported as either
method's standalone latency.  Search, policy, and MAPPO solvers are invoked
with the relevant team mask so unused actions are not computed and timed.
Run a small smoke test with:

```
MPLCONFIGDIR=/tmp/mpl python benchmark_baselines.py --seeds 1 --large-seeds 1 --episodes 3 \
  --methods Random Greedy "Policy Only" "Search Only" "Search + Learning" \
  --search-only-budget 20 --search-learning-budget 10 --large-budget 100 \
  --beta-policy 0.5 --beta-value 0.5 --output ../current/benchmarks/smoke
```

For the fastest latency/schema validation, omit the expensive reference and
MAPPO matchups:

```
MPLCONFIGDIR=/tmp/mpl python benchmark_baselines.py --seeds 1 --episodes 2 \
  --search-only-budget 20 --search-learning-budget 10 --round-robin \
  --methods "Policy Only" "Search Only" "Search + Learning" \
  --output /tmp/benchmark_smoke_current --no-progress
```

Train the MAPPO checkpoint before including that method:

```
python train_mappo.py --episodes 512 \
  --output ../current/baselines/mappo_global.pt
```

The full first-stage run uses the paper budget and five seeds:

```
MPLCONFIGDIR=/tmp/mpl python benchmark_baselines.py --seeds 5 --large-seeds 1 --episodes 200 \
  --search-only-budget 3000 --search-learning-budget 1000 --large-budget 50000 \
  --beta-policy 0.5 --beta-value 0.5 \
  --round-robin \
  --methods "Policy Only" "Search Only" "Search + Learning" MAPPO "Large-budget MCTS" \
  --mappo-checkpoint ../current/baselines/mappo_ego_50k.pt \
  --output ../current/benchmarks/full
```

Independent matchups can be distributed across CPU worker processes. Use a
small worker count first because each MCTS matchup is CPU-intensive:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
python benchmark_baselines.py --seeds 5 --large-seeds 1 --episodes 200 \
  --search-only-budget 3000 --search-learning-budget 1000 --large-budget 50000 \
  --beta-policy 0.5 --beta-value 0.5 --workers 8 --parallel \
  --round-robin \
  --methods "Policy Only" "Search Only" "Search + Learning" MAPPO "Large-budget MCTS" \
  --mappo-checkpoint ../current/baselines/mappo_ego_50k.pt \
  --output ../current/benchmarks/full_parallel
```

In parallel mode the terminal progress bar counts environment steps for the
whole run, while its status line reports the active matchup, seed, episode,
step, active workers, and completed chunks.  Each completed chunk is also
printed as a separate line.  The default heartbeat is every five environment
steps; use `--progress-step-interval 1` when you need a heartbeat on every
step (with a small IPC overhead).

Any matchup containing Large-budget MCTS uses `--large-seeds` so that the
50,000-simulation reference does not multiply the full five-seed cost.  Other
matchups use `--seeds`.  The effective budgets, beta values, checkpoint paths,
and seed lists are saved to `config.json` in the output directory.

## Current research direction (2026-10-25)

The current plan is [geometry-guided neural tree expansion under local
occlusion](code/L45_RESEARCH_DIRECTIONS_20261025.md). First repair and audit
rollout legality and estimation reliability, then compare G, T and T+Geo.
Network training follows only if geometry-enhanced search and its teacher
provide additional value. The isolated reliability implementation is
`code/solvers/reliable_local_pursuit.py`; frozen experiments remain intact.
The first engineering audit is
[`current/l45_rollout_reliability_20261025/REPORT.md`](current/l45_rollout_reliability_20261025/REPORT.md).
The subsequent witnessed-recovery version is
`code/solvers/reliable_local_pursuit_v2.py`, with
[recovery and fixed-history coverage checks](current/l45_estimator_recovery_20261025/REPORT.md).
Recovery reliability and estimator coverage are reported separately; no new
neural training or deployment change follows from an engineering pass.
The subsequent [normal multimodal estimation validation](current/l45_multimodal_estimator_20261025/REPORT.md)
compares five estimators on fixed development and unseen histories. Its
candidate failed the frozen quality gates and was not adopted for search.
The later [wall-turn and geometric-export study](current/l45_corner_turn_estimator_20261025/REPORT.md)
passed its new-history estimator gates and provides a
[frozen shared research estimator](current/l45_corner_turn_estimator_20261025/frozen_estimator/README.md).
It preserves weighted-sample semantics and records initialization cost separately;
deployment, search-effect comparisons and neural training remain unchanged.
The subsequent [weighted search interface audit (2026-10-26)](current/l45_weighted_search_interface_v2_20261026/REPORT.md)
preserves frozen D and verifies weighted root sampling, its motion kernel,
legal simulation and exact replay on development histories. The full 50 ms
gate fails: first-occlusion graph reconstruction alone exceeds the budget.
Search-effect comparisons, independent efficacy confirmation and training
therefore remain gated; the initial failed geometry audit is also retained.
After explicit user authorization to evaluate the 1 s high-level research
period with a 200 ms controller budget, the
[frozen-D closed-loop study](current/l45_closed_loop_D_20261026/REPORT.md)
completed 72 development and 144 unseen-confirmation games. Confirmation
legal captures were G 42/48, T 38/48 and T+Geo 38/48. Neither T over G nor
T+Geo over T passed the preregistered additional-value test; T also had a
higher failure-penalized completion cost. No network training or deployment
change follows. The previous 50 ms failure remains a separate historical
result, and this study does not resolve that reconstruction bottleneck.
The subsequent [exact repeat and rollout diagnosis](current/l45_search_diagnosis_repeat_20261026/REPORT.md)
reran all 144 confirmation games with 144/144 identical scientific traces.
A common-root model experiment found that repeating a root displacement
undervalues the continuation of G compared with recomputing its geometric
action; extending the horizon alone did not reverse the action-value gaps.
This is a diagnostic on known histories, not an adopted search revision or
a new independent capture-benefit confirmation.
The [single-mechanism rollout-replanning study](current/l45_rollout_replan_exact_20261026/REPORT.md)
then completed 120 development and 240 new-confirmation games, keeping D
and the original G/T/T+Geo immutable. Exact geometry reuse was verified
bit-for-bit; an earlier slow execution prototype and its 62 completed
development games remain archived separately. Confirmation captures were
G 47/48, T 41/48, T+Geo 40/48, replanned T 43/48 and replanned T+Geo 40/48.
Neither revision passed the preregistered benefit tests. This modification
direction is stopped; no new modules, network training or deployment change
follow from the experiment.

## Local-observation search and RL

The obstacle/local-observation pipeline is documented in
[`code/SEARCH_RL.md`](code/SEARCH_RL.md). It adds anonymous team belief tracking,
root-sampled search, search distillation and PPO fine-tuning with independent
evaluation maps and resumable training. Build with `python code/build_local_rl.py`,
then launch `python -u code/run_search_rl_pipeline.py --workers 16`.
Outputs and progress are under `current/search_rl_v1/`; formal multi-seed runs
are gated by the pilot comparison. The new swept-collision protocol is opt-in.

## Error Messages

If you get an error message such as: 

```No such file or directory: '/home/ben/projects/decision_making/saved/example9/model_value_l0.pt```

it means that the solver tried to query a neural network oracle that does not exist. You can either disable neural network search or create a new model. 

To disable the neural network search, change `oracles_on` in `param.py` to `oracles_on = False`

To create a model, run `python train.py`. After training, you can query the newly created model (in `./current/models/`) by changing the `dirname` parameter in `param.py` to the corresponding location.
