
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

## Error Messages

If you get an error message such as: 

```No such file or directory: '/home/ben/projects/decision_making/saved/example9/model_value_l0.pt```

it means that the solver tried to query a neural network oracle that does not exist. You can either disable neural network search or create a new model. 

To disable the neural network search, change `oracles_on` in `param.py` to `oracles_on = False`

To create a model, run `python train.py`. After training, you can query the newly created model (in `./current/models/`) by changing the `dirname` parameter in `param.py` to the corresponding location.
