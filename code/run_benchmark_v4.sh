#!/bin/bash
# final_benchmark_v4: 5-method round-robin, 80 CPU workers, MAPPO on CPU.
# CUDA_VISIBLE_DEVICES must stay empty: each worker holding a CUDA context
# costs ~478MB, and 80 workers would OOM a 24GB GPU mid-run.
cd /home/sjc/decision_making/code || exit 1
exec env \
  CUDA_VISIBLE_DEVICES="" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/mpl \
  /home/sjc/miniconda3/envs/dm_env_py311/bin/python benchmark_baselines.py \
    --seeds 5 --large-seeds 1 --episodes 200 \
    --search-only-budget 3000 --search-learning-budget 1000 \
    --large-budget 50000 --beta-policy 0.5 --beta-value 0.5 \
    --parallel --workers 80 --chunk-episodes 25 --progress-step-interval 100 \
    --round-robin \
    --methods "Policy Only" "Search Only" "Search + Learning" MAPPO "Large-budget MCTS" \
    --mappo-checkpoint ../current/baselines/mappo_ego_50k.pt \
    --mappo-device auto \
    --output ../current/benchmarks/final_benchmark_v6
