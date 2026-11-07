#!/usr/bin/env bash
# V7 checkpoints/budgets, fixed obstacles, CPU episode-chunk parallelism.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$script_dir"

if [[ -n "${PYTHON:-}" ]]; then
  python_bin="$PYTHON"
elif [[ -x "$HOME/miniconda3/envs/dm_env_py311/bin/python" ]]; then
  python_bin="$HOME/miniconda3/envs/dm_env_py311/bin/python"
else
  python_bin="python3"
fi

layout="${OBSTACLE_LAYOUT:-two}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export PYTHONUNBUFFERED=1

# Explicit CPU inference lets all workers run without creating CUDA contexts.
# User arguments come last so --workers, --episodes, --output, etc. can override.
exec "$python_bin" "$script_dir/benchmark_baselines.py" \
  --obstacle-layout "$layout" \
  --seeds 5 --large-seeds 1 --episodes 200 \
  --search-only-budget 3000 --search-learning-budget 1000 --large-budget 50000 \
  --beta-policy 0.5 --beta-value 0.1 \
  --model-dir "$repo_root/current/memory_observation_v3/models" \
  --model-iteration 45 \
  --mappo-checkpoint "$repo_root/current/baselines/mappo_50k_v4.pt" \
  --mappo-device cpu --cpu-threads 1 \
  --parallel --workers "${WORKERS:-80}" --chunk-episodes 10 \
  --progress-step-interval 10 \
  --round-robin \
  --methods "Policy Only" "Search Only" "Search + Learning" MAPPO "Large-budget MCTS" \
  --output "$repo_root/current/benchmarks/obstacles_${layout}_v1" \
  "$@"
