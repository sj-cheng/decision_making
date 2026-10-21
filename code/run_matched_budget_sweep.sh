#!/usr/bin/env bash
set -euo pipefail

# Matched-budget sweep for the v5 direct-propagation protocol.
# Curves:
#   pursuit: Search Only(B)       vs Search + Learning(1000)
#   pursuit: Search + Learning(B) vs Search + Learning(1000)
#   evasion: Large-budget MCTS(LARGE_BUDGET) vs Search Only(B)
#   evasion: Large-budget MCTS(LARGE_BUDGET) vs Search + Learning(B)
#
# Quick pilot:
#   WORKERS=32 CHUNK_EPISODES=2 EPISODES=20 LARGE_SEEDS=1 ./run_matched_budget_sweep.sh
# WORKERS=1 runs serially for dedicated timing measurements.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/sjc/miniconda3/envs/dm_env_py311/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/current/benchmarks/matched_budget_parallel_v1}"
EPISODES="${EPISODES:-200}"
SEEDS="${SEEDS:-5}"
LARGE_SEEDS="${LARGE_SEEDS:-1}"
WARMUP="${WARMUP:-5}"
BETAS="${BETAS:-0.5}"
BUDGETS="${BUDGETS:-200 500 1000 3000 5000}"
LARGE_BUDGET="${LARGE_BUDGET:-50000}"
REFERENCE_BUDGET="${REFERENCE_BUDGET:-1000}"
WORKERS="${WORKERS:-32}"
CHUNK_EPISODES="${CHUNK_EPISODES:-5}"
CPU_THREADS="${CPU_THREADS:-1}"
NO_PROGRESS="${NO_PROGRESS:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
# Prevent each worker from opening another large BLAS/OpenMP thread pool.
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"

for setting in WORKERS CHUNK_EPISODES CPU_THREADS; do
  if ! [[ "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
    echo "$setting must be a positive integer" >&2
    exit 2
  fi
done

EXECUTION_ARGS=(--cpu-threads "$CPU_THREADS")
if (( WORKERS > 1 )); then
  EXECUTION_ARGS+=(--parallel --workers "$WORKERS" --chunk-episodes "$CHUNK_EPISODES")
  echo "Performance sweep: up to $WORKERS workers, $CHUNK_EPISODES episodes/chunk."
  echo "Concurrent latency samples are diagnostic; measure isolated latency with WORKERS=1."
else
  echo "Serial sweep: isolated solver timing requires no competing jobs."
fi
if [[ "$NO_PROGRESS" == 1 ]]; then
  EXECUTION_ARGS+=(--no-progress)
fi

mkdir -p "$OUTPUT_ROOT"
cd "$ROOT_DIR/code"

for BUDGET in $BUDGETS; do
  BASE_ARGS=(
    --seeds "$SEEDS"
    --large-seeds "$LARGE_SEEDS"
    --episodes "$EPISODES"
    --beta-policy "$BETAS"
    --beta-value "$BETAS"
    --timing-warmup-steps "$WARMUP"
    "${EXECUTION_ARGS[@]}"
  )

  "$PYTHON_BIN" benchmark_baselines.py "${BASE_ARGS[@]}" \
    --pursuer-method "Search Only" \
    --evader-method "Search + Learning" \
    --pursuer-budget "$BUDGET" \
    --evader-budget "$REFERENCE_BUDGET" \
    --output "$OUTPUT_ROOT/B${BUDGET}_pursuit_SO"

  "$PYTHON_BIN" benchmark_baselines.py "${BASE_ARGS[@]}" \
    --pursuer-method "Search + Learning" \
    --evader-method "Search + Learning" \
    --pursuer-budget "$BUDGET" \
    --evader-budget "$REFERENCE_BUDGET" \
    --output "$OUTPUT_ROOT/B${BUDGET}_pursuit_SL"

  "$PYTHON_BIN" benchmark_baselines.py "${BASE_ARGS[@]}" \
    --pursuer-method "Large-budget MCTS" \
    --evader-method "Search Only" \
    --pursuer-budget "$LARGE_BUDGET" \
    --evader-budget "$BUDGET" \
    --output "$OUTPUT_ROOT/B${BUDGET}_evasion_SO"

  "$PYTHON_BIN" benchmark_baselines.py "${BASE_ARGS[@]}" \
    --pursuer-method "Large-budget MCTS" \
    --evader-method "Search + Learning" \
    --pursuer-budget "$LARGE_BUDGET" \
    --evader-budget "$BUDGET" \
    --output "$OUTPUT_ROOT/B${BUDGET}_evasion_SL"
done

echo "Matched-budget sweep written to ${OUTPUT_ROOT}"
