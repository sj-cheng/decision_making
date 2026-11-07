#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
export TZ=Asia/Shanghai
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
E46_PYTHON=/home/sjc/miniconda3/envs/dm_env_py311/bin/python
E46_OUTPUT=/data2/sjc/decision_making/current/e46_geometry_teacher_20261105
mkdir -p -- "$E46_OUTPUT"
set +e
"$E46_PYTHON" -u code/run_e46_geometry_teacher.py --output "$E46_OUTPUT" >> "$E46_OUTPUT/console.log" 2>&1
E46_RUN_STATUS=$?
"$E46_PYTHON" -u code/analyze_e46_experiments.py --output "$E46_OUTPUT" >> "$E46_OUTPUT/analysis.log" 2>&1
E46_ANALYSIS_STATUS=$?
set -e
if [ "$E46_RUN_STATUS" -ne 0 ]; then
    exit "$E46_RUN_STATUS"
fi
exit "$E46_ANALYSIS_STATUS"
