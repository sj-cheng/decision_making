#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
export TZ=Asia/Shanghai
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
L5_FOLLOWUP_PYTHON=/home/sjc/miniconda3/envs/dm_env_py311/bin/python
L5_FOLLOWUP_OUTPUT=/data2/sjc/decision_making/current/l5_followup_20261107
mkdir -p -- "$L5_FOLLOWUP_OUTPUT"
exec "$L5_FOLLOWUP_PYTHON" -u code/scale_l5_followup.py launch --output "$L5_FOLLOWUP_OUTPUT" >> "$L5_FOLLOWUP_OUTPUT/console.log" 2>&1
