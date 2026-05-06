#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS=${1:-1}

if [ "$NUM_GPUS" -eq 1 ]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  python -m q_bypass_gpt.train --config configs/main_baseline.yaml
else
  torchrun --standalone --nproc_per_node=${NUM_GPUS} \
    -m q_bypass_gpt.train --config configs/main_baseline.yaml
fi
