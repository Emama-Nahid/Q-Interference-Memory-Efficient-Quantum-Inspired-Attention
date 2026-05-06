#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/ablation_no_phase.yaml}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python -m q_bypass_gpt.train --config "${CONFIG}"
