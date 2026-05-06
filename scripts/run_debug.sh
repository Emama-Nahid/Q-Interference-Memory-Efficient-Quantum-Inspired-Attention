#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python -m q_bypass_gpt.train --config configs/debug_qbypass.yaml
