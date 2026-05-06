#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

python -m q_bypass_gpt.train \
  --config configs/main_gptneo125_wikitext.yaml