#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

mkdir -p outputs/main_opt125_wikitext/logs

rm -f outputs/main_opt125_wikitext/logs/opt125_wikitext.log

CUDA_VISIBLE_DEVICES=1 python -m q_bypass_gpt.train \
  --config configs/main_opt125_wikitext.yaml \
  2>&1 | tee outputs/main_opt125_wikitext/logs/opt125_wikitext.log