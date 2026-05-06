#!/usr/bin/env bash
set -e

{
  echo "===== main_qhybrid_matched_d_nophase ====="
  grep "Final peak GPU memory during training" outputs/manual_logs/main_qhybrid_nophase_stdout.txt || true
  grep "Training complete. Best val loss" outputs/manual_logs/main_qhybrid_nophase_stdout.txt || true
  cat outputs/main_qhybrid_matched_d_nophase/metrics/test_eval.txt || true

  echo
  echo "===== tinystories_qhybrid_nophase ====="
  grep "Final peak GPU memory during training" outputs/manual_logs/tinystories_qhybrid_nophase_stdout.txt || true
  grep "Training complete. Best val loss" outputs/manual_logs/tinystories_qhybrid_nophase_stdout.txt || true
  cat outputs/tinystories_qhybrid_nophase/metrics/test_eval.txt || true

  echo
  echo "===== pile10k_qhybrid_nophase ====="
  grep "Final peak GPU memory during training" outputs/manual_logs/pile10k_qhybrid_nophase_stdout.txt || true
  grep "Training complete. Best val loss" outputs/manual_logs/pile10k_qhybrid_nophase_stdout.txt || true
  cat outputs/pile10k_qhybrid_nophase/metrics/test_eval.txt || true

  echo
  echo "===== smallc4_qhybrid_nophase ====="
  grep "Final peak GPU memory during training" outputs/manual_logs/smallc4_qhybrid_nophase_stdout.txt || true
  grep "Training complete. Best val loss" outputs/manual_logs/smallc4_qhybrid_nophase_stdout.txt || true
  cat outputs/smallc4_qhybrid_nophase/metrics/test_eval.txt || true
} | tee outputs/manual_logs/qhybrid_nophase_summary.txt