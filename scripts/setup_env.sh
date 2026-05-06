#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${1:-qbypassgpt}

conda create -n "${ENV_NAME}" python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

pip install -r requirements.txt
pip install -e .

echo "Environment ${ENV_NAME} is ready."
