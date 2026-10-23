#!/usr/bin/env bash
set -euo pipefail
# Activate the comparison environment BEFORE this script. Requires compatible CUDA toolkit/nvcc.
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
export MAX_JOBS=4
command -v nvcc >/dev/null || { echo 'nvcc missing: use an AutoDL CUDA 12.8 development image or install matching toolkit; keep your uploaded files on autodl-tmp.'; exit 1; }
nvcc --version
python -m pip install --no-build-isolation 'causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@857842b10546d86aaf05c52570a51fba2eec04d2'
python -m pip install --no-build-isolation 'mamba-ssm @ git+https://github.com/state-spaces/mamba.git@10b5d6358f27966f6a40e4bf0baa17a460688128'
python -c "import torch; from mamba_ssm import Mamba; m=Mamba(d_model=16,d_state=16,d_conv=4,expand=2).cuda(); print(m(torch.randn(1,32,16,device='cuda')).shape)"
python -m pip freeze > environment_actual.txt
