#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/verify_extra_addon.py
# Reuse the working environment. Never reinstall torch, CUDA, NumPy or Mamba here.
python - <<'PY'
import sys,importlib.metadata as m
from pathlib import Path
if sys.version_info[:2] != (3,10):raise SystemExit('Activate the existing tempograph-compare Python 3.10 environment.')
names=['torch','numpy','scipy','librosa','numba','setuptools','pytorch-lightning','torchmetrics','mamba-ssm','causal-conv1d','triton','huggingface-hub','transformers']
lines=[name+'=='+m.version(name) for name in names]
p=Path('downloads/extra_preserve_versions.txt');p.parent.mkdir(exist_ok=True);p.write_text('\n'.join(lines)+'\n')
print('Protecting existing versions:\n'+'\n'.join(lines))
PY
python -m pip install --upgrade-strategy only-if-needed -c downloads/extra_preserve_versions.txt 'accelerate==1.8.1' 'wandb==0.19.11' 'p-tqdm==1.4.2'
mkdir -p assets/flowerdance assets/finenet
python - <<'PY'
import torch,accelerate,wandb,p_tqdm
from mamba_ssm import Mamba
m=Mamba(d_model=16,d_state=16,d_conv=4,expand=2).cuda().eval()
with torch.no_grad():y=m(torch.randn(1,32,16,device='cuda'))
torch.cuda.synchronize()
assert torch.isfinite(y).all().item()
print('PASS: add-on dependencies and Mamba GPU smoke test. Checkpoint loading still needs checking.')
PY
