#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
export HF_HOME=/root/autodl-tmp/huggingface
export TORCH_HOME=/root/autodl-tmp/torch-cache
ENV_DIR=/root/autodl-tmp/envs/tempograph-compare
source /root/miniconda3/etc/profile.d/conda.sh
if [ ! -x "$ENV_DIR/bin/python" ]; then conda create -y -p "$ENV_DIR" python=3.10; fi
conda activate "$ENV_DIR"
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.26.4 scipy==1.13.1 matplotlib==3.9.4 pillow==11.1.0 librosa==0.11.0 soundfile==0.13.1 numba==0.60.0 pyyaml omegaconf==2.3.0 einops==0.8.1 einx vector-quantize-pytorch==1.23.2 kornia==0.8.1 tqdm gdown py7zr smplx==0.1.28 pyrender==0.1.45 trimesh==4.6.6 opencv-python-headless==4.11.0.86 pytorch-lightning==1.9.5 torchmetrics==0.11.4 transformers==4.49.0 tensorboard rich easydict ninja packaging essentia==2.1b6.dev1110
python -m pip install --no-build-isolation chumpy==0.70
python scripts/link_assets.py
python -m pip freeze > environment_actual.txt
python -c "import torch; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0)); print((torch.ones(2,device='cuda')+1).cpu())"
printf '\nEnvironment prepared. MEGADance also requires: bash scripts/install_mamba.sh\n'
