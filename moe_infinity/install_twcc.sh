# ml list:   1) cuda/12.6   2) miniconda3/24.11.1   3) gcc/11.5.0
git clone https://github.com/NVIDIA/cutlass.git $HOME/cutlass
uv venv --python 3.10
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install wheel
uv pip install -e . --no-build-isolation
