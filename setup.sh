#!/bin/bash
#
# This script automates the full setup and training launch for nanochat
# on an aarch64 machine with CUDA 13.0 and Ubuntu 24.04.
#
# It includes all fixes from our debugging session:
# 1. Installs 'clang' system dependency for Triton.
# 2. Creates the correct 'pyproject.toml' to target CUDA 13.0.
# 3. Runs 'uv sync --extra gpu' *twice* to fix the PyTorch CPU downgrade bug.
# 4. Sets all environment variables for Triton/CUDA.
# 5. Runs the final 'torchrun' training command.
#
# Usage:
# 1. Save this file as setup_and_train_nanochat.sh
# 2. Make it executable: chmod +x setup_and_train_nanochat.sh
# 3. Run it: ./setup_and_train_nanochat.sh
#

# Exit immediately if any command fails
set -e

echo "--- [Step 1/11] Cloning nanochat repo ---"
git clone https://github.com/karpathy/nanochat.git
cd nanochat

echo "--- [Step 2/11] Creating pyproject.toml for CUDA 13.0 ---"
# This 'cat' command creates the entire file in one go.
cat << 'EOF' > pyproject.toml
[project]
name = "nanochat"
version = "0.1.0"
description = "the minimal full-stack ChatGPT clone"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "datasets>=4.0.0",
    "fastapi>=0.117.1",
    "files-to-prompt>=0.6",
    "numpy==1.26.4",
    "psutil>=7.1.0",
    "regex>=2025.9.1",
    "setuptools>=80.9.0",
    "tiktoken>=0.11.0",
    "tokenizers>=0.22.0",
    "torch>=2.9.0",
    "triton>=3.5.0",
    "uvicorn>=0.36.0",
    "wandb>=0.21.3",
]

[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[tool.maturin]
module-name = "rustbpe"
bindings = "pyo3"
python-source = "."
manifest-path = "rustbpe/Cargo.toml"

[dependency-groups]
dev = [
    "maturin>=1.9.4",
    "pytest>=8.0.0",
]

[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

# target torch to cuda 13.0 or CPU
[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-cu130", extra = "gpu" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[project.optional-dependencies]
cpu = [
    "torch>=2.9.0",
]
gpu = [
    "torch>=2.9.0",
]

[tool.uv]
conflicts = [
    [
        { extra = "cpu" },
        { extra = "gpu" },
    ],
]
EOF

echo "--- [Step 3/11] Installing system dependencies (uv, clang) ---"
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt-get update
sudo apt-get install -y clang

echo "--- [Step 4/11] Creating venv and installing Python deps (GPU Pass 1) ---"
[ -d ".venv" ] || uv venv
# Run 'uv sync' using the uv binary *inside* the new venv
# This is the first pass to get the GPU-enabled PyTorch
.venv/bin/uv sync --extra gpu

echo "--- [Step 5/11] Installing Rust and building tokenizer ---"
# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# Source the new env variables for the *current* script's shell
source "$HOME/.cargo/env"
# Build the rustbpe Tokenizer.
# WARNING: This command will downgrade PyTorch to CPU. We fix this next.
.venv/bin/uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

echo "--- [Step 6/11] Re-installing PyTorch GPU (Fixing downgrade) ---"
# This is the CRITICAL FIX.
# 'uv run' from the previous step reverts PyTorch to the CPU version.
# We must re-sync to get the GPU version back.
.venv/bin/uv sync --extra gpu

echo "--- [Step 7/11] Verifying PyTorch GPU installation ---"
.venv/bin/python -c "import torch; \
    assert torch.cuda.is_available(), 'PyTorch CUDA is NOT available!'; \
    print('PyTorch CUDA check: OK (', torch.cuda.get_device_name(0), ')')"

echo "--- [Step 8/11] Downloading and training tokenizer ---"
# Download the training dataset
.venv/bin/python -m nanochat.dataset -n 240
# Train the tokenizer
.venv/bin/python -m scripts.tok_train --max_chars=2000000000
# Evaluate the tokenizer
.venv/bin/python -m scripts.tok_eval

echo "--- [Step 9/11] Downloading eval bundle ---"
curl -L -o eval_bundle.zip https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
unzip -q eval_bundle.zip
rm eval_bundle.zip
# Make sure the cache directory exists
mkdir -p "$HOME/.cache/nanochat"
# Move the bundle, but don't error if it's already there
if [ ! -d "$HOME/.cache/nanochat/eval_bundle" ]; then
    mv eval_bundle "$HOME/.cache/nanochat/"
else
    echo "eval_bundle already exists. Removing temp download."
    rm -r eval_bundle
fi

echo "--- [Step 10/11] Setting CUDA/Triton environment variables ---"
# These exports are set *for this script* and will be passed to torchrun
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}
echo "TRITON_PTXAS_PATH set to: $TRITON_PTXAS_PATH"

echo "--- [Step 11/11] Starting pre-training ---"
# We're all set. Launch the training.
.venv/bin/torchrun --standalone --nproc_per_node=gpu -m scripts.base_train -- --depth=20

echo "--- Training complete (or script finished) ---"
