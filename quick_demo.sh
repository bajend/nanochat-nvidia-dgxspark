#!/bin/bash

# Quick Demo Script for DGX Spark nanochat
# This script runs a minimal version for testing/demonstration
# NOT for production training - use dgx_spark_run.sh for that

echo "=========================================="
echo "DGX Spark nanochat - Quick Demo"
echo "=========================================="
echo ""
echo "This demo will:"
echo "1. Set up the environment"
echo "2. Build the tokenizer"
echo "3. Run minimal training (for testing)"
echo "4. Launch interactive chat"
echo ""
echo "Note: This is a DEMO. For full training, use dgx_spark_run.sh"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Setup
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat-demo"
mkdir -p $NANOCHAT_BASE_DIR

echo ""
echo "[1/5] Installing dependencies..."
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

echo ""
echo "[2/5] Installing Rust and building tokenizer..."
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

echo ""
echo "[3/5] Downloading minimal dataset..."
python -m nanochat.dataset -n 2  # Just 2 shards for demo

echo ""
echo "[4/5] Training tokenizer (this may take a few minutes)..."
python -m scripts.tok_train --max_chars=500000000  # 500M chars instead of 2B

echo ""
echo "[5/5] Demo complete!"
echo ""
echo "=========================================="
echo "What to do next:"
echo "=========================================="
echo ""
echo "1. For FULL TRAINING (8-12 hours):"
echo "   bash dgx_spark_run.sh"
echo ""
echo "2. To test the tokenizer:"
echo "   python -m scripts.tok_eval"
echo ""
echo "3. To train a tiny model for testing (NOT production):"
echo "   python -m scripts.base_train -- --depth=8 --device_batch_size=2 --num_iterations=100"
echo ""
echo "4. Read the documentation:"
echo "   cat README.md"
echo "   cat DGX_SPARK_SETUP.md"
echo "   cat QUICK_REFERENCE.md"
echo ""
echo "=========================================="
echo ""
echo "Environment is ready! Virtual environment is activated."
echo ""
