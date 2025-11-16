#!/bin/bash

# DGX Spark Optimized nanochat Training Script
# This script is optimized for the NVIDIA DGX Spark desktop AI supercomputer:
# - Single Blackwell GPU (no multi-GPU setup)
# - 128GB unified memory
# - ARM architecture support
# - Smaller model/batch sizes optimized for desktop training

# Expected runtime: ~8-12 hours on DGX Spark
# This will produce a smaller but capable ChatGPT-style model

# 1) Example launch (simplest):
# bash dgx_spark_run.sh
# 2) Example launch in a screen session:
# screen -L -Logfile dgx_spark.log -S dgx_spark bash dgx_spark_run.sh
# 3) Example launch with wandb logging:
# WANDB_RUN=dgx_spark screen -L -Logfile dgx_spark.log -S dgx_spark bash dgx_spark_run.sh

# Default intermediate artifacts directory
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies with GPU support
uv sync --extra gpu
# activate venv
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup (optional)
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Initialize report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo (ARM-compatible)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download initial dataset shards (~800MB)
python -m nanochat.dataset -n 8
# Download additional shards in background (optimized for smaller model)
# DGX Spark: Using d16 model (~300M params)
# Chinchilla: 300M * 20 = 6B tokens needed
# At 4.8 chars/token: 6B * 4.8 = 28.8B chars
# At 250M chars/shard: 28.8B / 250M = 115 shards
# Round up to 120 for safety
python -m nanochat.dataset -n 120 &
DATASET_DOWNLOAD_PID=$!

# Train the tokenizer with vocab size 65536 on ~2B characters
python -m scripts.tok_train --max_chars=2000000000
# Evaluate the tokenizer
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# DGX Spark configuration: Single GPU, smaller model
# Using depth=16 for ~300M parameter model (fits well in DGX Spark memory)
# Reduced device_batch_size to 4 for single GPU training
python -m scripts.base_train -- --depth=16 --device_batch_size=4 --run=$WANDB_RUN

# Evaluate the model
python -m scripts.base_loss
python -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining

# Download synthetic identity conversations
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# Run midtraining (single GPU, smaller batch)
python -m scripts.mid_train -- --device_batch_size=4 --run=$WANDB_RUN
python -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# Supervised Finetuning

python -m scripts.chat_sft -- --device_batch_size=4 --run=$WANDB_RUN
python -m scripts.chat_eval -- -i sft

# Chat with the model over CLI
# python -m scripts.chat_cli -p "Why is the sky blue?"

# Or chat via WebUI (recommended)
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Reinforcement Learning (Optional)
# Commented out by default to save time

# python -m scripts.chat_rl -- --device_batch_size=4 --run=$WANDB_RUN
# python -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report
python -m nanochat.report generate

echo ""
echo "======================================================================"
echo "DGX Spark training complete!"
echo "Your nanochat model is ready to use."
echo ""
echo "To chat via CLI:"
echo "  python -m scripts.chat_cli"
echo ""
echo "To chat via Web UI:"
echo "  python -m scripts.chat_web"
echo "  Then visit http://localhost:8000 in your browser"
echo ""
echo "See report.md for detailed metrics and evaluation results."
echo "======================================================================"
