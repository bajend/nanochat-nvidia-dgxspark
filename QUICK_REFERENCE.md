# Quick Reference Guide - DGX Spark nanochat

## One-Liner Commands

### Installation & Setup
```bash
# Clone repository
git clone https://github.com/bajend/nanochat-nvidia-dgxspark.git
cd nanochat-nvidia-dgxspark

# Full training run
bash dgx_spark_run.sh

# Training with logging to file
screen -L -Logfile dgx_spark.log -S dgx_spark bash dgx_spark_run.sh
```

### Chat with Your Model
```bash
# Activate environment first
source .venv/bin/activate

# Web UI (recommended)
python -m scripts.chat_web

# CLI chat
python -m scripts.chat_cli

# Single query
python -m scripts.chat_cli -p "Explain quantum computing"
```

### Monitoring
```bash
# Watch training log
tail -f dgx_spark.log

# GPU status
watch -n 1 nvidia-smi

# System resources
htop

# Disk usage
du -sh ~/.cache/nanochat/*
```

### Screen Session Management
```bash
# List sessions
screen -ls

# Reattach to session
screen -r dgx_spark

# Detach from session (inside screen)
Ctrl-a d

# Kill session
screen -X -S dgx_spark quit
```

## File Locations

```
~/.cache/nanochat/          # Main cache directory
├── data/                   # Downloaded dataset shards
├── checkpoints/            # Training checkpoints
├── tokenizer/              # Trained tokenizer files
└── identity_conversations.jsonl  # Personality data
```

## Quick Configuration Changes

### Model Size (in dgx_spark_run.sh)

**Small (150M params, ~4-6 hours):**
```bash
python -m scripts.base_train -- --depth=12 --device_batch_size=8 --run=$WANDB_RUN
```

**Medium (300M params, ~8-12 hours) - Default:**
```bash
python -m scripts.base_train -- --depth=16 --device_batch_size=4 --run=$WANDB_RUN
```

**Large (450M params, ~12-16 hours):**
```bash
python -m scripts.base_train -- --depth=18 --device_batch_size=2 --run=$WANDB_RUN
```

### Data Download Amount

```bash
# Less data (faster, less capable)
python -m nanochat.dataset -n 60

# Default
python -m nanochat.dataset -n 120

# More data (slower, more capable)
python -m nanochat.dataset -n 240
```

## Common Issues & Solutions

### Out of Memory
```bash
# Reduce batch size
--device_batch_size=2
# Or reduce model size
--depth=12
```

### Dataset Download Fails
```bash
# Retry manually
python -m nanochat.dataset -n 120

# Check connection
ping huggingface.co
```

### Rust Build Fails
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Rebuild tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Python Package Issues
```bash
# Reinstall dependencies
uv sync --extra gpu

# Or force reinstall
rm -rf .venv
uv venv
uv sync --extra gpu
```

## Performance Tips

### Maximize GPU Utilization
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check batch size isn't too small
# Increase if GPU memory allows
--device_batch_size=8
```

### Speed Up Dataset Download
```bash
# Download in background while tokenizer trains
python -m nanochat.dataset -n 120 &

# Use faster storage (NVMe SSD)
export NANOCHAT_BASE_DIR="/fast/ssd/nanochat"
```

### Optimize ARM Performance
```bash
# Set CPU threads
export OMP_NUM_THREADS=1

# Enable PyTorch optimizations
export TORCH_COMPILE=1
```

## Testing Commands

### Test Tokenizer
```bash
source .venv/bin/activate
python -m pytest tests/test_rustbpe.py -v
```

### Test Engine
```bash
python -m pytest tests/test_engine.py -v
```

### Quick Inference Test
```bash
python -m scripts.chat_cli -p "Hello, who are you?"
```

## Environment Variables

```bash
# Cache directory
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

# WandB tracking
export WANDB_RUN="my_experiment"
export WANDB_PROJECT="nanochat-dgx-spark"

# CPU threads
export OMP_NUM_THREADS=1

# CUDA settings
export CUDA_LAUNCH_BLOCKING=0
```

## Cleanup Commands

```bash
# Remove datasets only (keep models)
rm -rf ~/.cache/nanochat/data/

# Remove old checkpoints
rm -rf ~/.cache/nanochat/checkpoints/

# Remove everything (fresh start)
rm -rf ~/.cache/nanochat/

# Remove virtual environment
rm -rf .venv/
```

## Useful Aliases (add to ~/.bashrc)

```bash
# Add to ~/.bashrc
alias nanochat-train='cd ~/nanochat-nvidia-dgxspark && bash dgx_spark_run.sh'
alias nanochat-chat='cd ~/nanochat-nvidia-dgxspark && source .venv/bin/activate && python -m scripts.chat_web'
alias nanochat-cli='cd ~/nanochat-nvidia-dgxspark && source .venv/bin/activate && python -m scripts.chat_cli'
alias nanochat-status='nvidia-smi && du -sh ~/.cache/nanochat/*'
alias nanochat-logs='tail -f ~/nanochat-nvidia-dgxspark/dgx_spark.log'
```

## Training Stages Timeline

```
00:00 - 00:30   Tokenizer training
00:30 - 05:00   Base pretraining
05:00 - 07:00   Midtraining
07:00 - 09:00   Supervised fine-tuning
09:00 - 09:30   Evaluation
09:30 - 12:00   Optional: Reinforcement learning
```

## Estimated Resource Usage

```
Disk Space:
- Initial clone:     ~10 MB
- Dependencies:      ~5 GB
- Datasets:         ~12 GB
- Checkpoints:       ~5 GB
- Total:            ~22 GB

Memory:
- Training:         ~60 GB GPU + System
- Inference:        ~10 GB

Network:
- Dataset download: ~12 GB
- Dependencies:      ~2 GB
```

## Advanced Usage

### Resume from Checkpoint
```bash
# Training automatically resumes from latest checkpoint
# Just re-run the script
bash dgx_spark_run.sh
```

### Export Model
```bash
# Model is saved in:
~/.cache/nanochat/checkpoints/

# Copy to deploy elsewhere
cp -r ~/.cache/nanochat/checkpoints/final_model /path/to/deployment/
```

### Custom Training Data
```bash
# Replace identity_conversations.jsonl with your data
# See: https://github.com/karpathy/nanochat/discussions/139
```

## Getting Help

1. Check README.md
2. Check DGX_SPARK_SETUP.md
3. Check COMPARISON.md
4. Original nanochat: https://github.com/karpathy/nanochat
5. Open an issue on GitHub

## Version Info

```bash
# Check versions
python --version
nvidia-smi
rustc --version
uv --version
```

## Benchmarking

```bash
# Run evaluations only
source .venv/bin/activate
python -m scripts.chat_eval -- -i sft

# Specific benchmark
python -m scripts.chat_eval -- -i sft -a GSM8K
```

---

**Quick Start Reminder:**
```bash
bash dgx_spark_run.sh  # Start training
# Wait 8-12 hours
python -m scripts.chat_web  # Chat with your model!
```
