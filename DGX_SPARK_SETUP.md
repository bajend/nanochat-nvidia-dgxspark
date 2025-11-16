# DGX Spark Setup Guide

This guide provides detailed instructions for setting up and running nanochat on your NVIDIA DGX Spark.

## System Requirements

### Hardware
- NVIDIA DGX Spark desktop AI supercomputer
- Minimum 50GB free disk space
- Internet connection for downloading datasets

### Software
- DGX OS (NVIDIA's custom Linux distribution)
- Python 3.10 or higher (should be pre-installed)
- Rust toolchain (will be installed by the script)
- CUDA drivers (should be pre-installed on DGX OS)

## Initial Setup

### 1. Clone the Repository

```bash
git clone https://github.com/bajend/nanochat-nvidia-dgxspark.git
cd nanochat-nvidia-dgxspark
```

### 2. Verify CUDA and GPU

Before starting, verify your GPU is properly detected:

```bash
nvidia-smi
```

You should see your Blackwell GPU listed with available memory (~128GB unified).

### 3. Check Python Version

```bash
python --version
```

Ensure you have Python 3.10 or higher.

## Running Your First Training

### Option 1: Quick Test (CPU/MPS mode for testing)

If you want to test the pipeline without full GPU training:

```bash
# This will run a minimal version for testing
bash dgx_spark_run.sh
```

### Option 2: Full Training Run

For the complete training pipeline (~8-12 hours):

```bash
# Using screen to run in background
screen -L -Logfile dgx_spark.log -S dgx_spark bash dgx_spark_run.sh
```

**Screen session commands:**
- Detach: `Ctrl-a d`
- Reattach: `screen -r dgx_spark`
- List sessions: `screen -ls`
- Kill session: `screen -X -S dgx_spark quit`

### Monitoring Progress

While training is running, you can monitor progress:

```bash
# Watch the log file
tail -f dgx_spark.log

# Or if you're in the screen session, you'll see live output
```

## Training Stages

The training pipeline consists of several stages:

### Stage 1: Tokenizer Training (~30 minutes)
- Downloads ~800MB of training data
- Trains a BPE tokenizer with 65,536 vocabulary
- Builds Rust tokenizer library

### Stage 2: Base Pretraining (~4-6 hours)
- Downloads additional dataset shards (~12GB total)
- Trains a depth=16 Transformer model (~300M parameters)
- Runs on raw text data from FineWeb-EDU

### Stage 3: Midtraining (~1-2 hours)
- Teaches the model conversational structure
- Trains on synthetic conversations
- Adds special tokens for chat formatting

### Stage 4: Supervised Fine-Tuning (~1-2 hours)
- Fine-tunes for specific tasks
- Improves instruction following
- Enhances chat quality

### Stage 5: Evaluation
- Runs benchmark tasks (ARC, MMLU, GSM8K, HumanEval, etc.)
- Generates comprehensive report

### Optional: Reinforcement Learning (~2 hours)
- Currently commented out by default
- Can be enabled for math reasoning improvements

## After Training

### Inference and Chat

Once training completes, activate the virtual environment and start chatting:

```bash
# Activate virtual environment
source .venv/bin/activate

# Option 1: Web UI (recommended)
python -m scripts.chat_web
# Then visit http://localhost:8000 in your browser

# Option 2: Command-line chat
python -m scripts.chat_cli

# Option 3: Single query
python -m scripts.chat_cli -p "Explain quantum computing in simple terms"
```

### Accessing the Model from Outside

If you want to access the web UI from another device:

```bash
# Find your DGX Spark's IP address
ip addr show

# Then start the web server
python -m scripts.chat_web --host 0.0.0.0 --port 8000

# Access from another device: http://<DGX_SPARK_IP>:8000
```

## Advanced Configuration

### Memory Optimization

If you encounter memory issues, you can reduce the model size:

**Small Model (d12, ~150M params):**
```bash
# Edit dgx_spark_run.sh and change:
python -m scripts.base_train -- --depth=12 --device_batch_size=8 --run=$WANDB_RUN
```

**Medium Model (d16, ~300M params) - Default:**
```bash
python -m scripts.base_train -- --depth=16 --device_batch_size=4 --run=$WANDB_RUN
```

**Large Model (d18, ~450M params):**
```bash
python -m scripts.base_train -- --depth=18 --device_batch_size=2 --run=$WANDB_RUN
```

### Training Data Amount

To train on more or less data, adjust the number of shards:

```bash
# Fewer shards (faster, less capable model)
python -m nanochat.dataset -n 60

# More shards (slower, more capable model)
python -m nanochat.dataset -n 240
```

### WandB Integration

For experiment tracking with Weights & Biases:

```bash
# Install wandb if not already installed
pip install wandb

# Login to wandb
wandb login

# Run with wandb tracking
WANDB_RUN=my_experiment bash dgx_spark_run.sh
```

## Storage Management

### Data Locations

- **Cache directory**: `~/.cache/nanochat/` (datasets, checkpoints)
- **Checkpoints**: Stored in cache during training
- **Final model**: In cache after training completes
- **Virtual environment**: `.venv/` in project directory

### Cleaning Up

To free up disk space after training:

```bash
# Remove downloaded datasets (keep models)
rm -rf ~/.cache/nanochat/data/

# Remove old checkpoints (keep final model)
# Be careful with this - only remove if you're sure
rm -rf ~/.cache/nanochat/checkpoints/

# Remove everything (start fresh)
rm -rf ~/.cache/nanochat/
```

## Troubleshooting

### Common Issues

**Issue: "CUDA out of memory"**
```bash
# Reduce batch size in dgx_spark_run.sh
--device_batch_size=2  # or even =1
```

**Issue: "Failed to download dataset"**
```bash
# Check internet connection
ping huggingface.co

# Retry download manually
python -m nanochat.dataset -n 8
```

**Issue: "Rust compiler not found"**
```bash
# Install Rust manually
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

**Issue: "Package installation fails"**
```bash
# Update uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Try installing again
uv sync --extra gpu
```

### Getting Help

1. Check the main [nanochat repository](https://github.com/karpathy/nanochat) for general issues
2. Review [nanochat discussions](https://github.com/karpathy/nanochat/discussions)
3. Open an issue on this repository for DGX Spark-specific problems

## Performance Notes

### Expected Performance on DGX Spark

- **Training Speed**: ~2-3x slower than 8xH100 (due to single GPU)
- **Inference Speed**: Excellent for local deployment
- **Memory Efficiency**: Better than discrete GPU setups due to unified memory
- **Power Consumption**: ~140W (very efficient for AI workload)

### Comparison Table

| Configuration | Time | Model Size | Quality |
|--------------|------|------------|---------|
| Minimal (d12) | ~4-6 hours | 150M params | Basic |
| Default (d16) | ~8-12 hours | 300M params | Good |
| Large (d18) | ~12-16 hours | 450M params | Better |

## Next Steps

After successfully training your model:

1. Experiment with different prompts and conversation styles
2. Try customizing the model's personality ([guide](https://github.com/karpathy/nanochat/discussions/139))
3. Fine-tune on your own data
4. Deploy as a local ChatGPT alternative
5. Share your results with the community!

## Resources

- [Original nanochat README](https://github.com/karpathy/nanochat)
- [DGX Spark Documentation](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- [PyTorch ARM Support](https://pytorch.org/)
- [DeepWiki nanochat](https://deepwiki.com/karpathy/nanochat)
