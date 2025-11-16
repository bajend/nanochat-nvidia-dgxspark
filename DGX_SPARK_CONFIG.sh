# DGX Spark Configuration
# This file documents the recommended configurations for NVIDIA DGX Spark

# ============================================================================
# HARDWARE SPECIFICATIONS
# ============================================================================
# CPU: 20-core ARM (Grace: 10x Cortex-X925, 10x Cortex-A725)
# GPU: NVIDIA Blackwell Architecture
# Memory: 128GB unified LPDDR5x @ 273 GB/s
# Storage: 1TB or 4TB NVMe SSD
# Network: 10 GbE + ConnectX-7 (200 Gbps)
# Power: 240W external supply (140W SoC TDP)

# ============================================================================
# RECOMMENDED MODEL SIZES FOR DGX SPARK
# ============================================================================

# SMALL (Development/Testing)
# - Depth: 12 layers
# - Parameters: ~150M
# - Batch size: 8
# - Training time: ~4-6 hours
# - Memory usage: ~40GB
SMALL_DEPTH=12
SMALL_BATCH=8

# MEDIUM (Default - Recommended)
# - Depth: 16 layers
# - Parameters: ~300M
# - Batch size: 4
# - Training time: ~8-12 hours
# - Memory usage: ~60GB
MEDIUM_DEPTH=16
MEDIUM_BATCH=4

# LARGE (Advanced)
# - Depth: 18 layers
# - Parameters: ~450M
# - Batch size: 2
# - Training time: ~12-16 hours
# - Memory usage: ~80GB
LARGE_DEPTH=18
LARGE_BATCH=2

# EXTRA LARGE (Maximum)
# - Depth: 20 layers
# - Parameters: ~561M
# - Batch size: 1
# - Training time: ~16-24 hours
# - Memory usage: ~100GB
XLARGE_DEPTH=20
XLARGE_BATCH=1

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Tokenizer training data (initial)
TOKENIZER_DATA_SHARDS=8  # ~800MB

# Pretraining data shards by model size
# Formula: (params * 20 tokens) * 4.8 chars/token / 250M chars/shard
SMALL_DATA_SHARDS=60    # 150M params -> ~60 shards (~6GB)
MEDIUM_DATA_SHARDS=120  # 300M params -> ~120 shards (~12GB)
LARGE_DATA_SHARDS=180   # 450M params -> ~180 shards (~18GB)
XLARGE_DATA_SHARDS=240  # 561M params -> ~240 shards (~24GB)

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Learning rates (auto-adjusted by scripts)
BASE_LR=0.001
MID_LR=0.0001
SFT_LR=0.00001

# Gradient accumulation steps (auto-calculated)
# For single GPU: effective_batch = device_batch * accum_steps
# Target effective batch size: 256
# accum_steps = 256 / device_batch_size

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

# Cache directory for datasets and checkpoints
CACHE_DIR="$HOME/.cache/nanochat"

# Checkpoint frequency (save every N steps)
CHECKPOINT_FREQ=1000

# Maximum checkpoints to keep
MAX_CHECKPOINTS=3

# ============================================================================
# COMPUTE OPTIMIZATIONS
# ============================================================================

# Number of CPU threads for data loading
# Set to 1 to avoid oversubscription on ARM
OMP_NUM_THREADS=1

# Enable PyTorch optimizations
TORCH_COMPILE=1
CUDA_LAUNCH_BLOCKING=0

# ARM-specific optimizations
# Use Neon SIMD instructions
ARM_NEON_ENABLE=1

# ============================================================================
# INFERENCE SETTINGS
# ============================================================================

# Web server configuration
WEB_HOST="0.0.0.0"
WEB_PORT=8000

# Generation parameters
MAX_GEN_LENGTH=2048
TEMPERATURE=0.8
TOP_P=0.95
TOP_K=50

# KV cache configuration
KV_CACHE_ENABLED=1
KV_CACHE_MAX_TOKENS=4096

# ============================================================================
# EVALUATION BENCHMARKS
# ============================================================================

# Enabled benchmarks (all by default)
EVAL_ARC=1
EVAL_MMLU=1
EVAL_GSM8K=1
EVAL_HUMANEVAL=1
EVAL_CORE=1
EVAL_CHATCORE=1

# ============================================================================
# WANDB CONFIGURATION (Optional)
# ============================================================================

# Set to enable wandb logging
# export WANDB_RUN=dgx_spark_experiment
# export WANDB_PROJECT=nanochat-dgx-spark
# export WANDB_ENTITY=your_username

# ============================================================================
# QUICK LAUNCH EXAMPLES
# ============================================================================

# Example 1: Small model for testing
# python -m scripts.base_train -- --depth=12 --device_batch_size=8

# Example 2: Default medium model
# python -m scripts.base_train -- --depth=16 --device_batch_size=4

# Example 3: Large model
# python -m scripts.base_train -- --depth=18 --device_batch_size=2

# Example 4: Maximum size model
# python -m scripts.base_train -- --depth=20 --device_batch_size=1

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

# Track GPU utilization
# watch -n 1 nvidia-smi

# Monitor system resources
# htop

# Check disk usage
# df -h ~/.cache/nanochat

# Monitor training logs
# tail -f dgx_spark.log

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# If OOM (Out of Memory):
# 1. Reduce --device_batch_size
# 2. Reduce --depth (smaller model)
# 3. Enable gradient checkpointing (if implemented)

# If slow training:
# 1. Ensure GPU is being utilized (nvidia-smi)
# 2. Check no CPU throttling (htop)
# 3. Verify SSD performance (not network storage)

# If dataset download fails:
# 1. Check internet connection
# 2. Retry: python -m nanochat.dataset -n <num_shards>
# 3. Check available disk space

# ============================================================================
# END OF CONFIGURATION
# ============================================================================
