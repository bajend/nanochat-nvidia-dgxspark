# nanochat-nvidia-dgxspark

> Karpathy's nanochat optimized for NVIDIA DGX Spark

This is a fork of [Andrej Karpathy's nanochat](https://github.com/karpathy/nanochat) specifically optimized to run on the **NVIDIA DGX Spark** desktop AI supercomputer.

## What is DGX Spark?

The NVIDIA DGX Spark is a compact desktop AI supercomputer featuring:
- **CPU**: 20-core ARM (Grace Blackwell Superchip)
- **GPU**: Single Blackwell GPU with 5th Gen Tensor Cores
- **Memory**: 128GB unified LPDDR5x system memory
- **AI Performance**: Up to 1,000 TOPS (inference)
- **Form Factor**: Ultra-compact desktop (150mm x 150mm x 50.5mm)

## What is nanochat?

nanochat is a minimal, full-stack implementation of a ChatGPT-like Large Language Model (LLM) pipeline. It covers:
- Tokenization (custom Rust BPE tokenizer)
- Pretraining on FineWeb-EDU
- Midtraining for conversational abilities
- Supervised Fine-Tuning (SFT)
- Optional Reinforcement Learning
- Evaluation benchmarks
- Web UI for chatting with your model

## DGX Spark Optimizations

This version adapts the original 8xH100 GPU setup to run efficiently on DGX Spark:

1. **Single GPU Training**: Removed distributed training (no `torchrun` multi-GPU)
2. **Smaller Model**: Using depth=16 (~300M parameters) instead of depth=20 (~561M parameters)
3. **Reduced Batch Sizes**: device_batch_size=4 (vs 32) to fit in single GPU memory
4. **ARM Architecture**: Native support for ARM-based Grace CPU
5. **Optimized Dataset**: Downloading only necessary data shards (~12GB vs ~24GB)
6. **Unified Memory**: Takes advantage of DGX Spark's 128GB unified memory architecture

## Quick Start

### Prerequisites

- NVIDIA DGX Spark with DGX OS
- ~50GB free disk space for data and checkpoints
- 8-12 hours for full training run

### Running the Full Pipeline

The fastest way to train and deploy your own ChatGPT-style model:

```bash
bash dgx_spark_run.sh
```

For long-running training, use a screen session:

```bash
screen -L -Logfile dgx_spark.log -S dgx_spark bash dgx_spark_run.sh
```

To detach from screen: `Ctrl-a d`  
To reattach: `screen -r dgx_spark`  
To view logs: `tail -f dgx_spark.log`

### Optional: WandB Logging

To enable Weights & Biases logging:

```bash
# First, login to wandb
wandb login

# Then run with WANDB_RUN environment variable
WANDB_RUN=dgx_spark bash dgx_spark_run.sh
```

## After Training

Once training completes, you can interact with your model:

### Chat via Web UI (Recommended)

```bash
source .venv/bin/activate
python -m scripts.chat_web
```

Then visit `http://localhost:8000` in your browser for a ChatGPT-like interface.

### Chat via CLI

```bash
source .venv/bin/activate
python -m scripts.chat_cli
```

Or for a single query:

```bash
python -m scripts.chat_cli -p "Why is the sky blue?"
```

### View Training Report

```bash
cat report.md
```

This contains detailed metrics, evaluation results, and benchmark scores.

## Expected Results

The DGX Spark configuration (depth=16, ~300M parameters) will produce a model that:
- Has basic conversational abilities
- Can follow simple instructions
- Makes more mistakes than larger models but is still fun to chat with
- Trains in approximately 8-12 hours on DGX Spark
- Similar capability to early GPT-2 era models

### Example Benchmark Scores (Expected)

| Metric          | BASE     | MID      | SFT      |
|-----------------|----------|----------|----------|
| CORE            | ~0.20    | -        | -        |
| ARC-Challenge   | -        | ~0.25    | ~0.26    |
| ARC-Easy        | -        | ~0.32    | ~0.35    |
| GSM8K           | -        | ~0.02    | ~0.03    |
| HumanEval       | -        | ~0.05    | ~0.07    |
| MMLU            | -        | ~0.28    | ~0.29    |

*Note: These are approximate estimates. Actual results may vary.*

## Customization

### Adjust Model Size

To train a larger or smaller model, edit `dgx_spark_run.sh`:

```bash
# For a smaller model (~150M params, faster training):
python -m scripts.base_train -- --depth=12 --device_batch_size=8 --run=$WANDB_RUN

# For a larger model (~450M params, longer training):
python -m scripts.base_train -- --depth=18 --device_batch_size=2 --run=$WANDB_RUN
```

### Skip Stages

You can skip certain training stages by commenting them out in `dgx_spark_run.sh`:
- Comment out RL section to skip reinforcement learning (saves ~2 hours)
- Comment out evaluation sections for faster iteration during development

### Custom Identity

To give your model a custom personality, see the [identity customization guide](https://github.com/karpathy/nanochat/discussions/139) and modify the `identity_conversations.jsonl` file.

## File Structure

```
.
├── dgx_spark_run.sh          # Main training script for DGX Spark
├── nanochat/                 # Core Python package
│   ├── gpt.py               # GPT Transformer model
│   ├── engine.py            # Inference engine
│   ├── tokenizer.py         # BPE tokenizer wrapper
│   ├── dataloader.py        # Distributed data loader
│   └── ...                  # Other utilities
├── scripts/                  # Training and evaluation scripts
│   ├── base_train.py        # Pretraining
│   ├── mid_train.py         # Midtraining
│   ├── chat_sft.py          # Supervised fine-tuning
│   ├── chat_web.py          # Web UI server
│   └── ...                  # Other scripts
├── rustbpe/                  # Rust BPE tokenizer
├── tasks/                    # Evaluation tasks
└── tests/                    # Unit tests

```

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors, reduce the batch size:

```bash
python -m scripts.base_train -- --depth=16 --device_batch_size=2 --run=$WANDB_RUN
```

### Slow Training

DGX Spark uses a single GPU, so training is naturally slower than 8-GPU setups. Expected times:
- Tokenizer training: ~30 minutes
- Base pretraining: ~4-6 hours
- Midtraining: ~1-2 hours
- SFT: ~1-2 hours

### ARM Compatibility Issues

This code is designed to work on ARM architecture. If you encounter issues:
- Ensure you're using the ARM-compatible Rust toolchain
- Check that PyTorch is installed correctly for ARM
- Verify uv is properly configured for ARM64

## Differences from Original nanochat

| Feature | Original (8xH100) | DGX Spark Version |
|---------|------------------|-------------------|
| GPUs | 8x H100 (80GB each) | 1x Blackwell GPU |
| Model Size | d20 (561M params) | d16 (300M params) |
| Batch Size | 32 per device | 4 per device |
| Training Time | ~4 hours | ~8-12 hours |
| Data Downloaded | ~24GB | ~12GB |
| Architecture | x86_64 | ARM64 (aarch64) |
| Multi-GPU | Yes (DDP) | No (single GPU) |

## License

MIT - Same as original nanochat

## Acknowledgements

- Original [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy
- Optimized for NVIDIA DGX Spark desktop AI supercomputer
- Built on PyTorch, Rust, and the amazing open-source AI community

## Citation

If you use this DGX Spark adaptation in your work:

```bibtex
@misc{nanochat-dgx-spark,
  author = {Original: Andrej Karpathy, DGX Spark Adaptation},
  title = {nanochat-nvidia-dgxspark: nanochat optimized for NVIDIA DGX Spark},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/bajend/nanochat-nvidia-dgxspark}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to improve the DGX Spark optimization.
