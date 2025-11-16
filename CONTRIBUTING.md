# Contributing to nanochat-nvidia-dgxspark

Thank you for your interest in contributing to the DGX Spark adaptation of nanochat! This guide will help you get started.

## Project Goals

This project aims to:
1. Make nanochat accessible on NVIDIA DGX Spark hardware
2. Optimize for single-GPU, ARM-based training
3. Maintain compatibility with the original nanochat codebase
4. Provide excellent documentation and user experience
5. Enable local, private LLM training and deployment

## How to Contribute

### Reporting Issues

**Before creating an issue:**
- Check existing issues to avoid duplicates
- Verify the issue is specific to DGX Spark (not original nanochat)
- Test with the latest version

**When creating an issue, include:**
- DGX Spark hardware specs
- DGX OS version
- Steps to reproduce
- Error messages and logs
- Expected vs actual behavior

### Suggesting Features

We welcome feature suggestions! Please:
- Explain the use case
- Describe expected behavior
- Consider DGX Spark constraints (single GPU, ARM, etc.)
- Propose implementation if possible

### Code Contributions

#### Areas for Contribution

1. **Performance Optimizations**
   - ARM-specific optimizations
   - Memory efficiency improvements
   - Training speed enhancements
   - Inference optimizations

2. **Documentation**
   - Tutorials and guides
   - Example notebooks
   - Troubleshooting tips
   - Use case examples

3. **Testing**
   - Unit tests
   - Integration tests
   - Hardware compatibility tests
   - Benchmark improvements

4. **Features**
   - Multi-DGX Spark clustering
   - Advanced inference features
   - Custom data loading
   - Monitoring and visualization

5. **Bug Fixes**
   - DGX Spark-specific issues
   - ARM compatibility
   - Memory leaks
   - Edge cases

#### Not Accepting

- Changes that break compatibility with original nanochat
- Features requiring multiple GPUs (defeats DGX Spark purpose)
- Major architectural changes (propose first)
- Non-ARM platform support (out of scope)

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/nanochat-nvidia-dgxspark.git
cd nanochat-nvidia-dgxspark
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number-description
```

### 3. Set Up Development Environment

```bash
# Install dependencies
uv sync --extra gpu

# Activate environment
source .venv/bin/activate

# Install development tools
pip install pytest black flake8
```

### 4. Make Changes

Follow these guidelines:
- Keep changes minimal and focused
- Test on actual DGX Spark hardware when possible
- Update documentation if needed
- Add tests for new features
- Follow existing code style

### 5. Test Your Changes

```bash
# Run existing tests
python -m pytest tests/ -v

# Test specific functionality
python -m scripts.chat_cli -p "Test query"

# Lint code
black nanochat/ scripts/
flake8 nanochat/ scripts/
```

### 6. Commit Changes

Use clear, descriptive commit messages:

```bash
# Good commit messages:
git commit -m "Optimize ARM memory allocation in dataloader"
git commit -m "Fix tokenizer build on ARM64 architecture"
git commit -m "Add troubleshooting guide for OOM errors"

# Bad commit messages:
git commit -m "Fix bug"
git commit -m "Update code"
git commit -m "Changes"
```

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a PR on GitHub with:
- Clear description of changes
- Why the changes are needed
- How to test the changes
- Screenshots (if UI changes)
- Link to related issues

## Code Style Guidelines

### Python

- Follow PEP 8
- Use type hints where appropriate
- Add docstrings for functions and classes
- Keep functions focused and small
- Use meaningful variable names

Example:
```python
def calculate_tokens_needed(params: int, chinchilla_ratio: int = 20) -> int:
    """
    Calculate number of tokens needed for training based on Chinchilla scaling.
    
    Args:
        params: Number of model parameters
        chinchilla_ratio: Tokens per parameter (default: 20)
    
    Returns:
        Number of tokens needed for training
    """
    return params * chinchilla_ratio
```

### Shell Scripts

- Use bash (not sh)
- Add comments for complex logic
- Handle errors appropriately
- Use meaningful variable names
- Follow existing script style

Example:
```bash
#!/bin/bash

# Configure DGX Spark specific settings
DEPTH=16
BATCH_SIZE=4

# Train with error handling
if ! python -m scripts.base_train -- --depth=$DEPTH --device_batch_size=$BATCH_SIZE; then
    echo "Training failed!"
    exit 1
fi
```

### Documentation

- Use Markdown
- Include code examples
- Add screenshots for visual features
- Keep language clear and concise
- Test examples work correctly

## Testing Requirements

### For Bug Fixes

- Add test that reproduces the bug
- Verify fix resolves the issue
- Ensure no regressions

### For New Features

- Add unit tests
- Add integration test if applicable
- Test on DGX Spark hardware
- Update documentation

### Testing Checklist

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing on DGX Spark (if possible)
- [ ] No new warnings or errors
- [ ] Documentation updated
- [ ] Examples work as described

## Pull Request Process

1. **Create PR** with clear description
2. **Address feedback** from reviewers
3. **Update documentation** if needed
4. **Ensure tests pass**
5. **Squash commits** if requested
6. **Wait for approval** from maintainers

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Commit messages clear
- [ ] No merge conflicts
- [ ] Tested on DGX Spark (if applicable)

## LLM Disclosure Policy

Following the original nanochat policy:

**When submitting a PR, please declare:**
- Any parts with substantial LLM contribution
- Code you have not written personally
- Code you do not fully understand

This helps maintain code quality and learning opportunities.

## Communication

- **Issues**: For bug reports and feature requests
- **Discussions**: For questions and ideas
- **Pull Requests**: For code contributions
- **Email**: For private/security issues

## Recognition

Contributors will be:
- Listed in contributors section
- Credited in release notes
- Acknowledged in documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License (same as the project).

## Questions?

- Check the README.md
- Review existing issues and PRs
- Ask in GitHub Discussions
- Contact maintainers

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Provide constructive feedback
- Focus on technical merit
- Help others learn and grow
- Maintain professionalism

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or inflammatory comments
- Personal attacks
- Publishing private information
- Unethical or illegal activity

## Getting Started Checklist

- [ ] Read this guide
- [ ] Check existing issues
- [ ] Fork repository
- [ ] Set up development environment
- [ ] Make small test change
- [ ] Submit your first PR!

## Examples of Good Contributions

### Example 1: Performance Optimization

```
Title: Optimize memory allocation for ARM architecture

Description:
- Reduced memory overhead by 15% on ARM
- Implemented NEON SIMD optimizations
- Tested on DGX Spark with d16 model

Changes:
- Modified nanochat/dataloader.py
- Added ARM-specific memory pooling
- Updated documentation

Testing:
- Ran full training pipeline
- Verified memory usage reduction
- No performance regressions
```

### Example 2: Documentation Improvement

```
Title: Add troubleshooting guide for common OOM errors

Description:
- Documented common out-of-memory scenarios
- Added solutions and workarounds
- Included configuration examples

Changes:
- Added TROUBLESHOOTING.md
- Updated README with link
- Added memory usage examples

Testing:
- Verified all solutions work
- Tested on DGX Spark hardware
- Screenshots included
```

### Example 3: Bug Fix

```
Title: Fix tokenizer build failure on ARM64

Description:
- Resolves issue #123
- Tokenizer failed to compile on ARM
- Added ARM64 target specification

Changes:
- Updated rustbpe/Cargo.toml
- Added ARM64 build flags
- Updated build documentation

Testing:
- Built successfully on DGX Spark
- Tokenizer training completes
- No regressions on other platforms
```

## Thank You!

Your contributions make this project better for everyone using DGX Spark for LLM development. We appreciate your time and effort!

---

**Ready to contribute?** Check out the [open issues](https://github.com/bajend/nanochat-nvidia-dgxspark/issues) or start with improving documentation!
