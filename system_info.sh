#!/bin/bash

# System Information Script
# Prints Python version, CUDA, PyTorch CUDA version, and OS details

echo "==================== SYSTEM INFORMATION ===================="
echo

# Python Version
echo "🐍 PYTHON VERSION:"
if command -v python3 &> /dev/null; then
    python3 --version
elif command -v python &> /dev/null; then
    python --version
else
    echo "Python not found"
fi
echo

# CUDA Version
echo "🚀 CUDA VERSION:"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release" | sed 's/.*release /CUDA /'
    echo "CUDA Toolkit Location: $(which nvcc)"
elif command -v nvidia-smi &> /dev/null; then
    echo "CUDA Driver Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)"
    echo "CUDA Runtime Version: $(nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits | head -1)"
else
    echo "CUDA not found or not available"
fi
echo

# PyTorch CUDA Version
echo "🔥 PYTORCH CUDA VERSION:"
if command -v python3 &> /dev/null; then
    python3 -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
if torch.cuda.is_available():
    print(f'PyTorch CUDA Version: {torch.version.cuda}')
    print(f'CUDA Device Count: {torch.cuda.device_count()}')
    if torch.cuda.device_count() > 0:
        print(f'CUDA Device Name: {torch.cuda.get_device_name(0)}')
        print(f'CUDA Capability: {torch.cuda.get_device_capability(0)}')
else:
    print('CUDA not available in PyTorch')
" 2>/dev/null || echo "PyTorch not installed or Python not available"
elif command -v python &> /dev/null; then
    python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
if torch.cuda.is_available():
    print(f'PyTorch CUDA Version: {torch.version.cuda}')
    print(f'CUDA Device Count: {torch.cuda.device_count()}')
    if torch.cuda.device_count() > 0:
        print(f'CUDA Device Name: {torch.cuda.get_device_name(0)}')
        print(f'CUDA Capability: {torch.cuda.get_device_capability(0)}')
else:
    print('CUDA not available in PyTorch')
" 2>/dev/null || echo "PyTorch not installed or Python not available"
else
    echo "Python not available to check PyTorch"
fi
echo

# Operating System Information
echo "💻 OPERATING SYSTEM:"
if [ -f /etc/os-release ]; then
    source /etc/os-release
    echo "Distribution: $PRETTY_NAME"
    echo "Version: $VERSION"
    echo "Version ID: $VERSION_ID"
    echo "ID: $ID"
    echo "ID Like: $ID_LIKE"
else
    echo "Distribution: $(uname -s)"
fi
echo

# Architecture
echo "🏗️ ARCHITECTURE:"
echo "Machine Architecture: $(uname -m)"
echo "Processor: $(uname -p)"
echo "Hardware Platform: $(uname -i)"
echo

# Compilation Information
echo "⚙️ COMPILATION:"
echo "Kernel: $(uname -r)"
echo "Kernel Version: $(uname -v)"
if command -v gcc &> /dev/null; then
    echo "GCC Version: $(gcc --version | head -1)"
else
    echo "GCC: Not installed"
fi
if command -v clang &> /dev/null; then
    echo "Clang Version: $(clang --version | head -1)"
else
    echo "Clang: Not installed"
fi
echo

# Additional Distribution Details
echo "📦 DISTRIBUTION DETAILS:"
if [ -f /etc/lsb-release ]; then
    echo "LSB Release Info:"
    cat /etc/lsb-release | sed 's/^/  /'
fi

if command -v lsb_release &> /dev/null; then
    echo "LSB Description: $(lsb_release -d | cut -f2)"
    echo "LSB Codename: $(lsb_release -c | cut -f2)"
fi

# Package Manager / Installer Type
echo
echo "📋 INSTALLER TYPE:"
if command -v apt &> /dev/null; then
    echo "Package Manager: APT (Debian/Ubuntu)"
elif command -v yum &> /dev/null; then
    echo "Package Manager: YUM (Red Hat/CentOS)"
elif command -v dnf &> /dev/null; then
    echo "Package Manager: DNF (Fedora)"
elif command -v pacman &> /dev/null; then
    echo "Package Manager: Pacman (Arch Linux)"
elif command -v zypper &> /dev/null; then
    echo "Package Manager: Zypper (openSUSE)"
elif command -v apk &> /dev/null; then
    echo "Package Manager: APK (Alpine Linux)"
else
    echo "Package Manager: Unknown or not detected"
fi

# System Uptime and Load
echo
echo "⏱️ SYSTEM STATUS:"
echo "Uptime: $(uptime -p 2>/dev/null || uptime)"
echo "Load Average: $(uptime | awk -F'load average:' '{print $2}')"

# Memory Information
echo
echo "💾 MEMORY:"
if command -v free &> /dev/null; then
    free -h
else
    echo "Memory information not available"
fi

echo
echo "==================== END OF REPORT ===================="