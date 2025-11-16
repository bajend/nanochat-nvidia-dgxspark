#!/usr/bin/env python3
"""
DGX Spark nanochat - Environment Validation Script

This script validates that your DGX Spark is properly configured
for running nanochat training.
"""

import sys
import subprocess
import platform
import os

def print_section(title):
    """Print a section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_command(cmd, name):
    """Check if a command exists"""
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=True)
        print(f"✓ {name} is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"✗ {name} is NOT installed")
        return False

def check_python_packages():
    """Check Python package availability"""
    packages = ["torch", "numpy", "datasets", "fastapi"]
    results = []
    
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg} is installed")
            results.append(True)
        except ImportError:
            print(f"✗ {pkg} is NOT installed")
            results.append(False)
    
    return all(results)

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU detected: {gpu_name}")
            print(f"  Total memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("✗ No CUDA GPU detected")
            print("  nanochat can run on CPU but will be very slow")
            return False
    except ImportError:
        print("✗ PyTorch not installed, cannot check GPU")
        return False

def check_disk_space():
    """Check available disk space"""
    try:
        home = os.path.expanduser("~")
        stat = os.statvfs(home)
        free_gb = (stat.f_bavail * stat.f_frsize) / 1e9
        print(f"  Free disk space in home: {free_gb:.1f} GB")
        
        if free_gb < 50:
            print("⚠ Warning: Less than 50 GB free. nanochat needs ~50 GB")
            return False
        else:
            print("✓ Sufficient disk space available")
            return True
    except Exception as e:
        print(f"✗ Could not check disk space: {e}")
        return False

def check_memory():
    """Check system memory"""
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
            total_mem = int([l for l in lines if 'MemTotal' in l][0].split()[1]) / 1e6
            print(f"  Total system memory: {total_mem:.1f} GB")
            
            if total_mem < 100:
                print("⚠ Warning: DGX Spark should have ~128 GB memory")
                return False
            else:
                print("✓ Sufficient system memory")
                return True
    except Exception as e:
        print(f"✗ Could not check memory: {e}")
        return False

def main():
    """Main validation function"""
    print_section("DGX Spark Environment Validation")
    
    print(f"\nPython version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    
    # Check if ARM architecture
    if platform.machine() not in ['aarch64', 'arm64']:
        print(f"\n⚠ Warning: Expected ARM64 architecture, got {platform.machine()}")
        print("  This version is optimized for DGX Spark (ARM)")
    else:
        print(f"\n✓ Correct architecture: {platform.machine()}")
    
    print_section("System Requirements")
    check_disk_space()
    check_memory()
    
    print_section("Required Commands")
    all_commands = True
    all_commands &= check_command("python3", "Python 3")
    all_commands &= check_command("git", "Git")
    
    print_section("Optional Commands (will be installed by script)")
    check_command("uv", "uv (Python package manager)")
    check_command("cargo", "Rust/Cargo")
    
    print_section("Python Packages")
    packages_ok = check_python_packages()
    
    if not packages_ok:
        print("\n  Run 'uv sync --extra gpu' to install packages")
    
    print_section("GPU Check")
    gpu_ok = check_gpu()
    
    print_section("Summary")
    print()
    
    if all_commands and gpu_ok:
        print("✓ Your system appears ready for DGX Spark nanochat!")
        print("\nNext steps:")
        print("  1. Run: bash dgx_spark_run.sh")
        print("  2. Or for a quick demo: bash quick_demo.sh")
    elif all_commands and not gpu_ok:
        print("⚠ System is ready but no GPU detected")
        print("  Training will be very slow on CPU")
        print("\nYou can still proceed:")
        print("  bash quick_demo.sh")
    else:
        print("✗ Some requirements are missing")
        print("\nPlease install missing components and run this script again")
        print("  See DGX_SPARK_SETUP.md for detailed instructions")
    
    print()

if __name__ == "__main__":
    main()
