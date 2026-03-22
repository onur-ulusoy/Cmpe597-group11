#!/bin/bash

# CMPE 597 Project Universal Setup Script
# Works on: macOS (Apple Silicon/Intel) and Linux (CUDA/CPU)

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   CMPE 597 Project Environment Setup   ${NC}"
echo -e "${BLUE}========================================${NC}\n"

# [1/6] Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     OS_TYPE=Linux;;
    Darwin*)    OS_TYPE=Mac;;
    *)          OS_TYPE="UNKNOWN:${OS}"
esac
echo -e "${YELLOW}[1/6] Detected System: ${OS_TYPE}${NC}"

# [2/6] Check Python
echo -e "${YELLOW}[2/6] Checking Python version...${NC}"
if command -v python3 &>/dev/null; then
    python3 --version
else
    echo -e "${RED}❌ Python3 not found! Please install Python 3.8+.${NC}"; exit 1;
fi

# [3/6] Create & Activate venv
if [ -d "venv" ]; then
    echo -e "${YELLOW}[3/6] Virtual environment already exists. Activating...${NC}"
else
    echo -e "${YELLOW}[3/6] Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}\n"

# [4/6] Upgrade pip
echo -e "${YELLOW}[4/6] Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}✅ pip upgraded${NC}\n"

# [5/6] Install Dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}[5/6] Installing dependencies from requirements.txt...${NC}"
    echo "This may take a few minutes..."
    pip install -r requirements.txt --quiet
    echo -e "${GREEN}✅ Dependencies installed${NC}\n"
else
    echo -e "${RED}❌ requirements.txt not found!${NC}"
    echo "Please create requirements.txt with: torch, transformers, peft, open_clip_torch, etc."
    exit 1
fi

# [6/6] Universal Hardware Test
echo -e "${YELLOW}[6/6] Testing Hardware Acceleration (CUDA/MPS/CPU)...${NC}\n"

python3 << 'PYTEST'
import torch
import sys
import platform

def print_status(name, status):
    icon = "✅" if status else "❌"
    print(f"{icon} {name}")

print("-" * 40)
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print("-" * 40)

# Check for Imports
try:
    import transformers
    import peft
    import open_clip
    import PIL
    print("✅ Libraries: Transformers, PEFT, OpenCLIP, Pillow loaded.")
except ImportError as e:
    print(f"❌ Library Error: {e}")
    sys.exit(1)

print("-" * 40)
print("Hardware Acceleration Check:")

device = "cpu"
batch_rec = 8

# 1. Check CUDA (Linux/Windows NVIDIA)
if torch.cuda.is_available():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ CUDA Available: {gpu_name} ({vram:.1f} GB VRAM)")
    batch_rec = 32 if vram > 16 else 16

# 2. Check MPS (Mac Apple Silicon)
elif torch.backends.mps.is_available():
    device = "mps"
    print(f"✅ MPS (Metal Performance Shaders) Available")
    # Mac Unified Memory is hard to detect via PyTorch, assuming decent M-series
    batch_rec = 32 

# 3. Fallback to CPU
else:
    print("⚠️  No GPU detected. Using CPU (Training will be slow).")
    batch_rec = 4

# Test Tensor Allocation
try:
    x = torch.randn(10, 10).to(device)
    y = torch.randn(10, 10).to(device)
    z = x @ y
    print(f"✅ Tensor computation test passed on: {device.upper()}")
except Exception as e:
    print(f"❌ Hardware test failed: {e}")
    sys.exit(1)

print("-" * 40)
print("Recommended Training Settings:")
print(f"Device: {device}")
print(f"Batch Size: {batch_rec}")
print("-" * 40)

PYTEST

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ Setup Complete! Ready to train.${NC}"
    echo -e "${GREEN}========================================${NC}\n"
    echo -e "${BLUE}To start, run:${NC}"
    echo -e "  ${YELLOW}source venv/bin/activate${NC}"
else
    echo -e "\n${RED}❌ Setup failed during testing.${NC}"
    exit 1
fi
