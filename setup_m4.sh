#!/bin/bash

# CMPE 597 Project Setup Script for Mac M4
# Creates venv, installs dependencies, and tests everything

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}CMPE 597 Project Setup for Mac M4${NC}"
echo -e "${BLUE}========================================${NC}\n"

# [1/8] Check Python
echo -e "${YELLOW}[1/8] Checking Python version...${NC}"
python3 --version || { echo -e "${RED}❌ Python3 not found!${NC}"; exit 1; }
echo -e "${GREEN}✅ Python3 found${NC}\n"

# [2/8] Create venv
if [ -d "venv" ]; then
    echo -e "${YELLOW}[2/8] Virtual environment already exists${NC}\n"
else
    echo -e "${YELLOW}[2/8] Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}\n"
fi

# [3/8] Activate venv
echo -e "${YELLOW}[3/8] Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}\n"

# [4/8] Upgrade pip
echo -e "${YELLOW}[4/8] Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}✅ pip upgraded${NC}\n"

# [5/8] Install PyTorch
echo -e "${YELLOW}[5/8] Installing PyTorch (Apple Silicon optimized)...${NC}"
echo "This may take 2-3 minutes..."
pip install torch torchvision torchaudio --quiet
echo -e "${GREEN}✅ PyTorch installed${NC}\n"

# [6/8] Install OpenCLIP
echo -e "${YELLOW}[6/8] Installing OpenCLIP...${NC}"
pip install open-clip-torch --quiet
echo -e "${GREEN}✅ OpenCLIP installed${NC}\n"

# [7/8] Install other dependencies
echo -e "${YELLOW}[7/8] Installing Transformers, Pillow, and utilities...${NC}"
pip install transformers pillow tqdm numpy peft --quiet
echo -e "${GREEN}✅ All dependencies installed${NC}\n"

# [8/8] Test dependencies
echo -e "${YELLOW}[8/8] Testing all dependencies...${NC}\n"

python3 << 'PYTEST'
import sys

def test_import(module_name, display_name=None):
    if display_name is None:
        display_name = module_name
    try:
        __import__(module_name)
        print(f"✅ {display_name}")
        return True
    except ImportError as e:
        print(f"❌ {display_name}: {e}")
        return False

print("Testing Python packages:")
print("-" * 40)

all_ok = True
all_ok &= test_import("torch", "PyTorch")
all_ok &= test_import("torchvision", "TorchVision")
all_ok &= test_import("open_clip", "OpenCLIP")
all_ok &= test_import("transformers", "Transformers")
all_ok &= test_import("PIL", "Pillow")
all_ok &= test_import("tqdm", "tqdm")
all_ok &= test_import("numpy", "NumPy")

print("\n" + "=" * 40)
print("Device Availability:")
print("-" * 40)

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {sys.version.split()[0]}")

mps_available = torch.backends.mps.is_available()
print(f"MPS (Apple GPU): {mps_available}")
print(f"CUDA: {torch.cuda.is_available()}")

if mps_available:
    print("\n✅ GPU acceleration available (MPS)")
    recommended_device = "mps"
    recommended_batch = 32
    
    # Test MPS
    try:
        device = torch.device("mps")
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = x @ y
        print("✅ MPS device test passed")
    except Exception as e:
        print(f"⚠️  MPS test failed: {e}")
        recommended_device = "cpu"
        recommended_batch = 8
else:
    print("\n⚠️  MPS not available, using CPU")
    recommended_device = "cpu"
    recommended_batch = 8

print("\n" + "=" * 40)
print("Recommended Settings:")
print("-" * 40)
print(f"Device: {recommended_device}")
print(f"Batch size: {recommended_batch}")
print(f"Text batch size: {recommended_batch * 2}")

sys.exit(0 if all_ok else 1)
PYTEST

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ Setup completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}\n"
    
    echo -e "${BLUE}To activate the environment:${NC}"
    echo -e "  ${YELLOW}source venv/bin/activate${NC}\n"
    
    echo -e "${BLUE}Installed packages:${NC}"
    pip list | grep -E "torch|clip|transformers|Pillow|tqdm|numpy"
    
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}❌ Setup failed during testing${NC}"
    echo -e "${RED}========================================${NC}\n"
    exit 1
fi
