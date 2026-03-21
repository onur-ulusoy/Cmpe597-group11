#!/bin/bash

# CMPE 597 - MemeCap Dataset Download Script
# Downloads MemeCap dataset from Hugging Face

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MemeCap Dataset Download${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check if data already exists
if [ -f "data/memes-test.json" ] && [ -f "data/memes-trainval.json" ] && [ -d "data/memes" ] && [ "$(ls -A data/memes 2>/dev/null)" ]; then
    echo -e "${GREEN}✅ Dataset already exists!${NC}"
    image_count=$(ls data/memes 2>/dev/null | wc -l | tr -d ' ')
    echo -e "Images: ${image_count} files\n"
    
    read -p "Re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Skipping download.${NC}"
        exit 0
    fi
    rm -rf data/memes/*
fi

# Check if datasets library is installed
echo -e "${YELLOW}[1/3] Checking dependencies...${NC}"
if ! python3 -c "import datasets" 2>/dev/null; then
    echo -e "${YELLOW}Installing Hugging Face datasets library...${NC}"
    pip install -q datasets
    echo -e "${GREEN}✅ Library installed${NC}\n"
else
    echo -e "${GREEN}✅ Dependencies ready${NC}\n"
fi

# Create data directory
echo -e "${YELLOW}[2/3] Creating directories...${NC}"
mkdir -p data/memes
echo -e "${GREEN}✅ Directories created${NC}\n"

# Download using Python script
echo -e "${YELLOW}[3/3] Downloading from Hugging Face...${NC}"
echo -e "${BLUE}This will download ~1.6GB of data (5-10 minutes)${NC}\n"

python3 << 'PYTHON_SCRIPT'
import json
from datasets import load_dataset
from pathlib import Path
from PIL import Image
import sys
import re

print("📥 Loading dataset from Hugging Face...")
try:
    dataset = load_dataset("Leonardo6/memecap", split="train")
    print(f"✅ Loaded {len(dataset)} samples\n")
    
    # Create data structures
    trainval_data = []
    test_data = []
    
    # Use first 90% as train, last 10% as test (to match original split)
    split_idx = int(len(dataset) * 0.9)
    
    print("💾 Processing and saving images...")
    for idx, item in enumerate(dataset):
        try:
            # Extract image and metadata
            img = item['images'][0]  # First image
            messages = item['messages']
            
            # Parse the meme info from messages
            user_msg = messages[0]['content']
            assistant_msg = messages[1]['content']
            
            # Extract title from user message (between quotes after "title")
            title = "Unknown"
            title_match = re.search(r'title\s+"([^"]+)"', user_msg)
            if title_match:
                title = title_match.group(1)
            
            # Extract meme text from user message (the text shown in the meme)
            meme_text = ""
            text_match = re.search(r'text\s+"([^"]*)"', user_msg)
            if text_match:
                meme_text = text_match.group(1)
            
            # Save image (convert RGBA to RGB if needed)
            img_filename = f"{idx:05d}.jpg"
            img_path = Path(f"data/memes/{img_filename}")
            
            # Convert RGBA to RGB before saving as JPEG
            if img.mode == 'RGBA':
                # Create white background
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                rgb_img.save(img_path, 'JPEG', quality=95)
            elif img.mode == 'P':
                # Convert palette mode to RGB
                img.convert('RGB').save(img_path, 'JPEG', quality=95)
            elif img.mode != 'RGB':
                # Convert any other mode to RGB
                img.convert('RGB').save(img_path, 'JPEG', quality=95)
            else:
                img.save(img_path, 'JPEG', quality=95)
            
            # Create entry matching the expected format
            # Expected format based on MemeCapDataset.from_paths():
            # {"img": "filename.jpg", "text": ["meme text"], "meme_captions": ["caption"]}
            entry = {
                "img": img_filename,
                "text": [meme_text] if meme_text else [""],
                "meme_captions": [assistant_msg],
                "title": title  # Additional field for reference
            }
            
            # Split into train/test
            if idx < split_idx:
                trainval_data.append(entry)
            else:
                test_data.append(entry)
            
            # Progress indicator
            if (idx + 1) % 500 == 0:
                print(f"  ✓ Processed {idx + 1}/{len(dataset)} images...")
        
        except Exception as e:
            print(f"  ⚠️  Warning: Failed to process image {idx}: {e}")
            continue
    
    # Save JSON files
    print("\n💾 Saving JSON files...")
    with open("data/memes-trainval.json", "w") as f:
        json.dump(trainval_data, f, indent=2)
    print(f"✅ Saved {len(trainval_data)} training samples")
    
    with open("data/memes-test.json", "w") as f:
        json.dump(test_data, f, indent=2)
    print(f"✅ Saved {len(test_data)} test samples")
    
    # Show sample entry
    print(f"\n📋 Sample entry structure:")
    if trainval_data:
        sample = trainval_data[0]
        print(f"  img: {sample['img']}")
        print(f"  text: {sample['text'][0][:50]}..." if sample['text'][0] else "  text: (empty)")
        print(f"  meme_captions: {sample['meme_captions'][0][:80]}...")
        print(f"  title: {sample['title'][:50]}...")
    
    print(f"\n✅ Successfully downloaded {len(trainval_data) + len(test_data)} images!")
    
except Exception as e:
    print(f"\n❌ Error: {e}", file=sys.stderr)
    print("\n💡 If download fails, see MANUAL_DOWNLOAD.md for alternative methods.", file=sys.stderr)
    sys.exit(1)
PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    echo -e "\n${RED}❌ Download failed.${NC}"
    echo -e "${YELLOW}See MANUAL_DOWNLOAD.md for alternative download methods.${NC}\n"
    exit 1
fi

# Verify dataset
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Dataset Verification${NC}"
echo -e "${BLUE}========================================${NC}"

if [ -f "data/memes-test.json" ]; then
    test_count=$(python3 -c "import json; print(len(json.load(open('data/memes-test.json'))))" 2>/dev/null || echo "?")
    echo -e "${GREEN}✅ Test set: ${test_count} samples${NC}"
else
    echo -e "${RED}❌ Test JSON not found${NC}"
fi

if [ -f "data/memes-trainval.json" ]; then
    train_count=$(python3 -c "import json; print(len(json.load(open('data/memes-trainval.json'))))" 2>/dev/null || echo "?")
    echo -e "${GREEN}✅ Training set: ${train_count} samples${NC}"
else
    echo -e "${RED}❌ Training JSON not found${NC}"
fi

if [ -d "data/memes" ] && [ "$(ls -A data/memes 2>/dev/null)" ]; then
    image_count=$(ls data/memes 2>/dev/null | wc -l | tr -d ' ')
    echo -e "${GREEN}✅ Images: ${image_count} files${NC}"
    
    # Check total size
    total_size=$(du -sh data/memes 2>/dev/null | cut -f1)
    echo -e "${GREEN}✅ Total size: ${total_size}${NC}"
else
    echo -e "${RED}❌ No images found${NC}"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Dataset ready!${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo -e "${BLUE}Next steps:${NC}"
echo -e "  ${YELLOW}# Quick test (5 samples)${NC}"
echo -e "  ${YELLOW}python3 zero_shot/evaluation.py --limit 5 --data_dir data --image_root data/memes --model_family openclip --input_type type1 --device mps --output_dir outputs/test${NC}\n"
