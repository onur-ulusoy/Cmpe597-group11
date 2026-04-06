#!/bin/bash

# Configuration
DEVICE="cuda" # Or "mps" for Mac, "cpu" for low-end
BATCH_SIZE=128
EPOCHS=10
FEATURE_DIR="data/features/openclip_vit_l_14"

# 1. Ensure dependencies are installed
echo "[*] Installing dependencies..."
pip install -r requirements.txt

# 2. Pre-extract CLIP Features (Crucial for speed)
# This only needs to be run once.
echo "[*] Starting Feature Extraction..."
python src/tasks/classification/train/extract_features.py \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --output_dir $FEATURE_DIR

# 3. Train the Classifier
# This will be very fast (seconds per epoch) once features are extracted.
echo "[*] Starting 10-Epoch Training..."
python src/tasks/classification/train/caption_type.py \
    --epochs $EPOCHS \
    --feature_dir $FEATURE_DIR \
    --device $DEVICE \
    --output_dir outputs/classification/train/
