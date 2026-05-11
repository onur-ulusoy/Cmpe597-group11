import argparse
import os
import sys
import json
import torch
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.getcwd())

from src.models.pretrained.openclip import OpenCLIPBackend

import torch.nn.functional as F
from PIL import Image
import hashlib

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = self.preprocess(Image.open(p).convert("RGB"))
        return img, str(p.name)

def get_text_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

def extract_features(json_path, image_root, output_dir, backend, batch_size=64):
    with open(json_path, "r") as f:
        data = json.load(f)
        
    os.makedirs(output_dir, exist_ok=True)
    
    unique_images = list(set([item["img_fname"] for item in data]))
    image_paths = [Path(image_root) / fname for fname in unique_images]
    
    # 1. Extract and save image embeddings
    img_save_dir = os.path.join(output_dir, "images")
    os.makedirs(img_save_dir, exist_ok=True)
    
    img_dataset = ImageDataset(image_paths, backend.preprocess)
    img_loader = torch.utils.data.DataLoader(img_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    print(f"Extracting {len(image_paths)} images from {json_path}...")
    for images, fnames in tqdm(img_loader, desc="Images"):
        # Check if entire batch exists
        if all([os.path.exists(os.path.join(img_save_dir, f"{fname}.pt")) for fname in fnames]):
            continue
            
        images = images.to(backend.device)
        with torch.no_grad():
            feats = backend.model.encode_image(images)
            feats = F.normalize(feats, p=2, dim=-1)
        for fname, feat in zip(fnames, feats):
            torch.save(feat.cpu(), os.path.join(img_save_dir, f"{fname}.pt"))
            
    # 2. Extract and save text embeddings
    # We collect all unique captions (meme and literal)
    all_texts = set()
    for item in data:
        all_texts.update(item["meme_captions"])
        all_texts.update(item["img_captions"])
        
    unique_texts = list(all_texts)
    text_save_dir = os.path.join(output_dir, "texts")
    os.makedirs(text_save_dir, exist_ok=True)
    
    print(f"Extracting {len(unique_texts)} texts...")
    for i in tqdm(range(0, len(unique_texts), batch_size), desc="Texts"):
        batch_texts = unique_texts[i:i+batch_size]
        
        # Check if entire batch exists
        if all([os.path.exists(os.path.join(text_save_dir, f"{get_text_hash(text)}.pt")) for text in batch_texts]):
            continue
            
        feats = backend.encode_texts(batch_texts, batch_size=len(batch_texts))
        for text, feat in zip(batch_texts, feats):
            t_hash = get_text_hash(text)
            torch.save(feat, os.path.join(text_save_dir, f"{t_hash}.pt"))
            
    # Save mapping for texts
    mapping_path = os.path.join(output_dir, "text_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            existing_mapping = json.load(f)
        existing_mapping.update({text: get_text_hash(text) for text in unique_texts})
        mapping = existing_mapping
    else:
        mapping = {text: get_text_hash(text) for text in unique_texts}
        
    with open(mapping_path, "w") as f:
        json.dump(mapping, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, default="data/memes-trainval.json")
    parser.add_argument("--test_json", type=str, default="data/memes-test.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--output_dir", type=str, default="data/features/openclip_vit_l_14")
    parser.add_argument("--model_name", type=str, default="ViT-L-14")
    parser.add_argument("--pretrained", type=str, default="laion2b_s32b_b82k")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    from src.tasks.caption_classification.train.caption_type import get_device
    device = get_device(args.device)
    backend = OpenCLIPBackend(args.model_name, args.pretrained, device)
    
    print("Pre-extracting features for Train/Val...")
    extract_features(args.train_json, args.image_root, args.output_dir, backend, args.batch_size)
    
    print("\nPre-extracting features for Test...")
    extract_features(args.test_json, args.image_root, args.output_dir, backend, args.batch_size)
    
    print("\nExtraction complete!")
