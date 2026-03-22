import json
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch

class MemeCapTrainDataset(Dataset):
    def __init__(self, json_path, image_root, processor):
        self.image_root = Path(image_root)
        self.processor = processor
        
        print(f"📂 Loading dataset from: {json_path}")
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        self.valid_data = []
        missing_count = 0
        
        # Debug: Check keys in the first item
        if len(self.data) > 0:
            print(f"🔍 First JSON keys: {list(self.data[0].keys())}")

        for item in self.data:
            # FIX: Check both 'img' (Train) and 'img_fname' (Test) keys
            img_fname = item.get('img') or item.get('img_fname')
            
            if not img_fname:
                continue
                
            # Try to resolve the path
            image_path = self._resolve_path(img_fname)
            
            if image_path:
                item['resolved_path'] = image_path
                self.valid_data.append(item)
            else:
                missing_count += 1
                if missing_count == 1:
                    print(f"❌ First missing image: {img_fname}")

        print(f"✅ Loaded {len(self.valid_data)} valid samples.")
        if missing_count > 0:
            print(f"⚠️  Skipped {missing_count} missing images.")

    def _resolve_path(self, img_fname):
        # 1. Check exact path
        candidate = self.image_root / img_fname
        if candidate.exists():
            return candidate
            
        # 2. Check just the filename (ignore folders in json)
        basename = Path(img_fname).name
        candidate = self.image_root / basename
        if candidate.exists():
            return candidate
            
        return None

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        image_path = item['resolved_path']
        
        # 1. Load Image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a dummy to prevent crash (shouldn't happen with valid_data)
            image = Image.new('RGB', (224, 224))
        
        # 2. Get Caption
        captions = item.get('meme_captions', "")
        if isinstance(captions, list):
            caption = captions[0] if captions else ""
        else:
            caption = captions
            
        # 3. Process
        # The OpenClipAdapter handles the tokenizer details
        processed = self.processor(
            images=image, 
            text=[caption], 
            return_tensors="pt"
        )
        
        return {
            "pixel_values": processed['pixel_values'].squeeze(0),
            "input_ids": processed['input_ids'].squeeze(0),
            "attention_mask": processed['attention_mask'].squeeze(0)
        }
