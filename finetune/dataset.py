import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import os

class MemeCapTrainDataset(Dataset):
    def __init__(self, json_path, image_root, processor):
        self.image_root = Path(image_root)
        self.processor = processor
        
        print(f"📂 Loading dataset from: {json_path}")
        print(f"🖼️  Image root: {self.image_root.resolve()}")

        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        self.valid_data = []
        missing_count = 0
        
        # Debug: Print the first filename we are looking for
        if len(self.data) > 0:
            print(f"🔍 First JSON entry 'img': {self.data[0].get('img')}")

        for item in self.data:
            # FIX: The key is 'img', not 'img_fname'
            img_fname = item.get('img')
            
            if not img_fname:
                continue
                
            # Try to resolve the path (handle subfolders or direct filenames)
            image_path = self._resolve_path(img_fname)
            
            if image_path:
                item['resolved_path'] = image_path
                self.valid_data.append(item)
            else:
                missing_count += 1
                # Print the first missing example to help debug
                if missing_count == 1:
                    print(f"❌ Could not find first image at: {self.image_root / img_fname}")

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
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a dummy tensor to prevent crash
            raise e
        
        captions = item.get('meme_captions', "")
        if isinstance(captions, list):
            caption = captions[0]
        else:
            caption = captions
            
        inputs = self.processor(
            text=[caption],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77 
        )
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }
