import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class MemeCapTrainDataset(Dataset):
    def __init__(self, json_path, image_root, processor):
        self.image_root = image_root
        self.processor = processor
        
        print(f"📂 Loading dataset from: {json_path}")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Get Image Path
        img_name = item.get('img_fname', 'unknown.jpg')
        image_path = os.path.join(self.image_root, img_name)

        # 2. Get Caption (Robust)
        # Handle cases where meme_captions is None or empty
        captions_list = item.get('meme_captions', [])
        if captions_list is None: 
            captions_list = []
            
        if isinstance(captions_list, list) and len(captions_list) > 0:
            caption = captions_list[0]
        else:
            caption = "meme" 

        # 3. Get Title (Robust)
        title = item.get('title', "") 

        # --- CRITICAL FIX: FORCE STRING TYPE ---
        # This prevents "float object is not iterable" if title is NaN/None/Number
        if title is None: title = ""
        title = str(title)
        
        if caption is None: caption = ""
        caption = str(caption)
        # ---------------------------------------

        # 4. Load Image
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            # Create black image if file missing/corrupt
            # print(f"⚠️ Warning: Could not open {image_path}") # Optional: uncomment to debug missing files
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # 5. Process Inputs
        # Squeeze to remove batch dimension [1, ...] -> [...]
        
        processed_image = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        processed_caption = self.processor(text=caption, return_tensors="pt")["input_ids"].squeeze(0)
        processed_title = self.processor(text=title, return_tensors="pt")["input_ids"].squeeze(0)

        return {
            "pixel_values": processed_image,
            "input_ids": processed_caption,
            "title_ids": processed_title 
        }
