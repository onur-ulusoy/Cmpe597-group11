import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class MemeCapTrainDataset(Dataset):
    def __init__(self, json_path, image_root, processor):
        """
        Args:
            json_path: Path to memes-trainval.json
            image_root: Path to image directory
            processor: CLIPProcessor (handles both image and text processing)
        """
        self.image_root = Path(image_root)
        self.processor = processor
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        # Filter out entries where image doesn't exist to prevent crashing
        self.valid_data = []
        for item in self.data:
            img_fname = item.get('img_fname')
            if img_fname and (self.image_root / img_fname).exists():
                self.valid_data.append(item)
                
        print(f"Loaded {len(self.valid_data)} training samples.")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        
        # 1. Load Image
        image_path = self.image_root / item['img_fname']
        image = Image.open(image_path).convert("RGB")
        
        # 2. Load Text (The Meme Caption)
        # Some entries might have a list of captions, take the first one
        captions = item.get('meme_captions', "")
        if isinstance(captions, list):
            caption = captions[0]
        else:
            caption = captions
            
        # 3. Process using CLIP Processor
        # This returns a dict with 'pixel_values', 'input_ids', 'attention_mask'
        inputs = self.processor(
            text=[caption],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77 
        )
        
        # Remove batch dimension added by processor (1, C, H, W) -> (C, H, W)
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }
