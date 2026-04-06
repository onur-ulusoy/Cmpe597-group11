import json
import os
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset
from PIL import Image

@dataclass
class ClassificationSample:
    image_path: Path
    text: str
    label: int  # 1 for metaphorical, 0 for literal
    post_id: str

def load_classification_records(
    json_path: str, 
    image_root: str, 
    limit: Optional[int] = None
) -> List[ClassificationSample]:
    """
    Loads MemeCap records and converts them into classification pairs.
    Metaphorical (meme_caption) = 1
    Literal (img_caption) = 0
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    samples = []
    skipped = 0
    
    for item in data:
        post_id = item.get("post_id", "unknown")
        img_name = item.get("img_fname")
        if not img_name:
            continue
            
        img_path = Path(image_root) / img_name
        if not img_path.exists():
            skipped += 1
            continue
            
        meme_caps = item.get("meme_captions", [])
        img_caps = item.get("img_captions", [])
        
        # Positive samples (Metaphorical)
        for m_cap in meme_caps:
            samples.append(ClassificationSample(
                image_path=img_path,
                text=m_cap,
                label=1,
                post_id=post_id
            ))
            
        # Negative samples (Literal)
        for i_cap in img_caps:
            samples.append(ClassificationSample(
                image_path=img_path,
                text=i_cap,
                label=0,
                post_id=post_id
            ))
            
        if limit and len(samples) >= limit:
            break
            
    print(f"[Dataset] Loaded {len(samples)} classification samples from {json_path} (Skipped: {skipped})")
    return samples

class MemeCapClassificationDataset(Dataset):
    def __init__(self, samples: List[ClassificationSample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        return {
            "image": image,
            "text": sample.text,
            "label": sample.label,
            "post_id": sample.post_id
        }
