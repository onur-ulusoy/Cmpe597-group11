import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def simple_tokenize(text: str) -> List[str]:
    if text is None:
        return []
    text = text.lower().strip()
    return re.findall(r"\b\w+\b", text)


def first_nonempty_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for x in value:
            if isinstance(x, str) and x.strip():
                return x.strip()
    return ""


def try_resolve_image_path(image_root: Path, img_fname: str) -> Optional[Path]:
    candidate = image_root / img_fname
    if candidate.exists():
        return candidate.resolve()

    basename = Path(img_fname).name
    candidate2 = image_root / basename
    if candidate2.exists():
        return candidate2.resolve()

    matches = list(image_root.rglob(basename))
    if matches:
        return matches[0].resolve()

    return None


class Vocab:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, stoi: Dict[str, int]):
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}
        self.pad_idx = stoi[self.PAD_TOKEN]
        self.unk_idx = stoi[self.UNK_TOKEN]

    @classmethod
    def build(cls, texts: List[str], min_freq: int = 1):
        counter = Counter()
        for text in texts:
            counter.update(simple_tokenize(text))

        stoi = {
            cls.PAD_TOKEN: 0,
            cls.UNK_TOKEN: 1,
        }

        for token, freq in counter.items():
            if freq >= min_freq:
                stoi[token] = len(stoi)

        return cls(stoi)

    def encode(self, text: str, max_length: int) -> List[int]:
        tokens = simple_tokenize(text)
        ids = [self.stoi.get(tok, self.unk_idx) for tok in tokens[:max_length]]
        if len(ids) < max_length:
            ids += [self.pad_idx] * (max_length - len(ids))
        return ids

    def __len__(self):
        return len(self.stoi)


def build_image_transform(image_size: int = 224, train: bool = True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if train:
        return transforms.Compose([
            # 1. Resize directly to square (distorts aspect ratio but keeps text)
            transforms.Resize((image_size, image_size)),
            
            # 2. Mild Color Jitter (helps generalization)
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            
            # 3. Random Horizontal Flip? NO. 
            # Flipping a meme mirrors the text, making it unreadable.
            
            transforms.ToTensor(),
            normalize,
        ])

    # Validation/Test
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

def load_memecap_records(json_path: str, image_root: str):
    json_path = Path(json_path)
    image_root = Path(image_root)

    print(f"[Dataset] JSON path: {json_path.resolve()}")
    print(f"[Dataset] Image root: {image_root.resolve()}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    records = []
    skipped_missing_caption = 0
    skipped_missing_image = 0
    missing_examples = []

    for item in raw_data:
        img_fname = first_nonempty_string(item.get("img_fname") or item.get("img"))
        caption = first_nonempty_string(item.get("meme_captions") or item.get("caption"))
        title = first_nonempty_string(item.get("title", ""))

        if not img_fname or not caption:
            skipped_missing_caption += 1
            continue

        image_path = try_resolve_image_path(image_root, img_fname)
        if image_path is None:
            skipped_missing_image += 1
            if len(missing_examples) < 10:
                missing_examples.append(img_fname)
            continue

        records.append({
            "image_path": str(image_path),
            "caption": caption,
            "title": title,
        })

    print(f"[Dataset] Loaded {len(records)} samples from {json_path}")
    print(f"[Dataset] Skipped missing caption/metadata: {skipped_missing_caption}")
    print(f"[Dataset] Skipped missing image files     : {skipped_missing_image}")

    return records


def build_vocab_from_records(records, min_freq: int = 1, include_titles: bool = True) -> Vocab:
    texts = []
    for item in records:
        if item["caption"]:
            texts.append(item["caption"])
        if include_titles and item["title"]:
            texts.append(item["title"])

    vocab = Vocab.build(texts, min_freq=min_freq)
    print(f"[Vocab] Size: {len(vocab)}")
    return vocab


class MemeCapCustomDataset(Dataset):
    def __init__(self, records, vocab: Vocab, max_text_len: int = 40, image_transform=None):
        self.records = records
        self.vocab = vocab
        self.max_text_len = max_text_len
        self.image_transform = image_transform or build_image_transform()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        sample = self.records[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.image_transform(image)

        caption_ids = self.vocab.encode(sample["caption"], self.max_text_len)
        caption_ids = torch.tensor(caption_ids, dtype=torch.long)
        attention_mask = (caption_ids != self.vocab.pad_idx).long()

        return {
            "image": image,
            "caption_ids": caption_ids,
            "caption_mask": attention_mask,
            "raw_caption": sample["caption"],
            "raw_title": sample["title"],
            "image_path": sample["image_path"],
            "index": idx,
        }