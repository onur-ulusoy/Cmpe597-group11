import re
from collections import Counter
from typing import Dict, List
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def simple_tokenize(text: str) -> List[str]:
    if text is None: return []
    text = text.lower().strip()
    return re.findall(r"\b\w+\b", text)

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

        stoi = {cls.PAD_TOKEN: 0, cls.UNK_TOKEN: 1}
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

    def __len__(self): return len(self.stoi)

def build_image_transform(image_size: int = 224, train: bool = True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

def build_vocab_from_records(records, min_freq: int = 1, include_titles: bool = True) -> Vocab:
    texts = []
    for item in records:
        if item.caption: texts.append(item.caption)
        if include_titles and item.title: texts.append(item.title)
    vocab = Vocab.build(texts, min_freq=min_freq)
    print(f"[Vocab] Size: {len(vocab)}")
    return vocab

class MemeCapCustomDataset(Dataset):
    def __init__(self, records, vocab: Vocab, max_text_len: int = 40, image_transform=None):
        self.records = records
        self.vocab = vocab
        self.max_text_len = max_text_len
        self.image_transform = image_transform or build_image_transform()

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        sample = self.records[idx]

        image = Image.open(sample.image_path).convert("RGB")
        image = self.image_transform(image)

        caption_ids = self.vocab.encode(sample.caption, self.max_text_len)
        caption_ids = torch.tensor(caption_ids, dtype=torch.long)
        caption_mask = (caption_ids != self.vocab.pad_idx).long()

        title_ids = self.vocab.encode(sample.title, self.max_text_len)
        title_ids = torch.tensor(title_ids, dtype=torch.long)
        title_mask = (title_ids != self.vocab.pad_idx).long()

        return {
            "image": image,
            "caption_ids": caption_ids,
            "caption_mask": caption_mask,
            "title_ids": title_ids,
            "title_mask": title_mask,
            "raw_caption": sample.caption,
            "raw_title": sample.title,
            "image_path": sample.image_path,
            "index": idx,
        }