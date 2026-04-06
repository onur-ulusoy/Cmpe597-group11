from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

class OpenCLIPBackend:
    def __init__(self, model_name: str, pretrained: str, device: str):
        import open_clip

        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, image_paths: Sequence[Path], batch_size: int) -> torch.Tensor:
        all_feats = []
        for start in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images (OpenCLIP)"):
            batch_paths = image_paths[start:start + batch_size]
            images = [self.preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            batch = torch.stack(images).to(self.device)
            feats = self.model.encode_image(batch)
            feats = F.normalize(feats, p=2, dim=-1)
            all_feats.append(feats.cpu())
        return torch.cat(all_feats, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], batch_size: int) -> torch.Tensor:
        all_feats = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding texts (OpenCLIP)"):
            batch_texts = list(texts[start:start + batch_size])
            tokens = self.tokenizer(batch_texts).to(self.device)
            feats = self.model.encode_text(tokens)
            feats = F.normalize(feats, p=2, dim=-1)
            all_feats.append(feats.cpu())
        return torch.cat(all_feats, dim=0)