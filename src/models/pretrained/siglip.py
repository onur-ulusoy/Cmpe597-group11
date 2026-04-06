from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

class SigLIPBackend:
    def __init__(self, checkpoint: str, device: str):
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(checkpoint, use_fast=False)
        self.model = AutoModel.from_pretrained(checkpoint).to(device)
        self.model.eval()

    def _extract_image_features(self, outputs) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            return outputs.image_embeds
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        raise RuntimeError(f"Could not extract image features from output type {type(outputs)}")

    def _extract_text_features(self, outputs) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return outputs
        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            return outputs.text_embeds
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        raise RuntimeError(f"Could not extract text features from output type {type(outputs)}")

    @torch.no_grad()
    def encode_images(self, image_paths: Sequence[Path], batch_size: int) -> torch.Tensor:
        all_feats = []
        for start in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images (SigLIP)"):
            batch_paths = image_paths[start:start + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]

            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model.get_image_features(**inputs)
            feats = self._extract_image_features(outputs)
            feats = F.normalize(feats.float(), p=2, dim=-1)
            all_feats.append(feats.cpu())

        return torch.cat(all_feats, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], batch_size: int) -> torch.Tensor:
        all_feats = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding texts (SigLIP)"):
            batch_texts = list(texts[start:start + batch_size])

            inputs = self.processor(
                text=batch_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model.get_text_features(**inputs)
            feats = self._extract_text_features(outputs)
            feats = F.normalize(feats.float(), p=2, dim=-1)
            all_feats.append(feats.cpu())

        return torch.cat(all_feats, dim=0)