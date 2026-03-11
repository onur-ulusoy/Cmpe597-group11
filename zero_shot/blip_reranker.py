from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from tqdm import tqdm


class BLIPReranker:
    def __init__(self, checkpoint: str, device: str):
        from transformers import AutoProcessor, BlipForImageTextRetrieval

        self.device = device
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model = BlipForImageTextRetrieval.from_pretrained(checkpoint).to(device)
        self.model.eval()

    def _extract_matching_score(self, outputs) -> torch.Tensor:
        """
        BLIP image-text retrieval returns a BlipImageTextMatchingModelOutput
        whose relevant field is usually `itm_score`.

        We convert it to a 1D score per pair:
        - if shape [B, 2], use P(match) = softmax[:, 1]
        - if shape [B, 1], use that single score
        - if another tensor-like format appears, flatten carefully
        """
        if hasattr(outputs, "itm_score") and outputs.itm_score is not None:
            score = outputs.itm_score

            if score.ndim == 2 and score.shape[1] == 2:
                # class 0 = not matched, class 1 = matched
                return torch.softmax(score.float(), dim=1)[:, 1].cpu()

            if score.ndim == 2 and score.shape[1] == 1:
                return score[:, 0].float().cpu()

            if score.ndim == 1:
                return score.float().cpu()

            return score.view(-1).float().cpu()

        # fallback for possible API differences
        if hasattr(outputs, "logits_per_image") and outputs.logits_per_image is not None:
            score = outputs.logits_per_image
            return score.view(-1).float().cpu()

        raise RuntimeError(
            f"Could not extract a BLIP matching score from output type {type(outputs)}"
        )

    @torch.no_grad()
    def rerank_topk(
        self,
        image_paths: Sequence[Path],
        captions: Sequence[str],
        base_scores: torch.Tensor,
        top_k: int,
        batch_size: int,
        blend_lambda: float,
    ) -> torch.Tensor:
        reranked = base_scores.clone()
        n = base_scores.shape[0]

        for qi in tqdm(range(n), desc="BLIP reranking"):
            k = min(top_k, n)
            top_idx = torch.topk(base_scores[qi], k=k).indices.tolist()
            image = Image.open(image_paths[qi]).convert("RGB")
            score_chunks = []

            for start in range(0, len(top_idx), batch_size):
                idx_chunk = top_idx[start:start + batch_size]
                text_chunk = [captions[j] for j in idx_chunk]

                inputs = self.processor(
                    images=[image] * len(text_chunk),
                    text=text_chunk,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Use the image-text matching head for reranking.
                outputs = self.model(**inputs, use_itm_head=True)
                blip_scores = self._extract_matching_score(outputs)
                score_chunks.append(blip_scores)

            blip_scores = torch.cat(score_chunks, dim=0)

            fused_scores = (
                blend_lambda * base_scores[qi, top_idx].cpu()
                + (1.0 - blend_lambda) * blip_scores
            )
            reranked[qi, top_idx] = fused_scores.to(reranked.device)

        return reranked