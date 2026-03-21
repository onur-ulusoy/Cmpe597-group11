import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def load_json_list(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


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


@dataclass
class MemeSample:
    idx: int
    post_id: str
    image_path: Path
    title: str
    caption: str
    img_fname: str


class MemeCapDataset:
    def __init__(self, samples: List[MemeSample], json_path: Path, missing_examples: List[str]):
        self.samples = samples
        self.json_path = json_path
        self.missing_examples = missing_examples

    @classmethod
    def from_paths(
        cls,
        data_dir: str = "data",
        test_json: Optional[str] = None,
        image_root: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> "MemeCapDataset":
        data_dir = Path(data_dir)
        json_path = Path(test_json) if test_json else data_dir / "memes-test.json"
        image_root_path = Path(image_root) if image_root else data_dir / "memes"

        if not json_path.exists():
            raise FileNotFoundError(f"Could not find test json: {json_path}")
        if not image_root_path.exists():
            raise FileNotFoundError(f"Could not find image root: {image_root_path}")

        raw_samples = load_json_list(json_path)
        if limit is not None:
            raw_samples = raw_samples[:limit]

        parsed: List[MemeSample] = []
        skipped_missing_caption = 0
        skipped_missing_image = 0
        missing_examples: List[str] = []

        for idx, item in enumerate(raw_samples):
            # Handle both "img_fname" and "img" keys
            img_fname = first_nonempty_string(item.get("img_fname") or item.get("img"))
            
            # Handle title (may be missing)
            title = first_nonempty_string(item.get("title", ""))
            
            # Handle both "meme_captions" (list) and "caption" (string)
            caption = first_nonempty_string(
                item.get("meme_captions") or item.get("caption")
            )
            
            post_id = str(item.get("post_id", idx))

            if not img_fname or not caption:
                skipped_missing_caption += 1
                continue

            image_path = try_resolve_image_path(image_root_path, img_fname)
            if image_path is None:
                skipped_missing_image += 1
                if len(missing_examples) < 20:
                    missing_examples.append(img_fname)
                continue

            parsed.append(
                MemeSample(
                    idx=idx,
                    post_id=post_id,
                    image_path=image_path,
                    title=title,
                    caption=caption,
                    img_fname=img_fname,
                )
            )

        if not parsed:
            raise ValueError("No valid samples loaded.")

        print(f"Loaded {len(parsed)} valid test samples from {json_path}")
        print(f"Skipped missing caption/metadata: {skipped_missing_caption}")
        print(f"Skipped missing image files     : {skipped_missing_image}")
        if missing_examples:
            print("\nExample missing img_fname values:")
            for x in missing_examples[:10]:
                print(f"  - {x}")

        return cls(parsed, json_path, missing_examples)


def compute_ranks(score_matrix: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(score_matrix, dim=1, descending=True)
    gold = torch.arange(score_matrix.shape[0], device=score_matrix.device).unsqueeze(1)
    ranks = (order == gold).nonzero(as_tuple=False)[:, 1] + 1
    return ranks


def compute_metrics(score_matrix: torch.Tensor) -> Dict[str, float]:
    ranks = compute_ranks(score_matrix).float()
    return {
        "R@1": (ranks <= 1).float().mean().item(),
        "R@5": (ranks <= 5).float().mean().item(),
        "MRR": (1.0 / ranks).mean().item(),
        "MedR": ranks.median().item(),
        "MeanR": ranks.mean().item(),
    }


def print_metrics(name: str, metrics: Dict[str, float]) -> None:
    print(f"\n=== {name} ===")
    print(f"R@1   : {metrics['R@1'] * 100:.2f}%")
    print(f"R@5   : {metrics['R@5'] * 100:.2f}%")
    print(f"MRR   : {metrics['MRR'] * 100:.2f}%")
    print(f"MedR  : {metrics['MedR']:.2f}")
    print(f"MeanR : {metrics['MeanR']:.2f}")


def make_query_embeddings(
    image_emb: torch.Tensor,
    title_emb: Optional[torch.Tensor],
    input_type: str,
    alpha: float,
) -> torch.Tensor:
    image_emb = l2_normalize(image_emb)

    if input_type == "type1":
        return image_emb

    if input_type == "type2":
        if title_emb is None:
            raise ValueError("title embeddings are required for type2")
        title_emb = l2_normalize(title_emb)
        fused = alpha * image_emb + (1.0 - alpha) * title_emb
        return l2_normalize(fused)

    raise ValueError(f"Unknown input_type: {input_type}")


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_per_query_csv(path: Path, dataset: MemeCapDataset, score_matrix: torch.Tensor) -> None:
    ranks = compute_ranks(score_matrix).cpu().tolist()
    topk = torch.topk(score_matrix.cpu(), k=min(5, score_matrix.shape[1]), dim=1).indices.tolist()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_index",
            "post_id",
            "img_fname",
            "gold_rank",
            "gold_caption",
            "top1_index",
            "top1_caption",
            "top5_indices",
        ])

        for i, sample in enumerate(dataset.samples):
            top1_idx = topk[i][0]
            writer.writerow([
                i,
                sample.post_id,
                sample.img_fname,
                ranks[i],
                sample.caption,
                top1_idx,
                dataset.samples[top1_idx].caption,
                " ".join(map(str, topk[i])),
            ])
