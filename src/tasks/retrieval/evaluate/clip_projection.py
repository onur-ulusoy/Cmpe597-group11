#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import open_clip

sys.path.append(os.getcwd())
from src.common.metrics import compute_recall_metrics


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, output_dim=256, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


class ProjectionRetrievalModel(nn.Module):
    def __init__(self, input_dim=768, output_dim=256, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.query_proj = ProjectionHead(input_dim, output_dim, hidden_dim, dropout)
        self.caption_proj = ProjectionHead(input_dim, output_dim, hidden_dim, dropout)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, query_embs, caption_embs):
        q = self.query_proj(query_embs)
        c = self.caption_proj(caption_embs)
        logit_scale = self.logit_scale.exp().clamp(max=100)
        return logit_scale * q @ c.T


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_first_string(value):
    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
        return value[0]
    if isinstance(value, str):
        return value
    return ""


def build_metaphor_text(item):
    metaphors = item.get("metaphors", [])
    parts = []

    if isinstance(metaphors, list):
        for m in metaphors:
            if isinstance(m, dict):
                metaphor = m.get("metaphor", "")
                meaning = m.get("meaning", "")
                if metaphor and meaning:
                    parts.append(f"{metaphor} means {meaning}")
                elif metaphor:
                    parts.append(str(metaphor))
                elif meaning:
                    parts.append(str(meaning))

    return "; ".join(parts)


def load_memecap_items(json_path, image_root):
    data = load_json(json_path)
    items = []

    for item in data:
        img_fname = item.get("img_fname", "")
        image_path = os.path.join(image_root, img_fname)

        caption = get_first_string(item.get("meme_captions", ""))
        title = item.get("title", "")
        image_caption = get_first_string(item.get("img_captions", ""))
        metaphor_text = build_metaphor_text(item)

        if not caption:
            continue
        if not img_fname or not os.path.exists(image_path):
            continue

        items.append(
            {
                "image_path": image_path,
                "caption": caption,
                "title": title if isinstance(title, str) else "",
                "image_caption": image_caption,
                "metaphor_text": metaphor_text,
                "post_id": item.get("post_id", ""),
            }
        )

    return items


def build_target_text(item, mode):
    caption = item["caption"]
    title = item["title"]
    image_caption = item["image_caption"]
    metaphor_text = item["metaphor_text"]

    if mode == "caption":
        return caption

    if mode == "title_caption":
        return f"{title} [SEP] {caption}" if title else caption

    if mode == "caption_imagecaption":
        return f"{caption} [SEP] {image_caption}" if image_caption else caption

    if mode == "caption_metaphor":
        return f"{caption} [SEP] {metaphor_text}" if metaphor_text else caption

    raise ValueError(f"Unknown target_text_mode: {mode}")


def encode_images(model, preprocess, image_paths, device, batch_size):
    model.eval()
    all_embs = []

    with torch.no_grad():
        for start in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[start:start + batch_size]
            images = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (224, 224), (0, 0, 0))
                images.append(preprocess(img))

            images = torch.stack(images).to(device)
            embs = model.encode_image(images)
            embs = F.normalize(embs, p=2, dim=-1)
            all_embs.append(embs.cpu())

    return torch.cat(all_embs, dim=0)


def encode_texts(model, tokenizer, texts, device, batch_size):
    model.eval()
    all_embs = []

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[start:start + batch_size]
            tokens = tokenizer(batch_texts).to(device)
            embs = model.encode_text(tokens)
            embs = F.normalize(embs, p=2, dim=-1)
            all_embs.append(embs.cpu())

    return torch.cat(all_embs, dim=0)


def make_query_embeddings(image_embs, title_embs, model_type, fusion_alpha):
    if model_type == "type1":
        return F.normalize(image_embs, p=2, dim=-1)

    if model_type == "type2":
        image_embs = F.normalize(image_embs, p=2, dim=-1)
        title_embs = F.normalize(title_embs, p=2, dim=-1)
        query = fusion_alpha * image_embs + (1.0 - fusion_alpha) * title_embs
        return F.normalize(query, p=2, dim=-1)

    raise ValueError(f"Unknown model_type: {model_type}")


def extract_or_load_features(items, ckpt_args, eval_args, device, out_dir):
    model_type = ckpt_args["model_type"]
    target_text_mode = eval_args.target_text_mode or ckpt_args["target_text_mode"]

    cache_path = out_dir / f"test_clip_features_{model_type}_{target_text_mode}.pt"

    if eval_args.use_cache and cache_path.exists():
        print(f"Loading cached features: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print("\nExtracting test OpenCLIP features...")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        ckpt_args["clip_model"],
        pretrained=ckpt_args["pretrained"],
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(ckpt_args["clip_model"])

    image_paths = [x["image_path"] for x in items]
    titles = [x["title"] for x in items]
    target_texts = [build_target_text(x, target_text_mode) for x in items]

    image_embs = encode_images(clip_model, preprocess, image_paths, device, eval_args.batch_size)
    title_embs = encode_texts(clip_model, tokenizer, titles, device, eval_args.batch_size)
    caption_embs = encode_texts(clip_model, tokenizer, target_texts, device, eval_args.batch_size)

    del clip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    features = {
        "image_embs": image_embs,
        "title_embs": title_embs,
        "caption_embs": caption_embs,
        "titles": titles,
        "target_texts": target_texts,
        "image_paths": image_paths,
    }

    if eval_args.use_cache:
        print(f"Saving feature cache: {cache_path}")
        torch.save(features, cache_path)

    return features


@torch.no_grad()
def get_scores(model, query_embs, caption_embs, device):
    model.eval()

    query_embs = query_embs.to(device)
    caption_embs = caption_embs.to(device)

    q = model.query_proj(query_embs)
    c = model.caption_proj(caption_embs)

    return (q @ c.T).cpu()


def evaluate_scores(scores):
    image_to_text = compute_recall_metrics(scores, ks=(1, 5, 10))
    text_to_image = compute_recall_metrics(scores.T, ks=(1, 5, 10))

    mean_metrics = {
        f"mean_{k}": (image_to_text[k] + text_to_image[k]) / 2.0
        for k in image_to_text
    }

    return {
        "image_to_text": image_to_text,
        "text_to_image": text_to_image,
        "mean": mean_metrics,
    }


def compute_ranks(scores):
    n = scores.size(0)
    sorted_idx = scores.argsort(dim=1, descending=True)

    ranks = []
    for i in range(n):
        rank = (sorted_idx[i] == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    return ranks


def plot_rank_histogram(ranks, path):
    clipped = [min(r, 50) for r in ranks]

    plt.figure(figsize=(8, 5))
    plt.hist(clipped, bins=50)
    plt.xlabel("Correct Caption Rank")
    plt.ylabel("Number of Queries")
    plt.title("Rank Distribution of Correct Captions")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_similarity_matrix(scores, path, max_items=64):
    n = min(max_items, scores.size(0))
    sub = scores[:n, :n].numpy()

    plt.figure(figsize=(7, 6))
    plt.imshow(sub, aspect="auto")
    plt.colorbar(label="Similarity")
    plt.xlabel("Candidate Captions")
    plt.ylabel("Query Memes")
    plt.title(f"Similarity Matrix Sample ({n}x{n})")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_alpha_sweep(alpha_results, path):
    alphas = [x["alpha"] for x in alpha_results]
    r1 = [x["metrics"]["image_to_text"]["R@1"] * 100 for x in alpha_results]
    r5 = [x["metrics"]["image_to_text"]["R@5"] * 100 for x in alpha_results]
    mrr = [x["metrics"]["image_to_text"]["MRR"] * 100 for x in alpha_results]

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, r1, marker="o", label="R@1")
    plt.plot(alphas, r5, marker="o", label="R@5")
    plt.plot(alphas, mrr, marker="o", label="MRR")
    plt.xlabel("Fusion Alpha")
    plt.ylabel("Metric (%)")
    plt.title("Type 2 Fusion Alpha Sweep")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint["args"]

    model_type = ckpt_args["model_type"]
    target_text_mode = args.target_text_mode or ckpt_args["target_text_mode"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        {
            "checkpoint": args.checkpoint,
            "checkpoint_args": ckpt_args,
            "eval_args": vars(args),
            "resolved_model_type": model_type,
            "resolved_target_text_mode": target_text_mode,
        },
        output_dir / "eval_args.json",
    )

    model = ProjectionRetrievalModel(
        input_dim=checkpoint["input_dim"],
        output_dim=ckpt_args["proj_dim"],
        hidden_dim=ckpt_args["hidden_dim"],
        dropout=ckpt_args["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    items = load_memecap_items(args.test_json, args.image_root)
    features = extract_or_load_features(items, ckpt_args, args, device, output_dir)

    fusion_alpha = args.fusion_alpha
    if fusion_alpha is None:
        fusion_alpha = ckpt_args["fusion_alpha"]

    query_embs = make_query_embeddings(
        features["image_embs"],
        features["title_embs"],
        model_type,
        fusion_alpha,
    )

    caption_embs = features["caption_embs"]

    scores = get_scores(model, query_embs, caption_embs, device)
    metrics = evaluate_scores(scores)

    ranks = compute_ranks(scores)

    save_json(metrics, output_dir / "final_test_metrics.json")
    plot_rank_histogram(ranks, output_dir / "rank_histogram.png")
    plot_similarity_matrix(scores, output_dir / "similarity_matrix_sample.png")

    print("\nFinal Test Metrics")
    print(json.dumps(metrics, indent=4))

    if model_type == "type2" and args.alpha_sweep:
        alpha_results = []

        for alpha in np.arange(0.0, 1.01, 0.1):
            alpha = round(float(alpha), 2)

            query_alpha = make_query_embeddings(
                features["image_embs"],
                features["title_embs"],
                model_type,
                alpha,
            )

            scores_alpha = get_scores(model, query_alpha, caption_embs, device)
            metrics_alpha = evaluate_scores(scores_alpha)

            alpha_results.append(
                {
                    "alpha": alpha,
                    "metrics": metrics_alpha,
                }
            )

        save_json(alpha_results, output_dir / "alpha_sweep.json")
        plot_alpha_sweep(alpha_results, output_dir / "alpha_sweep.png")

    print(f"\nSaved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_json", type=str, default="data/memes-test.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--output_dir", type=str, default="outputs/retrieval/clip_projection_eval")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--use_cache", action="store_true")

    parser.add_argument(
        "--target_text_mode",
        choices=["caption", "title_caption", "caption_imagecaption", "caption_metaphor"],
        default=None,
        help="Optional override. By default uses the mode saved in the checkpoint.",
    )

    parser.add_argument(
        "--fusion_alpha",
        type=float,
        default=None,
        help="Optional override. By default uses the alpha saved in the checkpoint.",
    )

    parser.add_argument("--alpha_sweep", action="store_true")

    args = parser.parse_args()
    main(args)