#!/usr/bin/env python3
import argparse
import json
import os
import random
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
from src.common.utils import set_seed


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
        logits = logit_scale * q @ c.T
        return logits, q, c


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def split_items(items, val_ratio=0.1, seed=42):
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    val_size = int(len(items) * val_ratio)
    return items[val_size:], items[:val_size]


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


def extract_or_load_features(items, args, split_name, run_dir, device):
    cache_path = run_dir / f"{split_name}_clip_features_{args.model_type}_{args.target_text_mode}.pt"

    if args.use_cache and cache_path.exists():
        print(f"Loading cached features: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print(f"\nExtracting frozen OpenCLIP features for {split_name}...")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model,
        pretrained=args.pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)

    image_paths = [x["image_path"] for x in items]
    titles = [x["title"] for x in items]
    target_texts = [build_target_text(x, args.target_text_mode) for x in items]

    image_embs = encode_images(clip_model, preprocess, image_paths, device, args.feature_batch_size)
    title_embs = encode_texts(clip_model, tokenizer, titles, device, args.feature_batch_size)
    caption_embs = encode_texts(clip_model, tokenizer, target_texts, device, args.feature_batch_size)

    del clip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    features = {
        "image_embs": image_embs,
        "title_embs": title_embs,
        "caption_embs": caption_embs,
        "target_texts": target_texts,
        "titles": titles,
        "image_paths": image_paths,
    }

    if args.use_cache:
        print(f"Saving feature cache: {cache_path}")
        torch.save(features, cache_path)

    return features


def make_query_embeddings(image_embs, title_embs, model_type, fusion_alpha):
    if model_type == "type1":
        return F.normalize(image_embs, p=2, dim=-1)

    if model_type == "type2":
        image_embs = F.normalize(image_embs, p=2, dim=-1)
        title_embs = F.normalize(title_embs, p=2, dim=-1)
        query = fusion_alpha * image_embs + (1.0 - fusion_alpha) * title_embs
        return F.normalize(query, p=2, dim=-1)

    raise ValueError(f"Unknown model_type: {model_type}")


def contrastive_loss(logits, label_smoothing=0.0):
    n = logits.size(0)
    labels = torch.arange(n, device=logits.device)
    loss_q = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    loss_c = F.cross_entropy(logits.T, labels, label_smoothing=label_smoothing)
    return (loss_q + loss_c) / 2.0


@torch.no_grad()
def compute_hard_negative_indices(query_embs, caption_embs, top_h):
    scores = query_embs @ caption_embs.T
    n = scores.size(0)
    scores[torch.arange(n), torch.arange(n)] = -1e9
    return scores.topk(k=top_h, dim=1).indices


def hard_negative_loss(q_proj, pos_c_proj, hard_c_proj, logit_scale):
    bsz, h, dim = hard_c_proj.shape

    pos_scores = (q_proj * pos_c_proj).sum(dim=-1, keepdim=True)
    hard_scores = torch.bmm(hard_c_proj, q_proj.unsqueeze(-1)).squeeze(-1)

    logits = torch.cat([pos_scores, hard_scores], dim=1) * logit_scale
    labels = torch.zeros(bsz, dtype=torch.long, device=q_proj.device)

    return F.cross_entropy(logits, labels)


@torch.no_grad()
def evaluate_model(model, query_embs, caption_embs, device):
    model.eval()

    query_embs = query_embs.to(device)
    caption_embs = caption_embs.to(device)

    q = model.query_proj(query_embs)
    c = model.caption_proj(caption_embs)

    scores = q @ c.T

    image_to_text = compute_recall_metrics(scores.cpu(), ks=(1, 5, 10))
    text_to_image = compute_recall_metrics(scores.T.cpu(), ks=(1, 5, 10))

    mean_metrics = {
        f"mean_{k}": (image_to_text[k] + text_to_image[k]) / 2.0
        for k in image_to_text
    }

    return {
        "image_to_text": image_to_text,
        "text_to_image": text_to_image,
        "mean": mean_metrics,
    }


def selection_score(metrics):
    return metrics["image_to_text"]["R@5"] + 0.5 * metrics["image_to_text"]["R@1"]


def plot_loss_curve(history, path):
    epochs = [x["epoch"] for x in history["epochs"]]
    losses = [x["train_loss"] for x in history["epochs"]]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Projection Head Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_recall_curve(history, path):
    epochs = [x["epoch"] for x in history["epochs"]]
    r1 = [x["val_metrics"]["image_to_text"]["R@1"] * 100 for x in history["epochs"]]
    r5 = [x["val_metrics"]["image_to_text"]["R@5"] * 100 for x in history["epochs"]]
    r10 = [x["val_metrics"]["image_to_text"]["R@10"] * 100 for x in history["epochs"]]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, r1, marker="o", label="R@1")
    plt.plot(epochs, r5, marker="o", label="R@5")
    plt.plot(epochs, r10, marker="o", label="R@10")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Recall (%)")
    plt.title("Validation Retrieval Recall")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    run_dir = Path(args.output_dir) / args.model_type / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), run_dir / "args.json")

    all_train_items = load_memecap_items(args.train_json, args.image_root)
    train_items, val_items = split_items(all_train_items, val_ratio=args.val_ratio, seed=args.seed)

    print(f"Train items: {len(train_items)}")
    print(f"Val items: {len(val_items)}")

    train_features = extract_or_load_features(train_items, args, "train", run_dir, device)
    val_features = extract_or_load_features(val_items, args, "val", run_dir, device)

    train_query = make_query_embeddings(
        train_features["image_embs"],
        train_features["title_embs"],
        args.model_type,
        args.fusion_alpha,
    )
    train_caption = train_features["caption_embs"]

    val_query = make_query_embeddings(
        val_features["image_embs"],
        val_features["title_embs"],
        args.model_type,
        args.fusion_alpha,
    )
    val_caption = val_features["caption_embs"]

    input_dim = train_query.shape[1]

    model = ProjectionRetrievalModel(
        input_dim=input_dim,
        output_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    hard_indices = None
    if args.use_hard_negatives:
        hard_indices = compute_hard_negative_indices(
            train_query,
            train_caption,
            args.num_hard_negatives,
        )

    n_train = train_query.size(0)
    all_indices = torch.arange(n_train)

    history = {
        "epochs": [],
        "best_epoch": None,
        "best_score": None,
        "best_val_metrics": None,
    }

    best_score = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()

        perm = all_indices[torch.randperm(n_train)]
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(range(0, n_train, args.batch_size), desc=f"Epoch {epoch}/{args.epochs}")

        for start in pbar:
            batch_idx = perm[start:start + args.batch_size]

            if len(batch_idx) < 2:
                continue

            q_batch = train_query[batch_idx].to(device)
            c_batch = train_caption[batch_idx].to(device)

            logits, q_proj, pos_c_proj = model(q_batch, c_batch)

            loss = contrastive_loss(
                logits,
                label_smoothing=args.label_smoothing,
            )

            if args.use_hard_negatives:
                h_idx = hard_indices[batch_idx]
                hard_caption_embs = train_caption[h_idx.reshape(-1)].to(device)
                hard_c_proj = model.caption_proj(hard_caption_embs)
                hard_c_proj = hard_c_proj.view(len(batch_idx), args.num_hard_negatives, -1)

                hn_loss = hard_negative_loss(
                    q_proj=q_proj,
                    pos_c_proj=pos_c_proj,
                    hard_c_proj=hard_c_proj,
                    logit_scale=model.logit_scale.exp().clamp(max=100),
                )

                loss = loss + args.hard_negative_weight * hn_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(num_batches, 1)

        val_metrics = evaluate_model(model, val_query, val_caption, device)
        score = selection_score(val_metrics)

        history["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_metrics": val_metrics,
                "selection_score": score,
            }
        )

        print(f"\nEpoch {epoch} | loss={avg_loss:.4f}")
        print(f"Val image-to-text: {val_metrics['image_to_text']}")
        print(f"Val text-to-image: {val_metrics['text_to_image']}")
        print(f"Selection score: {score:.6f}")

        if score > best_score:
            best_score = score
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            history["best_epoch"] = epoch
            history["best_score"] = best_score
            history["best_val_metrics"] = val_metrics

            torch.save(
                {
                    "model_state_dict": best_state,
                    "args": vars(args),
                    "input_dim": input_dim,
                    "best_epoch": epoch,
                    "best_score": best_score,
                    "best_val_metrics": val_metrics,
                },
                run_dir / "best_projection_model.pt",
            )

            print(f"Saved best checkpoint: {run_dir / 'best_projection_model.pt'}")

        save_json(history, run_dir / "train_history.json")
        plot_loss_curve(history, run_dir / "loss_curve.png")
        plot_recall_curve(history, run_dir / "recall_curve.png")

    print("\nTraining complete.")
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Best score: {history['best_score']}")
    print(f"Saved outputs to: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", choices=["type1", "type2"], required=True)
    parser.add_argument("--train_json", type=str, default="data/memes-trainval.json")
    parser.add_argument("--image_root", type=str, default="data/memes")

    parser.add_argument("--output_dir", type=str, default="outputs/retrieval/clip_projection")
    parser.add_argument("--run_name", type=str, default="default")

    parser.add_argument("--clip_model", type=str, default="ViT-L-14")
    parser.add_argument("--pretrained", type=str, default="laion2b_s32b_b82k")

    parser.add_argument(
        "--target_text_mode",
        choices=["caption", "title_caption", "caption_imagecaption", "caption_metaphor"],
        default="caption",
    )

    parser.add_argument("--fusion_alpha", type=float, default=0.5)

    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--feature_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--use_hard_negatives", action="store_true")
    parser.add_argument("--num_hard_negatives", type=int, default=10)
    parser.add_argument("--hard_negative_weight", type=float, default=0.2)

    args = parser.parse_args()
    train(args)