#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

import open_clip
from peft import LoraConfig, get_peft_model

sys.path.append(os.getcwd())

from src.common.utils import set_seed, plot_loss
from src.common.dataset import load_memecap_records
from src.common.metrics import compute_recall_metrics


class OpenClipAdapter:
    def __init__(self, preprocess_fn, tokenizer):
        self.preprocess = preprocess_fn
        self.tokenizer = tokenizer

    def __call__(self, images=None, text=None, return_tensors="pt", **kwargs):
        data = {}

        if images is not None:
            if isinstance(images, list):
                pixel_values = torch.stack([self.preprocess(img) for img in images])
            else:
                pixel_values = self.preprocess(images).unsqueeze(0)
            data["pixel_values"] = pixel_values

        if text is not None:
            input_ids = self.tokenizer(text)
            data["input_ids"] = input_ids

        return data


class MemeCapFinetuneDataset(Dataset):
    def __init__(self, records, processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        sample = self.records[idx]

        try:
            image = Image.open(sample.image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        pixel_values = self.processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)

        caption_ids = self.processor(
            text=sample.caption,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        title_ids = self.processor(
            text=sample.title,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": caption_ids,
            "title_ids": title_ids,
        }


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def split_records(records, val_ratio=0.1, seed=42):
    records = list(records)
    rng = random.Random(seed)
    rng.shuffle(records)

    val_size = int(len(records) * val_ratio)
    val_records = records[:val_size]
    train_records = records[val_size:]

    return train_records, val_records


def get_openclip_base(peft_model):
    """
    PEFT wraps the OpenCLIP model. In the previous implementation,
    encode_image and encode_text were accessed through model.model.
    This helper keeps that behavior explicit.
    """
    return peft_model.model


def compute_query_features(base_model, images, titles, task, fusion_alpha):
    img_feats = base_model.encode_image(images)

    if task == "type1":
        query_feats = img_feats

    elif task == "type2":
        title_feats = base_model.encode_text(titles)

        img_feats = F.normalize(img_feats, p=2, dim=-1)
        title_feats = F.normalize(title_feats, p=2, dim=-1)

        query_feats = fusion_alpha * img_feats + (1.0 - fusion_alpha) * title_feats

    else:
        raise ValueError(f"Unknown task: {task}")

    query_feats = F.normalize(query_feats, p=2, dim=-1)
    return query_feats


def train_one_epoch(model, dataloader, optimizer, device, args):
    model.train()

    base_model = get_openclip_base(model)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        images = batch["pixel_values"].to(device, non_blocking=True)
        captions = batch["input_ids"].to(device, non_blocking=True)
        titles = batch["title_ids"].to(device, non_blocking=True)

        query_feats = compute_query_features(
            base_model=base_model,
            images=images,
            titles=titles,
            task=args.task,
            fusion_alpha=args.fusion_alpha,
        )

        cap_feats = base_model.encode_text(captions)
        cap_feats = F.normalize(cap_feats, p=2, dim=-1)

        logit_scale = base_model.logit_scale.exp()

        logits_per_query = logit_scale * query_feats @ cap_feats.t()
        logits_per_cap = logits_per_query.t()

        labels = torch.arange(images.size(0), device=device, dtype=torch.long)

        loss = (
            loss_img(logits_per_query, labels)
            + loss_txt(logits_per_cap, labels)
        ) / 2.0

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss


@torch.no_grad()
def encode_loader_for_retrieval(model, dataloader, device, args):
    model.eval()
    base_model = get_openclip_base(model)

    query_embs = []
    caption_embs = []

    for batch in tqdm(dataloader, desc="Encoding validation", leave=False):
        images = batch["pixel_values"].to(device, non_blocking=True)
        captions = batch["input_ids"].to(device, non_blocking=True)
        titles = batch["title_ids"].to(device, non_blocking=True)

        query_feats = compute_query_features(
            base_model=base_model,
            images=images,
            titles=titles,
            task=args.task,
            fusion_alpha=args.fusion_alpha,
        )

        cap_feats = base_model.encode_text(captions)
        cap_feats = F.normalize(cap_feats, p=2, dim=-1)

        query_embs.append(query_feats.cpu())
        caption_embs.append(cap_feats.cpu())

    query_embs = torch.cat(query_embs, dim=0)
    caption_embs = torch.cat(caption_embs, dim=0)

    return query_embs, caption_embs


@torch.no_grad()
def evaluate_retrieval(model, dataloader, device, args):
    query_embs, caption_embs = encode_loader_for_retrieval(
        model=model,
        dataloader=dataloader,
        device=device,
        args=args,
    )

    scores = query_embs @ caption_embs.T
    metrics = compute_recall_metrics(scores.cpu(), ks=(1, 5, 10))

    return metrics


def validation_score(metrics):
    """
    A simple scalar for checkpoint selection.
    Prioritizes R@5, while still rewarding R@1.
    """
    return metrics["R@5"] + 0.5 * metrics["R@1"]


def format_metrics(metrics):
    return (
        f"R@1={metrics['R@1'] * 100:.2f} | "
        f"R@5={metrics['R@5'] * 100:.2f} | "
        f"R@10={metrics['R@10'] * 100:.2f} | "
        f"MRR={metrics['MRR'] * 100:.2f}"
    )


def train(args):
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_save_dir = os.path.join("outputs", "retrieval", "finetune", args.task)
    os.makedirs(base_save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_save_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    best_dir = os.path.join(base_save_dir, "best_lora")
    plot_file = os.path.join(run_dir, "loss_plot.png")
    history_file = os.path.join(run_dir, "train_history.json")

    print(f"Device: {device}")
    print(f"Training retrieval LoRA task: {args.task.upper()}")
    print(f"Run directory: {run_dir}")
    print(f"Best adapter directory: {best_dir}")

    model_name = args.model_name
    pretrained = args.pretrained

    print(f"\nLoading OpenCLIP: {model_name} / {pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )

    tokenizer = open_clip.get_tokenizer(model_name)
    processor = OpenClipAdapter(preprocess, tokenizer)

    print("\nApplying LoRA...")
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    for param in model.parameters():
        param.requires_grad = False

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.to(device)

    print("\nLoading MemeCap records...")
    records = load_memecap_records(args.train_json, args.image_root)
    train_records, val_records = split_records(
        records,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"Total records: {len(records)}")
    print(f"Train records: {len(train_records)}")
    print(f"Validation records: {len(val_records)}")

    train_dataset = MemeCapFinetuneDataset(train_records, processor)
    val_dataset = MemeCapFinetuneDataset(val_records, processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_history = []
    history = {
        "args": vars(args),
        "train_loss": [],
        "val_metrics": [],
        "best_epoch": None,
        "best_score": None,
        "best_metrics": None,
        "run_dir": run_dir,
        "best_dir": best_dir,
    }

    best_score = -1.0
    best_epoch = -1
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        avg_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
        )

        loss_history.append(avg_loss)
        history["train_loss"].append({
            "epoch": epoch,
            "loss": avg_loss,
        })

        print(f"Train loss: {avg_loss:.4f}")

        val_metrics = evaluate_retrieval(
            model=model,
            dataloader=val_loader,
            device=device,
            args=args,
        )

        score = validation_score(val_metrics)

        history["val_metrics"].append({
            "epoch": epoch,
            **val_metrics,
            "selection_score": score,
        })

        print(f"Validation: {format_metrics(val_metrics)}")
        print(f"Selection score: {score:.6f}")

        plot_loss(loss_history, plot_file)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = val_metrics

            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)

            model.save_pretrained(best_dir)

            print(f"New best checkpoint saved to: {best_dir}")

            history["best_epoch"] = best_epoch
            history["best_score"] = best_score
            history["best_metrics"] = best_metrics

        save_json(history, history_file)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation metrics: {format_metrics(best_metrics)}")
    print(f"Best adapter saved to: {best_dir}")

    final_summary_path = os.path.join(base_save_dir, "best_summary.json")
    save_json(
        {
            "task": args.task,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_metrics": best_metrics,
            "best_adapter": best_dir,
            "run_dir": run_dir,
            "fusion_alpha": args.fusion_alpha,
        },
        final_summary_path,
    )

    print(f"Best summary saved to: {final_summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True, choices=["type1", "type2"])

    parser.add_argument("--train_json", type=str, default="data/memes-trainval.json")
    parser.add_argument("--image_root", type=str, default="data/memes")

    parser.add_argument("--model_name", type=str, default="ViT-L-14")
    parser.add_argument("--pretrained", type=str, default="laion2b_s32b_b82k")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["c_fc", "c_proj", "out_proj"],
        help="LoRA target module names.",
    )

    parser.add_argument(
        "--fusion_alpha",
        type=float,
        default=0.5,
        help="Only used for type2. Query = alpha * image + (1-alpha) * title.",
    )

    args = parser.parse_args()
    train(args)