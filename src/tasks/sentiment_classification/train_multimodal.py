#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split

from transformers import CLIPProcessor, CLIPModel

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


EMOTION_7_TO_ID = {
    "Anger": 0,
    "Disgust": 1,
    "Fear": 2,
    "Joy": 3,
    "Neutral": 4,
    "Sadness": 5,
    "Surprise": 6,
}

EMOTION_3_TO_ID = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_label_maps(label_set):
    if label_set == "7class":
        label_to_id = EMOTION_7_TO_ID
    elif label_set == "3class":
        label_to_id = EMOTION_3_TO_ID
    else:
        raise ValueError(f"Unknown label_set: {label_set}")

    id_to_label = {v: k for k, v in label_to_id.items()}
    return label_to_id, id_to_label


def get_meme_caption(item):
    meme_caps = item.get("meme_captions", [])

    if (
        isinstance(meme_caps, list)
        and len(meme_caps) > 0
        and isinstance(meme_caps[0], str)
        and meme_caps[0].strip()
    ):
        return meme_caps[0].strip()

    title = item.get("title", "")
    if isinstance(title, str) and title.strip():
        return title.strip()

    return ""


def extract_clip_features(
    data,
    image_root,
    model,
    processor,
    device,
    label_key,
    label_to_id,
    batch_size=64,
):
    valid_items = []

    skipped_bad_label = 0
    skipped_missing_text = 0
    skipped_missing_image = 0
    skipped_image_error = 0

    for item in data:
        label_str = item.get(label_key, None)

        if label_str not in label_to_id:
            skipped_bad_label += 1
            continue

        text = get_meme_caption(item)
        if not text:
            skipped_missing_text += 1
            continue

        img_fname = item.get("img_fname", "")
        img_path = os.path.join(image_root, img_fname)

        if not img_fname or not os.path.exists(img_path):
            skipped_missing_image += 1
            continue

        valid_items.append((text, img_path, label_to_id[label_str]))

    print(f"Valid samples for CLIP extraction: {len(valid_items)}")
    print(f"Skipped bad/missing labels: {skipped_bad_label}")
    print(f"Skipped missing text: {skipped_missing_text}")
    print(f"Skipped missing image path: {skipped_missing_image}")

    image_features_list = []
    text_features_list = []
    labels_list = []

    model.eval()

    for start in tqdm(range(0, len(valid_items), batch_size), desc="Extracting CLIP features"):
        batch = valid_items[start:start + batch_size]

        texts = []
        images = []
        labels = []

        for text, img_path, label_id in batch:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                skipped_image_error += 1
                continue

            texts.append(text)
            images.append(image)
            labels.append(label_id)

        if not texts:
            continue

        with torch.no_grad():
            inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(device)

            img_embeds = model.get_image_features(pixel_values=inputs.pixel_values)
            txt_embeds = model.get_text_features(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )

            img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
            txt_embeds = txt_embeds / txt_embeds.norm(p=2, dim=-1, keepdim=True)

        image_features_list.append(img_embeds.cpu())
        text_features_list.append(txt_embeds.cpu())
        labels_list.append(torch.tensor(labels, dtype=torch.long))

    print(f"Skipped image loading errors: {skipped_image_error}")

    if not image_features_list:
        raise RuntimeError("No valid samples were extracted. Check image paths and label_key.")

    image_features = torch.cat(image_features_list, dim=0)
    text_features = torch.cat(text_features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    print(f"Image feature shape: {tuple(image_features.shape)}")
    print(f"Text feature shape: {tuple(text_features.shape)}")
    print(f"Label distribution IDs: {Counter(labels.tolist())}")

    return image_features, text_features, labels


class LateFusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=7, dropout=0.5):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, img_features, txt_features):
        fused = torch.cat([img_features, txt_features], dim=1)
        return self.network(fused)


class GatedFusionMLP(nn.Module):
    """
    Slightly more custom multimodal fusion:
    learns a gate between image and text embeddings, then classifies.
    Useful as an ablation against simple concatenation.
    """
    def __init__(self, feature_dim, hidden_dim=256, num_classes=7, dropout=0.5):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, img_features, txt_features):
        concat = torch.cat([img_features, txt_features], dim=1)
        gate = self.gate(concat)
        fused = gate * txt_features + (1.0 - gate) * img_features
        return self.classifier(fused)


def compute_class_weights(labels, num_classes, device):
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)

    total = counts.sum()
    weights = total / (num_classes * counts)

    return weights.to(device)


def evaluate(model, loader, device, id_to_label):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for img_feat, txt_feat, labels in loader:
            img_feat = img_feat.to(device)
            txt_feat = txt_feat.to(device)

            outputs = model(img_feat, txt_feat)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    target_names = [id_to_label[i] for i in range(len(id_to_label))]

    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(id_to_label))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(len(id_to_label))),
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds,
        "labels": all_labels,
    }


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    train_labels_for_weights,
    device,
    id_to_label,
    epochs=20,
    lr=5e-4,
    weight_decay=1e-2,
    label_smoothing=0.1,
    model_name="Multimodal Model",
):
    num_classes = len(id_to_label)
    class_weights = compute_class_weights(train_labels_for_weights, num_classes, device)

    print(f"\nClass weights for {model_name}:")
    for i, w in enumerate(class_weights.detach().cpu().tolist()):
        print(f"  {id_to_label[i]}: {w:.4f}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0

    print(f"\n{'=' * 60}")
    print(f"Training {model_name}")
    print(f"{'=' * 60}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for img_feat, txt_feat, labels in train_loader:
            img_feat = img_feat.to(device)
            txt_feat = txt_feat.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(img_feat, txt_feat)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, device, id_to_label)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Val Weighted F1: {val_metrics['weighted_f1']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device, id_to_label)

    print(f"\n>>> BEST VALIDATION EPOCH FOR {model_name}: {best_epoch}")
    print(f">>> FINAL TEST RESULTS FOR {model_name}")
    print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Macro F1:    {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {test_metrics['weighted_f1']:.4f}")

    return test_metrics, best_epoch


def make_train_val_loaders(img_feat, txt_feat, labels, batch_size, val_ratio, seed):
    dataset = TensorDataset(img_feat, txt_feat, labels)

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_indices = train_dataset.indices
    train_labels_for_weights = labels[train_indices]

    return train_loader, val_loader, train_labels_for_weights


def save_text_report(result_path, results, fusion_type):
    m = results["test_metrics"]

    result_text = (
        f"\n{'=' * 60}\n"
        f"TASK 2.3.c: MULTIMODAL SENTIMENT CLASSIFICATION\n"
        f"{'=' * 60}\n"
        f"Fusion Type : {fusion_type}\n"
        f"Accuracy    : {m['accuracy']:.4f}\n"
        f"Macro F1    : {m['macro_f1']:.4f}\n"
        f"Weighted F1 : {m['weighted_f1']:.4f}\n"
        f"{'=' * 60}\n"
    )

    print(result_text)

    with open(result_path, "w", encoding="utf-8") as f:
        f.write(result_text)


def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    label_to_id, id_to_label = get_label_maps(args.label_set)
    num_classes = len(label_to_id)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(vars(args), out_dir / "train_multimodal_args.json")

    print("\nLoading label JSON files...")
    train_data = load_json(args.train_labels)
    test_data = load_json(args.test_labels)

    print(f"Train JSON samples: {len(train_data)}")
    print(f"Test JSON samples: {len(test_data)}")
    print(f"Using label_key: {args.label_key}")
    print(f"Using label_set: {args.label_set}")

    cache_path = out_dir / f"clip_multimodal_cache_{args.label_set}_{args.cache_name}.pt"

    if args.use_cache and cache_path.exists():
        print(f"\nLoading cached CLIP features from: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")

        train_img_feat = cache["train_img_feat"]
        train_txt_feat = cache["train_txt_feat"]
        train_labels = cache["train_labels"]
        test_img_feat = cache["test_img_feat"]
        test_txt_feat = cache["test_txt_feat"]
        test_labels = cache["test_labels"]

    else:
        print(f"\nLoading CLIP model: {args.clip_model}")
        clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
        clip_processor = CLIPProcessor.from_pretrained(args.clip_model)

        print("\nExtracting train CLIP features...")
        train_img_feat, train_txt_feat, train_labels = extract_clip_features(
            data=train_data,
            image_root=args.image_root,
            model=clip_model,
            processor=clip_processor,
            device=device,
            label_key=args.label_key,
            label_to_id=label_to_id,
            batch_size=args.feature_batch_size,
        )

        print("\nExtracting test CLIP features...")
        test_img_feat, test_txt_feat, test_labels = extract_clip_features(
            data=test_data,
            image_root=args.image_root,
            model=clip_model,
            processor=clip_processor,
            device=device,
            label_key=args.label_key,
            label_to_id=label_to_id,
            batch_size=args.feature_batch_size,
        )

        del clip_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if args.use_cache:
            print(f"\nSaving CLIP feature cache to: {cache_path}")
            torch.save(
                {
                    "train_img_feat": train_img_feat,
                    "train_txt_feat": train_txt_feat,
                    "train_labels": train_labels,
                    "test_img_feat": test_img_feat,
                    "test_txt_feat": test_txt_feat,
                    "test_labels": test_labels,
                },
                cache_path,
            )

    feature_dim = train_img_feat.shape[1]
    fused_dim = feature_dim * 2

    print(f"\nCLIP feature dimension: {feature_dim}")
    print(f"Fused feature dimension: {fused_dim}")

    train_loader, val_loader, train_labels_for_weights = make_train_val_loaders(
        img_feat=train_img_feat,
        txt_feat=train_txt_feat,
        labels=train_labels,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    test_loader = DataLoader(
        TensorDataset(test_img_feat, test_txt_feat, test_labels),
        batch_size=args.batch_size,
        shuffle=False,
    )

    if args.fusion_type == "late_concat":
        model = LateFusionMLP(
            input_dim=fused_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout,
        ).to(device)
        model_name = "Late Fusion Concatenation MLP"

    elif args.fusion_type == "gated":
        model = GatedFusionMLP(
            feature_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout,
        ).to(device)
        model_name = "Gated Fusion MLP"

    else:
        raise ValueError(f"Unknown fusion_type: {args.fusion_type}")

    test_metrics, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_labels_for_weights=train_labels_for_weights,
        device=device,
        id_to_label=id_to_label,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        model_name=model_name,
    )

    final_results = {
        "label_set": args.label_set,
        "label_key": args.label_key,
        "clip_model": args.clip_model,
        "fusion_type": args.fusion_type,
        "best_epoch_by_val_macro_f1": best_epoch,
        "test_metrics": {
            k: v for k, v in test_metrics.items()
            if k not in ["predictions", "labels"]
        },
    }

    save_json(final_results, out_dir / "multimodal_results.json")
    save_text_report(out_dir / "multimodal_results.txt", final_results, args.fusion_type)

    torch.save(model.state_dict(), out_dir / f"multimodal_{args.fusion_type}_mlp.pt")

    print(f"\nSaved all outputs to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_labels", type=str, required=True)
    parser.add_argument("--test_labels", type=str, required=True)
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--label_key", type=str, required=True)
    parser.add_argument("--label_set", type=str, choices=["7class", "3class"], default="7class")

    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--output_dir", type=str, default="outputs/sentiment_classification/multimodal")
    parser.add_argument("--cache_name", type=str, default="default")
    parser.add_argument("--use_cache", action="store_true")

    parser.add_argument("--fusion_type", type=str, choices=["late_concat", "gated"], default="late_concat")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--feature_batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)