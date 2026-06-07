import argparse
import os
import sys
import random
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from src.common.classification_dataset import (
    load_classification_records,
    MemeCapClassificationDataset,
)
from src.common.classification_metrics import (
    compute_classification_metrics,
    print_classification_report,
)
from src.common.utils import set_seed, save_json, save_checkpoint, load_checkpoint
from src.models.pretrained.openclip import OpenCLIPBackend
from src.models.custom.caption_classification_model import MemeClassificationModel


def get_device(preference=None):
    if preference:
        return preference
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_records(records, val_ratio=0.1, seed=42):
    records = list(records)
    rng = random.Random(seed)
    rng.shuffle(records)

    val_size = int(len(records) * val_ratio)
    val_records = records[:val_size]
    train_records = records[val_size:]

    return train_records, val_records


def get_text_hash(text):
    import hashlib
    return hashlib.md5(text.strip().encode()).hexdigest()


class FeatureDataset(MemeCapClassificationDataset):
    def __init__(self, records, feature_dir=None):
        super().__init__(records)
        self.feature_dir = feature_dir

        if feature_dir:
            with open(os.path.join(feature_dir, "text_mapping.json"), "r", encoding="utf-8") as f:
                self.text_mapping = json.load(f)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.feature_dir:
            img_path = os.path.join(
                self.feature_dir,
                "images",
                f"{sample.image_path.name}.pt",
            )

            text = sample.text.strip()

            if text in self.text_mapping:
                t_hash = self.text_mapping[text]
            else:
                # fallback if mapping stores hashes implicitly
                t_hash = get_text_hash(text)

            text_path = os.path.join(
                self.feature_dir,
                "texts",
                f"{t_hash}.pt",
            )

            return {
                "img_emb": torch.load(img_path, map_location="cpu"),
                "text_emb": torch.load(text_path, map_location="cpu"),
                "label": sample.label,
            }

        return {
            "image_path": str(sample.image_path),
            "text": sample.text,
            "label": sample.label,
        }


def train_one_epoch(
    classifier,
    dataloader,
    backend,
    optimizer,
    criterion,
    device,
    use_features=False,
):
    classifier.train()

    running_loss = 0.0
    all_scores = []
    all_preds = []
    all_labels = []

    progress = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress:
        labels = batch["label"].to(device).float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        if use_features:
            img_emb = batch["img_emb"].to(device)
            text_emb = batch["text_emb"].to(device)
        else:
            img_paths = batch["image_path"]
            texts = batch["text"]

            with torch.no_grad():
                img_emb = backend.encode_images(img_paths, batch_size=len(img_paths)).to(device)
                text_emb = backend.encode_texts(texts, batch_size=len(texts)).to(device)

        logits = classifier(img_emb, text_emb)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        scores = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        preds = (scores >= 0.5).astype(int)

        all_scores.extend(scores.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["label"].numpy().tolist())

        progress.set_postfix(loss=f"{loss.item():.4f}")

    metrics = compute_classification_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_scores),
    )

    return running_loss / max(len(dataloader), 1), metrics


@torch.no_grad()
def evaluate(
    classifier,
    dataloader,
    backend,
    criterion,
    device,
    use_features=False,
    threshold=0.5,
):
    classifier.eval()

    running_loss = 0.0
    all_scores = []
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        labels = batch["label"].to(device).float().unsqueeze(1)

        if use_features:
            img_emb = batch["img_emb"].to(device)
            text_emb = batch["text_emb"].to(device)
        else:
            img_paths = batch["image_path"]
            texts = batch["text"]

            img_emb = backend.encode_images(img_paths, batch_size=len(img_paths)).to(device)
            text_emb = backend.encode_texts(texts, batch_size=len(texts)).to(device)

        logits = classifier(img_emb, text_emb)
        loss = criterion(logits, labels)

        running_loss += loss.item()

        scores = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (scores >= threshold).astype(int)

        all_scores.extend(scores.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["label"].numpy().tolist())

    metrics = compute_classification_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_scores),
    )

    return running_loss / max(len(dataloader), 1), metrics, np.array(all_labels), np.array(all_scores)


def tune_threshold(labels, scores):
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.linspace(0.05, 0.95, 91):
        preds = (scores >= threshold).astype(int)
        metrics = compute_classification_metrics(labels, preds, scores)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)

    return best_threshold, best_f1


def plot_loss_curve(history, out_path):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Caption Classification Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_f1_curve(history, out_path):
    epochs = list(range(1, len(history["val_metrics"]) + 1))
    val_f1 = [m["f1"] for m in history["val_metrics"]]
    val_acc = [m["accuracy"] for m in history["val_metrics"]]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_f1, marker="o", label="Val F1")
    plt.plot(epochs, val_acc, marker="o", label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_plots(history, run_dir):
    plot_loss_curve(history, os.path.join(run_dir, "loss_curve.png"))
    plot_f1_curve(history, os.path.join(run_dir, "val_f1_curve.png"))


def main(args):
    set_seed(args.seed)
    device = get_device(args.device)

    print(f"[Device] {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    all_train_records = load_classification_records(
        args.train_json,
        args.image_root,
        limit=args.limit,
    )

    test_records = load_classification_records(
        args.test_json,
        args.image_root,
        limit=args.limit,
    )

    train_records, val_records = split_records(
        all_train_records,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"Train samples: {len(train_records)}")
    print(f"Val samples: {len(val_records)}")
    print(f"Test samples: {len(test_records)}")

    train_dataset = FeatureDataset(train_records, feature_dir=args.feature_dir)
    val_dataset = FeatureDataset(val_records, feature_dir=args.feature_dir)
    test_dataset = FeatureDataset(test_records, feature_dir=args.feature_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )

    save_json(os.path.join(run_dir, "args.json"), vars(args))

    backend = None
    if args.feature_dir is None:
        backend = OpenCLIPBackend(args.model_name, args.pretrained, device)

    classifier = MemeClassificationModel(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_epoch = -1
    best_threshold = 0.5

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_metrics": [],
        "val_metrics": [],
        "best_epoch": None,
        "best_val_f1": None,
        "best_threshold": None,
        "test_metrics": None,
    }

    log_file = open(os.path.join(run_dir, "train.log"), "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    use_features = args.feature_dir is not None

    log(f"Starting training at {timestamp}")
    log(f"Args: {vars(args)}")

    for epoch in range(1, args.epochs + 1):
        log(f"\n--- Epoch {epoch}/{args.epochs} ---")

        avg_train_loss, train_metrics = train_one_epoch(
            classifier,
            train_loader,
            backend,
            optimizer,
            criterion,
            device,
            use_features=use_features,
        )

        avg_val_loss, val_metrics_default, val_labels, val_scores = evaluate(
            classifier,
            val_loader,
            backend,
            criterion,
            device,
            use_features=use_features,
            threshold=0.5,
        )

        tuned_threshold, tuned_val_f1 = tune_threshold(val_labels, val_scores)
        val_preds_tuned = (val_scores >= tuned_threshold).astype(int)
        val_metrics = compute_classification_metrics(val_labels, val_preds_tuned, val_scores)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_metrics"].append(train_metrics)
        history["val_metrics"].append(val_metrics)

        log(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        log(f"Val tuned threshold: {tuned_threshold:.4f}")

        for k, v in val_metrics.items():
            log(f"Val {k}: {v:.4f}")

        epoch_path = os.path.join(run_dir, f"epoch_{epoch}.pt")
        save_checkpoint(epoch_path, classifier, optimizer, None, epoch, val_metrics["f1"])

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_threshold = tuned_threshold

            best_path = os.path.join(run_dir, "best_classifier.pt")
            save_checkpoint(best_path, classifier, optimizer, None, epoch, best_val_f1)

            history["best_epoch"] = best_epoch
            history["best_val_f1"] = best_val_f1
            history["best_threshold"] = best_threshold

            log(f"[*] Saved new best model by validation F1: {best_val_f1:.4f}")

        last_path = os.path.join(run_dir, "last_classifier.pt")
        save_checkpoint(last_path, classifier, optimizer, None, epoch, val_metrics["f1"])

        save_json(os.path.join(run_dir, "history.json"), history)
        save_plots(history, run_dir)

    log(f"\nBest epoch: {best_epoch}")
    log(f"Best validation F1: {best_val_f1:.4f}")
    log(f"Best validation threshold: {best_threshold:.4f}")

    # Final test evaluation ONCE using best checkpoint and validation-tuned threshold.
    best_path = os.path.join(run_dir, "best_classifier.pt")
    load_checkpoint(best_path, classifier, device=device)
    classifier.eval()

    avg_test_loss, test_metrics, test_labels, test_scores = evaluate(
        classifier,
        test_loader,
        backend,
        criterion,
        device,
        use_features=use_features,
        threshold=best_threshold,
    )

    history["test_metrics"] = test_metrics
    save_json(os.path.join(run_dir, "history.json"), history)
    save_json(os.path.join(run_dir, "final_metrics.json"), {
        "test_loss": avg_test_loss,
        "threshold": best_threshold,
        "metrics": test_metrics,
    })

    log("\n[Final Test Metrics]")
    log(f"Test Loss: {avg_test_loss:.4f}")
    log(f"Threshold from validation: {best_threshold:.4f}")

    for k, v in test_metrics.items():
        log(f"Test {k}: {v:.4f}")

    print_classification_report("Final Test Evaluation", test_metrics)

    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_json", type=str, default="data/memes-trainval.json")
    parser.add_argument("--test_json", type=str, default="data/memes-test.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--output_dir", type=str, default="outputs/caption_classification/train")

    parser.add_argument("--model_name", type=str, default="ViT-L-14")
    parser.add_argument("--pretrained", type=str, default="laion2b_s32b_b82k")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--feature_dir", type=str, default=None)

    parser.add_argument("--val_ratio", type=float, default=0.1)

    args = parser.parse_args()
    main(args)