import argparse
import os
import sys
import json

import numpy as np
import torch
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
from src.common.utils import load_checkpoint, save_json
from src.models.pretrained.openclip import OpenCLIPBackend
from src.models.custom.caption_classification_model import MemeClassificationModel


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PathDataset(MemeCapClassificationDataset):
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "image_path": str(sample.image_path),
            "text": sample.text,
            "label": sample.label,
        }


@torch.no_grad()
def run_inference(model, backend, dataloader, device, threshold):
    model.eval()

    all_scores = []
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Inference"):
        img_paths = batch["image_path"]
        texts = batch["text"]
        labels = batch["label"].numpy()

        img_emb = backend.encode_images(img_paths, batch_size=len(img_paths)).to(device)
        text_emb = backend.encode_texts(texts, batch_size=len(texts)).to(device)

        logits = model(img_emb, text_emb)
        scores = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (scores >= threshold).astype(int)

        all_scores.extend(scores.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    return np.array(all_labels), np.array(all_preds), np.array(all_scores)


def plot_confusion_matrix(labels, preds, out_path):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Caption Classification Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Literal", "Meme"], rotation=20)
    plt.yticks([0, 1], ["Literal", "Meme"])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_score_histogram(labels, scores, out_path):
    plt.figure(figsize=(8, 5))
    plt.hist(scores[labels == 0], bins=30, alpha=0.6, label="Literal")
    plt.hist(scores[labels == 1], bins=30, alpha=0.6, label="Meme")
    plt.xlabel("Predicted probability of meme/metaphorical class")
    plt.ylabel("Count")
    plt.title("Prediction Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    test_records = load_classification_records(
        args.test_json,
        args.image_root,
        limit=args.limit,
    )

    test_dataset = PathDataset(test_records)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    backend = OpenCLIPBackend(args.model_name, args.pretrained, device)

    model = MemeClassificationModel(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    threshold = args.threshold

    if args.history_json and os.path.exists(args.history_json):
        with open(args.history_json, "r", encoding="utf-8") as f:
            history = json.load(f)

        if "best_threshold" in history and history["best_threshold"] is not None:
            threshold = float(history["best_threshold"])
            print(f"Using validation-tuned threshold from history: {threshold:.4f}")

    labels, preds, scores = run_inference(
        model,
        backend,
        test_loader,
        device,
        threshold,
    )

    metrics = compute_classification_metrics(labels, preds, scores)
    print_classification_report("Final Evaluation", metrics)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.dirname(args.checkpoint)

    os.makedirs(output_dir, exist_ok=True)

    save_json(os.path.join(output_dir, "final_metrics.json"), {
        "threshold": threshold,
        "metrics": metrics,
    })

    plot_confusion_matrix(
        labels,
        preds,
        os.path.join(output_dir, "confusion_matrix.png"),
    )

    plot_score_histogram(
        labels,
        scores,
        os.path.join(output_dir, "score_histogram.png"),
    )

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_json", type=str, default="data/memes-test.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--history_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--model_name", type=str, default="ViT-L-14")
    parser.add_argument("--pretrained", type=str, default="laion2b_s32b_b82k")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()
    main(args)