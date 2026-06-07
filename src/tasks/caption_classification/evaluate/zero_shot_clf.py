import argparse
import sys
import os
from pathlib import Path
import random

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from src.common.classification_dataset import load_classification_records
from src.common.classification_metrics import compute_classification_metrics, print_classification_report
from src.common.utils import save_json, set_seed
from src.models.pretrained.openclip import OpenCLIPBackend


def split_records(records, val_ratio=0.1, seed=42):
    records = list(records)
    rng = random.Random(seed)
    rng.shuffle(records)

    val_size = int(len(records) * val_ratio)
    val_records = records[:val_size]

    return val_records


def compute_scores(samples, backend, batch_size):
    all_scores = []
    all_labels = []

    for start in tqdm(range(0, len(samples), batch_size), desc="Classification pairs"):
        batch_samples = samples[start:start + batch_size]

        img_paths = [s.image_path for s in batch_samples]
        texts = [s.text for s in batch_samples]
        labels = [s.label for s in batch_samples]

        img_emb = backend.encode_images(img_paths, batch_size=batch_size)
        text_emb = backend.encode_texts(texts, batch_size=batch_size)

        img_emb = F.normalize(img_emb, p=2, dim=-1)
        text_emb = F.normalize(text_emb, p=2, dim=-1)

        similarities = (img_emb * text_emb).sum(dim=-1).cpu().numpy()

        # Assumption:
        # Higher CLIP similarity means literal caption.
        # Lower similarity means meme/metaphorical caption.
        # Therefore score for class 1 = 1 - similarity.
        scores = 1.0 - similarities

        all_scores.extend(scores.tolist())
        all_labels.extend(labels)

    return np.array(all_labels), np.array(all_scores)


def tune_threshold(labels, scores):
    best_f1 = -1.0
    best_threshold = 0.5

    for threshold in np.linspace(scores.min(), scores.max(), 100):
        preds = (scores >= threshold).astype(int)
        metrics = compute_classification_metrics(labels, preds, scores)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)

    return best_threshold, best_f1


def plot_confusion_matrix(labels, preds, out_path):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Zero-Shot Confusion Matrix")
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


def plot_score_histogram(labels, scores, threshold, out_path):
    plt.figure(figsize=(8, 5))
    plt.hist(scores[labels == 0], bins=30, alpha=0.6, label="Literal")
    plt.hist(scores[labels == 1], bins=30, alpha=0.6, label="Meme")
    plt.axvline(threshold, linestyle="--", label=f"Threshold={threshold:.3f}")
    plt.xlabel("Zero-shot score: 1 - CLIP similarity")
    plt.ylabel("Count")
    plt.title("Zero-Shot Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_zero_shot_classification(args):
    set_seed(args.seed)

    train_records = load_classification_records(
        json_path=args.train_json,
        image_root=args.image_root,
        limit=args.limit,
    )

    val_records = split_records(
        train_records,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    test_records = load_classification_records(
        json_path=args.test_json,
        image_root=args.image_root,
        limit=args.limit,
    )

    backend = OpenCLIPBackend(
        model_name=args.openclip_model_name,
        pretrained=args.openclip_pretrained,
        device=args.device,
    )

    print("Computing validation scores for threshold tuning...")
    val_labels, val_scores = compute_scores(val_records, backend, args.batch_size)

    best_threshold, best_val_f1 = tune_threshold(val_labels, val_scores)

    print(f"Validation-tuned threshold: {best_threshold:.4f}")
    print(f"Validation F1 at threshold: {best_val_f1:.4f}")

    print("Computing test scores...")
    test_labels, test_scores = compute_scores(test_records, backend, args.batch_size)

    test_preds = (test_scores >= best_threshold).astype(int)
    final_metrics = compute_classification_metrics(test_labels, test_preds, test_scores)

    run_name = f"Zero-Shot_{args.openclip_model_name}_ValThresh{best_threshold:.3f}"
    print_classification_report(run_name, final_metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(output_dir / "zero_shot_metrics.json", {
        "run_name": run_name,
        "model_name": args.openclip_model_name,
        "threshold_source": "validation_split_from_trainval",
        "threshold": float(best_threshold),
        "val_f1": float(best_val_f1),
        "metrics": final_metrics,
    })

    plot_confusion_matrix(
        test_labels,
        test_preds,
        output_dir / "confusion_matrix.png",
    )

    plot_score_histogram(
        test_labels,
        test_scores,
        best_threshold,
        output_dir / "score_histogram.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_json", type=str, default="data/memes-trainval.json")
    parser.add_argument("--test_json", type=str, default="data/memes-test.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--output_dir", type=str, default="outputs/caption_classification/zero_shot")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--openclip_model_name", type=str, default="ViT-L-14")
    parser.add_argument("--openclip_pretrained", type=str, default="laion2b_s32b_b82k")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()
    run_zero_shot_classification(args)