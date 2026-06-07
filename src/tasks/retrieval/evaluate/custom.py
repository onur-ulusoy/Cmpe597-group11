import argparse
import os
import sys
sys.path.append(os.getcwd())

import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.common.dataset import load_memecap_records
from src.common.utils import save_json, load_checkpoint
from src.common.metrics import compute_recall_metrics
from src.models.custom.data_utils import (
    MemeCapCustomDataset,
    Vocab,
    build_image_transform,
)
from src.models.custom.cross_modal_retrieval_model import MatchingModel


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def encode_dataset(model, dataloader, device):
    model.eval()
    meme_embs = []
    caption_embs = []

    for batch in dataloader:
        images = batch["image"].to(device)
        caption_ids = batch["caption_ids"].to(device)
        caption_mask = batch["caption_mask"].to(device)
        title_ids = batch["title_ids"].to(device)
        title_mask = batch["title_mask"].to(device)

        meme_emb = model.encode_meme(
            images,
            title_ids,
            title_mask,
            normalize=True,
        )
        caption_emb = model.encode_caption(
            caption_ids,
            caption_mask,
            normalize=True,
        )

        meme_embs.append(meme_emb.cpu())
        caption_embs.append(caption_emb.cpu())

    meme_embs = torch.cat(meme_embs, dim=0)
    caption_embs = torch.cat(caption_embs, dim=0)
    return meme_embs, caption_embs


@torch.no_grad()
def compute_score_matrix(model, dataloader, device):
    meme_embs, caption_embs = encode_dataset(model, dataloader, device)
    score_matrix = meme_embs @ caption_embs.T
    return score_matrix


def evaluate_bidirectional(score_matrix):
    image_to_text = compute_recall_metrics(score_matrix.cpu(), ks=(1, 5, 10))
    text_to_image = compute_recall_metrics(score_matrix.T.cpu(), ks=(1, 5, 10))

    mean_metrics = {}
    for key in image_to_text:
        mean_metrics[f"mean_{key}"] = (image_to_text[key] + text_to_image[key]) / 2.0

    return {
        "image_to_text": image_to_text,
        "text_to_image": text_to_image,
        "mean": mean_metrics,
    }


def compute_ranks(score_matrix):
    n = score_matrix.size(0)
    sorted_idx = score_matrix.argsort(dim=1, descending=True)

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
    plt.title("Rank Distribution of Correct Captions (last bin = rank ≥ 50)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_similarity_matrix(score_matrix, path, max_items=64):
    n = min(max_items, score_matrix.size(0))
    sub = score_matrix[:n, :n].cpu().numpy()

    plt.figure(figsize=(7, 6))
    plt.imshow(sub, aspect="auto")
    plt.colorbar(label="Similarity")
    plt.xlabel("Candidate Captions")
    plt.ylabel("Query Memes")
    plt.title(f"Similarity Matrix Sample ({n}x{n})")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_topk_examples(score_matrix, records, path, top_k=5, max_examples=30):
    sorted_idx = score_matrix.argsort(dim=1, descending=True)

    examples = []
    n = min(len(records), max_examples)

    for i in range(n):
        top_indices = sorted_idx[i, :top_k].tolist()

        item = {
            "query_index": i,
            "image_path": getattr(records[i], "image_path", ""),
            "title": getattr(records[i], "title", ""),
            "correct_caption": getattr(records[i], "caption", ""),
            "correct_rank": int((sorted_idx[i] == i).nonzero(as_tuple=True)[0].item() + 1),
            "top_retrieved": [],
        }

        for j in top_indices:
            item["top_retrieved"].append(
                {
                    "candidate_index": int(j),
                    "score": float(score_matrix[i, j].item()),
                    "caption": getattr(records[j], "caption", ""),
                    "is_correct": bool(j == i),
                }
            )

        examples.append(item)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=4, ensure_ascii=False)


def main(args):
    device = get_device()
    print(f"[Device] {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)

    vocab = Vocab(checkpoint["vocab_stoi"])
    model_args = checkpoint["args"]

    model_type = model_args.get("model_type", "type1")
    print(f"[Info] Loaded a {model_type.upper()} checkpoint.")

    model = MatchingModel(
        vocab_size=len(vocab),
        pad_idx=vocab.pad_idx,
        model_type=model_type,
        feat_dim=model_args["feat_dim"],
        word_dim=model_args["word_dim"],
        text_hidden_dim=model_args["text_hidden_dim"],
        text_num_layers=model_args["text_num_layers"],
        text_dropout=model_args["text_dropout"],
        image_dropout=model_args["image_dropout"],
    ).to(device)

    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    test_records = load_memecap_records(args.test_json, args.image_root)

    dataset = MemeCapCustomDataset(
        records=test_records,
        vocab=vocab,
        max_text_len=args.max_text_len,
        image_transform=build_image_transform(args.image_size, train=False),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    score_matrix = compute_score_matrix(model, dataloader, device)
    metrics = evaluate_bidirectional(score_matrix)
    ranks = compute_ranks(score_matrix)

    print(f"[{model_type.upper()} Test Metrics]")
    print("\nImage-to-text:")
    for k, v in metrics["image_to_text"].items():
        print(f"{k}: {v}")

    print("\nText-to-image:")
    for k, v in metrics["text_to_image"].items():
        print(f"{k}: {v}")

    print("\nMean:")
    for k, v in metrics["mean"].items():
        print(f"{k}: {v}")

    out_dir = os.path.dirname(args.output_json)
    os.makedirs(out_dir, exist_ok=True)

    final_out_path = os.path.join(out_dir, f"{model_type}_test_metrics.json")
    save_json(final_out_path, metrics)
    print(f"[Info] Saved metrics to {final_out_path}")

    rank_hist_path = os.path.join(out_dir, f"{model_type}_rank_histogram.png")
    sim_matrix_path = os.path.join(out_dir, f"{model_type}_similarity_matrix_sample.png")
    examples_path = os.path.join(out_dir, f"{model_type}_topk_examples.json")

    plot_rank_histogram(ranks, rank_hist_path)
    plot_similarity_matrix(score_matrix, sim_matrix_path, max_items=args.sim_matrix_items)
    save_topk_examples(
        score_matrix,
        test_records,
        examples_path,
        top_k=args.top_k,
        max_examples=args.max_examples,
    )

    print("[Info] Saved visualizations:")
    print(f"  - {rank_hist_path}")
    print(f"  - {sim_matrix_path}")
    print(f"  - {examples_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_json", type=str, default="data/memes-test.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--output_json", type=str, default="outputs/retrieval/custom/eval_metrics.json")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_text_len", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_examples", type=int, default=30)
    parser.add_argument("--sim_matrix_items", type=int, default=64)

    args = parser.parse_args()
    main(args)