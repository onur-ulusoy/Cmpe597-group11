import json
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_recall_at_k(score_matrix: torch.Tensor, ks=(1, 5, 10)):
    ranks = []
    sorted_idx = torch.argsort(score_matrix, dim=1, descending=True)

    for i in range(score_matrix.size(0)):
        rank = (sorted_idx[i] == i).nonzero(as_tuple=False).item() + 1
        ranks.append(rank)

    metrics = {}
    for k in ks:
        metrics[f"R@{k}"] = 100.0 * sum(r <= k for r in ranks) / len(ranks)

    metrics["MedR"] = int(np.median(ranks))
    metrics["MRR"] = float(np.mean([1.0 / r for r in ranks]))
    return metrics


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_score, vocab, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_val_score": best_val_score,
        "vocab_stoi": vocab.stoi,
        "args": vars(args),
    }, path)


def load_checkpoint(path, model, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt