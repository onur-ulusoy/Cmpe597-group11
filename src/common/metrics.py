import torch
import numpy as np

def compute_recall_metrics(score_matrix: torch.Tensor, ks=(1, 5, 10)):
    # Computes R@K, Mean Reciprocal Rank (MRR), and Median Rank.
    sorted_idx = torch.argsort(score_matrix, dim=1, descending=True)
    
    ranks = []
    for i in range(score_matrix.size(0)):
        rank = (sorted_idx[i] == i).nonzero(as_tuple=False).item() + 1
        ranks.append(rank)

    metrics = {}
    for k in ks:
        metrics[f"R@{k}"] = float(sum(r <= k for r in ranks) / len(ranks))

    metrics["MedR"] = int(np.median(ranks))
    metrics["MeanR"] = float(np.mean(ranks))
    metrics["MRR"] = float(np.mean([1.0 / r for r in ranks]))
    
    return metrics

def print_metrics(name: str, metrics: dict):
    print(f"\n=== {name} ===")
    print(f"R@1   : {metrics['R@1'] * 100:.2f}%")
    print(f"R@5   : {metrics['R@5'] * 100:.2f}%")
    if 'R@10' in metrics:
        print(f"R@10  : {metrics['R@10'] * 100:.2f}%")
    print(f"MRR   : {metrics['MRR'] * 100:.2f}%")
    print(f"MedR  : {metrics['MedR']:.2f}")
    print(f"MeanR : {metrics['MeanR']:.2f}")