import json
import os
import glob
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def log_metrics(log_path, epoch, avg_loss):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    mode = 'a' if os.path.exists(log_path) else 'w'
    with open(log_path, mode) as f:
        f.write(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}\n")
    print(f"Logged metrics to {log_path}")

def plot_loss(loss_history, save_path):
    epochs = range(1, len(loss_history) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history, marker='o', linestyle='-', color='#2c3e50', linewidth=2)
    plt.title('Training Loss per Epoch', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Contrastive Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(epochs)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_score, vocab=None, args=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_val_score": best_val_score,
        "vocab_stoi": vocab.stoi if vocab else None,
        "args": vars(args) if args else None,
    }, path)

def load_checkpoint(path, model, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt

def get_latest_checkpoint(base_output_dir="outputs/finetune"):
    runs = sorted(glob.glob(os.path.join(base_output_dir, "*")))
    runs = [r for r in runs if os.path.isdir(r)]
    if not runs:
        raise ValueError(f"No training runs found in {base_output_dir}")
    
    latest_run = runs[-1]
    epochs = glob.glob(os.path.join(latest_run, "lora_epoch_*"))
    if not epochs:
        raise ValueError(f"No checkpoint folders found in {latest_run}")
    
    epochs.sort(key=lambda x: int(x.split('_')[-1]))
    latest_checkpoint = epochs[-1]
    print(f"Auto-detected latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint