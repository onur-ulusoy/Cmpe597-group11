import argparse
import math
import os
import random
import sys
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.common.dataset import load_memecap_records
from src.common.metrics import compute_recall_metrics
from src.common.utils import set_seed, save_json, save_checkpoint, load_checkpoint
from src.models.custom.data_utils import (
    MemeCapCustomDataset,
    build_image_transform,
    build_vocab_from_records,
)
from src.models.custom.loss import total_loss
from src.models.custom.cross_modal_retrieval_model import MatchingModel


def get_device():
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


def build_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.1 + 0.9 * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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

        meme_emb = model.encode_meme(images, title_ids, title_mask, normalize=True)
        caption_emb = model.encode_caption(caption_ids, caption_mask, normalize=True)

        meme_embs.append(meme_emb.cpu())
        caption_embs.append(caption_emb.cpu())

    meme_embs = torch.cat(meme_embs, dim=0)
    caption_embs = torch.cat(caption_embs, dim=0)
    return meme_embs, caption_embs


@torch.no_grad()
def evaluate_matching(model, dataloader, device):
    model.eval()
    meme_embs, caption_embs = encode_dataset(model, dataloader, device)

    score_matrix = meme_embs @ caption_embs.T
    metrics = compute_recall_metrics(score_matrix.cpu(), ks=(1, 5, 10))
    return metrics


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    grad_clip,
    label_smoothing,
    scaler,
    use_amp,
):
    model.train()
    running_loss = 0.0
    progress = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        caption_ids = batch["caption_ids"].to(device, non_blocking=True)
        caption_mask = batch["caption_mask"].to(device, non_blocking=True)
        title_ids = batch["title_ids"].to(device, non_blocking=True)
        title_mask = batch["title_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device, enabled=use_amp):
            pos_out = model(
                images,
                title_ids,
                title_mask,
                caption_ids,
                caption_mask,
            )

            loss, parts = total_loss(
                image_emb=pos_out["image_emb"],
                text_emb=pos_out["text_emb"],
                logit_scale=pos_out["logit_scale"],
                label_smoothing=label_smoothing,
            )

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return {"loss": running_loss / max(len(dataloader), 1)}


def plot_train_loss(history, out_path):
    if not history["train"]:
        return

    epochs = list(range(1, len(history["train"]) + 1))
    losses = [x["loss"] for x in history["train"]]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Custom Retrieval Model Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_val_recall(history, out_path):
    if not history["val_metrics"]:
        return

    epochs = list(range(1, len(history["val_metrics"]) + 1))
    r1 = [m["R@1"] * 100 for m in history["val_metrics"]]
    r5 = [m["R@5"] * 100 for m in history["val_metrics"]]
    r10 = [m["R@10"] * 100 for m in history["val_metrics"]]
    mrr = [m["MRR"] * 100 for m in history["val_metrics"]]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, r1, marker="o", label="R@1")
    plt.plot(epochs, r5, marker="o", label="R@5")
    plt.plot(epochs, r10, marker="o", label="R@10")
    plt.plot(epochs, mrr, marker="o", label="MRR")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Metric (%)")
    plt.title("Custom Retrieval Model Validation Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_train_loss_and_val_r5(history, out_path):
    if not history["train"] or not history["val_metrics"]:
        return

    epochs = list(range(1, len(history["train"]) + 1))
    losses = [x["loss"] for x in history["train"]]
    r5 = [m["R@5"] * 100 for m in history["val_metrics"]]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(epochs, losses, marker="o", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")

    ax2 = ax1.twinx()
    ax2.plot(epochs, r5, marker="s", label="Val R@5")
    ax2.set_ylabel("Validation R@5 (%)")

    plt.title("Training Loss vs Validation R@5")
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_all_plots(history, out_dir):
    plot_train_loss(history, os.path.join(out_dir, "loss_curve.png"))
    plot_val_recall(history, os.path.join(out_dir, "val_recall_curve.png"))
    plot_train_loss_and_val_r5(history, os.path.join(out_dir, "loss_vs_val_r5.png"))


def main(args):
    set_seed(args.seed)
    device = get_device()
    print(f"[Device] {device} | [Model Type] {args.model_type.upper()}")

    run_output_dir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(run_output_dir, exist_ok=True)

    all_train_records = load_memecap_records(args.train_json, args.image_root)
    test_records = load_memecap_records(args.test_json, args.image_root)

    train_records, val_records = split_records(
        all_train_records,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    vocab = build_vocab_from_records(
        train_records,
        min_freq=args.min_freq,
        include_titles=True,
    )

    train_dataset = MemeCapCustomDataset(
        train_records,
        vocab,
        args.max_text_len,
        build_image_transform(args.image_size, train=True),
    )
    val_dataset = MemeCapCustomDataset(
        val_records,
        vocab,
        args.max_text_len,
        build_image_transform(args.image_size, train=False),
    )
    test_dataset = MemeCapCustomDataset(
        test_records,
        vocab,
        args.max_text_len,
        build_image_transform(args.image_size, train=False),
    )

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
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = MatchingModel(
        vocab_size=len(vocab),
        pad_idx=vocab.pad_idx,
        model_type=args.model_type,
        feat_dim=args.feat_dim,
        word_dim=args.word_dim,
        text_hidden_dim=args.text_hidden_dim,
        text_num_layers=args.text_num_layers,
        text_dropout=args.text_dropout,
        image_dropout=args.image_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )
    scheduler = build_cosine_scheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
    )

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_path = os.path.join(run_output_dir, f"best_{args.model_type}.pt")

    best_val_score = -1.0
    best_epoch = -1

    history = {
        "args": vars(args),
        "train": [],
        "val_metrics": [],
        "best_epoch": None,
        "best_val_score": None,
        "final_test_metrics": None,
    }

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        train_stats = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            label_smoothing=args.label_smoothing,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_metrics = evaluate_matching(model, val_loader, device)
        val_score = val_metrics["R@5"] + 0.5 * val_metrics["R@1"]

        history["train"].append(train_stats)
        history["val_metrics"].append(val_metrics)

        print(
            f"[Epoch {epoch}] "
            f"loss={train_stats['loss']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )
        print(f"[Val] {val_metrics}")
        print(f"[Selection Score] {val_score:.6f}")

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch

            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_score,
                vocab,
                args,
            )
            print(f"[Checkpoint] Saved best model to {ckpt_path}")

            history["best_epoch"] = best_epoch
            history["best_val_score"] = best_val_score

        save_json(os.path.join(run_output_dir, "train_history.json"), history)
        save_all_plots(history, run_output_dir)

        scheduler.step()

    print(f"\nBest validation checkpoint: epoch {best_epoch} with score={best_val_score:.6f}")

    best_ckpt = load_checkpoint(ckpt_path, model, device=device)
    final_test_metrics = evaluate_matching(model, test_loader, device)

    history["best_epoch"] = int(best_ckpt["epoch"])
    history["final_test_metrics"] = final_test_metrics

    save_json(os.path.join(run_output_dir, "train_history.json"), history)
    save_json(os.path.join(run_output_dir, "final_test_metrics.json"), final_test_metrics)
    save_all_plots(history, run_output_dir)

    print("\n[Final Test Metrics]")
    for k, v in final_test_metrics.items():
        print(f"{k}: {v}")

    print(f"\n[Info] Saved outputs to {run_output_dir}")
    print(f"[Info] Saved checkpoint to {ckpt_path}")
    print(f"[Info] Saved visualizations:")
    print(f"  - {os.path.join(run_output_dir, 'loss_curve.png')}")
    print(f"  - {os.path.join(run_output_dir, 'val_recall_curve.png')}")
    print(f"  - {os.path.join(run_output_dir, 'loss_vs_val_r5.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["type1", "type2"],
        default="type1",
        help="type1 = image only, type2 = image + title",
    )

    parser.add_argument("--train_json", type=str, default="data/memes-trainval.json")
    parser.add_argument("--test_json", type=str, default="data/memes-test.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--output_dir", type=str, default="outputs/retrieval/custom")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_text_len", type=int, default=40)
    parser.add_argument("--min_freq", type=int, default=2)

    parser.add_argument("--feat_dim", type=int, default=256)
    parser.add_argument("--word_dim", type=int, default=256)
    parser.add_argument("--text_hidden_dim", type=int, default=256)
    parser.add_argument("--text_num_layers", type=int, default=1)
    parser.add_argument("--text_dropout", type=float, default=0.15)
    parser.add_argument("--image_dropout", type=float, default=0.05)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    main(args)