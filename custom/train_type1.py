import argparse
import math
import os
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    MemeCapCustomDataset,
    build_image_transform,
    build_vocab_from_records,
    load_memecap_records,
)
from loss import total_loss
from model import Type1MatchingModel
from utils import (
    compute_recall_at_k,
    load_checkpoint,
    save_checkpoint,
    save_json,
    set_seed,
)

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
    image_embs = []
    text_embs = []

    for batch in dataloader:
        images = batch["image"].to(device)
        caption_ids = batch["caption_ids"].to(device)
        caption_mask = batch["caption_mask"].to(device)

        # Normalize MUST be True for pure cosine-similarity retrieval
        image_emb = model.encode_image(images, normalize=True)
        text_emb = model.encode_text(caption_ids, caption_mask, normalize=True)

        image_embs.append(image_emb.cpu())
        text_embs.append(text_emb.cpu())

    image_embs = torch.cat(image_embs, dim=0)
    text_embs = torch.cat(text_embs, dim=0)
    return image_embs, text_embs

@torch.no_grad()
def evaluate_type1_matching(model, dataloader, device):
    model.eval()

    image_embs, text_embs = encode_dataset(model, dataloader, device)
    image_embs = image_embs.to(device)
    text_embs = text_embs.to(device)

    # Compute cosine similarity matrix directly (Matrix Mult of L2 normalized vectors)
    # The score matrix will be [N_images, N_texts]
    score_matrix = image_embs @ text_embs.T

    metrics = compute_recall_at_k(score_matrix.cpu(), ks=(1, 5, 10))
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

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device, enabled=use_amp):
            pos_out = model(images, caption_ids, caption_mask)
            
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

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
        )

    n = len(dataloader)
    return {
        "loss": running_loss / n,
        "loss_contrastive": running_loss / n,
    }

def main(args):
    set_seed(args.seed)
    device = get_device()
    print(f"[Device] {device}")

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
        records=train_records,
        vocab=vocab,
        max_text_len=args.max_text_len,
        image_transform=build_image_transform(args.image_size, train=True),
    )
    val_dataset = MemeCapCustomDataset(
        records=val_records,
        vocab=vocab,
        max_text_len=args.max_text_len,
        image_transform=build_image_transform(args.image_size, train=False),
    )
    test_dataset = MemeCapCustomDataset(
        records=test_records,
        vocab=vocab,
        max_text_len=args.max_text_len,
        image_transform=build_image_transform(args.image_size, train=False),
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

    model = Type1MatchingModel(
        vocab_size=len(vocab),
        pad_idx=vocab.pad_idx,
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
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
    )

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "best_type1_match.pt")

    best_val_score = -1.0
    best_epoch = -1

    history = {
        "train": [],
        "val_metrics": [],
        "best_epoch": None,
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

        val_metrics = evaluate_type1_matching(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        val_score = val_metrics["R@5"] + 0.5 * val_metrics["R@1"]

        history["train"].append(train_stats)
        history["val_metrics"].append(val_metrics)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch}] "
            f"loss={train_stats['loss']:.4f} "
            f"lr={current_lr:.6f}"
        )
        print(f"[Val] {val_metrics}")

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_score=best_val_score,
                vocab=vocab,
                args=args,
            )
            print(f"[Checkpoint] Saved best model to {ckpt_path}")

        save_json(os.path.join(args.output_dir, "train_history_type1_match.json"), history)
        scheduler.step()

    print(f"\nBest validation checkpoint: epoch {best_epoch} with score={best_val_score:.3f}")

    best_ckpt = load_checkpoint(ckpt_path, model, device=device)

    final_test_metrics = evaluate_type1_matching(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    history["best_epoch"] = int(best_ckpt["epoch"])
    history["final_test_metrics"] = final_test_metrics

    save_json(os.path.join(args.output_dir, "train_history_type1_match.json"), history)
    save_json(os.path.join(args.output_dir, "final_test_metrics_type1_match.json"), final_test_metrics)

    print("\n[Final Test Metrics]")
    for k, v in final_test_metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_json", type=str, default="data/memes-trainval.json")
    parser.add_argument("--test_json", type=str, default="data/memes-test.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--output_dir", type=str, default="outputs/custom_type1_match")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_text_len", type=int, default=40)
    parser.add_argument("--min_freq", type=int, default=2) # Adjusted from 1 to 2

    parser.add_argument("--feat_dim", type=int, default=256)
    parser.add_argument("--word_dim", type=int, default=256) # Adjusted from 128
    parser.add_argument("--text_hidden_dim", type=int, default=256) # Adjusted from 128
    parser.add_argument("--text_num_layers", type=int, default=1)

    parser.add_argument("--text_dropout", type=float, default=0.15)
    parser.add_argument("--image_dropout", type=float, default=0.05)

    parser.add_argument("--batch_size", type=int, default=32) # Adjusted from 16
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4) # Adjusted from 5e-5
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    main(args)