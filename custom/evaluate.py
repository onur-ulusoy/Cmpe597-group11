import argparse
import os
import sys
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader

from dataset import load_memecap_records
from utils import save_json, load_checkpoint
from metrics import compute_recall_metrics
from custom_dataset import MemeCapCustomDataset, Vocab, build_image_transform
from model import MatchingModel

def get_device():
    if torch.cuda.is_available(): return "cuda"
    elif torch.backends.mps.is_available(): return "mps"
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

        meme_emb = model.encode_meme(images, title_ids, title_mask, normalize=True)
        caption_emb = model.encode_caption(caption_ids, caption_mask, normalize=True)

        meme_embs.append(meme_emb.cpu())
        caption_embs.append(caption_emb.cpu())

    meme_embs = torch.cat(meme_embs, dim=0)
    caption_embs = torch.cat(caption_embs, dim=0)
    return meme_embs, caption_embs

@torch.no_grad()
def evaluate_matching(model, dataloader, device):
    meme_embs, caption_embs = encode_dataset(model, dataloader, device)
    meme_embs = meme_embs.to(device)
    caption_embs = caption_embs.to(device)

    score_matrix = meme_embs @ caption_embs.T
    return compute_recall_metrics(score_matrix.cpu(), ks=(1, 5, 10))

def main(args):
    device = get_device()
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
        num_workers=0,
    )

    metrics = evaluate_matching(model, dataloader, device)

    print(f"[{model_type.upper()} Test Metrics]")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        os.makedirs(out_dir, exist_ok=True)
        final_out_path = os.path.join(out_dir, f"{model_type}_test_metrics.json")
        save_json(final_out_path, metrics)
        print(f"[Info] Saved metrics to {final_out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_json", type=str, default="data/memes-test.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--output_json", type=str, default="outputs/custom_models/eval_metrics.json")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_text_len", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    main(args)