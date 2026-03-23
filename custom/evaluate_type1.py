import argparse

import torch
from torch.utils.data import DataLoader

from dataset import MemeCapCustomDataset, Vocab, build_image_transform, load_memecap_records
from model import Type1MatchingModel
from utils import compute_recall_at_k, load_checkpoint, save_json

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@torch.no_grad()
def encode_dataset(model, dataloader, device):
    model.eval()
    image_embs = []
    text_embs = []

    for batch in dataloader:
        images = batch["image"].to(device)
        caption_ids = batch["caption_ids"].to(device)
        caption_mask = batch["caption_mask"].to(device)

        image_emb = model.encode_image(images, normalize=True)
        text_emb = model.encode_text(caption_ids, caption_mask, normalize=True)

        image_embs.append(image_emb.cpu())
        text_embs.append(text_emb.cpu())

    image_embs = torch.cat(image_embs, dim=0)
    text_embs = torch.cat(text_embs, dim=0)
    return image_embs, text_embs

@torch.no_grad()
def evaluate_matching(model, dataloader, device):
    image_embs, text_embs = encode_dataset(model, dataloader, device)
    image_embs = image_embs.to(device)
    text_embs = text_embs.to(device)

    # Calculate full score matrix 
    score_matrix = image_embs @ text_embs.T

    return compute_recall_at_k(score_matrix.cpu(), ks=(1, 5, 10))

def main(args):
    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)

    vocab = Vocab(checkpoint["vocab_stoi"])
    model_args = checkpoint["args"]

    model = Type1MatchingModel(
        vocab_size=len(vocab),
        pad_idx=vocab.pad_idx,
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

    print("[Type1 Test Metrics]")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if args.output_json:
        save_json(args.output_json, metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_json", type=str, default="data/memes-test.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--output_json", type=str, default="outputs/custom_type1_match/test_metrics.json")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_text_len", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    main(args)