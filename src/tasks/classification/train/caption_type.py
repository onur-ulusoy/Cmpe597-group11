import argparse
import os
import sys
import random
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.common.classification_dataset import load_classification_records, MemeCapClassificationDataset
from src.common.classification_metrics import compute_classification_metrics, print_classification_report
from src.common.utils import set_seed, save_json, save_checkpoint
from src.models.pretrained.openclip import OpenCLIPBackend
from src.models.custom.classification_model import MemeClassificationModel

def get_device(preference=None):
    if preference: return preference
    if torch.cuda.is_available(): return "cuda"
    elif torch.backends.mps.is_available(): return "mps"
    return "cpu"

def train_one_epoch(classifier, dataloader, backend, optimizer, criterion, device):
    classifier.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        img_paths = batch["image_path"] if "image_path" in batch else None
        # Note: dataset returns PIL image, but for frozen CLIP we might prefer paths or pre-extracted
        # To keep it simple, we use the backend's encode_images which takes paths
        
        texts = batch["text"]
        labels = batch["label"].to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            # Extract features from frozen CLIP backend
            img_emb = backend.encode_images(img_paths, batch_size=len(img_paths)).to(device)
            text_emb = backend.encode_texts(texts, batch_size=len(texts)).to(device)
            
        # Forward through trainable classifier
        logits = classifier(img_emb, text_emb)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = torch.sigmoid(logits).round().detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].numpy())
        
        progress.set_postfix(loss=f"{loss.item():.4f}")
        
    return running_loss / len(dataloader), all_labels, all_preds

@torch.no_grad()
def evaluate(classifier, dataloader, backend, criterion, device):
    classifier.eval()
    running_loss = 0.0
    all_scores = []
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        img_paths = batch["image_path"]
        texts = batch["text"]
        labels = batch["label"].to(device).float().unsqueeze(1)
        
        img_emb = backend.encode_images(img_paths, batch_size=len(img_paths)).to(device)
        text_emb = backend.encode_texts(texts, batch_size=len(texts)).to(device)
        
        logits = classifier(img_emb, text_emb)
        loss = criterion(logits, labels)
        
        running_loss += loss.item()
        scores = torch.sigmoid(logits).cpu().numpy()
        preds = scores.round()
        
        all_scores.extend(scores)
        all_preds.extend(preds)
        all_labels.extend(batch["label"].numpy())
        
    metrics = compute_classification_metrics(np.array(all_labels), np.array(all_preds), np.array(all_scores))
    return running_loss / len(dataloader), metrics

def main(args):
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"[Device] {device}")
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Dataset
    train_records = load_classification_records(args.train_json, args.image_root, limit=args.limit)
    test_records = load_classification_records(args.test_json, args.image_root, limit=args.limit)
    
    # We use a custom version of the dataset that returns paths for the CLIP backend
    class PathDataset(MemeCapClassificationDataset):
        def __getitem__(self, idx):
            sample = self.samples[idx]
            return {"image_path": str(sample.image_path), "text": sample.text, "label": sample.label}
            
    train_dataset = PathDataset(train_records)
    test_dataset = PathDataset(test_records)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    
    # CLIP Backend (Frozen)
    backend = OpenCLIPBackend(args.model_name, args.pretrained, device)
    
    # MLP Classifier (Trainable)
    classifier = MemeClassificationModel(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    best_f1 = -1
    history = {"train_loss": [], "test_metrics": []}
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        avg_train_loss, _, _ = train_one_epoch(classifier, train_loader, backend, optimizer, criterion, device)
        avg_test_loss, metrics = evaluate(classifier, test_loader, backend, criterion, device)
        
        history["train_loss"].append(avg_train_loss)
        history["test_metrics"].append(metrics)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
        print_classification_report("Test", metrics)
        
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            ckpt_path = os.path.join(run_dir, "best_classifier.pt")
            save_checkpoint(ckpt_path, classifier, optimizer, None, epoch, best_f1)
            print(f"[*] Saved new best model to {ckpt_path}")
            
        save_json(os.path.join(run_dir, "history.json"), history)

if __name__ == "__main__":
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, default="data/memes-trainval.json")
    parser.add_argument("--test_json", type=str, default="data/memes-test.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--output_dir", type=str, default="outputs/classification/train")
    
    parser.add_argument("--model_name", type=str, default="ViT-L-14")
    parser.add_argument("--pretrained", type=str, default="laion2b_s32b_b82k")
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or mps")
    
    args = parser.parse_args()
    main(args)
