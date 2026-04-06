import argparse
import os
import sys
import random
from datetime import datetime
import json
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

def train_one_epoch(classifier, dataloader, backend, optimizer, criterion, device, use_features=False):
    classifier.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        labels = batch["label"].to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        
        if use_features:
            img_emb = batch["img_emb"].to(device)
            text_emb = batch["text_emb"].to(device)
        else:
            img_paths = batch["image_path"]
            texts = batch["text"]
            with torch.no_grad():
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
def evaluate(classifier, dataloader, backend, criterion, device, use_features=False):
    classifier.eval()
    running_loss = 0.0
    all_scores = []
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        labels = batch["label"].to(device).float().unsqueeze(1)
        
        if use_features:
            img_emb = batch["img_emb"].to(device)
            text_emb = batch["text_emb"].to(device)
        else:
            img_paths = batch["image_path"]
            texts = batch["text"]
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
    
    # Dataset Mapping
    class FeatureDataset(MemeCapClassificationDataset):
        def __init__(self, records, feature_dir=None):
            super().__init__(records)
            self.feature_dir = feature_dir
            if feature_dir:
                with open(os.path.join(feature_dir, "text_mapping.json"), "r") as f:
                    self.text_mapping = json.load(f)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            if self.feature_dir:
                img_path = os.path.join(self.feature_dir, "images", f"{sample.image_fname}.pt")
                t_hash = self.text_mapping[sample.text]
                text_path = os.path.join(self.feature_dir, "texts", f"{t_hash}.pt")
                return {
                    "img_emb": torch.load(img_path, map_location="cpu"),
                    "text_emb": torch.load(text_path, map_location="cpu"),
                    "label": sample.label
                }
            return {"image_path": str(sample.image_path), "text": sample.text, "label": sample.label}
            
    train_dataset = FeatureDataset(train_records, feature_dir=args.feature_dir)
    test_dataset = FeatureDataset(test_records, feature_dir=args.feature_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    
    # Save args
    save_json(os.path.join(run_dir, "args.json"), vars(args))
    
    # CLIP Backend (Frozen)
    backend = OpenCLIPBackend(args.model_name, args.pretrained, device)
    
    # MLP Classifier (Trainable)
    classifier = MemeClassificationModel(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    best_f1 = -1
    history = {"train_loss": [], "test_metrics": []}
    
    # Logging
    log_file = open(os.path.join(run_dir, "train.log"), "w")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
        
    log(f"Starting training at {timestamp}")
    log(f"Args: {vars(args)}")
    
    for epoch in range(1, args.epochs + 1):
        log(f"\n--- Epoch {epoch}/{args.epochs} ---")
        use_features = args.feature_dir is not None
        avg_train_loss, _, _ = train_one_epoch(classifier, train_loader, backend, optimizer, criterion, device, use_features=use_features)
        avg_test_loss, metrics = evaluate(classifier, test_loader, backend, criterion, device, use_features=use_features)
        
        history["train_loss"].append(avg_train_loss)
        history["test_metrics"].append(metrics)
        
        log(f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
        for k, v in metrics.items():
            log(f"Test {k}: {v:.4f}")
            
        # Save per-epoch checkpoint
        epoch_path = os.path.join(run_dir, f"epoch_{epoch}.pt")
        save_checkpoint(epoch_path, classifier, optimizer, None, epoch, metrics["f1"])
        
        # Save best checkpoint
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_path = os.path.join(run_dir, "best_classifier.pt")
            save_checkpoint(best_path, classifier, optimizer, None, epoch, best_f1)
            log(f"[*] Saved new best model (F1: {best_f1:.4f})")
            
        # Save last checkpoint
        last_path = os.path.join(run_dir, "last_classifier.pt")
        save_checkpoint(last_path, classifier, optimizer, None, epoch, metrics["f1"])
            
        save_json(os.path.join(run_dir, "history.json"), history)
        
    log_file.close()

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
    parser.add_argument("--feature_dir", type=str, default=None, help="Path to pre-extracted CLIP features")
    
    args = parser.parse_args()
    main(args)
