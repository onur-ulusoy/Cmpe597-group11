import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(os.getcwd())

from src.common.classification_dataset import load_classification_records, MemeCapClassificationDataset
from src.common.classification_metrics import compute_classification_metrics, print_classification_report
from src.common.utils import load_checkpoint, save_json
from src.models.pretrained.openclip import OpenCLIPBackend
from src.models.custom.classification_model import MemeClassificationModel

def get_device():
    if torch.cuda.is_available(): return "cuda"
    elif torch.backends.mps.is_available(): return "mps"
    return "cpu"

@torch.no_grad()
def main(args):
    device = get_device()
    print(f"Using device: {device}")
    
    # Dataset
    test_records = load_classification_records(args.test_json, args.image_root, limit=args.limit)
    
    class PathDataset(MemeCapClassificationDataset):
        def __getitem__(self, idx):
            sample = self.samples[idx]
            return {"image_path": str(sample.image_path), "text": sample.text, "label": sample.label}
            
    test_dataset = PathDataset(test_records)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Backend
    backend = OpenCLIPBackend(args.model_name, args.pretrained, device)
    
    # Model
    model = MemeClassificationModel(hidden_dim=args.hidden_dim, dropout=0.1).to(device)
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    
    all_scores = []
    all_preds = []
    all_labels = []
    
    for batch in tqdm(test_loader, desc="Inference"):
        img_paths = batch["image_path"]
        texts = batch["text"]
        labels = batch["label"].numpy()
        
        img_emb = backend.encode_images(img_paths, batch_size=len(img_paths)).to(device)
        text_emb = backend.encode_texts(texts, batch_size=len(texts)).to(device)
        
        logits = model(img_emb, text_emb)
        scores = torch.sigmoid(logits).cpu().numpy()
        preds = scores.round()
        
        all_scores.extend(scores.flatten())
        all_preds.extend(preds.flatten())
        all_labels.extend(labels)
        
    metrics = compute_classification_metrics(np.array(all_labels), np.array(all_preds), np.array(all_scores))
    print_classification_report("Final Evaluation", metrics)
    
    # Save results
    output_dir = os.path.dirname(args.checkpoint)
    save_json(os.path.join(output_dir, "final_metrics.json"), metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json", type=str, default="data/memes-test.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_classifier.pt")
    
    parser.add_argument("--model_name", type=str, default="ViT-L-14")
    parser.add_argument("--pretrained", type=str, default="laion2b_s32b_b82k")
    
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    
    args = parser.parse_args()
    main(args)
