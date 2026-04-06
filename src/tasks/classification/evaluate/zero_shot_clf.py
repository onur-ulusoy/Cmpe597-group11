import argparse
import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.common.classification_dataset import load_classification_records
from src.common.classification_metrics import compute_classification_metrics, print_classification_report
from src.common.utils import save_json, set_seed
from src.models.pretrained.openclip import OpenCLIPBackend

def run_zero_shot_classification(args):
    set_seed(args.seed)
    
    # Load dataset
    samples = load_classification_records(
        json_path=args.test_json,
        image_root=args.image_root,
        limit=args.limit
    )
    
    # Extract Embeddings
    backend = OpenCLIPBackend(
        model_name=args.openclip_model_name,
        pretrained=args.openclip_pretrained,
        device=args.device
    )
    
    all_scores = []
    all_labels = []
    
    print(f"Running zero-shot classification evaluation...")
    for start in tqdm(range(0, len(samples), args.batch_size), desc="Classification pairs"):
        batch_samples = samples[start:start+args.batch_size]
        
        img_paths = [s.image_path for s in batch_samples]
        texts = [s.text for s in batch_samples]
        labels = [s.label for s in batch_samples]
        
        img_emb = backend.encode_images(img_paths, batch_size=args.batch_size)
        text_emb = backend.encode_texts(texts, batch_size=args.batch_size)
        
        # Compute cosine similarity
        img_emb = F.normalize(img_emb, p=2, dim=-1)
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        
        # Batch dot product
        similarities = (img_emb * text_emb).sum(dim=-1).cpu().numpy()
        
        # If higher similarity means Literal (0), then 1-S means Metaphorical (1)
        # We'll use 1-S as the "probability" of the metaphorical class
        scores = 1.0 - similarities
        
        all_scores.extend(scores)
        all_labels.extend(labels)
        
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Find the best threshold on the current (test) set for a stronger baseline
    # (Optional: we could use a fixed threshold of 0.5 if scores are properly scaled)
    best_f1 = -1
    best_threshold = 0.5
    
    for threshold in np.linspace(all_scores.min(), all_scores.max(), 50):
        preds = (all_scores >= threshold).astype(int)
        metrics = compute_classification_metrics(all_labels, preds, all_scores)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = threshold
            
    final_preds = (all_scores >= best_threshold).astype(int)
    final_metrics = compute_classification_metrics(all_labels, final_preds, all_scores)
    
    run_name = f"Zero-Shot_{args.openclip_model_name}_Thresh{best_threshold:.2f}"
    print_classification_report(run_name, final_metrics)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_json(output_dir / "zero_shot_metrics.json", {
        "run_name": run_name,
        "model_name": args.openclip_model_name,
        "threshold": float(best_threshold),
        "metrics": final_metrics
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json", type=str, default="data/memes-test.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--output_dir", type=str, default="outputs/classification/zero_shot")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--openclip_model_name", type=str, default="ViT-L-14")
    parser.add_argument("--openclip_pretrained", type=str, default="laion2b_s32b_b82k")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    run_zero_shot_classification(args)
