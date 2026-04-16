import argparse
import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.getcwd())

from src.common.dataset import load_memecap_records
from src.common.metrics import compute_recall_metrics
from src.common.utils import get_latest_checkpoint
from src.models.pretrained.lora import LoRAOpenCLIPBackend

def evaluate_model(task_name, checkpoint_path, records, args, device):
    """Loads a specific model and evaluates it using unified metrics."""
    print(f"--- Evaluating {task_name} ---")
    print(f"Loading adapter: {checkpoint_path}")

    try:
        backend = LoRAOpenCLIPBackend(
            model_name="ViT-L-14",
            pretrained="laion2b_s32b_b82k",
            checkpoint_path=checkpoint_path,
            device=device
        )
    except Exception as e:
        print(f" Error loading LoRA: {e}")
        return None

    print("Generating Embeddings...")
    image_paths = [s.image_path for s in records]
    captions = [s.caption for s in records]
    titles = [s.title for s in records]

    img_embs = backend.encode_images(image_paths, batch_size=args.batch_size)
    cap_embs = backend.encode_texts(captions, batch_size=args.batch_size)
    
    if task_name == "Type 1":
        scores = img_embs @ cap_embs.T
        
    elif task_name == "Type 2":
        title_embs = backend.encode_texts(titles, batch_size=args.batch_size)
        
        img_embs = F.normalize(img_embs, p=2, dim=-1)
        title_embs = F.normalize(title_embs, p=2, dim=-1)
        
        query_embs = img_embs + title_embs
        query_embs = F.normalize(query_embs, p=2, dim=-1)
        
        scores = query_embs @ cap_embs.T

    metrics = compute_recall_metrics(scores.cpu(), ks=(1, 5, 10))
    
    del backend
    torch.cuda.empty_cache()
    
    return metrics

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Dual Evaluation on {device}")

    print("Loading Test Data...")
    try:
        records = load_memecap_records(args.test_json, args.image_root)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    results = []

    # Evaluate Type 1
    ckpt_t1 = args.ckpt_type1
    if not ckpt_t1:
        try:
            ckpt_t1 = get_latest_checkpoint("outputs/retrieval/finetune/type1")
        except:
            ckpt_t1 = None
    
    if ckpt_t1:
        metrics = evaluate_model("Type 1", ckpt_t1, records, args, device)
        if metrics:
            results.append(("Fine-Tuned (Type 1)", "Image Only", metrics))
    else:
        print("Skipping Type 1: No valid checkpoint found in outputs/retrieval/finetune_type1")

    # Evaluate Type 2
    ckpt_t2 = args.ckpt_type2
    if not ckpt_t2:
        try:
            ckpt_t2 = get_latest_checkpoint("outputs/retrieval/finetune/type2")
        except:
            ckpt_t2 = None
        
    if ckpt_t2:
        metrics = evaluate_model("Type 2", ckpt_t2, records, args, device)
        if metrics:
            results.append(("Fine-Tuned (Type 2)", "Image + Title", metrics))
    else:
        print("Skipping Type 2: No valid checkpoint found in outputs/retrieval/finetune_type2")

    if not results:
        print("No models evaluated. Please train the models first.")
        return

    print("\n\n## Final Results Comparison\n")
    print("| Model Source | Input Type | R@1 | R@5 | MRR |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    for model_name, input_type, m in results:
        r1 = m['R@1'] * 100
        r5 = m['R@5'] * 100
        mrr = m['MRR'] * 100
        print(f"| {model_name} | {input_type} | {r1:.2f} | {r5:.2f} | {mrr:.2f} |")
    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_type1", type=str, default=None)
    parser.add_argument("--ckpt_type2", type=str, default=None)
    parser.add_argument("--test_json", type=str, default="data/memes-test.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--batch_size", type=int, default=128) 
    
    args = parser.parse_args()
    main(args)