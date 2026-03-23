import argparse
import torch
import sys
import os
import glob

# --- PATH FIX ---
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "zero_shot"))

from zero_shot.common import MemeCapDataset, compute_metrics, l2_normalize
from finetune.lora_backend import LoRAOpenCLIPBackend

def get_latest_valid_checkpoint(base_dir):
    """Finds the most recent timestamped folder and a VALID lora checkpoint inside it."""
    if not os.path.exists(base_dir):
        return None
    
    # 1. Find all timestamp folders, sort by date (newest last)
    runs = sorted(glob.glob(os.path.join(base_dir, "*")))
    
    # Iterate backwards (newest runs first) to find one with valid checkpoints
    for run_dir in reversed(runs):
        if not os.path.isdir(run_dir):
            continue
            
        # 2. Find epoch folders
        epochs = glob.glob(os.path.join(run_dir, "lora_epoch_*"))
        if not epochs:
            continue
            
        # Sort by epoch number (highest last)
        epochs.sort(key=lambda x: int(x.split("_")[-1]))
        
        # 3. Check for adapter_config.json (CRITICAL CHECK)
        for checkpoint in reversed(epochs):
            if os.path.isfile(os.path.join(checkpoint, "adapter_config.json")):
                return checkpoint
    
    return None

def evaluate_model(task_name, checkpoint_path, dataset, args, device):
    """Loads a specific model and evaluates it."""
    print(f"\n🔄 --- Evaluating {task_name} ---")
    print(f"📂 Loading adapter: {checkpoint_path}")

    # 1. Load Model
    try:
        backend = LoRAOpenCLIPBackend(
            model_name="ViT-L-14",
            pretrained="laion2b_s32b_b82k",
            checkpoint_path=checkpoint_path,
            device=device
        )
    except Exception as e:
        print(f"❌ Error loading LoRA: {e}")
        return None

    # 2. Generate Embeddings
    print("🧠 Generating Embeddings...")
    image_paths = [s.image_path for s in dataset.samples]
    captions = [s.caption for s in dataset.samples]
    titles = [s.title for s in dataset.samples]

    img_embs = backend.encode_images(image_paths, batch_size=args.batch_size)
    cap_embs = backend.encode_texts(captions, batch_size=args.batch_size)
    
    # 3. Compute Scores based on Task
    if task_name == "Type 1":
        # Image Only
        scores = img_embs @ cap_embs.T
        
    elif task_name == "Type 2":
        # Image + Title Fusion
        title_embs = backend.encode_texts(titles, batch_size=args.batch_size)
        
        # Normalize first
        img_embs = l2_normalize(img_embs)
        title_embs = l2_normalize(title_embs)
        
        # Fusion: Average
        query_embs = img_embs + title_embs
        query_embs = l2_normalize(query_embs)
        
        scores = query_embs @ cap_embs.T

    # 4. Metrics
    metrics = compute_metrics(scores)
    
    # Clean up
    del backend
    torch.cuda.empty_cache()
    
    return metrics

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Dual Evaluation on {device}")

    # 1. Load Dataset
    print("📂 Loading Test Data...")
    try:
        dataset = MemeCapDataset.from_paths(
            data_dir="data",
            test_json=args.test_json,
            image_root=args.image_root
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    results = []

    # --- 2. Evaluate Type 1 (Image Only) ---
    ckpt_t1 = args.ckpt_type1
    if not ckpt_t1:
        ckpt_t1 = get_latest_valid_checkpoint("outputs/finetune_type1")
    
    if ckpt_t1:
        metrics = evaluate_model("Type 1", ckpt_t1, dataset, args, device)
        if metrics:
            results.append(("Fine-Tuned (Type 1)", "Image Only", metrics))
    else:
        print("⚠️ Skipping Type 1: No valid checkpoint found in outputs/finetune_type1")

    # --- 3. Evaluate Type 2 (Image + Title) ---
    ckpt_t2 = args.ckpt_type2
    if not ckpt_t2:
        ckpt_t2 = get_latest_valid_checkpoint("outputs/finetune_type2")
        
    if ckpt_t2:
        metrics = evaluate_model("Type 2", ckpt_t2, dataset, args, device)
        if metrics:
            results.append(("Fine-Tuned (Type 2)", "Image + Title", metrics))
    else:
        print("⚠️ Skipping Type 2: No valid checkpoint found in outputs/finetune_type2")

    # --- 4. Print Final Table ---
    if not results:
        print("\n❌ No models evaluated. Please train the models first.")
        return

    print("\n\n## 📊 Final Results Comparison\n")
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
    parser.add_argument("--ckpt_type1", type=str, default=None, help="Path to Type 1 checkpoint")
    parser.add_argument("--ckpt_type2", type=str, default=None, help="Path to Type 2 checkpoint")
    parser.add_argument("--test_json", type=str, default="data/memes-test.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--batch_size", type=int, default=128) 
    
    args = parser.parse_args()
    main(args)
