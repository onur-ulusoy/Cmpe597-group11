import argparse
import torch
import sys
import os

# --- PATH FIX ---
# 1. Add root to path
sys.path.append(os.getcwd())
# 2. Add zero_shot to path so internal imports like 'from common import...' work
sys.path.append(os.path.join(os.getcwd(), "zero_shot"))

import utils
# Now we can import these safely
from zero_shot.common import MemeCapDataset, compute_metrics, l2_normalize
from finetune.lora_backend import LoRAOpenCLIPBackend

def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Inference on {device} (OpenCLIP LoRA)...")

    # 1. Resolve Checkpoint Path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        # Try to find the latest checkpoint automatically
        try:
            checkpoint_path = utils.get_latest_checkpoint(args.output_dir)
        except Exception as e:
            print(f"⚠️ Could not find latest checkpoint automatically: {e}")
            print("Please provide --checkpoint_path explicitly.")
            return

    print(f"📂 Loading adapter from: {checkpoint_path}")

    # 2. Initialize Backend (Loads Model + LoRA)
    try:
        backend = LoRAOpenCLIPBackend(
            model_name="ViT-L-14",
            pretrained="laion2b_s32b_b82k",
            checkpoint_path=checkpoint_path,
            device=device
        )
    except Exception as e:
        print(f"\n❌ Error loading LoRA backend: {e}")
        print("Did you run the NEW train.py? Old checkpoints might be incompatible.")
        return

    # 3. Load Data
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
    
    image_paths = [s.image_path for s in dataset.samples]
    captions = [s.caption for s in dataset.samples]
    titles = [s.title for s in dataset.samples]

    # 4. Generate Embeddings
    print("🧠 Generating Embeddings...")
    
    # Batch processing is handled inside the backend, but let's be safe
    img_embs = backend.encode_images(image_paths, batch_size=args.batch_size)
    cap_embs = backend.encode_texts(captions, batch_size=args.batch_size)
    title_embs = backend.encode_texts(titles, batch_size=args.batch_size)

    results = []

    # --- Evaluate Type 1 (Image Only) ---
    scores_t1 = img_embs @ cap_embs.T
    metrics_t1 = compute_metrics(scores_t1)
    results.append(("Fine-Tuned CLIP", "Type 1", metrics_t1))

    # --- Evaluate Type 2 (Image + Title) ---
    alpha = args.alpha
    # Fusion: alpha * Image + (1-alpha) * Title
    query_embs_t2 = (alpha * img_embs) + ((1 - alpha) * title_embs)
    query_embs_t2 = l2_normalize(query_embs_t2)
    
    scores_t2 = query_embs_t2 @ cap_embs.T
    metrics_t2 = compute_metrics(scores_t2)
    results.append(("Fine-Tuned CLIP", "Type 2", metrics_t2))

    # 5. Print Markdown Table
    print("\n## 6. Results\n")
    print("| Model | Input Type | R@1 | R@5 | MRR |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    for model_name, input_type, m in results:
        r1 = m['R@1'] * 100
        r5 = m['R@5'] * 100
        mrr = m['MRR'] * 100
        print(f"| {model_name} | {input_type} | {r1:.2f} | {r5:.2f} | {mrr:.2f} |")
    
    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to lora_epoch_X folder")
    parser.add_argument("--output_dir", type=str, default="outputs/finetune", help="Base dir to search for latest")
    parser.add_argument("--test_json", type=str, default="data/memes-test.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=128) 
    
    args = parser.parse_args()
    evaluate(args)
