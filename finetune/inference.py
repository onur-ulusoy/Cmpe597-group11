import argparse
import torch
import sys
import os
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel

# Add root to path to import utils and zero_shot.common
sys.path.append(os.getcwd())
import utils
from zero_shot.common import MemeCapDataset, compute_metrics, l2_normalize

def get_device():
    if torch.cuda.is_available(): return "cuda"
    elif torch.backends.mps.is_available(): return "mps"
    return "cpu"

def encode_batches(model, processor, inputs, mode, device, batch_size=32):
    """
    Encodes a list of images or texts into embeddings.
    mode: 'image' or 'text'
    """
    embeddings = []
    
    for i in tqdm(range(0, len(inputs), batch_size), desc=f"Encoding {mode}s"):
        batch = inputs[i : i + batch_size]
        
        if mode == 'image':
            processed = processor(images=batch, return_tensors="pt", padding=True)
        else:
            processed = processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77)
            
        pixel_values = processed.get("pixel_values")
        input_ids = processed.get("input_ids")
        attention_mask = processed.get("attention_mask")
        
        if pixel_values is not None: pixel_values = pixel_values.to(device)
        if input_ids is not None: input_ids = input_ids.to(device)
        if attention_mask is not None: attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            if mode == 'image':
                emb = model.get_image_features(pixel_values=pixel_values)
            else:
                emb = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        
        embeddings.append(emb.cpu())
        
    return torch.cat(embeddings, dim=0)

def evaluate(args):
    device = get_device()
    print(f"🚀 Inference on {device}...")

    # 1. Resolve Checkpoint Path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = utils.get_latest_checkpoint(args.output_dir)

    print(f"📂 Loading adapter from: {checkpoint_path}")

    # 2. Load Model (Base + LoRA)
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    base_model = CLIPModel.from_pretrained(model_id)
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.to(device)
    model.eval()

    # 3. Load Data (Using common.py logic)
    print("📂 Loading Test Data...")
    dataset = MemeCapDataset.from_paths(
        data_dir="data",
        test_json=args.test_json,
        image_root=args.image_root
    )
    
    # Extract raw lists
    images = [s.image_path for s in dataset.samples] # Paths (common.py handles loading)
    # We need to load actual PIL images for the processor
    from PIL import Image
    loaded_images = [Image.open(p).convert("RGB") for p in images]
    
    captions = [s.caption for s in dataset.samples]
    titles = [s.title for s in dataset.samples]

    # 4. Generate Embeddings
    print("🧠 Generating Embeddings...")
    
    # Image Embeddings
    img_embs = encode_batches(model, processor, loaded_images, 'image', device)
    img_embs = l2_normalize(img_embs)
    
    # Caption (Target) Embeddings
    cap_embs = encode_batches(model, processor, captions, 'text', device)
    cap_embs = l2_normalize(cap_embs)
    
    # Title Embeddings (Only needed for Type 2)
    title_embs = encode_batches(model, processor, titles, 'text', device)
    title_embs = l2_normalize(title_embs)

    results = []

    # --- Evaluate Type 1 (Image Only) ---
    scores_t1 = img_embs @ cap_embs.T
    metrics_t1 = compute_metrics(scores_t1)
    results.append(("Fine-Tuned CLIP", "Type 1", metrics_t1))

    # --- Evaluate Type 2 (Image + Title) ---
    # Formula: alpha * img + (1-alpha) * title
    alpha = args.alpha
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
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Specific path to lora_epoch_X folder")
    parser.add_argument("--output_dir", type=str, default="outputs/finetune", help="Base dir to search for latest")
    parser.add_argument("--test_json", type=str, default="data/memes-test.json")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for image in Type 2 (Image + Title)")
    
    args = parser.parse_args()
    evaluate(args)
