import json
import os
import argparse
import random
from collections import Counter
from tqdm import tqdm
import torch
from transformers import pipeline
from pathlib import Path

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def save_markdown_report(samples, image_root, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rel_prefix = os.path.relpath(image_root, start=os.path.dirname(output_path))
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Manual Quality Check (Random Samples)\n\n")
        for i, item in enumerate(samples):
            img_fname = item.get("img_fname", "")
            meme_caps = item.get("meme_captions", [])
            text = meme_caps[0] if isinstance(meme_caps, list) and len(meme_caps) > 0 else item.get("title", "")
            label = item.get("sentiment_label", "")
            score = item.get("sentiment_score", 0.0)
            
            img_path = os.path.join(rel_prefix, img_fname).replace("\\", "/")
            
            f.write(f"### {i+1}. Predicted Label: **{label}** (Score: {score:.4f})\n")
            f.write(f"- **Meme Text**: {text}\n\n")
            f.write(f"<img src='{img_path}' width='400'>\n\n")
            f.write("---\n\n")

def main(args):
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "text-classification", 
        model=args.model_name, 
        device=device, 
        truncation=True, 
        max_length=512
    )

    model_slug = args.model_name.split("/")[-1].replace("-", "_")
    output_base = Path(args.output_dir) / model_slug
    output_base.mkdir(parents=True, exist_ok=True)
    
    all_processed_data = []
    imbalance_report = {}

    for split_name, input_path in [("train", args.train_input), ("test", args.test_input)]:
        print(f"\nProcessing {split_name} split using {model_slug}...")
        data = load_json(input_path)
        
        label_counter = Counter()
        processed_data = []

        for item in tqdm(data):
            meme_caps = item.get("meme_captions", [])
            text = meme_caps[0] if isinstance(meme_caps, list) and len(meme_caps) > 0 else item.get("title", "")
            
            if not text: continue

            result = classifier(text)[0]
            label = result["label"]
            score = result["score"]

            item["sentiment_label"] = label
            item["sentiment_score"] = score
            processed_data.append(item)
            all_processed_data.append(item)
            label_counter[label] += 1

        out_name = f"{split_name}_labels.json"
        save_json(processed_data, output_base / out_name)
        
        total = sum(label_counter.values())
        dist = {lbl: {"count": cnt, "percentage": (cnt/total)*100} for lbl, cnt in label_counter.items()}
        imbalance_report[split_name] = dist

    report_name = "imbalance_report.json"
    save_json(imbalance_report, output_base / report_name)
    
    if len(all_processed_data) >= args.num_samples:
        random.seed(42)
        sampled_items = random.sample(all_processed_data, args.num_samples)
        md_name = "manual_check.md"
        save_markdown_report(sampled_items, args.image_root, output_base / md_name)

    print(f"\nGeneration complete for {model_slug}. Outputs saved to {output_base}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", type=str, default="data/memes-trainval.json")
    parser.add_argument("--test_input", type=str, default="data/memes-test.json")
    parser.add_argument("--output_dir", type=str, default="outputs/sentiment_classification/labels")
    parser.add_argument("--image_root", type=str, default="data/memes")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--model_name", type=str, default="j-hartmann/emotion-english-distilroberta-base")
    args = parser.parse_args()
    main(args)